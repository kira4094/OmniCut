"""
OmniCut — equirectangular to perspective projection engine.

Replaces aliceVision_split360Images.exe with full multi-axis control.
"""
from __future__ import annotations

import subprocess
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """ZYX Euler → 3×3 rotation matrix.  Angles in degrees."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    # R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    return np.array([
        [cy*cr + sy*sp*sr,  -cy*sr + sy*sp*cr,  sy*cp],
        [cp*sr,              cp*cr,              -sp   ],
        [-sy*cr + cy*sp*sr,  sy*sr + cy*sp*cr,   cy*cp],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Pose generation
# ---------------------------------------------------------------------------

_Pose = dict[str, float | np.ndarray]  # {yaw, pitch, roll, rotation}

def generate_poses(
    yaw_count: int = 8,   yaw_start: float = 0.0,   yaw_end: float = 360.0,
    pitch_count: int = 1, pitch_start: float = 0.0, pitch_end: float = 0.0,
    roll_count: int = 1,  roll_start: float = 0.0,  roll_end: float = 0.0,
) -> list[_Pose]:
    """Cartesian product of yaw/pitch/roll ranges → list of camera poses.

    When *count is 1 the range collapses to *start (the single angle).
    """
    yaws   = np.linspace(yaw_start,   yaw_end,   int(yaw_count),   endpoint=False) if yaw_count > 1   else np.array([yaw_start])
    pitches = np.linspace(pitch_start, pitch_end, int(pitch_count)) if pitch_count > 1 else np.array([pitch_start])
    rolls  = np.linspace(roll_start,  roll_end,  int(roll_count))  if roll_count > 1  else np.array([roll_start])

    return [
        dict(yaw=float(y), pitch=float(p), roll=float(r),
             rotation=rotation_matrix(y, p, r))
        for y, p, r in product(yaws, pitches, rolls)
    ]


# ---------------------------------------------------------------------------
# Remap-map precomputation  (the heavy maths, done once per pose)
# ---------------------------------------------------------------------------

def build_remap(
    eq_w: int, eq_h: int,
    out_res: int, fov_rad: float,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute (map_x, map_y) for cv2.remap().

    Pipeline per output pixel:
      (x,y) → camera-space ray → rotate → equirectangular (u,v)
    """
    xs = np.arange(out_res, dtype=np.float64)
    ys = np.arange(out_res, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    cx = (out_res - 1) / 2.0
    cy = (out_res - 1) / 2.0
    focal = (out_res / 2.0) / np.tan(fov_rad / 2.0)

    # Camera-space ray directions (looking along +Z, Y-up)
    dx = (xx - cx) / focal
    dy = (cy - yy) / focal  # negate so +Y is up
    dz = np.ones_like(dx)

    rays = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    # Apply rotation: world_ray = R @ camera_ray
    rays_world = rays @ rotation.T

    # Equirectangular projection
    lon = np.arctan2(rays_world[:, 0], rays_world[:, 2])
    lat = np.arcsin(np.clip(rays_world[:, 1], -1.0, 1.0))

    u = ((lon / (2.0 * np.pi)) + 0.5) * (eq_w - 1)
    v = (0.5 - (lat / np.pi)) * (eq_h - 1)

    return (u.reshape(out_res, out_res).astype(np.float32),
            v.reshape(out_res, out_res).astype(np.float32))


# ---------------------------------------------------------------------------
# Single image processing
# ---------------------------------------------------------------------------

ProgressCB = Optional[Callable[[int, int], None]]

def process_image(
    src: np.ndarray,
    eq_w: int, eq_h: int,
    poses: list[_Pose],
    out_res: int = 1200,
    fov_deg: float = 90.0,
    quality: int = 95,
    out_dir: str | Path = ".",
    source_stem: str = "image",
    progress_callback: ProgressCB = None,
) -> list[Path]:
    """Process one equirectangular image → perspective views (one per pose).

    *source_stem* is the original filename (without extension), used in output names.
    Returns list of written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fov_rad = np.radians(fov_deg)

    # Precompute all remap maps (one per pose)
    maps = [build_remap(eq_w, eq_h, out_res, fov_rad, p["rotation"]) for p in poses]

    paths: list[Path] = []
    n = len(poses)
    for idx, ((mx, my), pose) in enumerate(zip(maps, poses)):
        perspective = cv2.remap(src, mx, my, cv2.INTER_LINEAR)
        fname = f"{source_stem}_{idx:04d}.jpg"
        fpath = out_dir / fname
        cv2.imwrite(str(fpath), perspective, [cv2.IMWRITE_JPEG_QUALITY, quality])
        paths.append(fpath)
        if progress_callback:
            progress_callback(idx + 1, n)
    return paths


# ---------------------------------------------------------------------------
# Batch folder processing
# ---------------------------------------------------------------------------

def process_batch(
    input_dir: str | Path,
    poses: list[_Pose],
    out_res: int = 1200,
    fov_deg: float = 90.0,
    quality: int = 95,
    output_dir: str | Path | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    progress_callback: ProgressCB = None,
) -> int:
    """Process every image in *input_dir* through every pose.

    All outputs go flat into *output_dir* (no subdirectories).
    Naming: {source_stem}_{global_seq:04d}.jpg

    Returns total file count written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "split_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    fov_rad = np.radians(fov_deg)

    images = sorted([p for p in input_dir.iterdir()
                     if p.suffix.lower() in extensions and p.is_file()])
    if not images:
        return 0

    # Precompute remap maps once (use first image for dimensions)
    sample = cv2.imread(str(images[0]))
    if sample is None:
        return 0
    eq_h, eq_w = sample.shape[:2]
    maps = [build_remap(eq_w, eq_h, out_res, fov_rad, p["rotation"]) for p in poses]

    total = len(images) * len(poses)
    done = 0
    for fi, img_path in enumerate(images):
        src = cv2.imread(str(img_path))
        if src is None:
            continue
        stem = img_path.stem
        for ((mx, my), _) in zip(maps, poses):
            perspective = cv2.remap(src, mx, my, cv2.INTER_LINEAR)
            fname = f"{stem}_{done:04d}.jpg"
            cv2.imwrite(str(output_dir / fname), perspective,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            done += 1
            if progress_callback:
                progress_callback(done, total)
    return total


# ---------------------------------------------------------------------------
# FFmpeg video frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 1,
    ffmpeg_path: str | Path = "ffmpeg.exe",
) -> Path:
    """Extract frames at *fps* from *video_path* via FFmpeg.  Returns *output_dir*."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(output_dir / "image_%04d.jpg")
    cmd = [
        str(ffmpeg_path),
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-qscale:v", "1",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (code {result.returncode}): {result.stderr}")
    return output_dir


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Run a quick test on sample data if available."""
    import sys
    data = Path(__file__).with_name("_data")
    if not data.is_dir():
        print("No _data/ dir found — skipping demo", file=sys.stderr)
        return

    jpgs = list(data.glob("*.jpg")) + list(data.glob("*.png"))
    if not jpgs:
        print("_data/ has no images — skipping demo", file=sys.stderr)
        return

    first = str(jpgs[0])
    print(f"Demo: {first}")
    src = cv2.imread(first)
    h, w = src.shape[:2]

    poses = generate_poses(yaw_count=4, pitch_count=3, pitch_start=-30, pitch_end=30)
    out = Path("_demo_out")
    paths = process_image(src, w, h, poses, out_res=800, fov_deg=90, out_dir=out,
                          progress_callback=lambda d, t: print(f"  {d}/{t}"))
    print(f"  → {len(paths)} images in {out}/")
    for p in paths:
        print(f"    {p.name}")


if __name__ == "__main__":
    _demo()
