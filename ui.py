"""OmniCut — Tkinter GUI for 360° video frame extraction & image splitting."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

import cv2

from projection import generate_poses, process_image, process_batch, extract_frames


_APP_DIR = Path(__file__).resolve().parent


def _clamp(v: str, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except ValueError:
        return lo


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class Application(ttk.Frame):
    """OmniCut main window with two tabs."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.master = master
        master.title("OmniCut")
        master.resizable(False, False)

        self._processing = False
        self._cancel_flag = False

        self._build_widgets()
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # -----------------------------------------------------------------------
    # Widgets
    # -----------------------------------------------------------------------

    def _build_widgets(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.X)

        self._tab_video = ttk.Frame(nb, padding=8)
        self._tab_split = ttk.Frame(nb, padding=8)
        nb.add(self._tab_video, text="  视频抽帧  ")
        nb.add(self._tab_split, text="  图片裁切  ")

        self._build_video_tab()
        self._build_split_tab()

        # Shared progress + action bar
        self._build_status_section()
        self._build_action_section()

    # -- Tab 1: 视频抽帧 -----------------------------------------------------

    def _build_video_tab(self) -> None:
        f = self._tab_video

        # Path
        r1 = ttk.Frame(f)
        r1.pack(fill=tk.X, pady=4)
        ttk.Label(r1, text="Video:").pack(side=tk.LEFT)
        self._v_path = tk.StringVar()
        ttk.Entry(r1, textvariable=self._v_path, width=50).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r1, text="Browse", command=lambda: self._browse_video()).pack(side=tk.LEFT)

        # FPS + FFmpeg
        r2 = ttk.Frame(f)
        r2.pack(fill=tk.X, pady=4)
        ttk.Label(r2, text="FPS:").pack(side=tk.LEFT)
        self._v_fps = tk.StringVar(value="1")
        ttk.Spinbox(r2, textvariable=self._v_fps, from_=0.1, to=30,
                    increment=0.5, width=6).pack(side=tk.LEFT, padx=4)

        self._v_dl_btn = ttk.Button(r2, text="Download FFmpeg",
                                    command=self._download_ffmpeg)
        self._v_dl_btn.pack(side=tk.LEFT, padx=(10, 0))
        self._v_ff_status = tk.StringVar(value="")
        ttk.Label(r2, textvariable=self._v_ff_status,
                  foreground="gray").pack(side=tk.LEFT, padx=(6, 0))

        # Output
        r3 = ttk.Frame(f)
        r3.pack(fill=tk.X, pady=4)
        ttk.Label(r3, text="Output:").pack(side=tk.LEFT)
        self._v_out = tk.StringVar(value=str(_APP_DIR / "output"))
        ttk.Entry(r3, textvariable=self._v_out, width=45).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r3, text="Browse", command=lambda: self._browse_out(self._v_out)).pack(side=tk.LEFT)

        self._update_ffmpeg_status()
        self._v_run_btn = ttk.Button(f, text="Extract Frames", command=self._run_video, width=14)
        self._v_run_btn.pack(anchor="w", pady=(8, 0))

    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if p:
            self._v_path.set(p)

    # -- Tab 2: 图片裁切 -----------------------------------------------------

    def _build_split_tab(self) -> None:
        f = self._tab_split

        # Input
        g1 = ttk.LabelFrame(f, text="Input Source", padding=6)
        g1.pack(fill=tk.X, pady=(0, 6))

        r_in = ttk.Frame(g1)
        r_in.pack(fill=tk.X)
        ttk.Label(r_in, text="Type:").pack(side=tk.LEFT)
        self._s_type = tk.StringVar(value="Image Folder")
        cb = ttk.Combobox(r_in, textvariable=self._s_type,
                          values=["Single Image", "Image Folder"],
                          state="readonly", width=14)
        cb.pack(side=tk.LEFT, padx=4)

        ttk.Label(r_in, text="Path:").pack(side=tk.LEFT, padx=(8, 0))
        self._s_path = tk.StringVar()
        ttk.Entry(r_in, textvariable=self._s_path, width=40).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r_in, text="Browse", command=self._browse_split).pack(side=tk.LEFT)

        # Camera poses
        g2 = ttk.LabelFrame(f, text="Camera Poses", padding=6)
        g2.pack(fill=tk.X, pady=(0, 6))

        def _pose_row(parent: ttk.Frame, label: str, d_start: float, d_end: float, d_cnt: int):
            sv_s = tk.StringVar(value=str(d_start))
            sv_e = tk.StringVar(value=str(d_end))
            sv_c = tk.StringVar(value=str(d_cnt))
            r = ttk.Frame(parent)
            r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=label, width=5, anchor="e").pack(side=tk.LEFT)
            sb_s = ttk.Spinbox(r, textvariable=sv_s, from_=-180, to=360, increment=5, width=6)
            sb_s.pack(side=tk.LEFT, padx=1)
            ttk.Label(r, text="→").pack(side=tk.LEFT, padx=1)
            sb_e = ttk.Spinbox(r, textvariable=sv_e, from_=-180, to=360, increment=5, width=6)
            sb_e.pack(side=tk.LEFT, padx=1)
            ttk.Label(r, text="Count:").pack(side=tk.LEFT, padx=(4, 1))
            sb_c = ttk.Spinbox(r, textvariable=sv_c, from_=1, to=36, width=4)
            sb_c.pack(side=tk.LEFT)
            for w in (sb_s, sb_e, sb_c):
                w.bind("<<Increment>>", self._update_s_total, add="+")
                w.bind("<<Decrement>>", self._update_s_total, add="+")
                w.bind("<KeyRelease>", self._update_s_total, add="+")
            return sv_s, sv_e, sv_c

        self._s_yaw_s, self._s_yaw_e, self._s_yaw_c = _pose_row(g2, "Yaw", 0, 360, 8)
        self._s_pitch_s, self._s_pitch_e, self._s_pitch_c = _pose_row(g2, "Pitch", 0, 0, 1)
        self._s_total_label = ttk.Label(g2, text="Total views: 8")
        self._s_total_label.pack(anchor="w", pady=(2, 0))
        self._update_s_total()

        # Output settings
        g3 = ttk.LabelFrame(f, text="Output Settings", padding=6)
        g3.pack(fill=tk.X)

        r_o1 = ttk.Frame(g3)
        r_o1.pack(fill=tk.X, pady=2)
        ttk.Label(r_o1, text="FOV:").pack(side=tk.LEFT)
        self._s_fov = tk.StringVar(value="90")
        ttk.Entry(r_o1, textvariable=self._s_fov, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r_o1, text="°").pack(side=tk.LEFT)
        ttk.Label(r_o1, text="  Resolution:").pack(side=tk.LEFT, padx=(12, 0))
        self._s_res = tk.StringVar(value="2048")
        ttk.Entry(r_o1, textvariable=self._s_res, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r_o1, text="px").pack(side=tk.LEFT)
        ttk.Label(r_o1, text="  Quality:").pack(side=tk.LEFT, padx=(12, 0))
        self._s_q = tk.StringVar(value="100")
        ttk.Entry(r_o1, textvariable=self._s_q, width=4).pack(side=tk.LEFT, padx=4)

        r_o2 = ttk.Frame(g3)
        r_o2.pack(fill=tk.X, pady=2)
        ttk.Label(r_o2, text="Output:").pack(side=tk.LEFT)
        self._s_out = tk.StringVar(value=str(_APP_DIR / "output"))
        ttk.Entry(r_o2, textvariable=self._s_out, width=40).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r_o2, text="Browse", command=lambda: self._browse_out(self._s_out)).pack(side=tk.LEFT)

        self._s_run_btn = ttk.Button(f, text="Run Split", command=self._run_split, width=14)
        self._s_run_btn.pack(anchor="w", pady=(8, 0))

    def _browse_split(self) -> None:
        if self._s_type.get() == "Single Image":
            p = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        else:
            p = filedialog.askdirectory(title="Select Image Folder")
        if p:
            self._s_path.set(p)

    def _update_s_total(self, *_args) -> None:
        def _v(v, default):
            try:
                return max(1, int(v.get()))
            except (ValueError, tk.TclError):
                return default
        n = _v(self._s_yaw_c, 8) * _v(self._s_pitch_c, 3)
        self._s_total_label.config(text=f"Total views: {n}")

    # -- Shared: Progress + Actions ------------------------------------------

    def _build_status_section(self) -> None:
        f = ttk.LabelFrame(self, text="Progress", padding=6)
        f.pack(fill=tk.X, pady=(6, 6))

        self._progress = ttk.Progressbar(f, mode="determinate")
        self._progress.pack(fill=tk.X)

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(f, textvariable=self._status_var).pack(anchor="w", pady=(2, 0))

    def _build_action_section(self) -> None:
        f = ttk.Frame(self)
        f.pack(fill=tk.X)

        self._cancel_btn = ttk.Button(f, text="Cancel", command=self._cancel,
                                      width=10, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.RIGHT)

    # -- Shared: FFmpeg download ---------------------------------------------

    def _update_ffmpeg_status(self) -> None:
        from ffmpeg_dl import ffmpeg_path
        if ffmpeg_path():
            self._v_ff_status.set("Ready")
            self._v_dl_btn.config(state=tk.DISABLED)
        else:
            self._v_ff_status.set("Not installed")
            self._v_dl_btn.config(state=tk.NORMAL)

    def _download_ffmpeg(self) -> None:
        self._v_dl_btn.config(state=tk.DISABLED, text="Downloading…")
        self._v_ff_status.set("Downloading…")
        t = threading.Thread(target=self._dl_thread, daemon=True)
        t.start()

    def _dl_thread(self) -> None:
        from ffmpeg_dl import download_ffmpeg
        try:
            download_ffmpeg()
            self.master.after(0, lambda: (
                self._v_ff_status.set("Ready"),
                self._v_dl_btn.config(text="Download FFmpeg", state=tk.DISABLED),
            ))
        except Exception as e:
            self.master.after(0, lambda: (
                self._v_ff_status.set(f"Failed: {e}"),
                self._v_dl_btn.config(text="Download FFmpeg", state=tk.NORMAL),
            ))

    # -- Shared: utils -------------------------------------------------------

    @staticmethod
    def _browse_out(var: tk.StringVar) -> None:
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            var.set(d)

    # -----------------------------------------------------------------------
    # Actions — Video
    # -----------------------------------------------------------------------

    def _run_video(self) -> None:
        if self._processing:
            return
        path = self._v_path.get().strip()
        if not path or not Path(path).exists():
            self._status_var.set("Error: video file not found")
            return

        from ffmpeg_dl import ffmpeg_path as _fp
        ffmpeg = _fp()
        if ffmpeg is None:
            self._status_var.set("Error: FFmpeg not installed. Click Download FFmpeg first.")
            return

        out_dir = Path(self._v_out.get())
        if not out_dir.is_absolute():
            out_dir = _APP_DIR / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        fps = float(self._v_fps.get())

        self._start_processing()
        args = (path, ffmpeg, out_dir, fps)
        t = threading.Thread(target=self._thread_video, args=args, daemon=True)
        t.start()

    def _thread_video(self, path: str, ffmpeg: Path, out_dir: Path, fps: float) -> None:
        try:
            extract_frames(path, out_dir, fps=fps, ffmpeg_path=ffmpeg)
            self.master.after(0, lambda: self._on_complete())
        except Exception as exc:
            self.master.after(0, lambda m=str(exc): self._on_error(m))

    # -----------------------------------------------------------------------
    # Actions — Split
    # -----------------------------------------------------------------------

    def _run_split(self) -> None:
        if self._processing:
            return
        path = self._s_path.get().strip()
        if not path or not Path(path).exists():
            self._status_var.set("Error: input path does not exist")
            return

        try:
            poses = generate_poses(
                yaw_count=int(self._s_yaw_c.get()),
                yaw_start=_clamp(self._s_yaw_s.get(), -180, 360),
                yaw_end=_clamp(self._s_yaw_e.get(), -180, 360),
                pitch_count=int(self._s_pitch_c.get()),
                pitch_start=_clamp(self._s_pitch_s.get(), -90, 90),
                pitch_end=_clamp(self._s_pitch_e.get(), -90, 90),
            )
        except (ValueError, tk.TclError) as e:
            self._status_var.set(f"Error: invalid pose parameters — {e}")
            return
        if not poses:
            self._status_var.set("Error: at least one pose required")
            return

        out_dir = Path(self._s_out.get())
        if not out_dir.is_absolute():
            out_dir = _APP_DIR / out_dir

        try:
            res = int(self._s_res.get())
        except ValueError:
            res = 2048
        try:
            fov = float(self._s_fov.get())
        except ValueError:
            fov = 90.0
        try:
            q = int(self._s_q.get())
        except ValueError:
            q = 100

        self._start_processing()
        args = (path, poses, out_dir, res, fov, q)
        t = threading.Thread(target=self._thread_split, args=args, daemon=True)
        t.start()

    def _thread_split(self, path: str, poses: list, out_dir: Path,
                      res: int, fov: float, quality: int) -> None:
        input_type = self._s_type.get()

        def progress(done: int, total: int) -> None:
            if self._cancel_flag:
                raise _CancelException()
            pct = min(100, int(done / total * 100))
            self.master.after(0, lambda: (
                setattr(self._progress, "value", pct),
                self._status_var.set(f"{done}/{total} ({pct}%)"),
            ))

        try:
            if input_type == "Single Image":
                src = cv2.imread(path)
                if src is None:
                    raise RuntimeError(f"Could not read {path}")
                h, w = src.shape[:2]
                stem = Path(path).stem
                process_image(src, w, h, poses, out_res=res, fov_deg=fov,
                              quality=quality, out_dir=out_dir,
                              source_stem=stem, progress_callback=progress)
            else:
                process_batch(Path(path), poses, out_res=res, fov_deg=fov,
                              quality=quality, output_dir=out_dir,
                              progress_callback=progress)
            self.master.after(0, self._on_complete)
        except _CancelException:
            self.master.after(0, lambda: self._status_var.set("Cancelled"))
            self.master.after(0, self._enable_ui)
        except Exception as exc:
            self.master.after(0, lambda m=str(exc): self._on_error(m))

    # -----------------------------------------------------------------------
    # Shared state
    # -----------------------------------------------------------------------

    def _start_processing(self) -> None:
        self._processing = True
        self._cancel_flag = False
        self._progress["value"] = 0
        self._cancel_btn.config(state=tk.NORMAL)

    def _cancel(self) -> None:
        self._cancel_flag = True
        self._status_var.set("Cancelling…")

    def _on_complete(self) -> None:
        self._status_var.set("Complete!")
        self._progress["value"] = 100
        self._enable_ui()

    def _on_error(self, msg: str) -> None:
        self._status_var.set(f"Error: {msg}")
        self._enable_ui()

    def _enable_ui(self) -> None:
        self._processing = False
        self._cancel_btn.config(state=tk.DISABLED)


class _CancelException(Exception):
    pass


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
