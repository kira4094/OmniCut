"""
OmniCut — Tkinter GUI for multi-axis 360° image splitting.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Optional

import cv2

from projection import generate_poses, process_image, process_batch


_APP_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: str, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except ValueError:
        return lo


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class Application(ttk.Frame):
    """Main GUI for OmniCut."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.master = master
        master.title("OmniCut — 360° Multi-Axis Splitter")
        master.resizable(False, False)

        self._processing = False
        self._cancel_flag = False

        self._build_widgets()
        self.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    # -----------------------------------------------------------------------
    # Widget construction
    # -----------------------------------------------------------------------

    def _build_widgets(self) -> None:
        self._build_input_section()
        self._build_poses_section()
        self._build_output_section()
        self._build_status_section()
        self._build_action_section()

    # -- Input --------------------------------------------------------------

    def _build_input_section(self) -> None:
        f = ttk.LabelFrame(self, text="Input Source", padding=8)
        f.pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(f)
        row.pack(fill=tk.X)

        ttk.Label(row, text="Type:").pack(side=tk.LEFT)
        self._input_type = tk.StringVar(value="Image Folder")
        cb = ttk.Combobox(row, textvariable=self._input_type,
                          values=["Single Image", "Image Folder"],
                          state="readonly", width=16)
        cb.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(row, text="Path:").pack(side=tk.LEFT, padx=(12, 0))
        self._path_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self._path_var, width=50)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))

        ttk.Button(row, text="Browse", command=self._browse).pack(side=tk.LEFT)

    def _browse(self) -> None:
        t = self._input_type.get()
        path: str | None = None
        if t == "Single Image":
            path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        else:  # Image Folder
            path = filedialog.askdirectory(title="Select Image Folder")
        if path:
            self._path_var.set(path)

    # -- Camera Poses -------------------------------------------------------

    def _build_poses_section(self) -> None:
        f = ttk.LabelFrame(self, text="Camera Poses", padding=8)
        f.pack(fill=tk.X, pady=(0, 8))

        # Column headers
        hdr = ttk.Frame(f)
        hdr.pack(fill=tk.X)
        for i, txt in enumerate(["Axis", "Start", "End", "Count"]):
            ttk.Label(hdr, text=txt, width=8, anchor="w").grid(row=0, column=i, padx=2)

        def _make_row(axis: str, default_start: float, default_end: float,
                      default_count: int) -> tuple:
            sv_start = tk.StringVar(value=str(default_start))
            sv_end = tk.StringVar(value=str(default_end))
            sv_cnt = tk.StringVar(value=str(default_count))
            row = ttk.Frame(f)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=axis, width=6, anchor="e").pack(side=tk.LEFT, padx=(0, 6))
            sb_s = ttk.Spinbox(row, textvariable=sv_start, from_=-180, to=360, increment=5, width=6)
            sb_s.pack(side=tk.LEFT, padx=1)
            ttk.Label(row, text="→").pack(side=tk.LEFT, padx=2)
            sb_e = ttk.Spinbox(row, textvariable=sv_end, from_=-180, to=360, increment=5, width=6)
            sb_e.pack(side=tk.LEFT, padx=1)
            ttk.Label(row, text="Count:").pack(side=tk.LEFT, padx=(8, 2))
            sb_c = ttk.Spinbox(row, textvariable=sv_cnt, from_=1, to=36, width=4)
            sb_c.pack(side=tk.LEFT)
            for w in (sb_s, sb_e, sb_c):
                w.bind("<<Increment>>", self._update_total, add="+")
                w.bind("<<Decrement>>", self._update_total, add="+")
                w.bind("<KeyRelease>", self._update_total, add="+")
            return sv_start, sv_end, sv_cnt

        self._yaw_s, self._yaw_e, self._yaw_c = _make_row("Yaw", 0, 360, 8)
        self._pitch_s, self._pitch_e, self._pitch_c = _make_row("Pitch", 0, 0, 1)

        self._total_label = ttk.Label(f, text="Total views: 8")
        self._total_label.pack(anchor="w", pady=(4, 0))
        self._update_total()

    def _update_total(self, *_args) -> None:
        def _v(v: tk.StringVar, default: int) -> int:
            try:
                return max(1, int(v.get()))
            except (ValueError, tk.TclError):
                return default

        n = _v(self._yaw_c, 8) * _v(self._pitch_c, 3)
        self._total_label.config(text=f"Total views: {n}")

    # -- Output Settings ----------------------------------------------------

    def _build_output_section(self) -> None:
        f = ttk.LabelFrame(self, text="Output Settings", padding=8)
        f.pack(fill=tk.X, pady=(0, 8))

        self._out_dir_var = tk.StringVar(value=str(_APP_DIR / "output"))

        r = ttk.Frame(f)
        r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="FOV:").pack(side=tk.LEFT)
        self._fov_var = tk.StringVar(value="90")
        ttk.Entry(r, textvariable=self._fov_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r, text="°").pack(side=tk.LEFT)

        ttk.Label(r, text="  Resolution:").pack(side=tk.LEFT, padx=(12, 0))
        self._res_var = tk.StringVar(value="2048")
        ttk.Entry(r, textvariable=self._res_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r, text="px").pack(side=tk.LEFT)

        ttk.Label(r, text="  Quality:").pack(side=tk.LEFT, padx=(12, 0))
        self._q_var = tk.StringVar(value="100")
        ttk.Entry(r, textvariable=self._q_var, width=4).pack(side=tk.LEFT, padx=4)

        # Output path
        r2 = ttk.Frame(f)
        r2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(r2, text="Output:").pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self._out_dir_var, width=45).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r2, text="Browse", command=self._browse_out).pack(side=tk.LEFT)

    def _browse_out(self) -> None:
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self._out_dir_var.set(d)

    # -- Status Bar ---------------------------------------------------------

    def _build_status_section(self) -> None:
        f = ttk.LabelFrame(self, text="Progress", padding=8)
        f.pack(fill=tk.X, pady=(0, 8))

        self._progress = ttk.Progressbar(f, mode="determinate")
        self._progress.pack(fill=tk.X)

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(f, textvariable=self._status_var).pack(anchor="w", pady=(4, 0))

    # -- Action Buttons -----------------------------------------------------

    def _build_action_section(self) -> None:
        f = ttk.Frame(self)
        f.pack(fill=tk.X)

        self._run_btn = ttk.Button(f, text="Run", command=self._run, width=12)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 6))

        self._cancel_btn = ttk.Button(f, text="Cancel", command=self._cancel,
                                      width=12, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT)

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def _run(self) -> None:
        if self._processing:
            return

        path = self._path_var.get().strip()
        if not path or not Path(path).exists():
            self._status_var.set("Error: input path does not exist")
            return

        # Build poses
        try:
            poses = generate_poses(
                yaw_count=int(self._yaw_c.get()),
                yaw_start=_clamp(self._yaw_s.get(), -180, 360),
                yaw_end=_clamp(self._yaw_e.get(), -180, 360),
                pitch_count=int(self._pitch_c.get()),
                pitch_start=_clamp(self._pitch_s.get(), -90, 90),
                pitch_end=_clamp(self._pitch_e.get(), -90, 90),
            )
        except (ValueError, tk.TclError) as e:
            self._status_var.set(f"Error: invalid pose parameters — {e}")
            return

        if not poses:
            self._status_var.set("Error: at least one pose required")
            return

        out_dir = Path(self._out_dir_var.get())
        if not out_dir.is_absolute():
            out_dir = _APP_DIR / out_dir

        try:
            res = int(self._res_var.get())
        except ValueError:
            res = 2048
        try:
            fov = float(self._fov_var.get())
        except ValueError:
            fov = 90.0
        try:
            quality = int(self._q_var.get())
        except ValueError:
            quality = 100

        self._processing = True
        self._cancel_flag = False
        self._run_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._progress["value"] = 0

        args = (path, poses, out_dir, res, fov, quality)
        t = threading.Thread(target=self._process_thread, args=args, daemon=True)
        t.start()

    def _cancel(self) -> None:
        self._cancel_flag = True
        self._status_var.set("Cancelling… (finishing current image)")

    def _process_thread(self, input_path: str, poses: list, out_dir: Path,
                        res: int, fov: float, quality: int) -> None:
        """Run in background thread."""
        input_type = self._input_type.get()

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
                src = cv2.imread(input_path)
                if src is None:
                    raise RuntimeError(f"Could not read {input_path}")
                h, w = src.shape[:2]
                stem = Path(input_path).stem
                process_image(src, w, h, poses, out_res=res, fov_deg=fov,
                              quality=quality, out_dir=out_dir,
                              source_stem=stem,
                              progress_callback=progress)
            else:  # Image Folder
                process_batch(Path(input_path), poses, out_res=res, fov_deg=fov,
                              quality=quality, output_dir=out_dir,
                              progress_callback=progress)

            self.master.after(0, lambda: self._on_complete(out_dir))

        except _CancelException:
            self.master.after(0, lambda: self._status_var.set("Cancelled"))
            self.master.after(0, self._enable_ui)
        except Exception as exc:
            msg = str(exc)
            self.master.after(0, lambda m=msg: self._on_error(m))

    def _on_complete(self, out_dir: Path) -> None:
        self._status_var.set("Complete!")
        self._progress["value"] = 100
        self._enable_ui()

    def _on_error(self, msg: str) -> None:
        self._status_var.set(f"Error: {msg}")
        self._enable_ui()

    def _enable_ui(self) -> None:
        self._processing = False
        self._run_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)


class _CancelException(Exception):
    """Raised inside *progress_callback* when user cancels."""
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
