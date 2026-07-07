"""FFmpeg download helper."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

_FFMPEG_URL = (
    "https://www.gyan.dev/ffmpeg/builds/"
    "ffmpeg-release-essentials.zip"
)
_FFMPEG_EXE = Path(__file__).parent / "ffmpeg.exe"


def ffmpeg_path() -> Path | None:
    """Return path to ffmpeg.exe if it exists, else None."""
    return _FFMPEG_EXE if _FFMPEG_EXE.is_file() else None


def download_ffmpeg(progress_cb=None) -> Path:
    """Download ffmpeg.exe from gyan.dev.  Returns path to exe.

    *progress_cb* is called with (downloaded_bytes, total_bytes) during download.
    """
    import sys as _sys

    print("Downloading FFmpeg (~55 MB) …", file=_sys.stderr)
    resp = urlopen(_FFMPEG_URL)
    total = int(resp.headers.get("Content-Length", 0))

    chunks = []
    received = 0
    while True:
        chunk = resp.read(65536)
        if not chunk:
            break
        chunks.append(chunk)
        received += len(chunk)
        if progress_cb:
            progress_cb(received, total)

    data = b"".join(chunks)
    z = zipfile.ZipFile(io.BytesIO(data))
    for name in z.namelist():
        if name.endswith("/ffmpeg.exe"):
            with z.open(name) as src, open(_FFMPEG_EXE, "wb") as dst:
                dst.write(src.read())
            _FFMPEG_EXE.chmod(0o755)
            print("Done.", file=_sys.stderr)
            return _FFMPEG_EXE

    raise RuntimeError("ffmpeg.exe not found in the archive")
