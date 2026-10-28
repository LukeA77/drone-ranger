"""Short looping animated WebP clips for the README, cut from the showcase clips
(scripts/make_showcase.py).

Usage: python scripts/make_readme_media.py

GitHub shows animated images inline in a README but does not play video files from the
repository, so each stage gets a looping clip of a few seconds. Animated WebP is used rather than
GIF: the same clips are 3 to 5 times smaller at better quality. Each must stay under MAX_MB.
They hold dataset frames and are published under the datasets' terms (D-619).
"""

from __future__ import annotations

import json
import subprocess
import sys

from aerialrange.utils.config import REPO_ROOT
from aerialrange.viz.video import ffmpeg

SRC = REPO_ROOT / "cache/video/showcase"
OUT = REPO_ROOT / "docs/media"
MAX_MB = 8.0
# name: (source clip, start s, duration s, width px, fps)
CLIPS = {
    "tracking_hero": ("01_tracking.mp4", 0.2, 7.0, 900, 12),
    "ranging_real_late": ("03_ranging_oblique.mp4", 13.0, 8.0, 900, 10),
    "tracking": ("01_tracking.mp4", 17.3, 6.0, 900, 12),
    "before_after": ("02_before_after.mp4", 0.0, 6.0, 900, 12),
    "range_flag": ("05_range_flag.mp4", 6.0, 7.0, 540, 12),
    "ranging_overhead": ("04_ranging_overhead.mp4", 1.0, 7.0, 800, 10),
}


def make(name: str, src: str, start: float, dur: float, width: int, fps: int) -> float:
    out = OUT / f"{name}.webp"
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur)]
    cmd += ["-i", str(SRC / src), "-vf", f"fps={fps},scale={width}:-1:flags=lanczos"]
    cmd += ["-c:v", "libwebp_anim", "-quality", "72", "-compression_level", "4", "-loop", "0"]
    cmd += ["-an", str(out)]
    subprocess.run(cmd, check=True)
    return out.stat().st_size / 1e6


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rec, ok = {}, True
    for name, spec in CLIPS.items():
        mb = make(name, *spec)
        rec[name] = {
            "source": f"cache/video/showcase/{spec[0]}",
            "start_s": spec[1],
            "seconds": spec[2],
            "width": spec[3],
            "fps": spec[4],
            "size_mb": round(mb, 2),
        }
        ok &= mb <= MAX_MB
        print(f"{name}: {mb:.1f} MB")
    (OUT / "media.json").write_text(json.dumps(rec, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
