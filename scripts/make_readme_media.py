"""Short looping animated WebP clips for the README, cut from the showcase clips
(scripts/make_showcase.py) and from the 3D field-view video (scripts/make_video_field_view.py).

Usage: python scripts/make_readme_media.py [clip name ...]

GitHub shows animated images inline in a README but does not play video files from the
repository, so each stage gets a looping clip of a few seconds. Animated WebP is used rather than
GIF: the same clips are 3 to 5 times smaller at better quality. Each must stay under MAX_MB.
They hold dataset frames and are published under the datasets' terms (D-619).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from aerialrange.utils.config import REPO_ROOT
from aerialrange.viz.video import ffmpeg

SRC = REPO_ROOT / "cache/video/showcase"
OUT = REPO_ROOT / "docs/media"
MAX_MB = 8.0
# name: (source clip, start s, duration s, width px, fps)
CLIPS = {
    "tracking_hero": ("01_tracking.mp4", 0.2, 7.0, 900, 12),
    "tracking_v2_hero": ("06_tracking_v2.mp4", 0.2, 7.0, 900, 12),
    "ranging_real_late": ("03_ranging_oblique.mp4", 13.0, 8.0, 900, 10),
    "tracking": ("01_tracking.mp4", 17.3, 6.0, 900, 12),
    "before_after": ("02_before_after.mp4", 0.0, 6.0, 900, 12),
    "range_flag": ("05_range_flag.mp4", 6.0, 7.0, 540, 12),
    "ranging_overhead": ("04_ranging_overhead.mp4", 1.0, 7.0, 800, 10),
}
# Loops from the 3D field-view video, some played faster:
# name: (source, start s, duration s, width px, fps, speed, quality)
FIELD = REPO_ROOT / "cache/video/field_view.mp4"
FIELD_CLIPS = {
    "field_view": (FIELD, 0.0, 13.5, 900, 12, 1.0, 62),  # the flight up to the first pins
    "field_map": (FIELD, 13.5, 16.7, 900, 10, 1.5, 60),  # the rest of the flight
    "field_team": (FIELD, 30.2, 30.4, 900, 10, 1.6, 62),  # the team search, side by side
}


def make(
    name: str,
    src: str | Path,
    start: float,
    dur: float,
    width: int,
    fps: int,
    speed: float = 1.0,
    quality: int = 72,
) -> float:
    out = OUT / f"{name}.webp"
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    if speed != 1.0:
        vf = f"fps={fps},setpts=PTS/{speed},scale={width}:-2:flags=lanczos"
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur)]
    cmd += ["-i", str(SRC / src), "-vf", vf]
    cmd += ["-c:v", "libwebp_anim", "-quality", str(quality), "-compression_level", "4"]
    cmd += ["-loop", "0", "-an", str(out)]
    subprocess.run(cmd, check=True)
    return out.stat().st_size / 1e6


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:]) or set(CLIPS) | set(FIELD_CLIPS)  # optional: rebuild only those
    path = OUT / "media.json"
    rec, ok = (json.loads(path.read_text()) if path.exists() else {}), True
    for name, spec in CLIPS.items():
        if name not in only:
            continue
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
    for name, spec in FIELD_CLIPS.items():
        if name not in only:
            continue
        if not spec[0].exists():
            print(f"{name}: {spec[0]} missing: run scripts/make_video_field_view.py first")
            ok = False
            continue
        mb = make(name, *spec)
        rec[name] = {
            "source": spec[0].relative_to(REPO_ROOT).as_posix(),
            "start_s": spec[1],
            "seconds": spec[2],
            "width": spec[3],
            "fps": spec[4],
            "speed": spec[5],
            "size_mb": round(mb, 2),
        }
        ok &= mb <= MAX_MB
        print(f"{name}: {mb:.1f} MB")
    (OUT / "media.json").write_text(json.dumps(rec, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
