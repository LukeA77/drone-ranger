"""Download and extract Mid-Air archives listed in the site's download-form file.

Usage: python scripts/download_midair.py [--only Kite_training/sunny] [--skip color_left]
       [--min-free-gb 20] [--limit N]

Reads `<data_root>/<midair_subdir>/download_config.txt`. Each archive is downloaded to a
`.part` file, checked, extracted into `<map>/<climate>/[<modality>/<trajectory>/]` and then
deleted. Finished items get a `.done` marker so the script can be re-run to resume.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from aerialrange.utils.config import REPO_ROOT, load_config

CHUNK_BYTES = 1 << 20


def _target_dir(base: Path, url_path: str) -> Path:
    """`Kite_training/sunny/depth/trajectory_0000/frames.zip` -> base/Kite_training/sunny/depth/
    trajectory_0000; `.../sensor_records.zip` -> base/Kite_training/sunny."""
    return base / Path(url_path).parent


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> int:
    dest = dest.resolve()
    n = 0
    for info in zf.infolist():
        out = (dest / info.filename).resolve()
        if dest not in out.parents and out != dest:
            raise RuntimeError(f"unsafe path in archive: {info.filename}")
        if info.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK_BYTES)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="keep URLs whose path starts with this")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", nargs="*", default=[], help="modalities to skip, e.g. color_left")
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="stop before an archive if free disk space is below this",
    )
    args = ap.parse_args(argv)

    cfg = load_config("data")
    base = REPO_ROOT / cfg.require("data_root") / cfg.require("midair_subdir")
    urls = [u.strip() for u in (base / "download_config.txt").read_text().splitlines()]
    items = []
    for u in urls:
        if "/MidAir/" not in u:
            continue
        path = u.split("/MidAir/", 1)[1].split("?", 1)[0]
        parts = path.split("/")
        if len(parts) > 3 and parts[2] in args.skip:
            continue
        if args.only is None or path.startswith(args.only):
            items.append((u, path))
    items = items[: args.limit]
    print(f"{len(items)} archives to process into {base}", flush=True)

    for i, (url, path) in enumerate(items, 1):
        dest = _target_dir(base, path)
        marker = dest / f".{Path(path).stem}.done"
        if marker.exists():
            print(f"[{i}/{len(items)}] skip (done) {path}", flush=True)
            continue
        free_gb = shutil.disk_usage(base).free / 2**30
        if free_gb < args.min_free_gb:
            print(f"STOPPED: {free_gb:.1f} GB free < {args.min_free_gb} GB", flush=True)
            return 2
        dest.mkdir(parents=True, exist_ok=True)
        part = dest / f"{Path(path).name}.part"
        t0 = time.time()
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(url, timeout=120) as r, part.open("wb") as f:
                    shutil.copyfileobj(r, f, CHUNK_BYTES)
                with zipfile.ZipFile(part) as zf:
                    bad = zf.testzip()
                    if bad:
                        raise RuntimeError(f"corrupt member {bad}")
                    n = _safe_extract(zf, dest)
                break
            except Exception as e:  # noqa: BLE001 - report and retry any download failure
                print(f"[{i}/{len(items)}] attempt {attempt} failed for {path}: {e}", flush=True)
                if attempt == 3:
                    part.unlink(missing_ok=True)
                    print(f"[{i}/{len(items)}] GAVE UP {path}", flush=True)
                    n = -1
        if n < 0:
            continue
        mb = part.stat().st_size / 2**20
        part.unlink()
        marker.write_text(url.split("?", 1)[0] + "\n")
        print(
            f"[{i}/{len(items)}] ok {path}: {mb:.0f} MB, {n} files, {time.time() - t0:.0f} s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
