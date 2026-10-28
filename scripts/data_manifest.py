"""Record what data was used: file counts, sizes and a SHA-256 tree hash per dataset folder
(file names, sizes and checksums for docs/model_and_data.md").

Usage: python scripts/data_manifest.py

The original archives are no longer on disk, so each extracted split folder gets one tree
hash: SHA-256 over the sorted (relative path, size, file SHA-256) records. Two copies of a
folder with identical contents give the same hash. Writes results/data_manifest.json, which
docs/model_and_data.md cites. Heavy on disk I/O; run when nothing else is reading the data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from aerialrange.utils.config import REPO_ROOT, load_config

CHUNK = 1 << 20


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def tree_manifest(folder: Path) -> dict:
    tree = hashlib.sha256()
    n = size = 0
    for p in sorted(x for x in folder.rglob("*") if x.is_file() and not x.name.startswith(".")):
        rel = p.relative_to(folder).as_posix()
        s = p.stat().st_size
        tree.update(f"{rel}\t{s}\t{file_sha256(p)}\n".encode())
        n += 1
        size += s
    return {"files": n, "bytes": size, "tree_sha256": tree.hexdigest()}


def main() -> int:
    data = load_config("data")
    root = REPO_ROOT / data.require("data_root")
    folders = []
    for sub in ("visdrone_mot_subdir", "visdrone_det_subdir"):
        folders += sorted(p for p in (root / data.require(sub)).iterdir() if p.is_dir())
    climate = root / data.require("midair_subdir") / data.require("midair_climate")
    folders += [climate / m for m in ("depth", "segmentation", "normals") if (climate / m).is_dir()]
    out = {"generated_utc": datetime.now(UTC).isoformat(timespec="seconds"), "folders": {}}
    if (climate / "sensor_records.hdf5").exists():
        h5 = climate / "sensor_records.hdf5"
        out["folders"][h5.relative_to(root).as_posix()] = {
            "files": 1,
            "bytes": h5.stat().st_size,
            "tree_sha256": file_sha256(h5),
        }
    for f in folders:
        print(f"hashing {f.relative_to(root)} ...", flush=True)
        out["folders"][f.relative_to(root).as_posix()] = tree_manifest(f)
    dest = REPO_ROOT / "results" / "data_manifest.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
