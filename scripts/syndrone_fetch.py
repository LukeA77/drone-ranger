"""Copy selected SynDrone members to the local cache (`<data_root>/syndrone`).

Examples:
    python scripts/syndrone_fetch.py --town Town01 --modality camera bboxes
    python scripts/syndrone_fetch.py --town Town01 --modality rgb --heights 80 --frames 0:3000:25

Members already present locally are skipped. Each batch checks free space first and stops
above the floor (`configs/data.yaml` disk_floor_gb), so a run can be resumed.
Every member is CRC-checked against the archive before it is written.
"""

from __future__ import annotations

import argparse
import sys
import time

from aerialrange.data.syndrone_archive import HEIGHTS, MemberStore, free_gb, member_name
from aerialrange.utils.config import REPO_ROOT, load_config


def frames_arg(s: str) -> range:
    a, b, *c = (int(x) for x in s.split(":"))
    return range(a, b, c[0] if c else 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument(
        "--modality",
        nargs="+",
        required=True,
        choices=["rgb", "semantic", "depth", "camera", "bboxes"],
    )
    ap.add_argument("--heights", nargs="+", type=int, default=list(HEIGHTS))
    ap.add_argument("--frames", type=frames_arg, default=range(0, 3000))
    ap.add_argument("--batch", type=int, default=300, help="members per free-space check")
    args = ap.parse_args(argv)
    data = load_config("data")
    root = REPO_ROOT / data.require("data_root") / data.require("syndrone_subdir")
    floor = float(data.require("disk_floor_gb"))
    store = MemberStore(root, data.require("syndrone_base_url"))
    names: list[str] = []
    for m in args.modality:
        for f in args.frames:
            if m == "bboxes":
                names.append(member_name(args.town, m, f))
            else:
                names += [member_name(args.town, m, f, h) for h in args.heights]
    names = [n for n in dict.fromkeys(names) if not store.local_path(n).exists()]
    print(f"{len(names)} members to fetch; {free_gb(root):.1f} GB free (floor {floor} GB)")
    t0, done, nbytes = time.time(), 0, 0
    for i in range(0, len(names), args.batch):
        if free_gb(root) < floor:
            print(f"stopped: free space below {floor} GB; rerun to resume")
            return 2
        for _, d in store.iter_read(names[i : i + args.batch], keep=True):
            done += 1
            nbytes += len(d)
        rate = nbytes / 1e6 / max(time.time() - t0, 1e-9)
        print(f"  {done}/{len(names)}  {nbytes / 1e9:.2f} GB  {rate:.1f} MB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
