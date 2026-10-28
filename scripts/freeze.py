"""Freeze the settings of a phase before its one-time test run, and check the freeze.

Usage: python scripts/freeze.py --phase phase2            (write the manifest)
       python scripts/freeze.py --phase phase2 --check    (exit 1 if anything changed)

The manifest (results/frozen/<phase>.json) records the SHA-256 of every config and fitted-value
file the phase's test run reads, the git commit, and whether the working tree had uncommitted
changes to those files. `--check` recomputes the hashes; the test-run queue runs it first and
stops if any file differs, so settings cannot drift between freezing and testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime

from aerialrange.utils.config import REPO_ROOT

PHASES = {
    "phase2": [
        "configs/data.yaml",
        "configs/geometry.yaml",
        "configs/uncertainty.yaml",
        "results/splits.json",
        "results/midair/calibration_fit.json",
        "results/midair/terrain_calibration_fit.json",
    ],
    "phase4_down": [
        "configs/auair.yaml",
        "configs/trackers/botsort_cmc_tuned_B2_yolo26n_960.yaml",
        "results/auair/clip_labels.csv",
        "results/auair/camera_scale.json",
        "results/auair/repeatability_calibration.json",
        "results/visdrone/val/pixel_noise_B2_yolo26n_960.json",
    ],
    "phase4_view": [
        "configs/auair.yaml",
        "configs/trackers/botsort_cmc_tuned_B2_yolo26n_960.yaml",
        "results/auair/clip_labels.csv",
        "results/auair/camera_scale.json",
        "results/auair/repeatability_calibration.json",
        "results/auair/tilt_summary.json",
        "results/auair/stationary_B2_yolo26n_960.json",
        "results/auair/stationary_truth.json",
        "results/visdrone/val/pixel_noise_B2_yolo26n_960.json",
        "scripts/auair_repeatability_view.py",
        "scripts/auair_tilt.py",
        "scripts/auair_stationary.py",
        "src/aerialrange/evaluate/auair_world.py",
        "src/aerialrange/evaluate/tilt_motion.py",
        "src/aerialrange/evaluate/stationary.py",
    ],
    "phase3": [
        "configs/detector.yaml",
        "configs/tracker.yaml",
        "configs/trackers/botsort_cmc.yaml",
    ],
}


def sha256(rel: str) -> str:
    return hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)
    return r.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--extra", nargs="*", default=[], help="more files (e.g. chosen weights)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    dest = REPO_ROOT / "results" / "frozen" / f"{args.phase}.json"
    if args.check:
        m = json.loads(dest.read_text())
        changed = [f for f, h in m["files"].items() if sha256(f) != h]
        if changed:
            print(f"FROZEN SETTINGS CHANGED since {m['frozen_utc']}: {changed}")
            return 1
        print(f"{args.phase}: {len(m['files'])} files unchanged since {m['frozen_utc']}")
        return 0
    files = PHASES[args.phase] + args.extra
    dirty = git("status", "--porcelain", "--", *files)
    m = {
        "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "phase": args.phase,
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes_to_frozen_files": dirty.splitlines(),
        "files": {f: sha256(f) for f in files},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
