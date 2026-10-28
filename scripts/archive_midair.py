"""Move Mid-Air trajectories between local disk and an archive folder, verified file by file.

The archive is any folder (for example a synced cloud drive), set with the environment variable
AERIALRANGE_ARCHIVE_ROOT; frames go to <archive>/midair/<climate>/.

Usage:
  python scripts/archive_midair.py --to-drive --splits test calibration
  python scripts/archive_midair.py --restore --splits test

--to-drive: for each COMPLETE trajectory of the given splits (all three modality markers
present, no partial downloads), copy every frame of each modality folder to the archive, check size
and SHA-256 of the copy against the original, then delete the local frame. The small
`.frames.done` markers stay on the laptop so the downloader does not fetch the data again; an
`.archived` marker records where the frames went. If the archive is a Google Drive folder,
pauses while Drive's local upload cache is growing (backlog above its starting size), so the
local disk cannot fill up.
--restore: copy archived frames back, verified the same way, and remove the `.archived` marker
(the archive copy is kept).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aerialrange.utils.config import REPO_ROOT, load_config

ARCHIVE_ENV = "AERIALRANGE_ARCHIVE_ROOT"
ARCHIVE_ROOT = Path(os.environ[ARCHIVE_ENV]) / "midair" if os.environ.get(ARCHIVE_ENV) else None
MODALITIES = ("depth", "segmentation", "normals", "color_left")
REQUIRED = ("depth", "segmentation", "normals")
LOG = REPO_ROOT / "cache" / "midair_archive_log.json"
CHUNK = 1 << 20


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def drive_cache_gb() -> float:
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "DriveFS"
    if not root.exists():
        return 0.0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**30


def wait_for_drive(baseline_gb: float, max_growth_gb: float, min_free_gb: float) -> None:
    """Pause while Drive's local cache has grown more than `max_growth_gb` above its size when
    this run started (a real upload backlog; Drive also keeps already-uploaded files cached),
    or while the laptop has less than `min_free_gb` free."""
    while True:
        growth = drive_cache_gb() - baseline_gb
        free = shutil.disk_usage(REPO_ROOT).free / 2**30
        if growth <= max_growth_gb and free >= min_free_gb:
            return
        print(
            f"  waiting for Drive upload: backlog {growth:.1f} GB, laptop free {free:.1f} GB",
            flush=True,
        )
        time.sleep(60)


def move_verified(src: Path, dst: Path) -> None:
    """Copy src to dst, verify size and SHA-256, then delete src."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if dst.stat().st_size != src.stat().st_size or sha256(dst) != sha256(src):
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"verification failed for {src}")
    src.unlink()


def restore_verified(src: Path, dst: Path, attempts: int = 5, wait_s: float = 30.0) -> bool:
    """Copy a Drive file back and verify size and SHA-256. A local copy that already matches is
    kept (an interrupted restore resumes). Transient Drive errors (for example timeouts) are
    retried. Returns True if a copy was made."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size and sha256(dst) == sha256(src):
        return False
    for k in range(attempts):
        try:
            shutil.copy2(src, dst)
            if dst.stat().st_size == src.stat().st_size and sha256(dst) == sha256(src):
                return True
            raise RuntimeError(f"verification failed restoring {src}")
        except OSError:
            if k == attempts - 1:
                raise
            print(f"  Drive error on {src.name}, retrying in {wait_s:.0f} s", flush=True)
            time.sleep(wait_s)
    return True


def trajectory_complete(local: Path, name: str) -> bool:
    for m in REQUIRED:
        d = local / m / name
        if not (d / ".frames.done").exists() or any(d.glob("*.part")):
            return False
    return True


def select_names(
    splits: dict[str, list[str]], chosen: list[str], only: list[str] | None
) -> list[str]:
    """Trajectories of the chosen splits, optionally restricted to `only` (each of which must
    belong to one of those splits, so a typo cannot silently move nothing)."""
    names = [n for s in chosen for n in splits[s]]
    if only is None:
        return names
    stray = sorted(set(only) - set(names))
    if stray:
        raise ValueError(f"not in splits {chosen}: {stray}")
    return [n for n in names if n in set(only)]


def release_trajectory(local: Path, remote: Path, name: str) -> dict[str, int]:
    """Free laptop space after a restored trajectory has been processed (batch test runs).

    For each Drive-backed modality, the local frames are deleted only if the Drive folder holds
    a file of the same name and size for every local frame; then an `.archived` marker is
    written, exactly as after `--to-drive`. Colour frames (downloaded from the dataset site,
    not kept on Drive) are deleted with a `.released` marker; the downloader can fetch them
    again. Raises before deleting anything if any Drive copy is missing or differs in size.
    """
    plan = []
    for m in MODALITIES:
        d = local / m / name
        if not d.is_dir():
            continue
        files = [f for f in sorted(d.iterdir()) if f.is_file() and not f.name.startswith(".")]
        if m in REQUIRED:
            r = remote / m / name
            for f in files:
                c = r / f.name
                if not c.is_file() or c.stat().st_size != f.stat().st_size:
                    raise RuntimeError(f"{name}/{m}: no matching Drive copy of {f.name}")
            plan.append((d, files, ".archived", str(r) + "\n"))
        else:
            plan.append((d, files, ".released", "colour frames deleted after processing\n"))
    counts = {}
    for d, files, marker, text in plan:
        for f in files:
            f.unlink()
        (d / marker).write_text(text)
        counts[d.parent.name] = len(files)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--to-drive", action="store_true")
    g.add_argument("--restore", action="store_true")
    g.add_argument("--release", action="store_true", help="delete local copies kept on Drive")
    g.add_argument(
        "--prepare-download",
        action="store_true",
        help="clear the archived and done markers so the downloader fetches these again from "
        "the dataset site (faster than reading back from Drive; same files)",
    )
    ap.add_argument(
        "--splits", nargs="+", required=True, choices=["development", "calibration", "test"]
    )
    ap.add_argument("--trajectories", nargs="+", default=None, help="only these, within --splits")
    ap.add_argument("--max-backlog-gb", type=float, default=6.0)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    args = ap.parse_args(argv)

    data = load_config("data")
    climate = data.require("midair_climate")
    local = REPO_ROOT / data.require("data_root") / data.require("midair_subdir") / climate
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["midair"]["splits"]
    names = select_names(splits, args.splits, args.trajectories)
    log = json.loads(LOG.read_text()) if LOG.exists() else []
    if args.prepare_download:
        for name in names:
            for m in REQUIRED:
                for marker in (".archived", ".frames.done"):
                    (local / m / name / marker).unlink(missing_ok=True)
            print(f"{name}: markers cleared for re-download", flush=True)
        return 0
    if ARCHIVE_ROOT is None:
        print(f"Set {ARCHIVE_ENV} to the archive folder first.")
        return 1
    remote = ARCHIVE_ROOT / climate
    if not ARCHIVE_ROOT.parent.exists():
        print(f"Archive folder not found: {ARCHIVE_ROOT.parent}")
        return 1
    if args.release:
        # Only trajectories this script has put on Drive (or brought back from it); the release
        # itself checks a same-name, same-size Drive copy of every frame before deleting.
        known = {e["trajectory"] for e in log if e["action"] in ("to_drive", "restored")}
        for name in names:
            if name not in known:
                print(f"{name}: no Drive copy recorded by this script, not released")
                continue
            counts = release_trajectory(local, remote, name)
            log.append(
                {
                    "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "trajectory": name,
                    "action": "released",
                    "files": sum(counts.values()),
                }
            )
            LOG.write_text(json.dumps(log, indent=1) + "\n")
            print(f"{name}: released {counts}", flush=True)
        return 0
    baseline = drive_cache_gb()
    print(f"Drive cache at start: {baseline:.1f} GB (already-uploaded files, not a backlog)")

    for name in names:
        present = [m for m in MODALITIES if (local / m / name).is_dir()]
        marked = [m for m in present if (local / m / name / ".archived").exists()]
        archived = bool(marked)  # at least partly on Drive (restore brings back what is there)
        if args.to_drive:
            # Done only when EVERY modality folder is marked; a run interrupted between
            # modalities resumes with the rest instead of skipping the trajectory.
            if present and len(marked) == len(present):
                print(f"{name}: already on Drive")
                continue
            if not trajectory_complete(local, name):
                print(f"{name}: not completely downloaded yet, skipped")
                continue
        elif not archived:
            print(f"{name}: not archived, nothing to restore")
            continue
        n_files = n_bytes = 0
        for m in MODALITIES:
            a, b = (
                (local / m / name, remote / m / name)
                if args.to_drive
                else (remote / m / name, local / m / name)
            )
            if not a.is_dir():
                continue
            if args.to_drive:
                wait_for_drive(baseline, args.max_backlog_gb, args.min_free_gb)
            files = [f for f in sorted(a.iterdir()) if f.is_file() and not f.name.startswith(".")]
            for f in files:
                n_bytes += f.stat().st_size
                if args.to_drive:
                    move_verified(f, b / f.name)
                else:
                    restore_verified(f, b / f.name)
                n_files += 1
            marker = local / m / name / ".archived"
            if args.to_drive:
                marker.write_text(str(b) + "\n")
            else:
                marker.unlink(missing_ok=True)
        action = "to_drive" if args.to_drive else "restored"
        log.append(
            {
                "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "trajectory": name,
                "action": action,
                "files": n_files,
                "gb": round(n_bytes / 2**30, 3),
            }
        )
        LOG.write_text(json.dumps(log, indent=1) + "\n")
        print(f"{name}: {action}, {n_files} files, {n_bytes / 2**30:.2f} GB verified", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
