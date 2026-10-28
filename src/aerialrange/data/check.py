"""Presence, count and schema checks for the datasets.

Each dataset can be present as an extracted folder or as the original zip archive. Zips are
read in place (listing and annotation text only); nothing is extracted or modified.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

# VisDrone MOT annotation: frame, target id, left, top, width, height, score, category,
# truncation, occlusion. Category 0 = ignored region; 1..11 classes.
VISDRONE_MOT_COLUMNS = 10
# VisDrone DET annotation: left, top, width, height, score, category, truncation, occlusion.
# This column layout is an expectation to verify, not a verified fact; a mismatch is reported.
VISDRONE_DET_COLUMNS = 8
VISDRONE_CATEGORIES = range(0, 12)

VISDRONE_MOT_SPLITS = (
    "VisDrone2019-MOT-train",
    "VisDrone2019-MOT-val",
    "VisDrone2019-MOT-test-dev",
)
VISDRONE_DET_SPLITS = ("VisDrone2019-DET-train", "VisDrone2019-DET-val")
MIDAIR_HDF5_GROUPS = ("camera_data", "gps", "groundtruth", "imu")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass
class CheckResult:
    """Outcome of checking one dataset component."""

    name: str
    status: str  # "ok", "missing", "partial", "schema_error"
    details: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


class Tree:
    """Read-only view over a folder or a zip archive, with POSIX relative paths."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.is_zip = root.suffix.lower() == ".zip"
        self._zip = zipfile.ZipFile(root) if self.is_zip else None

    def files(self) -> Iterator[PurePosixPath]:
        if self._zip is not None:
            for n in self._zip.namelist():
                if not n.endswith("/"):
                    yield PurePosixPath(n)
        else:
            for p in self.root.rglob("*"):
                if p.is_file():
                    yield PurePosixPath(p.relative_to(self.root.parent).as_posix())

    def read_text(self, path: PurePosixPath) -> str:
        if self._zip is not None:
            return self._zip.read(str(path)).decode("utf-8", errors="replace")
        return (self.root.parent / Path(path)).read_text(encoding="utf-8", errors="replace")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def find_source(base: Path, name: str) -> Path | None:
    """Return `base/name` if it is a folder, else `base/name.zip` if present, else None."""
    if (base / name).is_dir():
        return base / name
    if (base / f"{name}.zip").is_file():
        return base / f"{name}.zip"
    return None


def _parse_rows(text: str, n_cols: int, where: str, problems: list[str]) -> list[list[int]]:
    rows: list[list[int]] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if fields and fields[-1] == "":
            fields = fields[:-1]  # tolerate one trailing comma
        if len(fields) != n_cols:
            problems.append(f"{where}:{i}: {len(fields)} columns, expected {n_cols}")
            continue
        try:
            rows.append([int(f) for f in fields])
        except ValueError:
            problems.append(f"{where}:{i}: non-integer field in {line!r}")
    return rows


def _cap(problems: list[str], limit: int = 20) -> list[str]:
    if len(problems) <= limit:
        return problems
    return problems[:limit] + [f"... and {len(problems) - limit} more"]


def check_visdrone_mot(base: Path, split: str) -> CheckResult:
    """Check one VisDrone MOT split: sequences, frames and annotation schema."""
    src = find_source(base, split)
    if src is None:
        return CheckResult(
            split, "missing", {"looked_for": [str(base / split), str(base / f"{split}.zip")]}
        )
    tree = Tree(src)
    try:
        frames: dict[str, int] = {}
        ann_files: dict[str, PurePosixPath] = {}
        for p in tree.files():
            parts = p.parts
            if len(parts) >= 4 and parts[1] == "sequences" and p.suffix.lower() in IMAGE_SUFFIXES:
                frames[parts[2]] = frames.get(parts[2], 0) + 1
            elif len(parts) == 3 and parts[1] == "annotations" and p.suffix == ".txt":
                ann_files[p.stem] = p
        problems: list[str] = []
        seqs, anns = set(frames), set(ann_files)
        for s in sorted(seqs - anns):
            problems.append(f"sequence {s} has no annotation file")
        for s in sorted(anns - seqs):
            problems.append(f"annotation {s} has no sequence frames")
        n_boxes = n_ignored = 0
        categories: set[int] = set()
        for stem, path in sorted(ann_files.items()):
            rows = _parse_rows(tree.read_text(path), VISDRONE_MOT_COLUMNS, path.name, problems)
            for r in rows:
                cat = r[7]
                categories.add(cat)
                if cat not in VISDRONE_CATEGORIES:
                    problems.append(f"{path.name}: category {cat} outside 0..11")
                n_ignored += cat == 0
            n_boxes += len(rows)
            if rows and stem in frames and max(r[0] for r in rows) > frames[stem]:
                problems.append(
                    f"{path.name}: max frame index {max(r[0] for r in rows)} > "
                    f"{frames[stem]} frames on disk"
                )
        details = {
            "source": str(src),
            "format": "zip" if tree.is_zip else "folder",
            "sequences": len(seqs),
            "annotation_files": len(anns),
            "frames": sum(frames.values()),
            "boxes": n_boxes,
            "ignored_region_rows": n_ignored,
            "categories_seen": sorted(categories),
        }
    finally:
        tree.close()
    status = "schema_error" if problems else "ok"
    return CheckResult(split, status, details, _cap(problems))


def check_visdrone_det(base: Path, split: str) -> CheckResult:
    """Check one VisDrone image-detection split: images, annotations and schema."""
    src = find_source(base, split)
    if src is None:
        return CheckResult(
            split, "missing", {"looked_for": [str(base / split), str(base / f"{split}.zip")]}
        )
    tree = Tree(src)
    try:
        images: set[str] = set()
        ann_files: dict[str, PurePosixPath] = {}
        for p in tree.files():
            parts = p.parts
            if len(parts) == 3 and parts[1] == "images" and p.suffix.lower() in IMAGE_SUFFIXES:
                images.add(p.stem)
            elif len(parts) == 3 and parts[1] == "annotations" and p.suffix == ".txt":
                ann_files[p.stem] = p
        problems: list[str] = []
        missing_ann = images - set(ann_files)
        if missing_ann:
            problems.append(f"{len(missing_ann)} images have no annotation file")
        n_boxes = 0
        for path in ann_files.values():
            rows = _parse_rows(tree.read_text(path), VISDRONE_DET_COLUMNS, path.name, problems)
            n_boxes += len(rows)
            problems.extend(
                f"{path.name}: category {r[5]} outside 0..11"
                for r in rows
                if r[5] not in VISDRONE_CATEGORIES
            )
        details = {
            "source": str(src),
            "format": "zip" if tree.is_zip else "folder",
            "images": len(images),
            "annotation_files": len(ann_files),
            "boxes": n_boxes,
        }
    finally:
        tree.close()
    status = "schema_error" if problems else "ok"
    return CheckResult(split, status, details, _cap(problems))


def parse_midair_download_list(path: Path) -> dict[str, dict[str, set[str]]]:
    """Read the Mid-Air download-form URL list into {"map/climate": {modality: {trajectory}}}.

    Only the URL path is used; the query string (a download token) is discarded.
    """
    expected: dict[str, dict[str, set[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().split("?", 1)[0]
        if "/MidAir/" not in line:
            continue
        parts = line.split("/MidAir/", 1)[1].split("/")
        if len(parts) < 3:
            continue
        key = f"{parts[0]}/{parts[1]}"
        mods = expected.setdefault(key, {})
        if parts[2] == "sensor_records.zip":
            mods.setdefault("sensor_records", set())
        elif len(parts) >= 4:
            mods.setdefault(parts[2], set()).add(parts[3])
    return expected


def _check_hdf5(path: Path) -> tuple[dict[str, Any], list[str]]:
    import h5py

    problems: list[str] = []
    with h5py.File(path, "r") as f:
        trajs = sorted(f.keys())
        for t in trajs:
            missing = [g for g in MIDAIR_HDF5_GROUPS if g not in f[t]]
            if missing:
                problems.append(f"{path.name}/{t}: missing groups {missing}")
    return {"trajectories_in_hdf5": len(trajs)}, problems


def check_midair(
    base: Path, climates: list[str] | None = None, modalities: list[str] | None = None
) -> list[CheckResult]:
    """Compare Mid-Air on disk with the download list (if present), optionally restricted to
    the configured climates and the modalities the evaluation needs."""
    if not base.is_dir():
        return [CheckResult("midair", "missing", {"looked_for": str(base)})]
    dl = base / "download_config.txt"
    expected = parse_midair_download_list(dl) if dl.is_file() else {}
    if not expected:
        # No download list: discover climates from sensor_records.hdf5 files.
        for h5 in base.rglob("sensor_records.hdf5"):
            rel = h5.parent.relative_to(base).as_posix()
            expected.setdefault(rel, {"sensor_records": set()})
    if climates is not None:
        expected = {k: v for k, v in expected.items() if k in climates}
    if modalities is not None:
        expected = {
            k: {m: s for m, s in v.items() if m in modalities or m == "sensor_records"}
            for k, v in expected.items()
        }
    if not expected:
        return [CheckResult("midair", "missing", {"looked_for": str(base), "found": "nothing"})]
    results = []
    for key, mods in sorted(expected.items()):
        cdir = base / key
        problems: list[str] = []
        details: dict[str, Any] = {"folder": str(cdir)}
        h5 = cdir / "sensor_records.hdf5"
        if h5.is_file():
            info, probs = _check_hdf5(h5)
            details.update(info)
            problems.extend(probs)
            details["sensor_records"] = "hdf5"
        elif (cdir / "sensor_records.zip").is_file():
            details["sensor_records"] = "zip only (not extracted)"
            problems.append("sensor_records.zip present but not extracted")
        else:
            details["sensor_records"] = "missing"
            problems.append("sensor_records.hdf5 missing")
        n_expected = n_present = 0
        for mod, trajs in sorted(mods.items()):
            if mod == "sensor_records":
                continue
            present = 0
            frames = 0
            for t in sorted(trajs):
                tdir = cdir / mod / t
                imgs = (
                    [p for p in tdir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
                    if tdir.is_dir()
                    else []
                )
                if imgs:
                    present += 1
                    frames += len(imgs)
                elif (tdir / "frames.zip").is_file():
                    problems.append(f"{mod}/{t}: frames.zip present but not extracted")
            n_expected += len(trajs)
            n_present += present
            details[mod] = {
                "trajectories_expected": len(trajs),
                "present": present,
                "frames": frames,
            }
        if n_present == 0 and details["sensor_records"] == "missing":
            status = "missing"
        elif problems or n_present < n_expected:
            status = "partial"
        else:
            status = "ok"
        if n_present < n_expected:
            problems.append(f"{n_expected - n_present} of {n_expected} trajectory folders missing")
        results.append(CheckResult(f"midair/{key}", status, details, _cap(problems)))
    return results


def check_auair(base: Path | None) -> CheckResult:
    """AU-AIR presence only; its schema is discovered in Phase 4."""
    if base is None:
        return CheckResult(
            "auair", "missing", {"note": "data.auair_subdir is null (not downloaded)"}
        )
    if not base.is_dir() or not any(base.iterdir()):
        return CheckResult("auair", "missing", {"looked_for": str(base)})
    n_files = sum(1 for p in base.rglob("*") if p.is_file())
    return CheckResult(
        "auair",
        "ok",
        {"folder": str(base), "files": n_files, "note": "presence only; schema checked in Phase 4"},
    )


def check_all(
    data_root: Path,
    visdrone_mot: Path,
    visdrone_det: Path,
    midair: Path,
    auair: Path | None,
    midair_climates: list[str] | None = None,
    midair_modalities: list[str] | None = None,
) -> list[CheckResult]:
    """Run every check. `data_root` is reported; the other paths are dataset folders."""
    results = [check_visdrone_mot(visdrone_mot, s) for s in VISDRONE_MOT_SPLITS]
    results += [check_visdrone_det(visdrone_det, s) for s in VISDRONE_DET_SPLITS]
    results += check_midair(midair, midair_climates, midair_modalities)
    results.append(check_auair(auair))
    return results
