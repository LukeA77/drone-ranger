"""Write the schema.json of a results folder from its file kinds (build2 Section 15).

Usage: python scripts/results_schema.py [--check] [folder ...]   (default: every folder in KINDS)

Each folder's kinds of file are listed below: a file-name pattern (matched against the path
inside the folder, first match wins), a definition name, the script that writes it and what it
holds. Each definition is inferred from every existing file of that kind:
- keys present in every file are required, and each key's type is the union of the types seen
  (null allowed where any file has null);
- objects keyed by data (towns, sequences, heights, bins, flights, seeds) are checked as maps:
  every value against one merged definition;
- numbers named as shares, probabilities, coverage, AP, IDF1, HOTA, recall or precision must
  lie in 0 to 1, and counts, sizes, times and absolute errors must not be negative, wherever
  every existing value already does (otherwise the rule is not applied to that key);
- objects deeper than four levels are checked for type only.
So the schema records the contract the current files meet, and tests/test_results_schema.py
checks every file, including future ones, against it; a file that matches no pattern fails.
--check writes nothing and exits 1 if a schema would change.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerialrange.utils.config import REPO_ROOT  # noqa: E402

Kind = tuple[str, str, str]  # pattern, definition, description

KINDS: dict[str, dict[str, Any]] = {
    "syndrone": {
        "title": "SynDrone end-to-end results (Phases 7 to 10)",
        "kinds": [
            (
                "*/eval_*.json",
                "eval",
                "scripts/syndrone_evaluate.py: detection, tracking, range and speed per group, "
                "pooled with 95 % intervals resampling sequences and towns, by height and per "
                "sequence",
            ),
            (
                "*/search_*.json",
                "search",
                "scripts/syndrone_search_metrics.py: ground searched, false alarms per km2 and "
                "share of people found (D-749)",
            ),
            (
                "*/breakdowns_*.json",
                "breakdowns",
                "scripts/syndrone_breakdowns.py: range and speed error by camera angle, true "
                "range, object size, height and class",
            ),
            (
                "*/compare_*.json",
                "compare",
                "scripts/syndrone_compare.py: one run against another on the same sequences",
            ),
            (
                "*/speed_display_fit_*.json",
                "speed_display_fit",
                "scripts/syndrone_speed_display_fit.py: when a speed is shown (D-710, D-739)",
            ),
            (
                "*/speed_truth_*.json",
                "speed_truth",
                "scripts/syndrone_speed_truth.py: speed truth per town (D-703)",
            ),
            (
                "*/speed_compare_*.json",
                "speed_compare",
                "scripts/syndrone_speed_compare.py: speed model comparisons (D-735)",
            ),
            (
                "*/z0_check_*.json",
                "z0_check",
                "scripts/syndrone_z0_check.py: base-height fit per town (D-706)",
            ),
            (
                "*/z0_walkers_check.json",
                "z0_walkers_check",
                "scripts/syndrone_z0_walkers.py: base height from walkers (D-720)",
            ),
            (
                "*/interval_calibration_*.json",
                "interval_calibration",
                "scripts/syndrone_interval_calibration.py: range and speed interval calibration "
                "(D-737)",
            ),
            (
                "*/contact_fit_*.json",
                "contact_fit",
                "scripts/syndrone_contact_fit.py: contact-point correction (D-743)",
            ),
            (
                "*/footprint_sizes.json",
                "footprint_sizes",
                "scripts/syndrone_footprint_sizes.py: object footprints (D-736)",
            ),
            (
                "*/imm_fit.json",
                "imm_fit",
                "scripts/syndrone_imm_fit.py: motion-model parameters (D-728)",
            ),
            (
                "*/imm_grid*.json",
                "imm_grid",
                "scripts/syndrone_imm_grid.py: motion-model grid (D-728, D-746)",
            ),
        ],
    },
    "midair": {
        "title": "Mid-Air results: version 1 ranging (Phases 2 and 4) and version 2 fusion (Phase "
        "11)",
        "kinds": [
            (
                "*/navigation/ranging_*.json",
                "nav_ranging",
                "scripts/msckf_ranging.py: terrain-from-motion ranging with fused, exact and "
                "assumed poses (D-741)",
            ),
            (
                "*/navigation/openvins_compare_*.json",
                "openvins_compare",
                "scripts/msckf_openvins_compare.py: against OpenVINS (D-741)",
            ),
            (
                "*/navigation/gps_gate_check.json",
                "gps_gate_check",
                "scripts/msckf_gps_gate_check.py: the GPS outlier gate rule (D-744)",
            ),
            (
                "*/navigation/gnss_check.json",
                "gnss_check",
                "scripts/msckf_gnss_check.py: GPS jumps in the data",
            ),
            (
                "*/navigation/gnss_fit.json",
                "gnss_fit",
                "scripts/msckf_gnss_fit.py: GPS noise model",
            ),
            (
                "*/navigation/imu_frames.json",
                "imu_frames",
                "scripts/msckf_imu_frames.py: IMU frame check",
            ),
            (
                "*/navigation/imu_noise_check.json",
                "imu_noise_check",
                "scripts/msckf_noise_check.py: IMU noise check",
            ),
            (
                "*/navigation/*.json",
                "nav_run",
                "scripts/msckf_midair.py: one filter run, per flight and pooled (D-741)",
            ),
        ],
        "v1": "version 1 result (Phases 2 and 4), scripts/midair_*.py and scripts/terrain_*.py; "
        "docs/limitations.md and the version 1 sections of the README",
    },
    "edge": {
        "title": "Edge results: speed, parity, quantisation and the cross-platform check (Phases "
        "5, 8 and 12)",
        "kinds": [
            (
                "benchmark_*.json",
                "benchmark",
                "scripts/benchmark_pipeline.py, benchmark_threaded.py, benchmark_v2.py: pipeline "
                "timing on one machine (D-724)",
            ),
            (
                "parity_*.json",
                "parity",
                "scripts/export_*.py and check_yolo_onnx.py: ONNX against the training code on "
                "the same frames",
            ),
            ("quantize_*.json", "quantize", "scripts/quantize_onnx.py: INT8 versions (D-731)"),
            (
                "platform_compare_*.json",
                "platform_compare",
                "scripts/platform_check.py --compare: x86-64 against ARM64 outputs (D-726)",
            ),
            (
                "platform_*.json",
                "platform",
                "scripts/platform_check.py: outputs on one machine (D-726)",
            ),
            (
                "pi_estimate.json",
                "pi_estimate",
                "scripts/estimate_pi.py: Raspberry Pi 5 speed, estimated, not measured (D-723)",
            ),
            ("candidates.json", "candidates", "scripts/candidates.py: detector candidates (D-729)"),
            ("pareto.json", "pareto", "scripts/pareto.py: version 1 speed and accuracy frontier"),
            ("ranging_kalman_cost.json", "ranging_kalman_cost", "time cost of the range filter"),
        ],
    },
    "thermal": {
        "title": "Thermal detection results (Phase 13, D-751)",
        "kinds": [
            (
                "detect_*.json",
                "detect",
                "scripts/hituav_eval.py: detection AP on HIT-UAV per group, validation or (once) "
                "test",
            ),
        ],
    },
    "birdsai": {
        "title": "Thermal tracking on BIRDSAI (Phase 13, D-754)",
        "kinds": [
            (
                "*/grid_*.json",
                "grid",
                "scripts/track_grid.py --dataset birdsai: the tracker threshold grid for people on "
                "the training videos, each point's metrics and the selected point (D-733 rule)",
            ),
        ],
    },
    "hituav": {
        "title": "HIT-UAV split (Phase 13, D-751)",
        "kinds": [
            (
                "splits.json",
                "splits",
                "scripts/hituav_coco_package.py: the day split and the official split",
            ),
        ],
    },
}

PROB = re.compile(
    r"(^|_)(ap50|ap|idf1|hota|coverage|share|recall|precision|p_found|accuracy|agreement|pd|found|fraction)"
    r"(_|$)|^(mota_|mean_idf1)",
    re.I,
)
NONNEG = re.compile(
    r"(^|_)(n|count|counts|switches|frames|km2|ms|seconds|s|mb|bytes|size|sizes|abs|width|height|px)(_|$)",
    re.I,
)
DATA_KEY = re.compile(
    r"^(Town\w+|h\d+|\d[\d.\-]*(-inf|-\d[\d.]*)?|uav\d+\w*|trajectory_\d+|\w+_\d{3,}|seed\d*|"
    r"(sunny|cloudy|foggy|sunset)_\w+|PLE_\w+|kite_\w+|0\.\d+|[\w.]+_vs_[\w.]+)$"
)
MAX_DEPTH = 4


def _types(values: list[Any]) -> list[str]:
    out = []
    for v in values:
        if v is None:
            t = "null"
        elif isinstance(v, bool):
            t = "boolean"
        elif isinstance(v, int):
            t = "integer"
        elif isinstance(v, float):
            t = "number"
        elif isinstance(v, str):
            t = "string"
        elif isinstance(v, list):
            t = "array"
        else:
            t = "object"
        out.append(t)
    s = set(out)
    if {"integer", "number"} <= s:
        s.discard("integer")
    order = ["object", "array", "string", "number", "integer", "boolean", "null"]
    return [t for t in order if t in s]


def infer(values: list[Any], key: str = "", depth: int = 0) -> dict[str, Any]:
    types = _types(values)
    node: dict[str, Any] = {"type": types[0] if len(types) == 1 else types}
    present = [v for v in values if v is not None]
    nums = [v for v in present if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if nums and len(nums) == len(present):
        if PROB.search(key) and all(0 <= v <= 1 for v in nums):
            node.update(minimum=0, maximum=1)
        elif NONNEG.search(key) and all(v >= 0 for v in nums):
            node["minimum"] = 0
    dicts = [v for v in present if isinstance(v, dict)]
    if dicts and depth < MAX_DEPTH:
        keys = {k for d in dicts for k in d}
        if keys and sum(bool(DATA_KEY.match(k)) for k in keys) >= max(1, len(keys) // 2):
            node["additionalProperties"] = infer(
                [v for d in dicts for v in d.values()], key, depth + 1
            )
        else:
            node["required"] = sorted(set.intersection(*(set(d) for d in dicts)))
            node["properties"] = {
                k: infer([d[k] for d in dicts if k in d], k, depth + 1) for k in sorted(keys)
            }
    lists = [v for v in present if isinstance(v, list)]
    items = [x for v in lists for x in v]
    if (
        lists
        and items
        and depth < MAX_DEPTH
        and all(not isinstance(x, (dict, list)) for x in items)
    ):
        node["items"] = infer(items, key, depth + 1)
    return node


def _v1_kinds(folder: Path) -> list[Kind]:
    """One kind per version 1 file name, deepest folder first (Mid-Air only)."""
    out: dict[str, Kind] = {}
    for f in sorted(folder.rglob("*.json")):
        rel = f.relative_to(folder).as_posix()
        if "/navigation/" in rel or f.name == "schema.json":
            continue
        parts = rel.split("/")
        if len(parts) == 1:
            pat, name = rel, f"v1_{f.stem}"
        else:
            sub = "/".join(parts[1:-1])
            pat = f"*/{sub + '/' if sub else ''}{f.name}"
            name = "v1_" + (sub.replace("/", "_") + "_" if sub else "") + f.stem
        out[pat] = (pat, name, "")
    return sorted(out.values(), key=lambda k: -k[0].count("/"))


def build(name: str) -> dict[str, Any]:
    spec = KINDS[name]
    folder = REPO_ROOT / "results" / name
    kinds: list[Kind] = list(spec["kinds"])
    if "v1" in spec:
        kinds += [(p, n, spec["v1"]) for p, n, _ in _v1_kinds(folder)]
    files: dict[str, list[Any]] = {n: [] for _, n, _ in kinds}
    for f in sorted(folder.rglob("*.json")):
        rel = f.relative_to(folder).as_posix()
        if rel == "schema.json":
            continue
        hit = next((n for p, n, _ in kinds if fnmatch.fnmatch(rel, p)), None)
        if hit is None:
            raise SystemExit(f"results/{name}/{rel}: no kind matches; add one to KINDS")
        files[hit].append(json.loads(f.read_text(encoding="utf-8")))
    defs = {}
    for _, n, desc in kinds:
        if not files[n]:
            raise SystemExit(f"results/{name}: kind {n} has no files")
        defs[n] = {"description": desc, **infer(files[n])}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"results/{name}/schema.json",
        "title": spec["title"],
        "description": (
            "x-files maps each file's path inside this folder to its definition (first match "
            "applies); tests/test_results_schema.py checks every file against it. Written by "
            "scripts/results_schema.py from the existing files: required keys are those every "
            "file has; shares and probabilities lie in 0 to 1; counts, sizes and times are not "
            "negative. Each definition's description names the script that writes the file."
        ),
        "x-files": {p: f"#/$defs/{n}" for p, n, _ in kinds},
        "$defs": defs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="*", default=sorted(KINDS))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    changed = []
    for name in args.folders:
        text = json.dumps(build(name), indent=1, ensure_ascii=False) + "\n"
        dest = REPO_ROOT / "results" / name / "schema.json"
        old = dest.read_text(encoding="utf-8") if dest.exists() else None
        if old != text:
            changed.append(name)
            if not args.check:
                dest.write_text(text, encoding="utf-8", newline="\n")
        print(f"results/{name}/schema.json: {'changed' if old != text else 'unchanged'}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
