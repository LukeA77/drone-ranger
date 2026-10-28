"""Combine the validation tracker grids (D-319, D-320) into one table and apply the selection rule.

Usage: python scripts/grid_summary.py

Reads results/visdrone/val/grid_<dets>__<tracker>[__ext_*].json, merges each detector and
tracker's main and extension points, and picks the best point by the D-319 rule (mean IDF1
over groups, rounded to 4 dp, then mean MOTA). Writes results/visdrone/val/grid_summary.json
and a CSV beside it: one row per detector and tracker, at the library default and at the
selected point, plus whether the selected point lies on the edge of the searched values.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime

from aerialrange.utils.config import REPO_ROOT

VAL = REPO_ROOT / "results" / "visdrone" / "val"
KEYS = ("track_high_thresh", "new_track_thresh")


def _row(p: dict) -> dict:
    o = {g: p["metrics"][g]["OVERALL"] for g in p["metrics"]}
    row = {k: p["overrides"][k] for k in KEYS}
    row |= {"mean_idf1": p["mean_idf1"], "mean_mota": p["mean_mota"]}
    for g, m in sorted(o.items()):
        for k in ("idf1", "mota", "num_switches", "precision", "recall"):
            row[f"{g}_{k}"] = m[k]
    return row


def main() -> int:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sources: dict[tuple[str, str], list[str]] = defaultdict(list)
    for f in sorted(VAL.glob("grid_*__*.json")):
        d = json.loads(f.read_text())
        key = (d["dets"], d["tracker"])
        groups[key] += d["points"]
        sources[key].append(f.name)
    rows = []
    for (dets, trk), pts in sorted(groups.items()):
        best = max(pts, key=lambda r: (round(r["mean_idf1"], 4), r["mean_mota"]))
        default = next(
            p
            for p in pts
            if p["overrides"] == {"track_high_thresh": 0.25, "new_track_thresh": 0.25}
        )
        tried = {k: sorted({p["overrides"][k] for p in pts}) for k in KEYS}
        edge = [k for k in KEYS if best["overrides"][k] in (tried[k][0], tried[k][-1])]
        for label, p in (("default", default), ("selected", best)):
            rows.append({"dets": dets, "tracker": trk, "point": label} | _row(p))
        rows[-1]["on_edge"] = ",".join(edge)
        rows[-1]["n_points"] = len(pts)
        rows[-1]["sources"] = ";".join(sources[(dets, trk)])
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "val",
        "selection_rule": "max mean IDF1 over groups (rounded to 4 dp), then mean MOTA (D-319)",
        "note": "Development numbers on validation; not reportable test results.",
        "rows": rows,
    }
    (VAL / "grid_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (VAL / "grid_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        edge = f"edge: {r['on_edge']}" if r.get("on_edge") else ""
        print(
            f"{r['dets']:24s} {r['tracker']:16s} {r['point']:8s} "
            f"high {r['track_high_thresh']:.2f} new {r['new_track_thresh']:.2f} "
            f"mean IDF1 {r['mean_idf1']:.3f} person {r['person_idf1']:.3f} "
            f"vehicle {r['vehicle_idf1']:.3f} {edge}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
