"""Measure the simulated drones' camera from SynDrone runs (Phase 14 sensor model, D-752).

Usage: python scripts/multidrone_sensor_fit.py [--name p10_v2_cal --split test --allow-test]

The simulation needs, per camera height, the probability that a person under the camera is
detected during one pass of the camera over them. It is measured here from a frozen run's
outputs (no new run): for each findable person (the evaluation's valid truth), each pass is a
stretch of frames in which they are findable and at least the lowest reliable depression below
the horizon (the searched-ground rule, D-749), split where they are out of view for longer than
the tracker keeps a lost track (max_lost frames). A pass counts as detected when a track
matches the person (IoU at least 0.5, after ignore regions, the evaluation's rules) in any of its
frames. Passes of one person are treated as independent by the simulation; correlation between
them is a stated limit.

Also written, per height: false alarms per km2 of ground searched (from the run's search
measures, D-749) and the median calibrated 68 % range half-width of matched person rows (the
size of a detection's position error).

Writes results/multi_drone/sensor_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from aerialrange.evaluate.syndrone_eval import (
    TruthRules,
    classify_truth,
    keep_not_ignored,
    match_pairs,
)
from aerialrange.utils.config import REPO_ROOT, load_config

TRUTH = REPO_ROOT / "cache" / "syndrone" / "truth"
RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
HEIGHTS = (20, 50, 80)
N_BOOT = 1000
GROUP = "person"


def mask_ltwh(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["mask_left"],
            df["mask_top"],
            df["mask_right"] - df["mask_left"],
            df["mask_bottom"] - df["mask_top"],
        ]
    )


def passes_of(frames: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """Split sorted frame numbers into passes where the gap exceeds max_gap frames."""
    if len(frames) == 0:
        return []
    cut = np.nonzero(np.diff(frames) > max_gap + 1)[0] + 1
    return np.split(frames, cut)


def sequence_passes(
    town: str, h: int, name: str, rules: TruthRules, min_dep: float, max_gap: int
) -> tuple[list[dict], list[float]]:
    tr = pd.read_csv(TRUTH / town / "objects.csv.gz")
    tr = classify_truth(tr[tr["height"] == h], rules)
    tr = tr[tr["status"] != "absent"]
    tracks = pd.read_csv(RUNS / name / f"{town}_h{h}" / "tracks.csv")
    tracks["frame0"] = tracks["frame"] - 1
    tg = tr[tr["group"] == GROUP]
    valid = tg[tg["status"] == "valid"]
    ign_sel = ((tr["status"] == "ignore") & (tr["group"] == GROUP)) | tr["group"].isna()
    ign_by_f = {f: d for f, d in tr[ign_sel].groupby("frame")}
    trk_by_f = {f: d for f, d in tracks[tracks["class"] == GROUP].groupby("frame0")}
    matched: set[tuple[int, int]] = set()  # (truth id, frame)
    halves: list[float] = []
    for f, v in valid.groupby("frame"):
        t = trk_by_f.get(f)
        if t is None:
            continue
        vb = mask_ltwh(v)
        ib = mask_ltwh(ign_by_f[f]) if f in ign_by_f else np.zeros((0, 4))
        t = t[keep_not_ignored(t[["left", "top", "width", "height"]].to_numpy(), vb, ib)]
        for a, b, _ in match_pairs(vb, t[["left", "top", "width", "height"]].to_numpy()):
            matched.add((int(v.iloc[a]["id"]), int(f)))
            row = t.iloc[b]
            if bool(row["valid"]) and np.isfinite(row["range_half_68_m"]):
                halves.append(float(row["range_half_68_m"]))
    out = []
    usable = valid[valid["depression_deg"] >= min_dep]
    for tid, d in usable.groupby("id"):
        d = d.sort_values("frame")
        fr = d["frame"].to_numpy(int)
        for p in passes_of(fr, max_gap):
            sel = d[d["frame"].isin(p)]
            out.append(
                {
                    "seq": f"{town}_h{h}",
                    "height": h,
                    "id": int(tid),
                    "n_frames": int(len(p)),
                    "box_h_px_median": float((sel["mask_bottom"] - sel["mask_top"]).median()),
                    "depression_deg_median": float(sel["depression_deg"].median()),
                    "detected": any((int(tid), int(f)) in matched for f in p),
                }
            )
    return out, halves


def boot_share(df: pd.DataFrame, rng: np.random.Generator) -> list[float] | None:
    seqs = sorted(df["seq"].unique())
    if len(seqs) < 2:
        return None
    by = {s: (int(g["detected"].sum()), len(g)) for s, g in df.groupby("seq")}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(seqs, len(seqs))
        k = sum(by[s][0] for s in pick)
        n = sum(by[s][1] for s in pick)
        vals.append(k / n if n else np.nan)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="p10_v2_cal")
    ap.add_argument("--split", default="test")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        print("reading test-town outputs needs --allow-test (outputs of the one-time run only)")
        return 1
    search = json.loads(
        (REPO_ROOT / "results" / "syndrone" / args.split / f"search_{args.name}.json").read_text()
    )
    ev = json.loads(
        (REPO_ROOT / "results" / "syndrone" / args.split / f"eval_{args.name}.json").read_text()
    )
    min_dep = float(search["min_depression_deg"])
    run_cfg = load_config("run")
    trk_path = REPO_ROOT / run_cfg.require("tracker")["config"]
    max_gap = int(yaml.safe_load(trk_path.read_text())["max_lost"])
    cfg = load_config("syndrone")
    rules = TruthRules(
        float(cfg.require("truth_min_visibility")), int(cfg.require("truth_min_instance_px"))
    )
    rows, halves_by_h = [], {h: [] for h in HEIGHTS}
    for seq in ev["sequences"]:
        town, h = seq.rsplit("_h", 1)
        p, hv = sequence_passes(town, int(h), args.name, rules, min_dep, max_gap)
        rows += p
        halves_by_h[int(h)] += hv
        print(f"{seq}: {len(p)} passes, {sum(r['detected'] for r in p)} detected", flush=True)
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    res: dict = {
        "name": args.name,
        "split": args.split,
        "sequences": ev["sequences"],
        "group": GROUP,
        "min_depression_deg": min_dep,
        "pass_split_gap_frames": max_gap,
        "n_boot": N_BOOT,
        "by_height": {},
    }
    for h in HEIGHTS:
        d = df[df["height"] == h]
        fa = search[GROUP]["by_height"][f"h{h}"]
        q = np.quantile(d["n_frames"], [0.25, 0.5, 0.75]).tolist() if len(d) else []
        by_len = {}
        if len(d) >= 4:
            edges = np.unique(np.quantile(d["n_frames"], [0, 0.25, 0.5, 0.75, 1.0]))
            for j, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
                lo = d["n_frames"] >= a if j == 0 else d["n_frames"] > a
                s = d[lo & (d["n_frames"] <= b)]  # bins (a, b], the first [a, b]
                by_len[f"{int(a)}-{int(b)}"] = {
                    "n": len(s),
                    "pd": float(s["detected"].mean()) if len(s) else None,
                }
        res["by_height"][f"h{h}"] = {
            "n_passes": len(d),
            "n_people": int(d.groupby("seq")["id"].nunique().sum()) if len(d) else 0,
            "pd_per_pass": float(d["detected"].mean()) if len(d) else None,
            "pd_ci95_resample_sequences": boot_share(d, rng) if len(d) else None,
            "pass_frames_quartiles": q,
            "pd_by_pass_frames": by_len,
            "box_h_px_median": float(d["box_h_px_median"].median()) if len(d) else None,
            "false_alarms_per_km2": fa["false_alarms_per_km2"],
            "false_alarm_tracks": fa["n_false_alarm_tracks"],
            "ground_searched_km2": fa["ground_searched_km2"],
            "range_half_68_m_median": float(np.median(halves_by_h[h])) if halves_by_h[h] else None,
        }
    out = REPO_ROOT / "results" / "multi_drone" / f"sensor_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["by_height"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
