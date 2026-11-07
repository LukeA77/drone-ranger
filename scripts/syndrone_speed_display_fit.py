"""Fit when speed is displayed on SynDrone development towns.

Reads the matched speed rows written by scripts/syndrone_evaluate.py
(`cache/syndrone/runs/<name>/speed_matches_development.csv.gz`). For each candidate pair
(speed sigma maximum, minimum track age), a speed is "displayed" when the rule in
`pipeline/speed_display.py` passes (sigma, age, and the windowed NIS test). The chosen pair
is the one that displays speed on the largest share of matched track-frames while at
least `--target` of displayed speeds are within `--tol` m/s of truth (pooled over
development sequences); ties go to the stricter pair. Writes
`results/syndrone/development/speed_display_fit_<name>.json`, including the share of tracks
that ever show a speed and a bootstrap interval for the accuracy, resampling sequences.

Defaults (D-706): tolerance 0.5 m/s, because walkers move at 1 to 2 m/s
(`spawn_utils.py`), so a displayed speed within 0.5 m/s keeps a walking person
distinguishable from a standing one; target 90 %.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from aerialrange.pipeline.speed_display import SpeedDisplayRule
from aerialrange.utils.config import REPO_ROOT, load_config

RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
SIGMAS = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
AGES = (2, 5, 10, 15, 25, 50)


def displayed(df: pd.DataFrame, rule: SpeedDisplayRule) -> np.ndarray:
    nis_n = np.minimum(df["age"].to_numpy(int), rule.nis_window)
    return np.array(
        [
            rule.ok(s, int(a), m, int(n))
            for s, a, m, n in zip(df["sigma_ms"], df["age"], df["nis_mean"], nis_n, strict=True)
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1_accurate")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--target", type=float, default=0.90)
    args = ap.parse_args(argv)
    sd = load_config("run").require("speed_display")
    df = pd.read_csv(RUNS / args.name / "speed_matches_development.csv.gz")
    df = df[np.isfinite(df["sigma_ms"]) & np.isfinite(df["pred_ms"])]
    df["ok"] = np.abs(df["pred_ms"] - df["truth_ms"]) <= args.tol
    df["seq"] = df["town"] + "_h" + df["height"].astype(str)
    grid = []
    for s in SIGMAS:
        for a in AGES:
            rule = SpeedDisplayRule(s, a, int(sd["nis_window"]), float(sd["nis_conf"]))
            show = displayed(df, rule)
            n = int(show.sum())
            grid.append(
                {
                    "sigma_max_ms": s,
                    "min_age": a,
                    "displayed_share": float(show.mean()),
                    "n_displayed": n,
                    "accuracy": float(df.loc[show, "ok"].mean()) if n else float("nan"),
                    "_show": show,
                }
            )
    feasible = [g for g in grid if g["n_displayed"] >= 100 and g["accuracy"] >= args.target]
    if not feasible:
        print("no setting meets the target; nothing chosen")
        best = None
    else:
        best = max(feasible, key=lambda g: (g["displayed_share"], -g["sigma_max_ms"], g["min_age"]))
    res: dict = {
        "name": args.name,
        "tolerance_ms": args.tol,
        "target": args.target,
        "nis_window": int(sd["nis_window"]),
        "nis_conf": float(sd["nis_conf"]),
        "n_matched_track_frames": int(len(df)),
        "grid": [{k: v for k, v in g.items() if k != "_show"} for g in grid],
        "chosen": None,
    }
    # the most accurate setting that shows enough speeds to judge (same >= 100 rule as
    # the choice), so a report can say how far short of the target the filter falls
    judged = [g for g in grid if g["n_displayed"] >= 100]
    if judged:
        top = max(judged, key=lambda g: g["accuracy"])
        res["most_accurate"] = {k: v for k, v in top.items() if k != "_show"}
    if best is not None:
        show = best["_show"]
        sub = df.assign(show=show)
        seqs = sorted(sub["seq"].unique())
        rng = np.random.default_rng(0)
        accs = []
        for _ in range(1000):
            pick = rng.choice(seqs, len(seqs))
            d = pd.concat([sub[sub["seq"] == q] for q in pick])
            if d["show"].any():
                accs.append(d.loc[d["show"], "ok"].mean())
        tracks = sub.groupby(["seq", "group", "track_id"])["show"].any()
        res["chosen"] = {
            "sigma_max_ms": best["sigma_max_ms"],
            "min_age": best["min_age"],
            "displayed_share_of_track_frames": best["displayed_share"],
            "accuracy": best["accuracy"],
            "accuracy_ci95_resample_sequences": [
                float(np.percentile(accs, 2.5)),
                float(np.percentile(accs, 97.5)),
            ],
            "share_of_tracks_ever_showing_speed": float(tracks.mean()),
            "share_by_group": {
                g: float(tracks.xs(g, level="group").mean())
                for g in sub["group"].unique()
                if g in tracks.index.get_level_values("group")
            },
            "accuracy_by_group": {
                g: float(d.loc[d["show"], "ok"].mean()) if d["show"].any() else None
                for g, d in sub.groupby("group")
            },
        }
    out = REPO_ROOT / "results" / "syndrone" / "development"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"speed_display_fit_{args.name}.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(res["chosen"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
