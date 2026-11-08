"""Footprint sizes of SynDrone actors by group, from the true 3D boxes (D-736).

Usage: python scripts/syndrone_footprint_sizes.py [--every 300]

For the development towns, every `--every`-th frame's 3D boxes (`SynDroneSequence.boxes`) are
read, and each actor's footprint (the bottom face) gives its length (longer side) and width;
an actor counts once (its median over the frames seen). Groups are those of the scoring
(people 40; vehicles car, truck, bus, 100 to 102). Per group: mean, standard deviation and
percentiles of length and width, and of the support half-width averaged over headings,
(L + W) / pi, which places the footprint centre when the heading is unknown
(`pipeline/footprint.py`).
Output: results/syndrone/development/footprint_sizes.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import numpy as np

from aerialrange.data.syndrone import BOTTOM, SynDroneSequence
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.evaluate.syndrone_eval import GROUPS
from aerialrange.utils.config import REPO_ROOT, load_config

GROUP_TAGS = GROUPS  # as scored: people 40; vehicles car, truck, bus (100 to 102)


def footprint_dims(corners: np.ndarray) -> tuple[float, float]:
    """Length (longer side) and width of the bottom face of one box (8, 3)."""
    b = corners[BOTTOM][:, :2]
    c = b.mean(axis=0)
    order = np.argsort(np.arctan2(b[:, 1] - c[1], b[:, 0] - c[0]))
    b = b[order]
    sides = np.linalg.norm(np.roll(b, -1, axis=0) - b, axis=1)
    a, w = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    return float(max(a, w)), float(min(a, w))


def summary(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "p10_p50_p90": np.percentile(x, [10, 50, 90]).tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=300)
    args = ap.parse_args(argv)
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text("utf-8"))["syndrone"][
        "splits"
    ]
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    dims: dict[str, dict[tuple[str, int], list]] = {g: {} for g in GROUP_TAGS}
    for town in splits["development"]:
        seq = SynDroneSequence(store, town, 50)
        for k in range(0, 3000, args.every):
            bx = seq.boxes(k)
            for i in np.flatnonzero(~bx.level):
                for g, tags in GROUP_TAGS.items():
                    if bx.tags[i] in tags:
                        dims[g].setdefault((town, int(bx.ids[i])), []).append(
                            footprint_dims(bx.corners[i])
                        )
        print(f"{town}: done", flush=True)
    out: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-736",
        "towns": splits["development"],
        "every_nth_frame": args.every,
        "groups": {},
    }
    for g, per in dims.items():
        lw = np.array([np.median(np.array(v), axis=0) for v in per.values()])
        if not len(lw):
            continue
        L, W = lw[:, 0], lw[:, 1]
        out["groups"][g] = {
            "actors": int(len(lw)),
            "length_m": summary(L),
            "width_m": summary(W),
            "mean_support_half_width_m": summary((L + W) / np.pi),
            # what the pipeline's footprint correction needs (configs/run.yaml `footprint`)
            "moments": {
                "length_mean_m": float(np.mean(L)),
                "width_mean_m": float(np.mean(W)),
                "length_var_m2": float(np.var(L, ddof=1)),
                "width_var_m2": float(np.var(W, ddof=1)),
                "length_width_cov_m2": float(np.cov(L, W)[0, 1]),
            },
        }
    dest = REPO_ROOT / "results" / "syndrone" / "development" / "footprint_sizes.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["groups"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
