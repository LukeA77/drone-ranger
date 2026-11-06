"""Evaluate a pipeline run on SynDrone sequences against truth.

Inputs: truth (`cache/syndrone/truth/<town>/objects.csv.gz`, scripts/syndrone_z0_check.py),
speed truth (`cache/syndrone/speed/<town>.csv.gz`) and pipeline outputs
(`cache/syndrone/runs/<name>/<town>_h<h>/`, scripts/syndrone_run.py).

Reports, per group (person, vehicle), pooled over sequences with 95 % bootstrap intervals
that resample whole sequences, and separately whole towns:
- detection AP50, overall and by truth box height (0-16-32-64-128-inf px);
- tracking IDF1, MOTA (py-motmetrics) and HOTA (TrackEval's algorithm, `track/hota.py`);
- range: relative error of matched tracks against the footprint centre and the near edge,
  and coverage of the pipeline's range intervals at 68 / 90 / 95 %;
- speed: error of matched tracks against truth, coverage of the speed intervals, and the
  same for speeds the display rule shows.

Truth 2D boxes are the instance-mask boxes (visible extent, D-704). Results go to
`results/syndrone/<split>/eval_<name>.json`; refuses to evaluate test towns unless
--allow-test is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

import motmetrics as mm
import numpy as np
import pandas as pd

from aerialrange.data.syndrone import FPS
from aerialrange.detect.evaluate import ap_from_pieces, ap_pieces
from aerialrange.evaluate.syndrone_eval import (
    GROUPS,
    SIZE_BINS_PX,
    TruthRules,
    classify_truth,
    interval_coverage,
    iou_matrix,
    keep_not_ignored,
    match_pairs,
)
from aerialrange.track.hota import hota_combine, hota_sequence, summary
from aerialrange.utils.config import REPO_ROOT, load_config

TRUTH = REPO_ROOT / "cache" / "syndrone" / "truth"
SPEED = REPO_ROOT / "cache" / "syndrone" / "speed"
RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
N_BOOT = 1000
LEVELS = (0.68, 0.90, 0.95)


def splits() -> dict:
    return json.loads((REPO_ROOT / "results" / "splits.json").read_text("utf-8"))["syndrone"][
        "splits"
    ]


def mask_ltwh(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["mask_left"],
            df["mask_top"],
            df["mask_right"] - df["mask_left"],
            df["mask_bottom"] - df["mask_top"],
        ]
    )


def evaluate_sequence(town: str, h: int, name: str, rules: TruthRules, cmap: dict) -> dict:
    tr = pd.read_csv(TRUTH / town / "objects.csv.gz")
    tr = classify_truth(tr[tr["height"] == h], rules)
    tr = tr[tr["status"] != "absent"]
    run = RUNS / name / f"{town}_h{h}"
    tracks = pd.read_csv(run / "tracks.csv")
    tracks["frame0"] = tracks["frame"] - 1
    dets = pd.read_csv(run / "detections.csv")
    dets["frame0"] = dets["frame"] - 1
    dets["group"] = dets["cls"].map(lambda c: cmap.get(int(c)))
    sp = pd.read_csv(SPEED / f"{town}.csv.gz").set_index(["id", "frame"])
    frames = range(int(max(tracks["frame0"].max(), dets["frame0"].max())) + 1)
    out: dict = {"town": town, "height": h, "n_frames": len(frames)}
    for g in GROUPS:
        tg = tr[tr["group"] == g]
        valid_by_f = {f: d for f, d in tg[tg["status"] == "valid"].groupby("frame")}
        # Ignore boxes: any non-valid visible box (other classes, level, occluded, small).
        # Ignore regions for group g: its own ignored objects, plus boxes in no group (riders,
        # two-wheelers, trams, map boxes). The other group's objects are not ignored, so a
        # detection of the wrong class is a false positive, as in version 1.
        ign_sel = ((tr["status"] == "ignore") & (tr["group"] == g)) | tr["group"].isna()
        ign_by_f = {f: d for f, d in tr[ign_sel].groupby("frame")}
        dg = dets[dets["group"] == g]
        dg_by_f = {f: d for f, d in dg.groupby("frame0")}
        trk = tracks[tracks["class"] == g]
        trk_by_f = {f: d for f, d in trk.groupby("frame0")}
        gt_img, det_img = {}, {}
        acc = mm.MOTAccumulator(auto_id=False)
        g_ids, t_ids, sims = [], [], []
        gid_map: dict[int, int] = {}
        tid_map: dict[int, int] = {}
        rng_rows, spd_rows = [], []
        for f in frames:
            v = valid_by_f.get(f)
            vb = mask_ltwh(v) if v is not None else np.zeros((0, 4))
            ib = mask_ltwh(ign_by_f[f]) if f in ign_by_f else np.zeros((0, 4))
            d = dg_by_f.get(f)
            db = (
                d[["left", "top", "width", "height"]].to_numpy()
                if d is not None
                else np.zeros((0, 4))
            )
            ds = d["score"].to_numpy() if d is not None else np.zeros(0)
            k = keep_not_ignored(db, vb, ib)
            gt_img[f], det_img[f] = vb, (db[k], ds[k])
            t = trk_by_f.get(f)
            if t is not None:
                tb = t[["left", "top", "width", "height"]].to_numpy()
                t = t[keep_not_ignored(tb, vb, ib)]
                tb = t[["left", "top", "width", "height"]].to_numpy()
            else:
                tb = np.zeros((0, 4))
            gids = v["id"].to_numpy(int) if v is not None else np.zeros(0, int)
            tids = t["track_id"].to_numpy(int) if t is not None else np.zeros(0, int)
            iou = iou_matrix(vb, tb)
            dist = np.where(iou >= 0.5, 1.0 - iou, np.nan)
            acc.update(gids.tolist(), tids.tolist(), dist, frameid=f)
            g_ids.append(np.array([gid_map.setdefault(int(i), len(gid_map)) for i in gids], int))
            t_ids.append(np.array([tid_map.setdefault(int(i), len(tid_map)) for i in tids], int))
            sims.append(iou)
            for a, b, _ in match_pairs(vb, tb):
                trow, prow = v.iloc[a], t.iloc[b]
                if bool(prow["valid"]):
                    rng_rows.append(
                        (
                            trow["range_m"],
                            trow["range_near_m"],
                            prow["range_m"],
                            prow["range_sigma_m"],
                            trow["depression_deg"],
                            trow["mask_bottom"] - trow["mask_top"],
                        )
                    )
                key = (int(trow["id"]), int(f))
                if key in sp.index and np.isfinite(prow["speed_ms"]):
                    st = sp.loc[key]
                    if np.isfinite(st["speed_ms"]):
                        spd_rows.append(
                            (
                                st["speed_ms"],
                                prow["speed_ms"],
                                prow["speed_sigma_ms"],
                                bool(prow.get("speed_shown", False)),
                                prow["track_age"],
                                prow["nis_mean"],
                                prow["track_id"],
                                trow["depression_deg"],
                            )
                        )
        pieces = {"all": ap_pieces(gt_img, det_img, 0.5)}
        for lo, hi in zip(SIZE_BINS_PX[:-1], SIZE_BINS_PX[1:], strict=True):
            pieces[f"{lo:g}-{hi:g}"] = ap_pieces(gt_img, det_img, 0.5, (lo, hi))
        mh = mm.metrics.create()
        m = mh.compute(
            acc,
            metrics=[
                "num_objects",
                "num_misses",
                "num_false_positives",
                "num_switches",
                "idtp",
                "idfp",
                "idfn",
            ],
            name="x",
        ).iloc[0]
        out[g] = {
            "ap_pieces": pieces,
            "mot": {k: float(m[k]) for k in m.index},
            "hota": hota_sequence(g_ids, t_ids, sims),
            "range": np.array(rng_rows, float).reshape(-1, 6),
            "speed": np.array(spd_rows, float).reshape(-1, 8),
            "n_valid_truth": int((tg["status"] == "valid").sum()),
        }
    return out


def pooled(seqs: list[dict], g: str) -> dict:
    """All metrics for group g pooled over the given sequence results."""
    res: dict = {}
    for b in seqs[0][g]["ap_pieces"]:
        res[f"ap50_{b}"] = ap_from_pieces([s[g]["ap_pieces"][b] for s in seqs])["ap"]
    mot = {k: sum(s[g]["mot"][k] for s in seqs) for k in seqs[0][g]["mot"]}
    n = max(mot["num_objects"], 1)
    res["mota"] = 1 - (mot["num_misses"] + mot["num_false_positives"] + mot["num_switches"]) / n
    res["idf1"] = 2 * mot["idtp"] / max(2 * mot["idtp"] + mot["idfp"] + mot["idfn"], 1)
    res["id_switches"] = mot["num_switches"]
    res.update(
        {
            f"hota_{k}": v
            for k, v in summary(hota_combine({i: s[g]["hota"] for i, s in enumerate(seqs)})).items()
        }
    )
    r = np.vstack([s[g]["range"] for s in seqs])
    if len(r):
        rel = (r[:, 2] - r[:, 0]) / r[:, 0]
        rel_near = (r[:, 2] - r[:, 1]) / r[:, 1]
        res["range_n"] = len(r)
        res["range_rel_err_median_abs"] = float(np.median(np.abs(rel)))
        res["range_rel_err_p90_abs"] = float(np.percentile(np.abs(rel), 90))
        res["range_rel_err_median_signed"] = float(np.median(rel))
        res["range_rel_err_near_edge_median_signed"] = float(np.median(rel_near))
        res.update(
            {
                f"range_coverage_{k}": v
                for k, v in interval_coverage(r[:, 2] - r[:, 0], r[:, 3], LEVELS).items()
            }
        )
    sp = np.vstack([s[g]["speed"] for s in seqs])
    if len(sp):
        err = sp[:, 1] - sp[:, 0]
        res["speed_n"] = len(sp)
        res["speed_err_median_abs_ms"] = float(np.median(np.abs(err)))
        res["speed_err_p90_abs_ms"] = float(np.percentile(np.abs(err), 90))
        res.update(
            {f"speed_coverage_{k}": v for k, v in interval_coverage(err, sp[:, 2], LEVELS).items()}
        )
        shown = sp[:, 3] > 0
        res["speed_shown_share"] = float(shown.mean())
        if shown.any():
            res["speed_shown_err_median_abs_ms"] = float(np.median(np.abs(err[shown])))
            res["speed_shown_err_p90_abs_ms"] = float(np.percentile(np.abs(err[shown]), 90))
            res.update(
                {
                    f"speed_shown_coverage_{k}": v
                    for k, v in interval_coverage(err[shown], sp[shown, 2], LEVELS).items()
                }
            )
    return res


HEADLINE = (
    "ap50_all",
    "idf1",
    "mota",
    "hota_HOTA",
    "range_rel_err_median_abs",
    "range_rel_err_p90_abs",
    "range_coverage_0.90",
    "speed_err_median_abs_ms",
    "speed_coverage_0.90",
)


def boot_all(
    units: list, g: str, flatten: Callable[[list], list]
) -> dict[str, tuple[float, float]]:
    """95 % percentile intervals of every headline metric, resampling whole units."""
    rng = np.random.default_rng(0)
    vals: dict[str, list[float]] = {k: [] for k in HEADLINE}
    for _ in range(N_BOOT):
        pick = [units[i] for i in rng.integers(0, len(units), len(units))]
        res = pooled(flatten(pick), g)
        for k in HEADLINE:
            v = res.get(k)
            if v is not None and np.isfinite(v):
                vals[k].append(v)
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in vals.items()
        if v
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1_accurate")
    ap.add_argument("--split", default="development")
    ap.add_argument("--towns", nargs="*")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        print("test towns are run once, after freezing; pass --allow-test")
        return 1
    towns = args.towns or splits()[args.split]
    assert all(t in splits()[args.split] for t in towns), "town not in this split"
    cfg = load_config("syndrone")
    rules = TruthRules(
        float(cfg.require("truth_min_visibility")), int(cfg.require("truth_min_instance_px"))
    )
    cmap = {
        int(k): v for k, v in load_config("detector").require("class_maps")["visdrone10"].items()
    }
    seqs = []
    for t in towns:
        for h in (20, 50, 80):
            if (RUNS / args.name / f"{t}_h{h}" / "tracks.csv").exists():
                seqs.append(evaluate_sequence(t, h, args.name, rules, cmap))
                print(f"evaluated {t} h{h}", flush=True)
    if not seqs:
        print("no runs found")
        return 1
    result: dict = {
        "name": args.name,
        "split": args.split,
        "towns": towns,
        "sequences": [f"{s['town']}_h{s['height']}" for s in seqs],
        "truth_rules": vars(rules),
        "n_boot": N_BOOT,
    }
    for g in GROUPS:
        res = pooled(seqs, g)
        by_town = [[s for s in seqs if s["town"] == t] for t in sorted({s["town"] for s in seqs})]
        ci_seq = boot_all(seqs, g, lambda u: u)
        ci_town = (
            boot_all(by_town, g, lambda u: [s for us in u for s in us]) if len(by_town) > 1 else {}
        )
        per_seq = {
            f"{s['town']}_h{s['height']}": {
                k: v for k, v in pooled([s], g).items() if k in HEADLINE
            }
            for s in seqs
        }
        by_height = {
            f"h{h}": pooled([s for s in seqs if s["height"] == h], g)
            for h in (20, 50, 80)
            if any(s["height"] == h for s in seqs)
        }
        result[g] = {
            "pooled": res,
            "ci95_resample_sequences": ci_seq,
            "ci95_resample_towns": ci_town,
            "by_height": by_height,
            "per_sequence": per_seq,
            "n_valid_truth": int(sum(s[g]["n_valid_truth"] for s in seqs)),
        }
    out = REPO_ROOT / "results" / "syndrone" / args.split
    out.mkdir(parents=True, exist_ok=True)
    cols = [
        "truth_ms",
        "pred_ms",
        "sigma_ms",
        "shown",
        "age",
        "nis_mean",
        "track_id",
        "depression_deg",
    ]
    frames = []
    for s in seqs:
        for g in GROUPS:
            df = pd.DataFrame(s[g]["speed"], columns=cols)
            df["town"], df["height"], df["group"] = s["town"], s["height"], g
            frames.append(df)
    RUNS.joinpath(args.name).mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(RUNS / args.name / f"speed_matches_{args.split}.csv.gz", index=False)
    (out / f"eval_{args.name}.json").write_text(
        json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8"
    )
    for g in GROUPS:
        print(g, {k: round(result[g]["pooled"].get(k, float("nan")), 3) for k in HEADLINE})
    print(f"speed sample rate: {FPS} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
