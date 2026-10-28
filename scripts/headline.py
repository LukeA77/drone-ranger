"""Collect the headline numbers for the documents into results/headline.json.

Usage: python scripts/headline.py

Every value is read from a results file produced by the code in this repository; the source file
of each block is recorded next to it. Documents cite these values through
{{RESULT:headline.json#<key>|<format>}} tokens, filled by scripts/fill_docs.py.
"""

from __future__ import annotations

import csv
import json
import sys

from aerialrange.utils.config import REPO_ROOT

R = REPO_ROOT / "results"


def load(rel: str) -> dict:
    return json.loads((R / rel).read_text(encoding="utf-8"))


def main() -> int:
    h: dict = {}
    # --- Mid-Air test (Phase 2, run once)
    ts = load("midair/test/terrain/summary.json")["data"]
    tc = load("midair/test/terrain/calibration.json")["data"]
    tf = load("midair/test/terrain/flag.json")["data"]
    prop = load("midair/test/propagation.json")["data"]
    slope = load("midair/test/slope.json")["data"]
    cal = load("midair/test/calibration_applied.json")
    tbin = load("midair/test/terrain/binned.json")["data"]["slope_deg"]["terrain_motion"]
    rep = [b for b in tbin if b.get("reported")]
    h["midair_test"] = {
        "_source": "results/midair/test/ (terrain/*, propagation, slope, calibration_applied)",
        "terrain_median": ts["terrain_motion"]["abs_rel_err_median"],
        "terrain_p90": ts["terrain_motion"]["abs_rel_err_p90"],
        "terrain_valid": ts["terrain_motion"]["valid_fraction"],
        "flat_true_median": ts["flat_true_height"]["abs_rel_err_median"],
        "flat_true_valid": ts["flat_true_height"]["valid_fraction"],
        "flat_alt_median": ts["flat_altimeter"]["abs_rel_err_median"],
        "flat_alt_valid": ts["flat_altimeter"]["valid_fraction"],
        "terrain_median_ci": ts["terrain_motion"]["abs_rel_err_median_ci"],
        "cov_raw_90": tc["raw"]["coverage"]["0.9"],
        "cov_cal_68": tc["conformal"]["0.68"],
        "cov_cal_90": tc["conformal"]["0.9"],
        "cov_cal_95": tc["conformal"]["0.95"],
        "cov_cal_99": tc["conformal"]["0.99"],
        "flag_auroc": tf["auroc"],
        "flag_auroc_dev": load("midair/development/terrain/flag.json")["data"]["auroc"],
        "n_test_flights": len(load("midair/test/terrain/flag.json")["meta"]["trajectories"]),
        "flag_precision": tf["precision"],
        "flag_recall": tf["recall"],
        "flat_z_std_before": cal["test_split"]["before"]["z_std"],
        "flat_z_std_after": cal["test_split"]["after"]["z_std"],
        "flat_flag_auroc": cal["flag_auroc_test"],
        "mc_over_fo_below_2deg": prop[0]["mc_over_fo_median"],
        "mc_over_fo_above_10deg_max": max(b["mc_over_fo_median"] for b in prop if b["lo"] >= 10),
        "flat_err_slope_0_5": slope["flat oracle"]["bins"][0]["stat"],
        "flat_err_slope_45_90": slope["flat oracle"]["bins"][-1]["stat"],
        "terrain_err_slope_first": rep[0]["stat"],
        "terrain_err_slope_last": rep[-1]["stat"],
        "terrain_err_slope_30_45": next(b["stat"] for b in rep if b["lo"] == 30.0),
    }
    # paired comparison on a common set of samples (D-620, scripts/midair_paired.py)
    pr = load("midair/test/terrain/paired.json")["terrain_vs_flat_true_height"]
    pt, pf = pr["methods"]["terrain_motion"], pr["methods"]["flat_true_height"]
    worst = max(
        pt["per_flight_median"],
        key=lambda k: pt["per_flight_median"][k] - pf["per_flight_median"][k],
    )
    h["midair_test"].update(
        {
            "paired_share": pr["share_of_land_samples"],
            "paired_terrain_median": pt["median"],
            "paired_terrain_median_ci": pt["median_ci"],
            "paired_terrain_p90": pt["p90"],
            "paired_flat_median": pf["median"],
            "paired_flat_median_ci": pf["median_ci"],
            "paired_flat_p90": pf["p90"],
            "paired_terrain_closer": pr["share_samples_terrain_closer"],
            "paired_flights_better": pr["flights_where_terrain_median_lower"],
            "paired_n_flights": pr["n_flights"],
            "paired_worst_flight": int(worst),
            "paired_worst_flight_terrain": pt["per_flight_median"][worst],
            "paired_worst_flight_flat": pf["per_flight_median"][worst],
        }
    )
    # --- robustness of the ranging claim (D-621 pose noise, D-622 contact point)
    ps = load("midair/development/terrain/pose_sweep.json")["conditions"]
    cpb = load("ranging/contact_point_budget.json")["classes"]
    h["robustness"] = {
        "_source": "results/midair/development/terrain/pose_sweep.json, "
        "results/ranging/contact_point_budget.json",
        "n_dev_flights": len(
            load("midair/development/terrain/pose_sweep.json")["meta"]["trajectories"]
        ),
        **{
            f"{c.replace('.', 'p')}_{m}": ps[c][m]["median"]
            for c in (
                "exact",
                "pos_jitter_0.02m",
                "pos_jitter_0.1m",
                "att_jitter_0.1deg",
                "att_jitter_0.5deg",
                "pos_bias_5m",
                "att_bias_2deg",
                "nav_good",
                "nav_typical",
                "nav_poor",
            )
            for m in ("terrain", "flat")
        },
        "car_contact_median_abs": cpb["car"]["bottom_centre_with_box_noise"]["all"]["median_abs"],
        "car_contact_median_signed": cpb["car"]["bottom_centre_with_box_noise"]["all"][
            "median_signed"
        ],
        "person_contact_median_abs": cpb["person"]["bottom_centre_with_box_noise"]["all"][
            "median_abs"
        ],
        "car_end_to_end_median": cpb["car"]["end_to_end_terrain_exact_poses"]["median_abs"],
        "car_end_to_end_p90": cpb["car"]["end_to_end_terrain_exact_poses"]["p90_abs"],
        "person_end_to_end_median": cpb["person"]["end_to_end_terrain_exact_poses"]["median_abs"],
    }
    # --- VisDrone test-dev (Phase 3, run once)
    tt = load("visdrone/test-dev/B2_yolo26n_960_tuned_botsort_cmc.json")["metrics"]
    td = load("visdrone/test-dev/detect_B2_yolo26n_960.json")
    tdr = td.get("results", td)
    bd = load("visdrone/test-dev/breakdowns_B2_yolo26n_960_tuned_botsort_cmc.json")["groups"]
    h["visdrone_test"] = {
        "_source": "results/visdrone/test-dev/",
        **{
            f"{g}_{k}": tt[g]["OVERALL"][m]
            for g in ("person", "vehicle")
            for k, m in (("idf1", "idf1"), ("mota", "mota"), ("idsw", "num_switches"))
        },
        "person_ap50": tdr["person"]["ap50"]["ap"],
        "vehicle_ap50": tdr["vehicle"]["ap50"]["ap"],
        "person_recall_16_32": bd["person"]["by_box_height"]["16-32"]["recall"],
        "person_recall_128": bd["person"]["by_box_height"]["128-inf"]["recall"],
        "vehicle_recall_16_32": bd["vehicle"]["by_box_height"]["16-32"]["recall"],
    }
    # --- VisDrone val (development): the H1 baselines
    gs = {
        (r["dets"], r["tracker"], r["point"]): r
        for r in load("visdrone/val/grid_summary.json")["rows"]
    }
    b0 = load("visdrone/val/B0_yolo26n_640_bytetrack.json")["metrics"]
    b0c = load("visdrone/val/B0_yolo26n_640_botsort_cmc.json")["metrics"]
    h["visdrone_val"] = {
        "_source": "results/visdrone/val/ (B0_*.json, grid_summary.json)",
        "b0_mean_idf1": (b0["person"]["OVERALL"]["idf1"] + b0["vehicle"]["OVERALL"]["idf1"]) / 2,
        "b0_cmc_mean_idf1": (b0c["person"]["OVERALL"]["idf1"] + b0c["vehicle"]["OVERALL"]["idf1"])
        / 2,
        "b2_640_default": gs[("B2_yolo26n_640", "botsort_cmc.yaml", "default")]["mean_idf1"],
        "b2_960_tuned": gs[("B2_yolo26n_960", "botsort_cmc.yaml", "selected")]["mean_idf1"],
        "b3_tuned": gs[("B3_yolo26n_640_ios0.5", "botsort_cmc.yaml", "selected")]["mean_idf1"],
        "bytetrack_960_tuned": gs[("B2_yolo26n_960", "bytetrack.yaml", "selected")]["mean_idf1"],
    }
    # --- AU-AIR (Phase 4)
    cs = load("auair/camera_scale.json")
    hv = load("auair/repeatability_view_held_out_B2_yolo26n_960.json")["results"]
    mv = load("auair/moving_B2_yolo26n_960.json")["summary"]
    ad = load("auair/detect_B2_yolo26n_960.json")["results"]["held_out"]
    dep = hv["by_depression_all_bottom_relative"]
    tilt = load("auair/tilt_summary.json")["clips"].values()
    with (REPO_ROOT / "results" / "auair" / "clips.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    alts = [float(r[k]) for r in rows for k in ("alt_min_m", "alt_max_m")]
    h["auair"] = {
        "_source": "results/auair/ (camera_scale, repeatability_view_held_out, moving, detect)",
        "focal_px": cs["focal_px"]["median"],
        "tilt_frame_share": sum(c["frames_with_tilt"] for c in tilt)
        / sum(c["frames"] for c in tilt),
        "alt_min_m": min(alts),
        "alt_max_m": max(alts),
        "focal_ci": cs["focal_px"]["ci95_clip_bootstrap"],
        "fov_deg": cs["focal_px"]["horizontal_fov_deg"],
        "heldout_segments": hv["n_segments"],
        "heldout_rel_all": hv["bottom"]["all_flight_relative"]["median_m"],
        "heldout_rel_all_ci": hv["bottom"]["all_flight_relative"]["ci95"],
        "heldout_rel_hover": hv["bottom"]["hover_relative"]["median_m"],
        "heldout_rel_dep_0_10": dep["0-10"]["median_m"],
        "heldout_rel_dep_20_40": dep["20-40"]["median_m"],
        "vehicle_speed_median_kmh": mv["vehicle"]["median_kmh"],
        "vehicle_share_above_70": mv["vehicle"]["share_above_70_kmh"],
        "person_speed_median_kmh": mv["person"]["median_kmh"],
        "heldout_vehicle_ap50": ad["vehicle"]["ap50"]["ap"],
        "heldout_person_ap50": ad["person"]["ap50"]["ap"],
    }
    # --- Chosen detector (model card)
    with (R / "visdrone/train/B2_yolo26n_960/results.csv").open(newline="") as fh:
        tr = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    best_ep = max(tr, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    rj = load("visdrone/train/B2_yolo26n_960/run.json")
    vd = load("visdrone/val/detect_B2_yolo26n_960.json")
    vdr = vd.get("results", vd)
    h["model"] = {
        "_source": "results/visdrone/train/B2_yolo26n_960/, results/visdrone/val/detect_*.json",
        "epochs": len(tr),
        "best_epoch": int(float(best_ep["epoch"])),
        "det_val_map50": float(best_ep["metrics/mAP50(B)"]),
        "det_val_map50_95": float(best_ep["metrics/mAP50-95(B)"]),
        "train_hours": rj["wall_seconds"] / 3600,
        "gpu": rj["gpu"],
        "imgsz": rj["imgsz"],
        "mot_val_person_ap50": vdr["person"]["ap50"]["ap"],
        "mot_val_vehicle_ap50": vdr["vehicle"]["ap50"]["ap"],
    }
    # --- Data facts (data cards)
    prep = load("visdrone/prep.json")["det"]
    sp = load("splits.json")["midair"]["splits"]
    tot = load("auair/totals.json")
    h["data"] = {
        "_source": "results/visdrone/prep.json, results/splits.json, results/auair/totals.json",
        "visdrone_excluded_images": sum(
            prep[s]["excluded_images_shared_video_with_mot_eval"] for s in prep
        ),
        "visdrone_det_train_images": prep["train"]["images"],
        "midair_dev": len(sp["development"]),
        "midair_cal": len(sp["calibration"]),
        "midair_test": len(sp["test"]),
        "midair_total": sum(len(v) for v in sp.values()),
        "auair_frames": tot["frames"],
        "auair_recordings": tot["recordings"],
        "auair_clips": tot["clips"],
    }
    # --- Edge (Phase 5)
    pa = {c["config"]: c for c in load("edge/pareto.json")["configs"]}
    best, fast = pa["p5_onnx960_bot8"], pa["p5_mt_onnx640_bot8_s2"]
    rk = load("edge/ranging_kalman_cost.json")["by_tracks"]
    h["edge"] = {
        "_source": "results/edge/ (pareto.json, benchmark_*.json, ranging_kalman_cost.json)",
        "accurate_fps": best["fps"],
        "accurate_idf1": best["mean_idf1"],
        "accurate_median_ms": best["median_ms"],
        "fast_fps": fast["fps"],
        "fast_idf1": fast["mean_idf1"],
        "fast_latency_median_ms": fast["median_ms"],
        "fast_latency_p99_ms": fast["p99_ms"],
        "pytorch_fps": pa["p5_pt960_bot2"]["fps"],
        "int8dyn_detect_ms": pa["p5_int8dyn960_bot2"]["stages_median_ms"]["detect"],
        "fp32_detect_ms": pa["p5_onnx960_bot2"]["stages_median_ms"]["detect"],
        "rk_per_object_60_ms": rk["60"]["per_object"]["median_ms"],
        "rk_batched_60_ms": rk["60"]["batched"]["median_ms"],
        "cpu": load("env.json")["cpu_model"],
    }
    (R / "headline.json").write_text(json.dumps(h, indent=1) + "\n")
    print(json.dumps(h, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
