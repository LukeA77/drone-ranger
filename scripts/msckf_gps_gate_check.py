"""Apply D-744's rule for the GPS outlier gate to the development runs.

Usage: python scripts/msckf_gps_gate_check.py

Runs (results/midair/development/navigation/<name>.json, from scripts/msckf_midair.py):
A msckf_sig4 (clean, no gate), B gate_clean, C nogate_burst, D gate_burst, E gate_step.
Rule, written before the runs (D-744): (1) in B all four pooled NEES lie within their 95 %
bands; (2) in D likewise, and every injected fix is rejected; (3) in E, on every flight, GPS
position is used again within 7 s plus one fix of the step (a reset) and afterwards no
rejection lasts longer than 7 s. Writes gps_gate_check.json next to the runs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from aerialrange.utils.config import REPO_ROOT

DIR = REPO_ROOT / "results" / "midair" / "development" / "navigation"
RUNS = {"A": "msckf_sig4", "B": "gate_clean", "C": "nogate_burst", "D": "gate_burst"}
RUNS["E"] = "gate_step"
STATES = ("attitude", "position", "velocity", "all9")
RESET_FIXES = 7  # gps_reset_s at 1 Hz


def in_band(run: dict) -> dict:
    n = run["pooled"]["nees_mean"]
    return {k: bool(n[f"{k}_band_95"][0] <= n[k] <= n[f"{k}_band_95"][1]) for k in STATES}


def summary(run: dict) -> dict:
    p = run["pooled"]
    return {
        "nees": {k: p["nees_mean"][k] for k in STATES},
        "nees_band_95": p["nees_mean"]["all9_band_95"],
        "rpe_0.04s_trans_median_m": p["rpe"]["1"]["trans_median_m"],
        "rpe_0.4s_trans_median_m": p["rpe"]["10"]["trans_median_m"],
        "tilt_median_deg": p["tilt_err_median_deg_median"],
        "yaw_median_deg": p["yaw_err_median_abs_deg_median"],
        "horizontal_rmse_m": p["ate_horizontal_rmse_m_median"],
    }


def longest_run(fixes: list[int]) -> int:
    best = cur = 0
    prev = None
    for f in sorted(fixes):
        cur = cur + 1 if prev is not None and f == prev + 1 else 1
        best, prev = max(best, cur), f
    return best


def main() -> int:
    runs = {k: json.loads((DIR / f"{v}.json").read_text()) for k, v in RUNS.items()}
    clean_rejected = {n: f["info"]["gps_events"] for n, f in runs["B"]["flights"].items()}
    c1 = all(in_band(runs["B"]).values())
    missed = {}
    for n, f in runs["D"]["flights"].items():
        rej = set(f["info"]["gps_events"]["pos_rejected"])
        miss = [i for i in f["injected_fixes"] if i not in rej]
        if miss:
            missed[n] = miss
    c2 = all(in_band(runs["D"]).values()) and not missed
    step = {}
    for n, f in runs["E"]["flights"].items():
        s = f["injected_fixes"][0]
        ev = f["info"]["gps_events"]
        resets = [r for r in ev["pos_reset"] if r >= s]
        first = resets[0] if resets else None
        after = [r for r in ev["pos_rejected"] if first is not None and r > first]
        step[n] = {
            "step_fix": s,
            "reset_fix": first,
            "fixes_to_reset": None if first is None else first - s,
            "later_resets": resets[1:],
            "longest_rejection_after_reset": longest_run(after),
        }
    c3 = all(
        v["reset_fix"] is not None
        and v["fixes_to_reset"] <= RESET_FIXES + 1
        and not v["later_resets"]
        and v["longest_rejection_after_reset"] <= RESET_FIXES
        for v in step.values()
    )
    res = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-744",
        "runs": RUNS,
        "summary": {k: summary(v) for k, v in runs.items()},
        "nees_in_band": {k: in_band(v) for k, v in runs.items()},
        "clean_gps_events_with_gate": clean_rejected,
        "burst_fixes_not_rejected": missed,
        "step": step,
        "conditions": {"1_clean_consistent": c1, "2_bursts_rejected": c2, "3_step_reset": c3},
        "adopted": bool(c1 and c2 and c3),
    }
    (DIR / "gps_gate_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("summary", "nees_in_band", "conditions")}, indent=1))
    print("ADOPTED" if res["adopted"] else "NOT ADOPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
