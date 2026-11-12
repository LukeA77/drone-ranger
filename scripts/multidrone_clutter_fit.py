"""Measure false-alarm recurrence, pass-to-pass correlation and a persistence filter for the
Phase 16 search simulation (D-764), at the height where people are found (20 m).

Usage:
  python scripts/multidrone_clutter_fit.py --name p10_v2_cal --split test --allow-test
  python scripts/multidrone_clutter_fit.py --name v2_dfine_cal_rural --split development

Read from a frozen run's outputs (no new run), with the evaluation's rules (as
scripts/multidrone_sensor_fit.py):

- A false-alarm track is a person track never matched to a valid person (IoU 0.5, after ignore
  regions), its site the median of its ground positions.
- A ground point's views are the frames whose searched ground (geomap.footprint_ring, the
  searched-ground rule of D-749) holds it, split into passes where it is out of view for longer
  than the tracker keeps a lost track (max_lost frames).
- Recurrence: on a later pass over a false-alarm site, another false-alarm track with a ground
  position within `radius_m` of the site. The radius is the search map's own cut: TAIL_SIGMAS
  times the camera's measured position error (the sensor file's 68 % half-width at 20 m).
  `q` is the share of later passes with a recurrence; `repeat_share` the share of false-alarm
  tracks that recur one on an earlier pass.
- Two kinds of false alarm, fitted by maximum likelihood over the sites' later passes: a share
  of sites are persistent look-alikes that fire with probability `fire_per_pass` on each later
  pass; the rest never fire again (`fit_persistent`), with intervals from resampling sites. The
  simulation places look-alikes once per search at the density that gives this share of the
  measured rate, so re-flown ground does not collect new ones (a model where every false alarm
  left a new site did, and its rate grew without bound on ground flown many times).
- Pass correlation: for each person, consecutive passes (as the sensor fit splits them), the
  detection rate after a detected pass and after a missed one.
- Persistence: only tracks lasting at least `m` frames (counted as the evaluation counts track
  life) are kept, then matching is redone. Values of `m`: 1 (no filter), the median life of
  false-alarm tracks, the median life of person tracks. For each `m` > 1 a camera file in the
  sensor fit's format is written (`sensor_<name>_m<m>.json`) so the simulation loads it the same
  way, with the recurrence values recomputed from the filtered tracks.

Writes results/multi_drone/clutter_<name>.json (and the sensor files).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multidrone_sensor_fit as sf  # noqa: E402
from aerialrange.data.syndrone import syndrone_camera  # noqa: E402
from aerialrange.data.syndrone_archive import MemberStore  # noqa: E402
from aerialrange.evaluate.syndrone_eval import (  # noqa: E402
    TruthRules,
    classify_truth,
    keep_not_ignored,
    match_pairs,
)
from aerialrange.pipeline.geomap import footprint_ring  # noqa: E402
from aerialrange.pipeline.range_speed import ranging_model  # noqa: E402
from aerialrange.search.belief import TAIL_SIGMAS  # noqa: E402
from aerialrange.utils.config import REPO_ROOT, load_config  # noqa: E402

HEIGHT = 20
GROUP = "person"
IMAGE_W, IMAGE_H = 1920, 1080
INFO_RADII_M = (2.0, 5.0, 10.0)  # also reported, for reading the radius' effect
N_BOOT_FIT = 200
OUT = REPO_ROOT / "results" / "multi_drone"


def split_passes(frames: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """Sorted frame numbers split into passes where the gap exceeds max_gap frames."""
    return sf.passes_of(np.asarray(frames, int), max_gap)


def pass_index(passes: list[np.ndarray], frame: int) -> int:
    """Index of the pass holding `frame`, or of the last pass starting before it (a track can
    start just outside the searched ground)."""
    k = -1
    for i, p in enumerate(passes):
        if p[0] <= frame:
            k = i
        if p[0] <= frame <= p[-1]:
            return i
    return k


def recurrence(
    sites: pd.DataFrame,
    fa_rows: pd.DataFrame,
    seen: dict[int, np.ndarray],
    max_gap: int,
    radius_m: float,
) -> tuple[list[bool], list[bool], list[list[bool]]]:
    """(one flag per later pass over a site: a recurrence?, one flag per site: does it recur
    one on an earlier pass?, per site the flags of its later passes in order).

    `sites`: per false-alarm track (index track_id): n, e, first, last frame.
    `fa_rows`: every false-alarm track row (track_id, frame0, ground_n_m, ground_e_m).
    `seen[tid]`: the frames whose searched ground holds that site."""
    later, repeat, per_site = [], [], []
    for tid, s in sites.iterrows():
        passes = split_passes(seen[tid], max_gap)
        if not passes:
            repeat.append(False)
            per_site.append([])
            continue
        i0, i1 = pass_index(passes, int(s["first"])), pass_index(passes, int(s["last"]))
        other = fa_rows[fa_rows["track_id"] != tid]
        near = other[
            np.hypot(other["ground_n_m"] - s["n"], other["ground_e_m"] - s["e"]) <= radius_m
        ]
        near_frames = set(near["frame0"].astype(int).tolist())
        flags = [bool(near_frames & set(p.tolist())) for p in passes[i1 + 1 :]]
        later += flags
        per_site.append(flags)
        repeat.append(any(near_frames & set(p.tolist()) for p in passes[: max(i0, 0)]))
    return later, repeat, per_site


def fit_persistent(per_site: list[list[bool]]) -> tuple[float, float]:
    """Maximum-likelihood (share, fire) of a two-kind false-alarm model: a share of the sites
    seen are persistent look-alikes that fire on each later pass with probability `fire`; the
    rest never fire again. Uses the sites with at least one later pass. Sites with the same
    (later passes, recurrences) are counted together."""
    nr = np.array([(len(f), sum(f)) for f in per_site if f], int).reshape(-1, 2)
    if not len(nr) or nr[:, 1].sum() == 0:
        return 0.0, 0.0
    pairs, counts = np.unique(nr, axis=0, return_counts=True)
    share = np.linspace(0.001, 0.999, 999)[:, None, None]
    fire = np.linspace(0.01, 1.0, 100)[None, :, None]
    n = pairs[:, 0].astype(float)[None, None, :]
    r = pairs[:, 1].astype(float)[None, None, :]
    with np.errstate(divide="ignore"):
        pers = share * fire**r * (1.0 - fire) ** (n - r)
        ll = (np.log(pers + (1.0 - share) * (r == 0)) * counts[None, None, :]).sum(axis=2)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return float(share[i, 0, 0]), float(fire[0, j, 0])


def ground_positions(t: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Each track row's ground position; older runs did not write it, so it is rebuilt from
    the slant range, the azimuth and the camera pose, as ranging.range_boxes made them."""
    if "ground_n_m" in t and t["ground_n_m"].notna().any():
        return t
    by_frame = meta.set_index("frame") if "frame" in meta.columns else meta
    pz = by_frame.loc[t["frame"].to_numpy()]
    horiz = np.sqrt(np.maximum(t["range_m"].to_numpy() ** 2 - pz["h_m"].to_numpy() ** 2, 0.0))
    az = np.radians(t["azimuth_deg"].to_numpy())
    t = t.copy()
    t["ground_n_m"] = pz["n_m"].to_numpy() + horiz * np.cos(az)
    t["ground_e_m"] = pz["e_m"].to_numpy() + horiz * np.sin(az)
    return t


def match_town(
    tr: pd.DataFrame, tracks: pd.DataFrame, keep_ids: set[int] | None
) -> tuple[pd.DataFrame, dict[int, list[int]], set[tuple[int, int]]]:
    """Kept person-track rows (after ignore regions), per-track [kept frames, matched frames],
    and the matched (truth id, frame) pairs. `keep_ids`: tracks to keep (None: all)."""
    tg = tr[tr["group"] == GROUP]
    valid_by_f = {f: d for f, d in tg[tg["status"] == "valid"].groupby("frame")}
    ign_sel = ((tr["status"] == "ignore") & (tr["group"] == GROUP)) | tr["group"].isna()
    ign_by_f = {f: d for f, d in tr[ign_sel].groupby("frame")}
    trk = tracks[tracks["class"] == GROUP]
    if keep_ids is not None:
        trk = trk[trk["track_id"].isin(keep_ids)]
    life: dict[int, list[int]] = {}
    matched: set[tuple[int, int]] = set()
    kept = []
    for f, t in trk.groupby("frame0"):
        v = valid_by_f.get(f)
        vb = sf.mask_ltwh(v) if v is not None else np.zeros((0, 4))
        ib = sf.mask_ltwh(ign_by_f[f]) if f in ign_by_f else np.zeros((0, 4))
        t = t[keep_not_ignored(t[["left", "top", "width", "height"]].to_numpy(), vb, ib)]
        for tid in t["track_id"]:
            life.setdefault(int(tid), [0, 0])[0] += 1
        for a, b, _ in match_pairs(vb, t[["left", "top", "width", "height"]].to_numpy()):
            life[int(t.iloc[b]["track_id"])][1] += 1
            matched.add((int(v.iloc[a]["id"]), int(f)))
        kept.append(t)
    rows = pd.concat(kept) if kept else trk.iloc[:0]
    return rows, life, matched


def person_passes(
    tr: pd.DataFrame, matched: set[tuple[int, int]], min_dep: float, max_gap: int
) -> list[dict]:
    """Passes of each valid person as the sensor fit splits them, detected or not."""
    tg = tr[(tr["group"] == GROUP) & (tr["status"] == "valid")]
    out = []
    for tid, d in tg[tg["depression_deg"] >= min_dep].groupby("id"):
        fr = d.sort_values("frame")["frame"].to_numpy(int)
        for p in split_passes(fr, max_gap):
            out.append(
                {
                    "id": int(tid),
                    "first": int(p[0]),
                    "n_frames": len(p),
                    "detected": any((int(tid), int(f)) in matched for f in p),
                }
            )
    return out


def pass_correlation(rows: list[dict], seq_key: str = "seq") -> dict:
    df = pd.DataFrame(rows).sort_values([seq_key, "id", "first"])
    pairs = []
    for _, d in df.groupby([seq_key, "id"]):
        det = d["detected"].to_numpy(bool)
        pairs += list(zip(det[:-1], det[1:], strict=True))
    a = np.array(pairs, bool).reshape(-1, 2)
    hit, miss = a[a[:, 0]], a[~a[:, 0]]
    return {
        "n_passes": len(df),
        "pd_per_pass": float(df["detected"].mean()),
        "n_pairs": len(a),
        "after_detected": {"n": len(hit), "pd": float(hit[:, 1].mean()) if len(hit) else None},
        "after_missed": {"n": len(miss), "pd": float(miss[:, 1].mean()) if len(miss) else None},
    }


def pd_bins(rows: list[dict], edges_keys: list[str]) -> dict:
    """Detection share by pass length in the sensor file's bins ("a-b": (a, b], first [a, b])."""
    df = pd.DataFrame(rows)
    out = {}
    for j, k in enumerate(edges_keys):
        a, b = (int(x) for x in k.split("-"))
        lo = df["n_frames"] >= a if j == 0 else df["n_frames"] > a
        s = df[lo & (df["n_frames"] <= b)]
        out[k] = {"n": len(s), "pd": float(s["detected"].mean()) if len(s) else None}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="p10_v2_cal")
    ap.add_argument("--split", default="test")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        print("reading test-town outputs needs --allow-test (outputs of the one-time run only)")
        return 1
    sensor = json.loads((OUT / f"sensor_{args.name}.json").read_text())
    s20 = sensor["by_height"][f"h{HEIGHT}"]
    radius = TAIL_SIGMAS * float(s20["range_half_68_m_median"])
    min_dep = float(sensor["min_depression_deg"])
    search = json.loads(
        (REPO_ROOT / "results" / "syndrone" / args.split / f"search_{args.name}.json").read_text()
    )
    area_km2 = float(search[GROUP]["by_height"][f"h{HEIGHT}"]["ground_searched_km2"])
    trk_cfg = REPO_ROOT / load_config("run").require("tracker")["config"]
    max_gap = int(yaml.safe_load(trk_cfg.read_text())["max_lost"])
    cfg = load_config("syndrone")
    rules = TruthRules(
        float(cfg.require("truth_min_visibility")), int(cfg.require("truth_min_instance_px"))
    )
    run_cfg = load_config("run")
    cam = syndrone_camera()
    camera = {"focal_px": cam.fx, "cx": cam.cx, "cy": cam.cy}
    model = ranging_model(lambda k: camera if k == "camera" else run_cfg.require(k))
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    from syndrone_refilter import flight_meta  # needs the dataset; imported only when run

    towns = [s.rsplit("_h", 1)[0] for s in sensor["sequences"] if s.endswith(f"_h{HEIGHT}")]
    data = []
    for town in towns:
        tr = pd.read_csv(sf.TRUTH / town / "objects.csv.gz")
        tr = classify_truth(tr[tr["height"] == HEIGHT], rules)
        tr = tr[tr["status"] != "absent"]
        tracks = pd.read_csv(sf.RUNS / args.name / f"{town}_h{HEIGHT}" / "tracks.csv")
        tracks["frame0"] = tracks["frame"] - 1
        meta = flight_meta(store, town, HEIGHT)
        tracks = ground_positions(tracks, meta)
        rings = []
        for _, pose in meta.iterrows():
            r = footprint_ring(pose, model, IMAGE_W, IMAGE_H, min_dep)
            rings.append(MPath(r) if r is not None else None)
        data.append((town, tr, tracks, rings))
        print(f"{town}_h{HEIGHT}: {len(tracks)} track rows, {len(rings)} frames", flush=True)

    def persistent(later: list[bool], repeat: list[bool], per_site: list[list[bool]]) -> dict:
        share, fire = fit_persistent(per_site)
        rng = np.random.default_rng(0)
        idx = [i for i, f in enumerate(per_site) if f]
        boots = []
        for _ in range(N_BOOT_FIT):
            pick = rng.choice(idx, len(idx)) if idx else []
            boots.append(fit_persistent([per_site[i] for i in pick]))
        ci = (
            {
                k: [float(np.percentile([b[j] for b in boots], q)) for q in (2.5, 97.5)]
                for j, k in enumerate(("share", "fire_per_pass"))
            }
            if boots
            else {}
        )
        return {
            "later_passes": len(later),
            "q": float(np.mean(later)) if later else None,
            "repeat_share": float(np.mean(repeat)) if repeat else None,
            "sites_with_later_passes": len(idx),
            "persistent": {
                "share": share,
                "fire_per_pass": fire,
                "ci95_resample_sites": ci,
                "rule": "maximum likelihood, two kinds: persistent look-alikes firing with "
                "fire_per_pass on each later pass, the rest never again",
            },
        }

    def fit(keep_min: int, radii: tuple[float, ...]) -> dict:
        fa_tracks, later, repeat, prow = 0, {r: [] for r in radii}, {r: [] for r in radii}, []
        per_site: dict[float, list[list[bool]]] = {r: [] for r in radii}
        lives = {"false_alarm": [], "person": []}
        for town, tr, tracks, rings in data:
            _, life0, _ = match_town(tr, tracks, None)
            keep = {t for t, (n, _) in life0.items() if n >= keep_min}
            rows, life, matched = match_town(tr, tracks, keep)
            fa_ids = {t for t, (_, m) in life.items() if m == 0}
            fa_tracks += len(fa_ids)
            lives["false_alarm"] += [n for t, (n, _) in life.items() if t in fa_ids]
            lives["person"] += [n for t, (n, _) in life.items() if t not in fa_ids]
            fa_rows = rows[rows["track_id"].isin(fa_ids) & rows["ground_n_m"].notna()]
            g = fa_rows.groupby("track_id")
            sites = g[["ground_n_m", "ground_e_m"]].median()
            sites.columns = ["n", "e"]
            sites["first"], sites["last"] = g["frame0"].min(), g["frame0"].max()
            pts = sites[["n", "e"]].to_numpy()
            inside = np.zeros((len(rings), len(sites)), bool)
            for f, ring in enumerate(rings):
                if ring is not None and len(pts):
                    inside[f] = ring.contains_points(pts)
            # rings are per frame from 1, frame0 = frame - 1: row f is frame0 f
            seen = {tid: np.nonzero(inside[:, j])[0] for j, tid in enumerate(sites.index)}
            for r in radii:
                lt, rp, ps = recurrence(sites, fa_rows, seen, max_gap, r)
                later[r] += lt
                repeat[r] += rp
                per_site[r] += ps
            for p in person_passes(tr, matched, min_dep, max_gap):
                prow.append({**p, "seq": town})
        res = {
            "min_track_frames": keep_min,
            "false_alarm_tracks": fa_tracks,
            "false_alarms_per_km2": fa_tracks / area_km2,
            "track_life_median_frames": {k: float(np.median(v)) for k, v in lives.items() if v},
            "recurrence": {f"{r:g}": persistent(later[r], repeat[r], per_site[r]) for r in radii},
            "pass_correlation": pass_correlation(prow),
            "pd_by_pass_frames": pd_bins(prow, list(s20["pd_by_pass_frames"])),
        }
        return res

    base = fit(1, (radius, *INFO_RADII_M))
    med = base["track_life_median_frames"]
    ms = sorted({int(round(med["false_alarm"])), int(round(med["person"]))} - {1})
    by_m = {"1": base}
    for m in ms:
        by_m[str(m)] = fit(m, (radius,))
        print(f"m = {m}: {by_m[str(m)]['false_alarm_tracks']} false-alarm tracks", flush=True)
    res = {
        "name": args.name,
        "split": args.split,
        "height_m": HEIGHT,
        "sequences": [f"{t}_h{HEIGHT}" for t in towns],
        "radius_m": radius,
        "radius_rule": "TAIL_SIGMAS x the sensor file's 68 % half-width at 20 m",
        "pass_split_gap_frames": max_gap,
        "ground_searched_km2": area_km2,
        "persistence_values": {
            "rule": "1 (none), median life of false-alarm tracks, median life of person tracks",
            "m": [1, *ms],
        },
        "by_min_track_frames": by_m,
    }
    # check: with no filter, the pass detection shares equal the sensor fit's
    for k, v in base["pd_by_pass_frames"].items():
        if v["n"] != s20["pd_by_pass_frames"][k]["n"]:
            print(
                f"warning: bin {k} has {v['n']} passes, "
                f"the sensor file {s20['pd_by_pass_frames'][k]['n']}"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"clutter_{args.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    for m in ms:
        f = by_m[str(m)]
        sm = json.loads(json.dumps(sensor))
        h = sm["by_height"][f"h{HEIGHT}"]
        n_det = sum(round(v["pd"] * v["n"]) for v in f["pd_by_pass_frames"].values() if v["n"])
        n_all = sum(v["n"] for v in f["pd_by_pass_frames"].values())
        h["pd_by_pass_frames"] = f["pd_by_pass_frames"]
        h["pd_per_pass"] = n_det / n_all
        h["false_alarms_per_km2"] = f["false_alarms_per_km2"]
        h["false_alarm_tracks"] = f["false_alarm_tracks"]
        sm["persistence_filter"] = {"min_track_frames": m, "source": f"clutter_{args.name}.json"}
        for other in ("h50", "h80"):  # not refitted: people are not found there (D-752)
            sm["by_height"].pop(other, None)
        (OUT / f"sensor_{args.name}_m{m}.json").write_text(json.dumps(sm, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "by_min_track_frames"}, indent=1))
    for m, f in by_m.items():
        print(m, f["false_alarms_per_km2"], f["recurrence"], f["pass_correlation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
