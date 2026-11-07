"""Check the recovered SynDrone base height z0 and the box reprojection.

Reads the truth chunks of one town (scripts/syndrone_truth.py) and writes:
- `cache/syndrone/truth/<town>/z0.csv`: per frame the joint fit, its standard error, the
  per-camera medians, and the final z0 used for truth;
- `cache/syndrone/truth/<town>/objects.csv.gz`: every truth row with z0-dependent geometry
  (range, near-edge range, depression, projected box) recomputed from the final z0;
- `results/syndrone/<split>/z0_check_<town>.json`: fit quality, camera agreement, smoothness,
  and the reprojection error per camera height against the instance masks;
- with --sheet, a contact sheet of projected 3D boxes over frames, in
  `cache/syndrone/figures/` (dataset imagery is not committed).

Final z0: the per-frame joint fit where it rests on at least `--min-boxes` vehicles, else
linear interpolation in time between such frames (z0 comes from a cubic spline of a flight
log, so it is smooth). Frames before the first or after the last fitted frame take the
nearest fit. The choice is reported (fitted / interpolated share). A town where no frame
stands on enough vehicles uses the walker fit instead (scripts/syndrone_z0_walker_series.py,
D-720), counted the same way; the source is reported.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys

import numpy as np
import pandas as pd

from aerialrange.data.syndrone import (
    BOTTOM,
    CameraRecord,
    SynDroneSequence,
    parse_boxes,
    syndrone_camera,
)
from aerialrange.data.syndrone import HEIGHT as IMG_H
from aerialrange.data.syndrone import WIDTH as IMG_W
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.evaluate.syndrone_eval import GROUPS
from aerialrange.evaluate.syndrone_truth import _nearest_on_polygon, project_ue
from aerialrange.utils.config import REPO_ROOT, load_config

TRUTH = REPO_ROOT / "cache" / "syndrone" / "truth"
EDGE_FIELDS = ("left", "top", "right", "bottom")


def split_of(town: str) -> str:
    s = json.loads((REPO_ROOT / "results" / "splits.json").read_text(encoding="utf-8"))
    return next(k for k, v in s["syndrone"]["splits"].items() if town in v)


def load_chunks(town: str) -> list[dict]:
    recs = []
    for p in sorted((TRUTH / town).glob("chunk_*.json.gz")):
        recs += json.loads(gzip.decompress(p.read_bytes()))
    frames = [r["frame"] for r in recs]
    if len(frames) != len(set(frames)):
        raise RuntimeError(f"{town}: overlapping truth chunks (a frame appears twice)")
    return sorted(recs, key=lambda r: r["frame"])


def final_z0(z: pd.DataFrame, min_boxes: int) -> tuple[np.ndarray, np.ndarray]:
    ok = (z["z0_n"] >= min_boxes) & np.isfinite(z["z0"])
    f = z["frame"].to_numpy()
    if not ok.any():
        raise RuntimeError("no frame has enough vehicles to fit z0")
    zf = np.interp(f, f[ok], z.loc[ok, "z0"].to_numpy())
    return zf, ok.to_numpy()


def reproj_stats(df: pd.DataFrame) -> dict:
    """Edge errors (projected 3D box minus instance-mask box) on clean views."""
    clean = df[(df["status_clean"]) & np.isfinite(df["mask_left"])]
    if not len(clean):
        return {"n": 0}
    e = np.column_stack([clean[f"proj_{k}"] - clean[f"mask_{k}"] for k in EDGE_FIELDS])
    cen = np.column_stack(
        [
            (e[:, 0] + e[:, 2]) / 2,  # horizontal centre offset
            (e[:, 1] + e[:, 3]) / 2,  # vertical centre offset
        ]
    )
    return {
        "n": int(len(clean)),
        "median_abs_edge_px": float(np.median(np.abs(e))),
        "p90_abs_edge_px": float(np.percentile(np.abs(e), 90)),
        "median_signed_edge_px": dict(
            zip(EDGE_FIELDS, np.median(e, 0).round(2).tolist(), strict=True)
        ),
        "median_abs_centre_offset_px": float(np.median(np.abs(cen))),
        "median_signed_centre_offset_px": np.median(cen, 0).round(3).tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--min-boxes", type=int, default=3)
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args(argv)
    recs = load_chunks(args.town)
    if len(recs) != 3000:
        print(f"only {len(recs)} frames of truth; run scripts/syndrone_truth.py first")
        return 1
    split = split_of(args.town)
    z = pd.DataFrame(
        [
            {
                "frame": r["frame"],
                "z0": r["z0"],
                "z0_se": r["z0_se"],
                "z0_scale": r["z0_scale"],
                "z0_n": r["z0_n"],
                **{f"z0_h{h}": r["per_view"][str(h)][0] for h in (20, 50, 80)},
                **{f"n_h{h}": r["per_view"][str(h)][1] for h in (20, 50, 80)},
            }
            for r in recs
        ]
    )
    source = "vehicles"
    if not ((z["z0_n"] >= args.min_boxes) & np.isfinite(z["z0"])).any():
        # No frame stands on enough vehicles (Town02: its vehicle actors are parked away from
        # roads). Use the walker fit, checked on development and calibration towns (D-720).
        wpath = TRUTH / args.town / "z0_walkers.csv"
        if not wpath.exists():
            print("no vehicle-fitted frame; run scripts/syndrone_z0_walker_series.py first")
            return 1
        w = pd.read_csv(wpath).set_index("frame").reindex(z["frame"])
        z["z0"] = w["z0_walkers"].to_numpy()
        z["z0_se"] = w["se_walkers"].to_numpy()
        z["z0_n"] = w["n_walkers"].to_numpy()
        source = "walkers"
    z["z0_final"], fitted = final_z0(z, args.min_boxes)
    z["fitted"] = fitted
    z.to_csv(TRUTH / args.town / "z0.csv", index=False)

    # Recompute z0-dependent geometry with the final z0.
    cam = syndrone_camera()
    rows = []
    for r in recs:
        k = r["frame"]
        z0 = float(z.at[k, "z0_final"])
        for h, objs in r["objects"].items():
            c = r["cameras"][h]
            rec = CameraRecord(c["x"], c["y"], c["h"], c["pitch"], c["yaw"], c["roll"])
            C = np.array([c["x"], c["y"], c["h"] + z0])
            for o in objs:
                f = np.array(o["foot_ue"])
                horiz = float(np.hypot(*(f[:2] - C[:2])))
                rows.append(
                    {
                        "frame": k,
                        "height": int(h),
                        **{
                            kk: o[kk]
                            for kk in (
                                "id",
                                "tag",
                                "level",
                                "n_instance_px",
                                "n_occluded_px",
                                "visibility",
                                "truncation",
                            )
                        },
                        "range_m": float(np.linalg.norm(f - C)),
                        "depression_deg": float(np.degrees(np.arctan2(C[2] - f[2], horiz))),
                        "foot_x": f[0],
                        "foot_y": f[1],
                        "foot_z": f[2],
                        "range_m_framefit": o["range_m"],
                        "range_near_m_framefit": o["range_near_m"],
                        **{f"mask_{n}": v for n, v in zip(EDGE_FIELDS, o["box_mask"], strict=True)},
                        "_rec": rec,
                        "_z0": z0,
                    }
                )
    df = pd.DataFrame(rows)
    # Chunks built before parse_boxes dropped duplicate actor entries (D-709) hold the
    # duplicate twice; drop it only if the two rows are identical.
    key = ["frame", "height", "id", "level"]
    dup = df.duplicated(key, keep="first")
    if dup.any():
        plain = df.drop(columns=["_rec"]).round(9)
        if plain.duplicated(keep="first").sum() != dup.sum():
            raise RuntimeError(f"{args.town}: same object twice in a frame with different truth")
        print(
            f"dropped {int(dup.sum())} duplicate truth rows (ids {sorted(df[dup]['id'].unique())})"
        )
        df = df[~dup].reset_index(drop=True)
    # Near-edge range and projected boxes need the corners: recompute from the box files.
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    seq = SynDroneSequence(store, args.town, 80)
    proj = {n: np.full(len(df), np.nan) for n in EDGE_FIELDS}
    near = np.full(len(df), np.nan)
    for k, g in df.groupby("frame"):
        b = parse_boxes(store.read(seq.member("bboxes", int(k))))
        pos = {int(i): j for j, i in enumerate(b.ids)}
        for idx, row in g.iterrows():
            j = pos[int(row["id"])]
            uv, _ = project_ue(cam, row["_rec"], row["_z0"], b.corners[j])
            proj["left"][idx], proj["top"][idx] = np.floor(uv.min(0))
            proj["right"][idx], proj["bottom"][idx] = np.ceil(uv.max(0))
            foot = b.corners[j, BOTTOM]
            rec = row["_rec"]
            C = np.array([rec.x, rec.y, rec.h + row["_z0"]])
            q = _nearest_on_polygon(foot[[0, 1, 3, 2], :2], C[:2])
            near[idx] = float(np.linalg.norm(np.array([q[0], q[1], foot[:, 2].mean()]) - C))
    for n in EDGE_FIELDS:
        df[f"proj_{n}"] = proj[n]
    df["range_near_m"] = near
    df = df.drop(columns=["_rec", "_z0"])
    in_group = df["tag"].isin([t for ts in GROUPS.values() for t in ts])
    df["status_clean"] = (
        in_group
        & ~df["level"]
        & (df["visibility"] >= 0.9)
        & (df["truncation"] == 0)
        & (df["n_instance_px"] >= 100)
    )
    df.to_csv(TRUTH / args.town / "objects.csv.gz", index=False)

    agree = {}
    for a, b in ((20, 50), (20, 80), (50, 80)):
        m = (z[f"n_h{a}"] >= 2) & (z[f"n_h{b}"] >= 2)
        dd = (z.loc[m, f"z0_h{a}"] - z.loc[m, f"z0_h{b}"]).to_numpy()
        agree[f"h{a}_minus_h{b}"] = {
            "frames": int(m.sum()),
            "median_m": float(np.median(dd)) if len(dd) else None,
            "mad_m": float(np.median(np.abs(dd - np.median(dd)))) if len(dd) else None,
        }
    fz = z.loc[z["fitted"]]
    resid = (
        fz["z0"] - np.interp(fz["frame"], fz["frame"].to_numpy()[::5], fz["z0"].to_numpy()[::5])
    ).to_numpy()
    reproj = {f"h{h}": reproj_stats(df[df["height"] == h]) for h in (20, 50, 80)}
    nadir = reproj["h80"].get("median_abs_edge_px")
    summary = {
        "town": args.town,
        "split": split,
        "frames": len(z),
        "z0_source": source,
        "fitted_share": float(z["fitted"].mean()),
        "min_boxes": args.min_boxes,
        "z0_m": {
            "median": float(z["z0_final"].median()),
            "min": float(z["z0_final"].min()),
            "max": float(z["z0_final"].max()),
        },
        "z0_se_m": {"median": float(fz["z0_se"].median()), "p90": float(fz["z0_se"].quantile(0.9))},
        "boxes_per_frame_median": float(z["z0_n"].median()),
        "camera_agreement": agree,
        "smoothness_resid_vs_5frame_interp_m": {
            "median_abs": float(np.median(np.abs(resid))),
            "p90_abs": float(np.percentile(np.abs(resid), 90)),
        },
        "reprojection": reproj,
        "reproj_px_max_proposal": None if nadir is None else nadir + 1.0,
        "reproj_rule": "proposal = straight-down (h80) median |edge error| + 1 px; at nadir a "
        "z0 error hardly moves the projection, so that median is the gap between "
        "the object's shape and its box (D-704)",
        "clean_view_definition": "person or vehicle actor, visibility >= 0.9, no truncation, "
        ">= 100 instance pixels",
    }
    out = REPO_ROOT / "results" / "syndrone" / split
    out.mkdir(parents=True, exist_ok=True)
    (out / f"z0_check_{args.town}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if args.sheet:
        contact_sheet(args.town, store, df, z)
    return 0


def contact_sheet(town: str, store: MemberStore, df: pd.DataFrame, z: pd.DataFrame) -> None:
    """Frames with projected 3D box wireframes (yellow) and instance-mask boxes (cyan)."""
    import cv2

    cam = syndrone_camera()
    tiles = []
    for h in (20, 50, 80):
        seq = SynDroneSequence(store, town, h)
        for k in (300, 1500, 2700):
            img = seq.rgb(k, keep=True)
            rec = seq.camera_record(k)
            b = parse_boxes(store.read(seq.member("bboxes", k)))
            g = df[(df["frame"] == k) & (df["height"] == h) & (df["n_instance_px"] > 0)]
            pos = {int(i): j for j, i in enumerate(b.ids)}
            for _, row in g.iterrows():
                uv, zc = project_ue(
                    cam, rec, float(z.at[k, "z0_final"]), b.corners[pos[int(row["id"])]]
                )
                if np.any(zc <= 0):
                    continue
                p = uv.astype(int)
                for a, c in (
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (0, 2),
                    (1, 3),
                    (4, 6),
                    (5, 7),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ):
                    cv2.line(img, tuple(p[a]), tuple(p[c]), (0, 230, 255), 1, cv2.LINE_AA)
                if np.isfinite(row["mask_left"]):
                    cv2.rectangle(
                        img,
                        (int(row["mask_left"]), int(row["mask_top"])),
                        (int(row["mask_right"]), int(row["mask_bottom"])),
                        (255, 255, 0),
                        1,
                    )
            cv2.putText(
                img,
                f"{town} {h} m frame {k}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            tiles.append(cv2.resize(img, (IMG_W // 2, IMG_H // 2), interpolation=cv2.INTER_AREA))
    rows = [np.hstack(tiles[i : i + 3]) for i in range(0, 9, 3)]
    out = REPO_ROOT / "cache" / "syndrone" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"reproj_{town}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"contact sheet: {out / f'reproj_{town}.jpg'}")


if __name__ == "__main__":
    sys.exit(main())
