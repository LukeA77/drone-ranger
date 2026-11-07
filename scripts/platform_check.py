"""Check that the pipeline gives the same outputs on two machines (x86 and ARM64).

Usage:
  python scripts/platform_check.py --tag x86 [--frames DIR] [--n 60] [--config run]
  python scripts/platform_check.py --compare results/edge/platform_x86.json \
         results/edge/platform_arm64.json

Run mode writes results/edge/platform_<tag>.json with two parts, both from `aerialrange run`:
- `demo`: the whole pipeline on the synthetic flight (`aerialrange demo`: boxes from the
  scene, so tracking, ranging, the Kalman filter bank and the outputs, no model);
- `real`: the whole pipeline with the configured ONNX detector on the first `n` frames of a
  real sequence (default: VisDrone MOT val uav0000086_00000_v, from the prepared data;
  --frames for any folder of JPEG frames). Real frames have no flight data, so the
  synthetic flight's first camera pose is used for every frame and the camera centre is put
  at the image centre: the ranges are not meaningful, only comparable.
It records the raw detections and every output row, the model file's SHA-256 and the
machine. Nothing here is timed (speed is not compared across machines, and emulated ARM64
says nothing about Pi speed).

Compare mode matches the two files frame by frame (same class, IoU >= 0.5, best overlaps
first) and passes if every check is within its tolerance (TOL below). Float32 kernels
differ between x86 (AVX) and ARM (NEON) builds of ONNX Runtime in the last bits, so a
detection right at the score floor can appear on one side only and a tracker decision on
a threshold can flip; the tolerances allow that and nothing larger. Writes
results/edge/platform_compare_<tagA>_<tagB>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "results" / "edge"
DEFAULT_SEQ = "uav0000086_00000_v"
TOL = {
    "detection_agreement_min": 0.99,  # share of detections matched, each side
    "box_max_diff_px": 0.5,  # matched detections and output rows
    "score_max_diff": 0.005,
    "row_agreement_min": 0.99,  # share of output rows matched, each side
    "id_consistency_min": 0.99,  # matched rows whose track ids map one to one
    "range_max_rel_diff": 0.001,  # matched valid rows
}
ROW_FIELDS = ["frame", "track_id", "class", "left", "top", "width", "height", "range_m", "valid"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(tracks_csv: Path) -> list[list]:
    df = pd.read_csv(tracks_csv)
    return df[ROW_FIELDS].astype(object).where(df[ROW_FIELDS].notna(), None).values.tolist()


def run_check(tag: str, frames_dir: Path | None, n: int, config: str) -> Path:
    import cv2

    from aerialrange.pipeline.demo import demo
    from aerialrange.pipeline.run import run

    det = load_config(config).require("detector")
    weights = REPO_ROOT / det["weights"]
    if frames_dir is None:
        seq = REPO_ROOT / "cache" / "visdrone" / "mot" / "val" / DEFAULT_SEQ
        info = json.loads((seq / "seqinfo.json").read_text())
        frames_dir = REPO_ROOT / Path(info["frame_dir"].replace("\\", "/"))
    frames = sorted(Path(frames_dir).glob("*.jpg"))[:n]
    if not frames:
        raise SystemExit(f"no JPEG frames in {frames_dir}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        demo_summary = demo(tmp / "demo")
        demo_rows = _rows(tmp / "demo" / "run" / "tracks.csv")
        pose = pd.read_csv(tmp / "demo" / "flight" / "metadata.csv").iloc[0]
        meta = pd.DataFrame(
            {
                "frame": np.arange(1, len(frames) + 1),
                "t_s": np.arange(len(frames)) / 25.0,
                **{c: pose[c] for c in ("n_m", "e_m", "h_m", "yaw_rad", "tilt_deg")},
            }
        )
        h, w = cv2.imread(str(frames[0])).shape[:2]
        dets: list[dict] = []
        run(
            (cv2.imread(str(p)) for p in frames),
            meta,
            tmp / "real",
            config=config,
            overrides={"camera": {"cx": w / 2, "cy": h / 2}},
            detections_out=dets,
            write_video=False,
        )
        real_rows = _rows(tmp / "real" / "tracks.csv")
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tag": tag,
        "machine": {
            "machine": platform.machine(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "onnxruntime": version("onnxruntime"),
            "opencv": cv2.__version__,
        },
        "config": config,
        "detector": {**det, "sha256": _sha256(weights)},
        "frames": f"{Path(frames_dir).name}: first {len(frames)} JPEG frames ({w}x{h})",
        "row_fields": ROW_FIELDS,
        "demo": {"summary": demo_summary, "rows": demo_rows},
        "real": {"detections": dets, "rows": real_rows},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"platform_{tag}.json"
    dest.write_text(json.dumps(out, default=float) + "\n")
    print(
        f"{tag}: demo {len(demo_rows)} rows; real {len(dets)} detections, "
        f"{len(real_rows)} rows; wrote {dest}"
    )
    return dest


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a, area_b = a[:, 2] * a[:, 3], b[:, 2] * b[:, 3]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def _match(
    fa: np.ndarray,
    ba: np.ndarray,
    ca: np.ndarray,
    fb: np.ndarray,
    bb: np.ndarray,
    cb: np.ndarray,
) -> list[tuple[int, int]]:
    """One-to-one pairs within each frame: same class, IoU >= 0.5, best overlaps first."""
    pairs = []
    for f in np.union1d(fa, fb):
        ia, ib = np.flatnonzero(fa == f), np.flatnonzero(fb == f)
        if not len(ia) or not len(ib):
            continue
        m = _iou(ba[ia], bb[ib]) * (ca[ia][:, None] == cb[ib][None, :])
        ii, jj = np.nonzero(m >= 0.5)
        used_a, used_b = set(), set()
        for k in np.argsort(-m[ii, jj], kind="stable"):
            i, j = ii[k], jj[k]
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                pairs.append((int(ia[i]), int(ib[j])))
    return pairs


def _compare_dets(a: list[dict], b: list[dict]) -> dict:
    def arr(d: list[dict]) -> tuple:
        df = pd.DataFrame(d, columns=["frame", "left", "top", "width", "height", "score", "cls"])
        return (
            df["frame"].to_numpy(),
            df[["left", "top", "width", "height"]].to_numpy(float),
            df["cls"].to_numpy(),
            df["score"].to_numpy(float),
        )

    fa, ba, ca, sa = arr(a)
    fb, bb, cb, sb = arr(b)
    p = _match(fa, ba, ca, fb, bb, cb)
    ia, ib = (np.array([x[k] for x in p], int) for k in (0, 1))
    return {
        "n_a": len(a),
        "n_b": len(b),
        "agreement_a": len(p) / max(len(a), 1),
        "agreement_b": len(p) / max(len(b), 1),
        "box_max_diff_px": float(np.abs(ba[ia] - bb[ib]).max()) if len(p) else 0.0,
        "score_max_diff": float(np.abs(sa[ia] - sb[ib]).max()) if len(p) else 0.0,
    }


def _compare_rows(a: list[list], b: list[list]) -> dict:
    da, db = (pd.DataFrame(r, columns=ROW_FIELDS) for r in (a, b))
    box = ["left", "top", "width", "height"]
    p = _match(
        da["frame"].to_numpy(),
        da[box].to_numpy(float),
        da["class"].to_numpy(),
        db["frame"].to_numpy(),
        db[box].to_numpy(float),
        db["class"].to_numpy(),
    )
    ia, ib = (np.array([x[k] for x in p], int) for k in (0, 1))
    ida, idb = da["track_id"].to_numpy()[ia], db["track_id"].to_numpy()[ib]
    # Ids are arbitrary labels: a pair is consistent if it follows the majority mapping both ways.
    fwd = {k: c.most_common(1)[0][0] for k, c in _groups(ida, idb).items()}
    bwd = {k: c.most_common(1)[0][0] for k, c in _groups(idb, ida).items()}
    consistent = sum(fwd[x] == y and bwd[y] == x for x, y in zip(ida, idb, strict=True))
    va = da["valid"].to_numpy(bool)[ia] & db["valid"].to_numpy(bool)[ib]
    ra, rb = da["range_m"].to_numpy(float)[ia][va], db["range_m"].to_numpy(float)[ib][va]
    return {
        "n_a": len(da),
        "n_b": len(db),
        "agreement_a": len(p) / max(len(da), 1),
        "agreement_b": len(p) / max(len(db), 1),
        "box_max_diff_px": (
            float(np.abs(da[box].to_numpy(float)[ia] - db[box].to_numpy(float)[ib]).max())
            if len(p)
            else 0.0
        ),
        "id_consistency": consistent / max(len(p), 1),
        "range_max_rel_diff": float(np.max(np.abs(ra - rb) / np.abs(rb))) if len(ra) else 0.0,
    }


def _groups(x: np.ndarray, y: np.ndarray) -> dict:
    g: dict = {}
    for a, b in zip(x, y, strict=True):
        g.setdefault(a, Counter())[b] += 1
    return g


def compare(path_a: Path, path_b: Path, out_dir: Path = OUT) -> int:
    a, b = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
    if a["detector"]["sha256"] != b["detector"]["sha256"]:
        print("the two runs used different model files")
        return 1
    res = {
        "demo_rows": _compare_rows(a["demo"]["rows"], b["demo"]["rows"]),
        "real_detections": _compare_dets(a["real"]["detections"], b["real"]["detections"]),
        "real_rows": _compare_rows(a["real"]["rows"], b["real"]["rows"]),
    }
    checks: dict[str, bool] = {}
    d = res["real_detections"]
    checks["detection_agreement"] = (
        min(d["agreement_a"], d["agreement_b"]) >= TOL["detection_agreement_min"]
    )
    checks["detection_boxes"] = d["box_max_diff_px"] <= TOL["box_max_diff_px"]
    checks["detection_scores"] = d["score_max_diff"] <= TOL["score_max_diff"]
    for part in ("demo_rows", "real_rows"):
        r = res[part]
        checks[f"{part}_agreement"] = (
            min(r["agreement_a"], r["agreement_b"]) >= TOL["row_agreement_min"]
        )
        checks[f"{part}_boxes"] = r["box_max_diff_px"] <= TOL["box_max_diff_px"]
        checks[f"{part}_ids"] = r["id_consistency"] >= TOL["id_consistency_min"]
        checks[f"{part}_ranges"] = r["range_max_rel_diff"] <= TOL["range_max_rel_diff"]
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "a": {"tag": a["tag"], "machine": a["machine"]},
        "b": {"tag": b["tag"], "machine": b["machine"]},
        "detector": a["detector"],
        "frames": a["frames"],
        "tolerance": TOL,
        "results": res,
        "checks": {k: bool(v) for k, v in checks.items()},
        "passed": bool(all(checks.values())),
    }
    dest = out_dir / f"platform_compare_{a['tag']}_{b['tag']}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"checks": out["checks"], "passed": out["passed"]}, indent=1))
    print(f"wrote {dest}")
    return 0 if out["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--frames", type=Path, default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--config", default="run")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.tag:
        ap.error("--tag is required to run the check")
    run_check(args.tag, args.frames, args.n, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
