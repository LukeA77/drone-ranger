"""Search map of a flight: the map viewer with the ground image, the video and the finds (D-750).

Usage: python scripts/make_search_map.py auair --clip 35 --source cache/runs/v2_clip35
       python scripts/make_search_map.py syndrone --town Town07 --height 50 --source v2_dfine
       [--name <output folder name>] [--config run]

Writes cache/maps/<name>/: map.html (open it in a browser; no network needed), map.geojson,
ground.webp (the flight's own frames laid on the ground, pipeline/mosaic.py) and video.mp4 (the
flight's frames, H.264, for the linked video panel).

- AU-AIR: the source run's tracks are replayed through the range and speed stage with the
  configuration given, as scripts/auair_map.py does; the origin is the clip's first GPS fix.
- SynDrone (development and calibration towns only): the source run's tracks
  (cache/syndrone/runs/<source>/) are replayed as scripts/syndrone_refilter.py does, with exact
  poses and the interval calibration of the version 2 detector, so the map shows the product
  pipeline. SynDrone has no geographic position, so the town is placed at latitude 0,
  longitude 0 for display (stated in the page). The simulation's true positions of the people
  and vehicles that count as findable (in view, big enough, not hidden: the evaluation's valid
  truth) are added as a layer, every video frame.
Frames: every frame of an AU-AIR clip; every second SynDrone frame (the detector's own stride)
for the video and every tenth for the ground image, which is enough overlap at SynDrone's
speeds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aerialrange.geometry.frames import ned_to_geodetic
from aerialrange.pipeline.geomap import build_geojson, min_reliable_depression_deg, write_search_map
from aerialrange.pipeline.intervals import config_from_calibration
from aerialrange.pipeline.mosaic import Camera, build_ground_image
from aerialrange.pipeline.range_speed import (
    RangeSpeedStage,
    exact_pose_overrides,
    ranging_model,
    replay,
)
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syndrone_refilter import flight_meta  # noqa: E402


def truth_match(tracks: pd.DataFrame, truth: pd.DataFrame, flight: pd.DataFrame) -> dict[str, str]:
    """Per track, against the simulation's truth:
    - "real": matched a valid truth box of its group (IoU at least 0.5) in some frame;
    - "near": never matched, but in some frame it was kept, a valid truth object of its group
      stood inside the track's own 95 % position ellipse there (the map's ellipse: ground
      covariance calibrated along the line of sight, geomap.calibrate_along_sight), so a crew
      sent to it would find that object; on small boxes the IoU rule misses such tracks;
    - "false alarm": kept in some frame, neither of the above.
    "real" and "false alarm" together are D-749's search counts per track (near tracks count
    as false alarms there)."""
    from scipy.stats import chi2

    from aerialrange.evaluate.syndrone_eval import keep_not_ignored, match_pairs
    from aerialrange.pipeline.geomap import ELLIPSE_LEVEL, calibrate_along_sight
    from syndrone_evaluate import mask_ltwh

    gate = float(chi2.ppf(ELLIPSE_LEVEL, 2))
    cov_cols = ["ground_cov_nn_m2", "ground_cov_ne_m2", "ground_cov_ee_m2"]
    c = tracks[cov_cols].to_numpy(float)
    covs = np.stack([c[:, [0, 1]], c[:, [1, 2]]], axis=1)
    covs = calibrate_along_sight(covs, tracks, flight)
    row_of = {k: i for i, k in enumerate(tracks.index)}
    kept: set[int] = set()
    real: set[int] = set()
    near: set[int] = set()
    tr_by_f = {f: d for f, d in truth.groupby("frame")}
    for f, tk in tracks.groupby("frame"):
        tf = tr_by_f.get(int(f) - 1)
        for g in ("person", "vehicle"):
            t = tk[tk["class"] == g]
            if t.empty:
                continue
            if tf is None:
                kept.update(int(i) for i in t["track_id"])
                continue
            v = tf[(tf["group"] == g) & (tf["status"] == "valid")]
            ig = tf[((tf["status"] == "ignore") & (tf["group"] == g)) | tf["group"].isna()]
            vb, ib = mask_ltwh(v), mask_ltwh(ig)
            tb = t[["left", "top", "width", "height"]].to_numpy(float)
            k = keep_not_ignored(tb, vb, ib)
            t, tb = t[k], tb[k]
            kept.update(int(i) for i in t["track_id"])
            real.update(int(t["track_id"].iloc[b]) for _, b, _ in match_pairs(vb, tb))
            if len(v):
                foot = v[["foot_x", "foot_y"]].to_numpy(float)
                for idx, r in t.iterrows():
                    tid = int(r["track_id"])
                    if tid in real or tid in near or not np.isfinite(r["ground_n_m"]):
                        continue
                    cv = covs[row_of[idx]]
                    if not np.all(np.isfinite(cv)) or np.linalg.det(cv) <= 0:
                        continue
                    dd = foot - [r["ground_n_m"], r["ground_e_m"]]
                    m2 = np.einsum("ij,jk,ik->i", dd, np.linalg.inv(cv), dd)
                    if np.any(m2 <= gate):
                        near.add(tid)
    label = {}
    for i in kept:
        label[str(i)] = "real" if i in real else "near" if i in near else "false alarm"
    return label


OUT = REPO_ROOT / "cache" / "maps"
VIDEO_WIDTH = 960
SYNDRONE_VIDEO_STEP = 2
SYNDRONE_GROUND_STEP = 10
READ_BATCH = 100  # frames per read (compressed, about 50 MB)
SYNDRONE_ORIGIN = (0.0, 0.0, 0.0)  # display only: SynDrone has no geographic position
CAL = "results/syndrone/calibration/interval_calibration_v2_dfine.json"


class VideoWriter:
    """Frames piped to ffmpeg as raw BGR, written as H.264 (yuv420p, fast start) at a width of
    VIDEO_WIDTH, so the browser can play and seek it."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise SystemExit("ffmpeg not found on PATH (needed for the video)")
        self.size = (VIDEO_WIDTH, round(height * VIDEO_WIDTH / width / 2) * 2)
        self.p = subprocess.Popen(
            [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{fps:g}",
             "-i", "-", "-c:v", "libx264", "-crf", "28", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    def write(self, img: np.ndarray) -> None:
        self.p.stdin.write(cv2.resize(img, self.size, interpolation=cv2.INTER_AREA).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise SystemExit("ffmpeg failed")


def boxes_by_frame(tracks: pd.DataFrame, frames: set[int]) -> dict[str, list]:
    """The viewer's video boxes: frame -> [track, left, top, width, height, class, flag]."""
    out: dict[str, list] = {}
    t = tracks[tracks["frame"].isin(frames)]
    box = t[["left", "top", "width", "height"]].to_numpy(float).round(1)
    for f, tid, b, c, fl in zip(t["frame"], t["track_id"], box, t["class"], t["flag"], strict=True):
        out.setdefault(str(int(f)), []).append([int(tid), *b.tolist(), str(c), str(fl)])
    return out


def lonlat(n: np.ndarray, e: np.ndarray, origin: tuple[float, float, float]) -> np.ndarray:
    lat, lon, _ = ned_to_geodetic(np.c_[n, e, np.zeros(len(n))], *origin)
    return np.c_[np.round(lon, 7), np.round(lat, 7)]


def auair(args: argparse.Namespace, cfg) -> tuple:  # noqa: ANN001
    clip = REPO_ROOT / "cache" / "auair" / "flights" / f"clip_{args.clip:02d}"
    meta = pd.read_csv(clip / "metadata.csv")
    frames = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv")
    first = frames[frames["clip"] == args.clip].sort_values("t_ms").iloc[0]
    if abs(first["n_m"]) > 1e-9 or abs(first["e_m"]) > 1e-9:
        raise SystemExit("the clip's first fix is not its local origin")
    origin = (float(first["lat_deg"]), float(first["lon_deg"]), float(first["alt_m"]))
    src = pd.read_csv(REPO_ROOT / args.source / "tracks.csv")
    tracks = replay(src, meta.set_index("frame"), RangeSpeedStage.from_config(cfg.require))
    c = cfg.require("camera")
    first_img = cv2.imread(str(clip / "frames" / "000001.jpg"))
    cam = Camera(c["focal_px"], c["cx"], c["cy"], first_img.shape[1], first_img.shape[0])
    fl = meta.set_index("frame")
    # the clip's frames are unevenly spaced: play the video at their average rate (real time)
    fps = (len(meta) - 1) / float(meta["t_s"].iloc[-1] - meta["t_s"].iloc[0])

    def read(f: int) -> np.ndarray:
        return cv2.imread(str(clip / "frames" / f"{f:06d}.jpg"))

    vframes = list(fl.index)
    gframes = set(vframes)
    model = ranging_model(cfg.require)
    title = f"AU-AIR clip {args.clip}"
    return (tracks, meta, origin, cam, read, vframes, gframes, fps, None, model, title, None)


def syndrone(args: argparse.Namespace, cfg) -> tuple:  # noqa: ANN001
    from aerialrange.data.syndrone import FPS, SynDroneSequence, decode_rgb, syndrone_camera
    from aerialrange.data.syndrone_archive import MemberStore
    from aerialrange.evaluate.syndrone_eval import TruthRules, classify_truth

    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text("utf-8"))["syndrone"][
        "splits"
    ]
    if args.town in sp["test"]:
        raise SystemExit(f"{args.town} is a test town")
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    sc = syndrone_camera()
    camd = {"focal_px": sc.fx, "cx": sc.cx, "cy": sc.cy}
    ov = {"camera": camd, **exact_pose_overrides(cfg.require("kalman"))}
    ov["intervals"] = config_from_calibration(REPO_ROOT / CAL)

    def get(k: str):  # noqa: ANN202
        if k == "intervals":
            return ov[k]
        base = cfg.require(k)
        return {**base, **ov.get(k, {})} if isinstance(base, dict) else ov.get(k, base)

    meta = flight_meta(store, args.town, args.height)
    src = pd.read_csv(
        REPO_ROOT
        / "cache"
        / "syndrone"
        / "runs"
        / args.source
        / f"{args.town}_h{args.height}"
        / "tracks.csv"
    )
    tracks = replay(src, meta, RangeSpeedStage.from_config(get))
    seq = SynDroneSequence(store, args.town, args.height)
    n = len(meta)
    vframes = [k + 1 for k in range(0, n, SYNDRONE_VIDEO_STEP)]
    gframes = {k + 1 for k in range(0, n, SYNDRONE_GROUND_STEP)}

    def read_all(frames: list[int]) -> Iterator[tuple[int, np.ndarray]]:
        """Frames in the order given. The store yields members in archive order, not in the
        order asked for, so each batch is read whole (compressed bytes) and put in order."""
        for i in range(0, len(frames), READ_BATCH):
            batch = frames[i : i + READ_BATCH]
            names = {seq.member("rgb", f - 1): f for f in batch}
            got = {names[n]: raw for n, raw in store.iter_read(list(names))}
            for f in batch:
                yield f, decode_rgb(got[f])

    # findable truth (the evaluation's valid boxes), in the video frames
    cfgs = load_config("syndrone")
    rules = TruthRules(
        float(cfgs.require("truth_min_visibility")), int(cfgs.require("truth_min_instance_px"))
    )
    tr = pd.read_csv(REPO_ROOT / "cache" / "syndrone" / "truth" / args.town / "objects.csv.gz")
    tr = classify_truth(tr[tr["height"] == args.height], rules)
    tr = tr[tr["status"] != "absent"]
    matched = truth_match(tracks, tr, meta)
    tr = tr[(tr["status"] == "valid") & tr["group"].notna() & (tr["frame"] + 1).isin(vframes)]
    ll = lonlat(tr["foot_x"].to_numpy(float), tr["foot_y"].to_numpy(float), SYNDRONE_ORIGIN)
    truth: dict[str, list] = {}
    for (f, g), (lo, la) in zip(zip(tr["frame"] + 1, tr["group"], strict=True), ll, strict=True):
        truth.setdefault(str(int(f)), []).append([float(lo), float(la), str(g)])
    first_img = next(read_all([1]))[1]
    cam = Camera(sc.fx, sc.cx, sc.cy, first_img.shape[1], first_img.shape[0])
    meta = meta.reset_index()
    return (tracks, meta, SYNDRONE_ORIGIN, cam, read_all, vframes, gframes,
            FPS / SYNDRONE_VIDEO_STEP, (truth, matched), ranging_model(get),
            f"SynDrone {args.town}, {args.height} m (simulated)",
            "Simulated town, placed at latitude 0, longitude 0 for display")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    a = sub.add_parser("auair")
    a.add_argument("--clip", type=int, required=True)
    a.add_argument("--source", type=Path, required=True)
    s = sub.add_parser("syndrone")
    s.add_argument("--town", required=True)
    s.add_argument("--height", type=int, required=True)
    s.add_argument("--source", required=True)
    for p in (a, s):
        p.add_argument("--config", default="run")
        p.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.kind == "auair":
        res = auair(args, cfg)
        name = args.name or f"auair_clip{args.clip:02d}"
    else:
        res = syndrone(args, cfg)
        name = args.name or f"syndrone_{args.town}_h{args.height}"
    tracks, meta, origin, cam, reader, vframes, gframes, fps, truth, model, title, note = res
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(out / "tracks.csv", index=False)
    gj = build_geojson(tracks, meta, origin, model, cam.width_px, cam.height_px, name=name)
    fl = meta.set_index("frame")
    vw = VideoWriter(out / "video.mp4", fps, cam.width_px, cam.height_px)

    def frames_for_ground() -> Iterator[tuple[int, np.ndarray]]:
        """Every video frame is written to the video as it is read; the ground image takes the
        subset it needs, so each frame is read once."""
        it = reader(vframes) if args.kind == "syndrone" else ((f, reader(f)) for f in vframes)
        for f, img in it:
            vw.write(img)
            if f in gframes:
                yield f, img

    ground = build_ground_image(
        frames_for_ground(), fl.loc[sorted(gframes)], cam, min_reliable_depression_deg(model)
    )
    vw.close()
    extras = {
        "title": title,
        "boxes": boxes_by_frame(tracks, set(vframes)),
        "video": {"frames": vframes, "fps": fps, "width": cam.width_px, "height": cam.height_px},
    }
    if truth is not None:
        extras["truth"], extras["truth_match"] = truth
    if note:
        extras["note"] = note
    write_search_map(out, gj, extras, ground, origin, out / "video.mp4")
    objs = [f["properties"] for f in gj["features"] if f["properties"]["kind"] == "object"]
    people = [o for o in objs if o["class"] == "person"]
    print(
        f"{title}: {len(objs)} objects ({len(people)} people, "
        f"{sum(o['reliable_share'] > 0 for o in people)} with a reliable range); "
        f"ground image {ground.image.shape[1]} x {ground.image.shape[0]} at {ground.res_m:.2f} m "
        f"from {ground.frames_used} frames; searched {gj['drone_ranger']['searched_km2']:.3f} km2"
        if ground
        else f"{title}: no ground image"
    )
    print("wrote", out / "map.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
