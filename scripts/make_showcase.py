"""Showcase clips with no words (2026-10-23, D-618): footage with the system's output
drawn on it, and nothing else (no titles, captions, legends or credits on screen; I write the
story and the dataset credits in the post).

Usage: python scripts/make_showcase.py [--only 1 3]

Writes to cache/video/showcase/ (dataset imagery, not committed, D-323):
  01_tracking.mp4        fine-tuned detector + tuned BoT-SORT on four VisDrone test-dev clips
                         (saved tracks of the one-time test run);
                         a box and identity number per track, one colour per identity
  02_before_after.mp4    two validation clips side by side: pretrained (left), fine-tuned (right);
                         validation, because the pretrained baseline was never run on test-dev
  03_ranging_oblique.mp4 AU-AIR held-out clip 20 (frames 40-158): range +/- sigma per vehicle,
                         coloured by reliability (green reliable, amber uncertain), ground
                         rings 10/25/50/100 m
  04_ranging_overhead.mp4 AU-AIR held-out clip 44, straight down, same overlay
  05_range_flag.mp4      Mid-Air development flight 0026: range samples stuck to the ground
                         (projected with the drone's pose in every frame); green flag quiet,
                         amber flag raised; circle within 10 % of truth, cross more than 10 % off
Inputs: saved validation tracks (cache/visdrone/runs), the accurate AU-AIR runs
(cache/runs/auair_clip20_acc, auair_clip44_acc) and the Mid-Air caches (cache/midair).
Each clip stays within `{{MAX_VIDEO_MB}}`; results/video/showcase.json records them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import cv2
import numpy as np
import pandas as pd

from aerialrange.data.midair import WATER_CLASS, MidAirTrajectory, midair_camera
from aerialrange.evaluate.conformal import apply_conformal
from aerialrange.evaluate.flag import FlagModel
from aerialrange.geometry.camera import project_world_point
from aerialrange.pipeline.overlay import STATUS, draw_rings
from aerialrange.pipeline.ranging import CameraPose
from aerialrange.utils.config import REPO_ROOT, load_config
from aerialrange.viz import video as v

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from terrain_analyse import FIT, dep_band, flag_features, methods  # noqa: E402

TRAJ = "trajectory_0026"
# Track identity colours (fixed order; every box also carries its id as text, so colour never
# stands alone) and the reliability status colours of the pipeline overlay.
ID_COLOURS = [
    v.hex_bgr(h)
    for h in ("#2a78d6", "#eb6834", "#1baf7a", "#9b59d0", "#e0447c", "#3fb8c4", "#c9a227")
]
GOOD, WARN = STATUS["reliable"], STATUS["uncertain"]
DARK, WHITE = (24, 24, 24), (250, 250, 250)


OUT = REPO_ROOT / "cache/video/showcase"
HD = (1920, 1080)
FPS = 25
VIS_CLIPS = [  # (sequence, first frame, last frame), validation split
    ("uav0000305_00000_v", 1, 184),
    ("uav0000137_00458_v", 1, 233),
    ("uav0000086_00000_v", 1, 250),
    ("uav0000117_02622_v", 1, 250),
]
VIS_RUNS = ("B0_yolo26n_640_botsort_cmc", "B2_yolo26n_960_tuned")
# Clip 01 uses the saved tracks of the one-time test-dev run (frozen settings; nothing is re-run
# on test data). Chosen for clarity, not typicality: the README quotes the test-set scores.
TEST_RUN = "B2_yolo26n_960_tuned_botsort_cmc"
TEST_CLIPS = [
    ("uav0000297_02761_v", 1, 200),
    ("uav0000073_00600_v", 1, 200),
    ("uav0000120_04775_v", 1, 200),
    ("uav0000077_00720_v", 1, 200),
]
AUAIR = {
    3: (20, "cache/runs/auair_clip20_acc", 40, 158),  # from 40: skips the take-off
    4: (44, "cache/runs/auair_clip44_acc", 1, 65),
}
MIDAIR_SECONDS, MIDAIR_POINTS, HALF_WINDOW = 30, 40, 12
BG = (17, 17, 17)


def midair_scores() -> dict[str, np.ndarray]:
    """Terrain-from-motion estimates of one development flight with the frozen calibration
    and flag (fitted on the calibration flights; the same code path as terrain_analyse)."""
    fit = json.loads(FIT.read_text())
    model = fit["conformal"]
    model["global"] = {float(k): float(x) for k, x in model["global"].items()}
    model["groups"] = {
        int(k): {(float(kk) if kk != "n" else kk): vv for kk, vv in g.items()}
        for k, g in model["groups"].items()
    }
    flag = FlagModel.from_dict(fit["flag_model"])
    cache = REPO_ROOT / "cache/midair"
    r = dict(np.load(cache / f"{TRAJ}.npz"))
    r.update(np.load(cache / f"{TRAJ}.terrain.npz"))
    m = methods(r)["terrain_motion"]
    ok = (r["cls"] != WATER_CLASS) & m["valid"] & np.isfinite(m["sigma"]) & (m["sigma"] > 0)
    s68 = np.full(len(ok), np.nan)
    s68[ok] = apply_conformal(m["sigma"][ok], model, 0.68, groups=dep_band(r)[ok])
    X = flag_features(r, s68)
    fin = ok & np.all(np.isfinite(X), axis=1)
    score = np.full(len(ok), np.nan)
    score[fin] = flag.score(X[fin])
    return {
        "frame": r["frame"],
        "u": r["u"],
        "v": r["v"],
        "ok": fin,
        "true": r["true_range_m"],
        "est": m["range"],
        "bad": np.abs(m["rel"]) > fit["err_threshold_frac"],
        "flag": score >= fit["flag_threshold"]["threshold"],
        "thr": fit["err_threshold_frac"],
    }


def vis_tracks(run: str, seq: str, split: str = "val") -> dict[int, list]:
    out: dict[int, list] = {}
    for g in ("person", "vehicle"):
        f = REPO_ROOT / "cache/visdrone/runs" / run / split / f"{seq}_{g}.txt"
        if f.stat().st_size == 0:
            continue
        for fr, i, x, y, w, h, _ in np.loadtxt(f, delimiter=",").reshape(-1, 7):
            out.setdefault(int(fr), []).append((int(i), x, y, w, h))
    return out


def draw_tracks(img: np.ndarray, rows: list, s: float, thick: int, font_scale: float) -> None:
    for i, x, y, w, h in rows:
        c = ID_COLOURS[i % len(ID_COLOURS)]
        p1, p2 = (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s))
        cv2.rectangle(img, p1, p2, c, thick, cv2.LINE_AA)
        lab = str(i)
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), DARK, -1)
        cv2.putText(
            img,
            lab,
            (p1[0] + 3, p1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            WHITE,
            1,
            cv2.LINE_AA,
        )


def frame_paths(seq: str, split: str = "val") -> list:
    info = json.loads((REPO_ROOT / "cache/visdrone/mot" / split / seq / "seqinfo.json").read_text())
    return sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))


def clip_tracking(wr: v.Writer) -> dict:
    for n, (seq, a, b) in enumerate(TEST_CLIPS):
        tr, paths = vis_tracks(TEST_RUN, seq, "test-dev"), frame_paths(seq, "test-dev")
        for f in range(a, b + 1):
            src = cv2.imread(str(paths[f - 1]))
            s = HD[0] / src.shape[1]
            img = cv2.resize(src, HD, interpolation=cv2.INTER_AREA)
            draw_tracks(img, tr.get(f, []), s, 3, 0.7)
            if f == a and n:
                wr.crossfade(img, 0.5)
            wr.write(img)
    return {"run": TEST_RUN, "split": "test-dev", "clips": TEST_CLIPS}


def clip_before_after(wr: v.Writer) -> dict:
    pw = (HD[0] - 3 * 12) // 2
    for n, (seq, a, b) in enumerate(VIS_CLIPS[:2]):
        tr = [vis_tracks(r, seq) for r in VIS_RUNS]
        paths = frame_paths(seq)
        for f in range(a, b + 1):
            src = cv2.imread(str(paths[f - 1]))
            ph = round(pw * src.shape[0] / src.shape[1])
            img = np.empty((HD[1], HD[0], 3), np.uint8)
            img[:] = BG
            y0 = (HD[1] - ph) // 2
            for k in range(2):
                pane = cv2.resize(src, (pw, ph), interpolation=cv2.INTER_AREA)
                draw_tracks(pane, tr[k].get(f, []), pw / src.shape[1], 2, 0.5)
                v.paste(img, pane, 12 + k * (pw + 12), y0)
            if f == a and n:
                wr.crossfade(img, 0.5)
            wr.write(img)
    return {"left": VIS_RUNS[0], "right": VIS_RUNS[1], "clips": VIS_CLIPS[:2]}


def clip_ranging(wr: v.Writer, key: int) -> dict:
    clip, run, a, b = AUAIR[key]
    cam = load_config("run").require("camera")
    rings = load_config("run").require("overlay")["rings_m"]
    flight = REPO_ROOT / "cache/auair/flights" / f"clip_{clip:02d}"
    meta = pd.read_csv(flight / "metadata.csv").set_index("frame")
    tracks = pd.read_csv(REPO_ROOT / run / "tracks.csv")
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    frames = sorted((flight / "frames").iterdir())
    for k in range(a, b + 1):
        img = cv2.imread(str(frames[k - 1]))
        m = meta.loc[k]
        pose = CameraPose(*(float(m[c]) for c in ("n_m", "e_m", "h_m", "yaw_rad", "tilt_deg")))
        draw_rings(img, pose, float(cam["focal_px"]), float(cam["cx"]), float(cam["cy"]), rings)
        labels, placed = [], []
        g = by_frame.get(k)
        if g is not None:
            for _, r in g.sort_values("height", ascending=False).iterrows():
                col = STATUS[r["flag"]]
                x1, y1 = int(r["left"]), int(r["top"])
                x2, y2 = int(r["left"] + r["width"]), int(r["top"] + r["height"])
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 4, cv2.LINE_AA)
                if r["flag"] == "invalid":
                    continue
                lab = f"{r['range_m']:.0f} ± {r['range_sigma_m']:.0f} m"
                tw = v.font(28, True).getlength(lab)
                lx, ly = int(min(max(6, x1), img.shape[1] - tw - 10)), max(34, y1 - 8)
                rect = (lx - 6, ly - 34, lx + tw + 12, ly + 4)
                if any(
                    not (rect[2] < q[0] or q[2] < rect[0] or rect[3] < q[1] or q[3] < rect[1])
                    for q in placed
                ):
                    continue
                placed.append(rect)
                labels.append((lab, (lx, ly), 28, WHITE, True))
        v.text(img, labels, box=DARK, anchor="ld")
        if img.shape[:2] != (HD[1], HD[0]):
            img = cv2.resize(img, HD, interpolation=cv2.INTER_AREA)
        wr.write(img, FPS // 5)  # 5 fps source, real time
    return {"clip": clip, "run": run, "frames": [a, b]}


def clip_flag(wr: v.Writer) -> dict:
    """Range samples of each one-per-second analysis frame, projected into the 12 frames
    either side with the drone's pose, so they stay on the ground as the video plays."""
    data = load_config("data")
    root = (
        REPO_ROOT
        / data.require("data_root")
        / data.require("midair_subdir")
        / data.require("midair_climate")
    )
    d = midair_scores()
    traj = MidAirTrajectory(root, TRAJ)
    rng = np.random.default_rng(load_config("project").require("seed"))
    keys = np.unique(d["frame"])
    n_keys = MIDAIR_SECONDS
    keys = keys[(len(keys) - n_keys) // 2 :][:n_keys]
    pts = {}
    for f in keys:
        j = np.flatnonzero(d["ok"] & (d["frame"] == f))
        pts[int(f)] = rng.choice(j, min(MIDAIR_POINTS, len(j)), replace=False)
    r = dict(np.load(REPO_ROOT / f"cache/midair/{TRAJ}.npz"))
    first, last = int(keys[0]) - HALF_WINDOW, int(keys[-1]) + HALF_WINDOW
    cam = None
    side = HD[1]
    for k in range(first, last + 1):
        src = cv2.imread(str(traj.frame_path("color_left", k)))
        if cam is None:
            cam = midair_camera(src.shape[1], src.shape[0])
        s = side / src.shape[0]
        img = cv2.resize(src, (side, side), interpolation=cv2.INTER_CUBIC)
        key = int(keys[np.argmin(np.abs(keys - k))])
        if abs(key - k) <= HALF_WINDOW:
            pose = traj.pose(k)
            for j in pts[key]:
                uv, front = project_world_point(cam, pose, r["true_point_w"][j])
                if not front:
                    continue
                c = (int(uv[0] * s), int(uv[1] * s))
                col = WARN if d["flag"][j] else GOOD
                if d["bad"][j]:
                    for lw, cc in ((9, DARK), (4, col)):
                        cv2.line(
                            img, (c[0] - 11, c[1] - 11), (c[0] + 11, c[1] + 11), cc, lw, cv2.LINE_AA
                        )
                        cv2.line(
                            img, (c[0] - 11, c[1] + 11), (c[0] + 11, c[1] - 11), cc, lw, cv2.LINE_AA
                        )
                else:
                    cv2.circle(img, c, 10, DARK, -1, cv2.LINE_AA)
                    cv2.circle(img, c, 7, col, -1, cv2.LINE_AA)
        wr.write(img)
    traj.close()
    return {
        "trajectory": "trajectory_0026",
        "split": "development",
        "frames": [first, last],
        "points_per_second": MIDAIR_POINTS,
    }


CLIPS = {
    1: ("01_tracking", HD, clip_tracking),
    2: ("02_before_after", HD, clip_before_after),
    3: ("03_ranging_oblique", HD, lambda wr: clip_ranging(wr, 3)),
    4: ("04_ranging_overhead", HD, lambda wr: clip_ranging(wr, 4)),
    5: ("05_range_flag", (HD[1], HD[1]), clip_flag),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=sorted(CLIPS))
    args = ap.parse_args(argv)
    max_mb = float(load_config("release").require("max_video_mb"))
    rec_path = REPO_ROOT / "results/video/showcase.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"clips": {}}
    for i in args.only:
        name, size, fn = CLIPS[i]
        master = OUT / f"{name}_master.mp4"
        wr = v.Writer(master, size, FPS)
        info = fn(wr)
        wr.close()
        out = OUT / f"{name}.mp4"
        mode = v.encode_quality(master, out, max_mb, wr.seconds)
        master.unlink()
        mb = out.stat().st_size / 1e6
        rec["clips"][name] = {
            "file": str(out.relative_to(REPO_ROOT)).replace("\\", "/"),
            "seconds": round(wr.seconds, 2),
            "size_mb": round(mb, 2),
            "size": list(size),
            "encoding": mode,
            "within_size_limit": mb <= max_mb,
            **info,
        }
        print(f"{name}: {wr.seconds:.1f} s, {mb:.1f} MB ({mode})", flush=True)
    rec["generated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    rec["max_video_mb"] = max_mb
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return 0 if all(c["within_size_limit"] for c in rec["clips"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
