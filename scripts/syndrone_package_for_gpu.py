"""Package SynDrone development frames for fine-tuning the detector on a cloud GPU (6.5 b).

Usage: python scripts/syndrone_package_for_gpu.py [--stride 5]

Development towns only (results/splits.json). Every `stride`-th frame of each camera
(consecutive 25 Hz frames are near-duplicates). Labels in YOLO format with the version 1
VisDrone class list, so the fine-tuned model is a drop-in for the version 1 detector:
person -> pedestrian, car -> car, truck -> truck, bus -> bus, from the instance-mask boxes of
valid truth (D-704, D-705). Ignored objects (riders, two-wheelers, trams, map boxes, and
person or vehicle views below the visibility rule) are filled with the image's mean colour,
so the detector is not taught that real objects are background (D-707); pixels
inside a labelled box are never filled (D-711).

Folds (leave one town out, D-707): for town T held out, training uses the other two
development towns; model selection (early stopping) uses the last 20 % of frames of each
training town, a contiguous block, so selection frames are separated in time from training
frames; T is never seen and is scored afterwards with scripts/syndrone_run.py.

Writes cache/gpu_package/syndrone_dev_yolo.zip (images/<town>/..., labels/<town>/...,
fold list files and a YAML per fold) and manifest.json with counts and the zip's SHA-256.
Uploaded by hand as a PRIVATE Kaggle dataset; nothing here uploads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd

from aerialrange.data.syndrone import HEIGHT, N_FRAMES, WIDTH, SynDroneSequence
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.data.visdrone import CATEGORIES, TRAIN_CLASSES
from aerialrange.evaluate.syndrone_eval import TruthRules, classify_truth
from aerialrange.utils.config import REPO_ROOT, load_config

TRUTH = REPO_ROOT / "cache" / "syndrone" / "truth"
OUT = REPO_ROOT / "cache" / "gpu_package"
VD_CLASS = {40: "pedestrian", 100: "car", 101: "truck", 102: "bus"}
SELECT_SHARE = 0.2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args(argv)
    towns = json.loads((REPO_ROOT / "results" / "splits.json").read_text("utf-8"))["syndrone"][
        "splits"
    ]["development"]
    cfg = load_config("syndrone")
    rules = TruthRules(
        float(cfg.require("truth_min_visibility")), int(cfg.require("truth_min_instance_px"))
    )
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    names = [CATEGORIES[c] for c in TRAIN_CLASSES]
    cls_index = {t: names.index(n) for t, n in VD_CLASS.items()}
    frames = range(0, N_FRAMES, args.stride)
    cut = int(N_FRAMES * (1 - SELECT_SHARE))
    OUT.mkdir(parents=True, exist_ok=True)
    zpath = OUT / "syndrone_dev_yolo.zip"
    lists: dict[str, dict[str, list[str]]] = {t: {"train": [], "select": []} for t in towns}
    counts: dict = {}
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for town in towns:
            p = TRUTH / town / "objects.csv.gz"
            if not p.exists():
                print(f"{town}: no truth yet (scripts/syndrone_z0_check.py)")
                return 1
            tr = classify_truth(pd.read_csv(p), rules)
            tr = tr[(tr["status"] != "absent") & tr["frame"].isin(list(frames))]
            n_lab = n_ign = n_spared = 0
            for h in (20, 50, 80):
                seq = SynDroneSequence(store, town, h)
                g = {f: x for f, x in tr[tr["height"] == h].groupby("frame")}
                for k in frames:
                    img = seq.rgb(k)
                    rows = g.get(k, tr.iloc[0:0])
                    lines = []
                    mean = img.reshape(-1, 3).mean(0)
                    fill = np.zeros(img.shape[:2], bool)  # ignored objects to blank out
                    keep = np.zeros(img.shape[:2], bool)  # labelled objects, never blanked
                    for _, r in rows.iterrows():
                        x0, y0, x1, y1 = (
                            r["mask_left"],
                            r["mask_top"],
                            r["mask_right"],
                            r["mask_bottom"],
                        )
                        if not np.isfinite(x0):
                            continue
                        sl = (slice(int(y0), int(y1) + 1), slice(int(x0), int(x1) + 1))
                        if r["status"] == "valid" and int(r["tag"]) in cls_index:
                            cx, cy = (x0 + x1) / 2 / WIDTH, (y0 + y1) / 2 / HEIGHT
                            w, hh = (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT
                            lines.append(
                                f"{cls_index[int(r['tag'])]} {cx:.6f} {cy:.6f} {w:.6f} {hh:.6f}"
                            )
                            keep[sl] = True
                            n_lab += 1
                        else:
                            fill[sl] = True
                            n_ign += 1
                    # A map box can sit exactly on a labelled actor (D-704); blanking it would
                    # erase the labelled object and teach the detector that a grey patch is a
                    # car. Labelled pixels are therefore never blanked (D-711).
                    img[fill & ~keep] = mean
                    n_spared += int((fill & keep).any())
                    stem = f"{town}/h{h}_{k:05d}"
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    z.writestr(f"syndrone_dev_yolo/images/{stem}.jpg", buf.tobytes())
                    z.writestr(f"syndrone_dev_yolo/labels/{stem}.txt", "\n".join(lines))
                    part = "train" if k < cut else "select"
                    lists[town][part].append(
                        f"./images/{stem}.jpg"
                    )  # "./": relative to the list file
            counts[town] = {
                "images": 3 * len(frames),
                "labels": n_lab,
                "ignored_filled": n_ign,
                "images_with_fill_overlapping_labels": n_spared,
            }
            print(town, counts[town], flush=True)
        for held in towns:
            train = [x for t in towns if t != held for x in lists[t]["train"]]
            select = [x for t in towns if t != held for x in lists[t]["select"]]
            z.writestr(f"syndrone_dev_yolo/fold_{held}_train.txt", "\n".join(train))
            z.writestr(f"syndrone_dev_yolo/fold_{held}_select.txt", "\n".join(select))
            z.writestr(
                f"syndrone_dev_yolo/fold_{held}.yaml",
                "# path is set at run time by the training notebook\n"
                f"train: fold_{held}_train.txt\nval: fold_{held}_select.txt\n"
                f"names: {json.dumps(names)}\n",
            )
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    manifest = {
        "towns": towns,
        "stride": args.stride,
        "select_share": SELECT_SHARE,
        "counts": counts,
        "zip": zpath.name,
        "sha256": sha,
        "folds": {
            t: {
                "train": len([x for u in towns if u != t for x in lists[u]["train"]]),
                "select": len([x for u in towns if u != t for x in lists[u]["select"]]),
            }
            for t in towns
        },
    }
    (OUT / "syndrone_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
