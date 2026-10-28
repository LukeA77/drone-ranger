"""Fine-tune the detector on VisDrone DET (baseline B2).

Usage: python scripts/train_detector.py --name B2_yolo26n_640 --model yolo26n.pt --imgsz 640
       [--epochs N] [--batch N] [--fraction F] [--resume]

Starts from the library's defaults (`{{TRAIN_HYPERPARAMS}}`); every argument that differs
from a default is recorded with the run. Training is resumable (--resume continues the last
checkpoint of the named run). Uses cache/visdrone/yolo/visdrone.yaml from
scripts/visdrone_prep.py --det (DET images sharing a source video with MOT val/test-dev are
excluded, D-308). Model selection uses DET val only. Weights stay under cache/train/ and are
never committed; the run summary and curves are copied to results/visdrone/train/<name>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime

from aerialrange.utils.config import REPO_ROOT

DATA = REPO_ROOT / "cache" / "visdrone" / "yolo" / "visdrone.yaml"
RUNS = REPO_ROOT / "cache" / "train"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="pretrained weights, e.g. yolo26n.pt")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="default: library default")
    ap.add_argument("--batch", type=int, default=None, help="default: library default")
    ap.add_argument("--fraction", type=float, default=None, help="train on a subset (smoke)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if not DATA.exists():
        print("missing dataset YAML: run scripts/visdrone_prep.py --det")
        return 1
    import ultralytics
    from ultralytics import YOLO

    overrides = {
        k: v
        for k, v in (("epochs", args.epochs), ("batch", args.batch), ("fraction", args.fraction))
        if v is not None
    }
    t0 = time.time()
    if args.resume:
        model = YOLO(RUNS / args.name / "weights" / "last.pt")
        model.train(resume=True)
    else:
        model = YOLO(REPO_ROOT / "weights" / args.model)
        model.train(
            data=str(DATA),
            imgsz=args.imgsz,
            project=str(RUNS),
            name=args.name,
            exist_ok=True,
            device="cpu",
            seed=0,
            deterministic=True,
            **overrides,
        )
    run_dir = RUNS / args.name
    out = REPO_ROOT / "results" / "visdrone" / "train" / args.name
    out.mkdir(parents=True, exist_ok=True)
    for f in ("results.csv", "args.yaml", "results.png"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, out / f)
    (out / "run.json").write_text(
        json.dumps(
            {
                "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "wall_seconds_this_session": time.time() - t0,
                "ultralytics": ultralytics.__version__,
                "base_weights": args.model,
                "imgsz": args.imgsz,
                "overrides_vs_library_defaults": overrides,
                "resumed": args.resume,
                "device": "cpu",
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
