"""Export a trained D-FINE candidate to ONNX and check parity with the authors' code.

Usage (in the D-FINE environment, cache/venvs/dfine, with the authors' repository at the
commit the run used, D-717):
  cache/venvs/dfine/Scripts/python.exe scripts/export_dfine.py --run dfine_n_640
         --repo cache/venvs/src/D-FINE [--n-frames 50]

Reads cache/train/<run>/ (the Kaggle output, gpu/train_dfine_kaggle.py): checkpoint_best.pth,
<run>.yml and run.json. Builds the model from the run's config (pointed at the local
repository), loads the EMA weights the authors' code evaluates, and exports the network
without its post-processor (outputs `boxes`, `logits`; fixed input, batch 1) to
weights/<run>.onnx, so decoding runs in the project's code (`DFINEOnnx`). Parity: on the same
seeded VisDrone MOT val frames as the other exports, the authors' validation transform,
network and post-processor (PyTorch) against the project's ONNX detector; largest box and
score differences of matched detections and the share matched both ways (IoU >= 0.5, same
class). Writes results/edge/parity_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aerialrange.detect.onnx_detectors import DFINEOnnx  # noqa: E402
from aerialrange.utils.config import load_config  # noqa: E402
from export_rfdetr import compare  # noqa: E402

PREP = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"


def local_config(run_dir: Path, run: str, repo: Path) -> Path:
    """The run's config with its include pointed at the local repository."""
    text = (run_dir / f"{run}.yml").read_text()
    inc = re.search(r"__include__: \['([^']+)'\]", text)
    if inc is None:
        raise ValueError("run config has no __include__")
    rel = inc.group(1)[inc.group(1).index("configs/") :]
    text = text.replace(inc.group(1), (repo / rel).as_posix())
    dest = run_dir / f"{run}_local.yml"
    dest.write_text(text)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--score-floor", type=float, default=None, help="default: run config's")
    args = ap.parse_args(argv)
    args.repo = args.repo.resolve()
    run_dir = REPO_ROOT / "cache" / "train" / args.run
    info = json.loads((run_dir / "run.json").read_text())
    head = subprocess.run(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if head != info["commit"]:
        print(f"repository at {head}, run used {info['commit']}: check out that commit")
        return 1
    sys.path.insert(0, str(args.repo))
    import torch
    import torch.nn as nn
    from PIL import Image
    from src.core import YAMLConfig  # the authors' package, from --repo
    from src.data.transforms import ConvertPILImage, Resize

    cfg = YAMLConfig(str(local_config(run_dir, args.run, args.repo)))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(run_dir / "checkpoint_best.pth", map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    net = cfg.model.deploy().eval()
    post = cfg.postprocessor.eval()
    res = int(info["input_size"])

    class Raw(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net

        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.net(images)
            return out["pred_boxes"], out["pred_logits"]

    dest = REPO_ROOT / "weights" / f"{args.run}.onnx"
    tmp = run_dir / "export.onnx"
    torch.onnx.export(
        Raw().eval(),
        (torch.rand(1, 3, res, res),),
        str(tmp),
        input_names=["images"],
        output_names=["boxes", "logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    shutil.move(tmp, dest)

    floor = args.score_floor
    if floor is None:
        floor = float(load_config("run_accurate").require("detector")["score_floor"])
    onnx_det = DFINEOnnx(dest, score_floor=floor)
    # The authors' validation ops (configs/dfine/include/dataloader.yml), their own classes.
    resize, to_tensor = Resize(size=[res, res]), ConvertPILImage(dtype="float32", scale=True)
    frames = []
    for seq in sorted(p for p in PREP.iterdir() if p.is_dir()):
        meta = json.loads((seq / "seqinfo.json").read_text())
        frames += sorted((REPO_ROOT / meta["frame_dir"]).glob("*.jpg"))
    rng = np.random.default_rng(load_config("project").require("seed"))
    pick = sorted(rng.choice(len(frames), size=min(args.n_frames, len(frames)), replace=False))
    rows = []
    for k in pick:
        img = cv2.imread(str(frames[k]))
        h, w = img.shape[:2]
        x = to_tensor(resize(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        with torch.no_grad():
            r = post(net(x[None]), torch.tensor([[w, h]]))[0]  # sizes as (w, h), as the authors'
        keep = (r["scores"] > floor) & (r["labels"] > 0)
        a = (
            r["boxes"][keep].numpy().astype(float),
            r["scores"][keep].numpy().astype(float),
            r["labels"][keep].numpy().astype(int) - 1,
        )
        d = onnx_det(img)
        rows.append(compare(a, (d.xyxy, d.score, d.cls)))
    diffs = sum((r["box_diffs"] for r in rows), [])
    tot_a = sum(r["n_a"] for r in rows)
    tot_b = sum(r["n_b"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "commit": head,
        "onnx": dest.name,
        "input_hw": [onnx_det.h, onnx_det.w],
        "score_floor": floor,
        "n_frames": len(pick),
        "torch": torch.__version__,
        "onnxruntime": version("onnxruntime"),
        "reference": "authors' validation transform, network and post-processor (PyTorch)",
        "max_box_diff_px": max(r["max_box_diff_px"] for r in rows),
        "p99_box_diff_px": float(np.percentile(diffs, 99)) if diffs else 0.0,
        "max_score_diff": max(r["max_score_diff"] for r in rows),
        "agreement_library_side": matched / max(tot_a, 1),
        "agreement_onnx_side": matched / max(tot_b, 1),
        "detections_library": tot_a,
        "detections_onnx": tot_b,
    }
    path = REPO_ROOT / "results" / "edge" / f"parity_{args.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
