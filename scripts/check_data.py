"""Check that the datasets are present and match their expected schemas.

Usage: python scripts/check_data.py [--data-root PATH]

Prints a report and writes `results/data_check.json`. Exits 1 if anything is missing or
fails a schema check. Never downloads, extracts or modifies data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from aerialrange.data.check import check_all
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "results" / "data_check.json"
# Modalities the Mid-Air evaluation reads (colour images are optional and unused).
MIDAIR_MODALITIES = ["depth", "segmentation", "normals"]


def _resolve(root: Path, sub: str | None) -> Path | None:
    return None if sub is None else (root / sub).resolve()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config("data")
    root = args.data_root or os.environ.get("AERIALRANGE_DATA_ROOT") or cfg.require("data_root")
    root = Path(root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    if not root.is_dir():
        print(f"data root not found: {root}")
        return 1

    results = check_all(
        root,
        visdrone_mot=_resolve(root, cfg.get("visdrone_mot_subdir")) or root,
        visdrone_det=_resolve(root, cfg.get("visdrone_det_subdir")) or root,
        midair=_resolve(root, cfg.get("midair_subdir")) or root / "midair",
        auair=_resolve(root, cfg.get("auair_subdir")),
        midair_climates=[cfg.get("midair_climate")] if cfg.get("midair_climate") else None,
        midair_modalities=MIDAIR_MODALITIES,
    )

    print(f"Data root: {root}\n")
    for r in results:
        print(f"[{r.status.upper():>12}] {r.name}")
        for k, v in r.details.items():
            print(f"{'':16}{k}: {v}")
        for p in r.problems:
            print(f"{'':16}! {p}")
    ok = all(r.status == "ok" for r in results)
    print(f"\n{sum(r.status == 'ok' for r in results)} of {len(results)} components ok.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data_root": str(root), "all_ok": ok, "results": [asdict(r) for r in results]}
    # Committed file: store paths relative to the repo so no local user paths leak.
    text = json.dumps(payload, indent=2).replace(json.dumps(str(REPO_ROOT))[1:-1], "<repo>")
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
