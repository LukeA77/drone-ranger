"""Write `results/env.json` describing the compute environment ({{COMPUTE_PROFILE}})."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aerialrange.utils.env_report import env_report

OUT = Path(__file__).resolve().parents[1] / "results" / "env.json"


def main() -> int:
    report = env_report()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
