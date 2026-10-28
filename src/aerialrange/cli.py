"""Command-line entry point `aerialrange`. Sub-commands mirror the Makefile targets."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Annotated

import typer

from aerialrange.utils.config import REPO_ROOT

app = typer.Typer(help="Aerial tracking and ranging.", no_args_is_help=True)


def _run_script(name: str) -> None:
    runpy.run_path(str(REPO_ROOT / "scripts" / name), run_name="__main__")


@app.command("data-check")
def data_check() -> None:
    """Check dataset presence and schemas."""
    _run_script("check_data.py")


@app.command("env")
def env() -> None:
    """Write results/env.json."""
    _run_script("env_report.py")


@app.command("run")
def run_cmd(
    source: Annotated[Path, typer.Option(help="Video file or folder of frames")],
    metadata: Annotated[Path, typer.Option(help="Flight-data CSV, one row per frame")],
    config: Annotated[str, typer.Option(help="Config name in configs/")] = "run",
    out: Annotated[Path, typer.Option(help="Output folder")] = Path("cache/run"),
    boxes: Annotated[Path | None, typer.Option(help="Boxes CSV instead of a model")] = None,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames")] = None,
) -> None:
    """Detect, track, range and annotate one flight: tracks.csv, annotated.mp4, timing.json."""
    from aerialrange.pipeline.run import run

    timing = run(source, metadata, out, config=config, boxes_csv=boxes, max_frames=max_frames)
    typer.echo(json.dumps(timing["stages"]["end_to_end"], indent=1))
    typer.echo(f"wrote {out}")


@app.command("demo")
def demo_cmd(
    out: Annotated[Path, typer.Option(help="Output folder")] = Path("cache/demo"),
) -> None:
    """Run the whole pipeline on a synthetic flight (no datasets) and check it against truth."""
    from aerialrange.pipeline.demo import demo

    typer.echo(json.dumps(demo(out), indent=1))
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
