from typer.testing import CliRunner

from aerialrange.cli import app
from aerialrange.utils.env_report import env_report


def test_env_report_has_required_fields():
    r = env_report()
    for key in ("cpu_model", "logical_cores", "memory_gb", "gpus", "os", "python", "packages"):
        assert key in r
    assert r["packages"]["numpy"]


def test_cli_lists_commands():
    out = CliRunner().invoke(app, ["--help"])
    assert out.exit_code == 0
    assert "data-check" in out.output and "env" in out.output
