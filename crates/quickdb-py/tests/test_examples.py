"""The shipped examples run and print what they promise."""

import pathlib
import subprocess
import sys

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def test_agent_memory_example_runs():
    out = subprocess.run([sys.executable, str(EXAMPLES / "agent_memory.py")], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    assert "decisions on record: 2" in out.stdout
    assert "Q: الفريق الميداني" in out.stdout
