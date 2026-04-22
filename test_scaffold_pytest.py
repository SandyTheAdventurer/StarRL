import subprocess
import sys
import os
import pytest


def test_scaffold_actions():
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "test_scaffold_runner.py")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Test failed: {result.stderr}"