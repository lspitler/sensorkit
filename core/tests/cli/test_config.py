# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys

import pytest


def config_load(path) -> subprocess.CompletedProcess:
    # A fresh process, since config loading is memoized per process.
    return subprocess.run(
        [sys.executable, "-m", "sensorkit.cli", "config", "load", str(path)],
        env={**os.environ, "SENSORKIT_BACKEND": "fake"},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("version: 1\nbogus: {}\n", id="unknown-section"),
        pytest.param("version: 999\n", id="unsupported-version"),
        pytest.param("version: 1\nsensors: [{}]\n", id="invalid-section"),
    ],
)
def test_load_failure_exits_nonzero(tmp_path, content):
    path = tmp_path / "sensorkit.yaml"
    path.write_text(content)

    assert config_load(path).returncode != 0


def test_load_missing_file_exits_nonzero(tmp_path):
    assert config_load(tmp_path / "missing.yaml").returncode != 0


def test_load_valid_config_exits_zero(tmp_path):
    path = tmp_path / "sensorkit.yaml"
    path.write_text("version: 1\n")

    assert config_load(path).returncode == 0
