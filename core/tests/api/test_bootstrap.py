# SPDX-License-Identifier: Apache-2.0
import pathlib
import sys

import pytest

import sensorkit.api as sk


@pytest.mark.asyncio
async def test_import_modules_twice_with_file_import(tmp_path: pathlib.Path, monkeypatch):
    module = tmp_path / "bootstrap_file_import.py"
    module.write_text(
        "from pydantic import BaseModel\n"
        "\n"
        "import sensorkit.api as sk\n"
        "\n"
        "@sk.declare_keyword(key='BootstrapFileImport')\n"
        "class BootstrapFileImport(BaseModel):\n"
        "    pass\n"
    )
    # Keep any default config in the working directory out of the imports.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SENSORKIT_IMPORTS", str(module))

    try:
        await sk.import_modules()
        loaded = sys.modules[module.stem]
        await sk.import_modules()

        assert sys.modules[module.stem] is loaded
    finally:
        sys.modules.pop(module.stem, None)
