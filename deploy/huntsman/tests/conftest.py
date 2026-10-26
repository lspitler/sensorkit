"""Load deploy/huntsman/huntsman_dome.py once for every test module (as huntsman_dome_under_test).

The plugin is a loose file, not a package, and it declares SensorKit keywords
at import, which may only happen once per process.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_NAME = "huntsman_dome_under_test"

if _NAME in sys.modules:
    hd = sys.modules[_NAME]
else:
    _path = pathlib.Path(__file__).parents[1] / "huntsman_dome.py"
    _spec = importlib.util.spec_from_file_location(_NAME, _path)
    hd = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = hd
    _spec.loader.exec_module(hd)
