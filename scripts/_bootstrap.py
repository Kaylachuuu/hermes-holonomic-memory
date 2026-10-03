"""Make this folder importable as the package `holonomic`, whatever the folder is called.
Used by the tests and scripts; Hermes loads the package itself.

Development scripts live in scripts/ and tests/, never in the plugin's top folder:
Hermes executes every .py file it finds there when it loads the plugin."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "holonomic" not in sys.modules:
    spec = importlib.util.spec_from_file_location("holonomic", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["holonomic"] = module
    spec.loader.exec_module(module)
