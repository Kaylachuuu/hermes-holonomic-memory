"""Run the test suite without pytest installed (pytest also works: `pytest tests`)."""
import contextlib, importlib, inspect, sys, tempfile, time, traceback, types
from pathlib import Path

root = Path(__file__).parent
sys.path[:0] = [str(root), str(root / "tests")]
import _bootstrap  # noqa: E402,F401
try:
    import pytest  # noqa: F401
except ImportError:
    shim = types.ModuleType("pytest")

    @contextlib.contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"{exc.__name__} not raised")

    class approx:
        def __init__(self, v, rel=1e-6, abs=1e-6): self.v, self.tol = v, max(abs, rel * __builtins__.abs(v))
        def __eq__(self, o): return __builtins__.abs(o - self.v) <= self.tol
    shim.raises, shim.approx = raises, approx
    sys.modules["pytest"] = shim

failed = 0
for path in sorted((root / "tests").glob("test_*.py")):
    mod = importlib.import_module(path.stem)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp)) if inspect.signature(fn).parameters else fn()
            print(f"PASS {name} ({time.time() - t0:.2f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
print("\nall passed" if not failed else f"\n{failed} failed")
sys.exit(1 if failed else 0)
