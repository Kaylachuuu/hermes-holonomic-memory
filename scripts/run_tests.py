"""Run the test suite without pytest installed (pytest also works: `pytest tests`)."""
import builtins, contextlib, importlib, inspect, sys, tempfile, time, traceback, types
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "scripts"), str(root / "tests")]
    try:
        import pytest  # noqa: F401
    except ImportError:
        shim = types.ModuleType("pytest")

        @contextlib.contextmanager
        def raises(exc, match=None):
            try:
                yield
            except exc as caught:
                import re
                if match is not None and not re.search(match, str(caught)):
                    raise AssertionError(f"{exc.__name__} raised, but {match!r} is not in: {caught}")
                return
            raise AssertionError(f"{exc.__name__} not raised")

        class approx:
            def __init__(self, v, rel=1e-6, abs=1e-6):
                self.v, self.tol = v, max(abs, rel * builtins.abs(v))

            def __eq__(self, other):
                return builtins.abs(other - self.v) <= self.tol

        shim.raises, shim.approx = raises, approx
        shim.fixture = lambda *a, **k: (lambda fn: fn)
        sys.modules["pytest"] = shim
    import conftest

    failed = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            t0 = time.time()
            try:
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                    try:
                        fn(Path(tmp)) if inspect.signature(fn).parameters else fn()
                    finally:
                        conftest.close_all()        # Windows cannot delete a database that is still open
                print(f"PASS {name} ({time.time() - t0:.2f}s)")
            except Exception:
                failed += 1
                print(f"FAIL {name}\n{traceback.format_exc()}")
    print("\nall passed" if not failed else f"\n{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
