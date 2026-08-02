# Testing

Moving to the cctbx/phenix convention, so this code can be dropped into one of those trees
without rework. **This migration is partial** — see the inventory at the end.

## The pattern

A test is an ordinary program. No runner, no collection, no plugins:

- it lives at `python/pxviewer/regression/tst_<subject>.py`;
- it defines `exercise_*()` functions and a `run()` that calls them;
- it prints `OK` on success and exits nonzero on failure;
- it is registered in `python/pxviewer/run_tests.py`.

```bash
libtbx.python -m pxviewer.run_tests                                  # everything
libtbx.python python/pxviewer/regression/tst_validation_events.py    # one test
```

Once pxviewer is a configured libtbx module, `libtbx.run_tests_parallel module=pxviewer
nproc=8` works with no change — `run_tests.py` uses `libtbx.test_utils.run_tests` when the
module resolves and falls back to running the scripts itself when it does not, which is the
normal case while pxviewer is pip-installed.

Two conventions worth keeping:

**Optional dependencies are gated in the registry, not scattered through the scripts.**
pxviewer's GUI and live-session tests need PySide6, QtWebEngine and websockets, which a
headless cctbx build will not have. `run_tests.py` assembles its list accordingly and
announces what it skipped — the same thing mmtbx does with probe. `tst_utils.skip()` is for
the finer-grained cases inside a script.

**Tests must run from any directory.** Use `tst_utils.data_path()` rather than a path
relative to the repository root.

This is not a style preference — the pytest suite has *contradictory* working-directory
requirements and there is no directory that runs all of it. Measured:

Measured on `test_hotspots.py` before it was converted, against `test_desktop.py` which
still is not:

| | from repo root | from `python/` |
| --- | --- | --- |
| `tests/test_hotspots.py` (now converted) | 40 passed | 22 failed, 6 errors |
| `tests/test_desktop.py` | 4 spurious failures | passes |

One hardcoded `python/pxviewer/data/1tec.pdb`, which only resolves from the root; the other
resolves data through the installed package and only works from `python/`. Converted tests
resolve against the package and run from anywhere, which is what cctbx assumes -- every
`tst_*.py` here is verified from `/tmp`.

## Converting a pytest test

Most of it is mechanical. `libtbx.test_utils` already provides the equivalents:

| pytest | cctbx |
| --- | --- |
| `assert x == pytest.approx(y)` | `assert approx_equal(x, y)` |
| `with pytest.raises(E) as e:` | `with raises(E) as e:` (same shape, from `libtbx.test_utils`) |
| `pytest.importorskip("m")` | `if not have("m"): skip("...")` |
| `pytest.skip(reason)` | `skip(reason)` |
| `tmp_path` | `with tmp_dir() as path:` |
| `monkeypatch.setattr(o, "a", v)` | `with monkeypatched(o, "a", v):` |
| `@pytest.fixture(scope="module")` | a module-level function with a cache list |
| `@pytest.mark.parametrize` | a `for` loop over the cases |
| `capsys` | `contextlib.redirect_stdout(io.StringIO())` |

`have`, `skip`, `monkeypatched`, `tmp_dir`, `qt_application` and `data_path` are in
`pxviewer/regression/tst_utils.py` — the only things cctbx does not already supply.

Two things to watch when converting:

- **A fixture becomes a cached function, and the cache is per process.** Under `run_tests`
  every script is its own process, so there is no cross-test leakage to design around — but
  within a script the cached model *is* shared, so do not mutate it. `extract_bonds` builds
  restraints and reorders the hierarchy, for instance, so pass `geometry=` rather than
  letting it mutate a shared model.
- **`approx_equal` prints a diff and returns a bool**; it does not raise. Keep the `assert`.

## Inventory

Converted, and the pytest originals removed:

| test | exercises | list |
| --- | ---: | --- |
| `regression/tst_validation_events.py` | 14 | core |
| `regression/tst_hotspots.py` | 22 | core |
| `regression/tst_concern.py` | 5 | core |
| `regression/tst_hotspots_standalone.py` | 3 | core |
| `regression/tst_hotspots_gui.py` | 15 | gui |

`test_hotspots.py` became three files rather than one. Its 40 tests mixed pure calibration
with desktop-shell wiring, and splitting them along that seam is what lets the calibration
half -- the part that pins the science -- run in a headless cctbx build with no Qt at all.
The concern-import tests that need no viewer went to `tst_concern.py` for the same reason.

**Not yet converted: 34 files, 9,925 lines, still requiring pytest.**

The bulk is concentrated: `test_desktop.py` alone is 3,600 lines and `test_live.py` 1,063,
together nearly half the remainder. Both are GUI/live-session tests and would go in the
gated `gui_tests` list. `test_live_maps.py`, `test_volume_io.py` and `test_cctbx_io.py` are
the natural next ones -- small, pure computation, and they belong in `core_tests` where they
would run in a headless build.

Counts of what the remainder leans on, which is what keeps the conversion mechanical rather
than hard: 271 `importorskip`, 138 `tmp_path`, 53 `monkeypatch`, 52 `approx`, 44 `raises`,
16 fixtures, 10 `parametrize`, 2 `capsys`.
