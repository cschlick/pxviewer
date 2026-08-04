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

This is not a style preference — the pytest suite had *contradictory* working-directory
requirements, and there was no directory that ran all of it. Measured before either file was
converted:

| | from repo root | from `python/` |
| --- | --- | --- |
| `tests/test_hotspots.py` | 40 passed | 22 failed, 6 errors |
| `tests/test_desktop.py` | 4 spurious failures | passes |

One hardcoded `python/pxviewer/data/1tec.pdb`, which only resolves from the root; the other
resolved data through the installed package and only worked from `python/`. Four more
hardcoded `pxviewer/data/1ubq.pdb` turned up while converting `test_desktop.py`. Converted
tests resolve against the package and run from anywhere, which is what cctbx assumes — every
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
| `monkeypatch` / `mock` | **nothing — build the real thing.** See below. |
| `@pytest.fixture(scope="module")` | a module-level function with a cache list |
| `@pytest.mark.parametrize` | a `for` loop over the cases |
| `capsys` | `contextlib.redirect_stdout(io.StringIO())` |

`have`, `skip`, `tmp_dir`, `qt_application` and `data_path` are in
`pxviewer/regression/tst_utils.py` — the only things cctbx does not already supply.

**Do not patch or mock.** Not one of cctbx's 768 `tst_*.py` files does, and it is a habit
worth dropping rather than porting: in this domain the real thing is cheap to build, and
faking it tests less. Two substitutions cover essentially every case:

- *A test that needs a map should write one.* `VolumeData.from_numpy(...).write_map(path)` is
  two lines, and the CCP4 round trip it then exercises is usually part of what the test is
  about — `tst_concern.py` gained real coverage of NXSTART placement and anisotropic pixel
  sizes by making exactly this swap.
- *A test tempted to spy on which method got called should assert on the resulting state.*
  Watching for `set_volume_iso(vid, 1.5)` tests the implementation; reading the volume's `iso`
  back afterwards tests the behaviour, and survives the implementation changing.

The second substitution has been the more valuable one in practice, because the state is
usually a stronger assertion than the spy it replaces:

| the spy was | the state is | why it is better |
| --- | --- | --- |
| wrap `session.push` and count calls | read `session._frame_index` | it is what the wire protocol numbers frames with, so it moves if and only if a frame really went out |
| wrap `fmodel.update_all_scales` | compare `k_isotropic`, `k_anisotropic`, `k_masks` | catches a rescale reached *any* way, including from inside cctbx where no spy would see it |
| replace `_CATEGORIES` with a stub | run the shipping map against a small local class | tests the configuration that ships rather than one invented for the test |

The last of those is worth generalising: patching a module constant to something convenient
usually means the test no longer covers the real value. `tst_api_guide.py` needed a class
with one method in two different categories and one in none — and the real category map
already puts `select` and `color_by` in different groups, so a four-method class exercises
every path against the shipping configuration.

Two more entries earned their place while `test_desktop.py` was being split:

| the spy was | the state is | why it is better |
| --- | --- | --- |
| wrap `_viewport.load` to prove no reload | read `app._scene_counter` | the app's own counter, one per composed scene. Reading the scene *file* does not work: composing one is what writes it, so the observer causes the event |
| wrap `_clear_layout` to catch a pane rebuild | compare the pane's child widgets | a rebuild replaces them, so the same objects still being there *is* the claim |

Seven `monkeypatch` uses remain, all in the two files still on pytest.

Two things to watch when converting:

- **A fixture becomes a cached function, and the cache is per process.** Under `run_tests`
  every script is its own process, so there is no cross-test leakage to design around — but
  within a script the cached model *is* shared, so do not mutate it. `extract_bonds` builds
  restraints and reorders the hierarchy, for instance, so pass `geometry=` rather than
  letting it mutate a shared model.
- **`approx_equal` prints a diff and returns a bool**; it does not raise. Keep the `assert`.
- **`raises` cannot handle every exception.** It instantiates the class with no arguments to
  test `isinstance`, so it blows up on anything whose constructor requires them —
  `urllib.error.HTTPError` needs five. Fall back to `try` / `except` with
  `raise Exception_expected` in the try, which is the older cctbx idiom and always works.

## Inventory

Converted, and the pytest originals removed — **43 files, 426 exercises**:

| test | exercises | list |
| --- | ---: | --- |
| `regression/tst_analysis.py` | 8 | core |
| `regression/tst_api.py` | 13 | core |
| `regression/tst_api_guide.py` | 7 | core |
| `regression/tst_appserver.py` | 3 | gui |
| `regression/tst_bcif.py` | 5 | core |
| `regression/tst_cctbx_io.py` | 24 | gui |
| `regression/tst_concern.py` | 6 | core |
| `regression/tst_console.py` | 5 | gui |
| `regression/tst_data.py` | 10 | core |
| `regression/tst_demos.py` | 7 | core |
| `regression/tst_desktop_appearance.py` | 25 | gui |
| `regression/tst_desktop_interaction.py` | 22 | gui |
| `regression/tst_desktop_registry.py` | 16 | gui |
| `regression/tst_desktop_tables.py` | 16 | gui |
| `regression/tst_desktop_tutorials.py` | 10 | gui |
| `regression/tst_edits.py` | 4 | core |
| `regression/tst_geometry.py` | 6 | core |
| `regression/tst_gpu.py` | 15 | core |
| `regression/tst_gui_concurrency.py` | 4 | gui |
| `regression/tst_hotspots.py` | 22 | core |
| `regression/tst_hotspots_gui.py` | 15 | gui |
| `regression/tst_hotspots_standalone.py` | 3 | core |
| `regression/tst_hydrogens.py` | 1 | core |
| `regression/tst_kinemage.py` | 9 | core |
| `regression/tst_ligands.py` | 9 | core |
| `regression/tst_live_maps.py` | 6 | core |
| `regression/tst_loader.py` | 9 | core |
| `regression/tst_minimize.py` | 9 | core |
| `regression/tst_mvs.py` | 19 | core |
| `regression/tst_palettes.py` | 7 | core |
| `regression/tst_primitives.py` | 24 | core |
| `regression/tst_probe.py` | 5 | core |
| `regression/tst_qscore.py` | 5 | gui |
| `regression/tst_reflections.py` | 7 | core |
| `regression/tst_reflections_gui.py` | 9 | gui |
| `regression/tst_tug.py` | 13 | core |
| `regression/tst_tug_gui.py` | 3 | gui |
| `regression/tst_validation.py` | 8 | core |
| `regression/tst_validation_events.py` | 14 | core |
| `regression/tst_volume.py` | 6 | core |
| `regression/tst_volume_demos.py` | 6 | core |
| `regression/tst_volume_io.py` | 6 | core |
| `regression/tst_webapp.py` | 5 | gui |

**Five files were split rather than converted one-to-one**, each along the same seam: a
subject whose computation is pure cctbx and whose consequences are desktop wiring.
`test_hotspots.py` became `tst_hotspots.py`, `tst_hotspots_gui.py` and
`tst_hotspots_standalone.py`; `test_reflections.py` and `test_tug.py` each became a core
file and a `_gui` one. The split is what lets the half that pins the science — hotspot
calibration, phasing and sigma-scaling, what a tug does to coordinates — run in a headless
cctbx build with no Qt at all, which is the environment they are most likely to be run in
once upstreamed. The concern-import tests that need no viewer went to `tst_concern.py` for
the same reason.

`test_desktop.py` was split for a different reason: size. Its 90 tests and 3,600 lines
became five files grouped by subject — `tables`, `registry`, `appearance`, `interaction`
and `tutorials` — because a single 3,600-line `tst_*.py` would be unlike anything in cctbx
and unpleasant to run one exercise from. All five are gated, since all of them need Qt.

**Three assertions were already failing and are restated rather than ported.** Converting a
test means running it, which is how these surfaced; each was checked against the pytest
original first, so none is conversion damage.

- `test_analysis.py` demanded more than 200 severe clashes on 1TEC. That predated the 0.40 Å
  reporting gate and the hydrogen-bond exclusion, which together cut it to 172. It is now
  stated as the with-hydrogen versus bare ratio (172 against 10), which is the actual claim
  and does not move when the calibration is re-anchored.
- `test_desktop.py` asserted that the inactive minimize button had *no* stylesheet. It has
  one — the plain framed look. The accent is a `background:` fill, so that is what the
  exercise now looks for.
- The same file's `_detect_map_model_shift` tests passed only because the detector was
  patched out; see the note on building a misplaced pair, below.

**One test was dropped rather than converted.** `test_geometry.py` checked that
`build_geometry` returns `None` when no monomer library is present, by deleting two
environment variables and replacing `geometry._chem_data_geostd` with a stub. That branch
is not reachable here without patching — chem_data is installed and importable, so clearing
the environment alone still finds geostd. The reachable half of the guard is kept and the
gap is stated in the exercise's docstring, rather than reintroducing patching for one case.

**A patched test can pass for the wrong reason, and only building the real thing shows it.**
`test_desktop.py` replaced `_detect_map_model_shift` with one returning a chosen shift, so
the join between detecting a shift and applying it was never exercised. Writing a genuinely
misplaced pair instead — the real search recovers 2.98 Å from a 3.0 Å offset in well under a
second — turned up two facts the patched version could not have:

- Displacing a model with `shift_model_and_set_crystal_symmetry` writes an *undisplaced*
  file. `model_as_pdb` undoes the shift to recover the source coordinates, which is exactly
  what that API is for and what the production path is asserted to rely on elsewhere. The
  offset has to be a plain `set_sites_cart` move.
- A pair opened *together* can still need aligning. Pairing on load works from file metadata
  and cannot know about coordinates that are simply in the wrong place.

**Not yet converted: 2 files, 1,424 lines, still requiring pytest:**

- `test_live.py` (1063 lines)
- `test_gui_fuzz.py` (361 lines)

`test_live.py` is not hard in kind, only in size — the live-session protocol, which the
converted `tst_mvs.py`, `tst_primitives.py` and `tst_cctbx_io.py` already touch the edges
of. It will want splitting the way `test_desktop.py` was.

`test_gui_fuzz.py` is the one file with a genuine design question left. It builds a random
walk over the app's own controls, clicking, toggling and dragging real widgets, so the
modals it opens are real and its `guarded_modals` patches are load-bearing in a way
`test_gui_concurrency.py`'s were not. The substitute is already written: `closing_modals()`
in `tst_utils.py` cancels a dialog that really opened, and returns exactly what the stubs
returned. What remains to settle is the walk itself — how to seed it reproducibly under
one-process-per-script, and whether the parametrised seeds become a loop or separate
exercises.

`test_gpu.py` was expected to be the hard case — 109 lines carrying 24 `monkeypatch` uses,
which read as a file built entirely out of faked hardware states. It converted whole, with
nothing dropped and one patch left. Most of those 24 were `setenv`/`delenv`, and the
chooser reads the environment because the environment is genuinely its input: setting a
variable and restoring it afterwards is not a mock. Two more pointed `XDG_CACHE_HOME` at a
temp directory, which made the *real* cache file cheap to use — and using it added coverage
the original had none of, since the round trip through the machine signature and the
version stamp is what decides whether a remembered verdict is trusted (three new exercises:
a cache from another machine, from an older version, and unreadable).

That left `os.execv`, which cannot be substituted by building anything, because it does not
return — a test that let it fire would be replaced by the process it launched. It is now a
`restart=` parameter with `os.execv` as its default, the same injectable seam the module
already gives `log`. Worth stating plainly: **that is a change to production code made for
a test.** It is justified here because the parameter marks a real boundary and the module
had already drawn the same one for logging. It is not a licence to add a hook wherever a
patch used to be — the first question stays "what state does this leave behind?", and it
had an answer for the other 23 cases in this file.

Counts of what the remainder leans on: 9 `raises`, 7 `monkeypatch`, 6 `approx`,
4 `importorskip`, 4 fixtures, 3 `parametrize`.
