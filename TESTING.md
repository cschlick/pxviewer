# Testing

The cctbx/phenix convention throughout, so this code can be dropped into one of those
trees without rework. **The migration is complete**: no test uses pytest, and
`python/tests/` holds only a `conftest.py` the pattern does not need.

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

`have`, `skip`, `tmp_dir`, `qt_application`, `data_path`, `closing_modals` and
`shipped_defaults` are in `pxviewer/regression/tst_utils.py` — the only things cctbx does
not already supply. Two subject-specific helpers sit beside them: `gui_invariants.py` (the
bank a GUI fuzzer asserts) and `live_harness.py` (connecting to a session and reading its
wire).

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

No `monkeypatch` uses remain anywhere in the suite.

**A test that builds a `DesktopApp` must take `shipped_defaults()`.** The app reads four
`QSettings` keys at construction, so an exercise asserting "a non-polymer opens as
ball-and-stick" is otherwise asserting about whatever the person running the tests has
configured — and two exercises *write* those keys.

This was learned the expensive way. The first attempt redirected `QSettings` to a temporary
directory with `setDefaultFormat(IniFormat)` and `setPath(...)`. On macOS that is a silent
no-op: the two-argument constructor stays on `NativeFormat`, and `setPath` is documented not
to apply to it. So the exercises wrote the *real* preferences, and three other files — which
never touched a setting and passed on their own — failed in the registry afterwards, because
they now opened models against defaults a previous script had left behind.

Two things worth taking from it:

- **A file passing standalone says nothing about it passing in the registry** when the state
  it depends on lives outside the process. One-process-per-script isolates memory, not the
  filesystem or the user's preferences.
- **Verify that isolation isolates.** `QSettings("pxviewer", "pxviewer").fileName()` said
  `~/Library/Preferences/...` the whole time; one line would have caught it.

`shipped_defaults()` snapshots the named keys, clears them, and restores on the way out.
Snapshot-and-restore is weaker than redirection — a run killed mid-exercise can still leave a
key behind — so it is applied to every script that constructs the app, not only the two that
set a preference deliberately.

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

Converted, and the pytest originals removed — **49 files, 485 exercises**:

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
| `regression/tst_gui_fuzz.py` | 2 | gui |
| `regression/tst_hotspots.py` | 22 | core |
| `regression/tst_hotspots_gui.py` | 15 | gui |
| `regression/tst_hotspots_standalone.py` | 3 | core |
| `regression/tst_hydrogens.py` | 1 | core |
| `regression/tst_kinemage.py` | 9 | core |
| `regression/tst_ligands.py` | 9 | core |
| `regression/tst_live_attributes.py` | 12 | core |
| `regression/tst_live_frames.py` | 7 | core |
| `regression/tst_live_maps.py` | 6 | core |
| `regression/tst_live_overlays.py` | 12 | core |
| `regression/tst_live_selection.py` | 17 | core |
| `regression/tst_live_volumes.py` | 9 | core |
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

The two largest files were split for a different reason: size. `test_desktop.py`'s 90
tests and 3,600 lines became five files grouped by subject — `tables`, `registry`,
`appearance`, `interaction` and `tutorials` — and `test_live.py`'s 61 tests became
`frames`, `overlays`, `selection`, `volumes` and `attributes`. A single 3,600-line
`tst_*.py` would be unlike anything in cctbx and unpleasant to run one exercise from. The
desktop five are gated; the live five are not, since they need websockets but no Qt.

`test_live.py` also gained a helper module, `live_harness.py`. Sixty exercises shared one
shape — connect, consume the topology, provoke something, read the answer — written out
longhand each time, which buried the assertion in scaffolding. Two things it encodes were
open-coded inconsistently before: that the topology always arrives first, and that text and
binary messages interleave, so "the next message" is a race a command's echo can lose to a
coordinate frame.

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

**Every pytest file is converted.** `python/tests/` holds only `conftest.py`, which the
cctbx pattern does not use.

**Cancelling a real dialog needs `AA_DontUseNativeDialogs`.** `test_gui_fuzz.py` clicks
random widgets, so it really does open dialogs, and its `guarded_modals` patches were
load-bearing where `test_gui_concurrency.py`'s were not. Replacing them with
`closing_modals()` — a timer that cancels a dialog that really opened — is right, but it
is *not* sufficient on its own, which cost half an hour to learn:

`QFileDialog.getSaveFileName` on macOS opens a **native** `NSSavePanel`. That is not a Qt
widget, so it is not in `topLevelWidgets()`, and it runs its own modal loop inside AppKit
where nothing in Qt can reach it. The run simply hangs until it is killed. The stack from
`sample(1)` on the wedged process said so plainly — `QDialog::exec` →
`runApplicationModalPanel` → `NSApplication _doModalLoop:` — which is worth remembering as
the fastest way to tell a hung GUI test from a slow one.

Two things follow:

- `qt_application()` now sets `AA_DontUseNativeDialogs`, so every dialog is a Qt widget and
  `closing_modals()` can cancel it. Cancelling then returns exactly what the old stubs
  returned (`("", "")`, `([], "")`, an invalid `QColor`), so the replacement really is
  behaviour-for-behaviour — but only with that attribute set.
- **The offscreen fallback does not fire on this machine.** `qt_application()` defaults to
  the offscreen platform only when there is no display, and XQuartz is installed here, so
  `DISPLAY` is set and the platform stays `cocoa`. That is the intended behaviour — a
  machine with a display should exercise the real GPU path — but it means "it will be
  offscreen anyway" is not a safe assumption to build on.

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

## Memory

**A test should cost about what using the application costs.** That is the standard, and
three things broke it. All three are the same mistake in different clothes: Qt frees things
*through the event loop*, and a test script does not run one.

**`deleteLater` is a post, not a delete.** `QApplication.processEvents()` deliberately does
not deliver `DeferredDelete` events — deferring past the current loop is the entire point of
them. So every pane the app rebuilt sat in the posted-event queue for the whole run. Use
`process_events()` from `tst_utils` instead of `QApplication.processEvents()`: it delivers
them, the way a real event loop eventually would. One walk went from 3694 live widgets to a
flat 454 with no other change.

**`app.stop()` did not free the app.** Use `dispose(app)` from `tst_utils`, never a bare
`app.stop()`. Measured before the fix: each `DesktopApp` left 429 of its 430 widgets alive
for the rest of the process, so a script that built five carried all five to the end. Two
production gaps fed this and are now closed — `ControlsWindow` had no teardown at all, and
`DesktopApp` never released its main window or splash (the splash is held by the closure
connected to the page-load signal, so an app that never loads a page keeps one alive with
nothing left to dismiss it). After: 430 widgets built, **zero** residual, flat across apps.

**Freeing things exposed a use-after-free the leak had been hiding.** The first `dispose`
deleted the widgets as soon as `stop` returned — and a worker thread that posts its result
back with `run_on_main` could still have a callback queued. It then landed on a freed
object and took the process down with `libshiboken: Internal C++ object ... already
deleted`. Nondeterministic, so it crashed one run and printed `OK` on the next. Objects
that are never freed cannot be used after free, so the leak had been acting as a very
expensive safety net. `dispose` now calls `settle()` first, which pumps the loop until no
`pxviewer-` worker thread is alive, so callbacks land against live widgets; only then are
the deletes delivered. **If you add a teardown path, settle before you free.**

**A random walk that only grows stops being a random walk.** `tst_gui_fuzz.py` was weighted
towards loading, so it climbed to 15 models and 8 generated map+model groups — about 2 GB,
and a scene nobody builds. Worse as a *test*: the late steps all re-probed one enormous
state instead of many different ones. `Walk.make_room` now evicts at random to stay under
`MAX_MODELS` / `MAX_VOLUMES` / `MAX_GROUPS`, which bounds the footprint, keeps the walk
moving through varied small scenes, and exercises the remove paths on the way.

Together: `tst_gui_fuzz.py` fell from **2588 MB to 889 MB** and got *faster* (239 s → 159 s
on comparable runs), and within a walk RSS is now flat at ~520 MB where it used to climb
past 2260 MB. Since the runner is serial the suite costs whatever its heaviest script costs,
and that ceiling went from **1563 MB to ~1120 MB** — nothing now exceeds roughly what one
viewport and a loaded scene cost in the application itself, which was the point.

If you suspect a leak, measure rather than guess — `QApplication.allWidgets()` is the honest
signal, and it is the one to trust. **Peak RSS is a poor instrument**: `tst_live_maps.py`
measured 491, 745 and 827 MB on three consecutive runs of *identical* code, so a single
before/after pair proves nothing at that resolution. Two further traps: `ru_maxrss` is a
high-water mark and so cannot show memory coming back (the figures here are current RSS from
`ps`), and a peak sampled by polling misses short spikes entirely on fast scripts. Live
widget counts are exact, reproducible, and directly answer the question being asked.

## Running it

The registry is 49 scripts, one process each, run **serially** — `_run_standalone` does one
`subprocess.call` at a time, so the suite costs whatever its heaviest single script costs,
not the sum. Do not run two registries at once; that, not the tests themselves, is what once
took the machine down.

A single script is the normal unit of work:

```bash
libtbx.python python/pxviewer/regression/tst_desktop_tables.py
```

The slowest are `tst_desktop_tutorials.py` (real phasing, ligand fitting and refinement),
`tst_gui_fuzz.py` (five seeded walks) and `tst_qscore.py`.
