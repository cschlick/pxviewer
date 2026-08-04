"""Specify tests to be run in this and sub-directories.

The cctbx/phenix convention, so this module can be dropped into one of those trees without
rework: tests are standalone ``tst_*.py`` scripts that run under ``libtbx.python``, exit
nonzero on failure, and print ``OK`` at the end; this file is the registry that names them.

Run everything::

    libtbx.python -m pxviewer.run_tests

or, once pxviewer is a configured libtbx module::

    libtbx.run_tests_parallel module=pxviewer nproc=8

Run one test the same way you would in cctbx -- it is just a program::

    libtbx.python pxviewer/regression/tst_validation_events.py

**Optional dependencies are gated here, not inside the scripts.** pxviewer's GUI and
live-session tests need PySide6, QtWebEngine and websockets, which a headless cctbx build
will not have; the lists below are assembled accordingly, mirroring how mmtbx excludes its
MolProbity tests when probe is absent. A skipped group is announced rather than silently
dropped.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

# --- the tests -------------------------------------------------------------------
#
# $D is the module's dist path, filled in by libtbx.test_utils.run_tests. Keep each list
# in dependency order: cheap and self-contained first, so a broken environment fails fast.

#: Pure computation: cctbx, numpy, scipy. No Qt, no websockets, no display.
core_tests = [
    "$D/regression/tst_bcif.py",
    "$D/regression/tst_palettes.py",
    "$D/regression/tst_gpu.py",
    "$D/regression/tst_api_guide.py",
    "$D/regression/tst_kinemage.py",
    "$D/regression/tst_loader.py",
    "$D/regression/tst_volume_io.py",
    "$D/regression/tst_data.py",
    "$D/regression/tst_hydrogens.py",
    "$D/regression/tst_volume.py",
    "$D/regression/tst_volume_demos.py",
    "$D/regression/tst_edits.py",
    "$D/regression/tst_api.py",
    "$D/regression/tst_mvs.py",
    "$D/regression/tst_primitives.py",
    "$D/regression/tst_demos.py",
    "$D/regression/tst_geometry.py",
    "$D/regression/tst_minimize.py",
    "$D/regression/tst_tug.py",
    "$D/regression/tst_reflections.py",
    "$D/regression/tst_live_maps.py",
    "$D/regression/tst_validation.py",
    "$D/regression/tst_analysis.py",
    "$D/regression/tst_probe.py",
    "$D/regression/tst_ligands.py",
    "$D/regression/tst_validation_events.py",
    "$D/regression/tst_hotspots.py",
    "$D/regression/tst_concern.py",
    "$D/regression/tst_hotspots_standalone.py",
]

#: Need PySide6 + QtWebEngine (the desktop shell), websockets (the live session), or a
#: built frontend bundle (the webapp serves it).
gui_tests = [
    "$D/regression/tst_cctbx_io.py",
    "$D/regression/tst_qscore.py",
    "$D/regression/tst_reflections_gui.py",
    "$D/regression/tst_tug_gui.py",
    "$D/regression/tst_gui_concurrency.py",
    "$D/regression/tst_appserver.py",
    "$D/regression/tst_console.py",
    "$D/regression/tst_webapp.py",
    "$D/regression/tst_hotspots_gui.py",
]

tst_list = tuple(core_tests)
tst_list_expected_unstable = ()


def _have(*module_names):
    from pxviewer.regression.tst_utils import have

    return have(*module_names)


def _assemble():
    """Build the list, announcing anything the environment cannot run."""
    tests = list(core_tests)
    if _have("PySide6.QtWebEngineWidgets", "websockets"):
        tests += gui_tests
    elif gui_tests:
        print("Skipping %d GUI tests: PySide6 QtWebEngine / websockets not available"
              % len(gui_tests))
    return tuple(tests)


def run():
    try:
        import libtbx.load_env                      # noqa: F401
        from libtbx import test_utils
    except ImportError:
        return _run_standalone()

    try:
        build_dir = libtbx.env.under_build("pxviewer")
        dist_dir = libtbx.env.dist_path("pxviewer")
    except Exception:
        build_dir = dist_dir = None
    if not dist_dir:
        # pxviewer is pip-installed rather than configured as a libtbx module, which is the
        # normal case today -- dist_path returns None rather than raising. Fall back rather
        # than fail: the scripts are ordinary programs and do not need the build system.
        return _run_standalone()
    test_utils.run_tests(build_dir, dist_dir, _assemble())


def _run_standalone():
    """Run the registry without a configured libtbx module.

    ``libtbx.test_utils.run_tests`` needs pxviewer to be a build target to resolve ``$D``.
    Until it is, resolve ``$D`` to the package directory and run each script in turn. Same
    tests, same pass/fail, no build system required.
    """
    import subprocess

    dist = os.path.dirname(os.path.abspath(__file__))
    tests = _assemble()
    failures = []
    for entry in tests:
        args = [entry] if isinstance(entry, str) else list(entry)
        script = args[0].replace("$D", dist)
        print("=" * 72)
        print(script, *args[1:])
        sys.stdout.flush()
        code = subprocess.call([sys.executable, script] + args[1:])
        if code != 0:
            failures.append((script, code))
    print("=" * 72)
    if failures:
        print("FAILED %d of %d:" % (len(failures), len(tests)))
        for script, code in failures:
            print("  exit %d  %s" % (code, script))
        return 1
    print("OK  (%d tests)" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(run() or 0)
