"""The few helpers cctbx's own test_utils does not provide.

Everything else comes straight from ``libtbx.test_utils``: ``approx_equal`` for float
comparison, ``raises`` (already a context manager, same shape as pytest's), ``show_diff``,
``Exception_expected``, ``open_tmp_directory``.

One thing has no cctbx analogue: **optional dependencies.** pxviewer's GUI and live-session
tests need PySide6, QtWebEngine or websockets, none of which a headless cctbx build has.
Coarse gating belongs in ``run_tests.py`` (the pattern mmtbx uses for probe); :func:`have`
and :func:`skip` are for the finer-grained cases inside a script.

**There is deliberately no monkeypatch helper.** Not one of cctbx's 768 ``tst_*.py`` files
patches or mocks anything: they build real objects and write real files, because in this
domain the real thing is cheap to make. A test that needs a map should write one --
``VolumeData.from_numpy(...).write_map(path)`` is two lines, and the CCP4 round trip it then
exercises is usually part of what the test is about. A test tempted to spy on which method
got called should assert on the state that results instead.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import importlib
import os
import shutil
import sys
import tempfile


def have(*module_names):
    """True when every named module imports. Cheap enough to call at module scope."""
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception:
            return False
    return True


def skip(reason):
    """Report a test as skipped and exit cleanly.

    Exits 0 rather than raising: ``libtbx.test_utils.run_tests`` treats a nonzero exit as a
    failure, and a missing optional dependency is not one. The message is printed so a
    skipped run is visible rather than silently passing.
    """
    print("SKIP: %s" % reason)
    sys.stdout.flush()
    sys.exit(0)


@contextlib.contextmanager
def tmp_dir(suffix=""):
    """A temporary directory, removed afterwards."""
    path = tempfile.mkdtemp(suffix=suffix, prefix="pxviewer_tst_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def qt_application():
    """The process-wide QApplication, created once.

    Qt allows exactly one per process and cannot recreate it after teardown, so every GUI
    test in a script shares this. Tests run as separate processes under ``run_tests``, so
    there is no cross-test leakage to worry about.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def data_path(*parts):
    """A path under ``pxviewer/data``, independent of the working directory.

    The pytest suite used paths relative to the repository root, which only worked when it
    was run from there — one of the reasons several tests failed depending on where you
    stood. cctbx tests are run from anywhere, so resolve against the package.
    """
    import pxviewer

    return os.path.join(os.path.dirname(os.path.abspath(pxviewer.__file__)), "data", *parts)
