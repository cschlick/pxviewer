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

    The platform plugin is chosen the way the pytest ``conftest.py`` chose it: default to
    ``offscreen`` only where there is no display, since it needs neither a display nor a
    GPU and is where the suite is proven to pass headless. Where a display *is* present,
    leave Qt to pick the native platform, so the tests exercise the same GPU path the app
    would. Set ``QT_QPA_PLATFORM`` yourself to override.
    """
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Every dialog must be a Qt widget, so :func:`closing_modals` can find and cancel it.
    # A *native* dialog -- an NSSavePanel on macOS -- is not in topLevelWidgets() and runs
    # its own modal loop in AppKit, so nothing in Qt can dismiss it and a run that opens
    # one hangs until it is killed. That is not hypothetical: it wedged the widget fuzzer
    # for half an hour, because this machine has XQuartz installed, DISPLAY is therefore
    # set, and the platform stays cocoa rather than falling back to offscreen.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    return QApplication.instance() or QApplication([])


#: Every ``QSettings`` key the desktop reads or writes. Listed rather than discovered:
#: on macOS ``allKeys()`` also returns the whole NSGlobalDomain, so anything that walked
#: it would snapshot several hundred of the user's system preferences.
DESKTOP_SETTINGS_KEYS = (
    "defaults/focus_surroundings",
    "defaults/model_representations",
    "defaults/molstar_interactions",
    "defaults/shown_structure_types",
    "work_dir",
)


@contextlib.contextmanager
def shipped_defaults():
    """Run with the preferences a fresh install has, and put the user's back afterwards.

    **Any test that constructs a ``DesktopApp`` needs this**, not only the ones that write
    a preference. The app reads these keys at construction, so an exercise asserting that
    "a non-polymer opens as ball-and-stick" is really asserting about whatever the person
    running the tests happens to have configured.

    Snapshot-and-restore rather than redirection, because redirection does not work here:
    ``QSettings.setDefaultFormat`` leaves the two-argument constructor on ``NativeFormat``
    on macOS, and ``setPath`` is documented not to apply to that format. A run that dies
    between the snapshot and the restore therefore *can* leave a key behind -- which is
    exactly what happened once, and is why every desktop script takes this rather than
    only the two that set a preference deliberately.
    """
    from PySide6.QtCore import QSettings

    settings = QSettings("pxviewer", "pxviewer")
    saved = [(key, settings.contains(key), settings.value(key))
             for key in DESKTOP_SETTINGS_KEYS]
    for key in DESKTOP_SETTINGS_KEYS:
        settings.remove(key)
    settings.sync()
    try:
        yield settings
    finally:
        for key, existed, value in saved:
            if existed:
                settings.setValue(key, value)
            else:
                settings.remove(key)
        settings.sync()


@contextlib.contextmanager
def closing_modals(interval_ms=20):
    """Cancel any modal dialog that appears, so a stray one cannot hang the run.

    A timer rather than a patched ``QMessageBox.warning`` / ``QFileDialog.getSaveFileName``
    / ``QColorDialog.getColor``: the timer fires inside the nested event loop the dialog's
    ``exec()`` spins, so the dialog is really constructed, really shown, and really
    cancelled, and the calling handler gets its answer and carries on.

    That is not merely the no-patching rule applied for its own sake -- cancelling returns
    exactly what the stubs used to return (``("", "")``, ``([], "")``, an invalid
    ``QColor``), so this is a strict replacement that additionally covers the dialog
    construction and the app's own handling of a cancel.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog

    def cancel_visible_modals():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog) and widget.isVisible() and widget.isModal():
                widget.reject()

    timer = QTimer()
    timer.setInterval(interval_ms)
    timer.timeout.connect(cancel_visible_modals)
    timer.start()
    try:
        yield
    finally:
        timer.stop()


def process_events():
    """Let Qt catch up, *including* the deletions it has deferred.

    Use instead of a bare ``QApplication.processEvents()`` anywhere a test drives the UI
    repeatedly. ``processEvents`` deliberately does not deliver ``DeferredDelete`` -- the
    whole point of those events is to outlive the current event loop -- so a rebuilt pane
    posts its old widgets for deletion and they are never collected. Under a real event
    loop they would be; under a test that only calls ``processEvents`` they accumulate for
    the length of the run, which measured as a walk growing from 888 to 3694 live widgets
    without the scene itself getting any bigger.

    Delivering them here is what makes a test's memory resemble the application's.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def dispose(app):
    """Stop a ``DesktopApp`` and actually free it. Use instead of a bare ``app.stop()``.

    ``stop`` closes the windows and calls ``deleteLater`` on them, which only *posts* a
    deletion — and ``processEvents`` does not deliver ``DeferredDelete``, by design, since
    the events exist precisely to defer past the current event loop. In a desktop run the
    posted deletes are collected as Qt shuts down and nobody notices. In a test script
    nothing collects them, so every app a script builds stays resident: measured at 430
    live widgets each, and a script that built five carried all five to the end.

    Delivering them explicitly is what makes a multi-app script cost one app rather than
    all of them. After this, ``QApplication.allWidgets()`` is empty again.

    **The settle step is not optional.** A worker thread posts its result back with
    ``run_on_main``; if the widgets are freed while such a callback is still queued, it
    lands on a deleted C++ object and the process dies with ``libshiboken: Internal C++
    object ... already deleted`` -- a real use-after-free that the old leak was hiding,
    since objects that are never freed cannot be used after free. So: stop, let the
    workers finish and their callbacks run *against live widgets*, and only then deliver
    the deletes.
    """
    try:
        app.stop()
    finally:
        settle()              # workers finish and their callbacks land, widgets alive
        process_events()      # ... and only now deliver the deferred deletes


#: Worker threads the app starts are named for it, which is what makes them identifiable
#: without reaching into the app's internals.
WORKER_PREFIX = "pxviewer-"


def settle(timeout=120.0):
    """Pump the loop until no pxviewer worker thread is left running.

    Deliberately does *not* deliver ``DeferredDelete``: the point is to let everything
    still in flight land while the objects it refers to are alive.
    """
    import threading
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(t.name.startswith(WORKER_PREFIX) and t.is_alive()
                   for t in threading.enumerate()):
            break
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()   # let the last callbacks land


def data_path(*parts):
    """A path under ``pxviewer/data``, independent of the working directory.

    The pytest suite used paths relative to the repository root, which only worked when it
    was run from there — one of the reasons several tests failed depending on where you
    stood. cctbx tests are run from anywhere, so resolve against the package.
    """
    import pxviewer

    return os.path.join(os.path.dirname(os.path.abspath(pxviewer.__file__)), "data", *parts)
