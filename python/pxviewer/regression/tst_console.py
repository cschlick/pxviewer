"""The embedded IPython console (optional 'console' extra)."""

from __future__ import absolute_import, division, print_function

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from pxviewer.regression.tst_utils import have, qt_application, skip

if not have("PySide6", "qtconsole", "ipykernel"):
    skip("PySide6 / qtconsole / ipykernel not available")


def exercise_console_available():
    from pxviewer import console

    assert console.console_available() is True


def exercise_the_embedded_console_shares_live_objects():
    """The in-process kernel evaluates against the very objects we push in."""
    from pxviewer.console import EmbeddedConsole

    class FakeSession(object):
        marker = "live"

    console = EmbeddedConsole({"session": FakeSession(), "answer": 21})
    try:
        shell = console._manager.kernel.shell
        assert shell.user_ns["session"].marker == "live"   # the same object we pushed
        assert shell.ev("answer * 2") == 42                 # a real shell, not a stub
        console.push({"session": "rebound"})                # rebinding takes effect
        assert shell.user_ns["session"] == "rebound"
    finally:
        console.shutdown()


def exercise_the_console_suppresses_the_kernel_banner():
    """The widget squelches IPython's own banner so only our greeting shows."""
    from pxviewer.console import EmbeddedConsole

    console = EmbeddedConsole()
    try:
        # The kernel-info reply sets this trait; our observer must blank it out.
        console.widget.kernel_banner = "Python 3.12 ... IPython 9 ... Tip: ..."
        assert console.widget.kernel_banner == ""
    finally:
        console.shutdown()


def exercise_the_banner_fits_the_pane():
    """The console sits in the controls pane -- about 38 monospace columns. A wider banner
    wraps mid-sentence, which reads as a mess and is worse than saying less."""
    from pxviewer.console import BANNER_MAX_COLUMNS, default_banner

    too_wide = [l for l in default_banner().splitlines()
                if len(l) > BANNER_MAX_COLUMNS]
    assert not too_wide, "these wrap in the console: %s" % too_wide


def exercise_the_banner_points_at_the_names_in_scope():
    """It names what is actually bound, and where the cctbx objects are: a banner that
    advertises something not there, or that returns None, is worse than none."""
    from pxviewer.console import default_banner

    banner = default_banner()
    assert "session" in banner and "app" in banner
    assert "session.model" in banner        # the cctbx mmtbx.model.manager
    assert "group_mmm" in banner            # the cctbx map_model_manager
    assert "numpy" not in banner and "np =" not in banner


def run():
    qt_application()        # the console widgets need one, created once per process
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
