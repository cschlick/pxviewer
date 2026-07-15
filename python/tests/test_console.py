"""Tests for the embedded IPython console (optional 'console' extra)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("qtconsole")
pytest.importorskip("ipykernel")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_console_available():
    from pxviewer import console

    assert console.console_available() is True


def test_embedded_console_shares_live_objects(qapp):
    """The in-process kernel evaluates against the very objects we push in."""
    from pxviewer.console import EmbeddedConsole

    class FakeSession:
        marker = "live"

    console = EmbeddedConsole({"session": FakeSession(), "answer": 21})
    try:
        shell = console._manager.kernel.shell
        # The pushed object is the same one the kernel sees.
        assert shell.user_ns["session"].marker == "live"
        # It is a real IPython shell, not a stub.
        assert shell.ev("answer * 2") == 42
        # Rebinding (used to track the active model) takes effect.
        console.push({"session": "rebound"})
        assert shell.user_ns["session"] == "rebound"
    finally:
        console.shutdown()


def test_console_suppresses_kernel_banner(qapp):
    """The widget squelches IPython's own banner so only our greeting shows."""
    from pxviewer.console import EmbeddedConsole

    console = EmbeddedConsole()
    try:
        # The kernel-info reply sets this trait; our observer must blank it out.
        console.widget.kernel_banner = "Python 3.12 ... IPython 9 ... Tip: ..."
        assert console.widget.kernel_banner == ""
    finally:
        console.shutdown()
