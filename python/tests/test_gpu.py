"""Tests for the WebGL-backend chooser (pxviewer.gpu).

The QtWebEngine parts can't be unit-tested headlessly, but the decision logic — mode
resolution, the software-flag fallback, remembering a verdict, and the one-shot restart
— is pure and is what actually governs behaviour. os.execv is stubbed so the "restart"
is observable without replacing the process.
"""

import pytest

from pxviewer import gpu


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """A clean env and a throwaway cache dir for every test."""
    for var in ("QTWEBENGINE_CHROMIUM_FLAGS", "PXVIEWER_GPU", "_PXVIEWER_GL_RETRIED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    gpu._STATE["autofix"] = False
    yield


def _flags(monkeypatch):
    import os
    return os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS")


def test_resolve_mode_precedence(monkeypatch):
    assert gpu.resolve_mode("software") == "software"       # explicit wins
    monkeypatch.setenv("PXVIEWER_GPU", "hardware")
    assert gpu.resolve_mode(None) == "hardware"             # env next
    monkeypatch.delenv("PXVIEWER_GPU")
    assert gpu.resolve_mode(None) == "auto"                 # default


def test_resolve_mode_rejects_garbage():
    with pytest.raises(ValueError):
        gpu.resolve_mode("turbo")


def test_custom_flags_are_left_alone(monkeypatch):
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--my-own-flags")
    assert gpu.configure("auto", log=lambda *_: None) == "custom"
    assert _flags(monkeypatch) == "--my-own-flags"
    assert not gpu.autofix_enabled()


def test_software_mode_sets_flags(monkeypatch):
    assert gpu.configure("software", log=lambda *_: None) == "software"
    assert _flags(monkeypatch) == gpu.SOFTWARE_FLAGS
    assert not gpu.autofix_enabled()


def test_hardware_mode_sets_nothing_and_does_not_arm(monkeypatch):
    assert gpu.configure("hardware", log=lambda *_: None) == "hardware"
    assert _flags(monkeypatch) is None
    assert not gpu.autofix_enabled()


def test_auto_without_cache_arms_the_check(monkeypatch):
    assert gpu.configure("auto", log=lambda *_: None) == "hardware"
    assert _flags(monkeypatch) is None
    assert gpu.autofix_enabled()  # unknown outcome -> verify after load


def test_auto_does_not_arm_after_a_retry(monkeypatch):
    monkeypatch.setenv("_PXVIEWER_GL_RETRIED", "1")
    assert gpu.configure("auto", log=lambda *_: None) == "hardware"
    assert not gpu.autofix_enabled()


def test_marking_hardware_is_remembered(monkeypatch):
    gpu.configure("auto", log=lambda *_: None)
    assert gpu.autofix_enabled()
    gpu.mark_hardware_ok()
    assert not gpu.autofix_enabled()
    # A fresh configure now trusts the remembered verdict: no check, no flags.
    assert gpu.configure("auto", log=lambda *_: None) == "hardware"
    assert not gpu.autofix_enabled()


def test_missing_webgl_remembers_software_and_restarts(monkeypatch):
    execs = {}

    def fake_execv(path, argv):
        execs["path"], execs["argv"] = path, argv

    monkeypatch.setattr(gpu.os, "execv", fake_execv)
    gpu.configure("auto", log=lambda *_: None)
    gpu.on_webgl_missing(log=lambda *_: None)

    # It restarts, marking the child so it cannot loop and forcing software next time.
    assert execs, "expected a re-exec"
    import os
    assert os.environ["_PXVIEWER_GL_RETRIED"] == "1"
    assert os.environ["PXVIEWER_GPU"] == "software"

    # And the verdict is remembered, so the *next* launch goes straight to software.
    assert gpu.configure("auto", log=lambda *_: None) == "software"
    assert _flags(monkeypatch) == gpu.SOFTWARE_FLAGS


def test_missing_webgl_is_a_noop_when_not_armed(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu.os, "execv", lambda *a: calls.append(a))
    gpu.configure("hardware", log=lambda *_: None)  # not armed
    gpu.on_webgl_missing(log=lambda *_: None)
    assert not calls, "must not restart when the check was never armed"
