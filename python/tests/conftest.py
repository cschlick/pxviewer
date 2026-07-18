"""Shared pytest configuration.

The Qt-backed tests (desktop app, console, GUI fuzzers) need a platform plugin. Default
to ``offscreen`` where there is no display — it needs neither a display nor a GPU, and is
where the suite is proven to pass headless. Where a display is present, leave Qt to pick
the native platform, so the tests exercise the same GPU path the app would.

No WebGL/GPU flags are forced: a machine with a working GPU is never pushed onto software
rendering, and the viewports the GUI tests build are disposed on teardown (see
``ViewportWindow.close``) so their render processes do not pile up. Set ``QT_QPA_PLATFORM``
or ``QTWEBENGINE_CHROMIUM_FLAGS`` yourself to override either choice.

Applied before any test imports PySide6 — pytest imports a top-level conftest before
collection, which the platform choice requires.
"""

import os

if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
