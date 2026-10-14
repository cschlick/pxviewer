"""Regenerate a paper figure from a saved view (.pxview.json).

Usage::

    python regen_figure.py <view.pxview.json> <out.png> [--controls]

Loads the view into the real desktop app, waits for the Mol* page to come up
and the saved camera to land, then writes the browser-side render to
``out.png``. ``--controls`` composites a Qt grab of the controls window to
the left of the viewport, for full-GUI shots like the paper's Figure 1.

The viewport image comes from ``session.screenshot()`` — ``QWidget.grab()``
on a QWebEngineView returns a blank frame, so the browser-side capture is the
only faithful one (this is the same reason the visual harness has
``shot_viewport``). The controls window is a plain Qt widget, where grab is
fine, so the composite uses each capture path where it works.

Run with the pxviewer environment's python. Real windows open briefly —
``make_app`` uses the native platform because an offscreen WebEngine renders
nothing to photograph.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

from pxviewer import viewstate
from pxviewer.regression.visual.harness import (
    _call_session, dispose, make_app, pump, settle, wait_viewer)


def _composite(app, viewport_png: bytes) -> bytes:
    """Controls window (Qt grab) on the left, viewport render on the right."""
    from PIL import Image

    ctrl_path = Path(tempfile.mktemp(suffix=".png"))
    app._controls.widget().grab().save(str(ctrl_path))
    controls = Image.open(ctrl_path).convert("RGB")
    view = Image.open(io.BytesIO(viewport_png)).convert("RGB")
    height = view.height
    controls = controls.resize(
        (round(controls.width * height / controls.height), height))
    out = Image.new("RGB", (controls.width + view.width, height), "white")
    out.paste(controls, (0, 0))
    out.paste(view, (controls.width, 0))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def capture(view_path: str, out_png: str, *, with_controls: bool = False,
            settle_s: float = 5.0) -> Path:
    app = make_app()
    try:
        viewstate.load_view(app, view_path)
        pump(until=lambda: bool(app._models) or bool(app._volumes),
             timeout=60)
        # The restore reloads the page; the saved camera is an armed one-shot
        # delivered on the client's "ready", so wait for the page then settle.
        ok, _ = wait_viewer(app)
        if not ok:
            raise RuntimeError("viewport never came up")
        settle(settle_s)
        png = _call_session(app, "screenshot", 30)
        if not png:
            raise RuntimeError("screenshot came back empty")
        if with_controls:
            png = _composite(app, png)
        out = Path(out_png)
        out.write_bytes(png)
        return out
    finally:
        dispose(app)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    out = capture(args[0], args[1], with_controls="--controls" in argv)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
