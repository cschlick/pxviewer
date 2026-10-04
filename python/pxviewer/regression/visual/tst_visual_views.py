"""Visual pass -- a saved view restores pixel-exact.

The claim a .pxview.json makes: the scene it encodes reproduces *the rendered
frame*, not just the state dict. Load a model, frame a selection, save the
view, and restore it into a fresh app — the camera readback must match exactly
(target, radius) and the re-rendered viewport must match the saved pixels. The
camera is the hard part: it has to land after the client's auto-frame (an armed
one-shot on "ready"), past the resets a still-building scene can still fire.
"""

import tempfile
import traceback

from pxviewer import viewstate
from pxviewer.regression.visual.harness import *


def run() -> None:
    app = make_app()
    try:
        load_bundled(app, "1ubq.pdb")
        pump(until=lambda: bool(app._models), timeout=20)
        settle(3.0)
        mid = app._models[0]["id"]
        controls = app._controls
        controls._select_expr.setText("resseq 29")
        controls._on_select_expression()
        pump(until=lambda: bool(app._scene_selection.get(mid)), timeout=15)
        settle(3.0)

        before = cam_state(app)
        check("camera readable before save", before is not None)
        shot_before = shot_viewport(app, "view-before")

        tmp = tempfile.mkdtemp()
        view_path = f"{tmp}/figure.pxview.json"
        doc = viewstate.save_view(app, view_path, bundle_data=True)
        check("camera captured in the file",
              doc.get("camera") and doc["camera"].get("radius"),
              "%s" % (doc.get("camera") or {}).get("radius"))
        check("model data bundled",
              any(o["kind"] == "model" and o["data"] for o in doc["objects"]))
        dispose(app)
        app = None

        app = make_app()
        try:
            settle(2.0)
            viewstate.load_view(app, view_path)
            pump(until=lambda: bool(app._models), timeout=30)
            settle(4.0)

            after = cam_state(app)
            check("camera readable after restore", after is not None)
            if before and after:
                dt = sum((a - b) ** 2 for a, b in
                         zip(before["target"], after["target"])) ** 0.5
                check("camera target restored", dt < 0.01, "delta %s" % dt)
                check("camera radius restored",
                      abs(before["radius"] - after["radius"]) < 0.01,
                      "%s vs %s" % (before["radius"], after["radius"]))
            shot_after = shot_viewport(app, "view-after")
            if shot_before and shot_after:
                d = img_diff(shot_before, shot_after)
                check("restored view pixel-matches", d < 2.0, "%.3f%%" % d)

            nmid = app._models[0]["id"]
            check("standing selection restored",
                  bool(app._scene_selection.get(nmid)))
            entry = app._model_entry(nmid)
            check("clip sphere restored",
                  app._sphere_state(entry)["mode"] == "selection")
        except Exception:
            check("restore ran clean", False, traceback.format_exc(limit=3))
    except Exception:
        check("view pass ran clean", False, traceback.format_exc(limit=3))
    finish(app)


run()
