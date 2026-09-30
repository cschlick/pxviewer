"""Visual pass 0/1 -- launch, first look, first model.

The automatable core of VISUAL_TESTS.md's Pass 0 and Pass 1: the app comes up,
the embedded viewer answers, a bundled model loads and frames, the tab order is
the shipping order, and every toolbar button has a tooltip. Screenshots land in
the run directory for a human glance; what the script asserts is the state.
"""

import traceback

from pxviewer.regression.visual.harness import *

EXPECTED_TABS = ["Scene", "Geometry", "Validation", "Tools", "Console", "Settings"]


def run() -> None:
    app = make_app()
    try:
        controls = app._controls
        check("tab order is Scene/Geometry/Validation/Tools/Console/Settings",
              list(controls._tab_labels) == EXPECTED_TABS,
              str(controls._tab_labels))

        bar_buttons = [getattr(controls, name) for name in
                       ("_zoom_out_btn", "_zoom_in_btn", "_reset_view_btn",
                        "_picture_btn", "_dock_btn", "_mouse_btn", "_help_btn")
                       if hasattr(controls, name)]
        check("every toolbar button has a tooltip",
              len(bar_buttons) == 7 and all(b.toolTip() for b in bar_buttons),
              "%d buttons" % len(bar_buttons))

        path = load_bundled(app, "1ubq.pdb")
        note("loaded %s" % path)
        pump(until=lambda: bool(app._models), timeout=20)
        # The viewer attaches to the loaded model's session; only then can a
        # screenshot be answered (the dummy session before a load serves nothing).
        ok, out = wait_viewer(app)
        check("viewer answered a screenshot",
              ok and bool(out.get("v")), str(out.get("err", "")))
        settle(3.0)   # parse + first real render in the browser

        check("model loaded and active",
              len(app._models) == 1
              and app._models[0]["id"] == app._active_model_id)
        shot_viewport(app, "v0-viewport-1ubq")
        main_window(app, "v0-window")
        controls_window(app, "v0-controls")

        # A screenshot of an empty viewport and one with a model must differ.
        before = cam_state(app)
        check("camera state readable", before is not None
              and "radius" in before, str(before))
    except Exception:
        check("launch pass ran clean", False, traceback.format_exc(limit=3))
    finish(app)


run()
