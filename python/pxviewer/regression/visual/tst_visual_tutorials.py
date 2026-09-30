"""Visual pass 7 -- every tutorial starts, walks, and exits clean.

A smoke sweep over ``tutorial.all_tutorials()`` in the real rendered app: the
menu order matches the registry, each tutorial loads its own example (bundled
or demo data; the three fetch-dependent ones run only under
``PXVIEWER_VISUAL_NET=1``), the coach pane opens at "Step 1 / N" with real
text, and walking every step with Next/Skip reaches the end with the layout
restored. Per-step *actions* (what each step actually teaches) stay manual in
VISUAL_TESTS.md -- this sweep catches broken loaders, missing controls, wedged
advances, and exit-state leaks.
"""

import os
import traceback

from pxviewer.regression.tst_utils import closing_modals
from pxviewer.regression.visual.harness import *

from pxviewer import tutorial


#: Tutorials whose loader fetches real data (PDB/EMDB, ~160 MB for localres).
#: Offline by default; opt in with PXVIEWER_VISUAL_NET=1.
FETCH_TUTORIALS = {
    "A model with its map",
    "Real-space refine into cryo-EM density",
    "Look at local resolution",
}


def run() -> None:
    app = make_app()
    try:
        controls = app._controls
        vp = app._viewport
        tuts = tutorial.all_tutorials()

        check("tutorial registry is ten strong", len(tuts) == 10,
              str([t.title for t in tuts]))

        for tut in tuts:
            section(tut.title)
            slug = "".join(c if c.isalnum() else "-" for c in tut.title.lower())[:40]
            if tut.title in FETCH_TUTORIALS and not os.environ.get(
                    "PXVIEWER_VISUAL_NET"):
                note("skipped: fetches real data (PXVIEWER_VISUAL_NET=1 to run)")
                continue

            # An empty scene means no "clear or keep" dialog; close any leftover.
            app._clear_all()
            settle(1.0)

            with closing_modals():   # a loader failure's warning must not hang the run
                controls._start_tutorial(tut)
                pump(until=lambda: controls._tutorial is not None
                     or bool(app._models) or bool(app._volumes), timeout=10)
                pump(until=lambda: bool(app._models) or bool(app._volumes),
                     timeout=120)    # demo loaders compute on workers
                settle(3.0)

            if not check("%s: started" % tut.title,
                         controls._tutorial is not None):
                continue
            check("%s: coach shows step 1 of %d" % (tut.title, len(tut.steps)),
                  not vp.coach_bar.isHidden()
                  and vp.coach_title.text() == tut.title
                  and vp.coach_progress.text() == "Step 1 / %d" % len(tut.steps)
                  and bool(vp.coach_text.text().strip()),
                  vp.coach_progress.text())
            check("%s: coach height reserved" % tut.title,
                  vp.coach_text.minimumHeight() > 0,
                  str(vp.coach_text.minimumHeight()))
            shot_viewport(app, "v7-%s-start" % slug)
            main_window(app, "v7-%s-window" % slug)

            # Walk to the end: the 400 ms poll auto-advances satisfied steps;
            # Next/Skip carries the rest. Guarded so a wedged step is a FAIL,
            # not a hang.
            seen_steps = set()
            guard = 3 * len(tut.steps) + 2
            while controls._tutorial is not None and guard > 0:
                seen_steps.add(controls._tutorial_step)
                settle(0.6)
                if controls._tutorial is None:
                    break
                controls._tutorial_next()
                guard -= 1
            check("%s: walked all %d steps to the end" % (tut.title, len(tut.steps)),
                  controls._tutorial is None and seen_steps,
                  "visited %s" % sorted(seen_steps))
            check("%s: coach put away" % tut.title,
                  vp.coach_bar.isHidden()
                  and vp.coach_text.minimumHeight() == 0)
            shot_viewport(app, "v7-%s-after" % slug)

        check("no tutorial left running", controls._tutorial is None)
    except Exception:
        check("tutorials pass ran clean", False, traceback.format_exc(limit=3))
    finish(app)


run()
