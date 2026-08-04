"""A seeded random walk over the GUI, asserting the invariant bank after every step.

The cheap first cut of GUI fuzzing -- a plain seeded loop, no Hypothesis. It drives the
action surface in a random but *valid* order, and the value is entirely in what is checked
afterwards: state *combinations* no hand-written exercise reaches, with
``gui_invariants`` doing the catching.

Two walks, because they fail differently. The first drives ``DesktopApp`` methods, which
is where registry and pairing bugs live. The second clicks the real widgets, which is
where wiring bugs live -- the colour dialog reopening on OK was one, and no amount of
backend driving would have found it.

Both stay on fast synchronous actions on purpose. Minimize, tug and make_maps run on
background threads and take seconds; their races are a separate concern, covered by
``tst_gui_concurrency.py``.

Determinism: fixed seeds, both RNGs seeded, and every action appended to a log, so a
failure prints the exact sequence that produced it -- which is the minimal repro to paste
into a unit test.
"""

from __future__ import absolute_import, division, print_function

import os
import random
import sys

from pxviewer.regression.tst_utils import (
    closing_modals, data_path, dispose, have, process_events, qt_application,
    shipped_defaults, skip)

if not have("PySide6.QtWebEngineWidgets", "websockets", "numpy"):
    skip("PySide6 QtWebEngine / websockets not available")

import numpy as np                                   # noqa: E402

qt_application()

from PySide6.QtWidgets import (                     # noqa: E402
    QAbstractSlider, QCheckBox, QComboBox, QPushButton)

from pxviewer.desktop import DesktopApp             # noqa: E402
from pxviewer.regression.gui_invariants import assert_viewer_consistent   # noqa: E402

MODELS = ["1ubq.pdb", "1tec.pdb"]

#: Seeds run as a loop rather than as separate exercises: each is one sample of the same
#: question, and a failure names its seed, so splitting them would only make the output
#: longer without making a failure easier to place.
BACKEND_SEEDS = [0, 1, 2]
WIDGET_SEEDS = [0, 1]

STEPS_PER_WALK = 150
POKES_PER_WALK = 80

#: How large the walk lets a scene get before it evicts to make room (see
#: :meth:`Walk.make_room`). Chosen to match a working session rather than to be generous:
#: a few models, a few maps, and one or two paired groups is what the app is used with,
#: and holding more only makes the test heavier, not broader.
MAX_MODELS = 4
MAX_VOLUMES = 4
MAX_GROUPS = 2

#: How much of the trail to print on a failure. Enough to see how the state was built up
#: without burying the assertion that broke.
TRAIL = 15

#: Buttons that start background work -- Minimize, Stop, Add H + analyze, Build ligand.
#: They are icon-only, so they are recognised by their tooltip.
THREADED_TOOLTIPS = (
    "Minimize the active model", "Halt the run", "Add hydrogens with reduce2",
    "Build the ligand")


class Walk(object):
    """One random walk: the actions it can take, gated by what is currently loaded.

    An action returns a label, or ``None`` when its precondition was not met -- there is
    nothing to remove from an empty scene -- so the walk simply tries again rather than
    biasing itself towards whatever happens to be possible.
    """

    def __init__(self, app, rng):
        self.app = app
        self.rng = rng

    # -- helpers --------------------------------------------------------------

    def models(self):
        return list(self.app._models)

    def volumes(self):
        return list(self.app._volumes)

    def pick(self, sequence):
        return self.rng.choice(sequence) if sequence else None

    def make_room(self, entries, limit, remove):
        """Evict at random until there is room for one more, and say what went.

        The walk is weighted towards loading, so left alone it only ever grows: measured
        at 15 models and 8 generated map+model groups by the end of a single walk, around
        2 GB of them. That is not a scene anyone builds, and it is not even good fuzzing
        -- the later steps all re-test one enormous state instead of testing many
        different ones.

        Evicting rather than refusing keeps every step productive, keeps the walk moving
        through *different* small scenes, and exercises the remove paths on the way. The
        ceiling is what a real session looks like, so the footprint is too.
        """
        evicted = []
        while len(entries()) >= limit:
            before = len(entries())
            victim = self.pick(entries())
            if victim is None:                    # nothing left to evict; let it grow
                break
            remove(victim)
            if len(entries()) >= before:
                # The removal did not take. Stop rather than spin: a walk that hangs here
                # would look like a slow test rather than the bug it actually is.
                raise AssertionError(
                    "removing %r left %d entries" % (victim, len(entries())))
            evicted.append(victim)
        return " (evicted %d)" % len(evicted) if evicted else ""

    # -- actions --------------------------------------------------------------

    def load_model(self):
        room = self.make_room(
            self.models, MAX_MODELS, lambda m: self.app.remove_model(m["id"]))
        path = data_path(self.rng.choice(MODELS))
        self.app.load_file(path)
        return "load_model %s%s" % (os.path.basename(path), room)

    def load_volume(self):
        from pxviewer.volume_io import VolumeData

        room = self.make_room(
            self.volumes, MAX_VOLUMES, lambda v: self.app.remove_volume(v["id"]))
        self.app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        return "load_volume%s" % room

    def load_group(self):
        # The heaviest action by far: the density is computed, not read, and the group
        # keeps the map_model_manager that produced it.
        room = self.make_room(
            lambda: list(self.app._groups), MAX_GROUPS,
            lambda gid: self.app.remove_group(gid))
        self.app.load_map_model_demo(d_min=4.0)
        return "load_group (map+model)%s" % room

    def focus_object(self):
        objects = ([("model", m["id"]) for m in self.models()]
                   + [("volume", v["id"]) for v in self.volumes()]
                   + [("reflections", r["id"]) for r in self.app._reflections])
        picked = self.pick(objects)
        if picked is None:
            return None
        kind, ident = picked
        if kind == "model":
            self.app.set_active_model(ident)
        else:
            self.app._controls._update_appearance(kind, ident)
        return "focus %s %s" % (kind, ident)

    def toggle_visible(self):
        entry = self.pick(self.models() + self.volumes())
        if entry is None:
            return None
        wanted = not entry["visible"]
        if entry in self.models():
            self.app.set_model_visible(entry["id"], wanted)
        else:
            self.app.set_volume_visible(entry["id"], wanted)
        return "visible %s -> %s" % (entry["id"], wanted)

    def restyle_volume(self):
        volume = self.pick(self.volumes())
        if volume is None:
            return None
        vid = volume["id"]
        which = self.rng.choice(
            ["color", "iso", "opacity", "style", "clip", "radius", "mask"])
        if which == "color":
            self.app.set_volume_color(
                vid, self.rng.choice(["gold", "salmon", "#3fa9f5"]))
        elif which == "iso":
            self.app.set_volume_iso(vid, round(self.rng.uniform(0.5, 6.0), 2))
        elif which == "opacity":
            self.app.set_volume_opacity(vid, round(self.rng.uniform(0.2, 1.0), 2))
        elif which == "style":
            self.app.set_volume_style(
                vid, self.rng.choice(["surface", "wireframe", "mesh"]))
        elif which == "clip":
            front, back = sorted((self.rng.random(), self.rng.random()))
            self.app.set_volume_clip(vid, front, back)
        elif which == "radius":
            self.app.set_volume_radius(vid, self.rng.choice([None, 10.0, 20.0]))
        elif which == "mask":
            # Only valid when paired. Hitting the refusal is fine -- it is the documented
            # behaviour, and reaching it from a random state is worth doing.
            try:
                self.app.set_volume_mask(vid, self.rng.choice([None, 3.0]))
            except ValueError:
                pass
        return "restyle_volume %s %s" % (vid, which)

    def restyle_model(self):
        model = self.pick(self.models())
        if model is None:
            return None
        mid = model["id"]
        which = self.rng.choice(["rep", "color", "clip"])
        if which == "rep":
            self.app.set_model_representation(
                mid, self.rng.choice(["cartoon", "ball-and-stick"]))
        elif which == "color":
            self.app.set_model_color(
                mid, self.rng.choice([None, "element-symbol", "chain-id"]))
        elif which == "clip":
            front, back = sorted((self.rng.random(), self.rng.random()))
            self.app.set_model_clip(mid, front, back)
        return "restyle_model %s %s" % (mid, which)

    def pair(self):
        models, volumes = self.app.pairable()
        if not models or not volumes:
            return None
        self.app.pair_model_with_map(
            self.pick(models)["id"], self.pick(volumes)["id"])
        return "pair"

    def remove_object(self):
        entry = self.pick(
            self.models() + self.volumes() + list(self.app._reflections))
        if entry is None:
            return None
        if entry in self.models():
            self.app.remove_model(entry["id"])
        elif entry in self.volumes():
            self.app.remove_volume(entry["id"])
        else:
            self.app.remove_reflections(entry["id"])
        return "remove %s" % entry["id"]

    def remove_group(self):
        gid = self.pick(list(self.app._groups))
        if gid is None:
            return None
        self.app.remove_group(gid)
        return "remove_group %s" % gid

    def reset_view(self):
        self.app.reset_view()
        return "reset_view"

    #: Loading and focusing appear twice, which weights the walk towards building a scene
    #: up rather than tearing it down -- an empty scene has almost no actions available.
    ACTIONS = [
        load_model, load_model, load_volume, load_group, focus_object, focus_object,
        toggle_visible, restyle_volume, restyle_model, pair, remove_object,
        remove_group, reset_view,
    ]

    def step(self):
        return self.rng.choice(self.ACTIONS)(self)


def interactive_widgets(controls):
    """``(label, callable)`` for each enabled, visible control worth poking."""
    root = controls.widget()
    actions = []

    for button in root.findChildren(QPushButton):
        if not (button.isEnabled() and button.isVisibleTo(root)):
            continue
        if button.toolTip().startswith(THREADED_TOOLTIPS):
            continue
        # Menu buttons (Demos) trigger heavy loads the backend walk already covers, and
        # in a tight loop they just reload slow demos.
        if button.menu() is not None:
            continue
        actions.append(("click:%s" % (button.text() or button.toolTip()[:20]),
                        button.click))

    for combo in root.findChildren(QComboBox):
        if combo.isEnabled() and combo.isVisibleTo(root) and combo.count() > 1:
            actions.append(("combo", lambda c=combo: c.setCurrentIndex(
                (c.currentIndex() + 1) % c.count())))

    for check in root.findChildren(QCheckBox):
        if check.isEnabled() and check.isVisibleTo(root):
            actions.append(("check", lambda w=check: w.toggle()))

    for slider in root.findChildren(QAbstractSlider):
        if slider.isEnabled() and slider.isVisibleTo(root):
            middle = (slider.minimum() + slider.maximum()) // 2
            actions.append(("slider", lambda w=slider, v=middle: w.setValue(v)))

    return actions


def check(app, trail, seed, what):
    """Assert the bank, reporting the trail that led here rather than only the failure."""
    try:
        assert_viewer_consistent(app)
    except AssertionError as exc:
        raise AssertionError(
            "invariant broke after %d %s (seed %d):\n  %s\n-> %s"
            % (len(trail), what, seed, "\n  ".join(trail[-TRAIL:]), exc))


def desktop_for_fuzzing():
    """A running app with its controls **shown**.

    Shown deliberately: the widget invariants only bite on visible widgets, since
    orphaning a hidden one does not float it into a window. A hidden window would hide the
    very bug the stray-window check exists for.
    """
    app = DesktopApp(port=0)
    app._webapp.start()
    app._controls.widget().show()
    return app


def exercise_a_random_walk_over_the_actions_keeps_the_model_consistent():
    """Load, focus, pair, restyle, remove -- in a random but valid order."""
    for seed in BACKEND_SEEDS:
        rng = random.Random(seed)
        np.random.seed(seed)

        app = desktop_for_fuzzing()
        walk = Walk(app, rng)
        log = []
        try:
            for _ in range(STEPS_PER_WALK):
                label = walk.step()
                if label is None:
                    continue          # a precondition was not met; try again next step
                log.append(label)
                process_events()
                check(app, log, seed, "actions")
        finally:
            dispose(app)


def exercise_a_random_walk_over_the_widgets_keeps_the_model_consistent():
    """The same idea through the real controls: clicks, combos, checkboxes and sliders.

    The backend walk never touches a Qt signal; this does nothing else. Modals really open
    and are really cancelled by ``closing_modals``, so a control that opens one is
    exercised rather than skipped.
    """
    for seed in WIDGET_SEEDS:
        rng = random.Random(seed)
        np.random.seed(seed)

        app = desktop_for_fuzzing()
        try:
            # A scene with something of every kind to poke.
            app.load_file(data_path("1ubq.pdb"))
            app.load_map_model_demo(d_min=4.0)
            app.load_xray_demo(d_min=3.0)
            process_events()

            controls = app._controls
            poked = []
            for _ in range(POKES_PER_WALK):
                # Focus a random object first, so the Appearance controls exist to poke.
                items = app._loaded_summary()["items"]
                if items:
                    item = rng.choice(items)
                    if item["kind"] == "model":
                        app.set_active_model(item["id"])
                    else:
                        controls._update_appearance(item["kind"], item["id"])
                    process_events()

                actions = interactive_widgets(controls)
                if not actions:
                    continue
                label, act = rng.choice(actions)
                poked.append(label)
                act()
                process_events()
                check(app, poked, seed, "widget pokes")
        finally:
            dispose(app)


def run():
    # Both walks build a DesktopApp and the widget walk clicks whatever it finds, so the
    # preferences it reads are pinned and any dialog it opens is cancelled rather than
    # left blocking.
    with shipped_defaults(), closing_modals():
        for name, fn in sorted(globals().items()):
            if name.startswith("exercise"):
                print("  %s" % name)
                sys.stdout.flush()
                fn()
    print("OK")


if __name__ == "__main__":
    run()
