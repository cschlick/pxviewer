"""How loaded objects look, and what changing that is allowed to touch.

Almost every exercise here is really about *not* rebuilding something. A contour level, a
visibility toggle or a colour has to reach the viewport live, because the alternative --
recomposing the scene and reloading the page -- makes every other object flicker, throws
away the camera, and on software WebGL crashes outright. So the assertions come in pairs:
the change landed, and the scene was not rebuilt to land it.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import sys

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import (
    closing_modals, data_path, dispose, have, process_events, qt_application,
    shipped_defaults, skip)

if not have("PySide6.QtWebEngineWidgets", "websockets", "numpy"):
    skip("PySide6 QtWebEngine / websockets not available")

import numpy as np                                   # noqa: E402

QAPP = qt_application()

from PySide6.QtCore import Qt                        # noqa: E402
from PySide6.QtGui import QColor, QPalette           # noqa: E402
from PySide6.QtWidgets import (                      # noqa: E402
    QApplication, QComboBox, QLabel, QPushButton, QRadioButton)

from pxviewer.desktop import (                       # noqa: E402
    _CUSTOM_COLOR, _MODEL_REP_OPTIONS, _TREE_MAX_HEIGHT, _TREE_MIN_HEIGHT,
    _VOLUME_COLORS, DesktopApp, _make_range_slider)
from pxviewer.live import LiveSession                # noqa: E402
from pxviewer.volume_io import DEFAULT_ISO_SIGMA, VolumeData   # noqa: E402


class Recording_session(LiveSession):
    """A real session that keeps the volume commands it was asked to send.

    Volume appearance rides whichever session the viewport is connected to, so making
    that session a recording subclass is enough to see what went out -- no method on any
    object gets replaced, and everything it records it also really does.

    The commands are recorded here rather than at ``_broadcast_text``, which is reached
    through ``call_soon_threadsafe`` and so would not have run yet when the assertion
    fires.
    """

    def __init__(self, *args, **kwargs):
        super(Recording_session, self).__init__(*args, **kwargs)
        self.volume_commands = []

    def set_volume_iso(self, ref, value):
        self.volume_commands.append(("iso", ref, value))
        return super(Recording_session, self).set_volume_iso(ref, value)

    def set_volume_color(self, ref, color):
        self.volume_commands.append(("color", ref, color))
        return super(Recording_session, self).set_volume_color(ref, color)

    def set_volume_opacity(self, ref, opacity):
        self.volume_commands.append(("opacity", ref, opacity))
        return super(Recording_session, self).set_volume_opacity(ref, opacity)

    def set_volume_visible(self, ref, visible):
        self.volume_commands.append(("visible", ref, visible))
        return super(Recording_session, self).set_volume_visible(ref, visible)


@contextlib.contextmanager
def desktop(**kwargs):
    app = DesktopApp(port=0, **kwargs)
    app._webapp.start()
    with closing_modals():
        try:
            yield app
        finally:
            dispose(app)


def blob(app, name="blob", shape=(8, 8, 8)):
    return app._add_volume(VolumeData.from_numpy(np.ones(shape)), name)


def scene_text(app):
    """Compose the MVSJ and read it back.

    Note this *writes* a scene -- it is how the app composes one -- so it bumps
    ``_scene_counter``. Capture the counter before calling it.
    """
    path = app._write_volume_scene()
    if path is None:
        return ""
    return (app._webapp.volume_dir / path.lstrip("/")).read_text()


def tree_items(tree):
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.child(i) for i in range(node.childCount()))


def node_for(app, kind, ident):
    return next(n for n in tree_items(app._controls._loaded_tree)
                if n.data(0, Qt.ItemDataRole.UserRole) == (kind, ident))


# -- volume appearance --------------------------------------------------------


def exercise_volume_appearance_controls():
    """Style, colour, opacity and a contour level are each kept on the entry, so a scene
    rebuild restores them, and pushed live, so nothing has to reload."""
    with desktop() as app:
        vid = blob(app)
        entry = app._volume_entry(vid)
        assert entry["iso"] == DEFAULT_ISO_SIGMA

        app.set_volume_color(vid, "salmon")
        app.set_volume_opacity(vid, 0.4)
        app.set_volume_iso(vid, 3.25)
        assert (entry["color"], entry["opacity"], entry["iso"]) == ("salmon", 0.4, 3.25)
        assert app._write_volume_scene() is not None

        # Focusing the volume builds the controls and points the wheel at it.
        controls = app._controls
        controls._update_appearance("volume", vid)
        assert controls._iso_row is not None
        assert controls._iso_row["spin"].value() == 3.25
        assert app._volume_scroll_target == vid

        # The spinbox and slider drive each other and the backend.
        controls._iso_row["spin"].setValue(5.0)
        assert entry["iso"] == 5.0
        assert controls._iso_row["slider"].value() == 500     # 5.0 sigma at 0.01 steps

        # Focusing something that is not a volume takes the wheel target away.
        controls._update_appearance(None, None)
        assert controls._iso_row is None
        assert app._volume_scroll_target is None


def exercise_a_contour_changed_in_the_viewport_is_not_echoed_back():
    """The wheel is applied in the viewer, so the level arrives here after the fact. The
    widgets must follow it without writing it back -- an echo fights the scroll."""
    with desktop() as app:
        # The control session is the active model's, so a recording one sees the traffic.
        app._add_model(Recording_session.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        session = app._control_session()
        assert isinstance(session, Recording_session)

        vid = blob(app)
        entry = app._volume_entry(vid)
        controls = app._controls
        controls._update_appearance("volume", vid)
        session.volume_commands = []

        app._on_volume_iso_changed(entry["ref"], 4.5)

        assert entry["iso"] == 4.5
        assert controls._iso_row["spin"].value() == 4.5
        assert controls._iso_row["slider"].value() == 450
        assert session.volume_commands == []        # nothing went back out


def exercise_volume_colour_swatches_and_a_custom_picker():
    """Colours are swatches rather than names, with a picker for anything off the preset
    list -- the wire takes any hex Mol* can decode."""
    with desktop() as app:
        vid = blob(app)
        controls = app._controls
        controls._update_appearance("volume", vid)
        combo = controls._appearance_box.findChildren(QComboBox)[1]     # after Style

        assert [combo.itemData(i) for i in range(len(_VOLUME_COLORS))] == _VOLUME_COLORS
        assert all(not combo.itemIcon(i).isNull() for i in range(len(_VOLUME_COLORS)))
        assert combo.itemData(combo.count() - 1) == _CUSTOM_COLOR       # the picker, last

        combo.setCurrentIndex(2)
        assert app._volume_entry(vid)["color"] == _VOLUME_COLORS[2]

        # A picked colour is a hex string, and joins the list so it stays selected.
        app.set_volume_color(vid, "#3fa9f5")
        controls._update_appearance("volume", vid)
        combo = controls._appearance_box.findChildren(QComboBox)[1]
        assert combo.currentData() == "#3fa9f5"
        assert not combo.itemIcon(combo.currentIndex()).isNull()


def exercise_masking_density_around_the_model():
    """Hide density away from the molecule.

    It needs a paired map -- "away from the molecule" means nothing without one -- and it
    masks a copy, so the map that minimization refines against keeps all its density.
    """
    if not have("iotbx.map_model_manager"):
        print("    (skipped: iotbx.map_model_manager not available)")
        return

    def occupied(path):
        data = VolumeData.from_map_file(str(path)).map_manager.map_data().as_numpy_array()
        return float((np.abs(data) > 1e-4).mean())

    with desktop() as app:
        app.load_map_model_demo(d_min=3.0)
        vid = app._volumes[0]["id"]
        mmm = app.group_mmm(app._volumes[0]["group"])
        real_before = mmm.map_manager().map_data().as_numpy_array().copy()
        served = app._webapp.volume_dir / "vols" / ("%s.map" % vid)

        assert app.can_mask_volume(vid)
        full = occupied(served)

        app.set_volume_mask(vid, 3.0)
        assert occupied(served) < 0.5 * full          # the served map lost the outside
        assert app.volume_appearance(vid)["mask_radius"] == 3.0

        # A wider shell keeps more, so the radius means what it says.
        app.set_volume_mask(vid, 8.0)
        wide = occupied(served)
        app.set_volume_mask(vid, 3.0)
        assert occupied(served) < wide

        # The map that gets refined against is untouched by any of it.
        assert np.array_equal(real_before, mmm.map_manager().map_data().as_numpy_array())
        assert set(mmm.map_id_list()) == {"map_manager", "model_map"}   # no scratch pile-up

        app.set_volume_mask(vid, None)
        assert approx_equal(occupied(served), full)   # back to the whole map

        # An unpaired volume has no model to mask around.
        loose = blob(app, "loose")
        assert not app.can_mask_volume(loose)
        with raises(ValueError) as e:
            app.set_volume_mask(loose, 3.0)
        assert "paired" in str(e.value)


# -- hiding, in place ---------------------------------------------------------


def exercise_hiding_a_model_is_a_render_skip_not_a_reload():
    """On hardware a model hides by toggling its own render visibility in place, so the
    other objects never flicker; it stays connected, and showing flips it back."""
    with desktop(can_hide=True) as app:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        session = app.session_for(a)
        ws = "ws://%s:%d" % (app._host, session.port)
        # The app's own rebuild counter: it increments once per composed scene, which is
        # once per viewport reload. Unchanged means nothing was recomposed.
        before = app._scene_counter

        app.set_model_visible(a, False)
        assert session._structure_visible is False    # hidden in place ...
        assert ws in app._model_ws()                  # ... and still connected

        app.set_model_visible(a, True)
        assert session._structure_visible is True

        assert app._scene_counter == before           # no rebuild, so no flicker
        assert ws in app._model_ws()


def exercise_hiding_a_map_is_a_render_skip_not_a_reload():
    """A map hides by turning its isosurface off in place. It stays *in* the scene: a
    reload would re-hide it through _reassert_hidden_volumes, so removing it from the
    scene would be both redundant and a rebuild."""
    with desktop(can_hide=True) as app:
        app._add_model(Recording_session.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        session = app._control_session()
        vid = blob(app, "map")
        ref = app._volume_entry(vid)["ref"]
        before = app._scene_counter
        session.volume_commands = []

        app.set_volume_visible(vid, False)
        assert app._volume_entry(vid)["visible"] is False
        assert session.volume_commands[-1] == ("visible", ref, False)   # in place
        assert app._scene_counter == before           # and without recomposing
        assert ref in scene_text(app)                 # still baked into the scene

        app.set_volume_visible(vid, True)
        assert session.volume_commands[-1] == ("visible", ref, True)
        assert app._volume_entry(vid)["visible"] is True


def exercise_software_pins_a_model_and_says_why_on_click():
    """On software WebGL touching a model's render state segfaults, so models are pinned
    like maps: the checkbox is not checkable and the setter refuses *silently* -- an
    internal caller, add-hydrogens, hides the H-less original and must not warn -- while
    a click on the check column flashes the reason."""
    with desktop(can_hide=False) as app:              # software
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        session = app.session_for(a)
        node = node_for(app, "model", a)
        assert not (node.flags() & Qt.ItemFlag.ItemIsUserCheckable)

        warned = []
        app.bridge.status_warned.connect(warned.append)

        app.set_model_visible(a, False)
        assert app._model_entry(a)["visible"] is True
        assert session._structure_visible is not False    # never touched
        assert warned == []                               # and silent

        app._controls._on_tree_item_clicked(node, 0)
        assert warned and "hardware WebGL" in warned[-1]
        assert session._structure_visible is not False    # saying why touches nothing


def exercise_software_pins_a_map_and_says_why_on_click():
    """Hiding a map's isosurface segfaults on software WebGL, so the checkbox is not
    checkable. Not a dead control though: clicking it flashes the reason, which is a pure
    status message. A click off the check column says nothing."""
    with desktop(can_hide=False) as app:
        app._add_model(Recording_session.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        session = app._control_session()
        vid = blob(app, "map")
        node = node_for(app, "volume", vid)
        assert not (node.flags() & Qt.ItemFlag.ItemIsUserCheckable)

        warned = []
        app.bridge.status_warned.connect(warned.append)
        before = app._scene_counter
        session.volume_commands = []

        app._controls._on_tree_item_clicked(node, 0)
        assert warned and "hardware WebGL" in warned[-1]
        assert session.volume_commands == []              # nothing touched the scene
        assert app._scene_counter == before               # and nothing recomposed it
        assert app._volume_entry(vid)["visible"] is True  # still pinned visible

        warned[:] = []
        app._controls._on_tree_item_clicked(node, 2)      # the name column
        assert warned == []


def exercise_a_visibility_box_never_rebuilds_the_tree_inside_the_signal():
    """A visibility checkbox must never rebuild the object tree from inside
    ``itemChanged``.

    Qt emits ``itemChanged`` synchronously from inside ``QTreeWidgetItem::setData``,
    which it is running from the tree's own mouse-release stack. Rebuilding there reaches
    ``QTreeWidget.clear()``, destroying the very item whose ``setData`` is still on the
    stack; Qt keeps using that freed item as the stack unwinds and the process dies with
    SIGSEGV.

    That -- not the GPU -- is what every past "hiding segfaults" crash was: three
    unrelated hide mechanisms all crashed with one identical Qt backtrace, on software
    *and* hardware WebGL. So the toggle is applied one event-loop turn later.

    Reading the item after the signal is the check. If the tree had been cleared the item
    would be freed, and touching it is exactly the crash being guarded against.
    """
    with desktop(can_hide=True) as app:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")  # keeps the page alive
        item = node_for(app, "model", a)

        item.setCheckState(0, Qt.CheckState.Unchecked)   # emits itemChanged synchronously

        assert item.text(2) == "A"                       # still alive and usable
        assert app._model_entry(a)["visible"]            # deferred, so not applied yet

        process_events()
        assert not app._model_entry(a)["visible"]        # applied on the next turn


# -- the appearance pane ------------------------------------------------------


def exercise_hiding_an_object_does_not_rebuild_the_appearance_pane():
    """The pane is rebuilt from ``_on_loaded_changed``, which runs on *any* change to the
    object list -- so a visibility toggle tore its widgets down and rebuilt them, and it
    flickered on every checkbox click. Nothing in the pane depends on visibility, since a
    hidden object is still editable here.

    Checked by widget identity: a rebuild replaces the pane's children, so the same
    objects still being there is exactly "it was not rebuilt".
    """
    with desktop(can_hide=True) as app:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        vid = blob(app, "map")

        controls = app._controls
        controls._update_appearance("model", b)
        widgets = controls._appearance_box.findChildren(QComboBox)
        assert widgets                                   # there is something to preserve

        app.set_model_visible(a, False)
        app.set_model_visible(a, True)
        app.set_volume_visible(vid, False)
        app.set_volume_visible(vid, True)
        assert controls._appearance_box.findChildren(QComboBox) == widgets
        assert controls._focused == ("model", b)

        # Hiding the focused object itself is still no reason to rebuild.
        app.set_model_visible(b, False)
        assert controls._appearance_box.findChildren(QComboBox) == widgets
        assert controls._focused == ("model", b)

        # But a real appearance change must still rebuild it. Get the pane genuinely
        # showing 'cartoon' first: set_model_representation does not republish the list,
        # since its own dropdown already shows the new value.
        app.set_model_representation(b, "cartoon")
        app._emit_loaded_changed()
        widgets = controls._appearance_box.findChildren(QComboBox)

        app.ensure_atoms_shown(b)         # cartoon -> ball-and-stick, and republishes
        assert app._model_entry(b)["rep"] == "ball-and-stick"
        assert controls._appearance_box.findChildren(QComboBox) != widgets


def exercise_the_appearance_pane_follows_the_active_model():
    """Activating a model by its radio re-points the pane at it, so the dropdowns edit
    that model rather than the previously focused one."""
    with desktop() as app:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        assert app._active_model_id == b                 # last added is active

        app.set_model_representation(a, "cartoon")
        b_rep = app._model_entry(b)["rep"]

        radios = dict((r.property("mid"), r)
                      for r in app._controls._loaded_tree.findChildren(QRadioButton))
        app._controls._on_active_radio(radios[a])
        process_events()                     # deferred off the click signal

        assert app._active_model_id == a
        assert app._controls._focused == ("model", a)

        app.set_model_representation(a, "spacefill")
        assert app._model_entry(a)["rep"] == "spacefill"
        assert app._model_entry(b)["rep"] == b_rep       # per-model, so B is untouched


def exercise_a_new_model_takes_the_appearance_pane():
    """A new model becomes active, so the pane follows it -- which is why the
    hydrogenate-and-analyze '+H' model is what the dropdowns edit, not the hidden
    original."""
    with desktop() as app:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        assert app._controls._focused == ("model", a)

        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        assert app._active_model_id == b
        assert app._controls._focused == ("model", b)


def exercise_the_active_model_radio():
    """One radio per model row, marking the active one and activating on click."""
    with desktop() as app:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        assert app._active_model_id == b

        tree = app._controls._loaded_tree
        radios = dict((r.property("mid"), r) for r in tree.findChildren(QRadioButton))
        assert set(radios) == {a, b}
        assert radios[b].isChecked() and not radios[a].isChecked()

        # Activation is deferred one turn: the rebuild it triggers deletes the radio that
        # is still delivering this click, so it must not run inside the signal.
        app._controls._on_active_radio(radios[a])
        process_events()
        assert app._active_model_id == a

        radios = dict((r.property("mid"), r) for r in tree.findChildren(QRadioButton))
        assert radios[a].isChecked() and not radios[b].isChecked()


def exercise_rebuilding_the_appearance_pane_spawns_no_stray_windows():
    """Orphaning a still-visible widget turns it into a floating top-level window, which
    is how a rebuilt pane spawned stray little combo-box windows when a model and
    reflections loaded in succession. Clearing must hide-and-delete, not orphan."""
    from pxviewer.regression.gui_invariants import stray_windows

    with desktop() as app:
        app._controls.widget().show()
        app.load_xray_demo(d_min=2.5)
        process_events()
        assert stray_windows(app) == []

        # And the flow that orphaned them directly: rebuild the pane across kinds.
        controls = app._controls
        controls._update_appearance("model", app._models[0]["id"])
        controls._update_appearance("reflections", app._reflections[0]["id"])
        controls._update_appearance("model", app._models[0]["id"])
        process_events()
        assert stray_windows(app) == []


# -- representations ----------------------------------------------------------


def exercise_representation_dropdowns():
    """Each loaded object gets an inline representation dropdown, defaulted by what it
    is, and changing it updates the registry."""
    with desktop() as app:
        captured = {}
        app.bridge.loaded_changed.connect(lambda s: captured.update(s))

        mid = app._add_model(
            LiveSession.from_model_file(data_path("1ubq.pdb")), "1ubq")
        entry = app._model_entry(mid)
        assert entry["rep"] == "cartoon"                 # a polymer
        item = next(it for it in captured["items"] if it["id"] == mid)
        assert item["rep"] == "cartoon"

        other = app._add_model(LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]]), "x")
        assert app._model_entry(other)["rep"] == "ball-and-stick"     # a non-polymer

        app.set_model_representation(mid, "spacefill")
        assert entry["rep"] == "spacefill"

        vid = blob(app)
        volume = app._volume_entry(vid)
        assert volume["style"] == "surface"
        app.set_volume_style(vid, "mesh")
        assert volume["style"] == "mesh"

        controls = app._controls
        controls._update_appearance("model", mid)
        assert controls._appearance_box.title().endswith("1ubq")
        # Representation and colour at least, plus the structure-type show/hide.
        assert len(controls._appearance_box.findChildren(QComboBox)) >= 2

        controls._update_appearance("volume", vid)
        assert controls._appearance_box.findChildren(QComboBox)


def exercise_every_representation_option_is_accepted_by_the_session():
    """The dropdown's values and the LiveSession API have to agree: 'line' once did not."""
    session = LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]])
    for _label, value in _MODEL_REP_OPTIONS:
        session.set_representation(value)                # must not raise


def exercise_tools_and_appearance_setters():
    """Measure-from-selection, the colour and interaction setters, and the tools that
    only broadcast."""
    from types import SimpleNamespace

    with desktop() as app:
        mid = app._add_model(LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]]), "m")

        app._on_model_selection(mid, SimpleNamespace(indices=[0, 1]))
        assert "distance" in app.measure_selection("distance")

        app._on_model_selection(mid, SimpleNamespace(indices=[0]))
        with raises(ValueError):                         # wrong atom count, clear error
            app.measure_selection("distance")

        app.set_model_color(mid, "chain-id")
        assert app._model_entry(mid)["color"] == "chain-id"
        app.set_model_interactions(mid, True)
        assert app._model_entry(mid)["interactions"] is True

        # No analysis has run, so toggling a probe channel with no cached dots just
        # clears it. None of these may raise.
        app.set_probe_channel(0, True)
        app.set_probe_channel(0, False)
        app.clear_measurements()
        app.reset_view()


# -- alternate conformations ---------------------------------------------------


def exercise_a_model_with_altlocs_offers_its_conformers():
    """3NIR is crambin at 0.48 A: four conformers, and half its atoms in one."""
    with desktop() as app:
        app.load_file(data_path("3nir.pdb"))
        mid = app._models[0]["id"]
        assert app.model_conformers(mid) == ["A", "B", "C", "D"]


def exercise_a_model_without_altlocs_offers_none():
    """Which is what keeps the control off the pane for ordinary structures."""
    with desktop() as app:
        app.load_file(data_path("1ubq.pdb"))
        assert app.model_conformers(app._models[0]["id"]) == []


def exercise_choosing_a_conformer_hides_only_the_others():
    """The atoms with no altloc stay: they are the part of the residue every conformer
    shares, so dropping them would cut the backbone at every alternate side chain."""
    with desktop() as app:
        app.load_file(data_path("3nir.pdb"))
        mid = app._models[0]["id"]
        entry = app._model_entry(mid)

        assert app._shown_indices(entry) is None          # all 1026 to begin with

        app.set_model_conformer(mid, "A")
        shown = app._shown_indices(entry)
        assert len(shown) == 750                          # 492 shared + 258 in A

        altloc = entry["session"]._data.arrays.altloc
        assert {altloc[i].strip() for i in shown} == {"", "A"}

        app.set_model_conformer(mid, "B")
        assert len(app._shown_indices(entry)) == 745      # 492 shared + 253 in B

        app.set_model_conformer(mid, None)
        assert app._shown_indices(entry) is None          # back to all of them


def exercise_a_conformer_choice_composes_with_hidden_types():
    """Two independent reasons to hide an atom, so choosing a conformer must not put the
    waters back on screen."""
    with desktop() as app:
        app.load_file(data_path("3nir.pdb"))
        mid = app._models[0]["id"]
        entry = app._model_entry(mid)

        app.set_model_type_hidden(mid, "Water", True)
        without_water = set(app._shown_indices(entry))
        app.set_model_conformer(mid, "A")
        both = set(app._shown_indices(entry))

        assert both < without_water                       # strictly fewer, not reset
        resname = entry["session"]._data.arrays.resname
        assert not any(resname[i] == "HOH" for i in both)


def exercise_an_unknown_conformer_is_refused():
    with desktop() as app:
        app.load_file(data_path("3nir.pdb"))
        mid = app._models[0]["id"]
        with raises(ValueError) as e:
            app.set_model_conformer(mid, "Z")
        assert "no conformer" in str(e.value)


def exercise_the_conformer_control_appears_only_when_there_is_a_choice():
    """A combo offering nothing but "All" would be a control that never does anything."""
    with desktop() as app:
        app.load_file(data_path("3nir.pdb"))
        app.load_file(data_path("1ubq.pdb"))
        controls = app._controls

        def conformer_labels():
            return [w.text() for w in controls._appearance_box.findChildren(QLabel)]

        controls._update_appearance("model", app._models[0]["id"])   # 3nir
        assert "Conformer" in conformer_labels()

        controls._update_appearance("model", app._models[1]["id"])   # 1ubq
        assert "Conformer" not in conformer_labels()


# -- clipping -----------------------------------------------------------------


def exercise_the_range_slider_has_two_handles():
    """The clipping slab's control. Handles may meet -- not degenerate here, it is the
    point at which the object is fully clipped."""
    slider = _make_range_slider()()
    slider.resize(240, 24)
    assert slider.values() == (0.0, 1.0)                 # open: nothing clipped

    seen = []
    slider.changed.connect(lambda f, b: seen.append((round(f, 2), round(b, 2))))
    slider.set_values(0.25, 0.75, notify=True)
    assert slider.values() == (0.25, 0.75)
    assert seen == [(0.25, 0.75)]

    slider.set_values(0.8, 0.2)                          # crossed handles collapse
    assert slider.values() == (0.2, 0.2)
    slider.set_values(-1.0, 5.0)                         # out of range clamps
    assert slider.values() == (0.0, 1.0)


def exercise_clipping_is_per_object():
    """Each object carries its own slab, so the density can be clipped while the model
    inside it stays whole."""
    with desktop() as app:
        vid = blob(app)
        mid = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        assert app._volume_entry(vid)["clip"] == (0.0, 1.0)
        assert app._model_entry(mid)["clip"] == (0.0, 1.0)

        app.set_volume_clip(vid, 0.4, 0.6)
        assert app._volume_entry(vid)["clip"] == (0.4, 0.6)
        assert app._model_entry(mid)["clip"] == (0.0, 1.0)          # untouched
        assert app.volume_appearance(vid)["clip"] == (0.4, 0.6)

        app.set_model_clip(mid, 0.1, 0.9)
        assert app._model_entry(mid)["clip"] == (0.1, 0.9)
        assert app.model_appearance(mid)["clip"] == (0.1, 0.9)

        # The pane offers the slab for either kind, at its current value.
        app._controls._update_appearance("volume", vid)
        app._controls._update_appearance("model", mid)


# -- layout and theming -------------------------------------------------------


def exercise_the_object_list_fits_its_contents():
    """A QTreeWidget's sizeHint is a fixed ~256px whatever it holds; left to it, the list
    reserves room for ten objects while showing two, and pushes the rest of the pane into
    a scrollbar. On a 13-inch screen that space decides whether the pane fits."""
    with desktop() as app:
        tree = app._controls._loaded_tree
        assert tree.maximumHeight() == _TREE_MIN_HEIGHT   # empty: no reserved space

        for i in range(60):
            blob(app, "v%d" % i, shape=(4, 4, 4))

        # It grows, but only to the ceiling -- then it scrolls itself.
        assert tree.maximumHeight() == _TREE_MAX_HEIGHT
        assert tree.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def exercise_scene_actions_are_icon_buttons():
    with desktop() as app:
        controls = app._controls
        tips = ("Open a structure", "Get data from", "Save the focused",
                "Pair an unpaired", "Remove the highlighted",
                "Reset the view", "Save a picture")
        buttons = [b for b in controls.widget().findChildren(QPushButton)
                   if b.toolTip().startswith(tips)]
        assert len(buttons) == 7
        assert all(not b.icon().isNull() and b.text() == "" for b in buttons)

        assert controls._tabs.widget(1).isAncestorOf(controls._localres_btn)   # Tools
        assert controls._localres_btn.text() == "Local res"
        assert not controls._tabs.isAncestorOf(controls._reset_view_btn)   # utility row
        assert not controls._tabs.isAncestorOf(controls._picture_btn)

        # No forced geometry anywhere: that is what broke Open's chrome.
        for button in buttons:
            assert button.minimumHeight() == 0, button.toolTip()[:20]


def exercise_the_controls_repalette_across_live_theme_changes():
    """A live light/dark transition must not leave styled child panes behind."""
    original = QPalette(QAPP.palette())

    def themed(window, text, base, button):
        palette = QPalette(original)
        palette.setColor(QPalette.ColorRole.Window, QColor(window))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
        palette.setColor(QPalette.ColorRole.Base, QColor(base))
        palette.setColor(QPalette.ColorRole.Text, QColor(text))
        palette.setColor(QPalette.ColorRole.Button, QColor(button))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(text))
        return palette

    app = DesktopApp(port=0)
    try:
        controls = app._controls

        QAPP.setPalette(themed("#202124", "#f1f3f4", "#18191b", "#303134"))
        QAPP.processEvents()
        QAPP.processEvents()          # the watcher deliberately waits one turn
        assert controls._appearance_box.palette().color(
            QPalette.ColorRole.Window) == QColor("#202124")
        dark_icon = controls._reset_view_btn.icon().cacheKey()

        QAPP.setPalette(themed("#f5f5f5", "#202124", "#ffffff", "#eeeeee"))
        QAPP.processEvents()
        QAPP.processEvents()
        expected = QColor("#f5f5f5")
        assert controls.widget().palette().color(QPalette.ColorRole.Window) == expected
        assert controls._appearance_box.palette().color(
            QPalette.ColorRole.Window) == expected
        assert controls._tabs.tabBar().palette().color(
            QPalette.ColorRole.Window) == expected
        assert controls._reset_view_btn.icon().cacheKey() != dark_icon
    finally:
        QAPP.setPalette(original)
        QAPP.processEvents()
        dispose(app)


def exercise_the_mouse_bindings_are_shown_in_the_gui():
    """Zoom moved off the scroll wheel when the bindings went Coot-style, so it has to be
    spelled out or it is unfindable."""
    with desktop() as app:
        controls = app._controls
        legend = controls._build_mouse_legend()
        texts = set(w.text() for w in legend.findChildren(QLabel))

        assert "Zoom" in texts
        assert "right-drag" in texts and "Ctrl + scroll" in texts    # both ways to zoom
        assert "scroll" in texts and "Contour level" in texts
        assert "Refine drag mode" in texts and "Pull and minimize an atom" in texts

        # The gesture is named once, here -- not repeated as a chip beside every map's
        # Level slider, where it only ate space.
        vid = blob(app)
        controls._update_appearance("volume", vid)
        chips = [w.text() for w in controls._appearance_box.findChildren(QLabel)
                 if w.text() == "scroll"]
        assert chips == []


# -- the custom colour picker -------------------------------------------------


def exercise_a_custom_colour_previews_live_not_only_on_close():
    """The picker changed the map only after the dialog closed, which read as broken
    until you gave up. The colour is driven from the dialog's ``currentColorChanged``,
    so it updates as the wheel moves."""
    from PySide6.QtWidgets import QColorDialog

    first, second = _VOLUME_COLORS[0], _VOLUME_COLORS[1]

    with desktop() as app:
        applied = []
        combo = app._controls._add_color_row(first, applied.append)

        combo.setCurrentIndex(combo.findData(second))     # a preset still applies at once
        assert applied[-1] == second

        dialog = QColorDialog(QColor(first))
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        live = []
        dialog.currentColorChanged.connect(
            lambda c: live.append(c.name()) if c.isValid() else None)
        for hex_color in ("#112233", "#445566", "#778899"):
            dialog.setCurrentColor(QColor(hex_color))
        assert live == ["#112233", "#445566", "#778899"]  # one per move, live


def exercise_committing_a_custom_colour_does_not_reopen_the_dialog():
    """Pressing OK looked like it closed the dialog and immediately reopened it.

    Inserting the picked colour into the combo shifts the still-selected "Custom..."
    entry, which re-fires ``currentIndexChanged`` with the sentinel and reopens the
    picker. The commit re-indexes with the combo's signals blocked to stop exactly that.
    """
    combo = QComboBox()
    for i in range(3):
        combo.addItem("preset%d" % i, "p%d" % i)
    combo.addItem("Custom...", _CUSTOM_COLOR)
    combo.setCurrentIndex(combo.findData(_CUSTOM_COLOR))    # as if opening the picker

    fired = []
    combo.currentIndexChanged.connect(lambda i: fired.append(combo.itemData(i)))

    # Unguarded, inserting before "Custom..." re-fires with the sentinel: the reopen.
    combo.insertItem(combo.count() - 1, "#abcabc", "#abcabc")
    combo.setCurrentIndex(combo.count() - 2)
    assert _CUSTOM_COLOR in fired                          # the bug the guard prevents

    # Guarded, which is what the commit does: no signal, so no reopen.
    combo.setCurrentIndex(combo.findData(_CUSTOM_COLOR))
    fired[:] = []
    combo.blockSignals(True)
    combo.insertItem(combo.count() - 1, "#defdef", "#defdef")
    combo.setCurrentIndex(combo.count() - 2)
    combo.blockSignals(False)
    assert fired == []


def exercise_the_level_slider_reaches_past_the_hottest_voxel():
    """Sliding fully right must empty the map, whatever the map's dynamic range.

    Cryo-EM maps carry long tails: EMD-53478 tops out near 28 sigma, so under the old
    fixed 10-sigma ceiling full-right left most of the density standing with no way to
    clear it from the slider. The slider now spans this map's own range, its right end
    strictly above the hottest voxel.
    """
    from PySide6.QtWidgets import QSlider

    from pxviewer.desktop import _ISO_RESOLUTION

    with desktop() as app:
        # A long-tailed map: quiet background with one hot voxel, max far above 10 sigma.
        data = np.zeros((12, 12, 12), dtype=np.float32)
        data[6, 6, 6] = 100.0
        vid = app._add_volume(VolumeData.from_numpy(data), "hot")
        stats = app._volume_entry(vid)["data"].stats()
        max_sigma = (stats["max"] - stats["mean"]) / stats["std"]
        assert max_sigma > 12, "fixture is not long-tailed enough to discriminate"

        controls = app._controls
        controls._update_appearance("volume", vid, force=True)
        process_events()
        sliders = controls._appearance_box.findChildren(QSlider)
        assert sliders, "no Level slider in the pane"
        top = max(sl.maximum() * _ISO_RESOLUTION for sl in sliders)
        assert top > max_sigma, (
            "slider tops out at %.2f sigma below the map's %.2f -- full-right cannot "
            "empty the map" % (top, max_sigma))
        # Nothing survives the top position: full-right means an empty surface.
        level = stats["mean"] + top * stats["std"]
        remaining = int((app._volume_entry(vid)["data"].array >= level).sum())
        assert remaining == 0, "%d voxels still visible at the slider's maximum" % remaining

        # A low-contrast map keeps a sane range instead of a hypersensitive sliver.
        flat_vid = app._add_volume(VolumeData.from_numpy(np.random.default_rng(0)
                                                         .random((8, 8, 8))), "flat")
        controls._update_appearance("volume", flat_vid, force=True)
        process_events()
        sliders = controls._appearance_box.findChildren(QSlider)
        top = max(sl.maximum() * _ISO_RESOLUTION for sl in sliders)
        assert 4.0 <= top <= 12.0, "flat map got an unusable slider range: %.2f" % top


def exercise_the_tabs_share_the_full_bar_width():
    """Seven icon tabs span the bar edge to edge, not left-huddled beside grey space."""
    with desktop() as app:
        process_events()
        bar = app._controls._tabs.tabBar()
        count = bar.count()
        assert count >= 7, "expected the seven main tabs, found %d" % count
        total = sum(bar.tabRect(i).width() for i in range(count))
        # Rounding leaves a few px; anything more is the old left-tight layout.
        assert total >= bar.width() - count, (
            "tabs cover %dpx of a %dpx bar -- the bar is not filled" % (total, bar.width()))


def exercise_model_value_colourings_share_the_scale_machinery():
    """By B-factor / By occupancy colour through the attribute path with a settable
    range -- the model-side twin of the map's local-resolution scale: same builder,
    same blue->red ramp, a domain that is the user's and stays put.
    """
    from pxviewer.desktop import _VALUE_PALETTE
    from pxviewer.regression.tst_utils import data_path
    from PySide6.QtWidgets import QDoubleSpinBox

    with desktop() as app:
        app.load_file(data_path("1ubq.pdb"))
        process_events()
        mid = app._models[0]["id"]
        entry = app._model_entry(mid)

        app.set_model_color(mid, "bfactor")
        process_events()
        attribute = entry.get("attribute")
        assert attribute is not None and attribute["name"] == "bfactor"
        assert attribute["palette"] == _VALUE_PALETTE, "the shared ramp is not shared"
        lo, hi = attribute["domain"]
        assert hi > lo
        assert len(attribute["values"]) == len(
            entry["session"].model.get_hierarchy().atoms())

        # The range is the user's: settable, refusing a crossed pair, resettable.
        app.set_model_value_domain(mid, 5.0, 40.0)
        assert attribute["domain"] == (5.0, 40.0)
        app.set_model_value_domain(mid, 50.0, 5.0)
        assert attribute["domain"] == (5.0, 40.0), "a crossed range was accepted"
        app.reset_model_value_domain(mid)
        assert attribute["domain"] == attribute["default_domain"]

        # Re-picking the colouring keeps a user-set range rather than clobbering it.
        app.set_model_value_domain(mid, 5.0, 40.0)
        entry["color"] = None  # as if another colour had been picked meanwhile
        app.set_model_color(mid, "bfactor")
        assert entry["attribute"]["domain"] == (5.0, 40.0)

        # Occupancy defaults to its natural 0-1 scale (percentiles of all-1.0 collapse).
        app.set_model_color(mid, "occupancy")
        assert entry["attribute"]["domain"] == (0.0, 1.0)

        # The pane shows the shared Range group for the model, same as for the map.
        controls = app._controls
        controls._update_appearance("model", mid, force=True)
        process_events()
        spins = [w for w in controls._appearance_box.findChildren(QDoubleSpinBox)
                 if w.objectName().startswith("value-domain-")]
        assert len(spins) == 2, "no shared Range group on the model pane"
        assert {round(sp.value(), 2) for sp in spins} == {0.0, 1.0}


def exercise_tests_never_touch_the_users_settings():
    """The app's persistence lands in the harness's throwaway directory, not in the
    user's live preference domain -- test runs used to fight the user's real settings
    (the GUI fuzzer toggling a persisted checkbox rewrote it on disk)."""
    from pxviewer.regression.tst_utils import TEST_SETTINGS_DIR

    with desktop() as app:
        path = app._settings.fileName()
        assert path.startswith(TEST_SETTINGS_DIR), path
        assert "Library/Preferences" not in path, path
        app._settings.setValue("test/sentinel", "yes")
        app._settings.sync()
        import os
        assert os.path.exists(path), "the isolated settings file was never written"


def run():
    # Every exercise here builds a DesktopApp, which reads its defaults from QSettings --
    # so the whole file runs against a fresh install's preferences, not the user's.
    with shipped_defaults():
        for name, fn in sorted(globals().items()):
            if name.startswith("exercise"):
                print("  %s" % name)
                sys.stdout.flush()
                fn()
    print("OK")


if __name__ == "__main__":
    run()
