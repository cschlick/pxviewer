"""What the desktop holds: models, volumes, reflections, and the groups over them.

The distinction the whole file turns on is that **a group is not a pairing**. A group is
how the panel shows things that were opened together; a pairing is a cctbx
``map_model_manager`` that put a model and a map in a common frame. Objects can look
entirely compatible, sit in one group, and still not be paired -- and until they are,
there is no map to minimize into. Inferring the pairing from an eyeball compatibility
check is the regression several of these exercises guard.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import os
import sys
import time

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import (
    closing_modals, data_path, have, qt_application, skip, tmp_dir)

if not have("PySide6.QtWebEngineWidgets", "websockets",
            "iotbx.map_model_manager", "numpy"):
    skip("PySide6 QtWebEngine / websockets / iotbx.map_model_manager not available")

import numpy as np                               # noqa: E402

qt_application()

from PySide6.QtWidgets import QApplication       # noqa: E402

from pxviewer.desktop import DesktopApp, _GROUP_MEMBER_INDENT     # noqa: E402
from pxviewer.live import LiveSession            # noqa: E402
from pxviewer.volume_io import VolumeData        # noqa: E402

#: Phasing runs on a worker thread; generous, since it is seconds on this structure.
PHASE_TIMEOUT_S = 180


@contextlib.contextmanager
def desktop(**kwargs):
    app = DesktopApp(port=0, **kwargs)
    app._webapp.start()
    with closing_modals():
        try:
            yield app
        finally:
            app.stop()


@contextlib.contextmanager
def model_and_map_files(model_offset=None, boxed_origin=None, external_origin=None):
    """One structure written out as a separate model file and map file.

    ``model_offset`` displaces the model before writing, so the pair genuinely disagrees
    and the density knows by how much. ``boxed_origin`` writes the map away from the
    origin, and ``external_origin`` stamps a sub-voxel MRC ORIGIN record.

    The offset is a plain ``set_sites_cart`` move, deliberately **not**
    ``shift_model_and_set_crystal_symmetry``. The shift-aware call records the move in
    the model's history, and ``model_as_pdb`` then undoes it on the way out -- recovering
    the source coordinates, which is the whole point of that API and is asserted of the
    production path elsewhere in this file. It would write an *undisplaced* file here,
    leaving the alignment nothing to find and the exercise passing for the wrong reason.
    """
    from iotbx.map_model_manager import map_model_manager
    from scitbx.array_family import flex

    source = map_model_manager()
    source.generate_map()

    map_manager = source.map_manager().deep_copy()
    if boxed_origin is not None:
        map_manager.set_original_origin_and_gridding(original_origin=boxed_origin)
    if external_origin is not None:
        map_manager.set_output_external_origin(external_origin)

    model = source.model().deep_copy()
    if model_offset is not None:
        sites = model.get_sites_cart()
        model.set_sites_cart(sites + flex.vec3_double(
            sites.size(), tuple(float(v) for v in model_offset)))

    with tmp_dir() as directory:
        map_path = os.path.join(directory, "m.mrc")
        model_path = os.path.join(directory, "m.pdb")
        map_manager.write_map(map_path)
        with open(model_path, "w") as handle:
            handle.write(model.model_as_pdb())
        yield directory, model_path, map_path


def pump_until(predicate, what, timeout=PHASE_TIMEOUT_S):
    deadline = time.time() + timeout
    while time.time() < deadline and not predicate():
        QApplication.processEvents()
        time.sleep(0.05)
    QApplication.processEvents()
    assert predicate(), what


# -- the registries -----------------------------------------------------------


def exercise_a_volume_is_its_own_category():
    """Volumes never enter the model registry, and the one loaded is written where the
    browser can fetch it."""
    with desktop() as app:
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")

        assert len(app._volumes) == 1
        assert app._volumes[0]["visible"]
        assert not app._models
        assert app._write_volume_scene() is not None
        assert os.path.exists(
            str(app._webapp.volume_dir / "vols" / ("%s.map" % vid)))

        # This is a software app (the default), where hiding a map is refused, so it
        # stays in the scene until it is actually unloaded.
        app.set_volume_visible(vid, False)
        assert app._volume_entry(vid)["visible"] is True
        assert app._write_volume_scene() is not None

        app.remove_volume(vid)
        assert not app._volumes
        assert app._write_volume_scene() is None


def exercise_a_map_and_model_loaded_together_form_one_group():
    with desktop() as app:
        captured = {}
        app.bridge.loaded_changed.connect(lambda s: captured.update(s))

        with model_and_map_files() as (_dir, model_path, map_path):
            assert app.load_files([model_path, map_path]) == "group"

        assert len(app._models) == 1
        assert len(app._volumes) == 1
        gid = app._models[0]["group"]
        assert gid is not None
        assert app._volumes[0]["group"] == gid
        assert gid in app._groups

        # Model and volume coexist: the viewport gets both a socket and an MVSJ.
        assert len(app._visible_model_ws()) == 1
        assert app._write_volume_scene() is not None

        # The Loaded summary carries the group and both its items.
        assert gid in set(g["id"] for g in captured["groups"])
        assert set(it["kind"] for it in captured["items"]) == {"model", "volume"}

        app.remove_group(gid)
        assert not app._models
        assert not app._volumes
        assert gid not in app._groups


def exercise_the_map_model_demo_loads_as_a_group():
    with desktop() as app:
        assert app.load_map_model_demo(d_min=4.0) == "group"   # coarser = faster

        # One model and exactly one density map: the redundant model_map is dropped.
        assert len(app._models) == 1
        assert len(app._volumes) == 1
        gid = app._models[0]["group"]
        assert gid is not None
        assert app._volumes[0]["group"] == gid
        assert app._models[0]["session"]._n_atoms == 660       # ubiquitin
        assert len(app._visible_model_ws()) == 1
        assert app._write_volume_scene() is not None


def exercise_the_multi_model_registry():
    """Add overlays, hide switches, and the active model is what the table follows."""
    from pxviewer.appserver import find_frontend_dir, frontend_is_built

    frontend = find_frontend_dir()
    if frontend is None or not frontend_is_built(frontend):
        print("    (skipped: frontend not built)")
        return

    with desktop(can_hide=True) as app:          # hardware: models are allowed to hide
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(
            LiveSession.from_sites([[5, 0, 0], [6, 0, 0], [7, 0, 0]]), "B")

        assert len(app._models) == 2
        assert app._active_model_id == b
        assert len(app._visible_model_ws()) == 2      # both visible: drawn together
        assert app._session._n_atoms == 3             # active is B

        app.set_model_visible(a, False)
        assert len(app._visible_model_ws()) == 1

        app.set_active_model(a)
        assert app._session._n_atoms == 2   # the table follows A even while it is hidden

        app.remove_model(b)
        assert len(app._models) == 1
        assert app._active_model_id == a


# -- grouping is not pairing --------------------------------------------------


def exercise_a_group_keeps_the_manager_that_pairs_its_objects():
    """A group is not just a label: it holds the cctbx ``map_model_manager`` that put the
    model and map in a common frame.

    That manager is the only record of the pairing -- the DataManager keeps none, since
    ``get_map_model_manager`` evicts what it consumed.
    """
    with desktop() as app:
        app.load_map_model_demo(d_min=4.0)
        gid = app._models[0]["group"]
        mmm = app.group_mmm(gid)
        assert mmm is not None

        # The viewer shows the manager's own model, so minimizing it in place keeps the
        # pairing true rather than drifting away from it.
        assert app._models[0]["session"].model is mmm.model()
        # And the map offered for minimization is the manager's, not one picked out by
        # inspecting grids here.
        assert app.map_for_model() is mmm.map_manager().map_data()

        app.remove_group(gid)
        assert app.group_mmm(gid) is None


def exercise_looking_compatible_is_not_being_paired():
    """The regression this guards: deciding a model and a map go together by inspecting
    them. These two *are* mutually compatible by cctbx's own test and sit in one group,
    and they are still not paired, because nothing ever paired them."""
    from pxviewer.cctbx_io import read_model

    with desktop() as app:
        with model_and_map_files() as (_dir, model_path, map_path):
            model = read_model(model_path)
            volume = VolumeData.from_map_file(map_path)

        gid = app._new_group("hand-made")        # a group cctbx never paired
        app._add_model(LiveSession.from_cctbx_model(model), "m", group=gid)
        app._add_volume(volume, "map", group=gid)

        # The premise: they really would pass an eyeball compatibility check.
        assert volume.map_manager.origin_is_zero()
        assert volume.map_manager.is_compatible_model(model)

        # And it counts for nothing: no manager, no pairing, no map.
        assert app.group_mmm(gid) is None
        assert app.map_for_model() is None
        with raises(ValueError) as e:
            app.minimize_model(use_map=True)
        assert "not paired" in str(e.value)


def exercise_unpaired_objects_can_be_paired_explicitly():
    """Pairing is offered as an action rather than inferred, because it is one: cctbx
    relocates the model into a common frame with the map."""
    with desktop() as app, model_and_map_files() as (_d, model_path, map_path):
        app.load_files([model_path])
        app.load_files([map_path])

        models, volumes = app.pairable()
        assert len(models) == 1 and len(volumes) == 1     # both unpaired, both offered
        assert app.map_for_model() is None
        assert app._controls._pair_btn.isEnabled()

        gid = app.pair_model_with_map(models[0]["id"], volumes[0]["id"])
        mmm = app.group_mmm(gid)
        assert mmm is not None
        assert app._models[0]["group"] == gid
        assert app._volumes[0]["group"] == gid

        # The pairing is cctbx's, and it is what now answers the map question.
        assert app.map_for_model() is mmm.map_manager().map_data()
        assert app._models[0]["session"].model is mmm.model()

        # Paired objects are no longer on offer for pairing, but Pair stays available as
        # the place to re-check an existing pair's alignment.
        assert app.pairable() == ([], [])
        assert app._controls._pair_btn.isEnabled()
        assert len(app.alignable()) == 1
        with raises(ValueError) as e:
            app.pair_model_with_map(app._models[0]["id"], app._volumes[0]["id"])
        assert "already paired" in str(e.value)

        assert app._controls._minimize_map_check.isEnabled()


def exercise_a_model_and_its_reflections_are_one_group_before_phasing():
    """A model opened with its reflections shows as one group straight away, and Make
    maps fills that group in rather than starting a second one.

    Grouping the pair before it is phased must not make the model look already paired --
    it has to stay phasable.
    """
    if not have("mmtbx.f_model"):
        print("    (skipped: mmtbx.f_model not available)")
        return

    with desktop() as app:
        app.load_xray_demo()
        QApplication.processEvents()

        model, reflections = app._models[0], app._reflections[0]
        gid = model["group"]
        assert gid is not None
        assert reflections["group"] == gid          # one group, from the moment it loads
        assert app.group_mmm(gid) is None           # but no manager yet: not paired
        assert model["id"] in [m["id"] for m in app.models_for_phasing()]

        groups_before = set(app._groups)
        app.make_maps(reflections["id"], model["id"])
        pump_until(lambda: app.map_for_model(model["id"]) is not None,
                   "phasing never produced a map")

        assert set(app._groups) == groups_before    # filled in, not a second group
        assert app.group_mmm(gid) is not None
        assert all(v["group"] == gid for v in app._volumes)


def exercise_separately_loaded_models_stay_out_of_a_group():
    """Grouping is for objects opened as a unit. Models opened on their own stay
    top-level, and must not be swept into a group formed by a later load."""
    with desktop() as app:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        assert app._model_entry(a)["group"] is None
        assert app._model_entry(b)["group"] is None

        app.load_xray_demo()                        # a later unit-load forms its own
        QApplication.processEvents()

        assert app._model_entry(a)["group"] is None
        assert app._model_entry(b)["group"] is None
        assert len(app._groups) == 1                # only the x-ray pair grouped


def exercise_a_standalone_object_is_not_indented_like_a_group_member():
    """Qt indents column 0 only, so without help a group member's *name* sits at the same
    x as a standalone object's and everything reads as belonging to the group above."""
    with desktop() as app:
        app.load_xray_demo()                        # a group: model + reflections
        QApplication.processEvents()
        app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "loose")
        QApplication.processEvents()

        tree = app._controls._loaded_tree
        roots = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        group = next(r for r in roots if r.childCount())
        loose = next(r for r in roots if not r.childCount() and r.text(2))

        assert group.child(0).text(2).startswith(_GROUP_MEMBER_INDENT)
        assert not loose.text(2).startswith(" ")
        assert loose.text(2).startswith("loose")


# -- aligning a pair the density disagrees with -------------------------------


def exercise_the_translation_search_leaves_its_model_untouched():
    """The search works on a deep copy and returns only the frame translation.

    cctbx's ``translation_search`` moves its input sites, so running it on the live model
    would relocate the thing on screen as a side effect of measuring it.
    """
    from iotbx.map_model_manager import map_model_manager

    source = map_model_manager()
    source.generate_map()
    model = source.model().deep_copy()
    model.shift_model_and_set_crystal_symmetry(
        shift_cart=(3.0, -2.0, 1.0),
        crystal_symmetry=source.map_manager().crystal_symmetry())
    before = model.get_sites_cart().deep_copy()

    found = DesktopApp._detect_map_model_shift(
        model, source.map_manager().map_data())

    assert approx_equal(tuple(found), (-3.0, 2.0, -1.0), eps=0.15)
    assert approx_equal(model.get_sites_cart().rms_difference(before), 0.0)


def exercise_pairing_can_detect_and_record_a_missing_shift_cart():
    """Opt-in density alignment recovers a shift the file never recorded.

    Built as a pair that genuinely disagrees -- the model is written 3 A off the density
    -- rather than by substituting the detector for one that returns a chosen answer. The
    detection and its application are the same feature, and separating them leaves the
    join untested; run together, the recovered shift is the one built in.
    """
    displacement = (-3.0, 0.0, 0.0)
    expected = tuple(-v for v in displacement)

    with desktop() as app, model_and_map_files(model_offset=displacement) as (
            _d, model_path, map_path):
        app.load_files([model_path])
        app.load_files([map_path])
        mid, vid = app._models[0]["id"], app._volumes[0]["id"]
        before = app._models[0]["session"].model.get_sites_cart().deep_copy()

        gid = app.pair_model_with_map(mid, vid, detect_shift=True)
        mmm = app.group_mmm(gid)
        applied = app._model_entry(mid)["detected_shift_cart"]

        assert approx_equal(tuple(applied), expected, eps=0.15)

        # Whatever was applied, the model moved by exactly it ...
        moved = mmm.model().get_sites_cart() - before
        assert approx_equal(tuple(moved.mean()), tuple(applied), eps=1e-4)
        # ... and it is recorded as the model's shift history. The exact Cartesian value
        # is deliberately not rounded into the map's integer-grid origin metadata.
        assert approx_equal(tuple(mmm.model().shift_cart()), tuple(applied), eps=1e-4)


def exercise_a_pair_loaded_together_can_still_be_aligned_afterwards():
    """Opening model and map together pairs them immediately, but Pair remains the entry
    point for density alignment -- and reuses that manager rather than nesting a new pair
    inside it.

    Pairing on load puts the two in a common frame from their file metadata; it cannot
    know about a model whose coordinates are simply in the wrong place. So a pair opened
    together can still be several angstrom out, and aligning it afterwards is a real
    correction rather than a formality.
    """
    offset = (-3.0, 0.0, 0.0)

    with desktop() as app, model_and_map_files(model_offset=offset) as (
            _d, model_path, map_path):
        app.load_files([model_path, map_path])
        mid, vid = app._models[0]["id"], app._volumes[0]["id"]
        gid = app._models[0]["group"]
        manager = app.group_mmm(gid)

        applied = app.align_paired_model_with_map(mid, vid, detect_shift=True)

        assert approx_equal(tuple(applied), tuple(-v for v in offset), eps=0.15)
        assert app.group_mmm(gid) is manager             # the same manager, not a new one
        assert len(app._models) == 1
        assert len(app._volumes) == 1
        assert app._model_entry(mid)["group"] == gid
        assert app._volume_entry(vid)["group"] == gid
        assert app._model_entry(mid)["session"]._last_frame is not None
        assert app._controls._pair_btn.isEnabled()


def exercise_a_sub_voxel_mrc_origin_seeds_the_alignment():
    """A real-world MRC ORIGIN may sit between voxels. cctbx warns and discards it, but
    it is still an exact Cartesian hypothesis, so opt-in pairing keeps it as the search's
    starting shift and lets density determine only the residual."""
    import contextlib as _contextlib
    import io

    external = (3.13, -2.27, 1.19)               # deliberately not voxel coordinates

    with desktop() as app, model_and_map_files(
            model_offset=external, external_origin=external) as (
            _d, model_path, map_path):
        noise = io.StringIO()
        with _contextlib.redirect_stdout(noise):
            app.load_files([model_path])
            app.load_files([map_path])
            gid = app.pair_model_with_map(
                app._models[0]["id"], app._volumes[0]["id"], detect_shift=True)

        applied = app._model_entry(app._models[0]["id"])["detected_shift_cart"]
        assert approx_equal(tuple(applied), tuple(-v for v in external), eps=0.15)
        # The origin was used rather than discarded with a warning.
        assert "External origin is not on a grid point" not in noise.getvalue()
        assert approx_equal(
            tuple(app.group_mmm(gid).model().shift_cart()), tuple(applied), eps=1e-4)


def exercise_pairing_a_boxed_map_keeps_model_and_map_drawn_together():
    """Pairing relocates the model into the map's frame -- several angstrom for a boxed
    map -- so the map the browser is served has to move with it, or the model is drawn
    away from its own density.

    cctbx writes a map back in the frame it was read in, which is right for saving a file
    and wrong for the copy on screen, so the two paths must differ.
    """
    with desktop() as app, model_and_map_files(boxed_origin=(10, 10, 10)) as (
            directory, model_path, map_path):
        app.load_files([model_path])
        app.load_files([map_path])
        before = app._models[0]["session"].model.get_sites_cart().as_numpy_array().copy()
        vid = app._volumes[0]["id"]

        app.pair_model_with_map(app._models[0]["id"], vid)

        # The model really did move: this is why pairing cannot be a passive label.
        after = app._models[0]["session"].model.get_sites_cart().as_numpy_array()
        assert np.linalg.norm((after - before).mean(axis=0)) > 1.0

        # The served map moved with it -- written in the frame the model is drawn in.
        served = str(app._webapp.volume_dir / "vols" / ("%s.map" % vid))
        assert VolumeData.from_map_file(served).map_manager.map_data().origin() == (0, 0, 0)

        # Saving the map for the user is a different job, and keeps the original frame.
        out = os.path.join(directory, "saved.mrc")
        app._volumes[0]["data"].write_map(out)
        assert VolumeData.from_map_file(out).map_manager.map_data().origin() == (10, 10, 10)


# -- writing objects back out -------------------------------------------------


def exercise_write_object():
    """A model's cctbx coordinates in either format, and a volume's map."""
    with desktop() as app, tmp_dir() as directory:
        mid = app._add_model(
            LiveSession.from_model_file(data_path("1ubq.pdb")), "1ubq")

        pdb = os.path.join(directory, "out.pdb")
        app.write_object("model", mid, pdb)
        assert "ATOM" in open(pdb).read()

        cif = os.path.join(directory, "out.cif")
        app.write_object("model", mid, cif)
        assert "_atom_site" in open(cif).read()

        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        mrc = os.path.join(directory, "out.mrc")
        app.write_object("volume", vid, mrc)
        assert os.path.getsize(mrc) > 0


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
