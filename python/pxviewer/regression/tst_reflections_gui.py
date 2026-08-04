"""What loading reflections does to the desktop scene.

The cctbx half -- reading an MTZ and computing maps from it -- is in
``tst_reflections.py`` and needs no Qt. Everything here is about the consequences: which
maps open by themselves, how they are styled, what they are grouped with, and what
happens to them when the model moves.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import os
import sys
import time

from libtbx.test_utils import raises

from pxviewer.regression.tst_utils import (
    data_path, dispose, have, process_events, qt_application, shipped_defaults, skip,
    tmp_dir)
if not have("PySide6.QtWebEngineWidgets", "websockets", "iotbx.data_manager", "numpy"):
    skip("PySide6 QtWebEngine / websockets / iotbx.data_manager not available")

import numpy as np                                   # noqa: E402

qt_application()

from PySide6.QtCore import QCoreApplication         # noqa: E402

from pxviewer.desktop import _VIEW_RADIUS_DEFAULT, DesktopApp     # noqa: E402
from pxviewer.palettes import load_palettes         # noqa: E402

MODEL = data_path("1ubq.pdb")
D_MIN = 2.0

#: Phasing runs on a worker thread. Generous: it is seconds on this structure.
PHASE_TIMEOUT_S = 90


@contextlib.contextmanager
def mtz(coefficients):
    """One of the two kinds of MTZ. See ``tst_reflections.mtz`` for what they mean.

    The file has to outlive the load -- phasing reads it back -- so the directory stays
    open for the whole exercise rather than just the write.
    """
    from pxviewer.cctbx_io import read_model

    f_calc = read_model(MODEL).get_xray_structure().structure_factors(d_min=D_MIN).f_calc()
    with tmp_dir() as directory:
        if coefficients:
            dataset = f_calc.as_mtz_dataset(column_root_label="2FOFCWT")
            dataset.add_miller_array(f_calc, column_root_label="FOFCWT")
            path = os.path.join(directory, "refine_maps.mtz")
        else:
            f_obs = abs(f_calc).set_observation_type_xray_amplitude()
            f_obs = f_obs.customized_copy(sigmas=f_obs.data() * 0.05)
            dataset = f_obs.as_mtz_dataset(column_root_label="F")
            dataset.add_miller_array(f_obs.generate_r_free_flags(fraction=0.05),
                                     column_root_label="R-free-flags")
            path = os.path.join(directory, "data.mtz")
        dataset.mtz_object().write(path)
        yield path


@contextlib.contextmanager
def desktop():
    """A running desktop app, stopped afterwards even if the body raises."""
    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        yield app
    finally:
        dispose(app)


def pump_until(predicate, what, timeout=PHASE_TIMEOUT_S):
    """Run the Qt loop until a worker's result lands on the main thread."""
    deadline = time.time() + timeout
    while time.time() < deadline and not predicate():
        process_events()
        time.sleep(0.05)
    process_events()
    assert predicate(), what


def phase(app, path):
    """Load a data MTZ and the model, phase them, return ``(rid, mid)``."""
    app.load_file(path)
    app.load_file(MODEL)
    rid, mid = app._reflections[0]["id"], app._models[0]["id"]
    app.make_maps(rid, mid)
    pump_until(lambda: app._volumes, "phasing produced no maps")
    return rid, mid


def palette_colors():
    return set(c for group in load_palettes() for c in group)


# -- loading ------------------------------------------------------------------


def exercise_reflections_load_as_an_object_that_draws_nothing():
    """Reflections are the one loaded thing with nothing to draw: density is an FFT away,
    and for amplitudes a model away too. They are kept rather than consumed into maps,
    because recomputing density when the model moves needs them still here."""
    with desktop() as app, mtz(coefficients=False) as path:
        assert app.load_file(path) == "reflections"
        assert len(app._reflections) == 1
        # Not a model, not a volume, and nothing composed into the scene.
        assert not app._models
        assert not app._volumes
        assert app._write_volume_scene() is None

        item = next(i for i in app._emitted_items() if i["kind"] == "reflections")
        assert item["visible"] is None                 # nothing to show or hide
        assert item["has_map_coefficients"] is False

        app.remove_reflections(app._reflections[0]["id"])
        assert app._reflections == []


def exercise_a_data_mtz_makes_no_maps():
    """Amplitudes cannot become density on their own -- the phases have to be computed
    against a model. Loading one draws nothing rather than guessing."""
    with desktop() as app, mtz(coefficients=False) as path:
        app.load_file(path)
        assert len(app._reflections) == 1
        assert app._volumes == []
        assert app._reflections[0]["group"] is None    # nothing to group it with


def exercise_a_refinement_mtz_opens_its_maps():
    """A file carrying map coefficients is a refinement result and the density is what it
    is for, so the maps are made on load rather than asked about -- Coot's Auto Open, and
    the reason it is how most people open an MTZ."""
    with desktop() as app, mtz(coefficients=True) as path:
        app.load_file(path)
        assert len(app._reflections) == 1
        assert len(app._volumes) == 2                  # 2FOFCWT and FOFCWT, unprompted

        by_name = dict((v["name"], v) for v in app._volumes)
        assert set(by_name) == {"2FOFCWT", "FOFCWT"}
        # A regular map opens in a palette colour at 1.5 sigma; a difference map keeps
        # the convention it has to keep -- green/red -- at 3.
        assert by_name["2FOFCWT"]["color"] in palette_colors()
        assert by_name["2FOFCWT"]["iso"] == 1.5
        assert (by_name["FOFCWT"]["color"], by_name["FOFCWT"]["iso"]) == ("green", 3.0)

        # Data and maps are one group: the maps came from the file and go with it.
        gid = app._reflections[0]["group"]
        assert gid is not None
        assert all(v["group"] == gid for v in app._volumes)
        assert app.group_mmm(gid) is None              # no model, so cctbx paired nothing
        assert app._write_volume_scene() is not None   # they really are in the scene

        app.remove_group(gid)
        assert not app._volumes
        assert not app._reflections


def exercise_difference_maps_get_a_negative_contour():
    """Green where the density wants more than the model has, red where it wants less. A
    difference map read at one sign is half a map, so it opens with both."""
    with desktop() as app, mtz(coefficients=True) as path:
        app.load_file(path)
        by_name = dict((v["name"], v) for v in app._volumes)

        assert by_name["FOFCWT"]["negative_color"] == "red"
        # The 2Fo-Fc map gets none: its negative side is noise, and a second isosurface
        # is not free.
        assert by_name["2FOFCWT"]["negative_color"] is None
        assert app._write_volume_scene() is not None


def exercise_maps_from_reflections_open_with_a_view_radius():
    """A map made from reflections fills the unit cell, so drawing all of it buries the
    model in density -- Coot has a radius for exactly this. A map read from a file is
    already a box around its subject, so it gets none."""
    from pxviewer.volume_io import VolumeData

    with desktop() as app, mtz(coefficients=True) as path:
        app.load_file(path)
        assert all(v["radius"] == _VIEW_RADIUS_DEFAULT for v in app._volumes)

        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "cryoem")
        assert app._volume_entry(vid)["radius"] is None

        app.set_volume_radius(vid, 20.0)
        assert app.volume_appearance(vid)["radius"] == 20.0
        app.set_volume_radius(vid, None)
        assert app.volume_appearance(vid)["radius"] is None


# -- phasing ------------------------------------------------------------------


def exercise_making_maps_pairs_them_with_the_model_that_phased_them():
    """The phases came from the model, so the maps and the model are inseparable -- one
    map_model_manager. Which is also what makes them usable together: masking, and
    minimizing into the density, work on X-ray maps the moment they exist."""
    with desktop() as app, mtz(coefficients=False) as path:
        app.load_file(path)
        app.load_file(MODEL)
        rid, mid = app._reflections[0]["id"], app._models[0]["id"]

        assert [m["id"] for m in app.models_for_phasing()] == [mid]
        assert app._volumes == []                      # amplitudes alone make nothing

        app.make_maps(rid, mid)
        pump_until(lambda: app._volumes, "phasing produced no maps")

        by_name = dict((v["name"], v) for v in app._volumes)
        assert set(by_name) == {"2mFo-DFc", "mFo-DFc"}
        assert by_name["2mFo-DFc"]["color"] in palette_colors()
        assert (by_name["mFo-DFc"]["color"], by_name["mFo-DFc"]["iso"]) == ("green", 3.0)

        gid = app._models[0]["group"]
        assert gid is not None
        assert app._reflections[0]["group"] == gid
        assert app.group_mmm(gid) is not None          # cctbx really paired them

        # The payoff: everything that needs a pair now works on X-ray maps.
        assert app.map_for_model(mid) is not None                  # minimize into density
        assert app.can_mask_volume(by_name["2mFo-DFc"]["id"])      # mask around the model

        # The fit is reported, and a paired model is no longer on offer to phase again.
        assert app._reflections[0]["r_work"] is not None
        assert app.models_for_phasing() == []
        with raises(ValueError) as e:
            app.make_maps(rid, mid)
        assert "already paired" in str(e.value)


def exercise_a_moved_model_knows_which_reflections_phased_it():
    """What lets Minimize recompute the density without being asked."""
    with desktop() as app, mtz(coefficients=False) as path:
        app.load_file(MODEL)
        assert app.reflections_for_model(app._models[0]["id"]) is None  # nothing phased it

        app._clear_all()
        rid, mid = phase(app, path)

        found = app.reflections_for_model(mid)
        assert found is not None
        assert found["id"] == rid


def exercise_updating_maps_replaces_them_in_place():
    """Why the reflections are kept at all: once the model moves, the maps describe a
    model that no longer exists -- the difference map most of all, since it answers "what
    does the density have that the model does not" about the old positions.

    Replaced in place, so a level or a colour the user set on them survives.
    """
    with desktop() as app, mtz(coefficients=False) as path:
        rid, _mid = phase(app, path)

        volume = next(v for v in app._volumes if v["name"] == "mFo-DFc")
        vid = volume["id"]
        app.set_volume_iso(vid, 4.25)                  # a setting the user made
        before_map = volume["data"].map_manager.map_data().as_numpy_array().copy()
        before_r = app._reflection_entry(rid)["r_work"]

        # The model moves, as Minimize -- or anything else -- would move it.
        model = app._models[0]["session"].model
        xrs = model.get_xray_structure().deep_copy_scatterers()
        xrs.shake_sites_in_place(mean_distance=0.3)
        model.set_sites_cart(xrs.sites_cart())

        app.update_maps(rid)
        pump_until(lambda: app._reflection_entry(rid)["r_work"] != before_r,
                   "the maps were never recomputed")

        after = next(v for v in app._volumes if v["name"] == "mFo-DFc")
        assert after["id"] == vid                      # the same object
        assert after["iso"] == 4.25                    # the user's setting survived
        assert not np.array_equal(
            before_map, after["data"].map_manager.map_data().as_numpy_array())

        # And the map that minimizing refines into is the fresh one, not the stale one.
        mmm = app.group_mmm(app._models[0]["group"])
        assert np.array_equal(
            mmm.map_manager().map_data().as_numpy_array(),
            next(v for v in app._volumes if v["name"] == "2mFo-DFc")
            ["data"].map_manager.map_data().as_numpy_array())


def exercise_the_xray_demo_loads_a_model_and_its_reflections():
    """The X-ray demo shows the density-from-data path without a real dataset: it
    generates amplitudes from the bundled model, writes an MTZ, and loads the two
    unpaired so Make maps is there to try.

    The MTZ has to outlive the load, since phasing reads it back -- a temporary that is
    cleaned up when the loader returns leaves a demo that cannot be run.
    """
    with desktop() as app:
        app.load_xray_demo(d_min=2.5)

        assert len(app._models) == 1
        assert len(app._reflections) == 1
        reflections = app._reflections[0]
        assert not reflections["data"].has_map_coefficients
        assert os.path.exists(reflections["data"].path)
        assert len(app.models_for_phasing()) == 1


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
