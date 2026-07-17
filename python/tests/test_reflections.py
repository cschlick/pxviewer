"""Tests for reading X-ray reflections with cctbx."""

from pathlib import Path

import pytest

pytest.importorskip("iotbx.data_manager")

MODEL = Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb"


def _mtz(tmp_path, *, coefficients: bool):
    """An MTZ of the two kinds that exist in practice: a refinement file carrying map
    coefficients, or a data file carrying amplitudes and free flags."""
    from pxviewer.cctbx_io import read_model

    f_calc = read_model(str(MODEL)).get_xray_structure().structure_factors(d_min=2.0).f_calc()
    if coefficients:
        dataset = f_calc.as_mtz_dataset(column_root_label="2FOFCWT")
        dataset.add_miller_array(f_calc, column_root_label="FOFCWT")
        path = tmp_path / "refine_maps.mtz"
    else:
        f_obs = abs(f_calc).set_observation_type_xray_amplitude()
        f_obs = f_obs.customized_copy(sigmas=f_obs.data() * 0.05)
        dataset = f_obs.as_mtz_dataset(column_root_label="F")
        dataset.add_miller_array(
            f_obs.generate_r_free_flags(fraction=0.05), column_root_label="R-free-flags")
        path = tmp_path / "data.mtz"
    dataset.mtz_object().write(str(path))
    return path


def test_file_kind_recognises_reflections():
    from pxviewer.loader import file_kind

    assert file_kind("data.mtz") == "reflections"
    assert file_kind("model.pdb") == "model"
    assert file_kind("map.mrc") == "volume"


def test_map_coefficients_are_cctbxs_call_not_ours(tmp_path):
    """Whether density needs a model is the fork the whole feature turns on, and cctbx
    answers it: map_coefficients is a child datatype of miller_array, so the DataManager
    already separates a refinement file from a data file. We never read column names."""
    from pxviewer.reflections import ReflectionData

    refinement = ReflectionData.from_file(str(_mtz(tmp_path, coefficients=True)))
    assert refinement.has_map_coefficients
    assert len(refinement.map_coefficient_arrays()) == 2  # 2FOFCWT and FOFCWT

    data = ReflectionData.from_file(str(_mtz(tmp_path, coefficients=False)))
    assert not data.has_map_coefficients
    assert data.map_coefficient_arrays() == []
    assert "F,SIGF" in data.labels


def test_reflection_metadata(tmp_path):
    from pxviewer.reflections import ReflectionData

    data = ReflectionData.from_file(str(_mtz(tmp_path, coefficients=False)))
    d_max, d_min = data.resolution_range
    assert d_min == pytest.approx(2.0, abs=0.01)
    assert d_max > d_min
    assert data.n_reflections > 1000
    assert data.crystal_symmetry.unit_cell().parameters()[0] == pytest.approx(50.84, abs=0.01)
    assert "amplitudes" in data.summary()


def test_reflections_load_as_an_object_that_draws_nothing(tmp_path):
    """Reflections are the one loaded thing with nothing to draw: density is an FFT
    away, and for amplitudes a model away too. They are kept rather than consumed into
    maps, because recomputing density when the model moves needs them still here."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        assert app.load_file(str(_mtz(tmp_path, coefficients=False))) == "reflections"
        assert len(app._reflections) == 1
        # Not a model, not a volume, and nothing composed into the scene.
        assert not app._models and not app._volumes
        assert app._write_volume_scene() is None

        item = next(i for i in app._emitted_items() if i["kind"] == "reflections")
        assert item["visible"] is None  # nothing to show or hide
        assert item["has_map_coefficients"] is False

        app.remove_reflections(app._reflections[0]["id"])
        assert app._reflections == []
    finally:
        app.stop()


def test_2fofc_is_not_a_difference_map():
    """The trap in the label table: "2FOFCWT" contains "FOFCWT", so a substring test
    calls the 2Fo-Fc map a difference map and contours the main map at 3 sigma in green.
    No file records which is which, so the table is unavoidable — matching it loosely
    is not."""
    from pxviewer.reflections import is_difference_map

    assert not is_difference_map("2FOFCWT,PH2FOFCWT")
    assert not is_difference_map("2FOFCWT_no_fill,PH2FOFCWT_no_fill")
    assert not is_difference_map("FWT,PHWT")            # refmac's regular map
    assert is_difference_map("FOFCWT,PHFOFCWT")         # phenix
    assert is_difference_map("FOFCWT_no_fill,PHFOFCWT_no_fill")
    assert is_difference_map("DELFWT,PHDELWT")          # refmac


def test_map_from_coefficients_is_sigma_scaled(tmp_path):
    """Contour levels are in sigma throughout the viewer, so a transformed map has to
    arrive on that scale: "1.5" must mean 1.5 standard deviations of this map."""
    import numpy as np

    from pxviewer.reflections import ReflectionData, map_from_coefficients

    data = ReflectionData.from_file(str(_mtz(tmp_path, coefficients=True)))
    mm = map_from_coefficients(data.map_coefficient_arrays()[0])
    grid = mm.map_data().as_numpy_array()
    assert grid.mean() == pytest.approx(0.0, abs=1e-6)
    assert grid.std() == pytest.approx(1.0, abs=1e-6)
    assert mm.map_data().origin() == (0, 0, 0)
    # Gridded at cctbx's default 1/3, which is Coot's default sampling too.
    assert grid.shape == (80, 72, 45)


def test_refinement_mtz_opens_its_maps(tmp_path):
    """A file carrying map coefficients is a refinement result and the density is what
    it is for, so the maps are made on load rather than asked about — Coot's Auto Open,
    and the reason it is how most people open an MTZ."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_file(str(_mtz(tmp_path, coefficients=True)))
        assert len(app._reflections) == 1
        assert len(app._volumes) == 2  # 2FOFCWT and FOFCWT, without being asked

        by_name = {v["name"]: v for v in app._volumes}
        assert set(by_name) == {"2FOFCWT", "FOFCWT"}
        # Convention, not the palette: blue at 1.5 sigma, difference green at 3.
        assert (by_name["2FOFCWT"]["color"], by_name["2FOFCWT"]["iso"]) == ("dodgerblue", 1.5)
        assert (by_name["FOFCWT"]["color"], by_name["FOFCWT"]["iso"]) == ("green", 3.0)

        # Data and maps are one group: the maps came from the file, and go with it.
        gid = app._reflections[0]["group"]
        assert gid is not None and all(v["group"] == gid for v in app._volumes)
        assert app.group_mmm(gid) is None  # no model, so cctbx paired nothing
        assert app._write_volume_scene() is not None  # they really are in the scene

        app.remove_group(gid)
        assert not app._volumes and not app._reflections
    finally:
        app.stop()


def test_maps_from_reflections_open_with_a_view_radius(tmp_path):
    """A map made from reflections fills the unit cell, so drawing all of it buries the
    model in density — Coot has a radius for exactly this. A map read from a file is
    already a box around its subject, so it gets none."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    import numpy as np
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pxviewer.desktop import _VIEW_RADIUS_DEFAULT, DesktopApp
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_file(str(_mtz(tmp_path, coefficients=True)))
        assert all(v["radius"] == _VIEW_RADIUS_DEFAULT for v in app._volumes)

        # A map from a file is already boxed: no radius.
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "cryoem")
        assert app._volume_entry(vid)["radius"] is None

        app.set_volume_radius(vid, 20.0)
        assert app.volume_appearance(vid)["radius"] == 20.0
        app.set_volume_radius(vid, None)
        assert app.volume_appearance(vid)["radius"] is None
    finally:
        app.stop()


def test_phased_maps_takes_a_live_model_and_scales(tmp_path):
    """Two things the density depends on. The model is a live object, not a filename —
    the viewer's model is often nowhere on disk (reduce2 built it, Minimize moved it),
    and recomputing density after it moves is why the reflections are kept. And the
    fmodel must be scaled: get_fmodel returns one that is not, and 2mFo-DFc from an
    unscaled fmodel is wrong in a way that looks plausible."""
    import numpy as np

    from pxviewer.cctbx_io import read_model
    from pxviewer.reflections import PHASED_MAP_TYPES, phased_maps

    model = read_model(str(MODEL))  # in memory only
    out = phased_maps(model, str(_mtz(tmp_path, coefficients=False)))

    assert set(out["maps"]) == set(PHASED_MAP_TYPES)
    assert 0.0 <= out["r_work"] <= 1.0 and 0.0 <= out["r_free"] <= 1.0
    for mm in out["maps"].values():
        grid = mm.map_data().as_numpy_array()
        assert grid.std() == pytest.approx(1.0, abs=1e-6)  # sigma-scaled, like the rest
        assert mm.is_compatible_model(model)               # in the model's frame


def test_2mfo_dfc_is_not_the_difference_map():
    """The same trap as the label table, in the map-type names: "2mFo-DFc" ends with
    "mFo-DFc", so a prefix or substring test styles the main map as a difference map."""
    from pxviewer.reflections import DIFFERENCE_MAP_TYPES

    assert "2mFo-DFc" not in DIFFERENCE_MAP_TYPES
    assert "mFo-DFc" in DIFFERENCE_MAP_TYPES


def test_making_maps_pairs_them_with_the_model_that_phased_them(tmp_path):
    """The phases came from the model, so the maps and the model are inseparable — one
    map_model_manager. Which is also what makes them usable together: masking, and
    minimizing into the density, work on X-ray maps the moment they exist."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    import time

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_file(str(_mtz(tmp_path, coefficients=False)))
        app.load_file(str(MODEL))
        rid, mid = app._reflections[0]["id"], app._models[0]["id"]
        assert [m["id"] for m in app.models_for_phasing()] == [mid]
        assert app._volumes == []  # amplitudes alone make nothing

        app.make_maps(rid, mid)
        deadline = time.time() + 90
        while not app._volumes and time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.05)
        QCoreApplication.processEvents()

        by_name = {v["name"]: v for v in app._volumes}
        assert set(by_name) == {"2mFo-DFc", "mFo-DFc"}
        assert by_name["2mFo-DFc"]["color"] == "dodgerblue"
        assert (by_name["mFo-DFc"]["color"], by_name["mFo-DFc"]["iso"]) == ("green", 3.0)

        gid = app._models[0]["group"]
        assert gid is not None and app._reflections[0]["group"] == gid
        mmm = app.group_mmm(gid)
        assert mmm is not None  # cctbx really paired them

        # The payoff: everything that needs a pair now works on X-ray maps.
        assert app.map_for_model(mid) is not None       # minimize into density
        assert app.can_mask_volume(by_name["2mFo-DFc"]["id"])  # mask around the model

        # The fit is reported, and a paired model is no longer on offer to phase again.
        assert app._reflections[0]["r_work"] is not None
        assert app.models_for_phasing() == []
        with pytest.raises(ValueError, match="already paired"):
            app.make_maps(rid, mid)
    finally:
        app.stop()


def test_a_data_mtz_makes_no_maps(tmp_path):
    """Amplitudes cannot become density on their own — the phases have to be computed
    against a model. Loading one draws nothing rather than guessing."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_file(str(_mtz(tmp_path, coefficients=False)))
        assert len(app._reflections) == 1
        assert app._volumes == []
        assert app._reflections[0]["group"] is None  # nothing to group it with
    finally:
        app.stop()
