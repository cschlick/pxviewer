"""Reading X-ray reflections, and turning them into density.

The pure-cctbx half: recognising what kind of MTZ is in hand, reading its metadata, and
computing maps from it. The desktop side of the same feature -- what loading one does to
the scene -- is in ``tst_reflections_gui.py``, which needs Qt.

Two kinds of MTZ exist in practice and the difference decides everything downstream: a
refinement file carries map coefficients and is density already, while a data file
carries amplitudes and needs a model before it is anything at all.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import os
import sys

from libtbx.test_utils import approx_equal

from pxviewer.regression.tst_utils import data_path, have, skip, tmp_dir

if not have("iotbx.data_manager", "numpy"):
    skip("iotbx.data_manager / numpy not available")

import numpy as np                                   # noqa: E402

MODEL = data_path("1ubq.pdb")
D_MIN = 2.0


@contextlib.contextmanager
def mtz(coefficients):
    """Write one of the two kinds of MTZ and yield its path.

    ``coefficients=True`` gives a refinement file (2FOFCWT and FOFCWT); ``False`` gives a
    data file (amplitudes with sigmas, plus free flags).
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


# -- recognising the file -----------------------------------------------------


def exercise_file_kind_recognises_reflections():
    from pxviewer.loader import file_kind

    assert file_kind("data.mtz") == "reflections"
    assert file_kind("model.pdb") == "model"
    assert file_kind("map.mrc") == "volume"


def exercise_map_coefficients_are_cctbxs_call_not_ours():
    """Whether density needs a model is the fork the whole feature turns on, and cctbx
    answers it: map_coefficients is a child datatype of miller_array, so the DataManager
    has already separated a refinement file from a data file. Column names are never
    read here -- there is no agreed vocabulary for them across programs."""
    from pxviewer.reflections import ReflectionData

    with mtz(coefficients=True) as path:
        refinement = ReflectionData.from_file(path)
        assert refinement.has_map_coefficients
        assert len(refinement.map_coefficient_arrays()) == 2      # 2FOFCWT and FOFCWT

    with mtz(coefficients=False) as path:
        data = ReflectionData.from_file(path)
        assert not data.has_map_coefficients
        assert data.map_coefficient_arrays() == []
        assert "F,SIGF" in data.labels


def exercise_reflection_metadata():
    from pxviewer.reflections import ReflectionData

    with mtz(coefficients=False) as path:
        data = ReflectionData.from_file(path)

        d_max, d_min = data.resolution_range
        assert approx_equal(d_min, D_MIN, eps=0.01)
        assert d_max > d_min
        assert data.n_reflections > 1000
        assert approx_equal(
            data.crystal_symmetry.unit_cell().parameters()[0], 50.84, eps=0.01)
        assert "amplitudes" in data.summary()


# -- naming traps -------------------------------------------------------------


def exercise_2fofc_is_not_a_difference_map():
    """The trap in the label table: "2FOFCWT" contains "FOFCWT", so a substring test
    calls the 2Fo-Fc map a difference map and contours the main map at 3 sigma in green.

    No file records which is which, so the table is unavoidable -- matching it loosely
    is not.
    """
    from pxviewer.reflections import is_difference_map

    assert not is_difference_map("2FOFCWT,PH2FOFCWT")
    assert not is_difference_map("2FOFCWT_no_fill,PH2FOFCWT_no_fill")
    assert not is_difference_map("FWT,PHWT")               # refmac's regular map
    assert is_difference_map("FOFCWT,PHFOFCWT")            # phenix
    assert is_difference_map("FOFCWT_no_fill,PHFOFCWT_no_fill")
    assert is_difference_map("DELFWT,PHDELWT")             # refmac


def exercise_2mfo_dfc_is_not_the_difference_map():
    """The same trap in the map-type names: "2mFo-DFc" ends with "mFo-DFc", so a prefix
    or substring test styles the main map as a difference map."""
    from pxviewer.reflections import DIFFERENCE_MAP_TYPES

    assert "2mFo-DFc" not in DIFFERENCE_MAP_TYPES
    assert "mFo-DFc" in DIFFERENCE_MAP_TYPES


# -- making density -----------------------------------------------------------


def exercise_a_map_from_coefficients_is_sigma_scaled():
    """Contour levels are in sigma throughout the viewer, so a transformed map has to
    arrive on that scale: "1.5" must mean 1.5 standard deviations of this map."""
    from pxviewer.reflections import ReflectionData, map_from_coefficients

    with mtz(coefficients=True) as path:
        data = ReflectionData.from_file(path)
        mm = map_from_coefficients(data.map_coefficient_arrays()[0])
        grid = mm.map_data().as_numpy_array()

        assert approx_equal(grid.mean(), 0.0, eps=1e-6)
        assert approx_equal(grid.std(), 1.0, eps=1e-6)
        assert mm.map_data().origin() == (0, 0, 0)
        # Gridded at cctbx's default 1/3, which is Coot's default sampling too.
        assert grid.shape == (80, 72, 45)


def exercise_phased_maps_take_a_live_model_and_scale_it():
    """Two things the density depends on.

    The model is a live object rather than a filename -- the viewer's model is often
    nowhere on disk, because reduce2 built it or Minimize moved it, and recomputing
    density after it moves is the reason the reflections are kept at all.

    And the fmodel has to be scaled: ``get_fmodel`` returns one that is not, and a
    2mFo-DFc map from an unscaled fmodel is wrong in a way that looks entirely plausible.
    """
    from pxviewer.cctbx_io import read_model
    from pxviewer.reflections import PHASED_MAP_TYPES, phased_maps

    model = read_model(MODEL)                          # in memory only
    with mtz(coefficients=False) as path:
        out = phased_maps(model, path)

    assert set(out["maps"]) == set(PHASED_MAP_TYPES)
    assert 0.0 <= out["r_work"] <= 1.0
    assert 0.0 <= out["r_free"] <= 1.0
    for name, mm in sorted(out["maps"].items()):
        grid = mm.map_data().as_numpy_array()
        assert approx_equal(grid.std(), 1.0, eps=1e-6), name    # sigma-scaled, like the rest
        assert mm.is_compatible_model(model), name             # and in the model's frame


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
