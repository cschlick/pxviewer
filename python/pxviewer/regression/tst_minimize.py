"""Geometry minimization, and the intermediate states that make it watchable.

Two targets: restraints alone, and restraints balanced against a map. Both stream every
conformation they pass through, which is what lets the viewer animate a run rather than
cut to the answer -- so the streaming is checked as carefully as the result.
"""

from __future__ import absolute_import, division, print_function

import sys

from pxviewer.regression.tst_utils import data_path, have, skip

if not have("iotbx.data_manager", "mmtbx.refinement.geometry_minimization", "numpy"):
    skip("iotbx.data_manager / mmtbx.refinement.geometry_minimization not available")

import numpy as np                                   # noqa: E402

from pxviewer.geometry import monomer_library_available   # noqa: E402

if not monomer_library_available():
    skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

UBIQUITIN = data_path("1ubq.pdb")

#: Enough displacement that the restraints have real work to do, small enough that the
#: minimizer still finds its way back.
SHAKE_A = 0.3

#: Near-ideal bonds. mmtbx reports rmsd in Angstroms; a converged run lands well under.
IDEAL_BOND_RMSD = 0.05


def shaken_model(distance=SHAKE_A):
    """1UBQ with its coordinates shaken, so there is geometry to relax."""
    from iotbx.data_manager import DataManager

    dm = DataManager()
    dm.process_model_file(UBIQUITIN)
    model = dm.get_model()
    xrs = model.get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=distance)
    model.set_sites_cart(xrs.sites_cart())
    return model


def shaken_model_and_map(distance=SHAKE_A, d_min=3.0):
    """A model with a density computed from it -- in a common frame -- and then shaken.

    Computing the map first means the density describes the *unshaken* coordinates, so it
    carries the answer the restraints cannot know.
    """
    from iotbx.data_manager import DataManager
    from iotbx.map_model_manager import map_model_manager

    dm = DataManager()
    dm.process_model_file(UBIQUITIN)
    mmm = map_model_manager(model=dm.get_model())
    mmm.generate_map(d_min=d_min)

    model = mmm.model()
    xrs = model.get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=distance)
    model.set_sites_cart(xrs.sites_cart())
    return model, mmm.map_manager().map_data()


def sites(model):
    return model.get_sites_cart().as_numpy_array()


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


# -- restraints only ----------------------------------------------------------


def exercise_minimize_geometry_improves_and_updates_the_model():
    from pxviewer.minimize import minimize_geometry

    model = shaken_model()
    before = sites(model).copy()
    stats = minimize_geometry(model)

    assert stats["bonds_after"] < stats["bonds_before"]
    assert stats["angles_after"] < stats["angles_before"]
    assert stats["bonds_after"] < IDEAL_BOND_RMSD
    # Minimized in place, so the tables, validation and Write all follow along.
    assert not np.array_equal(before, sites(model))


def exercise_minimize_streams_intermediate_states():
    """cctbx hands each intermediate conformation to the states collector, which is what
    makes the run watchable rather than a jump to the answer."""
    from pxviewer.minimize import minimize_geometry

    model = shaken_model()
    frames = []
    stats = minimize_geometry(model, on_state=frames.append)

    assert len(frames) > 10                             # a run, not a single jump
    assert stats["n_sent"] == len(frames) - 1           # every state, plus a forced final
    assert all(f.shape == (model.get_number_of_atoms(), 3) for f in frames)
    assert not np.array_equal(frames[0], frames[-1])    # it moved
    # And it lands on the real answer, not one step short of it.
    assert np.allclose(frames[-1], sites(model))


def exercise_stride_thins_the_stream_but_keeps_the_final_state():
    """Streaming every step of a 500-step run floods the socket; the last one still has
    to arrive, or the viewer is left a step behind the model."""
    from pxviewer.minimize import minimize_geometry

    model = shaken_model()
    frames = []
    stats = minimize_geometry(model, on_state=frames.append, stride=10)

    assert stats["n_sent"] < stats["n_states"] / 5      # meaningfully thinned
    assert np.allclose(frames[-1], sites(model))


def exercise_minimize_can_be_halted_and_keeps_the_progress_so_far():
    """Stop is a shorter run, not a discarded one: the model stays where it got to.
    scitbx.lbfgs halts when callback_after_step returns True."""
    from pxviewer.minimize import minimize_geometry

    model = shaken_model()
    frames = []
    stats = minimize_geometry(
        model, on_state=frames.append, should_stop=lambda: len(frames) >= 20)

    assert stats["stopped"] is True
    assert stats["n_states"] < 100                      # cut short; a full run is ~500
    # Partway down the hill but definitively on the way: better than the shaken start.
    assert stats["bonds_after"] < stats["bonds_before"]
    assert np.allclose(frames[-1], sites(model))


def exercise_minimize_runs_to_completion_when_never_asked_to_stop():
    from pxviewer.minimize import minimize_geometry

    stats = minimize_geometry(shaken_model(), should_stop=lambda: False)

    assert stats["stopped"] is False
    assert stats["bonds_after"] < IDEAL_BOND_RMSD       # converged, not cut short


# -- against a map ------------------------------------------------------------


def exercise_minimize_to_map_improves_geometry_and_streams():
    from pxviewer.minimize import minimize_to_map

    model, map_data = shaken_model_and_map()
    frames = []
    stats = minimize_to_map(model, map_data, on_state=frames.append)

    assert stats["bonds_after"] < stats["bonds_before"]
    assert stats["bonds_after"] < IDEAL_BOND_RMSD
    assert stats["weight"] > 0                          # cctbx derived the balance
    assert len(frames) > 10                             # watchable, not a jump
    assert np.allclose(frames[-1], sites(model))


def exercise_minimize_to_map_pulls_the_model_back_towards_the_density():
    """The point of the map target. The shaken model returns towards the coordinates the
    density was computed from -- which restraints alone have no way to know."""
    from pxviewer.minimize import minimize_geometry, minimize_to_map

    truth = sites(shaken_model_and_map(distance=0.0)[0])

    with_map, map_data = shaken_model_and_map()
    minimize_to_map(with_map, map_data)
    map_rmsd = rmsd(sites(with_map), truth)

    no_map = shaken_model_and_map()[0]
    minimize_geometry(no_map)
    geometry_rmsd = rmsd(sites(no_map), truth)

    assert map_rmsd < geometry_rmsd, (map_rmsd, geometry_rmsd)


def exercise_minimize_to_map_can_be_halted():
    """The map minimizer has no stop hook of its own, so the states collector unwinds it
    -- and the model still lands on the conformation it had reached."""
    from pxviewer.minimize import minimize_to_map

    model, map_data = shaken_model_and_map()
    frames = []
    stats = minimize_to_map(
        model, map_data, on_state=frames.append,
        should_stop=lambda: len(frames) >= 15)

    assert stats["stopped"] is True
    assert stats["n_states"] < 100                      # cut short; a full run is ~150
    assert stats["bonds_after"] < stats["bonds_before"]  # progress kept, not discarded
    assert np.allclose(frames[-1], sites(model))


def exercise_minimize_dispatches_on_whether_a_map_is_given():
    from pxviewer.minimize import minimize

    assert minimize(shaken_model())["weight"] is None    # restraints only
    model, map_data = shaken_model_and_map()
    assert minimize(model, map_data=map_data)["weight"] > 0     # map target


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
