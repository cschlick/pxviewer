"""Dragging an atom with the model giving way live.

A tug is a real-time minimization: the pointer sets a reference target, cctbx relaxes a
zone around it, and what the user sees is the geometry arguing back. The desktop wiring
around it -- scopes, continuous mode, frame de-duplication -- is in ``tst_tug_gui.py``.
"""

from __future__ import absolute_import, division, print_function

import sys

from libtbx.test_utils import approx_equal

from pxviewer.regression.tst_utils import data_path, have, skip

if not have("mmtbx.geometry_restraints.reference", "numpy"):
    skip("mmtbx.geometry_restraints.reference not available")

import numpy as np                                   # noqa: E402

from pxviewer.geometry import monomer_library_available   # noqa: E402

if not monomer_library_available():
    skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

MODEL = data_path("1ubq.pdb")

#: A mid-model atom, well away from either chain end.
ATOM = 300


def model():
    """A **fresh** model each call. Every exercise here moves atoms, and ``Tug`` also
    attaches a restraints manager to whatever it is handed."""
    from pxviewer.cctbx_io import read_model

    return read_model(MODEL)


def sites(model):
    return model.get_sites_cart().as_numpy_array()


def residue_groups(model):
    return list(list(list(model.get_hierarchy().models())[0].chains())[0].residue_groups())


def iseqs(residue_group):
    return set(np.asarray(residue_group.atoms().extract_i_seq(), int).tolist())


def residues_touched(groups, indices):
    got = set(np.asarray(indices, int).tolist())
    return [i for i, rg in enumerate(groups) if iseqs(rg) & got]


# -- what a tug does ----------------------------------------------------------


def exercise_a_tug_pulls_it_does_not_teleport():
    """The atom arrives where the geometry lets it, not where the pointer is -- that is
    the difference between dragging a model and editing coordinates."""
    from pxviewer.tug import Tug

    m = model()
    tug = Tug(m, ATOM)
    start = sites(m).copy()                  # after Tug processed it; see below
    target = start[ATOM] + np.array([3.0, 0.0, 0.0])

    for i in range(20):                      # as a pointer moves: in steps
        tug.move_to(start[ATOM] + np.array([3.0 * (i + 1) / 20, 0.0, 0.0]))
    tug.finish()

    now = sites(m)
    moved = np.linalg.norm(now - start, axis=1)
    assert 1.0 < moved[ATOM] < 3.0                       # it followed, but geometry argued
    assert np.linalg.norm(now[ATOM] - target) > 0.01     # it did not reach the pointer
    assert (moved > 0.05).sum() > 10                     # the neighbourhood gave way too

    # And the model is still a model: strained, not torn.
    energies = m.get_restraints_manager().geometry.energies_sites(
        sites_cart=m.get_sites_cart(), compute_gradients=False)
    assert energies.bond_deviations()[2] < 0.1


def exercise_only_the_zone_moves_and_it_stays_attached():
    """Two things at once.

    The zone is what makes this interactive at all -- the cost is its own size rather
    than the model's. And ``grm.select`` drops every restraint reaching out of it, so
    without pinned boundary atoms the zone drifts off, edges first.
    """
    from pxviewer.tug import Tug

    m = model()
    tug = Tug(m, ATOM, radius=8.0)
    start = sites(m).copy()
    assert tug.zone_size < len(start) / 4        # a fraction of the model, not all of it

    for i in range(20):
        tug.move_to(start[ATOM] + np.array([3.0 * (i + 1) / 20, 0.0, 0.0]))
    tug.finish()

    now = sites(m)
    outside = ~np.isin(np.arange(len(start)), tug._indices)
    assert np.linalg.norm(now[outside] - start[outside], axis=1).max() == 0.0

    # The zone stayed put rather than sailing off with the atom.
    drift = np.linalg.norm(
        now[tug._indices].mean(axis=0) - start[tug._indices].mean(axis=0))
    assert drift < 0.5


def exercise_restraints_are_built_once_not_per_drag():
    """Processing a model costs seconds. Doing it on every begin is a freeze at the start
    of every drag, which is exactly when it is least affordable."""
    from pxviewer.tug import Tug

    m = model()
    assert not m.restraints_manager_available()
    Tug(m, ATOM).finish()
    assert m.restraints_manager_available()

    grm = m.get_restraints_manager()
    Tug(m, 320).finish()                             # a second drag reuses them
    assert m.get_restraints_manager() is grm


# -- drag scopes --------------------------------------------------------------


def exercise_scope_modes_pick_the_right_atoms():
    """The scope decides what gives way: a sphere, a single residue, or a stretch of
    residues each side along the chain -- Coot's refine scopes."""
    from pxviewer.tug import Tug

    m = model()
    groups = residue_groups(m)
    index = 40
    atom = int(groups[index].atoms().extract_i_seq()[0])

    single = Tug(m, atom, mode="residues", flank=0)
    assert len(residues_touched(groups, single.indices)) == 1     # just the grabbed one
    single.finish()

    stretch = Tug(m, atom, mode="residues", flank=2)
    assert len(residues_touched(groups, stretch.indices)) == 5    # it and two each side
    # And it is a contiguous run in sequence, not a ball of neighbours.
    expected = set()
    for j in range(index - 2, index + 3):
        expected |= iseqs(groups[j])
    assert set(stretch.indices.tolist()) == expected
    stretch.finish()

    sphere = Tug(m, atom, mode="sphere", radius=8.0)
    assert len(residues_touched(groups, sphere.indices)) > 5      # more than a stretch
    sphere.finish()


def exercise_scope_stretch_clamps_at_the_chain_end():
    """A stretch near the start of a chain must not run off into the residue before it,
    or into a different chain block. It clamps."""
    from pxviewer.tug import Tug

    m = model()
    groups = residue_groups(m)
    atom = int(groups[0].atoms().extract_i_seq()[0])          # the first residue

    tug = Tug(m, atom, mode="residues", flank=3)
    assert residues_touched(groups, tug.indices) == [0, 1, 2, 3]
    tug.finish()


def exercise_scope_selection_is_exactly_the_picked_residues():
    """Selection scope moves the residues the user picked -- an arbitrary set, and
    nothing else -- whole-residue-expanded from whatever atoms were in the selection."""
    from pxviewer.tug import Tug

    m = model()
    groups = residue_groups(m)

    # A few atoms from two non-adjacent residues.
    selection = (list(np.asarray(groups[10].atoms().extract_i_seq(), int)[:3])
                 + list(np.asarray(groups[50].atoms().extract_i_seq(), int)[:2]))
    atom = int(groups[10].atoms().extract_i_seq()[0])

    tug = Tug(m, atom, mode="selection", selection=selection)
    assert set(tug.indices.tolist()) == iseqs(groups[10]) | iseqs(groups[50])
    tug.finish()

    # Grabbing an atom outside the selection still works: its residue joins the zone.
    empty = Tug(m, atom, mode="selection", selection=[])
    assert set(empty.indices.tolist()) == iseqs(groups[10])
    empty.finish()


def exercise_a_standalone_ligand_with_no_boundary_can_be_dragged():
    """A placed monomer is a whole model with nothing around it, so the drag zone reaches
    no boundary atoms to pin.

    That left the reference restraint list uninitialised, and re-aiming the pull
    dereferenced it -- a crash on the first move, so a placed ligand could not be dragged
    at all.
    """
    from pxviewer import ligands
    from pxviewer.tug import Tug

    m = ligands.build_ligand_model("GOL", (5.0, 5.0, 5.0))
    start = sites(m).copy()

    tug = Tug(m, 0)
    for i in range(10):                      # the move that used to raise on frame one
        tug.move_to((start[0] + np.array([1.5 * (i + 1) / 10, 0.0, 0.0])).tolist())
    tug.finish()

    moved = np.linalg.norm(sites(m) - start, axis=1)
    assert moved[0] > 0.5                            # the dragged atom followed
    assert (moved > 0.05).sum() == len(moved)        # the whole ligand came along


# -- density and settling -----------------------------------------------------


def exercise_density_is_what_makes_a_tug_correct_something():
    """Geometry cannot know where the atoms belong; density can.

    The map term is also the one that silently does nothing when it is wired up wrongly:
    lbfgs refines a copy and rebinds it, so handing it sites and hoping leaves them
    untouched -- and a tug that ignores the map looks exactly like one that uses it.
    """
    if not have("iotbx.map_model_manager"):
        print("    (skipped: iotbx.map_model_manager not available)")
        return
    from iotbx.map_model_manager import map_model_manager
    from scitbx.array_family import flex

    from pxviewer.tug import Tug

    mmm = map_model_manager(model=model())
    mmm.generate_map(d_min=2.0)
    truth = sites(mmm.model()).copy()
    map_data = mmm.map_manager().map_data()

    flex.set_random_seed(0)      # the shake is random; a corrective case must repeat
    xrs = model().get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=0.4)
    shaken = xrs.sites_cart().as_numpy_array().copy()

    def jiggle(use_map):
        """The same drag, from the same start, with and without density."""
        m = model()
        m.set_sites_cart(flex.vec3_double(shaken))
        tug = Tug(m, ATOM, map_data=map_data if use_map else None, map_weight=50.0)
        zone = tug._indices

        def rmsd():
            delta = sites(m)[zone] - truth[zone]
            return float(np.sqrt((delta ** 2).sum(axis=1).mean()))

        before = rmsd()
        here = sites(m)[ATOM]
        for i in range(25):
            tug.move_to(here + np.array([0.15 * np.sin(i / 3), 0.0, 0.0]))
        tug.finish()
        return before, rmsd()

    before_geometry, after_geometry = jiggle(False)
    before_map, after_map = jiggle(True)
    assert approx_equal(before_geometry, before_map)   # same start, or this proves nothing

    # Geometry alone cannot improve on a truth it cannot see; density moves towards it.
    assert after_map < before_map - 0.05
    assert after_map < after_geometry - 0.05


def exercise_continuous_mode_keeps_minimizing_between_targets():
    """With the target held still, the free-running steps keep reducing the strain where
    a single nudge would have stopped. That is what lets a held-still drag keep settling
    instead of freezing at the first thing it reached."""
    from pxviewer.tug import Tug

    m = model()
    xrs = m.get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=0.4)
    m.set_sites_cart(xrs.sites_cart())

    tug = Tug(m, ATOM)
    # Aim at the atom's own position and never move it, so only the free-run does work.
    tug.set_target(sites(m)[ATOM])
    after_one = tug.step().copy()
    for _ in range(20):
        after_many = tug.step()
    tug.finish()

    zone = tug._indices
    assert np.linalg.norm(after_many[zone] - after_one[zone], axis=1).max() > 0.02


def exercise_settle_comes_to_rest_in_place():
    """A released fling winds down to rest rather than stopping dead mid-motion.

    One continuous minimization, not stepped restarts -- those jitter. It holds the atom
    where it was let go and relaxes the fragment around it, decelerating to a stop.
    """
    from pxviewer.tug import Tug

    m = model()
    tug = Tug(m, ATOM)
    start = sites(m)[ATOM].copy()
    tug.set_target((start + [4.0, 0.0, 0.0]).tolist())
    tug.step()                                   # fling partway
    released = sites(m)[ATOM].copy()

    frames = []
    tug.settle(on_frame=lambda c: frames.append(c.copy()))
    tug.finish()

    assert len(frames) > 20                      # a real wind-down, not one jump
    zone = tug._indices
    motion = np.linalg.norm(np.diff(np.stack(frames), axis=0), axis=2)[:, zone].max(axis=1)
    assert motion[-1] < 0.02                     # decelerated to rest
    assert motion[-1] < motion[0]                # and it actually slowed down

    final = sites(m)[ATOM]
    assert np.linalg.norm(final - released) < 0.5    # settled where it was let go ...
    assert np.linalg.norm(final - start) > 1.0       # ... and the drag was kept, not undone


# -- the wire -----------------------------------------------------------------


def exercise_stale_drag_targets_are_dropped_but_not_the_last_one():
    """The pointer outruns cctbx, so every target but the newest is somewhere it has
    already left. Only *runs* of targets collapse: an end between them is a different
    thing being said, and the target before a release is where the user let go."""
    from pxviewer.desktop import _collapse_moves

    drag = [("begin", "m", 1, None), ("move", "m", 1, "a"), ("move", "m", 1, "b"),
            ("move", "m", 1, "c"), ("end", "m", 1, None)]
    assert [(i[0], i[3]) for i in _collapse_moves(drag)] == [
        ("begin", None), ("move", "c"), ("end", None)]

    # Two drags in one batch: neither loses its own last target.
    two = [("move", "m", 1, "a"), ("end", "m", 1, None),
           ("begin", "m", 2, None), ("move", "m", 2, "z")]
    assert [(i[0], i[3]) for i in _collapse_moves(two)] == [
        ("move", "a"), ("end", None), ("begin", None), ("move", "z")]


def collect_tug_messages(sent, expected_count):
    """Start a session, send ``sent`` over the socket, and return what the handler saw."""
    import asyncio
    import json

    import websockets

    from pxviewer.live import LiveSession

    session = LiveSession.from_sites([[0, 0, 0], [1, 0, 0]])
    session.start(port=0)
    seen = []
    session.on_tug(lambda action, atom, target: seen.append((action, atom, target)))
    try:
        async def scenario():
            url = "ws://%s:%d" % (session.host, session.port)
            async with websockets.connect(url) as ws:
                await ws.recv()                      # topology arrives first
                for message in sent:
                    await ws.send(json.dumps(message))
                for _ in range(50):
                    if len(seen) == expected_count:
                        break
                    await asyncio.sleep(0.05)

        asyncio.run(scenario())
    finally:
        session.stop()
    return seen


def exercise_a_drag_from_the_viewport_reaches_a_handler():
    """The browser says which atom and where the pointer is; what the model does about it
    is cctbx's business, not the browser's."""
    if not have("websockets"):
        print("    (skipped: websockets not available)")
        return
    seen = collect_tug_messages([
        {"type": "tug", "action": "begin", "atom": 1},
        {"type": "tug", "action": "move", "atom": 1, "target": [1.0, 2.0, 3.0]},
        {"type": "tug", "action": "end", "atom": 1},
    ], expected_count=3)

    assert seen == [("begin", 1, None),
                    ("move", 1, [1.0, 2.0, 3.0]),
                    ("end", 1, None)]


def exercise_arm_from_the_viewport_reaches_a_handler():
    """Pressing Shift sends an 'arm' with no atom, since the drag has grabbed nothing
    yet. It must still reach the tug handler -- that is what lets the app stop a running
    minimization the instant Shift goes down, before the pointer grabs."""
    if not have("websockets"):
        print("    (skipped: websockets not available)")
        return
    seen = collect_tug_messages([{"type": "tug", "action": "arm"}], expected_count=1)
    assert seen == [("arm", -1, None)]


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
