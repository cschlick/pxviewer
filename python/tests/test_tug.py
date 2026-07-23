"""Tests for dragging an atom with the model giving way live."""

from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb"


def _require_restraints():
    pytest.importorskip("mmtbx.geometry_restraints.reference")
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")


def _model():
    from pxviewer.cctbx_io import read_model

    return read_model(str(MODEL))


def test_a_tug_pulls_it_does_not_teleport():
    """The atom arrives where the geometry lets it, not where the pointer is — that is
    the difference between dragging a model and editing coordinates."""
    _require_restraints()
    from pxviewer.tug import Tug

    model = _model()
    tug = Tug(model, 300)
    start = model.get_sites_cart().as_numpy_array().copy()  # after process(), see below
    target = start[300] + np.array([3.0, 0.0, 0.0])
    for i in range(20):  # as a pointer moves: in steps
        tug.move_to(start[300] + np.array([3.0 * (i + 1) / 20, 0.0, 0.0]))
    tug.finish()

    now = model.get_sites_cart().as_numpy_array()
    moved = np.linalg.norm(now - start, axis=1)
    assert 1.0 < moved[300] < 3.0            # it followed, but the geometry argued
    assert np.linalg.norm(now[300] - target) > 0.01
    assert (moved > 0.05).sum() > 10         # the neighbourhood gave way with it

    # And the model is still a model: strained, not torn.
    energies = model.get_restraints_manager().geometry.energies_sites(
        sites_cart=model.get_sites_cart(), compute_gradients=False)
    assert energies.bond_deviations()[2] < 0.1


def test_scope_modes_pick_the_right_atoms():
    """The drag scope decides what gives way: a sphere, a single residue, or a stretch of
    residues each side along the chain (Coot's refine scopes)."""
    _require_restraints()
    from pxviewer.tug import Tug

    model = _model()
    hierarchy = model.get_hierarchy()
    chain = list(list(hierarchy.models())[0].chains())[0]
    groups = list(chain.residue_groups())
    idx = 40
    atom = int(groups[idx].atoms().extract_i_seq()[0])

    def residues_in(indices):
        got = set(indices.tolist())
        return sum(1 for rg in groups
                   if set(np.asarray(rg.atoms().extract_i_seq(), int).tolist()) & got)

    single = Tug(model, atom, mode="residues", flank=0)
    assert residues_in(single.indices) == 1               # just the grabbed residue
    single.finish()

    stretch = Tug(model, atom, mode="residues", flank=2)
    assert residues_in(stretch.indices) == 5              # it and two each side
    # and it is a contiguous run in sequence, not a ball of neighbours
    expected = set()
    for j in range(idx - 2, idx + 3):
        expected |= set(np.asarray(groups[j].atoms().extract_i_seq(), int).tolist())
    assert set(stretch.indices.tolist()) == expected
    stretch.finish()

    sphere = Tug(model, atom, mode="sphere", radius=8.0)
    assert residues_in(sphere.indices) > 5                # the neighbourhood, more than a stretch
    sphere.finish()


def test_scope_stretch_clamps_at_the_chain_end():
    """A stretch near the start of a chain does not run off into the residue before it (or
    into a different chain block); it clamps."""
    _require_restraints()
    from pxviewer.tug import Tug

    model = _model()
    chain = list(list(model.get_hierarchy().models())[0].chains())[0]
    groups = list(chain.residue_groups())
    atom = int(groups[0].atoms().extract_i_seq()[0])  # first residue

    tug = Tug(model, atom, mode="residues", flank=3)
    got = set(tug.indices.tolist())
    # residue 0 plus up to 3 after it — never a residue "before" residue 0.
    reached = [i for i, rg in enumerate(groups)
               if set(np.asarray(rg.atoms().extract_i_seq(), int).tolist()) & got]
    assert reached == [0, 1, 2, 3]
    tug.finish()


def test_only_the_zone_moves_and_it_stays_attached():
    """Two things at once. The zone is what makes this interactive at all — its cost is
    its own size, not the model's — and grm.select drops every restraint reaching out of
    it, so without pinned boundary atoms the zone drifts off, edges first."""
    _require_restraints()
    from pxviewer.tug import Tug

    model = _model()
    tug = Tug(model, 300, radius=8.0)
    start = model.get_sites_cart().as_numpy_array().copy()
    assert tug.zone_size < len(start) / 4  # a fraction of the model, not all of it

    for i in range(20):
        tug.move_to(start[300] + np.array([3.0 * (i + 1) / 20, 0.0, 0.0]))
    tug.finish()

    now = model.get_sites_cart().as_numpy_array()
    outside = ~np.isin(np.arange(len(start)), tug._indices)
    assert np.linalg.norm(now[outside] - start[outside], axis=1).max() == 0.0

    # The zone stayed put rather than sailing off with the atom.
    drift = np.linalg.norm(now[tug._indices].mean(axis=0) - start[tug._indices].mean(axis=0))
    assert drift < 0.5


def test_density_is_what_makes_a_tug_correct_something():
    """Geometry cannot know where the atoms belong; density can. The map term is also
    the one that silently does nothing if you get it wrong — that lbfgs refines a copy
    and rebinds it, so handing it sites and hoping leaves them untouched."""
    _require_restraints()
    pytest.importorskip("iotbx.map_model_manager")
    from iotbx.map_model_manager import map_model_manager
    from scitbx.array_family import flex

    from pxviewer.tug import Tug

    mmm = map_model_manager(model=_model())
    mmm.generate_map(d_min=2.0)
    truth = mmm.model().get_sites_cart().as_numpy_array().copy()
    map_data = mmm.map_manager().map_data()

    from scitbx.array_family import flex
    flex.set_random_seed(0)  # the shake is random; a corrective case must be reproducible

    shaken = _model()
    xrs = shaken.get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=0.4)
    shaken_sites = xrs.sites_cart().as_numpy_array().copy()

    def jiggle(use_map):
        """The same drag, from the same start, with and without density."""
        model = _model()
        model.set_sites_cart(flex.vec3_double(shaken_sites))
        tug = Tug(model, 300, map_data=map_data if use_map else None, map_weight=50.0)
        zone = tug._indices
        rmsd = lambda: float(np.sqrt(
            ((model.get_sites_cart().as_numpy_array()[zone] - truth[zone]) ** 2).sum(axis=1).mean()))
        before = rmsd()
        here = model.get_sites_cart().as_numpy_array()[300]
        for i in range(25):
            tug.move_to(here + np.array([0.15 * np.sin(i / 3), 0.0, 0.0]))
        tug.finish()
        return before, rmsd()

    before_geom, after_geom = jiggle(False)
    before_map, after_map = jiggle(True)
    assert before_geom == pytest.approx(before_map)  # the same start, or this proves nothing

    # Geometry alone cannot improve on the truth it cannot see; density moves toward it.
    assert after_map < before_map - 0.05
    assert after_map < after_geom - 0.05


def test_stale_drag_targets_are_dropped_but_not_the_last_one():
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


def test_a_drag_from_the_viewport_reaches_a_handler():
    """The browser says which atom and where the pointer is; what the model does about
    it is cctbx's business, not the browser's."""
    import asyncio
    import json

    websockets = pytest.importorskip("websockets")
    from pxviewer.live import LiveSession

    session = LiveSession.from_sites([[0, 0, 0], [1, 0, 0]])
    session.start(port=0)
    seen = []
    session.on_tug(lambda action, atom, target: seen.append((action, atom, target)))
    try:
        async def scenario():
            url = f"ws://{session.host}:{session.port}"
            async with websockets.connect(url) as ws:
                await ws.recv()
                await ws.send(json.dumps({"type": "tug", "action": "begin", "atom": 1}))
                await ws.send(json.dumps({"type": "tug", "action": "move", "atom": 1,
                                          "target": [1.0, 2.0, 3.0]}))
                await ws.send(json.dumps({"type": "tug", "action": "end", "atom": 1}))
                for _ in range(50):
                    if len(seen) == 3:
                        break
                    await asyncio.sleep(0.05)

        asyncio.run(scenario())
        assert seen == [("begin", 1, None), ("move", 1, [1.0, 2.0, 3.0]), ("end", 1, None)]
    finally:
        session.stop()


def test_arm_from_the_viewport_reaches_a_handler():
    """Pressing Shift sends a 'arm' message with no atom (the drag hasn't grabbed anything
    yet). It must still reach the tug handler — that is what lets the app stop a running
    minimization the instant Shift goes down, before the pointer grabs."""
    import asyncio
    import json

    websockets = pytest.importorskip("websockets")
    from pxviewer.live import LiveSession

    session = LiveSession.from_sites([[0, 0, 0], [1, 0, 0]])
    session.start(port=0)
    seen = []
    session.on_tug(lambda action, atom, target: seen.append((action, atom, target)))
    try:
        async def scenario():
            url = f"ws://{session.host}:{session.port}"
            async with websockets.connect(url) as ws:
                await ws.recv()
                await ws.send(json.dumps({"type": "tug", "action": "arm"}))
                for _ in range(50):
                    if seen:
                        break
                    await asyncio.sleep(0.05)

        asyncio.run(scenario())
        assert seen == [("arm", -1, None)]
    finally:
        session.stop()


def test_continuous_mode_keeps_minimizing_between_targets():
    """The difference the mode makes: with the target held still, the free-running steps
    keep reducing the strain, where a single nudge would have stopped. That is what lets
    a held-still drag keep settling instead of freezing at the first thing it reached."""
    _require_restraints()
    from pxviewer.tug import Tug

    model = _model()
    xrs = model.get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=0.4)
    model.set_sites_cart(xrs.sites_cart())

    tug = Tug(model, 300)
    # Aim at the atom's own position and never move it — only the free-run does anything.
    tug.set_target(model.get_sites_cart().as_numpy_array()[300])
    after_one = tug.step().copy()
    for _ in range(20):
        after_many = tug.step()
    tug.finish()

    # It kept moving the model with no new target — a single step had not reached the
    # bottom. (A non-continuous drag held still would have stopped at `after_one`, which
    # is why the wire de-dups: see DesktopApp._push_tug.)
    zone = tug._indices
    assert np.linalg.norm(after_many[zone] - after_one[zone], axis=1).max() > 0.02


def test_restraints_are_built_once_not_per_drag():
    """Processing a model costs seconds. Doing it on every begin is a freeze at the
    start of every drag, which is exactly when it is least affordable."""
    _require_restraints()
    from pxviewer.tug import Tug

    model = _model()
    assert not model.restraints_manager_available()
    Tug(model, 300).finish()
    assert model.restraints_manager_available()

    # A second drag reuses them rather than rebuilding.
    grm = model.get_restraints_manager()
    Tug(model, 320).finish()
    assert model.get_restraints_manager() is grm


def test_desktop_scope_reaches_the_tug():
    """The Settings 'Moves:' control sets the scope, and a drag started afterwards builds a
    Tug with it — a single residue moves far fewer atoms than the default sphere."""
    _require_restraints()
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_file(str(MODEL))
        mid = app._models[0]["id"]

        app.set_tug_scope(mode="sphere", radius=8.0)
        app._serve_tug(mid, "begin", 300, None)
        sphere_zone = app._tug.zone_size
        app._serve_tug(mid, "end", 300, None)

        app.set_tug_scope(mode="residues", flank=0)
        app._serve_tug(mid, "begin", 300, None)
        single_zone = app._tug.zone_size
        app._serve_tug(mid, "end", 300, None)

        assert single_zone < sphere_zone            # one residue is fewer atoms than a sphere
        assert app._tug is None                     # cleaned up after each drag
    finally:
        app.stop()


def test_desktop_continuous_free_runs_and_dedups(qapp=None):
    """The desktop wiring: in continuous mode the worker keeps stepping with no new
    message (so a held-still drag settles), and identical frames are not re-sent (so a
    converged geometry drag does not spam the wire)."""
    _require_restraints()
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_file(str(MODEL))
        mid = app._models[0]["id"]
        session = app._models[0]["session"]
        pushed = []
        # push takes an optional `changed` (the moved-atom set for delta frames); accept and
        # forward it so the mock matches the real signature.
        session.push = lambda c, changed=None, _p=session.push: (pushed.append(1), _p(c, changed=changed))[1]

        app.set_tug_continuous(True)
        start = session.model.get_sites_cart().as_numpy_array()[300].copy()
        app._serve_tug(mid, "begin", 300, None)
        app._serve_tug(mid, "move", 300, (start + [3.0, 0.0, 0.0]).tolist())

        # Free-run with no new message: the model keeps moving while it has somewhere
        # to go — this is what a held-still drag does in continuous mode.
        n_before = len(pushed)
        for _ in range(20):
            app._tug_relax()
        assert len(pushed) - n_before > 0     # it kept stepping with no new target

        # A frame identical to the last is not resent: once a drag has truly settled,
        # re-pushing the same conformation 30 times a second is pointless wire traffic.
        settled = len(pushed)
        last = app._tug_last
        app._push_tug(last.copy())            # same coordinates again
        assert len(pushed) == settled

        app._serve_tug(mid, "end", 300, None)
        assert app._tug is None
    finally:
        app.stop()


def test_settle_comes_to_rest_in_place():
    """A released fling should visibly wind down to rest, not stop dead mid-motion. The
    settle is one continuous minimization (not stepped restarts, which jitter): it holds
    the atom where it was let go and relaxes the fragment, decelerating to a stop."""
    _require_restraints()
    import numpy as np
    from pxviewer.tug import Tug

    model = _model()
    tug = Tug(model, 300)
    start = model.get_sites_cart().as_numpy_array()[300].copy()
    tug.set_target((start + [4.0, 0.0, 0.0]).tolist())
    tug.step()  # fling partway
    released = model.get_sites_cart().as_numpy_array()[300].copy()

    frames = []
    tug.settle(on_frame=lambda c: frames.append(c.copy()))
    tug.finish()

    assert len(frames) > 20  # a real wind-down, not one jump
    zone = tug._indices
    motion = np.linalg.norm(np.diff(np.stack(frames), axis=0), axis=2)[:, zone].max(axis=1)
    assert motion[-1] < 0.02          # decelerated to rest
    assert motion[-1] < motion[0]     # it actually slowed down

    # It settled where it was let go, not on toward the 4 A pull.
    final = model.get_sites_cart().as_numpy_array()[300]
    assert np.linalg.norm(final - released) < 0.5
    assert np.linalg.norm(final - start) > 1.0  # the drag was kept, not undone


def test_a_standalone_ligand_with_no_boundary_can_be_dragged():
    """A small, self-contained ligand (a placed monomer) is a whole model with nothing
    around it, so the drag zone reaches no boundary atoms to pin. That left the reference
    restraint list uninitialised, and re-aiming the pull dereferenced it — a crash on the
    first move, so a placed ligand could not be dragged at all. Here the atom follows and
    the rest of the ligand comes with it via geometry."""
    _require_restraints()
    import numpy as np
    from pxviewer import ligands
    from pxviewer.tug import Tug

    model = ligands.build_ligand_model("GOL", (5.0, 5.0, 5.0))
    start = model.get_sites_cart().as_numpy_array().copy()
    tug = Tug(model, 0)
    for i in range(10):  # the move that used to raise on the first frame
        tug.move_to((start[0] + np.array([1.5 * (i + 1) / 10, 0.0, 0.0])).tolist())
    tug.finish()

    now = model.get_sites_cart().as_numpy_array()
    moved = np.linalg.norm(now - start, axis=1)
    assert moved[0] > 0.5                       # the dragged atom followed
    assert (moved > 0.05).sum() == len(moved)   # the whole ligand came along
