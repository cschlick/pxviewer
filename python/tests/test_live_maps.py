"""LiveDifferenceMap: the warm real-time difference-map recompute.

Verifies the compute is correct (a moved atom lights up the difference map exactly where it
moved) and honest (frozen scales, so R-free is a fixed reference), which is the feasibility
question for recomputing density as a model moves.
"""

from __future__ import annotations

import numpy as np
import pytest


def _synthetic_mtz(model, d_min, out):
    f_obs = abs(model.get_xray_structure().structure_factors(d_min=d_min).f_calc())
    f_obs.set_observation_type_xray_amplitude()
    rfree = f_obs.generate_r_free_flags()
    ds = f_obs.as_mtz_dataset(column_root_label="F")
    ds.add_miller_array(rfree, column_root_label="FreeR_flag")
    ds.mtz_object().write(str(out))


@pytest.fixture(scope="module")
def _engine(tmp_path_factory):
    pytest.importorskip("mmtbx.f_model")
    from pxviewer.cctbx_io import read_model
    from pxviewer.loader import sample_structure_path
    from pxviewer.reflections import LiveDifferenceMap

    path = sample_structure_path()  # 1UBQ
    model = read_model(str(path))
    mtz = tmp_path_factory.mktemp("refl") / "data.mtz"
    _synthetic_mtz(model, 2.0, mtz)
    return LiveDifferenceMap(read_model(str(path)), mtz), read_model(str(path))


def test_moving_one_atom_lights_up_the_difference_map_there(_engine):
    """Displace a single atom by 1 A and the mFo-DFc map must show a strong peak where it
    left/entered — the map really is responding to the model, not returning noise. The map
    is sigma-scaled (unit std), so 'sigma' and the raw value coincide."""
    engine, model = _engine
    xrs = model.get_xray_structure().deep_copy_scatterers()

    # Baseline: the model fits its own data, so the difference map is quiet (a few sigma from
    # the bulk-solvent model, not flat). Moving an atom must dwarf that.
    baseline_peak = np.abs(engine.recompute(xray_structure=xrs).map_data().as_numpy_array()).max()

    sites = xrs.sites_cart()
    i = model.get_number_of_atoms() // 2
    x, y, z = sites[i]
    sites[i] = (x + 1.0, y, z)
    xrs.set_sites_cart(sites)

    grid = engine.recompute(xray_structure=xrs).map_data().as_numpy_array()
    # A hole where the atom was and density where the data still wants it: a strong, localised
    # signal, well above the quiet baseline and far above the map's unit noise.
    assert np.abs(grid).max() > 12.0
    assert np.abs(grid).max() > 2.0 * baseline_peak


def test_recompute_freezes_scales_but_r_free_tracks_the_model(_engine):
    """The honesty guarantee: recompute updates only f_calc and never rescales (so the map
    answers to the model, not to a re-fit of the experiment), yet R-free still *rises* when
    the model is made worse — real feedback, not a frozen or self-flattering number."""
    engine, model = _engine
    r_free_good = engine.r_free

    # Spy: recompute must not call the expensive scaling step.
    original = engine._fmodel.update_all_scales
    rescales = []
    engine._fmodel.update_all_scales = lambda *a, **k: (rescales.append(1), original(*a, **k))[1]
    try:
        xrs = model.get_xray_structure().deep_copy_scatterers()
        xrs.shake_sites_in_place(mean_distance=0.4)
        engine.recompute(xray_structure=xrs)
    finally:
        engine._fmodel.update_all_scales = original

    assert rescales == []                 # frozen scales: no rescale hidden in recompute
    assert engine.r_free > r_free_good     # a worse model reads as a worse R-free (honest)


def test_recompute_local_is_a_small_box_that_captures_the_signal(_engine):
    """recompute_local returns a tiny window around the tug point — far smaller than the
    whole-cell map — while still holding the difference peak from a moved atom there."""
    engine, model = _engine
    xrs = model.get_xray_structure().deep_copy_scatterers()
    sites = xrs.sites_cart()
    i = model.get_number_of_atoms() // 2
    x, y, z = sites[i]
    center = (x + 1.0, y, z)
    sites[i] = center
    xrs.set_sites_cart(sites)

    full = engine.recompute(xray_structure=xrs)
    box = engine.recompute_local(center, radius=5.0, xray_structure=xrs)

    full_pts = full.map_data().size()
    box_pts = box.map_data().size()
    assert box_pts < full_pts / 5             # a genuine crop, not the whole map
    assert max(box.map_data().all()) < 40     # a small window (~20 grid points a side)
    # the moved-atom difference peak is inside the window
    assert np.abs(box.map_data().as_numpy_array()).max() > 12.0


def test_recompute_accepts_numpy_sites(_engine):
    """The live loop holds coordinates as a numpy (N,3) array; recompute must take them."""
    engine, model = _engine
    sites = np.array(model.get_sites_cart(), dtype="float64")
    mm = engine.recompute(sites_cart=sites)
    assert mm.map_data().as_numpy_array().shape == mm.map_data().all()


def _moved_box(engine, model, radius=5.0):
    """A difference-map box around a single atom displaced 1 A; returns (box, center)."""
    xrs = model.get_xray_structure().deep_copy_scatterers()
    sites = xrs.sites_cart()
    i = model.get_number_of_atoms() // 2
    x, y, z = sites[i]
    center = (x + 1.0, y, z)
    sites[i] = center
    xrs.set_sites_cart(sites)
    return engine.recompute_local(center, radius=radius, xray_structure=xrs), center


def test_encode_map_box_carries_a_self_contained_affine(_engine):
    """The wire format is decodable with no crystallography: the grid is the small window,
    and the moved-atom peak decodes (via origin + i*stepX + j*stepY + k*stepZ) to the
    Cartesian place the atom went — so the browser can place the box from the payload alone."""
    import struct

    from pxviewer.volume_io import encode_map_box

    engine, model = _engine
    box, center = _moved_box(engine, model)
    body = encode_map_box(box, level=3.0)

    flags, level = struct.unpack_from("<If", body, 0)
    nx, ny, nz = struct.unpack_from("<iii", body, 8)
    origin = np.array(struct.unpack_from("<fff", body, 20))
    steps = [np.array(struct.unpack_from("<fff", body, 32 + 12 * j)) for j in range(3)]
    data = np.frombuffer(body, dtype="<f4", offset=68).reshape(nx, ny, nz)

    assert flags == 1 and level == pytest.approx(3.0)     # difference map, +/-3 sigma
    assert max(nx, ny, nz) < 40                            # a small window, not the whole cell
    assert data.size == nx * ny * nz
    peak = np.unravel_index(np.argmax(np.abs(data)), data.shape)
    cart = origin + peak[0] * steps[0] + peak[1] * steps[1] + peak[2] * steps[2]
    assert np.linalg.norm(cart - np.array(center)) < 2.5   # the peak lands at the tug


def test_live_session_streams_and_replays_the_map_box(_engine):
    """show_map_box broadcasts the density window as a tagged binary frame, and a late
    client is caught up with the current box on connect (like the last coordinate frame)."""
    import asyncio
    import struct

    websockets = pytest.importorskip("websockets")
    from pxviewer import LiveSession

    engine, model = _engine
    box, _ = _moved_box(engine, model)
    _TAG_MAP = 4

    session = LiveSession.from_sites([[float(i), 0.0, 0.0] for i in range(4)])
    session.start(port=0)

    async def _recv_map(ws):
        for _ in range(20):
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            if isinstance(msg, (bytes, bytearray)) and struct.unpack_from("<I", msg, 0)[0] == _TAG_MAP:
                return msg
        return None

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            session.show_map_box(box, level=3.0)
            msg = await _recv_map(ws)
            assert msg is not None and len(msg) > 4 + 68  # tag + header + at least some grid
        # A client that joins after the box is live still gets it (replayed on connect).
        async with websockets.connect(url) as late:
            assert await _recv_map(late) is not None

    try:
        asyncio.run(scenario())
    finally:
        session.stop()
