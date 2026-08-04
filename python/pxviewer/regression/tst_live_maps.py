"""LiveDifferenceMap: recomputing a difference map while the model moves.

The feasibility question for tugging against density is whether an mFo-DFc map can be
recomputed fast enough and honestly enough to steer by. "Honestly" is the harder half:
the scales stay frozen, so the map answers to the model rather than to a re-fit of the
experiment, and R-free stays a fixed reference that gets *worse* when the model does.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import os
import sys

from libtbx.test_utils import approx_equal

from pxviewer.regression.tst_utils import have, skip, tmp_dir

if not have("mmtbx.f_model", "numpy"):
    skip("mmtbx.f_model / numpy not available")

import numpy as np                                   # noqa: E402

D_MIN = 2.0

#: A 1 A displacement of one atom, in a map scaled to unit noise. The peak it leaves is
#: enormous -- a hole where the atom was plus density where the data still wants it.
DISPLACEMENT_A = 1.0
PEAK_SIGMA = 12.0


def write_synthetic_mtz(model, path, d_min=D_MIN):
    """Amplitudes calculated from the model itself, so it fits its own data perfectly."""
    f_obs = abs(model.get_xray_structure().structure_factors(d_min=d_min).f_calc())
    f_obs.set_observation_type_xray_amplitude()
    dataset = f_obs.as_mtz_dataset(column_root_label="F")
    dataset.add_miller_array(f_obs.generate_r_free_flags(),
                             column_root_label="FreeR_flag")
    dataset.mtz_object().write(path)


@contextlib.contextmanager
def engine():
    """A **fresh** engine and an untouched model, as ``(engine, model)``.

    Not cached, despite two reads of 1UBQ and an f_model setup: the whole subject here is
    state that recompute leaves behind -- R-free tracks the last model it was given -- so
    a shared engine would make one exercise's answer depend on which ran before it. It
    costs well under a second.
    """
    from pxviewer.cctbx_io import read_model
    from pxviewer.loader import sample_structure_path
    from pxviewer.reflections import LiveDifferenceMap

    path = str(sample_structure_path())              # 1UBQ
    with tmp_dir() as directory:
        mtz = os.path.join(directory, "data.mtz")
        write_synthetic_mtz(read_model(path), mtz)
        yield LiveDifferenceMap(read_model(path), mtz), read_model(path)


def displaced(model, distance=DISPLACEMENT_A):
    """A structure with one mid-model atom moved, and the place it moved to."""
    xrs = model.get_xray_structure().deep_copy_scatterers()
    sites = xrs.sites_cart()
    i = model.get_number_of_atoms() // 2
    x, y, z = sites[i]
    center = (x + distance, y, z)
    sites[i] = center
    xrs.set_sites_cart(sites)
    return xrs, center


def scales(fmodel):
    """The *stored* scale factors -- the ones a refit would rewrite.

    ``scale_k1()`` is deliberately not among them: it is the least-squares scale between
    f_obs and f_model, recomputed on demand, so it moves whenever f_calc does and would
    report a rescale on every recompute.
    """
    return {
        "k_isotropic": np.array(fmodel.k_isotropic()),
        "k_anisotropic": np.array(fmodel.k_anisotropic()),
        "k_masks": [np.array(k) for k in fmodel.k_masks()],   # bulk solvent
    }


# -- the map responds to the model --------------------------------------------


def exercise_moving_one_atom_lights_up_the_difference_map_there():
    """The map really is responding to the model rather than returning noise.

    Compared against its own baseline as well as an absolute cut: a model that fits its
    own data still leaves a few sigma from the bulk-solvent model, so "quiet" is not
    "flat", and the displaced atom has to dwarf that.
    """
    with engine() as (live, model):
        rest = model.get_xray_structure().deep_copy_scatterers()
        baseline = np.abs(
            live.recompute(xray_structure=rest).map_data().as_numpy_array()).max()

        moved, _center = displaced(model)
        grid = live.recompute(xray_structure=moved).map_data().as_numpy_array()

        assert np.abs(grid).max() > PEAK_SIGMA
        assert np.abs(grid).max() > 2.0 * baseline


def exercise_recompute_accepts_numpy_sites():
    """The live loop holds coordinates as a numpy (N,3) array, so recompute takes them."""
    with engine() as (live, model):
        sites = np.array(model.get_sites_cart(), dtype="float64")
        mm = live.recompute(sites_cart=sites)
        assert mm.map_data().as_numpy_array().shape == mm.map_data().all()


# -- the honesty guarantee ----------------------------------------------------


def exercise_recompute_freezes_the_scales_but_r_free_still_tracks_the_model():
    """recompute updates f_calc and nothing else, yet R-free still rises when the model
    is made worse -- real feedback rather than a frozen or self-flattering number.

    Checked by comparing the scale factors themselves rather than by watching whether
    ``update_all_scales`` was called. The factors are what a refit would rewrite, they
    stay bit-identical through a recompute that moves every atom, and a single
    ``update_all_scales`` changes all three -- so this catches a rescale however it is
    reached, including one buried inside cctbx where no spy would see it.
    """
    with engine() as (live, model):
        r_free_good = live.r_free
        before = scales(live._fmodel)

        xrs = model.get_xray_structure().deep_copy_scatterers()
        xrs.shake_sites_in_place(mean_distance=0.4)
        live.recompute(xray_structure=xrs)

        after = scales(live._fmodel)
        assert (after["k_isotropic"] == before["k_isotropic"]).all()
        assert (after["k_anisotropic"] == before["k_anisotropic"]).all()
        assert len(after["k_masks"]) == len(before["k_masks"])
        for now, then in zip(after["k_masks"], before["k_masks"]):
            assert (now == then).all()

        assert live.r_free > r_free_good


# -- the local box ------------------------------------------------------------


def exercise_recompute_local_is_a_small_box_that_still_holds_the_signal():
    """A tug needs the density around one point, not the whole cell -- but cropping is
    only useful if the peak it was cropped for is inside."""
    with engine() as (live, model):
        moved, center = displaced(model)

        full = live.recompute(xray_structure=moved)
        box = live.recompute_local(center, radius=5.0, xray_structure=moved)

        assert box.map_data().size() < full.map_data().size() / 5   # a genuine crop
        assert max(box.map_data().all()) < 40          # ~20 grid points a side
        assert np.abs(box.map_data().as_numpy_array()).max() > PEAK_SIGMA


def exercise_the_encoded_box_carries_a_self_contained_affine():
    """The wire format is decodable with no crystallography in the browser.

    Decoding the peak through origin + i*stepX + j*stepY + k*stepZ has to land where the
    atom actually went. That product is the whole contract: get the affine wrong and the
    density draws in the right shape at the wrong place, which looks like a modelling
    error rather than a bug.
    """
    import struct

    from pxviewer.volume_io import encode_map_box

    with engine() as (live, model):
        moved, center = displaced(model)
        box = live.recompute_local(center, radius=5.0, xray_structure=moved)
        body = encode_map_box(box, level=3.0)

        flags, level = struct.unpack_from("<If", body, 0)
        nx, ny, nz = struct.unpack_from("<iii", body, 8)
        origin = np.array(struct.unpack_from("<fff", body, 20))
        steps = [np.array(struct.unpack_from("<fff", body, 32 + 12 * j)) for j in range(3)]
        data = np.frombuffer(body, dtype="<f4", offset=68).reshape(nx, ny, nz)

        assert flags == 1                              # a difference map
        assert approx_equal(level, 3.0)                # contoured at +/- 3 sigma
        assert max(nx, ny, nz) < 40                    # a window, not the whole cell
        assert data.size == nx * ny * nz

        peak = np.unravel_index(np.argmax(np.abs(data)), data.shape)
        cart = origin + peak[0] * steps[0] + peak[1] * steps[1] + peak[2] * steps[2]
        assert np.linalg.norm(cart - np.array(center)) < 2.5


def exercise_a_live_session_streams_and_replays_the_map_box():
    """show_map_box broadcasts the window as a tagged binary frame, and a client that
    joins later is caught up with the current box -- the same replay the last coordinate
    frame gets, and for the same reason: a late viewer must not be left blank."""
    if not have("websockets"):
        print("    (skipped: websockets not available)")
        return
    import asyncio
    import struct

    import websockets

    from pxviewer import LiveSession

    TAG_MAP = 4

    async def receive_map(ws):
        for _ in range(20):
            message = await asyncio.wait_for(ws.recv(), timeout=5)
            if (isinstance(message, (bytes, bytearray))
                    and struct.unpack_from("<I", message, 0)[0] == TAG_MAP):
                return message
        return None

    with engine() as (live, model):
        moved, center = displaced(model)
        box = live.recompute_local(center, radius=5.0, xray_structure=moved)

        session = LiveSession.from_sites([[float(i), 0.0, 0.0] for i in range(4)])
        session.start(port=0)

        async def scenario():
            url = "ws://%s:%d" % (session.host, session.port)
            async with websockets.connect(url) as ws:
                await ws.recv()                        # topology arrives first
                session.show_map_box(box, level=3.0)
                message = await receive_map(ws)
                assert message is not None
                assert len(message) > 4 + 68           # tag + header + some grid

            async with websockets.connect(url) as late:
                assert await receive_map(late) is not None

        try:
            asyncio.run(scenario())
        finally:
            session.stop()


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
