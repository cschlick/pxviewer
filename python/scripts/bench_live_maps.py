"""Feasibility benchmark for real-time difference-map recompute (LiveDifferenceMap).

Measures the *warm* per-update cost — recompute f_calc from moved atoms + FFT the mFo-DFc
difference coefficients, reusing the scaled fmodel — against the *cold* full recompute
(update_all_scales + both maps) that make_maps does. The warm path is what a live loop
would run every frame.

    python scripts/bench_live_maps.py

Numbers on the dev box (2.0 A): ~660 atoms ~60 Hz, ~2700 atoms ~6 Hz. Compute scales with
reflection count (FFT) and atom count (f_calc), so it is interactive for small structures
and a few Hz for medium ones — fine for local fitting, not for whole-structure high-res.
"""

from __future__ import annotations

import time

from pxviewer.cctbx_io import read_model
from pxviewer.loader import sample_structure_path
from pxviewer.reflections import LiveDifferenceMap


def _synthetic_mtz(model, d_min, out):
    """Write amplitudes computed from the model, so the demo is self-contained."""
    xrs = model.get_xray_structure()
    f_obs = abs(xrs.structure_factors(d_min=d_min).f_calc())
    f_obs.set_observation_type_xray_amplitude()
    rfree = f_obs.generate_r_free_flags()
    ds = f_obs.as_mtz_dataset(column_root_label="F")
    ds.add_miller_array(rfree, column_root_label="FreeR_flag")
    ds.mtz_object().write(str(out))
    return f_obs.size()


def _time(fn, reps):
    fn()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps * 1e3


def main() -> None:
    import tempfile
    from pathlib import Path

    print(f"{'structure':>12} | {'atoms':>6} | {'refl':>7} | {'d_min':>5} | "
          f"{'warm recompute':>14} | {'rate':>7}")
    print("-" * 72)
    for fname in ("1ubq.pdb", "1tec.pdb"):
        path = sample_structure_path(fname)
        if path is None:
            continue
        model = read_model(str(path))
        for d_min in (3.0, 2.0):
            with tempfile.TemporaryDirectory() as td:
                mtz = Path(td) / "data.mtz"
                nrefl = _synthetic_mtz(model, d_min, mtz)
                engine = LiveDifferenceMap(read_model(str(path)), mtz)  # scales once
                xrs = engine._fmodel.xray_structure.deep_copy_scatterers()
                xrs.shake_sites_in_place(mean_distance=0.3)
                warm_ms = _time(lambda: engine.recompute(xray_structure=xrs), reps=6)
            print(f"{fname:>12} | {model.get_number_of_atoms():>6} | {nrefl:>7} | "
                  f"{d_min:>5.1f} | {warm_ms:>11.1f} ms | {1000 / warm_ms:>5.0f} Hz")


if __name__ == "__main__":
    main()
