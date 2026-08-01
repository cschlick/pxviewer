"""Generate raw and concern local map-model CC fields plus per-atom results."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scitbx.array_family import flex
from iotbx.data_manager import DataManager
from iotbx.map_model_manager import map_model_manager

from local_cc import (cc_to_concern, correlation_radius, model_envelope_mask,
                      per_atom_local_cc, rolling_local_cc)
from color_scale import display_contract, empirical_percentile_field


def _flex_map(array):
    result = flex.double(np.ascontiguousarray(array).ravel())
    result.reshape(flex.grid(array.shape))
    return result


def _write_like(template, array, path):
    template.customized_copy(
        map_data=_flex_map(array), use_deep_copy_for_map_data=False,
        wrapping=False).write_map(path)


def _display_map(template, array, target_pixel_size=None, binary=False):
    """Make a finite viewer map, optionally resampled on a coarser grid."""
    mm = template.customized_copy(
        map_data=_flex_map(array), use_deep_copy_for_map_data=False,
        wrapping=False)
    if target_pixel_size is not None:
        mm = mm.resample_on_different_grid(
            target_grid_spacing=float(target_pixel_size))
    data = np.asarray(mm.map_data().as_numpy_array(), dtype=float)
    if binary:
        data = (data >= 0.5).astype(float)
    else:
        data = np.clip(data, 0.0, 1.0)
    return mm.customized_copy(
        map_data=_flex_map(data), use_deep_copy_for_map_data=False,
        wrapping=False)


def _atom_rows(model, values):
    rows = []
    for atom, value in zip(model.get_hierarchy().atoms(), values):
        ag = atom.parent()
        rg = ag.parent()
        chain = rg.parent()
        rows.append({
            "chain_id": chain.id.strip(), "resseq": rg.resseq.strip(),
            "icode": rg.icode.strip(), "resname": ag.resname.strip(),
            "atom_name": atom.name.strip(), "altloc": ag.altloc.strip(),
            "x": atom.xyz[0], "y": atom.xyz[1], "z": atom.xyz[2],
            "local_cc": float(value) if np.isfinite(value) else None,
        })
    return rows


def _residue_rows(model, values):
    rows = []
    offset = 0
    for hierarchy_model in model.get_hierarchy().models():
        for chain in hierarchy_model.chains():
            for rg in chain.residue_groups():
                n_atoms = rg.atoms().size()
                residue_values = values[offset:offset + n_atoms]
                offset += n_atoms
                finite = residue_values[np.isfinite(residue_values)]
                rows.append({
                    "chain_id": chain.id.strip(), "resseq": rg.resseq.strip(),
                    "icode": rg.icode.strip(),
                    "resnames": sorted(rg.unique_resnames()),
                    "n_atoms": n_atoms, "n_finite": int(finite.size),
                    "local_cc": float(finite.mean()) if finite.size else None,
                })
    assert offset == len(values)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate raw voxel local map-model CC, bounded concern, "
                    "optional percentile, mask, and atom/residue JSON.")
    parser.add_argument("model", help="input PDB/mmCIF model")
    parser.add_argument("experimental_map", help="input experimental MRC/CCP4 map")
    parser.add_argument("resolution", type=float,
                        help="reported map resolution in A")
    parser.add_argument("out_dir", help="directory for maps, tables, and manifest")
    parser.add_argument("--radius", type=float,
                        help="override max(2.5, resolution) in Angstrom")
    parser.add_argument("--envelope-radius", type=float,
                        help="evaluation band; defaults to correlation radius")
    parser.add_argument("--box-cushion", type=float,
                        help="defaults to max(5, 2*radius) Angstrom")
    parser.add_argument("--scattering-table", default="electron",
                        help="model-map scattering table (default: electron)")
    parser.add_argument(
        "--output-pixel-size", type=float, metavar="ANGSTROM",
        help="coarser viewer-map pixel size in A; raw local CC stays native")
    args = parser.parse_args()

    radius = args.radius if args.radius is not None else correlation_radius(args.resolution)
    if radius <= 0:
        raise ValueError("radius must be positive")
    env_radius = args.envelope_radius if args.envelope_radius is not None else radius
    cushion = args.box_cushion if args.box_cushion is not None else max(5.0, 2.0 * radius)
    if args.output_pixel_size is not None and args.output_pixel_size <= 0:
        raise ValueError("output pixel size must be positive")

    dm = DataManager()
    dm.process_model_file(args.model)
    dm.process_real_map_file(args.experimental_map)
    model = dm.get_model(args.model).remove_hydrogens()
    mmm = map_model_manager(
        map_manager=dm.get_real_map(args.experimental_map), model=model.deep_copy())
    mmm.box_all_maps_around_model_and_shift_origin(box_cushion=cushion)
    model = mmm.model().remove_hydrogens()
    mmm.set_model(model, overwrite=True)
    mmm.set_scattering_table(args.scattering_table)
    mmm.generate_map(model=model, d_min=args.resolution, map_id="model_map")

    exp_mm = mmm.map_manager()
    exp = exp_mm.map_data()
    calculated = mmm.get_map_manager_by_id("model_map").map_data()
    uc = exp_mm.crystal_symmetry().unit_cell()
    sites = model.get_sites_cart()
    envelope = model_envelope_mask(uc, exp.all(), sites, env_radius)
    cc_field, kernel = rolling_local_cc(
        exp, calculated, uc, radius, valid_mask=envelope)
    concern_field = cc_to_concern(cc_field)
    atom_cc = per_atom_local_cc(exp, calculated, uc, sites, radius)

    native_pixel_sizes = tuple(float(x) for x in exp_mm.pixel_sizes())
    if (args.output_pixel_size is not None and
            args.output_pixel_size < max(native_pixel_sizes) - 1e-6):
        raise ValueError(
            "output pixel size %.3f A is finer than native map pixels %s; "
            "omit it or choose a coarser value" %
            (args.output_pixel_size, native_pixel_sizes))
    concern_mm = _display_map(
        exp_mm, concern_field, args.output_pixel_size, binary=False)
    mask_mm = _display_map(
        exp_mm, envelope.astype(float), args.output_pixel_size, binary=True)
    display_concern = np.asarray(
        concern_mm.map_data().as_numpy_array(), dtype=float)
    display_mask = np.asarray(mask_mm.map_data().as_numpy_array(), dtype=bool)
    percentile_field, color_metadata = empirical_percentile_field(
        display_concern, support_mask=display_mask)
    percentile_mm = concern_mm.customized_copy(
        map_data=_flex_map(percentile_field), use_deep_copy_for_map_data=False,
        wrapping=False)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.basename(args.model).split(".")[0]
    raw_path = os.path.join(args.out_dir, stem + "_local_cc.ccp4")
    concern_path = os.path.join(args.out_dir, stem + "_local_cc_concern.ccp4")
    mask_path = os.path.join(args.out_dir, stem + "_local_cc_mask.ccp4")
    percentile_path = os.path.join(
        args.out_dir, stem + "_local_cc_percentile.ccp4")
    # Raw NaNs preserve 'not evaluated'. The separate mask supports viewers that
    # do not handle NaN voxels. Concern is zero outside the envelope.
    _write_like(exp_mm, cc_field, raw_path)
    concern_mm.write_map(concern_path)
    mask_mm.write_map(mask_path)
    percentile_mm.write_map(percentile_path)

    atom_path = os.path.join(args.out_dir, stem + "_local_cc_atoms.json")
    with open(atom_path, "w") as handle:
        json.dump(_atom_rows(model, atom_cc), handle, indent=2)
        handle.write("\n")
    residue_path = os.path.join(args.out_dir, stem + "_local_cc_residues.json")
    with open(residue_path, "w") as handle:
        json.dump(_residue_rows(model, atom_cc), handle, indent=2)
        handle.write("\n")
    finite = cc_field[np.isfinite(cc_field)]
    manifest = {
        **display_contract(),
        "model": os.path.abspath(args.model),
        "experimental_map": os.path.abspath(args.experimental_map),
        "resolution_angstrom": args.resolution,
        "correlation_radius_angstrom": radius,
        "envelope_radius_angstrom": env_radius,
        "box_cushion_angstrom": cushion,
        "scattering_table": args.scattering_table,
        "calculation_pixel_size_angstrom": list(native_pixel_sizes),
        "output_pixel_size_requested_angstrom": args.output_pixel_size,
        "output_pixel_size_actual_angstrom": [
            float(x) for x in concern_mm.pixel_sizes()],
        "calculation_grid": list(exp.all()),
        "output_grid": list(concern_mm.map_data().all()),
        "raw_local_cc_grid": "native calculation grid; not resampled",
        "kernel_grid_points": int(kernel.sum()),
        "evaluated_voxels": int(np.isfinite(cc_field).sum()),
        "local_cc_min": float(finite.min()) if finite.size else None,
        "local_cc_max": float(finite.max()) if finite.size else None,
        "local_cc_mean": float(finite.mean()) if finite.size else None,
        "concern_transform": "(1-local_cc)/2 over Pearson range [-1,1]",
        "color_scaling": color_metadata,
        "outputs": {"raw_local_cc": raw_path, "concern": concern_path,
                    "envelope_mask": mask_path, "per_atom": atom_path,
                    "per_residue": residue_path,
                    "color_percentile": percentile_path},
    }
    manifest_path = os.path.join(args.out_dir, stem + "_local_cc.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
