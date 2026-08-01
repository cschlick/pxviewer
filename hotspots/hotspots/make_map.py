"""LEGACY additive-severity prototype; use make_concern_maps.py instead.

Compute the hotspot field for a model and write it as a CCP4 map, alongside a
copy of the model, so both can be loaded together in a viewer.

  libtbx.python make_map.py [model.cif|.pdb] [out_dir]
"""
import os
import sys

import numpy as np

from events import load_hierarchy, extract_all
from field import compute_field, write_ccp4


def main(model_path, out_dir, spacing=1.0, sigma=2.0, use_hydrogens=True):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(model_path).split(".")[0]

    h = load_hierarchy(model_path)
    out = extract_all(h, use_hydrogens=use_hydrogens)
    events, manifest = out["events"], out["manifest"]
    fld = compute_field(events, spacing=spacing, sigma=sigma)

    map_path = os.path.join(out_dir, f"{stem}_hotspot.ccp4")
    write_ccp4(fld, map_path)

    # write the model as PDB next to it so the viewer can overlay them
    pdb_path = os.path.join(out_dir, f"{stem}.pdb")
    open(pdb_path, "w").write(h.as_pdb_string())

    d = fld.data
    print(f"model:      {model_path}")
    print(f"manifest:   {manifest}")
    print(f"field:      shape={d.shape} spacing={spacing} sigma={sigma} "
          f"(severity units, 1.0 == outlier threshold)")
    print(f"            max={d.max():.2f}  mean={d.mean():.3f}  "
          f"p99={float(np.percentile(d, 99)):.2f}")
    print(f"\nwrote map:  {map_path}")
    print(f"wrote model:{pdb_path}")
    print(f"\nColor on an ABSOLUTE domain (severity units), never auto-scaled:")
    print(f"  0.0 - 0.5  clean      -> transparent / background")
    print(f"  1.0        outlier    -> color knee (yellow)")
    print(f"  2.0+       coincidence-> red")
    return map_path, pdb_path, fld


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "/root/data/pdb_mmcif/te/1tec.cif.gz"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/root/hotspots/output"
    main(model, out_dir)
