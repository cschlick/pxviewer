"""Consistency constraint: our severity>=1.0 level set must reproduce the
analyzers' own outlier flags exactly. If it diverges, the severity mapping is
miscalibrated. This is the evidence the calibration is inherited, not invented.

Run under phenix python: libtbx.python hotspots/check_consistency.py [model]
"""
import sys
from events import load_hierarchy, extract_ramachandran, extract_rotamer, \
    _residue_atom_map


def check(hierarchy):
    amap = _residue_atom_map(hierarchy)
    ok = True
    for name, extract in (("rama", extract_ramachandran), ("rota", extract_rotamer)):
        events = extract(hierarchy, amap)
        ours = {e.meta["id"] for e in events if e.is_outlier}
        theirs = {e.meta["id"] for e in events if e.meta["outlier"]}
        missing = theirs - ours    # analyzer flagged, we didn't  (false negatives)
        extra = ours - theirs      # we flagged, analyzer didn't  (false positives)
        status = "OK" if not missing and not extra else "MISMATCH"
        if missing or extra:
            ok = False
        print(f"[{status}] {name}: analyzer={len(theirs)} ours={len(ours)} "
              f"missing={len(missing)} extra={len(extra)}")
        for i in sorted(missing):
            print(f"    MISSING (analyzer outlier, we didn't flag): {i}")
        for i in sorted(extra):
            print(f"    EXTRA   (we flagged, analyzer didn't):      {i}")
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/data/pdb_mmcif/te/1tec.cif.gz"
    print(f"model: {path}")
    ok = check(load_hierarchy(path))
    print("\nCONSISTENCY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
