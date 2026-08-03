"""The MolProbity validation registry and its six validators.

Each validator gets at least a smoke assertion on 1ubq. Three of them build restraints and
need a monomer library (rotamers, cablam, rama_z); the other three (ramachandran, cbetadev,
omegalyze) run off the hierarchy alone.
"""

from __future__ import absolute_import, division, print_function

import sys

from pxviewer import validation
from pxviewer.regression.tst_utils import data_path, have, skip

if not have("iotbx.data_manager"):
    skip("iotbx.data_manager not available")

#: Every validator that must be registered, in the stable (sorted-by-key) order.
EXPECTED_KEYS = ["cablam", "cbetadev", "omegalyze", "rama_z", "ramachandran", "rotamers"]

#: Validators that need a monomer library (restraints) to run.
NEEDS_MONOMER_LIB = set(["rotamers", "cablam", "rama_z"])

RAMA_COLUMNS = ["chain", "resid", "res", "phi", "psi", "type", "score"]

_cache = []


def model():
    """1UBQ, read once per process."""
    if not _cache:
        from iotbx.data_manager import DataManager

        dm = DataManager()
        dm.process_model_file(data_path("1ubq.pdb"))
        _cache.append(dm.get_model())
    return _cache[0]


def runnable(key, module):
    """Whether this validator can run here, reporting why if not."""
    if not have(module):
        print("  skipping %s: %s not available" % (key, module))
        return False
    if key in NEEDS_MONOMER_LIB:
        from pxviewer.geometry import monomer_library_available

        if not monomer_library_available():
            print("  skipping %s: no monomer library" % key)
            return False
    return True


def smoke(key):
    """Run one validator on 1ubq and assert the shared result invariants."""
    spec = dict((s.key, s) for s in validation.validators())[key]
    result = spec.run(model())
    assert isinstance(result, validation.ValidationResult)
    assert result.key == key
    assert result.title == spec.title
    assert result.columns                                  # a non-empty header
    assert all(len(row) == len(result.columns) for row in result.rows)
    # Markup is a list of kinemage primitives, each a dict with kind + colour.
    assert isinstance(result.markup, list)
    assert all(m["kind"] in set(["vectors", "dots", "balls", "triangles"])
               and len(m["color"]) == 3 for m in result.markup)
    assert isinstance(result.summary, str) and result.summary
    return result


# --- registry ----------------------------------------------------------------


def exercise_validators_list_all_six_in_stable_order():
    assert [spec.key for spec in validation.validators()] == EXPECTED_KEYS


def exercise_channel_for_is_distinct_per_validator():
    channels = dict((key, validation.channel_for(key)) for key in EXPECTED_KEYS)
    # One distinct channel each, all clear of the probe2 channels (0 and 1).
    assert len(set(channels.values())) == len(EXPECTED_KEYS)
    assert min(channels.values()) >= validation.CHANNEL_BASE
    # Channels follow the stable validator order from CHANNEL_BASE.
    assert channels == dict(
        (key, validation.CHANNEL_BASE + i) for i, key in enumerate(EXPECTED_KEYS))


# --- per-validator smoke tests -----------------------------------------------


def exercise_ramachandran_on_ubiquitin():
    if not runnable("ramachandran", "mmtbx.validation.ramalyze"):
        return
    result = smoke("ramachandran")
    assert result.columns == RAMA_COLUMNS
    assert len(result.rows) == 74


def exercise_rotamers_on_ubiquitin():
    if not runnable("rotamers", "mmtbx.validation.rotalyze"):
        return
    result = smoke("rotamers")
    assert "rotamer" in result.columns
    assert result.rows                        # 1ubq has side chains to score


def exercise_cbetadev_on_ubiquitin():
    if not runnable("cbetadev", "mmtbx.validation.cbetadev"):
        return
    result = smoke("cbetadev")
    assert "deviation" in result.columns
    assert result.rows


def exercise_omegalyze_on_ubiquitin():
    if not runnable("omegalyze", "mmtbx.validation.omegalyze"):
        return
    result = smoke("omegalyze")
    assert "omega" in result.columns
    assert result.rows


def exercise_cablam_on_ubiquitin():
    if not runnable("cablam", "mmtbx.validation.cablam"):
        return
    result = smoke("cablam")
    assert "cablam" in result.columns
    assert result.rows


def exercise_rama_z_on_ubiquitin():
    if not runnable("rama_z", "mmtbx.validation.rama_z"):
        return
    result = smoke("rama_z")
    # Whole-model metric: the four fixed regions, and no per-residue markup.
    assert result.columns == ["region", "z_score", "std_err"]
    assert result.markup == []
    assert [row[0] for row in result.rows] == ["Helix", "Sheet", "Loop", "Whole"]


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
