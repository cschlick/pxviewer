"""Tests for the MolProbity validation registry and the six validators.

Each validator gets at least a smoke assertion on 1ubq. Three of them build
restraints and need a monomer library (rotamers, cablam, rama_z); those are
guarded by :func:`pxviewer.geometry.monomer_library_available`. The other three
(ramachandran, cbetadev, omegalyze) run off the hierarchy alone.
"""

from pathlib import Path

import pytest

from pxviewer import validation

UBIQUITIN = Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb"

# Every validator that must be registered, in the stable (sorted-by-key) order.
EXPECTED_KEYS = ["cablam", "cbetadev", "omegalyze", "rama_z", "ramachandran", "rotamers"]

# Validators that need a monomer library (restraints) to run.
NEEDS_MONOMER_LIB = {"rotamers", "cablam", "rama_z"}

RAMA_COLUMNS = ["chain", "resid", "res", "phi", "psi", "type", "score"]


def _ubiquitin_model():
    from iotbx.data_manager import DataManager

    dm = DataManager()
    dm.process_model_file(str(UBIQUITIN))
    return dm.get_model()


def _require(key: str) -> None:
    """Skip when a validator can't run here (no data-manager / no monomer lib)."""
    pytest.importorskip("iotbx.data_manager")
    if key in NEEDS_MONOMER_LIB:
        from pxviewer.geometry import monomer_library_available

        if not monomer_library_available():
            pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")


# --- registry -------------------------------------------------------------

def test_validators_lists_all_six_in_stable_order():
    assert [spec.key for spec in validation.validators()] == EXPECTED_KEYS


def test_run_all_includes_ramachandran():
    assert "ramachandran" in {spec.key for spec in validation.validators()}


def test_channel_for_is_distinct_per_validator():
    channels = {key: validation.channel_for(key) for key in EXPECTED_KEYS}
    # One distinct channel each, all clear of the probe2 channels (0 and 1).
    assert len(set(channels.values())) == len(EXPECTED_KEYS)
    assert min(channels.values()) >= validation.CHANNEL_BASE
    # Channels follow the stable validator order from CHANNEL_BASE.
    assert channels == {
        key: validation.CHANNEL_BASE + i for i, key in enumerate(EXPECTED_KEYS)
    }


# --- per-validator smoke tests -------------------------------------------

def _smoke(key: str) -> validation.ValidationResult:
    """Run one validator on 1ubq and assert the shared result invariants."""
    _require(key)
    spec = {s.key: s for s in validation.validators()}[key]
    result = spec.run(_ubiquitin_model())
    assert isinstance(result, validation.ValidationResult)
    assert result.key == key
    assert result.title == spec.title
    assert result.columns  # a non-empty header
    assert all(len(row) == len(result.columns) for row in result.rows)
    assert all(len(m) == 3 for m in result.markers)
    assert isinstance(result.summary, str) and result.summary
    return result


def test_ramachandran_on_ubiquitin():
    pytest.importorskip("mmtbx.validation.ramalyze")
    result = _smoke("ramachandran")
    assert result.columns == RAMA_COLUMNS
    assert len(result.rows) == 74


def test_rotamers_on_ubiquitin():
    pytest.importorskip("mmtbx.validation.rotalyze")
    result = _smoke("rotamers")
    assert "rotamer" in result.columns
    assert result.rows  # 1ubq has side chains to score


def test_cbetadev_on_ubiquitin():
    pytest.importorskip("mmtbx.validation.cbetadev")
    result = _smoke("cbetadev")
    assert "deviation" in result.columns
    assert result.rows


def test_omegalyze_on_ubiquitin():
    pytest.importorskip("mmtbx.validation.omegalyze")
    result = _smoke("omegalyze")
    assert "omega" in result.columns
    assert result.rows


def test_cablam_on_ubiquitin():
    pytest.importorskip("mmtbx.validation.cablam")
    result = _smoke("cablam")
    assert "cablam" in result.columns
    assert result.rows


def test_rama_z_on_ubiquitin():
    pytest.importorskip("mmtbx.validation.rama_z")
    result = _smoke("rama_z")
    # Whole-model metric: the four fixed regions, and no per-residue markers.
    assert result.columns == ["region", "z_score", "std_err"]
    assert [row[0] for row in result.rows] == ["Helix", "Sheet", "Loop", "Whole"]
    assert result.markers == []
