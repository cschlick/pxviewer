"""Tests for reduce2-based hydrogen placement."""

from pathlib import Path

import pytest

UBIQUITIN = Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb"


def test_add_hydrogens_on_ubiquitin():
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("mmtbx.programs.reduce2")
    from pxviewer.hydrogens import add_hydrogens, hydrogens_available

    if not hydrogens_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

    from iotbx.data_manager import DataManager

    dm = DataManager()
    dm.process_model_file(str(UBIQUITIN))
    model = dm.get_model()
    before = list(model.get_hierarchy().atoms().extract_element())
    assert not any(e.strip().upper() == "H" for e in before)  # starts H-less

    h_model = add_hydrogens(model)
    after = list(h_model.get_hierarchy().atoms().extract_element())
    n_h = sum(1 for e in after if e.strip().upper() == "H")
    assert n_h > 500  # ubiquitin gains hundreds of hydrogens
    assert len(after) > len(before)
