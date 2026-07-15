"""Tests for the cctbx model bridge.

Skipped when cctbx isn't installed, so the rest of the suite still runs on a bare
environment; run in the pxviewer conda env (with cctbx-base) to exercise them.
"""

import asyncio
import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("iotbx.data_manager")

import numpy as np  # noqa: E402
from iotbx.data_manager import DataManager  # noqa: E402

from pxviewer.cctbx_io import (  # noqa: E402
    ModelData,
    first_model,
    load_model,
    model_from_sites,
    model_is_polymer,
    model_secondary_structure,
    model_to_arrays,
    read_model,
)
from pxviewer.data import encode_bcif_arrays, read_atoms  # noqa: E402
from pxviewer.live import LiveSession  # noqa: E402


def _model_from_str(text: str):
    dm = DataManager()
    return dm.get_model(dm.process_model_str("t", text))

websockets = pytest.importorskip("websockets")

_TAG_TOPOLOGY = 0
LYSOZYME = Path(__file__).parent / "data" / "1aki.pdb"


def test_read_model_and_extract_arrays():
    model = read_model(LYSOZYME)
    arrays = model_to_arrays(model)

    assert len(arrays) == 1079
    # Columns are all aligned to the same length (AtomArrays enforces this).
    assert arrays.x.shape[0] == len(arrays) == len(arrays.resname)
    # First atom of 1AKI is the backbone N of LYS 1, chain A.
    assert arrays.element[0] == "N"
    assert arrays.name[0] == "N"
    assert arrays.resname[0] == "LYS"
    assert arrays.chain[0] == "A"
    assert int(arrays.resseq[0]) == 1
    assert sorted(set(arrays.element)) == ["C", "N", "O", "S"]


def test_polymer_and_secondary_structure():
    model = read_model(LYSOZYME)
    assert model_is_polymer(model) is True

    ss = model_secondary_structure(model)
    assert ss, "1AKI has HELIX/SHEET records"
    kinds = {row[3] for row in ss}
    assert kinds <= {"helix", "sheet"}
    assert any(k == "helix" for *_, k in ss)
    # Rows are (chain, beg, end, kind) with integer residue bounds, beg <= end.
    for chain, beg, end, kind in ss:
        assert isinstance(beg, int) and isinstance(end, int)
        assert beg <= end
        assert chain == "A"


def test_arrays_encode_to_binarycif_roundtrip():
    arrays = model_to_arrays(read_model(LYSOZYME))
    data = encode_bcif_arrays(arrays, polymer=True)
    atoms = read_atoms_from_bytes(data)
    assert len(atoms) == len(arrays)
    assert atoms[0].resname == "LYS"


def read_atoms_from_bytes(data: bytes):
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".bcif", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return read_atoms(path)
    finally:
        os.unlink(path)


def test_live_session_from_model_file_streams_topology():
    session = LiveSession.from_model_file(LYSOZYME)
    assert session._n_atoms == 1079
    session.start(port=0)
    try:

        async def scenario():
            url = f"ws://{session.host}:{session.port}"
            async with websockets.connect(url) as ws:
                topo = await asyncio.wait_for(ws.recv(), timeout=5)
                assert isinstance(topo, (bytes, bytearray))
                tag = struct.unpack("<I", topo[:4])[0]
                assert tag == _TAG_TOPOLOGY
                assert len(topo) > 4  # BinaryCIF payload follows the tag

        asyncio.run(scenario())
    finally:
        session.stop()


def test_load_model_reduces_to_streamable_bundle():
    loaded = load_model(LYSOZYME)
    assert len(loaded.arrays) == 1079
    assert loaded.polymer is True
    assert loaded.secondary_structure
    assert loaded.model is not None  # the native model is retained

    session = LiveSession.from_model_file(LYSOZYME)
    assert session._n_atoms == 1079
    assert session.model is not None
    # metadata accessors work on cctbx-sourced atoms
    sel = session.select_by(ids=[1, 2, 3])
    assert sel.indices == [0, 1, 2]
    assert all(a.resname == "LYS" for a in sel.atoms)


# -- ModelData: cctbx selection + drift --------------------------------------

def test_model_backed_session_uses_cctbx_selection():
    session = LiveSession.from_model_file(LYSOZYME)
    sel = session.select_by(selection="chain A and resseq 5:14 and name CA")
    assert len(sel) == 10
    assert all(a.name == "CA" for a in sel.atoms)


def test_diff_detects_model_drift():
    loaded = load_model(LYSOZYME)
    data = ModelData(loaded.arrays, model=loaded.model)
    assert data.diff() is None  # in sync

    sites = loaded.model.get_sites_cart()
    sites[0] = (sites[0][0] + 5.0, sites[0][1], sites[0][2])
    loaded.model.set_sites_cart(sites)
    msg = data.diff()
    assert msg is not None and "drift" in msg


def test_selection_string_requires_model():
    data = ModelData(model_to_arrays(read_model(LYSOZYME)))  # no model attached
    with pytest.raises(ValueError, match="model-backed"):
        data.selection_indices("chain A")


# -- altloc: both conformers survive with distinct i_seq ---------------------

_ALTLOC_PDB = """\
ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  SER A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  CB  SER A   1       2.100   1.400   0.000  1.00  0.00           C
ATOM      4  OG ASER A   1       3.500   1.400   0.000  0.50  0.00           O
ATOM      5  OG BSER A   1       2.100   2.800   0.000  0.50  0.00           O
"""


def test_altlocs_kept_as_distinct_atoms():
    arrays = model_to_arrays(_model_from_str(_ALTLOC_PDB))
    assert len(arrays) == 5  # nothing flattened
    og = [i for i, nm in enumerate(arrays.name) if nm == "OG"]
    assert len(og) == 2
    assert sorted(arrays.altloc[i] for i in og) == ["A", "B"]  # distinct labels, own i_seq


def test_altloc_topology_writes_label_alt_id():
    session = LiveSession.from_cctbx_model(_model_from_str(_ALTLOC_PDB))
    assert session._n_atoms == 5
    # both conformers select
    assert len(session.select_by(selection="name OG")) == 2
    assert len(session.select_by(selection="altloc A")) == 1


# -- multi-MODEL reduced to model 1 ------------------------------------------

_NMR_PDB = """\
MODEL        1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  CA  ALA A   1       0.500   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       4.300   0.000   0.000  1.00  0.00           C
ENDMDL
"""


def test_multi_model_reduced_to_first():
    model = _model_from_str(_NMR_PDB)
    assert model.get_number_of_atoms() == 4  # both models
    reduced = first_model(model)
    assert reduced.get_number_of_atoms() == 2  # model 1 only

    session = LiveSession.from_cctbx_model(model)
    assert session._n_atoms == 2  # session took model 1


# -- model_from_sites helper -------------------------------------------------

def test_model_from_sites_roundtrips_coords_and_labels():
    sites = np.array([[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]], dtype=float)
    model = model_from_sites(sites, chains=["A", "A", "B"], resseqs=[1, 2, 3])
    arrays = model_to_arrays(model)
    assert len(arrays) == 3
    assert np.allclose(arrays.xyz, sites, atol=1e-3)
    assert arrays.chain == ["A", "A", "B"]
    # and the labels drive cctbx selection
    session = LiveSession.from_cctbx_model(model)
    assert session.select_by(selection="chain A").indices == [0, 1]
