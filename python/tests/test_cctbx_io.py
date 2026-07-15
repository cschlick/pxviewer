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

from pxviewer.cctbx_io import (  # noqa: E402
    load_model,
    model_is_polymer,
    model_secondary_structure,
    model_to_arrays,
    read_model,
)
from pxviewer.data import encode_bcif_arrays, read_atoms  # noqa: E402
from pxviewer.live import LiveSession  # noqa: E402

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

    # A polymer session built from it renders cartoon by secondary structure.
    session = LiveSession.from_arrays(
        loaded.arrays, polymer=loaded.polymer, secondary_structure=loaded.secondary_structure
    )
    assert session._n_atoms == 1079
    # metadata accessors work on cctbx-sourced atoms
    sel = session.select_by(ids=[1, 2, 3])
    assert sel.indices == [0, 1, 2]
    assert all(a.resname == "LYS" for a in sel.atoms)
