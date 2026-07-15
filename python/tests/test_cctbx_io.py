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
from pxviewer.data import encode_bcif_arrays  # noqa: E402
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
    import ciftools.serialization as cif_io

    arrays = model_to_arrays(read_model(LYSOZYME))
    block = cif_io.loads(encode_bcif_arrays(arrays, polymer=True), lazy=False)[0]
    site = block["atom_site"]
    assert site.n_rows == len(arrays)
    assert site["label_comp_id"].get_string(0) == "LYS"


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
    assert all(r == "LYS" for r in sel.resnames)


# -- ModelData: cctbx selection + drift --------------------------------------

def test_model_backed_session_uses_cctbx_selection():
    session = LiveSession.from_model_file(LYSOZYME)
    sel = session.select_by(selection="chain A and resseq 5:14 and name CA")
    assert len(sel) == 10
    assert all(n == "CA" for n in sel.names)


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


# -- per-atom attributes from mmCIF columns ----------------------------------

def _cif_with_columns(rows, extra_cols):
    """A minimal 3-atom mmCIF (chain A, resseq per row) with extra numeric columns.

    ``rows`` is a list of (resseq, {colname: value}); ``extra_cols`` names the extra
    columns (order matters for the header).
    """
    header = [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
        "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv",
        *extra_cols, "auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num",
    ]
    out = ["data_t", "loop_"] + ["_atom_site." + c for c in header]
    for i, (rs, extra) in enumerate(rows):
        vals = ["ATOM", str(i + 1), "C", "CA", ".", "ALA", "A", "1", str(rs),
                "%.1f" % (3.8 * (rs - 1)), "0", "0", "1", "0",
                *[str(extra[c]) for c in extra_cols], str(rs), "A", "1"]
        out.append(" ".join(vals))
    return "\n".join(out) + "\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_custom_atom_site_column_auto_exposed(tmp_path):
    cif = _cif_with_columns(
        [(1, {"plddt": 88.5}), (2, {"plddt": 72.1}), (3, {"plddt": 95.0})], ["plddt"]
    )
    session = LiveSession.from_model_file(_write(tmp_path, "m.cif", cif))
    assert "plddt" in session.attributes()
    assert list(session._attributes["plddt"]) == [88.5, 72.1, 95.0]
    # and it's usable for colouring
    session.color_by("plddt")


def test_pdb_load_has_no_custom_attributes():
    # A PDB has no room for arbitrary columns, so only the built-ins are present.
    session = LiveSession.from_model_file(LYSOZYME)
    assert session.attributes() == ["bfactor", "occupancy"]


def test_non_numeric_column_is_ignored(tmp_path):
    cif = _cif_with_columns(
        [(1, {"note": "aaa"}), (2, {"note": "bbb"}), (3, {"note": "ccc"})], ["note"]
    )
    session = LiveSession.from_model_file(_write(tmp_path, "m.cif", cif))
    assert "note" not in session.attributes()


def test_write_cif_roundtrips_attributes(tmp_path):
    cif = _cif_with_columns(
        [(1, {"plddt": 10.0}), (2, {"plddt": 20.0}), (3, {"plddt": 30.0})], ["plddt"]
    )
    session = LiveSession.from_model_file(_write(tmp_path, "m.cif", cif))
    session.set_attribute("score", [0.1, 0.2, 0.3])

    out = tmp_path / "out.cif"
    session.write_cif(out, attributes=["plddt", "score"])

    back = LiveSession.from_model_file(out)
    assert set(back.attributes()) >= {"plddt", "score"}
    assert [round(float(v), 3) for v in back._attributes["score"]] == [0.1, 0.2, 0.3]
    assert list(back._attributes["plddt"]) == [10.0, 20.0, 30.0]


def test_load_attributes_aligns_by_identity(tmp_path):
    model_cif = _cif_with_columns(
        [(1, {"plddt": 1.0}), (2, {"plddt": 2.0}), (3, {"plddt": 3.0})], ["plddt"]
    )
    session = LiveSession.from_model_file(_write(tmp_path, "m.cif", model_cif))

    # External file: same atoms, DIFFERENT order, a new column.
    ext = _cif_with_columns(
        [(3, {"energy": -3.0}), (1, {"energy": -1.0}), (2, {"energy": -2.0})], ["energy"]
    )
    loaded = session.load_attributes(_write(tmp_path, "e.cif", ext))
    assert loaded == ["energy"]
    # aligned back to the model's atom order (resseq 1, 2, 3)
    assert list(session._attributes["energy"]) == [-1.0, -2.0, -3.0]


def test_load_attributes_missing_atom_is_nan(tmp_path):
    model_cif = _cif_with_columns(
        [(1, {"x": 0}), (2, {"x": 0}), (3, {"x": 0})], ["x"]
    )
    session = LiveSession.from_model_file(_write(tmp_path, "m.cif", model_cif))
    # External file covers only residues 1 and 3.
    ext = _cif_with_columns([(1, {"q": 5.0}), (3, {"q": 7.0})], ["q"])
    session.load_attributes(_write(tmp_path, "e.cif", ext))
    q = session._attributes["q"]
    assert q[0] == 5.0 and np.isnan(q[1]) and q[2] == 7.0


def test_attribute_ops_need_a_model():
    session = LiveSession.from_sites([[0, 0, 0], [1, 0, 0]])  # synthetic, has a model
    assert session._data.model is not None  # from_sites is model-backed
