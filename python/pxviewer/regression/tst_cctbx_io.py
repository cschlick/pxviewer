"""The cctbx model bridge: arrays, topology, selection, and per-atom attributes."""

from __future__ import absolute_import, division, print_function

import asyncio
import os
import struct
import sys

from libtbx.test_utils import raises

from pxviewer.regression.tst_utils import data_path, have, skip, tmp_dir

if not have("iotbx.data_manager", "numpy", "websockets"):
    skip("iotbx.data_manager / numpy / websockets not available")

import numpy as np                                   # noqa: E402
import websockets                                    # noqa: E402
from iotbx.data_manager import DataManager           # noqa: E402

from pxviewer.cctbx_io import (ModelData, first_model, load_model,   # noqa: E402
                               model_from_sites, model_is_polymer,
                               model_secondary_structure, model_to_arrays,
                               read_model)
from pxviewer.data import encode_bcif_arrays         # noqa: E402
from pxviewer.live import LiveSession                # noqa: E402

_TAG_TOPOLOGY = 0
UBIQUITIN = data_path("1ubq.pdb")

_ALTLOC_PDB = """\
ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  SER A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  CB  SER A   1       2.100   1.400   0.000  1.00  0.00           C
ATOM      4  OG ASER A   1       3.500   1.400   0.000  0.50  0.00           O
ATOM      5  OG BSER A   1       2.100   2.800   0.000  0.50  0.00           O
"""

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


def model_from_str(text):
    dm = DataManager()
    return dm.get_model(dm.process_model_str("t", text))


def cif_with_columns(rows, extra_cols):
    """A minimal mmCIF (chain A, one atom per row) with extra numeric columns.

    ``rows`` is a list of (resseq, {colname: value}); ``extra_cols`` names the extra
    columns, and order matters for the header.
    """
    header = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
              "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
              "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv"] \
        + list(extra_cols) + ["auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num"]
    out = ["data_t", "loop_"] + ["_atom_site." + c for c in header]
    for i, (rs, extra) in enumerate(rows):
        vals = ["ATOM", str(i + 1), "C", "CA", ".", "ALA", "A", "1", str(rs),
                "%.1f" % (3.8 * (rs - 1)), "0", "0", "1", "0"] \
            + [str(extra[c]) for c in extra_cols] + [str(rs), "A", "1"]
        out.append(" ".join(vals))
    return "\n".join(out) + "\n"


def write(work, name, text):
    path = os.path.join(work, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


# -- arrays and topology ------------------------------------------------------


def exercise_read_model_and_extract_arrays():
    arrays = model_to_arrays(read_model(UBIQUITIN))

    assert len(arrays) == 660
    # Columns are all aligned to the same length (AtomArrays enforces this).
    assert arrays.x.shape[0] == len(arrays) == len(arrays.resname)
    # The first atom of 1UBQ is the backbone N of MET 1, chain A.
    assert arrays.element[0] == "N"
    assert arrays.name[0] == "N"
    assert arrays.resname[0] == "MET"
    assert arrays.chain[0] == "A"
    assert int(arrays.resseq[0]) == 1
    assert sorted(set(arrays.element)) == ["C", "N", "O", "S"]


def exercise_polymer_and_secondary_structure():
    model = read_model(UBIQUITIN)
    assert model_is_polymer(model) is True

    ss = model_secondary_structure(model)
    assert ss, "1UBQ has HELIX/SHEET records"
    kinds = set(row[3] for row in ss)
    assert kinds <= set(["helix", "sheet"])
    assert any(k == "helix" for row in ss for k in [row[3]])
    # Rows are (chain, beg, end, kind) with integer residue bounds, beg <= end.
    for chain, beg, end, kind in ss:
        assert isinstance(beg, int) and isinstance(end, int)
        assert beg <= end
        assert chain == "A"


def exercise_arrays_encode_to_binarycif_roundtrip():
    from pxviewer import bcif

    arrays = model_to_arrays(read_model(UBIQUITIN))
    site = bcif.decode(
        encode_bcif_arrays(arrays, polymer=True))["PXVIEWER"]["_atom_site"]
    assert len(site["label_comp_id"]) == len(arrays)
    assert site["label_comp_id"][0] == "MET"


def exercise_a_session_streams_topology_first():
    session = LiveSession.from_model_file(UBIQUITIN)
    assert session._n_atoms == 660
    session.start(port=0)
    try:
        async def scenario():
            url = "ws://%s:%d" % (session.host, session.port)
            async with websockets.connect(url) as ws:
                topo = await asyncio.wait_for(ws.recv(), timeout=5)
                assert isinstance(topo, (bytes, bytearray))
                assert struct.unpack("<I", topo[:4])[0] == _TAG_TOPOLOGY
                assert len(topo) > 4          # BinaryCIF payload follows the tag

        asyncio.run(scenario())
    finally:
        session.stop()


def exercise_load_model_reduces_to_a_streamable_bundle():
    loaded = load_model(UBIQUITIN)
    assert len(loaded.arrays) == 660
    assert loaded.polymer is True
    assert loaded.secondary_structure
    assert loaded.model is not None           # the native model is retained

    session = LiveSession.from_model_file(UBIQUITIN)
    assert session._n_atoms == 660
    assert session.model is not None
    sel = session.select_by(ids=[1, 2, 3])    # metadata accessors on cctbx atoms
    assert sel.indices == [0, 1, 2]
    assert all(r == "MET" for r in sel.resnames)


# -- ModelData: cctbx selection and drift -------------------------------------


def exercise_a_model_backed_session_uses_cctbx_selection():
    session = LiveSession.from_model_file(UBIQUITIN)
    sel = session.select_by(selection="chain A and resseq 5:14 and name CA")
    assert len(sel) == 10
    assert all(n == "CA" for n in sel.names)


def exercise_diff_detects_model_drift():
    loaded = load_model(UBIQUITIN)
    data = ModelData(loaded.arrays, model=loaded.model)
    assert data.diff() is None                # in sync

    sites = loaded.model.get_sites_cart()
    sites[0] = (sites[0][0] + 5.0, sites[0][1], sites[0][2])
    loaded.model.set_sites_cart(sites)
    msg = data.diff()
    assert msg is not None and "drift" in msg


def exercise_a_selection_string_requires_a_model():
    data = ModelData(model_to_arrays(read_model(UBIQUITIN)))   # no model attached
    with raises(ValueError) as e:
        data.selection_indices("chain A")
    assert "model-backed" in str(e.value)


# -- altlocs and multi-MODEL --------------------------------------------------


def exercise_altlocs_are_kept_as_distinct_atoms():
    arrays = model_to_arrays(model_from_str(_ALTLOC_PDB))
    assert len(arrays) == 5                   # nothing flattened
    og = [i for i, nm in enumerate(arrays.name) if nm == "OG"]
    assert len(og) == 2
    assert sorted(arrays.altloc[i] for i in og) == ["A", "B"]   # distinct, own i_seq


def exercise_altloc_topology_writes_label_alt_id():
    session = LiveSession.from_cctbx_model(model_from_str(_ALTLOC_PDB))
    assert session._n_atoms == 5
    assert len(session.select_by(selection="name OG")) == 2     # both conformers select
    assert len(session.select_by(selection="altloc A")) == 1


def exercise_multi_model_is_reduced_to_the_first():
    model = model_from_str(_NMR_PDB)
    assert model.get_number_of_atoms() == 4   # both models
    assert first_model(model).get_number_of_atoms() == 2        # model 1 only

    session = LiveSession.from_cctbx_model(model)
    assert session._n_atoms == 2              # the session took model 1


def exercise_model_from_sites_roundtrips_coords_and_labels():
    sites = np.array([[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]], dtype=float)
    model = model_from_sites(sites, chains=["A", "A", "B"], resseqs=[1, 2, 3])
    arrays = model_to_arrays(model)
    assert len(arrays) == 3
    assert np.allclose(arrays.xyz, sites, atol=1e-3)
    assert arrays.chain == ["A", "A", "B"]
    # ... and the labels drive cctbx selection.
    session = LiveSession.from_cctbx_model(model)
    assert session.select_by(selection="chain A").indices == [0, 1]


# -- per-atom attributes from mmCIF columns -----------------------------------


def exercise_a_custom_atom_site_column_is_exposed():
    with tmp_dir() as work:
        cif = cif_with_columns(
            [(1, {"plddt": 88.5}), (2, {"plddt": 72.1}), (3, {"plddt": 95.0})],
            ["plddt"])
        session = LiveSession.from_model_file(write(work, "m.cif", cif))
        assert "plddt" in session.attributes()
        assert list(session._attributes["plddt"]) == [88.5, 72.1, 95.0]
        session.color_by("plddt")             # and it is usable for colouring


def exercise_a_pdb_load_has_no_custom_attributes():
    """A PDB has no room for arbitrary columns, so only the built-ins are present."""
    session = LiveSession.from_model_file(UBIQUITIN)
    assert session.attributes() == ["bfactor", "occupancy"]


def exercise_a_non_numeric_column_is_ignored():
    with tmp_dir() as work:
        cif = cif_with_columns(
            [(1, {"note": "aaa"}), (2, {"note": "bbb"}), (3, {"note": "ccc"})],
            ["note"])
        session = LiveSession.from_model_file(write(work, "m.cif", cif))
        assert "note" not in session.attributes()


def exercise_write_cif_roundtrips_attributes():
    with tmp_dir() as work:
        cif = cif_with_columns(
            [(1, {"plddt": 10.0}), (2, {"plddt": 20.0}), (3, {"plddt": 30.0})],
            ["plddt"])
        session = LiveSession.from_model_file(write(work, "m.cif", cif))
        session.set_attribute("score", [0.1, 0.2, 0.3])

        out = os.path.join(work, "out.cif")
        session.write_cif(out, attributes=["plddt", "score"])

        back = LiveSession.from_model_file(out)
        assert set(back.attributes()) >= set(["plddt", "score"])
        assert [round(float(v), 3)
                for v in back._attributes["score"]] == [0.1, 0.2, 0.3]
        assert list(back._attributes["plddt"]) == [10.0, 20.0, 30.0]


def exercise_load_attributes_aligns_by_identity():
    with tmp_dir() as work:
        model_cif = cif_with_columns(
            [(1, {"plddt": 1.0}), (2, {"plddt": 2.0}), (3, {"plddt": 3.0})], ["plddt"])
        session = LiveSession.from_model_file(write(work, "m.cif", model_cif))

        # External file: the same atoms, a DIFFERENT order, and a new column.
        ext = cif_with_columns(
            [(3, {"energy": -3.0}), (1, {"energy": -1.0}), (2, {"energy": -2.0})],
            ["energy"])
        loaded = session.load_attributes(write(work, "e.cif", ext))
        assert loaded == ["energy"]
        # Aligned back to the model's atom order (resseq 1, 2, 3).
        assert list(session._attributes["energy"]) == [-1.0, -2.0, -3.0]


def exercise_a_missing_atom_loads_as_nan():
    with tmp_dir() as work:
        model_cif = cif_with_columns(
            [(1, {"x": 0}), (2, {"x": 0}), (3, {"x": 0})], ["x"])
        session = LiveSession.from_model_file(write(work, "m.cif", model_cif))
        # The external file covers only residues 1 and 3.
        ext = cif_with_columns([(1, {"q": 5.0}), (3, {"q": 7.0})], ["q"])
        session.load_attributes(write(work, "e.cif", ext))
        q = session._attributes["q"]
        assert q[0] == 5.0 and np.isnan(q[1]) and q[2] == 7.0


def exercise_from_sites_is_model_backed():
    session = LiveSession.from_sites([[0, 0, 0], [1, 0, 0]])
    assert session._data.model is not None


def exercise_load_attribute_text_positional():
    with tmp_dir() as work:
        session = LiveSession.from_sites(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        # A comment, a blank line, and a missing value.
        path = write(work, "scores.txt",
                     "# scores, one per atom\n0.5\n\n1.5\nnan\n2.5\n")
        assert session.load_attribute_text("score", path) == "score"
        v = session._attributes["score"]
        assert v[0] == 0.5 and v[1] == 1.5 and np.isnan(v[2]) and v[3] == 2.5


def exercise_load_attribute_text_wrong_length():
    with tmp_dir() as work:
        session = LiveSession.from_sites(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        path = write(work, "short.txt", "1\n2\n3\n")       # 3 values, 4 atoms
        with raises(ValueError) as e:
            session.load_attribute_text("bad", path)
        assert "4 atoms" in str(e.value)


def exercise_load_attribute_text_bad_line():
    with tmp_dir() as work:
        session = LiveSession.from_sites([[0, 0, 0], [1, 0, 0]])
        path = write(work, "bad.txt", "1\ntwo\n")
        with raises(ValueError) as e:
            session.load_attribute_text("bad", path)
        assert "one number per line" in str(e.value)


# -- DataManager ownership ----------------------------------------------------


def exercise_the_data_manager_is_the_callers_or_fresh_per_operation():
    """Nothing makes a private DataManager: callers pass one in, or get one scoped to the
    single operation -- the scope cctbx itself uses, since a program mutates the manager it
    is handed (ProgramTemplate binds it with set_program)."""
    from pxviewer.cctbx_io import data_manager

    mine = data_manager()
    assert data_manager(mine) is mine         # passed in -> used, not replaced
    assert data_manager() is not mine         # otherwise independent


def exercise_read_model_reads_through_a_given_data_manager():
    """Passing a DataManager in is what lets the caller own provenance: the file and the
    model it produced stay with the caller's manager."""
    from pxviewer.cctbx_io import data_manager

    dm = data_manager()
    model = read_model(UBIQUITIN, data_manager=dm)
    assert dm.get_model_names() == [str(UBIQUITIN)]
    assert dm.get_model(str(UBIQUITIN)) is model

    read_model(UBIQUITIN)                     # no manager given -> the caller's untouched
    assert len(dm.get_model_names()) == 1


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
