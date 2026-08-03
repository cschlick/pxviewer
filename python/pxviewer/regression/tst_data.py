"""The columnar BinaryCIF encoder (encode_bcif_arrays).

Decoded with :func:`pxviewer.bcif.decode`, the same module that wrote them -- a round trip.
(Cross-compatibility with an independent decoder is Mol* itself, at the other end of the
wire.) Category names come back with the leading underscore.
"""

from __future__ import absolute_import, division, print_function

import sys

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import have, skip

if not have("numpy"):
    skip("numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer import AtomArrays, bcif, encode_bcif_arrays      # noqa: E402


def protein_arrays(n=6):
    return AtomArrays(
        element=["C"] * n, name=["CA"] * n, resname=["ALA"] * n, chain=["A"] * n,
        resseq=list(range(1, n + 1)),
        x=np.arange(n, dtype=float), y=np.zeros(n), z=np.zeros(n))


def block(raw):
    return bcif.decode(raw)["PXVIEWER"]


def two_chains_with_waters():
    """The ordinary PDB shape: chains E and I, then each one's waters -- so the author
    chain id repeats in blocks that are not contiguous (1TEC is exactly this)."""
    chain = ["E"] * 3 + ["I"] * 2 + ["E"] * 2 + ["I"] * 1
    n = len(chain)
    return AtomArrays(
        element=["C"] * 5 + ["O"] * 3, name=["CA"] * 5 + ["O"] * 3,
        resname=["ALA"] * 5 + ["HOH"] * 3, chain=chain,
        resseq=[1, 2, 3, 1, 2, 101, 102, 101],
        x=np.arange(n, dtype=float), y=np.zeros(n), z=np.zeros(n))


def exercise_encode_has_atom_site_cell_and_symmetry():
    b = block(encode_bcif_arrays(protein_arrays()))
    assert "_atom_site" in b and "_cell" in b and "_symmetry" in b
    assert b["_symmetry"]["space_group_name_H-M"][0] == "P 1"
    assert len(b["_atom_site"]["id"]) == 6


def exercise_atom_site_columns_map_from_arrays():
    arrays = AtomArrays(
        element=["N", "C"], name=["N", "CA"], resname=["ALA", "ALA"], chain=["A", "A"],
        resseq=[1, 1], x=[1.0, 4.0], y=[2.0, 5.0], z=[3.0, 6.0])
    site = block(encode_bcif_arrays(arrays))["_atom_site"]
    assert site["type_symbol"][0] == "N"
    assert site["label_atom_id"][1] == "CA"
    assert approx_equal(site["Cartn_x"][0], 1.0)
    assert approx_equal(site["Cartn_y"][1], 5.0)


def exercise_optional_b_and_occupancy_columns():
    arrays = protein_arrays(2)
    arrays.b = np.array([11.0, 22.0], dtype=np.float32)
    arrays.occ = np.array([1.0, 0.5], dtype=np.float32)
    site = block(encode_bcif_arrays(arrays))["_atom_site"]
    assert "B_iso_or_equiv" in site and "occupancy" in site
    assert approx_equal(site["B_iso_or_equiv"][1], 22.0)


def exercise_polymer_flag_emits_entity():
    b = block(encode_bcif_arrays(protein_arrays(), polymer=True))
    assert "_entity" in b and "_entity_poly" in b
    assert b["_entity_poly"]["type"][0] == "polypeptide(L)"
    assert "label_entity_id" in b["_atom_site"]


def exercise_default_is_not_polymer():
    b = block(encode_bcif_arrays(protein_arrays()))
    assert "_entity" not in b
    assert "label_entity_id" not in b["_atom_site"]


def exercise_secondary_structure_categories():
    b = block(encode_bcif_arrays(
        protein_arrays(),
        secondary_structure=[("A", 1, 3, "helix"), ("A", 4, 6, "sheet")]))
    assert "_entity_poly" in b       # SS implies polymer
    conf = b["_struct_conf"]
    assert len(conf["id"]) == 1
    assert conf["conf_type_id"][0] == "HELX_P"
    assert int(conf["beg_label_seq_id"][0]) == 1
    assert int(conf["end_label_seq_id"][0]) == 3
    sheet = b["_struct_sheet_range"]
    assert len(sheet["id"]) == 1
    assert int(sheet["beg_label_seq_id"][0]) == 4


def exercise_secondary_structure_bad_kind():
    with raises(ValueError):
        encode_bcif_arrays(protein_arrays(),
                           secondary_structure=[("A", 1, 3, "coil")])


def exercise_a_repeated_chain_id_gets_one_label_asym_per_block():
    """Each contiguous run of the author chain must be its own ``label_asym_id``.

    An author chain id is not an mmCIF chain: a PDB reuses one for the protein and then
    again for that chain's waters. Writing it straight into ``label_asym_id`` tells the
    reader those blocks are one chain, and Mol* gathers each label's atoms together --
    reordering them against the file and silently breaking the contract that streamed
    coordinates are positionally aligned to the topology. On 1TEC that moved atoms by up
    to 58 A once a frame was pushed.
    """
    site = block(encode_bcif_arrays(two_chains_with_waters()))["_atom_site"]

    labels = list(site["label_asym_id"])
    assert labels == ["A", "A", "A", "B", "B", "C", "C", "D"]   # one per block, in order
    assert len(set(labels)) == 4                                 # and none of them reused
    # The author id is untouched, so the user still sees the chain they expect.
    assert list(site["auth_asym_id"]) == ["E", "E", "E", "I", "I", "E", "E", "I"]
    # Above all: the atoms stay in the order they were given.
    assert approx_equal([float(v) for v in site["Cartn_x"]], list(range(8)))


def exercise_secondary_structure_follows_the_relabelled_chains():
    """SS arrives keyed by author chain but is matched on ``label_asym_id``, so the ranges
    have to be retargeted -- otherwise Mol* finds no residues in them and the cartoon loses
    every helix and strand. A repeated chain's SS belongs to its polymer block."""
    b = block(encode_bcif_arrays(
        two_chains_with_waters(),
        secondary_structure=[("E", 1, 3, "helix"), ("I", 1, 2, "sheet")]))
    assert b["_struct_conf"]["beg_label_asym_id"][0] == "A"          # E's polymer block
    assert b["_struct_sheet_range"]["beg_label_asym_id"][0] == "B"   # I's polymer block


def exercise_a_single_chain_is_still_labelled_a():
    """The common case is unchanged -- one chain, one label."""
    site = block(encode_bcif_arrays(protein_arrays()))["_atom_site"]
    assert set(site["label_asym_id"]) == set(["A"])
    assert set(site["auth_asym_id"]) == set(["A"])


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
