"""Alternate conformations, from the file through to the bytes Mol* is handed.

The shipped model is 3NIR -- crambin at 0.48 A, 46 residues -- chosen because it is small
and unusually rich in the cases that break altloc handling: four labels (A, B, C and a
single lone D), three conformers on one tyrosine, and 44 water atoms carrying altlocs, so
the ``HETATM`` path is covered as well as ``ATOM``. Half its atoms sit in an alternate
conformation, which makes "the conformers were silently flattened" a loud failure rather
than a quiet one.

**What Mol* does with this was measured, not assumed.** The topology these exercises build
was fed to Mol* 5.10.1's own mmCIF reader and model builder (via node, against
``frontend/node_modules``), which reported: all 1026 atoms kept; ``label_alt_id`` carrying
A/B/C/D with the rest empty; and -- the property that matters -- **zero bonds between
atoms in different conformers**, so a serine's A-conformer OG is never bonded to its
B-conformer CB. Secondary structure survives alongside the altlocs, so cartoon still
draws. pxviewer's frontend contains no altloc code at all: this is Mol*'s default
behaviour, and these exercises pin the input it depends on.

One encoding detail is deliberate and worth stating, because it looks like a bug. A blank
altloc is written as the empty string with no BinaryCIF mask, where a canonical mmCIF file
would write ``.`` and mark the value absent. Mol* reads both to the same thing -- its
``str()`` returns ``""`` either way, and the built model stores ``""`` -- so the shortcut
is harmless. It is pinned below so that if it ever stops being harmless, a test says so.
"""

from __future__ import absolute_import, division, print_function

import collections
import struct
import sys

from pxviewer.regression.tst_utils import data_path, have, skip

if not have("iotbx.data_manager", "numpy", "msgpack"):
    skip("iotbx.data_manager / numpy / msgpack not available")

import msgpack                                        # noqa: E402

from pxviewer.cctbx_io import model_to_arrays, read_model   # noqa: E402
from pxviewer.live import LiveSession                 # noqa: E402

#: What 3NIR contains. Hard-coded rather than recomputed from the file: a test that derives
#: its expectation from the thing it is testing would still pass if the file were replaced.
TOTAL_ATOMS = 1026
ALTLOC_COUNTS = {"A": 258, "B": 253, "C": 22, "D": 1, "": 492}

_model_cache = []


def crambin():
    """The parsed model, read once -- reading it is ~1 s and nothing here mutates it."""
    if not _model_cache:
        _model_cache.append(read_model(data_path("3nir.pdb")))
    return _model_cache[0]


def atom_site_columns(session):
    """``{column name: column}`` from the BinaryCIF topology the session sends."""
    doc = msgpack.unpackb(session._topology, raw=False)
    categories = {c["name"]: c for c in doc["dataBlocks"][0]["categories"]}
    site = categories["_atom_site"]
    return site, {c["name"]: c for c in site["columns"]}


def string_column_values(column):
    """Decode a StringArray column back to one string per row."""
    encoding = column["data"]["encoding"][0]
    assert encoding["kind"] == "StringArray", encoding["kind"]
    data = column["data"]["data"]
    indices = [struct.unpack_from("<i", data, 4 * i)[0]
               for i in range(len(data) // 4)]
    offsets = [struct.unpack_from("<i", encoding["offsets"], 4 * i)[0]
               for i in range(len(encoding["offsets"]) // 4)]
    text = encoding["stringData"]
    uniques = [text[offsets[k]:offsets[k + 1]] for k in range(len(offsets) - 1)]
    return [uniques[i] for i in indices]


# -- the model reaches the arrays intact --------------------------------------


def exercise_every_conformer_survives_as_its_own_atom():
    """The failure this guards against is flattening -- keeping one conformer per atom,
    which silently discards half of this structure."""
    arrays = model_to_arrays(crambin())
    assert len(arrays) == TOTAL_ATOMS
    assert collections.Counter(arrays.altloc) == ALTLOC_COUNTS


def exercise_a_residue_can_carry_three_conformers():
    """Two is the common case and the one a naive implementation handles; TYR 29 has
    three, and the lone D atom elsewhere makes four labels in the file."""
    arrays = model_to_arrays(crambin())
    by_residue = collections.defaultdict(set)
    for resseq, alt in zip(arrays.resseq, arrays.altloc):
        if alt:
            by_residue[int(resseq)].add(alt)
    assert max(len(alts) for alts in by_residue.values()) == 3
    assert by_residue[29] == {"A", "B", "C"}


def exercise_waters_carry_altlocs_too():
    """Altlocs are not a polymer-only concern: 3NIR has them on waters, which arrive as
    HETATM and travel a different path through the hierarchy.

    The solvent is where this structure is at its most awkward -- all four labels appear
    on waters, and the single ``D`` atom in the entire file is one of them. An
    implementation that assumed altlocs come in pairs, or that only polymer residues have
    them, breaks here rather than somewhere subtle.
    """
    arrays = model_to_arrays(crambin())
    water_alts = [alt for resname, alt in zip(arrays.resname, arrays.altloc)
                  if resname == "HOH" and alt]
    assert collections.Counter(water_alts) == {"A": 22, "B": 20, "C": 1, "D": 1}

    # The lone D in the file is that water, not a polymer atom.
    assert sum(1 for alt in arrays.altloc if alt == "D") == 1


# -- selections address conformers --------------------------------------------


def exercise_each_conformer_can_be_selected_by_name():
    session = LiveSession.from_cctbx_model(crambin())
    assert session._n_atoms == TOTAL_ATOMS
    for label, expected in sorted(ALTLOC_COUNTS.items()):
        if not label:
            continue
        assert len(session.select_by(selection="altloc %s" % label)) == expected


def exercise_selecting_an_atom_name_returns_all_of_its_conformers():
    """The reason a viewer needs this: picking an atom that exists in two conformers must
    reach both, or the user edits one and wonders why the density still does not fit.

    THR 1 is modelled in two conformations, so every one of its atoms is doubled -- 133
    named heavy atoms across the structure are, which is what makes this the normal case
    here rather than a curiosity.
    """
    session = LiveSession.from_cctbx_model(crambin())
    assert len(session.select_by(selection="resseq 1 and name CA")) == 2
    assert len(session.select_by(selection="resseq 1 and name OG1")) == 2


# -- what actually goes to Mol* ------------------------------------------------


def exercise_the_topology_carries_label_alt_id():
    """The column is written conditionally -- only when some atom has an altloc -- so
    before this model was added the branch that writes it had never run in the suite."""
    session = LiveSession.from_cctbx_model(crambin())
    site, columns = atom_site_columns(session)
    assert site["rowCount"] == TOTAL_ATOMS
    assert "label_alt_id" in columns

    values = string_column_values(columns["label_alt_id"])
    assert len(values) == TOTAL_ATOMS
    assert collections.Counter(values) == ALTLOC_COUNTS


def exercise_a_model_without_altlocs_omits_the_column():
    """The other half of the conditional: 1UBQ has no alternate conformations, so the
    column is left out rather than sent as 1231 empty strings."""
    session = LiveSession.from_cctbx_model(read_model(data_path("1ubq.pdb")))
    _site, columns = atom_site_columns(session)
    assert "label_alt_id" not in columns


def exercise_a_blank_altloc_is_sent_as_the_empty_string():
    """Pins the shortcut described in the module docstring: no BinaryCIF mask, so a blank
    altloc is an empty string rather than mmCIF's absent ``.``.

    Mol* reads the two identically, which is why this is allowed to stand. If a future
    Mol* stops treating them the same, this exercise is where the assumption is written
    down -- change it here, and add the mask in ``bcif.string_column``.
    """
    session = LiveSession.from_cctbx_model(crambin())
    _site, columns = atom_site_columns(session)
    column = columns["label_alt_id"]

    assert column["mask"] is None                     # every row "present"
    assert string_column_values(column).count("") == ALTLOC_COUNTS[""]


def exercise_occupancies_accompany_the_conformers():
    """Alternate conformations are only interpretable with their occupancies -- two
    half-occupied conformers are one atom's worth of density, not two.

    Note what is *not* asserted. The textbook invariant is that an atom's conformers sum
    to 1.0, and it does not hold here: at 0.48 A the occupancies were refined freely, so
    the 273 conformer groups sum to anywhere between 0.34 and 1.12 and only 154 land
    within 2% of unity. Three atoms even carry an altloc at full occupancy. That is the
    deposited data, not a parsing error, and a test asserting the tidy version would fail
    on the real file -- so what is pinned is the range, which is what actually breaks if
    occupancies are dropped or defaulted to 1.
    """
    session = LiveSession.from_cctbx_model(crambin())
    _site, columns = atom_site_columns(session)
    assert "occupancy" in columns

    arrays = model_to_arrays(crambin())
    partial = [occ for occ, alt in zip(arrays.occ, arrays.altloc) if alt]
    assert len(partial) == 534
    assert all(0.0 < occ <= 1.0 for occ in partial)
    assert sum(1 for occ in partial if occ < 1.0) == 531   # all but three


def exercise_secondary_structure_survives_the_altlocs():
    """Crambin has a helix and two sheets. Cartoon is what the app draws by default, so
    an altloc-heavy model that lost its annotation would render as a tangle of tubes."""
    session = LiveSession.from_cctbx_model(crambin())
    doc = msgpack.unpackb(session._topology, raw=False)
    categories = {c["name"]: c for c in doc["dataBlocks"][0]["categories"]}
    assert categories["_struct_conf"]["rowCount"] == 1
    assert categories["_struct_sheet_range"]["rowCount"] == 2


# -- describing a picked atom --------------------------------------------------


def exercise_a_picked_atom_names_its_conformer():
    """Two conformers of an atom agree on name, residue and chain, so a label without the
    altloc names both of them identically -- which is what the viewer used to do."""
    from pxviewer.live import describe_atom

    a = {"id": 1, "name": "CA", "resname": "THR", "resseq": 1, "chain": "A", "altloc": "A"}
    b = dict(a, id=2, altloc="B")
    assert describe_atom(a) == "CA THR1 (alt A)"
    assert describe_atom(b) == "CA THR1 (alt B)"
    assert describe_atom(a) != describe_atom(b)


def exercise_an_atom_without_a_conformer_is_described_plainly():
    """The common case by far: no trailing "(alt )" noise on ordinary structures."""
    from pxviewer.live import describe_atom

    assert describe_atom(
        {"id": 1, "name": "CA", "resname": "GLY", "resseq": 7, "chain": "A",
         "altloc": ""}) == "CA GLY7"
    # A viewer built before altlocs were sent omits the key entirely.
    assert describe_atom(
        {"id": 1, "name": "CA", "resname": "GLY", "resseq": 7}) == "CA GLY7"
    assert describe_atom(None) == "empty space"


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
