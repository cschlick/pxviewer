"""The shared validation-extraction layer.

``pxviewer/validation_events.py`` is shared verbatim with the hotspots generator, which
resolves it by path rather than keeping a copy. These tests pin the two things that sharing
is supposed to guarantee: the same atoms get implicated as pxviewer's own scoring uses, and
the field-agreement check actually discriminates a correct field from a wrong one.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import data_path, have, skip

if not have("mmtbx", "numpy"):
    skip("mmtbx/numpy not available")

import numpy as np                                              # noqa: E402

from pxviewer import hotspots, validation_events as ve          # noqa: E402

_model_cache = []


def model():
    """The test model, read once per process."""
    if not _model_cache:
        from pxviewer.cctbx_io import read_model

        _model_cache.append(read_model(data_path("1tec.pdb")))
    return _model_cache[0]


# -- one definition, one file -------------------------------------------------


def exercise_single_copy():
    """A second copy that drifts looks like one definition while being two, which is the
    failure this module exists to prevent. hotspots/ resolves it by path instead."""
    import pxviewer

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(pxviewer.__file__))))
    copies = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git", "node_modules")]
        if "validation_events.py" in filenames:
            copies.append(os.path.join(dirpath, "validation_events.py"))
    assert len(copies) == 1, "expected exactly one validation_events.py, found %s" % copies


# -- the shared layer really is what pxviewer scores from ---------------------


def exercise_severity_uses_the_shared_extractor():
    """pxviewer's severity and the generator's concern must disagree about the *scale* and
    agree about the *place*. Both read atoms from this module."""
    m = model()
    n = m.get_number_of_atoms()
    hierarchy = m.get_hierarchy()
    names = [s.strip() for s in hierarchy.atoms().extract_name()]

    for metric, severity_fn, allowed, forbidden in (
        ("rama", hotspots.ramachandran_severity, ve.RAMA_ATOMS, None),
        ("rota", hotspots.rotamer_severity, None, ve.MAINCHAIN),
    ):
        events = (ve.extract_ramachandran(hierarchy) if metric == "rama"
                  else ve.extract_rotamer(hierarchy))
        cut = hotspots.RAMA_OUTLIER_PCT if metric == "rama" else hotspots.ROTA_OUTLIER_PCT
        from_events = set()
        for e in events:
            if hotspots._surprisal_severity(np.array([e.value]), cut)[0] > 0:
                from_events.update(e.atom_indices)

        from_severity = set(np.flatnonzero(severity_fn(m, n) > 0).tolist())
        assert from_severity == from_events, \
            "%s: severity and extractor disagree" % metric

        for i in from_events:
            if allowed is not None:
                assert names[i] in allowed
            if forbidden is not None:
                assert names[i] not in forbidden


def exercise_clash_events_are_per_contact():
    """probe2 emits a row per surface dot, so a single contact arrives dozens of times. A
    consumer depositing one kernel per event would weight a contact by how thoroughly it
    happened to be dotted."""
    events = ve.extract_clashes(model())
    assert events, "1TEC should have heavy-atom clashes"

    pairs = [tuple(e.detail["pair"]) for e in events]
    assert len(pairs) == len(set(pairs)), "each atom pair must appear exactly once"
    assert len(events) < 2000
    assert all(e.units == "angstrom" and e.value > 0 for e in events)
    assert all(e.outlier == (e.value >= ve.CLASH_OUTLIER_A) for e in events)


def exercise_native_values_are_never_calibrated():
    """Percentages stay percentages. If this module emitted a score, the projects would
    silently inherit one calibration and the split would be pointless."""
    rama = ve.extract_ramachandran(model().get_hierarchy())
    assert all(e.units == "percent" for e in rama)
    assert max(e.value for e in rama) > 1.0        # a percent, not a probability
    assert any(e.outlier for e in rama)


# -- the covalent channel -----------------------------------------------------


def exercise_bond_and_angle_native_deviations():
    """Deviations travel in their own units with the Z in detail, as everywhere else."""
    m = model()
    geometry = ve._restraints_geometry(m)
    bonds = ve.extract_bonds(m, geometry=geometry)
    angles = ve.extract_angles(m, geometry=geometry)
    assert bonds and angles

    assert all(e.units == "angstrom" and len(e.atom_indices) == 2 for e in bonds)
    assert all(e.units == "degree" and len(e.atom_indices) == 3 for e in angles)
    for events in (bonds, angles):
        e = events[0]
        assert approx_equal(e.value, abs(e.detail["delta"]))
        assert approx_equal(e.detail["ideal"] - e.detail["model"], e.detail["delta"])
        assert approx_equal(e.detail["z"], abs(e.detail["delta"]) / e.detail["sigma"])
        assert all(x.outlier == (x.detail["z"] >= 4.0) for x in events)

    rmsd = np.sqrt(np.mean([e.value ** 2 for e in bonds]))
    assert 0.0 < rmsd < 0.1


def exercise_injected_geometry_keeps_host_edits():
    """Building restraints inside the extractor would discard a host's custom restraints.

    pxviewer folds user bond/angle edits in through edits.build_restraints (one build path,
    one lock). A plain process(make_restraints=True) ignores those, and since an existing
    manager is reused rather than rebuilt, the edit-less one would be inherited by minimize
    and drag -- silently dropping the user's restraint.
    """
    if not have("rdkit", "mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping edits check: rdkit / monomer library not available")
        return
    from pxviewer import edits, ligands
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        print("  skipping edits check: no monomer library")
        return

    ligand = ligands.build_ligand_from_smiles("CCO", "EOH", (0, 0, 0))
    plain = ligand.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size()
    ligand.unset_restraints_manager()

    names = [a.name.strip() for a in ligand.get_hierarchy().atoms()]
    sels = [edits.selection_for_atom(ligand, names.index("C1")),
            edits.selection_for_atom(ligand, names.index("O1"))]
    scope = edits.empty_edits(ligand)
    edits.add_entry(
        scope, edits.new_entry(ligand, "bond", sels, ideal=2.4, sigma=0.02), "bond")
    edits.set_edits(ligand, scope)
    edits.build_restraints(ligand, force=True)

    events = ve.extract_bonds(ligand,
                              geometry=ligand.get_restraints_manager().geometry)
    assert len(events) == plain + 1
    assert any(approx_equal(e.detail["ideal"], 2.4, out=None) for e in events)

    edits.build_restraints(ligand)
    assert ligand.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size() == plain + 1


# -- the field/validation agreement check -------------------------------------


def _synthetic():
    """Ten atoms in a line 10 A apart; atoms 0-1 are the only outlier."""
    sites = np.zeros((10, 3))
    sites[:, 0] = np.arange(10) * 10.0
    events = [
        ve.ValidationEvent(metric="rama", value=0.01, units="percent", outlier=True,
                           atom_indices=(0, 1)),
        ve.ValidationEvent(metric="rama", value=1.0, units="percent", outlier=False,
                           atom_indices=(4,)),
        ve.ValidationEvent(metric="rama", value=50.0, units="percent", outlier=False,
                           atom_indices=(8,)),
    ]
    return sites, events


def exercise_agreement_catches_a_missed_outlier():
    """Recall is the guarantee such a field owes: never lose a flagged outlier."""
    sites, events = _synthetic()
    sampled = np.zeros(10)
    sampled[0] = 0.9          # only half the outlier is marked
    report = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                      hot_threshold=0.5)
    assert report.n_outlier_atoms == 2 and report.n_covered == 1
    assert approx_equal(report.recall, 0.5)
    assert [m["atom"] for m in report.missed] == [1]
    assert not report.ok


def exercise_agreement_accepts_spillover_but_not_signal_from_nowhere():
    """A hot atom beside a bad one is expected -- the splat is wider than a residue. A hot
    atom with nothing bad anywhere near it is the real defect."""
    sites, events = _synthetic()
    sampled = np.zeros(10)
    sampled[[0, 1]] = 1.0
    sampled[2] = 0.8          # 10 A away
    report = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                      hot_threshold=0.5, tolerance_a=12.0)
    assert report.recall == 1.0 and report.explained == 1.0 and report.ok

    tight = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                     hot_threshold=0.5, tolerance_a=4.0)
    assert tight.recall == 1.0
    assert [u["atom"] for u in tight.unexplained] == [2]
    assert approx_equal(tight.unexplained[0]["nearest_outlier_a"], 10.0)


def exercise_explanation_can_be_widened_past_the_cut():
    """A continuous field is built from every result, not just flagged ones, so judging it
    against outliers alone reports correct behaviour as failure."""
    sites, events = _synthetic()
    sampled = np.zeros(10)
    sampled[[0, 1]] = 1.0
    sampled[4] = 0.7          # allowed-but-unusual, which the concern curve still marks

    strict = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                      hot_threshold=0.5, tolerance_a=4.0)
    assert [u["atom"] for u in strict.unexplained] == [4]

    widened = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                       hot_threshold=0.5, tolerance_a=4.0,
                                       concerning=ve.worse_than_percent(2.0))
    assert widened.ok and widened.explained == 1.0
    assert widened.n_outlier_atoms == 2      # recall still keys off the outlier flag


def exercise_agreement_rejects_a_shuffled_field():
    """The negative control: if a wrong field passes, the check is worthless."""
    sites, events = _synthetic()
    sampled = np.zeros(10)
    sampled[[8, 9]] = 1.0     # hot at the far end, nowhere near the outlier
    report = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                      hot_threshold=0.5, tolerance_a=4.0,
                                      concerning=ve.worse_than_percent(2.0))
    assert report.recall == 0.0
    assert len(report.missed) == 2
    assert not report.ok


# -- the two roll-ups ---------------------------------------------------------


def _map_fit_events():
    """A cc gap over three atoms; a fourth atom was never measured."""
    return [
        ve.ValidationEvent(metric="cc_gap", value=-0.12, units="correlation",
                           outlier=False, atom_indices=(0,)),
        ve.ValidationEvent(metric="cc_gap", value=-0.05, units="correlation",
                           outlier=False, atom_indices=(1,)),
        ve.ValidationEvent(metric="cc_gap", value=0.03, units="correlation",
                           outlier=True, atom_indices=(2,)),
    ]


def exercise_per_atom_field_keeps_negatives():
    """Map-fit values are a continuous scale, not badness-from-zero. A negative correlation
    is the worst possible fit, and 0.0 is a real reading -- for a resolution in angstroms it
    is the *best* possible one."""
    values = ve.per_atom_field(_map_fit_events(), 4, metric="cc_gap")
    assert approx_equal(values[0], -0.12)
    assert approx_equal(values[1], -0.05)
    assert approx_equal(values[2], 0.03)
    assert np.isnan(values[3])            # never measured, not "fits perfectly"


def exercise_per_atom_refuses_map_fit_on_severity_defaults():
    """The corrupted field this prevents is silent, so the guard has to be loud."""
    events = _map_fit_events()
    with raises(ValueError) as e:
        ve.per_atom(events, 4, metric="cc_gap")
    assert "per_atom_field" in str(e.value)

    onto_severity = ve.per_atom(events, 4, metric="cc_gap",
                                transform=lambda ev: max(0.0, -ev.value))
    assert approx_equal(list(onto_severity), [0.12, 0.05, 0.0, 0.0])
    explicit = ve.per_atom(events, 4, metric="cc_gap", skip_nonpositive=False,
                           fill=np.nan)
    assert approx_equal(explicit[0], -0.12) and np.isnan(explicit[3])


def exercise_guard_only_inspects_the_requested_metric():
    """A geometry roll-up must not be blocked by map-fit events sharing the list."""
    mixed = _map_fit_events() + [
        ve.ValidationEvent(metric="clash", value=0.5, units="angstrom", outlier=True,
                           atom_indices=(0, 1))]
    assert approx_equal(list(ve.per_atom(mixed, 3, metric="clash")), [0.5, 0.5, 0.0])


def exercise_per_atom_rolls_up_with_max():
    """One mistake seen several times is still one mistake, and a sum would rank a large
    residue above a small one for no reason but atom count."""
    events = [
        ve.ValidationEvent(metric="clash", value=0.5, units="angstrom", outlier=True,
                           atom_indices=(0, 1)),
        ve.ValidationEvent(metric="clash", value=0.8, units="angstrom", outlier=True,
                           atom_indices=(1, 2)),
        ve.ValidationEvent(metric="rama", value=99.0, units="percent", outlier=False,
                           atom_indices=(0,)),
    ]
    assert approx_equal(list(ve.per_atom(events, 4, metric="clash")),
                        [0.5, 0.8, 0.8, 0.0])       # atom 1 keeps its worst, not 1.3
    assert approx_equal(list(ve.per_atom(events, 4, metric="clash", outliers_only=True)),
                        [0.5, 0.8, 0.8, 0.0])
    doubled = ve.per_atom(events, 4, metric="clash", transform=lambda e: e.value * 2)
    assert approx_equal(list(doubled), [1.0, 1.6, 1.6, 0.0])


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
