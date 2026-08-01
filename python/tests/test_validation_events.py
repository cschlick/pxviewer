"""The shared validation-extraction layer.

`pxviewer/validation_events.py` is copied verbatim into the sibling hotspots generator so
both projects localize validation identically. These tests pin the two things that copy is
supposed to guarantee: the same atoms get implicated as pxviewer's own scoring uses, and the
field-agreement check actually discriminates a correct field from a wrong one.
"""

import numpy as np
import pytest

pytest.importorskip("mmtbx")

from pxviewer import hotspots, validation_events as ve  # noqa: E402

_MODEL = "python/pxviewer/data/1tec.pdb"


@pytest.fixture(scope="module")
def model():
    from pxviewer.cctbx_io import read_model

    return read_model(_MODEL)


# -- the shared layer really is what pxviewer scores from ---------------------


def test_severity_implicates_exactly_the_atoms_the_shared_extractor_picks(model):
    """pxviewer's severity and the generator's concern must disagree about the *scale* and
    agree about the *place*. Both read atoms from this module, so a rule change cannot land
    in one project and not the other — this pins that they are in fact wired to it."""
    n = model.get_number_of_atoms()
    hierarchy = model.get_hierarchy()
    names = [s.strip() for s in hierarchy.atoms().extract_name()]

    for metric, severity_fn, allowed, forbidden in (
        ("rama", hotspots.ramachandran_severity, ve.RAMA_ATOMS, None),
        ("rota", hotspots.rotamer_severity, None, ve.MAINCHAIN),
    ):
        events = (ve.extract_ramachandran(hierarchy) if metric == "rama"
                  else ve.extract_rotamer(hierarchy))
        # Atoms the extractor implicates for results bad enough to score above zero.
        cut = hotspots.RAMA_OUTLIER_PCT if metric == "rama" else hotspots.ROTA_OUTLIER_PCT
        from_events = set()
        for e in events:
            if hotspots._surprisal_severity(np.array([e.value]), cut)[0] > 0:
                from_events.update(e.atom_indices)

        from_severity = set(np.flatnonzero(severity_fn(model, n) > 0).tolist())
        assert from_severity == from_events, f"{metric}: severity and extractor disagree"

        # And the localization rule itself still holds on the shared side.
        for i in from_events:
            if allowed is not None:
                assert names[i] in allowed
            if forbidden is not None:
                assert names[i] not in forbidden


def test_clash_events_are_one_per_contact_not_one_per_probe_dot(model):
    """probe2 emits a row per surface dot, so a single contact arrives dozens of times.

    A consumer that deposits one Gaussian per event would otherwise weight a contact by how
    thoroughly it happens to be dotted — and the generator sums within a metric before
    clipping, so that is the difference between a faithful field and a saturated one.
    """
    events = ve.extract_clashes(model)
    assert events, "1TEC should have heavy-atom clashes"

    pairs = [tuple(e.detail["pair"]) for e in events]
    assert len(pairs) == len(set(pairs)), "each atom pair must appear exactly once"
    # Far fewer events than dots, but the same atoms implicated as the per-dot roll-up.
    assert len(events) < 2000
    assert all(e.units == "angstrom" and e.value > 0 for e in events)
    # The outlier flag is MolProbity's reporting boundary, not a re-derivation.
    assert all(e.outlier == (e.value >= ve.CLASH_OUTLIER_A) for e in events)


def test_native_values_are_never_calibrated(model):
    """Percentages stay percentages. If this module ever started emitting a score, the two
    projects would silently inherit one calibration and the split would be pointless."""
    rama = ve.extract_ramachandran(model.get_hierarchy())
    assert all(e.units == "percent" for e in rama)
    assert max(e.value for e in rama) > 1.0        # a percent, not a probability
    assert any(e.outlier for e in rama)
    # The outlier flag is the validator's own, so it is recoverable exactly.
    assert all(e.outlier == (e.value <= 0.05) or not e.outlier for e in rama)


# -- the covalent channel -----------------------------------------------------


def test_bond_and_angle_events_carry_native_deviations(model):
    """Deviations travel in their own units with the Z in detail, like every other channel."""
    bonds = ve.extract_bonds(model)
    angles = ve.extract_angles(model)
    assert bonds and angles

    assert all(e.units == "angstrom" and len(e.atom_indices) == 2 for e in bonds)
    assert all(e.units == "degree" and len(e.atom_indices) == 3 for e in angles)
    for events in (bonds, angles):
        e = events[0]
        # value is |delta| and delta is ideal - model, both recorded.
        assert e.value == pytest.approx(abs(e.detail["delta"]))
        assert e.detail["ideal"] - e.detail["model"] == pytest.approx(e.detail["delta"])
        # Z is the deviation in sigmas, and the outlier flag is the 4-sigma cut.
        assert e.detail["z"] == pytest.approx(abs(e.detail["delta"]) / e.detail["sigma"])
        assert all(x.outlier == (x.detail["z"] >= 4.0) for x in events)

    # 1TEC is a real refined structure: deviations are small but not all zero.
    assert 0.0 < np.sqrt(np.mean([e.value ** 2 for e in bonds])) < 0.1


def test_extract_bonds_takes_an_injected_geometry_so_a_host_keeps_its_edits():
    """Building restraints inside the extractor would discard a host's custom restraints.

    pxviewer folds user bond/angle edits in through ``edits.build_restraints`` (one build
    path, one lock). A plain ``model.process(make_restraints=True)`` ignores those edits, and
    since an existing restraints manager is reused rather than rebuilt, the edit-less manager
    would then be inherited by minimize and drag — silently dropping the user's restraint.
    Passing ``geometry=`` is how a host stays in control of its own build.
    """
    pytest.importorskip("rdkit")
    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    from pxviewer import edits, ligands
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library")

    ligand = ligands.build_ligand_from_smiles("CCO", "EOH", (0, 0, 0))
    plain = ligand.get_restraints_manager().geometry.pair_proxies().bond_proxies.simple.size()
    ligand.unset_restraints_manager()

    names = [a.name.strip() for a in ligand.get_hierarchy().atoms()]
    sels = [edits.selection_for_atom(ligand, names.index("C1")),
            edits.selection_for_atom(ligand, names.index("O1"))]
    edits.set_edits(ligand, [{"kind": "bond", "selections": sels,
                              "ideal": 2.4, "sigma": 0.02}])
    edits.build_restraints(ligand, force=True)

    events = ve.extract_bonds(
        ligand, geometry=ligand.get_restraints_manager().geometry)
    assert len(events) == plain + 1
    assert any(e.detail["ideal"] == pytest.approx(2.4) for e in events)

    # And the host's manager is untouched, so a later build still has the edit.
    edits.build_restraints(ligand)
    assert ligand.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size() == plain + 1


# -- the sanity check ---------------------------------------------------------


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


def test_agreement_fails_a_field_that_misses_a_real_outlier():
    """Recall is the guarantee such a field owes: never lose a flagged outlier."""
    sites, events = _synthetic()
    sampled = np.zeros(10)
    sampled[0] = 0.9          # only half the outlier is marked
    report = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                      hot_threshold=0.5)
    assert report.n_outlier_atoms == 2 and report.n_covered == 1
    assert report.recall == pytest.approx(0.5)
    assert [m["atom"] for m in report.missed] == [1]
    assert not report.ok


def test_agreement_accepts_spillover_but_not_signal_from_nowhere():
    """A hot atom beside a bad one is expected — the splat is wider than a residue. A hot
    atom with nothing bad anywhere near it is the real defect."""
    sites, events = _synthetic()
    sampled = np.zeros(10)
    sampled[[0, 1]] = 1.0
    sampled[2] = 0.8          # 10 A away: outside the tolerance below
    report = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                      hot_threshold=0.5, tolerance_a=12.0)
    assert report.recall == 1.0 and report.explained == 1.0 and report.ok

    tight = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                     hot_threshold=0.5, tolerance_a=4.0)
    assert tight.recall == 1.0                      # still marks every outlier
    assert [u["atom"] for u in tight.unexplained] == [2]
    assert tight.unexplained[0]["nearest_outlier_a"] == pytest.approx(10.0)


def test_explanation_can_be_widened_past_the_outlier_cut():
    """A continuous field is built from every result, not just flagged ones, so judging it
    against outliers alone reports correct behaviour as failure."""
    sites, events = _synthetic()
    sampled = np.zeros(10)
    sampled[[0, 1]] = 1.0
    sampled[4] = 0.7          # an allowed-but-unusual residue the concern curve still marks

    strict = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                      hot_threshold=0.5, tolerance_a=4.0)
    assert [u["atom"] for u in strict.unexplained] == [4]

    widened = ve.check_field_agreement(events, sampled, sites, metric="rama",
                                       hot_threshold=0.5, tolerance_a=4.0,
                                       concerning=ve.worse_than_percent(2.0))
    assert widened.ok and widened.explained == 1.0
    # Recall still keys off the outlier flag, not the widened predicate.
    assert widened.n_outlier_atoms == 2


def test_agreement_rejects_a_shuffled_field():
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


def test_per_atom_field_keeps_negatives_and_marks_unmeasured_atoms():
    """Map-fit values are a continuous scale, not badness-from-zero.

    A negative correlation is the worst possible fit, not the absence of one, and 0.0 is a
    real reading — for a resolution in angstroms it is the *best* possible one. So nothing is
    filtered and unmeasured atoms are nan rather than zero.
    """
    values = ve.per_atom_field(_map_fit_events(), 4, metric="cc_gap")
    assert values[0] == pytest.approx(-0.12)
    assert values[1] == pytest.approx(-0.05)
    assert values[2] == pytest.approx(0.03)
    assert np.isnan(values[3])            # never measured, not "fits perfectly"


def test_per_atom_refuses_map_fit_events_on_the_severity_defaults():
    """The corrupted field this prevents is silent, so the guard has to be loud.

    With the severity defaults a normally-negative cc gap collapses to a field of zeros,
    indistinguishable from atoms that were never measured at all.
    """
    events = _map_fit_events()
    with pytest.raises(ValueError, match="per_atom_field"):
        ve.per_atom(events, 4, metric="cc_gap")

    # Saying what you mean is always allowed: an explicit transform maps a correlation onto a
    # badness scale on purpose, and explicit flags ask for the continuous behaviour.
    onto_severity = ve.per_atom(events, 4, metric="cc_gap",
                                transform=lambda e: max(0.0, -e.value))
    assert onto_severity.tolist() == [pytest.approx(0.12), pytest.approx(0.05), 0.0, 0.0]
    explicit = ve.per_atom(events, 4, metric="cc_gap", skip_nonpositive=False, fill=np.nan)
    assert explicit[0] == pytest.approx(-0.12) and np.isnan(explicit[3])


def test_the_guard_only_looks_at_the_metric_being_asked_for():
    """A caller rolling up a geometry metric must not be blocked by map-fit events that
    happen to share the list."""
    mixed = _map_fit_events() + [
        ve.ValidationEvent(metric="clash", value=0.5, units="angstrom", outlier=True,
                           atom_indices=(0, 1))]
    assert ve.per_atom(mixed, 3, metric="clash").tolist() == [0.5, 0.5, 0.0]


def test_per_atom_rolls_up_with_max_never_sum():
    """One mistake seen several times is still one mistake — and a sum would rank a large
    residue above a small one for no reason but atom count."""
    events = [
        ve.ValidationEvent(metric="clash", value=0.5, units="angstrom", outlier=True,
                           atom_indices=(0, 1)),
        ve.ValidationEvent(metric="clash", value=0.8, units="angstrom", outlier=True,
                           atom_indices=(1, 2)),
        ve.ValidationEvent(metric="rama", value=99.0, units="percent", outlier=False,
                           atom_indices=(0,)),
    ]
    values = ve.per_atom(events, 4, metric="clash")
    assert values.tolist() == [0.5, 0.8, 0.8, 0.0]   # atom 1 keeps its worst, not 1.3
    assert ve.per_atom(events, 4, metric="clash", outliers_only=True).tolist() == \
        [0.5, 0.8, 0.8, 0.0]
    # A transform is how each project applies its own calibration to shared events.
    doubled = ve.per_atom(events, 4, metric="clash", transform=lambda e: e.value * 2)
    assert doubled.tolist() == [1.0, 1.6, 1.6, 0.0]
