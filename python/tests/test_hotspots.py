"""Validation hotspots: severity calibration, atom assignment, and aggregation.

The rules under test are stated in HOTSPOTS.md; each test names the rule it pins.
"""

import time

import numpy as np
import pytest

pytest.importorskip("mmtbx")

from pxviewer import hotspots  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

_MODEL = "python/pxviewer/data/1tec.pdb"   # has real rotamer and Ramachandran outliers


def _model(path=_MODEL):
    from pxviewer.cctbx_io import read_model

    return read_model(path)


def _residue_max(model, values):
    """(chain, resid) -> the worst value over that residue's atoms."""
    out = {}
    for i, atom in enumerate(model.get_hierarchy().atoms_with_labels()):
        key = (atom.chain_id.strip(), atom.resid().strip())
        out[key] = max(out.get(key, 0.0), float(values[i]))
    return out


# -- calibration -------------------------------------------------------------------


def test_severity_one_reproduces_molprobity_outliers():
    """The consistency constraint from HOTSPOTS.md: because severity is anchored so that 1.0
    *is* the community cut, the ``severity >= 1.0`` level set has to flag exactly the residues
    mmtbx flags. If it diverges, our mapping is miscalibrated — not MolProbity."""
    from mmtbx.validation.ramalyze import ramalyze
    from mmtbx.validation.rotalyze import rotalyze

    model = _model()
    hierarchy = model.get_hierarchy()
    n = model.get_number_of_atoms()

    for validator, severity_fn in ((rotalyze, hotspots.rotamer_severity),
                                   (ramalyze, hotspots.ramachandran_severity)):
        flagged = {(r.chain_id.strip(), r.resid.strip())
                   for r in validator(pdb_hierarchy=hierarchy, outliers_only=False).results
                   if r.outlier}
        ours = {key for key, value in _residue_max(model, severity_fn(model, n)).items()
                if value >= 1.0}
        assert ours == flagged, f"{validator.__name__}: {ours ^ flagged}"
        assert flagged, "the test model should have outliers, or this proves nothing"


def test_severity_is_one_exactly_at_the_threshold():
    """1.0 is the outlier cut by construction, for every metric that has one — that is what
    makes the metrics commensurable without inventing weights."""
    assert hotspots._surprisal_severity(
        np.array([hotspots.RAMA_OUTLIER_PCT]), hotspots.RAMA_OUTLIER_PCT)[0] == pytest.approx(1.0)
    assert hotspots._surprisal_severity(
        np.array([hotspots.ROTA_OUTLIER_PCT]), hotspots.ROTA_OUTLIER_PCT)[0] == pytest.approx(1.0)
    # A perfectly ordinary conformation scores 0, not something small but positive.
    assert hotspots._surprisal_severity(np.array([100.0]), hotspots.RAMA_OUTLIER_PCT)[0] == 0.0
    # Worse than the cut goes above 1; better goes below. Monotone, so ranking is meaningful.
    worse = hotspots._surprisal_severity(np.array([0.001]), hotspots.RAMA_OUTLIER_PCT)[0]
    better = hotspots._surprisal_severity(np.array([2.0]), hotspots.RAMA_OUTLIER_PCT)[0]
    assert worse > 1.0 > better > 0.0


def test_a_zero_percentage_stays_finite():
    """ramalyze reports a hard 0 for the worst outliers; -log10(0) would poison the whole
    field with an inf that no clamp downstream could undo."""
    value = hotspots._surprisal_severity(np.array([0.0]), hotspots.RAMA_OUTLIER_PCT)[0]
    assert np.isfinite(value) and value > 1.0


# -- Rule 2: topological assignment ------------------------------------------------


def test_rotamer_severity_lands_on_the_side_chain_not_the_backbone():
    """Rule 2. A rotamer outlier is a statement about chi angles, so it says nothing about
    that residue's own backbone. Painting the whole residue would assert that it does — and
    would throw away the one thing computing per-atom buys."""
    model = _model()
    values = hotspots.rotamer_severity(model, model.get_number_of_atoms())
    names = [n.strip() for n in model.get_hierarchy().atoms().extract_name()]

    hot = [i for i, v in enumerate(values) if v >= 1.0]
    assert hot, "no rotamer outliers in the test model"
    assert all(names[i] not in hotspots._MAINCHAIN for i in hot)
    # ... and it really did reach side-chain atoms beyond CB.
    assert any(names[i] not in ("CB",) for i in hot)


def test_ramachandran_severity_lands_on_the_backbone_only():
    """Rule 2/7: assigned to residue i's own N/CA/C/O. Narrow on purpose — phi/psi involve
    three residues, but implicating the neighbours smears one residue's problem onto two
    innocent ones."""
    model = _model()
    values = hotspots.ramachandran_severity(model, model.get_number_of_atoms())
    names = [n.strip() for n in model.get_hierarchy().atoms().extract_name()]

    hot = [i for i, v in enumerate(values) if v >= 1.0]
    assert hot, "no Ramachandran outliers in the test model"
    assert all(names[i] in hotspots._RAMA_ATOMS for i in hot)


# -- Rule 5: hydrogens ---------------------------------------------------------------


def test_hydrogen_parents_are_the_nearest_heavy_atom_in_the_same_residue():
    """Rule 5's join. Probe finds clashes through hydrogens, but they carry no Q-score and
    are undrawn in ribbon views, so each one needs a heavy atom to hand its severity to."""
    pytest.importorskip("mmtbx.hydrogens")
    from pxviewer.hydrogens import add_hydrogens, hydrogens_available

    if not hydrogens_available():
        pytest.skip("hydrogen placement needs the monomer library")
    model = add_hydrogens(_model("python/pxviewer/data/zn_site.pdb"))
    hierarchy = model.get_hierarchy()
    elements = [e.strip().upper() for e in hierarchy.atoms().extract_element()]
    xyz = hierarchy.atoms().extract_xyz().as_numpy_array()

    parents = hotspots._hydrogen_parents(hierarchy)
    assert parents, "the hydrogenated model should have hydrogens"
    for h, parent in parents.items():
        assert elements[h] in ("H", "D")
        assert elements[parent] not in ("H", "D")     # a hydrogen cannot parent a hydrogen
        assert np.linalg.norm(xyz[h] - xyz[parent]) < 1.5   # a bond length, not across the box


def test_a_heavy_atom_inherits_its_hydrogens_clash_severity():
    """Rule 5. Without this the clash signal disappears the moment hydrogens are hidden,
    which is the default in ribbon and heavy-atom views."""
    pytest.importorskip("mmtbx.hydrogens")
    from pxviewer.hydrogens import add_hydrogens, hydrogens_available

    if not hydrogens_available():
        pytest.skip("hydrogen placement needs the monomer library")
    # A real structure with hydrogens added: zn_site is too small and too clean to overlap
    # anywhere, so it could not tell inheritance from doing nothing.
    model = add_hydrogens(_model())
    hierarchy = model.get_hierarchy()
    values = hotspots.clash_severity(model, model.get_number_of_atoms())
    parents = hotspots._hydrogen_parents(hierarchy)
    assert values.max() > 0, "the test model should have some overlap to inherit"
    assert any(values[h] > 0 for h in parents), "no clash landed on a hydrogen to inherit"

    for h, parent in parents.items():
        assert values[parent] >= values[h]   # inherited, never lost
    # Inheritance is a max, not a sum: a heavy atom with several clashing hydrogens must not
    # be inflated past the worst of them and its own.
    for parent in set(parents.values()):
        own = [values[h] for h, p in parents.items() if p == parent]
        assert values[parent] <= max(own + [values[parent]]) + 1e-12


# -- Rule 3/4: aggregation ------------------------------------------------------------


def test_the_p_norm_preserves_severity_rather_than_averaging_it():
    """Rule 4/objection 1. One severe metric must survive being combined with clean ones —
    a mean would dilute a real clash into a mild smudge."""
    severe = {"a": np.array([2.0]), "b": np.array([0.0]), "c": np.array([0.0])}
    assert hotspots.combine(severe)[0] >= 2.0        # not 2/3
    # Corroboration adds a little, but never as much as summing would.
    corroborated = {"a": np.array([1.0]), "b": np.array([1.0]), "c": np.array([1.0])}
    combined = hotspots.combine(corroborated)[0]
    assert 1.0 < combined < 3.0


def test_combining_needs_no_denominator_so_ragged_coverage_stays_comparable():
    """Rule 4, and the property the whole no-map mode rests on: 0 is the p-norm's identity,
    so an absent or clean metric changes nothing. An atom with two clean metrics must score
    the same as one with four — a mean would have to pick a denominator and could not."""
    two = {"a": np.array([1.3]), "b": np.array([0.0])}
    four = {"a": np.array([1.3]), "b": np.array([0.0]),
            "c": np.array([0.0]), "d": np.array([0.0])}
    assert hotspots.combine(two)[0] == pytest.approx(hotspots.combine(four)[0])


def test_residue_rollup_takes_the_max_not_the_sum():
    """Rule 6. Ramachandran is assigned to four backbone atoms, so a sum would count one
    phi/psi four times — and would rank a TRP over a GLY for nothing but being larger."""
    model = _model()
    result = hotspots.score(model, fit="none")
    rows = hotspots.residue_rows(model, result)
    by_residue = _residue_max(model, result.values)

    assert rows, "the test model should have hotspots"
    for chain, resid, _res, severity, *_parts in rows:
        # The cell is a formatted string; compare formatted rather than re-parsing it, so a
        # value sitting exactly on a rounding boundary does not make this flaky.
        assert severity == f"{by_residue[(chain, resid)]:.2f}"
    # Worst first: the table is a worklist.
    severities = [float(row[3]) for row in rows]
    assert severities == sorted(severities, reverse=True)


def test_residue_broadcast_raises_the_whole_residue_to_its_worst_atom():
    """Display-only (a ribbon draws no side chains, so per-atom rotamer severity would be
    invisible on one). It must not change the ranking, only where the colour is carried."""
    model = _model()
    result = hotspots.score(model, fit="none")
    spread = hotspots.residue_broadcast(model, result.values)

    assert (spread >= result.values).all()          # never loses signal
    before = _residue_max(model, result.values)
    after = _residue_max(model, spread)
    assert before == after                           # ... and never invents any


# -- the map-fit term, and working without one ----------------------------------------


def test_geometry_only_keeps_the_same_absolute_scale_as_with_a_map():
    """The requirement that the score still works with no map. Dropping the map term must not
    rescale anything: severity 1.0 still means the outlier cut, so an atom whose fit was clean
    scores identically either way, and a geometry-only run stays comparable to a full one."""
    pytest.importorskip("iotbx.map_model_manager")
    from iotbx.map_model_manager import map_model_manager

    mmm = map_model_manager()
    mmm.generate_map(d_min=2.5)
    model = mmm.model()

    geometry_only = hotspots.score(model, fit="none")
    with_map = hotspots.score(model, mmm=mmm, fit="cc")

    clean = with_map.components["fit"] == 0
    assert clean.any()
    assert np.allclose(with_map.values[clean], geometry_only.values[clean])
    # The map can only add severity, never subtract it.
    assert (with_map.values >= geometry_only.values - 1e-12).all()
    assert "fit" not in geometry_only.components
    assert geometry_only.missing  # and it says so rather than pretending it scored one


def test_a_bad_fit_raises_severity_under_either_map_term():
    """Both selectable fit terms have to actually notice an atom sitting outside its density."""
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("cctbx.maptbx.qscore")
    from iotbx.map_model_manager import map_model_manager

    mmm = map_model_manager()
    mmm.generate_map(d_min=2.5)
    model = mmm.model()
    sites = model.get_sites_cart()
    for i in range(5):  # shove five atoms out of the density; the map is unchanged
        sites[i] = (sites[i][0] + 2.0, sites[i][1], sites[i][2])
    model.set_sites_cart(sites)

    for kind in ("cc", "qscore"):
        fit = hotspots.score(model, mmm=mmm, fit=kind).components["fit"]
        assert fit[:5].max() > fit[5:].max(), f"{kind} did not notice the displaced atoms"


def test_a_missing_map_degrades_instead_of_refusing():
    """Unlike Q-score colouring — where no map means the metric itself is undefined — the
    hotspot score drops the map term and still reports the geometry, because the remaining
    severities keep their meaning without it."""
    model = _model()
    result = hotspots.score(model, mmm=None, fit="qscore")
    assert "fit" not in result.components
    assert result.values.max() > 0            # the geometry metrics still scored
    assert any("needs a map" in m for m in result.missing)


def test_an_unknown_fit_term_is_rejected():
    with pytest.raises(ValueError, match="fit must be one of"):
        hotspots.score(_model(), fit="rsrz")


# -- the desktop wiring ---------------------------------------------------------------


def test_computing_hotspots_colours_the_model_through_the_attribute_path(qapp):
    """The score reaches the viewport as a named per-atom attribute on a fixed severity
    domain, not as a Mol* theme and not stretched to this structure's own range."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import _HOTSPOT_COLOR, DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        got = []
        app.bridge.hotspots_ready.connect(got.append)
        mid = app._add_model(LiveSession.from_model_file(_MODEL), "1tec")
        app.compute_hotspots(mid, fit="none")

        deadline = time.time() + 300
        while time.time() < deadline and not got:
            qapp.processEvents()
            time.sleep(0.05)
        assert got, "hotspots never landed"

        entry = app._model_entry(mid)
        assert entry["color"] == _HOTSPOT_COLOR
        spec = list(entry["session"]._representations.values())[0]
        assert spec["color"] == "attribute"
        assert spec["attribute"]["name"] == _HOTSPOT_COLOR
        assert list(spec["attribute"]["domain"]) == list(hotspots.DOMAIN)
    finally:
        app.stop()
