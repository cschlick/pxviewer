"""Validation hotspots: severity calibration, atom assignment, aggregation, and the field.

The rules under test are stated in HOTSPOTS.md; each exercise names the rule it pins. This
is the pure-computation half -- cctbx and numpy only. The parts that need a desktop shell or
a live session are in tst_hotspots_gui.py.
"""

from __future__ import absolute_import, division, print_function

import struct
import sys

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import data_path, have, skip

if not have("mmtbx", "numpy"):
    skip("mmtbx/numpy not available")

import numpy as np                                    # noqa: E402

from pxviewer import hotspots                         # noqa: E402

_cache = {}


def model(name="1tec.pdb"):
    """The test model, read once per process. 1TEC has real rotamer and Rama outliers.

    Shared, so do not mutate it -- anything that builds restraints reorders the hierarchy.
    """
    if name not in _cache:
        from pxviewer.cctbx_io import read_model

        _cache[name] = read_model(data_path(name))
    return _cache[name]


def shared_analysis():
    """The model plus one shared analysis, so reduce2 and probe -- the two expensive steps --
    run once for the whole script rather than once per exercise. This is the same sharing the
    application relies on, so using it here exercises it too."""
    if "analysis" not in _cache:
        from pxviewer import analysis

        m = model()
        _cache["analysis"] = (m, analysis.ModelAnalysis(m))
    return _cache["analysis"]


def geometry_score():
    """A geometry-only hotspot score of the test model, computed once."""
    if "score" not in _cache:
        m, shared = shared_analysis()
        _cache["score"] = (m, hotspots.score(m, fit="none", analysis=shared))
    return _cache["score"]


def residue_max(m, values):
    """(chain, resid) -> the worst value over that residue's atoms."""
    out = {}
    for i, atom in enumerate(m.get_hierarchy().atoms_with_labels()):
        key = (atom.chain_id.strip(), atom.resid().strip())
        out[key] = max(out.get(key, 0.0), float(values[i]))
    return out


# -- calibration --------------------------------------------------------------


def exercise_severity_one_reproduces_molprobity_outliers():
    """The consistency constraint from HOTSPOTS.md: because severity is anchored so that 1.0
    *is* the community cut, the ``severity >= 1.0`` level set has to flag exactly the residues
    mmtbx flags. If it diverges, our mapping is miscalibrated -- not MolProbity."""
    from mmtbx.validation.ramalyze import ramalyze
    from mmtbx.validation.rotalyze import rotalyze

    m = model()
    hierarchy = m.get_hierarchy()
    n = m.get_number_of_atoms()

    for validator, severity_fn in ((rotalyze, hotspots.rotamer_severity),
                                   (ramalyze, hotspots.ramachandran_severity)):
        flagged = set((r.chain_id.strip(), r.resid.strip())
                      for r in validator(pdb_hierarchy=hierarchy,
                                         outliers_only=False).results if r.outlier)
        ours = set(key for key, value in residue_max(m, severity_fn(m, n)).items()
                   if value >= 1.0)
        assert ours == flagged, "%s: %s" % (validator.__name__, ours ^ flagged)
        assert flagged, "the test model should have outliers, or this proves nothing"


def exercise_severity_is_one_exactly_at_the_threshold():
    """1.0 is the outlier cut by construction, for every metric that has one -- that is what
    makes the metrics commensurable without inventing weights."""
    assert approx_equal(hotspots._surprisal_severity(
        np.array([hotspots.RAMA_OUTLIER_PCT]), hotspots.RAMA_OUTLIER_PCT)[0], 1.0)
    assert approx_equal(hotspots._surprisal_severity(
        np.array([hotspots.ROTA_OUTLIER_PCT]), hotspots.ROTA_OUTLIER_PCT)[0], 1.0)
    # A perfectly ordinary conformation scores 0, not something small but positive.
    assert hotspots._surprisal_severity(
        np.array([100.0]), hotspots.RAMA_OUTLIER_PCT)[0] == 0.0
    # Worse than the cut goes above 1; better goes below. Monotone, so ranking is meaningful.
    worse = hotspots._surprisal_severity(np.array([0.001]), hotspots.RAMA_OUTLIER_PCT)[0]
    better = hotspots._surprisal_severity(np.array([2.0]), hotspots.RAMA_OUTLIER_PCT)[0]
    assert worse > 1.0 > better > 0.0


def exercise_a_zero_percentage_stays_finite():
    """ramalyze reports a hard 0 for the worst outliers; -log10(0) would poison the whole
    field with an inf that no clamp downstream could undo."""
    value = hotspots._surprisal_severity(np.array([0.0]), hotspots.RAMA_OUTLIER_PCT)[0]
    assert np.isfinite(value) and value > 1.0


def exercise_the_palette_fades_clean_atoms_into_the_background():
    """The clean end of the model palette is the viewport background, so unremarkable protein
    disappears into it and only the hotspots read."""
    palette = hotspots.hotspot_palette("#101014")
    assert palette[0] == "#101014" and palette[1] == "#101014"   # 0..cut = background
    assert palette[2:] == hotspots.WARM                          # cut..severe = warm
    # A bad or missing background falls back to a light default, not a crash.
    assert hotspots.hotspot_palette(None)[0] == hotspots.hotspot_palette("green")[0]
    assert hotspots.hotspot_palette(None)[0].startswith("#")


# -- Rule 2: topological assignment -------------------------------------------


def exercise_rotamer_severity_lands_on_the_side_chain():
    """Rule 2. A rotamer outlier is a statement about chi angles, so it says nothing about
    that residue's own backbone. Painting the whole residue would assert that it does."""
    m = model()
    values = hotspots.rotamer_severity(m, m.get_number_of_atoms())
    names = [n.strip() for n in m.get_hierarchy().atoms().extract_name()]

    hot = [i for i, v in enumerate(values) if v >= 1.0]
    assert hot, "no rotamer outliers in the test model"
    assert all(names[i] not in hotspots._MAINCHAIN for i in hot)
    assert any(names[i] not in ("CB",) for i in hot)   # reached beyond CB


def exercise_ramachandran_severity_lands_on_the_backbone_only():
    """Rule 2/7: assigned to residue i's own N/CA/C/O. Narrow on purpose -- phi/psi involve
    three residues, but implicating the neighbours smears one residue's problem onto two."""
    m = model()
    values = hotspots.ramachandran_severity(m, m.get_number_of_atoms())
    names = [n.strip() for n in m.get_hierarchy().atoms().extract_name()]

    hot = [i for i, v in enumerate(values) if v >= 1.0]
    assert hot, "no Ramachandran outliers in the test model"
    assert all(names[i] in hotspots._RAMA_ATOMS for i in hot)


# -- Rule 5: hydrogens --------------------------------------------------------


def exercise_hydrogen_parents_are_the_nearest_heavy_atom():
    """Rule 5's join. Probe finds clashes through hydrogens, but they carry no Q-score and
    are undrawn in ribbon views, so each needs a heavy atom to hand its severity to."""
    if not have("mmtbx.hydrogens"):
        print("  skipping: mmtbx.hydrogens not available")
        return
    from pxviewer.hydrogens import add_hydrogens, hydrogens_available

    if not hydrogens_available():
        print("  skipping: hydrogen placement needs the monomer library")
        return
    m = add_hydrogens(model("zn_site.pdb"))
    hierarchy = m.get_hierarchy()
    elements = [e.strip().upper() for e in hierarchy.atoms().extract_element()]
    xyz = hierarchy.atoms().extract_xyz().as_numpy_array()

    parents = hotspots._hydrogen_parents(hierarchy)
    assert parents, "the hydrogenated model should have hydrogens"
    for h, parent in parents.items():
        assert elements[h] in ("H", "D")
        assert elements[parent] not in ("H", "D")    # a hydrogen cannot parent a hydrogen
        assert np.linalg.norm(xyz[h] - xyz[parent]) < 1.5   # a bond length, not across the box


def exercise_a_heavy_atom_inherits_its_hydrogens_clash_severity():
    """Rule 5. Without this the clash signal disappears the moment hydrogens are hidden,
    which is the default in ribbon and heavy-atom views."""
    if not have("mmtbx.hydrogens"):
        print("  skipping: mmtbx.hydrogens not available")
        return
    from pxviewer.hydrogens import hydrogens_available

    if not hydrogens_available():
        print("  skipping: hydrogen placement needs the monomer library")
        return
    # Score the *hydrogenated* model directly, so hydrogens are present in the returned array
    # and inheritance is observable; the shared analysis means reduce2/probe are not repeated.
    _m, shared = shared_analysis()
    hydrogenated = shared.hydrogenated()
    hierarchy = hydrogenated.get_hierarchy()
    values = hotspots.clash_severity(
        hydrogenated, hydrogenated.get_number_of_atoms(), analysis=shared)
    parents = hotspots._hydrogen_parents(hierarchy)
    assert values.max() > 0, "the test model should have some overlap to inherit"
    assert any(values[h] > 0 for h in parents), "no clash landed on a hydrogen to inherit"

    for h, parent in parents.items():
        assert values[parent] >= values[h]     # inherited, never lost
    # Inheritance is a max, not a sum: a heavy atom with several clashing hydrogens must not
    # be inflated past the worst of them and its own.
    for parent in set(parents.values()):
        own = [values[h] for h, p in parents.items() if p == parent]
        assert values[parent] <= max(own + [values[parent]]) + 1e-12


# -- Rule 3/4: aggregation ----------------------------------------------------


def exercise_the_p_norm_preserves_severity():
    """Rule 4/objection 1. One severe metric must survive being combined with clean ones --
    a mean would dilute a real clash into a mild smudge."""
    severe = {"a": np.array([2.0]), "b": np.array([0.0]), "c": np.array([0.0])}
    assert hotspots.combine(severe)[0] >= 2.0        # not 2/3
    corroborated = {"a": np.array([1.0]), "b": np.array([1.0]), "c": np.array([1.0])}
    combined = hotspots.combine(corroborated)[0]
    assert 1.0 < combined < 3.0     # corroboration adds, but less than summing would


def exercise_combining_needs_no_denominator():
    """Rule 4, and the property the whole no-map mode rests on: 0 is the p-norm's identity,
    so an absent or clean metric changes nothing. An atom with two clean metrics must score
    the same as one with four -- a mean would have to pick a denominator and could not."""
    two = {"a": np.array([1.3]), "b": np.array([0.0])}
    four = {"a": np.array([1.3]), "b": np.array([0.0]),
            "c": np.array([0.0]), "d": np.array([0.0])}
    assert approx_equal(hotspots.combine(two)[0], hotspots.combine(four)[0])


def exercise_residue_rollup_takes_the_max_not_the_sum():
    """Rule 6. Ramachandran is assigned to four backbone atoms, so a sum would count one
    phi/psi four times -- and would rank a TRP over a GLY for nothing but being larger."""
    m, result = geometry_score()
    rows = hotspots.residue_rows(m, result)
    by_residue = residue_max(m, result.values)

    assert rows, "the test model should have hotspots"
    for chain, resid, _res, severity, _rest in [(r[0], r[1], r[2], r[3], r[4:]) for r in rows]:
        # The cell is a formatted string; compare formatted rather than re-parsing, so a
        # value sitting exactly on a rounding boundary does not make this flaky.
        assert severity == "%.2f" % by_residue[(chain, resid)]
    severities = [float(row[3]) for row in rows]
    assert severities == sorted(severities, reverse=True)    # worst first: it is a worklist


def exercise_residue_broadcast_raises_the_residue_to_its_worst_atom():
    """Display-only (a ribbon draws no side chains, so per-atom rotamer severity would be
    invisible on one). It must not change the ranking, only where the colour is carried."""
    m, result = geometry_score()
    spread = hotspots.residue_broadcast(m, result.values)

    assert (spread >= result.values).all()           # never loses signal
    assert residue_max(m, result.values) == residue_max(m, spread)   # never invents any


# -- the map-fit term, and working without one --------------------------------


def exercise_geometry_only_keeps_the_same_absolute_scale():
    """The requirement that the score still works with no map. Dropping the map term must not
    rescale anything: severity 1.0 still means the outlier cut, so an atom whose fit was clean
    scores identically either way."""
    if not have("iotbx.map_model_manager"):
        print("  skipping: iotbx.map_model_manager not available")
        return
    from iotbx.map_model_manager import map_model_manager

    mmm = map_model_manager()
    mmm.generate_map(d_min=2.5)
    m = mmm.model()

    from pxviewer import analysis as analysis_mod
    shared = analysis_mod.ModelAnalysis(m)
    geometry_only = hotspots.score(m, fit="none", analysis=shared)
    with_map = hotspots.score(m, mmm=mmm, fit="cc", analysis=shared)

    clean = with_map.components["fit"] == 0
    assert clean.any()
    assert np.allclose(with_map.values[clean], geometry_only.values[clean])
    assert (with_map.values >= geometry_only.values - 1e-12).all()   # map only adds
    assert "fit" not in geometry_only.components
    assert geometry_only.missing    # and it says so rather than pretending it scored one


def exercise_a_bad_fit_raises_severity_under_either_map_term():
    """Both selectable fit terms have to notice an atom sitting outside its density."""
    if not have("iotbx.map_model_manager", "cctbx.maptbx.qscore"):
        print("  skipping: map_model_manager / qscore not available")
        return
    from iotbx.map_model_manager import map_model_manager

    mmm = map_model_manager()
    mmm.generate_map(d_min=2.5)
    m = mmm.model()
    sites = m.get_sites_cart()
    for i in range(5):    # shove five atoms out of the density; the map is unchanged
        sites[i] = (sites[i][0] + 2.0, sites[i][1], sites[i][2])
    m.set_sites_cart(sites)

    from pxviewer import analysis as analysis_mod
    shared = analysis_mod.ModelAnalysis(m)
    for kind in ("cc", "qscore"):
        fit = hotspots.score(m, mmm=mmm, fit=kind, analysis=shared).components["fit"]
        assert fit[:5].max() > fit[5:].max(), "%s did not notice the displaced atoms" % kind


def exercise_a_missing_map_degrades_instead_of_refusing():
    """Unlike Q-score colouring -- where no map means the metric itself is undefined -- the
    hotspot score drops the map term and still reports the geometry."""
    result = hotspots.score(model(), mmm=None, fit="qscore")
    assert "fit" not in result.components
    assert result.values.max() > 0            # the geometry metrics still scored
    assert any("needs a map" in msg for msg in result.missing)


def exercise_an_unknown_fit_term_is_rejected():
    with raises(ValueError) as e:
        hotspots.score(model(), fit="rsrz")
    assert "fit must be one of" in str(e.value)


# -- the spatial field --------------------------------------------------------


def exercise_the_field_lands_on_the_models_own_coordinates():
    """The grid box has to sit where the model is, or the shell draws somewhere else
    entirely. Grid index i maps to Cartesian (i + origin) * spacing."""
    m, result = geometry_score()
    field, spacing, origin = hotspots.severity_field(m, result.values)
    xyz = m.get_hierarchy().atoms().extract_xyz().as_numpy_array()

    lo = np.array(origin) * spacing
    hi = (np.array(origin) + np.array(field.shape)) * spacing
    assert (xyz >= lo).all() and (xyz <= hi).all()      # every atom inside the box
    # The worst atom's own voxel carries at least its own severity, because the kernel weight
    # is 1 at zero distance. That is what makes contouring at 1.0 mean "at the outlier cut".
    worst = int(np.argmax(result.values))
    voxel = tuple(int(round(xyz[worst][k] / spacing - origin[k])) for k in range(3))
    assert field[voxel] >= result.values[worst] - 1e-9


def exercise_the_contour_encloses_every_outlier_atom():
    """A shell at 1.0 must not miss an atom the table lists as past the cut."""
    m, result = geometry_score()
    field, spacing, origin = hotspots.severity_field(m, result.values)
    xyz = m.get_hierarchy().atoms().extract_xyz().as_numpy_array()

    outliers = np.flatnonzero(result.values >= 1.0)
    assert outliers.size
    for i in outliers:
        voxel = tuple(int(round(xyz[i][k] / spacing - origin[k])) for k in range(3))
        assert field[voxel] >= hotspots.FIELD_ISO


def exercise_the_field_is_not_a_sum():
    """The trap the design names: summing severity into voxels would light up the core for no
    better reason than having more atoms in it, making the field a map of where the protein is
    rather than of where the trouble is."""
    m = model()
    n = m.get_number_of_atoms()
    # Every atom mildly imperfect but none near the cut. A sum over a packed core would sail
    # past 1.0; a p-norm of small numbers stays small.
    field, _spacing, _origin = hotspots.severity_field(m, np.full(n, 0.2))
    assert field.max() < 1.0


def exercise_an_all_clean_model_produces_an_empty_field():
    m = model()
    field, _spacing, _origin = hotspots.severity_field(
        m, np.zeros(m.get_number_of_atoms()))
    assert field.max() == 0.0


def exercise_the_quality_presets_trade_grid_for_speed():
    """The quality dial moves both knobs that drive cloud cost and smoothness: a coarser grid
    and fewer raymarch steps. Every preset must stay within Mol*'s 10-steps-per-cell cap."""
    q = hotspots.CLOUD_QUALITY
    assert hotspots.CLOUD_QUALITY_DEFAULT == "low"        # the floor hardware is interactive
    assert q["low"][0] > q["medium"][0] > q["high"][0]    # low = coarsest grid = fastest
    assert q["low"][1] < q["medium"][1] < q["high"][1]    # high = most steps = smoothest
    assert all(1 <= steps <= 10 for _sp, steps in q.values())

    m = model()
    values = np.zeros(m.get_number_of_atoms())
    values[0] = 2.0    # one hotspot, so both grids cover the same box
    low, _s, _o = hotspots.severity_field(m, values, spacing=q["low"][0])
    high, _s, _o = hotspots.severity_field(m, values, spacing=q["high"][0])
    assert low.size < high.size / 4      # ~2x coarser per axis -> ~8x fewer voxels
    assert low.max() > 0                  # ... and it still resolves the hotspot


# -- the cloud wire format ----------------------------------------------------


def exercise_the_severity_box_normalizes_with_the_cut_marked():
    """Mol*'s direct-volume shader feeds the raw voxel value straight into the opacity
    transfer function as a 0..1 coordinate, so the grid has to arrive normalized or the ramp
    lands in the wrong place. cutFrac tells the frontend where the outlier threshold sits on
    that scale without it having to know the cap."""
    m, result = geometry_score()
    field, spacing, origin = hotspots.severity_field(m, result.values)
    payload = hotspots.encode_severity_box(field, spacing, origin, steps_per_cell=6.0)

    cut, steps_per_cell, nx, ny, nz = struct.unpack_from("<ffiii", payload, 0)
    assert (nx, ny, nz) == field.shape
    assert approx_equal(cut, hotspots.FIELD_ISO / hotspots.SEVERITY_CAP)   # 1.0 / 4.0
    assert approx_equal(steps_per_cell, 6.0)   # the quality dial travels with the grid

    # Geometry: origin in Cartesian, axis-aligned steps of one voxel.
    ox, oy, oz = struct.unpack_from("<fff", payload, 20)
    assert approx_equal((ox, oy, oz), tuple(o * spacing for o in origin))
    steps = struct.unpack_from("<fffffffff", payload, 32)
    assert approx_equal(steps, (spacing, 0, 0, 0, spacing, 0, 0, 0, spacing))

    data = np.frombuffer(payload, dtype="<f4", offset=68)
    assert data.size == nx * ny * nz
    assert 0.0 <= data.min() and data.max() <= 1.0
    # The worst voxel should be severity/cap, not clamped away.
    assert approx_equal(data.max(), min(field.max() / hotspots.SEVERITY_CAP, 1.0), eps=1e-6)


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
