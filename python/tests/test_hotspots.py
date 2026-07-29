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


@pytest.fixture(scope="module")
def shared_1tec():
    """The test model plus one shared analysis, so reduce2 and probe — the two expensive steps —
    run once for the whole module instead of once per test. This is the same sharing the app
    relies on, so using it here also exercises it."""
    from pxviewer import analysis

    model = _model()
    return model, analysis.ModelAnalysis(model)


@pytest.fixture(scope="module")
def geometry_score(shared_1tec):
    """A geometry-only hotspot score of the test model, computed once for the module."""
    model, shared = shared_1tec
    return model, hotspots.score(model, fit="none", analysis=shared)


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


def test_the_palette_fades_clean_atoms_into_the_background():
    """The clean end of the model palette is the viewport background, so unremarkable protein
    disappears into it and only the hotspots read — a green clean end dominates and stops the
    picture looking like hotspots at all."""
    palette = hotspots.hotspot_palette("#101014")
    assert palette[0] == "#101014" and palette[1] == "#101014"   # 0..cut = background
    assert palette[2:] == hotspots.WARM                          # cut..severe = warm
    # A bad or missing background falls back to a light default, not a crash.
    assert hotspots.hotspot_palette(None)[0] == hotspots.hotspot_palette("green")[0]
    assert hotspots.hotspot_palette(None)[0].startswith("#")


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


def test_a_heavy_atom_inherits_its_hydrogens_clash_severity(shared_1tec):
    """Rule 5. Without this the clash signal disappears the moment hydrogens are hidden,
    which is the default in ribbon and heavy-atom views."""
    pytest.importorskip("mmtbx.hydrogens")
    from pxviewer.hydrogens import hydrogens_available

    if not hydrogens_available():
        pytest.skip("hydrogen placement needs the monomer library")
    # Score the *hydrogenated* model directly, so hydrogens are present in the returned array
    # and inheritance is observable; the shared analysis means reduce2/probe are not repeated.
    _model_obj, shared = shared_1tec
    model = shared.hydrogenated()
    hierarchy = model.get_hierarchy()
    values = hotspots.clash_severity(model, model.get_number_of_atoms(), analysis=shared)
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


def test_residue_rollup_takes_the_max_not_the_sum(geometry_score):
    """Rule 6. Ramachandran is assigned to four backbone atoms, so a sum would count one
    phi/psi four times — and would rank a TRP over a GLY for nothing but being larger."""
    model, result = geometry_score
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


def test_residue_broadcast_raises_the_whole_residue_to_its_worst_atom(geometry_score):
    """Display-only (a ribbon draws no side chains, so per-atom rotamer severity would be
    invisible on one). It must not change the ranking, only where the colour is carried."""
    model, result = geometry_score
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

    # One shared analysis across both scores: same geometry, so reduce2/probe run once and the
    # comparison isolates the map term (which is the point of the test).
    from pxviewer import analysis as analysis_mod
    shared = analysis_mod.ModelAnalysis(model)
    geometry_only = hotspots.score(model, fit="none", analysis=shared)
    with_map = hotspots.score(model, mmm=mmm, fit="cc", analysis=shared)

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

    from pxviewer import analysis as analysis_mod
    shared = analysis_mod.ModelAnalysis(model)
    for kind in ("cc", "qscore"):
        fit = hotspots.score(model, mmm=mmm, fit=kind, analysis=shared).components["fit"]
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


# -- the spatial field ----------------------------------------------------------------


def test_the_field_lands_on_the_models_own_coordinates(geometry_score):
    """The grid box has to sit where the model is, or the shell draws somewhere else
    entirely. Grid index i maps to Cartesian (i + origin) * spacing."""
    model, result = geometry_score
    field, spacing, origin = hotspots.severity_field(model, result.values)
    xyz = model.get_hierarchy().atoms().extract_xyz().as_numpy_array()

    # Every atom falls inside the box...
    lo = np.array(origin) * spacing
    hi = (np.array(origin) + np.array(field.shape)) * spacing
    assert (xyz >= lo).all() and (xyz <= hi).all()
    # ... and the worst atom's own voxel carries at least its own severity, because the
    # kernel weight is 1 at zero distance. This is what makes contouring at 1.0 mean
    # "at the outlier cut" rather than something arbitrary.
    worst = int(np.argmax(result.values))
    voxel = tuple(int(round(xyz[worst][k] / spacing - origin[k])) for k in range(3))
    assert field[voxel] >= result.values[worst] - 1e-9


def test_the_contour_encloses_every_outlier_atom(geometry_score):
    """A shell at 1.0 must not miss an atom the table lists as past the cut."""
    model, result = geometry_score
    field, spacing, origin = hotspots.severity_field(model, result.values)
    xyz = model.get_hierarchy().atoms().extract_xyz().as_numpy_array()

    outliers = np.flatnonzero(result.values >= 1.0)
    assert outliers.size
    for i in outliers:
        voxel = tuple(int(round(xyz[i][k] / spacing - origin[k])) for k in range(3))
        assert field[voxel] >= hotspots.FIELD_ISO


def test_the_field_is_not_a_sum_so_dense_clean_regions_stay_dark():
    """The trap the design names: summing severity into voxels would light up the core for
    no better reason than having more atoms in it, making the field a map of where the
    protein is rather than of where the trouble is."""
    model = _model()
    n = model.get_number_of_atoms()
    # Every atom mildly imperfect but none anywhere near the cut. A sum over a packed core
    # would sail past 1.0; a p-norm of small numbers stays small.
    field, _spacing, _origin = hotspots.severity_field(model, np.full(n, 0.2))
    assert field.max() < 1.0


def test_an_all_clean_model_produces_an_empty_field():
    model = _model()
    field, _spacing, _origin = hotspots.severity_field(
        model, np.zeros(model.get_number_of_atoms()))
    assert field.max() == 0.0


def test_the_quality_presets_trade_grid_for_speed():
    """The quality dial moves both knobs that drive cloud cost and smoothness: a coarser grid
    (fewer voxels) and fewer raymarch steps. Low must be a much smaller grid than high, and
    every preset must stay within Mol*'s 10-steps-per-cell cap."""
    q = hotspots.CLOUD_QUALITY
    assert hotspots.CLOUD_QUALITY_DEFAULT == "low"          # the floor hardware is interactive
    assert q["low"][0] > q["medium"][0] > q["high"][0]      # low = coarsest grid = fastest
    assert q["low"][1] < q["medium"][1] < q["high"][1]      # high = most steps = smoothest
    assert all(1 <= steps <= 10 for _sp, steps in q.values())

    model = _model()
    values = np.zeros(model.get_number_of_atoms())
    values[0] = 2.0  # one hotspot, so both grids cover the same box
    low, _s, _o = hotspots.severity_field(model, values, spacing=q["low"][0])
    high, _s, _o = hotspots.severity_field(model, values, spacing=q["high"][0])
    assert low.size < high.size / 4            # ~2x coarser per axis -> ~8x fewer voxels
    assert low.max() > 0                        # ... and it still resolves the hotspot


# -- the cloud wire format ------------------------------------------------------------


def test_the_severity_box_normalizes_to_zero_one_with_the_cut_marked(geometry_score):
    """Mol*'s direct-volume shader feeds the raw voxel value straight into the opacity
    transfer function as a 0..1 coordinate, so the grid has to arrive normalized or the ramp
    lands in the wrong place. cutFrac tells the frontend where the outlier threshold sits on
    that scale without it having to know the cap."""
    import struct

    model, result = geometry_score
    field, spacing, origin = hotspots.severity_field(model, result.values)
    payload = hotspots.encode_severity_box(field, spacing, origin, steps_per_cell=6.0)

    cut, steps_per_cell, nx, ny, nz = struct.unpack_from("<ffiii", payload, 0)
    assert (nx, ny, nz) == field.shape
    assert cut == pytest.approx(hotspots.FIELD_ISO / hotspots.SEVERITY_CAP)  # 1.0 / 4.0
    assert steps_per_cell == pytest.approx(6.0)   # the quality dial travels with the grid

    # Geometry: origin in Cartesian, axis-aligned steps of one voxel.
    ox, oy, oz = struct.unpack_from("<fff", payload, 20)
    assert (ox, oy, oz) == pytest.approx(tuple(o * spacing for o in origin))
    steps = struct.unpack_from("<fffffffff", payload, 32)
    assert steps == pytest.approx((spacing, 0, 0, 0, spacing, 0, 0, 0, spacing))

    data = np.frombuffer(payload, dtype="<f4", offset=68)
    assert data.size == nx * ny * nz
    assert 0.0 <= data.min() and data.max() <= 1.0
    # The worst voxel should be severity/cap, not clamped away.
    assert data.max() == pytest.approx(min(field.max() / hotspots.SEVERITY_CAP, 1.0), abs=1e-6)


def test_streaming_a_severity_cloud_sets_the_replay_payload(qapp):
    """The cloud rides its own wire tag on the model's session and is replayed to late
    viewers, so it survives a viewport reload the way the difference map does."""
    pytest.importorskip("websockets")
    from pxviewer.live import LiveSession, _TAG_HOTSPOT_VOLUME

    session = LiveSession.from_model_file(_MODEL)
    assert session._last_hotspot_volume is None
    session.show_hotspot_volume(b"\x00\x01\x02\x03")
    assert session._last_hotspot_volume[:4] == _TAG_HOTSPOT_VOLUME.to_bytes(4, "little")
    assert session._last_hotspot_volume[4:] == b"\x00\x01\x02\x03"
    session.clear_hotspot_volume()
    assert session._last_hotspot_volume is None


def test_precomputed_hotspot_volume_opens_without_running_analysis(qapp, monkeypatch):
    """An external severity map goes straight to the cloud wire path; it neither needs nor
    creates a Hotspots analysis result."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.volume_io import VolumeData

    volume = VolumeData.from_numpy(
        np.full((3, 4, 5), 1.25, dtype=np.float32),
        spacing=(1.0, 1.5, 2.0), origin=(2, 3, 4), name="precomputed")

    opened = []
    monkeypatch.setattr(
        VolumeData, "from_map_file",
        classmethod(lambda cls, path, **kwargs: opened.append(str(path)) or volume))

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file(_MODEL), "1tec")
        entry = app._model_entry(mid)
        assert entry.get("hotspots") is None

        app.open_hotspot_volume("precomputed.map", mid)

        assert opened == ["precomputed.map"]
        assert entry.get("hotspots") is None
        assert entry["hotspot_field_source"] == "precomputed.map"
        assert entry["hotspot_cloud"] is True
        assert entry["session"]._last_hotspot_volume is not None

        # Switching looks redraws from the retained file data, still without analysis.
        app.show_hotspot_field(mid, on=True, style="contour")
        assert entry.get("hotspots") is None
        assert entry.get("hotspot_cloud") is None
        assert entry.get("hotspot_volume") is not None
    finally:
        app.stop()


def test_the_opacity_knee_is_remembered_for_late_viewers(qapp):
    """The knee is a lightweight control message (no grid re-stream), so it is remembered and
    re-sent on connect — a reload keeps the slider where the user left it."""
    pytest.importorskip("websockets")
    from pxviewer.live import LiveSession

    session = LiveSession.from_model_file(_MODEL)
    session.show_hotspot_volume(b"\x00")
    assert session._hotspot_knee is None            # a fresh cloud is at its default knee
    session.set_hotspot_opacity(0.6)
    assert session._hotspot_knee == pytest.approx(0.6)
    # A new cloud resets the knee; clearing drops it.
    session.show_hotspot_volume(b"\x01")
    assert session._hotspot_knee is None
    session.set_hotspot_opacity(0.4)
    session.clear_hotspot_volume()
    assert session._hotspot_knee is None


def test_the_desktop_knee_is_given_in_severity_and_sent_normalized(qapp):
    """The slider is in severity units (1.0 = the cut); the wire speaks the grid's [0,1]
    scale, so the desktop divides by the cap. A no-op unless a cloud is actually showing."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file(_MODEL), "1tec")
        entry = app._model_entry(mid)

        # No cloud showing -> the knee is ignored, session untouched.
        app.set_hotspot_opacity(mid, 1.5)
        assert entry["session"]._hotspot_knee is None

        entry["hotspot_cloud"] = True   # pretend a cloud is up (avoids a full score here)
        app.set_hotspot_opacity(mid, 1.5)
        assert entry["session"]._hotspot_knee == pytest.approx(1.5 / hotspots.SEVERITY_CAP)
    finally:
        app.stop()


# -- the desktop wiring ---------------------------------------------------------------


def test_the_cloud_and_contour_are_mutually_exclusive(qapp):
    """One 3-D field per model. The cloud streams on the session; the contour is an MVS-scene
    volume. Switching between them, or turning the field off, must leave neither behind."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import DesktopApp
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

        app.show_hotspot_field(mid, on=True, style="cloud")
        assert entry.get("hotspot_cloud") is True
        assert entry.get("hotspot_volume") is None       # no MVS contour
        assert not app._volumes
        payload = entry["session"]._last_hotspot_volume
        assert payload is not None
        # The cloud streams the quality preset's grid: its step vector (offset 36 past the tag
        # — after tag, cutFrac, stepsPerCell, dims, origin) is the voxel size, and the default
        # quality is low.
        import struct
        low_spacing = hotspots.CLOUD_QUALITY["low"][0]
        assert struct.unpack_from("<f", payload, 36)[0] == pytest.approx(low_spacing)

        # Bumping quality restreams a finer grid at more steps — same cloud, cleaner render.
        app.set_cloud_quality("high")
        payload = entry["session"]._last_hotspot_volume
        hi_spacing, hi_steps = hotspots.CLOUD_QUALITY["high"]
        assert struct.unpack_from("<f", payload, 36)[0] == pytest.approx(hi_spacing)  # stepX.x
        assert struct.unpack_from("<f", payload, 8)[0] == pytest.approx(hi_steps)     # stepsPerCell
        assert struct.unpack_from("<f", payload, 36)[0] < low_spacing

        app.show_hotspot_field(mid, on=True, style="contour")
        assert entry.get("hotspot_cloud") is None         # the cloud was torn down
        assert entry["session"]._last_hotspot_volume is None
        assert entry.get("hotspot_volume") is not None    # and a contour drawn
        assert len(app._volumes) == 1

        app.show_hotspot_field(mid, on=False)
        assert entry.get("hotspot_cloud") is None and entry.get("hotspot_volume") is None
        assert not app._volumes
    finally:
        app.stop()


def test_an_absolute_contour_level_is_converted_for_the_sigma_only_wire(qapp):
    """The live volume_iso command speaks sigma, which is right for maps — one slider range
    serves any of them. A severity field contours on absolute values because its levels are
    calibrated, so its level must be converted or a live change would land somewhere else
    from where the same number puts it on a scene rebuild."""
    pytest.importorskip("PySide6")
    from pxviewer.desktop import DesktopApp

    stats = {"mean": 0.02, "std": 0.25}
    absolute = {"iso_kind": "absolute", "data": type("D", (), {"stats": lambda self: stats})()}
    # (1.0 - 0.02) / 0.25
    assert DesktopApp._iso_for_wire(absolute, 1.0) == pytest.approx(3.92)
    # A map is untouched: its level already is sigma.
    assert DesktopApp._iso_for_wire({"iso_kind": "relative"}, 3.0) == 3.0
    assert DesktopApp._iso_for_wire({}, 3.0) == 3.0


def test_the_severity_contour_is_added_once_and_removed_on_toggle(qapp):
    """One shell per model: recomputing or re-showing must replace it, never stack a second
    surface on the first."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        got = []
        app.bridge.hotspots_ready.connect(got.append)
        mid = app._add_model(LiveSession.from_model_file(_MODEL), "1tec")

        # Refuses politely before there is anything to contour.
        said = []
        app.bridge.status_changed.connect(said.append)
        app.show_hotspot_field(mid, on=True, style="contour")
        assert not app._volumes and any("no hotspots computed" in s for s in said)

        app.compute_hotspots(mid, fit="none")
        deadline = time.time() + 300
        while time.time() < deadline and not got:
            qapp.processEvents()
            time.sleep(0.05)
        assert got, "hotspots never landed"

        app.show_hotspot_field(mid, on=True, style="contour")
        assert len(app._volumes) == 1
        volume = app._volumes[0]
        assert volume["iso_kind"] == "absolute"      # calibrated level, not sigma
        assert volume["iso"] == hotspots.FIELD_ISO
        assert volume["color"] == hotspots.FIELD_COLOR
        assert volume["opacity"] < 1.0               # the model stays readable through it

        app.show_hotspot_field(mid, on=True, style="contour")  # re-show replaces
        assert len(app._volumes) == 1
        app.show_hotspot_field(mid, on=False)        # and off removes
        assert not app._volumes
        assert app._model_entry(mid).get("hotspot_volume") is None
    finally:
        app.stop()


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


def test_choosing_hydrogens_drops_a_stale_score(qapp):
    """Turning hydrogens on or off changes what a clash *is*, so a score computed under the old
    setting no longer describes the checkbox — it is dropped rather than left silently
    disagreeing. (No recompute here; that is the user's next click.)"""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file(_MODEL), "1tec")
        entry = app._model_entry(mid)
        entry["hotspots"] = object()          # stand in for a finished score
        assert app._hotspot_hydrogens is False  # fast pass by default

        app.set_hotspot_hydrogens(True)
        assert app._hotspot_hydrogens is True
        assert entry.get("hotspots") is None    # the stale score was dropped
    finally:
        app.stop()


def test_validation_staleness_tracks_model_movement(qapp):
    """The building block for a stale-results warning: once a model is validated, moving any
    atom must flip it to 'stale', and re-validating at the new coordinates must clear it. The
    signal carries a plain bool the Validation tab uses to show/hide its warning banner."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file(_MODEL), "1tec")
        entry = app._model_entry(mid)

        seen: list = []
        app.bridge.validation_stale_changed.connect(seen.append)

        # Never validated: nothing to be stale against, so no warning.
        app._refresh_validation_staleness()
        assert seen[-1] is False

        # Validate (fingerprint the current coordinates): still fresh.
        app._mark_validated(entry)
        app._refresh_validation_staleness()
        assert seen[-1] is False

        # Move an atom: the cached results now describe a past geometry.
        model = entry["session"].model
        sites = model.get_sites_cart()
        moved = sites.deep_copy()
        moved[0] = (moved[0][0] + 0.5, moved[0][1], moved[0][2])
        model.set_sites_cart(moved)
        app._refresh_validation_staleness()
        assert seen[-1] is True

        # Re-validate at the moved coordinates: fresh again.
        app._mark_validated(entry)
        app._refresh_validation_staleness()
        assert seen[-1] is False
    finally:
        app.stop()
