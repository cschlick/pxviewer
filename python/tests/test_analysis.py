"""Shared per-model analysis: the Validation tab and the Hotspots score reuse one run of the
mmtbx analyzers instead of each paying for its own."""

import pytest

pytest.importorskip("mmtbx")

from pxviewer import analysis  # noqa: E402

_MODEL = "python/pxviewer/data/1tec.pdb"


def _model():
    from pxviewer.cctbx_io import read_model

    return read_model(_MODEL)


def test_the_analyzers_run_once_and_are_memoized():
    """Each analyzer computes on first use and is cached, so a second consumer reuses the
    exact result object rather than recomputing."""
    a = analysis.ModelAnalysis(_model())
    assert a.ramalyze() is a.ramalyze()     # same object -> computed once
    assert a.rotalyze() is a.rotalyze()
    assert set(a._cache) == {"ramalyze", "rotalyze"}  # probe only when asked


def test_for_model_reuses_a_matching_analysis_but_not_a_foreign_one():
    """A function handed a shared analysis should use it only if it is for the same model —
    otherwise it must make its own, never serve another model's results."""
    model = _model()
    a = analysis.ModelAnalysis(model)
    assert analysis.for_model(model, a) is a          # same model -> reuse
    assert analysis.for_model(model, None) is not a   # nothing given -> fresh
    other = _model()
    assert analysis.for_model(other, a) is not a      # wrong model -> fresh, not a's cache


def test_hotspots_and_validation_share_one_analysis():
    """The whole point: score() fills the cache (Ramachandran, rotamer, probe), and a
    subsequent validation reuses those same runs — and still produces correct results."""
    from pxviewer import hotspots, validation
    from mmtbx.validation.ramalyze import ramalyze

    model = _model()
    shared = analysis.ModelAnalysis(model)

    hotspots.score(model, fit="none", analysis=shared)
    # The default fast pass probes the bare model, so ramalyze/rotalyze/probe are cached but
    # reduce2 is not paid until hydrogens are asked for.
    assert {"ramalyze", "rotalyze", "probe:bare"} <= set(shared._cache)
    assert "hydrogenated" not in shared._cache
    ramalyze_obj = shared._cache["ramalyze"]

    results = {r.key: r for r in validation.run_all(model, shared)}
    assert shared._cache["ramalyze"] is ramalyze_obj   # reused, not recomputed
    # ... and the Ramachandran validator's answer matches a standalone run, so sharing did
    # not change what it reports.
    standalone = ramalyze(pdb_hierarchy=model.get_hierarchy(), outliers_only=False)
    assert len(results["ramachandran"].rows) == len(standalone.results)


def test_a_hotspot_score_leaves_the_clashes_tab_nothing_to_compute():
    """The big win: reduce2 and probe are the two expensive steps (tens of seconds), and the
    Clashes tab needs exactly what an *accurate* (hydrogen) hotspot score already ran. After
    one, its dots come straight from the cache."""
    pytest.importorskip("mmtbx.hydrogens")
    import time

    from pxviewer import hotspots
    from pxviewer.hydrogens import hydrogens_available

    if not hydrogens_available():
        pytest.skip("hydrogen placement needs the monomer library")
    model = _model()
    shared = analysis.ModelAnalysis(model)
    hotspots.score(model, fit="none", analysis=shared, use_hydrogens=True)

    started = time.time()
    contacts, clashes = shared.probe_dots_split()
    assert time.time() - started < 2.0, "the Clashes tab re-ran probe instead of reusing it"
    assert contacts and clashes            # and it is real data, not an empty cache hit


def test_the_fast_pass_skips_hydrogens_and_caches_separately():
    """The hotspot score can opt out of hydrogens for speed (skipping reduce2 and probing far
    fewer atoms). The two passes cache under distinct keys, so a session can use the fast one
    and later turn hydrogens on without discarding either — or the shared rama/rota runs."""
    pytest.importorskip("mmtbx.hydrogens")
    from pxviewer.hydrogens import hydrogens_available

    if not hydrogens_available():
        pytest.skip("hydrogen placement needs the monomer library")
    model = _model()
    a = analysis.ModelAnalysis(model)

    a.probe_dots(use_hydrogens=False)
    assert "probe:bare" in a._cache and "probe:h" not in a._cache   # nothing hydrogenated yet
    assert "hydrogenated" not in a._cache                            # reduce2 never ran

    a.probe_dots(use_hydrogens=True)
    assert {"probe:bare", "probe:h", "hydrogenated"} <= set(a._cache)  # both kept side by side
    assert a.probe_model(False) is model                              # fast pass = the bare model
    assert a.probe_model(True) is not model                           # accurate = the H-added one


def test_clashes_are_scored_with_hydrogens():
    """Clashes are overwhelmingly hydrogen-mediated, so probe must run on the hydrogenated
    model — scoring the bare model finds only a fraction of what MolProbity reports."""
    pytest.importorskip("mmtbx.hydrogens")
    from pxviewer.hydrogens import hydrogens_available

    if not hydrogens_available():
        pytest.skip("hydrogen placement needs the monomer library")
    model = _model()
    shared = analysis.ModelAnalysis(model)
    assert shared.probe_model(True) is not model          # accurate pass = the H-added copy
    assert shared.probe_model(True).get_number_of_atoms() > model.get_number_of_atoms()

    from pxviewer import hotspots

    values = hotspots.clash_severity(
        model, model.get_number_of_atoms(), analysis=shared, use_hydrogens=True)
    assert values.shape == (model.get_number_of_atoms(),)  # mapped back onto the scored model
    # Far more than the ~50 a hydrogen-less run finds on this structure.
    assert int((values >= 1.0).sum()) > 200


def test_validators_that_ignore_the_analysis_still_run():
    """Only the shared validators take an analysis; the rest keep their plain run(model). All
    must still run when an analysis is passed."""
    from pxviewer import validation

    model = _model()
    keys = {r.key for r in validation.run_all(model, analysis.ModelAnalysis(model))}
    assert {"ramachandran", "rotamers", "cablam", "cbetadev", "omegalyze", "rama_z"} <= keys


def test_moving_the_atoms_drops_the_shared_caches(qapp):
    """The analysis is bound to one geometry. When the model moves, the desktop must drop it
    (and the validation and hotspot caches with it) or a stale fit would be reused."""
    pytest.importorskip("PySide6")
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    try:
        entry = {"session": type("S", (), {"model": _model()})()}
        entry["analysis"] = app._model_analysis(entry)
        entry["validation"] = {"ramachandran": object()}
        entry["hotspots"] = object()
        assert entry["analysis"] is not None

        app._invalidate_model_state(entry)
        assert "analysis" not in entry
        assert "validation" not in entry
        assert "hotspots" not in entry
    finally:
        app.stop()


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
