"""One run of the mmtbx analyzers, shared by everything that needs it.

reduce2 and probe are the expensive steps -- tens of seconds on a small structure -- and
the Validation tab, the Clashes tab and the hotspot score all want the same answers from
them. ``ModelAnalysis`` runs each analyzer once and hands the same object to every
consumer; these exercises pin both halves of that: that the caching happens, and that
sharing does not change what any consumer reports.
"""

from __future__ import absolute_import, division, print_function

import sys
import time

from pxviewer.regression.tst_utils import data_path, have, skip

if not have("mmtbx", "numpy"):
    skip("mmtbx / numpy not available")

from pxviewer import analysis                    # noqa: E402

MODEL = data_path("1tec.pdb")


def model():
    """A **fresh** model each call: an analysis is bound to one geometry, and several
    exercises below turn on the caching that binding implies."""
    from pxviewer.cctbx_io import read_model

    return read_model(MODEL)


def hydrogens_ready():
    """Hydrogen placement needs the monomer library, which a bare build may not have."""
    if not have("mmtbx.hydrogens"):
        return False
    from pxviewer.hydrogens import hydrogens_available

    return hydrogens_available()


# -- caching and reuse --------------------------------------------------------


def exercise_the_analyzers_run_once_and_are_memoized():
    """A second consumer gets the same result object back, not a recomputation."""
    a = analysis.ModelAnalysis(model())

    assert a.ramalyze() is a.ramalyze()
    assert a.rotalyze() is a.rotalyze()
    assert set(a._cache) == {"ramalyze", "rotalyze"}   # probe runs only when asked for


def exercise_for_model_reuses_a_matching_analysis_but_not_a_foreign_one():
    """Handed a shared analysis, a function may use it only if it is for the same model.

    Serving another model's cached results would be silent and wrong -- the numbers would
    look plausible and describe a different structure.
    """
    m = model()
    a = analysis.ModelAnalysis(m)

    assert analysis.for_model(m, a) is a            # same model -> reuse
    assert analysis.for_model(m, None) is not a     # nothing given -> fresh
    assert analysis.for_model(model(), a) is not a  # a different model -> fresh


# -- what the sharing is for --------------------------------------------------


def exercise_hotspots_and_validation_share_one_analysis():
    """score() fills the cache and a later validation reuses it -- and still reports the
    same Ramachandran result a standalone run would."""
    from mmtbx.validation.ramalyze import ramalyze

    from pxviewer import hotspots, validation

    m = model()
    shared = analysis.ModelAnalysis(m)

    hotspots.score(m, fit="none", analysis=shared)
    # The default fast pass probes the bare model: rama/rota/probe are cached, but reduce2
    # is not paid until hydrogens are actually asked for.
    assert {"ramalyze", "rotalyze", "probe:bare"} <= set(shared._cache)
    assert "hydrogenated" not in shared._cache
    ramalyze_object = shared._cache["ramalyze"]

    results = dict((r.key, r) for r in validation.run_all(m, shared))
    assert shared._cache["ramalyze"] is ramalyze_object      # reused, not recomputed

    standalone = ramalyze(pdb_hierarchy=m.get_hierarchy(), outliers_only=False)
    assert len(results["ramachandran"].rows) == len(standalone.results)


def exercise_validators_that_ignore_the_analysis_still_run():
    """Only some validators take an analysis; passing one must not drop the rest."""
    from pxviewer import validation

    m = model()
    keys = set(r.key for r in validation.run_all(m, analysis.ModelAnalysis(m)))
    assert {"ramachandran", "rotamers", "cablam",
            "cbetadev", "omegalyze", "rama_z"} <= keys


# -- the two probe passes -----------------------------------------------------


def exercise_the_fast_pass_skips_hydrogens_and_caches_separately():
    """The hotspot score can opt out of hydrogens for speed, skipping reduce2 and probing
    far fewer atoms. The two passes cache under distinct keys, so a session can use the
    fast one and later turn hydrogens on without discarding either."""
    if not hydrogens_ready():
        print("    (skipped: hydrogen placement needs the monomer library)")
        return
    m = model()
    a = analysis.ModelAnalysis(m)

    a.probe_dots(use_hydrogens=False)
    assert "probe:bare" in a._cache
    assert "probe:h" not in a._cache
    assert "hydrogenated" not in a._cache          # reduce2 never ran

    a.probe_dots(use_hydrogens=True)
    assert {"probe:bare", "probe:h", "hydrogenated"} <= set(a._cache)
    assert a.probe_model(False) is m               # the fast pass probes the bare model
    assert a.probe_model(True) is not m            # the accurate one, the H-added copy


def exercise_clashes_are_scored_with_hydrogens():
    """Clashes are overwhelmingly hydrogen-mediated, so probe has to run on the
    hydrogenated model: scoring the bare one finds a fraction of what MolProbity does.

    Stated as a ratio rather than an absolute count. The absolute number moves whenever
    the clash calibration does -- the 0.40 A reporting gate and the exclusion of hydrogen
    bonds each cut it -- but the gap between the two passes is the claim, and it does not
    depend on where the severity scale is anchored. On 1TEC today: 172 atoms at or past
    the community cut with hydrogens, 10 without.
    """
    if not hydrogens_ready():
        print("    (skipped: hydrogen placement needs the monomer library)")
        return
    from pxviewer import hotspots

    m = model()
    shared = analysis.ModelAnalysis(m)
    n_atoms = m.get_number_of_atoms()

    assert shared.probe_model(True) is not m
    assert shared.probe_model(True).get_number_of_atoms() > n_atoms

    # One analysis, both passes: they cache under separate keys, so this is one reduce2
    # and two probes rather than two of each.
    with_h = hotspots.clash_severity(m, n_atoms, analysis=shared, use_hydrogens=True)
    bare = hotspots.clash_severity(m, n_atoms, analysis=shared, use_hydrogens=False)

    assert with_h.shape == (n_atoms,)              # mapped back onto the scored model
    severe_with_h = int((with_h >= 1.0).sum())     # 1.0 is the community outlier cut
    severe_bare = int((bare >= 1.0).sum())

    assert severe_with_h > 100, severe_with_h
    assert severe_with_h > 10 * severe_bare, (severe_with_h, severe_bare)


def exercise_a_hotspot_score_leaves_the_clashes_tab_nothing_to_compute():
    """The Clashes tab needs exactly what an accurate hotspot score already ran, so after
    one its dots come straight from the cache.

    Timed rather than inspected: what matters is that the tab does not stall for tens of
    seconds, and a cache key can be present while the value is still recomputed.
    """
    if not hydrogens_ready():
        print("    (skipped: hydrogen placement needs the monomer library)")
        return
    from pxviewer import hotspots

    m = model()
    shared = analysis.ModelAnalysis(m)
    hotspots.score(m, fit="none", analysis=shared, use_hydrogens=True)

    started = time.time()
    contacts, clashes = shared.probe_dots_split()
    elapsed = time.time() - started

    assert elapsed < 2.0, "the Clashes tab re-ran probe instead of reusing it (%.1fs)" % elapsed
    assert contacts and clashes                    # real data, not an empty cache hit


# -- invalidation -------------------------------------------------------------


class Stand_in_session(object):
    """Just enough of a session for the desktop's cache bookkeeping: it only reads
    ``.model`` to decide whether an analysis still applies."""

    def __init__(self, model):
        self.model = model


def exercise_moving_the_atoms_drops_the_shared_caches():
    """An analysis describes one geometry. When the model moves, the desktop must drop it
    along with the validation and hotspot results derived from it -- otherwise a stale fit
    is shown against coordinates that no longer produce it."""
    if not have("PySide6"):
        print("    (skipped: PySide6 not available)")
        return
    from pxviewer.regression.tst_utils import qt_application

    qt_application()
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    try:
        entry = {"session": Stand_in_session(model())}
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


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
