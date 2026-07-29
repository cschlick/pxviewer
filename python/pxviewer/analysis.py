"""One model, analyzed once — shared between the Validation tab and the Hotspots score.

Both features lean on the same expensive mmtbx analyzers: Ramachandran (``ramalyze``) and
rotamers (``rotalyze``) are computed by a validator *and* by the hotspot score, on the same
model. Running them twice is pure waste, and the user notices — a full validation runs again
the moment they ask for hotspots, and vice versa.

:class:`ModelAnalysis` wraps a model at a fixed geometry and memoizes those analyzers, so
whichever feature runs first pays for them and the other reuses the result. It is deliberately
*state-bound*: it caches results for one set of coordinates, and the desktop drops it whenever
the model moves (a minimization, a tug, an edit) so a stale fit can never be shown.

The two genuinely expensive steps are **reduce2** (adding hydrogens, ~10 s) and **probe2**
(~12-22 s) — an order of magnitude more than every other analyzer combined — so both live here
too, and the Clashes tab and the hotspot score share one run of each instead of paying
separately. That also fixes a correctness gap: clashes are overwhelmingly hydrogen-mediated, so
scoring them on a model without hydrogens misses most of them (52 flagged atoms vs 579 on
1TEC), and everything now goes through the same hydrogenated model MolProbity would use.
"""

from __future__ import annotations

from typing import Any, Callable


class ModelAnalysis:
    """Memoized ramalyze / rotalyze / probe for one model at one geometry.

    Create one per model and hand the *same* instance to both :func:`pxviewer.validation.run_all`
    and :func:`pxviewer.hotspots.score`; the second call reuses whatever the first computed.
    Throw it away (do not mutate) when the coordinates change.
    """

    def __init__(self, model: Any, *, data_manager: Any = None):
        self.model = model
        self._data_manager = data_manager
        self._cache: dict = {}

    def _memo(self, key: str, compute: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def ramalyze(self) -> Any:
        """The full ramalyze result (all residues, not outliers-only)."""
        from mmtbx.validation.ramalyze import ramalyze

        return self._memo(
            "ramalyze",
            lambda: ramalyze(pdb_hierarchy=self.model.get_hierarchy(), outliers_only=False))

    def rotalyze(self) -> Any:
        """The full rotalyze result (all residues, not outliers-only)."""
        from mmtbx.validation.rotalyze import rotalyze

        return self._memo(
            "rotalyze",
            lambda: rotalyze(pdb_hierarchy=self.model.get_hierarchy(), outliers_only=False))

    def hydrogenated(self) -> Any:
        """This model with explicit hydrogens added by reduce2 — memoized, because it is one of
        the two genuinely expensive steps in the whole validation stack (~10 s on a small
        protein).

        Returns the model unchanged if it already has hydrogens, or if reduce2 is unavailable
        (no monomer library) — callers get a usable model either way, just a less accurate one
        for clashes.
        """
        return self._memo("hydrogenated", self._add_hydrogens)

    def _add_hydrogens(self) -> Any:
        from .hydrogens import add_hydrogens, hydrogens_available

        elements = self.model.get_hierarchy().atoms().extract_element()
        if any(e.strip().upper() in ("H", "D") for e in elements):
            return self.model            # already hydrogenated; nothing to do
        if not hydrogens_available():
            return self.model            # no monomer library — fall back, less accurate
        try:
            return add_hydrogens(self.model)
        except Exception:  # pragma: no cover - reduce2 runtime errors
            return self.model

    def probe_model(self, use_hydrogens: bool = True) -> Any:
        """The model probe should run on — hydrogenated, or the bare one for a fast pass."""
        return self.hydrogenated() if use_hydrogens else self.model

    def probe_dots(self, use_hydrogens: bool = True) -> Any:
        """probe2 ``flat_results``, with or without hydrogens.

        **With** hydrogens is the MolProbity path and the accurate one: clashes are
        overwhelmingly hydrogen-mediated, so a bare run misses the great majority — on 1TEC it
        flags 52 atoms against 579. **Without** is a much cheaper heavy-atom-only pass (it skips
        reduce2 entirely and probes a third as many atoms), useful when responsiveness matters
        more than catching every clash.

        Both are memoized under their own key, so a session can use the fast pass and later turn
        hydrogens on without losing either result — or the shared Ramachandran/rotamer runs.
        """
        from .probe import run_probe_dots

        key = "probe:h" if use_hydrogens else "probe:bare"
        return self._memo(key, lambda: run_probe_dots(
            self.probe_model(use_hydrogens), data_manager=self._data_manager))

    def probe_dots_split(self, use_hydrogens: bool = True) -> Any:
        """The cached probe run split into ``(contacts, clashes)`` drawable dots."""
        from .probe import split_dots

        key = "probe_split:h" if use_hydrogens else "probe_split:bare"
        return self._memo(key, lambda: split_dots(self.probe_dots(use_hydrogens)))


def for_model(model: Any, analysis: Any = None, *, data_manager: Any = None) -> ModelAnalysis:
    """Return ``analysis`` if it already wraps ``model``, else a fresh :class:`ModelAnalysis`.

    Lets a function take an optional shared analysis and fall back to a private one without the
    caller having to build it — and guards against being handed one for a different model.
    """
    if isinstance(analysis, ModelAnalysis) and analysis.model is model:
        return analysis
    return ModelAnalysis(model, data_manager=data_manager)
