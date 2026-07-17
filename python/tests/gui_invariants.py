"""Invariants that must hold after any GUI action — the bank a fuzzer asserts.

The value of driving the app randomly is entirely in what you check afterwards: almost
every GUI bug found by hand this cycle kept the app running and just left it *wrong*
(stray windows, a control session pointed at a socket nobody was listening on, focus on
an object that had been removed). Each check here is one of those failure modes turned
into an assertion, so a random walk that reaches the state trips it.

``assert_viewer_consistent(app)`` runs the whole bank. It is written to work headless,
against a ``DesktopApp`` whose webapp has been started.
"""

from __future__ import annotations

from typing import Any, List


def assert_viewer_consistent(app: Any, *, check_windows: bool = True) -> None:
    """Assert every invariant. Raises AssertionError naming the first that fails."""
    _ids_are_unique(app)
    _groups_are_consistent(app)
    _focus_is_valid(app)
    _active_model_is_valid(app)
    _control_session_is_reachable(app)
    _derived_queries_point_at_live_objects(app)
    _summary_round_trips(app)
    if check_windows:
        assert_no_stray_windows(app)


def _entries(app):
    return list(app._models), list(app._volumes), list(app._reflections)


def _ids_are_unique(app) -> None:
    models, volumes, reflections = _entries(app)
    ids = [e["id"] for e in models + volumes + reflections]
    assert len(ids) == len(set(ids)), f"duplicate object ids: {ids}"


def _groups_are_consistent(app) -> None:
    models, volumes, reflections = _entries(app)
    members: dict = {}
    for e in models + volumes + reflections:
        gid = e.get("group")
        if gid is not None:
            members.setdefault(gid, []).append(e["id"])
            assert gid in app._groups, f"object {e['id']} in unknown group {gid}"
    # No orphan groups: an empty group should have been pruned.
    for gid in app._groups:
        assert members.get(gid), f"group {gid} has no members but was not pruned"
        # If cctbx paired the group, its map is offered for masking/minimize only with a
        # model present — the pairing is meaningless otherwise.
        mmm = app.group_mmm(gid)
        if mmm is not None:
            assert mmm.model() is not None, f"group {gid} has a manager but no model"


def _focus_is_valid(app) -> None:
    kind, ident = app._controls._focused
    if ident is None:
        return
    entry = {"model": app._model_entry, "volume": app._volume_entry,
             "reflections": app._reflection_entry}.get(kind, lambda _i: None)(ident)
    assert entry is not None, f"focused {kind} {ident} no longer exists"


def _active_model_is_valid(app) -> None:
    if app._active_model_id is not None:
        assert app._model_entry(app._active_model_id) is not None, \
            f"active model {app._active_model_id} no longer exists"
    # A focused *model* must be the active one (focusing a model activates it).
    kind, ident = app._controls._focused
    if kind == "model" and ident is not None:
        assert app._active_model_id == ident, \
            f"model {ident} is focused but active model is {app._active_model_id}"


def _control_session_is_reachable(app) -> None:
    """Volume commands ride the control session, and the page only connects to the
    visible models' sockets (or the dummy). A control session the page is not listening
    on means every volume control silently does nothing — the hidden-model bug."""
    control = app._control_session()
    if control is None:
        return
    if control is app._dummy:
        return
    reachable = {m["session"] for m in app._models if m["visible"]}
    assert control in reachable, \
        "control session is not a socket the viewport is connected to"


def _derived_queries_point_at_live_objects(app) -> None:
    """The helpers the UI asks about state must never name a removed object."""
    for m in app._models:
        app.map_for_model(m["id"])  # must not raise / must resolve
    for v in app._volumes:
        app.can_mask_volume(v["id"])
    phasable = app.models_for_phasing()
    live_ids = {m["id"] for m in app._models}
    assert all(m["id"] in live_ids for m in phasable), "models_for_phasing named a ghost"
    pair_models, pair_volumes = app.pairable()
    assert all(m["id"] in live_ids for m in pair_models)
    vol_ids = {v["id"] for v in app._volumes}
    assert all(v["id"] in vol_ids for v in pair_volumes)


def _summary_round_trips(app) -> None:
    """Every id the Loaded tree publishes resolves to a real entry, and the tree can be
    rebuilt from it without error."""
    summary = app._loaded_summary()
    resolve = {"model": app._model_entry, "volume": app._volume_entry,
               "reflections": app._reflection_entry}
    for item in summary["items"]:
        assert resolve[item["kind"]](item["id"]) is not None, \
            f"summary lists {item['kind']} {item['id']} which does not exist"
    group_ids = {g["id"] for g in summary["groups"]}
    assert group_ids == set(app._groups), "summary groups disagree with the registry"
    # Rebuilding the tree from the summary must not raise.
    app._controls._on_loaded_changed(summary)


# -- widget-level -------------------------------------------------------------

def stray_windows(app) -> List[Any]:
    """Parentless, visible input controls — which should never be top-level windows.

    Orphaning a still-visible widget (setParent(None)) turns it into a floating window;
    this is how a rebuilt Appearance pane spawned stray combo-box windows. Input controls
    are unambiguous: unlike a QWidget container (which the app's real windows are), a
    combo/slider/line-edit/button has no business being a window of its own.
    """
    from PySide6.QtWidgets import (
        QAbstractButton, QAbstractSlider, QApplication, QComboBox, QLineEdit,
    )

    kinds = (QComboBox, QAbstractSlider, QLineEdit, QAbstractButton)
    return [w for w in QApplication.topLevelWidgets()
            if isinstance(w, kinds) and w.parent() is None and w.isVisible()]


def assert_no_stray_windows(app) -> None:
    strays = stray_windows(app)
    assert not strays, "stray top-level widgets: " + ", ".join(
        f"{type(w).__name__}" for w in strays)
