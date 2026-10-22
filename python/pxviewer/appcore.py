"""The pxviewer application core: state and semantics with no widgets.

This is the GUI-agnostic heart of the app — the loaded-model registry, session
management, loading and fetching, selection, representations, validation,
jobs, the webapp/websocket server, the shared python kernel, and the MCP tool
surface. Everything a figure or an agent needs lives here; the desktop GUI is
an adapter on top (``desktop.DesktopApp`` subclasses this and attaches the
Qt widgets).

Two deliberate seams keep it UI-agnostic:

- **``self._viewport``** is the Mol* page *host*, not a widget. Desktop installs
  a ``QWebEngineView`` wrapper; a headless run may install a browser-driven one
  (see ``browser.py``); the default is a no-op stub. All the core ever calls is
  ``load(url)``/``close()``.
- **``self._controls``** is the controls window. The few semantic methods that
  *notify* the GUI (``_update_appearance``, ``_coordinates_moved``...) call it
  unconditionally — headless it is a ``_NullControls`` that no-ops, so the same
  code path runs unchanged.

QtCore *is* allowed here (bridge signals, the app-thread invoker, QSettings) —
it needs no display and no platform plugin. QtWidgets/ QtWebEngine are not:
importing this module and running the core must work with a plain
``QCoreApplication`` or no QApplication at all.
"""

from __future__ import annotations

import signal
import json
import math
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .demos import DEMOS, Player
from .loader import (
    FILE_DIALOG_FILTER,
    SAMPLE_STRUCTURE,
    file_kind,
    sample_structure_path,
)
from .webapp import Webapp

# Color a model by per-atom map-model correlation (cctbx's mmtbx.maps.correlation;
# see color_model_by_mapcc). Replaced two volume presentations of the same quantity.
_CC_COLOR = "mapcc"


# How long a settled drag waits before the whole-structure maps are re-phased. Long enough
# that a burst of pulls — the normal way anyone fits — costs one recompute rather than one
# per release, short enough that a user who has stopped fitting is not left looking at
# density for atoms that have moved. Restarted by every drag (see _schedule_map_update).
_DRAG_MAP_UPDATE_MS = 1500


# Color a model by aggregated validation severity (see hotspots.py and the Hotspots tab).
_HOTSPOT_COLOR = "hotspot"


# Markers dropped in the viewport ("place a marker" → click). Drawn as spheres on a
# markup channel well clear of the probe (0, 1) and validation (10+) channels.
_MARKER_CHANNEL = 90


_MARKER_COLOR = [255, 90, 90]


_MARKER_RADIUS = 0.5


# Color a model by its per-atom fit to the map. Not a Mol* theme — the values are computed
# by cctbx and pushed as an attribute (see qscore.py and color_model_by_qscore).
_QSCORE_COLOR = "qscore"


_STRUCTURE_TYPE_ORDER = ["Protein", "Nucleic acid", "Sugar", "Ion", "Water", "Ligand / other"]


# Least time between coordinate frames pushed during a drag.
#
# This was 0.05 (20 fps), then 0.025 (40 fps) — both set to keep frames from "backing up in
# the viewer." That was the right diagnosis of the wrong layer, twice over. The viewer now
# coalesces frames (see connectLive — only the newest conformation is drawn, the rest are
# dropped), so arriving too fast cannot queue. And the perf HUD showed the gate was not a
# ceiling on waste — it *was* the ceiling on motion: at 0.025 a real drag produced only
# ~11-30 fps, and the frontend sat idle ~26 ms of every 31, drawing faster than frames
# arrived (commit ~5 ms, draws 45/s, ~0 coalesced). The drag was starved, not flooded.
#
# 8 ms is where measured production tops out — the per-frame floor is then the cctbx step
# (~5 ms) plus GIL contention with the Qt thread, not this gate: a drag produces ~46 fps on
# a 2737-atom structure and ~88 fps on ubiquitin, both at their natural ceiling, lowering it
# further changes nothing. A small structure over-produces relative to the draw rate, but
# that is exactly the case coalescing makes free, so favor not throttling the big ones.
_TUG_PUSH_INTERVAL = 0.008


#: The blue->red value ramp, one definition for every user-scaled coloring: the map's
#: color-by-local-resolution surface uses these exact stops in the frontend
#: (LOCALRES_PALETTE in live.ts), and the model's B-factor/occupancy coloring sends
#: them over the attribute wire -- so "low is cool, high is hot" reads the same on a
#: surface and on atoms, and a figure can put the two side by side on one legend.
_VALUE_PALETTE = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"]


# Inline representation dropdowns in the Loaded tree (models vs maps differ).
# The model values must be types the LiveSession API accepts (see live.py's
# _STRUCTURE_REPR_TYPES / _REPR_ALIASES) — test_model_rep_options_are_valid guards this.
_MODEL_REP_OPTIONS = [
    ("Cartoon", "cartoon"),
    ("Ball & stick", "ball-and-stick"),
    ("Spacefill", "spacefill"),
    ("Surface", "surface"),
]


# How long the post-release wind-down plays for. The minimization itself converges in a
# fraction of a second; this stretches its states over a watchable settle so a released
# fling comes visibly to rest — the clearest signal that the fragment is done, not broken.
_TUG_SETTLE_DURATION = 1.2


#: **Every** per-atom value coloring, and the single place that says so. These are not
#: Mol* themes: each is an array of numbers traveling on ``entry["attribute"]``, drawn
#: by _apply_model_rep's attribute branch through an explicit (lo, hi) domain and
#: palette -- so each one gets the same user-settable Range group the map's
#: local-resolution scale has, and ``method`` names the routine that computes it.
#:
#: The calibrated ones (Q-score, map-model CC, severity) were once kept out of this
#: registry, on the reasoning that their domains carry meaning and are not the user's
#: to move. That protected comparability by *prohibition*, and cost the thing the
#: scale is for: a correlation whose values all sit between 0.6 and 0.95 is a
#: uniformly teal molecule on a 0..1 ramp, with no way to see the variation. The
#: default domain still carries the calibration and Reset is one click away, which
#: keeps the same color meaning the same thing by default without pretending the
#: user has nothing to say about it.
_MODEL_VALUE_COLORS = {
    "bfactor": {"label": "B-factor", "unit": " Å²", "method": "color_model_by_property"},
    "occupancy": {"label": "Occupancy", "unit": "", "method": "color_model_by_property"},
    _QSCORE_COLOR: {"label": "Q-score", "unit": "", "method": "color_model_by_qscore"},
    _CC_COLOR: {"label": "Map-model CC", "unit": "", "method": "color_model_by_mapcc"},
    _HOTSPOT_COLOR: {"label": "Hotspot severity", "unit": "",
                     "method": "color_model_by_hotspots"},
}



class _NullViewport:
    """Stand-in for the Mol* page host when no browser/view is attached."""

    def load(self, url: str) -> None:
        pass

    def close(self) -> None:
        pass

    def run_javascript(self, code: str, callback=None) -> None:
        pass

    def widget(self):
        return None

    def show(self) -> None:
        pass


class _NullControls:
    """Stand-in for the controls window when no GUI is attached.

    Any method call is a no-op; ``_focused`` reads as ``None``, which matches
    no (kind, id) pair — a headless app has no focused inspector.
    """

    _focused = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


# Representations that draw individual atoms: a palette color tints their *carbons* (O/N/S
# keep their standard hues). Ribbons/surfaces have no atoms to tint, so a palette color is
# applied uniformly instead. See _apply_model_rep.
_ATOM_REPS = frozenset({"ball-and-stick", "ball_and_stick", "spacefill", "sphere"})


# cctbx classifies each residue (common_residue_names_get_class) into these named
# structure types; we fold them into a small, friendly set for the show/hide menu.
_CLASS_TO_CATEGORY = {
    "common_amino_acid": "Protein",
    "d_amino_acid": "Protein",
    "modified_amino_acid": "Protein",
    "common_rna_dna": "Nucleic acid",
    "modified_rna_dna": "Nucleic acid",
    "ccp4_mon_lib_rna_dna": "Nucleic acid",
    "common_water": "Water",
    "common_saccharide": "Sugar",
    "common_element": "Ion",
    "common_small_molecule": "Ligand / other",
    "other": "Ligand / other",
}

def _model_rep_color(rep: str) -> str:
    """A sensible default color theme for a representation type (no palette in play)."""
    return "secondary-structure" if rep == "cartoon" else "element-symbol"


def _rep_shows_atoms(rep: str) -> bool:
    return rep in _ATOM_REPS


def _structure_type_groups(session) -> dict:
    """Map each present structure type -> its atom indices, via cctbx's residue class.

    Returned in a stable display order; only types actually present are included.
    """
    from iotbx.pdb import common_residue_names_get_class as get_class

    arrays = getattr(getattr(session, "_data", None), "arrays", None)
    if arrays is None:
        return {}
    category_of: dict = {}  # resname -> category (cache; few distinct resnames)
    groups: dict = {}
    for i, rn in enumerate(arrays.resname):
        cat = category_of.get(rn)
        if cat is None:
            cat = _CLASS_TO_CATEGORY.get(get_class(rn), "Ligand / other")
            category_of[rn] = cat
        groups.setdefault(cat, []).append(i)
    return {label: groups[label] for label in _STRUCTURE_TYPE_ORDER if label in groups}


def _dummy_session():
    from .live import LiveSession
    return LiveSession.from_sites([[100.0, 0.0, 0.0]])


def _restraints_built(entry) -> bool:
    """Whether the model's geometry restraints have been processed — the state
    the objects tree shows as its 'restraints' checkbox. pdb_interpretation
    sets the manager (and the monomer mappings the selection grammar needs)
    together; the check is the manager's availability."""
    model = getattr(entry.get("session"), "model", None)
    try:
        return bool(model is not None and model.restraints_manager_available())
    except Exception:  # pragma: no cover - defensive
        return False


def _edit_summaries(scope, model=None) -> list:
    """``[{"kind", "summary"}]`` for a model's edits scope, for the tree and edits list.

    A read-only projection of cctbx's own scope. It is deliberately not the thing that
    gets applied -- the scope is (see edits.build_restraints) -- so a field this summary
    has no room for cannot go missing from the restraints.

    Given ``model``, each entry also carries ``"i_seqs"`` — the atom indices the
    selections resolve to, in selection order — which is what
    :meth:`AppCore.show_restraint_notations` takes, closing the
    list-edits -> draw-edits loop the figfuzz maker had to reconstruct by hand.
    """
    from . import edits

    if scope is None:
        return []
    out = []
    for kind, obj in edits.entries(scope):
        item = {"kind": kind, "summary": edits.summarize(kind, obj)}
        if model is not None:
            try:
                item["i_seqs"] = [
                    i for sel in edits.selections_of(kind, obj)
                    for i in model.selection(sel).iselection()]
            except Exception:  # pragma: no cover - a selection the model can't parse
                pass
        out.append(item)
    return out


def _first_line(exc: Exception) -> str:
    """The first line of an exception's message, for a one-line status report.

    FetchError deliberately carries a multi-line explanation (url, what to try); the
    status bar has room for the first line of it, and the rest is on stderr.
    """
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0].strip()


def _bipolar_map(stats: dict) -> bool:
    """Whether a loaded grid is a difference map by content — a real negative
    lobe, not an ordinary map's noise floor.

    The file does not say (map formats record no map type), so this is a guess the
    way ``reflections.is_difference_map`` is a guess: the negative lobe must be
    deep on the sigma scale *and* a substantial fraction of the positive one —
    that pair separates a bipolar map from a normal map, whose minimum is a
    shallow negative tail next to a tall positive distribution.
    """
    std = stats.get("std") or 0.0
    if std <= 0:
        return False
    low = stats["min"] - stats["mean"]
    high = stats["max"] - stats["mean"]
    return low < -2.5 * std and abs(low) > 0.3 * abs(high)


def _make_bridge():
    """A QObject that marshals background-thread events onto the Qt GUI thread.

    Selections and demo status arrive on the WebSocket/demo threads; touching
    widgets from there is not allowed, so everything crosses over as a signal.
    """
    from PySide6.QtCore import QObject, Signal

    class _Bridge(QObject):
        scene_selection_changed = Signal(object)  # {"scene": {model_id: [indices]}, "changed": model_id}
        atom_picked = Signal(str, int, bool, str)  # (model id, atom index or -1, shift held, hit repr name) — a click, any mode
        status_changed = Signal(str)
        status_warned = Signal(str)  # like status_changed, but flashed so it is noticed
        interactions_changed = Signal(bool)
        structure_changed = Signal(object)  # the active LiveSession (or None)
        loaded_changed = Signal(object)     # {groups, items} for the Loaded tree
        restraints_changed = Signal(object)  # a model's restraints were rebuilt (model id)
        run_on_main = Signal(object)        # call a thunk on the GUI thread
        analysis_ready = Signal(object)     # clash/contact analysis finished (model id)
        validation_ready = Signal(object)   # validation finished: (model id, [ValidationResult])
        validation_stale_changed = Signal(bool)  # active model moved since it was last validated
        hotspots_ready = Signal(object)     # hotspots finished: (model id, Hotspots, columns, rows)
        concern_ready = Signal(object)      # concern fields imported: (model id, summary, columns, rows)
        busy_changed = Signal(object)       # (running: bool, label: str) — drives the busy bar
        minimizing_changed = Signal(bool)   # a minimization started (True) / finished (False)
        ligand_placed = Signal()            # a ligand was built and added (clear the inputs)
        volume_iso_changed = Signal(object)  # (volume id, level) changed in the viewport
        localres_shown = Signal()           # the viewport drew a localres coloring (it is usable now)

    return _Bridge()


def _thin_gestures(items):
    """Discard whole gestures the worker could not keep up with.

    When a batch drains with SEVERAL grab-and-release gestures queued, the older ones
    are input the system failed to process in time — serving them late replays clicks
    the user has already moved past (each begin pays a restraints build, each gesture
    streams motion), which is how a burst of quick tugs turned into seconds of
    catch-up. Kept: everything before the first begin (the tail — moves/end — of the
    gesture already in progress) and everything from the LAST begin on (the newest
    gesture, complete or still open). The rest is dropped as if those clicks never
    happened. Orphaned messages this can create for a *future* batch (an end whose
    begin was dropped) are already guarded no-ops in _serve_tug.
    """
    begins = [i for i, item in enumerate(items) if item[0] == "begin"]
    if len(begins) <= 1:
        return items
    return items[:begins[0]] + items[begins[-1]:]


def _collapse_moves(items):
    """Keep only the last of each run of drag targets.

    The pointer outruns cctbx, so every target but the last of a run is somewhere it has
    already left. Only *runs* collapse: a begin or an end between two moves is a
    different thing being said, and the move before a release is where the user let go.
    """
    kept = []
    for item in items:
        if item[0] == "move" and kept and kept[-1][0] == "move":
            kept[-1] = item
        else:
            kept.append(item)
    return kept


class AppCore:
    """Semantic application state and operations, independent of any GUI.

    What a non-Qt shell needs: model/session/volume/reflection registries, all
    load/fetch/validation/selection/representation methods, jobs, the kernel,
    the webapp server, and MCP attachment — plus the ``bridge`` signal bus a
    GUI (or a test, or an agent) subscribes to for change notifications.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5173,
                 can_hide: bool = False, viewport=None,
                 mcp_port: "int | None" = None):
        # QCoreApplication, not QApplication: the core only needs the QObject
        # machinery — bridge signals and the GUI-thread invoker marshal work
        # onto the app thread through it — which needs no display. A GUI
        # creates its QApplication before calling this; headless the core makes
        # a QCoreApplication itself.
        from PySide6.QtCore import QCoreApplication

        self._app = QCoreApplication.instance()
        if self._app is None:
            self._app = QCoreApplication(["pxviewer"])
        self._headless = not hasattr(self._app, "styleHints")

        self._host = host
        self._port = port
        # The HTTP MCP endpoint agents attach to (None -> MCP_PORT or
        # PXVIEWER_MCP_PORT). The webapp port above is the viewer's own
        # statics/ws server — unrelated.
        self._mcp_port = mcp_port
        # Whether objects can be hidden at all. Hiding recomposes the scene without the
        # hidden object and reloads the page — a clean context teardown that never disposes
        # anything mid-frame. A software renderer (SwiftShader) segfaults even on that, so
        # hiding is refused there and the tree checkboxes are non-checkable. run_desktop sets
        # this from the GL backend (hardware only). See pxviewer.gpu and [[gpu-webgl]].
        self._can_hide = bool(can_hide)

        self._webapp = Webapp(host=host, port=port)

        self._session: Optional[Any] = None  # the ACTIVE model session (drives the table)
        self._session_key: Optional[str] = None
        # Loaded models: {id, name, session, visible, group}. The viewport shows the
        # visible ones (one -> switch, several -> simultaneous). ``_session`` points at
        # the active model (drives the atoms table + selection sync).
        self._models: List[dict] = []
        self._model_counter = 0
        # Default-color palettes handed to each new family (model + its maps) as it opens.
        from .palettes import PaletteCycler

        self._palettes = PaletteCycler()
        self._active_model_id: Optional[str] = None
        self._focused_residue: Optional[tuple] = None  # (chain, resid) for space-bar nav
        # Loaded volumes (a distinct category — never in the atoms table / selection):
        # {id, name, data(VolumeData), visible, group, ref, map_url, iso, color}. Shown
        # as an MVSJ scene composed alongside the model ws in the one viewport.
        self._volumes: List[dict] = []
        self._volume_counter = 0
        # Loaded reflections: {id, name, data(ReflectionData), group}. The one loaded
        # thing that cannot be drawn — density is an FFT away, and for amplitudes a
        # model away too — so these have no visibility, no representation and no scene.
        # They are kept rather than consumed into maps: recomputing density after the
        # model moves is the point, and that needs the reflections still here.
        self._reflections: List[dict] = []
        self._reflection_counter = 0
        # Groups (a map_model_manager loaded together): {group_id: name}. Membership
        # is authoritative from cctbx — we never infer it.
        self._groups: dict = {}
        self._group_counter = 0
        # Where fetched files (models, maps, half-maps, reflections) are downloaded.
        # Persisted across runs via QSettings so a chosen directory sticks; the app has no
        # other settings today, so this is the whole of it. Defaults to ~/pxviewer-data.
        from PySide6.QtCore import QSettings
        from . import fetch as _fetch

        # PXVIEWER_SETTINGS_DIR redirects persistence into a plain ini file there --
        # the test harness sets it to a throwaway directory so tests and dev probes
        # never touch the user's live preference domain. (An env override rather than
        # QSettings.setPath because macOS's native format ignores setPath, and the
        # (org, app) constructor ignores setDefaultFormat -- both verified.)
        settings_dir = os.environ.get("PXVIEWER_SETTINGS_DIR")
        if settings_dir:
            self._settings = QSettings(os.path.join(settings_dir, "pxviewer.ini"),
                                       QSettings.Format.IniFormat)
        else:
            self._settings = QSettings("pxviewer", "pxviewer")
        self._work_dir = Path(
            self._settings.value("work_dir", str(_fetch.default_work_dir())))
        try:
            saved_reps = json.loads(
                str(self._settings.value("defaults/model_representations", '["cartoon"]')))
        except (TypeError, ValueError):
            saved_reps = ["cartoon"]
        valid_reps = {value for _label, value in _MODEL_REP_OPTIONS}
        self._default_model_reps = [
            rep for rep in saved_reps if rep in valid_reps] or ["cartoon"]
        try:
            saved_show = json.loads(str(self._settings.value(
                "defaults/shown_structure_types", json.dumps(_STRUCTURE_TYPE_ORDER))))
        except (TypeError, ValueError):
            saved_show = list(_STRUCTURE_TYPE_ORDER)
        self._default_shown_types = {
            label for label in saved_show if label in _STRUCTURE_TYPE_ORDER}
        self._default_model_interactions = (
            str(self._settings.value("defaults/molstar_interactions", "false")).lower()
            in ("1", "true", "yes"))
        # Mol*'s native click-focus display is retired (the stored preference is
        # deliberately ignored): a click now runs pxviewer's own selection pipeline,
        # and the local ball-and-stick context is the Selection pane's
        # "Neighborhood in ball-and-stick" — identical for clicked and typed
        # selections. The plumbing stays for the pick-mode suppression calls.
        self._focus_surroundings = False
        self._scene_counter = 0  # cache-buster for the composed volume MVSJ
        self._dummy: Optional[Any] = None  # persistent control ws when no model is visible
        self._batching = False  # defer viewport reload / signals during a group load
        # Scene-level selection: {model_id: [atom indices]}. Each model reports its
        # own picks independently (a selection may span models — e.g. protein +
        # ligand); the union across models is the scene selection. Mutated on the
        # WebSocket threads, read on the GUI thread, so guard it.
        self._scene_selection: dict = {}
        self._scene_lock = threading.Lock()
        # Serializes restraint builds. The pre-warm below runs on its own thread while a
        # drag can start on the tug worker, and both would otherwise process the same
        # model at once. Held only around the build, which is rare and per-model.
        self._restraints_lock = threading.Lock()
        # Labels of the long operations currently running, newest last — drives the busy
        # indicator (see run_background). A list, not a flag, so overlapping operations keep
        # it up until the last one finishes.
        self._busy_labels: list = []
        self._busy_lock = threading.Lock()
        # Recent background-job failures: an agent driving over MCP cannot see
        # the _warn flash, so the last few are kept here for status() to report.
        self._job_errors: list = []
        # Restraint-notation primitives currently drawn for the selected geometry rows.
        self._restraint_prim_ids: list = []
        self._restraint_prim_session = None
        self._markers: list = []  # placed markers: {id, name, position, atom, visible}
        self._marker_counter = 0
        self._player: Optional[Player] = None
        self._demo_thread: Optional[threading.Thread] = None
        self._selection_enabled = False
        self._tug_enabled = False
        self._computed_interactions_visible = False
        self._load_counter = 0

        self._stopped = False
        self._prev_sigint = None
        # The shared Python kernel (console.AppKernel) — the one namespace the
        # Console tab, the MCP ``python`` tool and any attached Jupyter client
        # all evaluate in. Lazy: a session that never needs a console never
        # pays for a kernel. ``_gui_invoker`` marshals callables onto the Qt
        # thread so kernel-side code can drive widgets safely.
        self._kernel = None
        self._gui_invoker = self._make_gui_invoker()
        self._sigint_installed = False
        self._sigint_timer = None
        self._minimize_stop = threading.Event()  # set to halt a running minimization
        # Set while no minimization is running. A drag waits on this before it builds
        # restraints on the model, so the minimizer's thread and the drag's never write the
        # same coordinates at once — the drag takes the model over from a clean stop.
        self._minimize_idle = threading.Event()
        self._minimize_idle.set()
        # Which model the run refines, so remove_model can stop a run on a
        # departing model instead of orphaning it over a dead session.
        self._minimize_mid: Optional[str] = None
        self._volume_scroll_target: Optional[str] = None  # volume the wheel contours
        # Zoom-button presses per direction — how the tutorial's zoom step knows
        # the camera was stepped (wheel zoom lives in the page and reports nothing).
        self._zoom_out_presses = 0
        self._zoom_in_presses = 0
        # Dragging atoms: explicitly armed, one drag at a time (there is one pointer). Continuous
        # relaxation is on by default — a drag settles as a living motion, which reads better
        # than a nudge-and-stop; the checkbox in the Refine drag box mirrors this.
        self._tug_into_density = False
        try:
            self._tug_settle_seconds = max(0.0, float(
                self._settings.value("drag/settle_seconds", _TUG_SETTLE_DURATION)))
        except (TypeError, ValueError):
            self._tug_settle_seconds = _TUG_SETTLE_DURATION
        try:
            self._tug_strength = max(0.05, float(
                self._settings.value("drag/pull_strength", 1.0)))
        except (TypeError, ValueError):
            self._tug_strength = 1.0
        self._tug_diagnostics = (
            str(self._settings.value("drag/diagnostics", "false")).lower()
            in ("1", "true", "yes"))
        self._tug_misses = 0
        self._tug_begin_walltime = None  # set at grab; cleared on the first pushed frame
        self._tug_cache = None  # (signature, finished Tug) — the last zone, reusable
        self._tug_continuous = True
        # Whether a settled drag re-phases the whole-structure maps (see
        # _queue_post_drag_map_update). On, because a minimization already does it and a
        # drag moves the model exactly as much.
        self._maps_after_drag = True
        self._drag_map_timer: Any = None
        self._drag_map_rid: Optional[str] = None  # which reflections that timer will re-phase
        # What a drag lets move (see tug.Tug): a sphere of a given radius, or a stretch of
        # residues (flank each side; 0 == single residue). Default is the sphere.
        self._tug_scope = {"mode": "sphere", "radius": 8.0, "flank": 0}
        self._tug: Any = None
        self._tug_model: Optional[str] = None
        self._tug_session: Any = None
        self._tug_last: Any = None
        self._tug_last_push: float = 0.0
        self._tug_queue: Any = None  # made with its worker on the first drag
        self._diff_suppressed: list = []  # static diff maps hidden behind the live window
        self._stale_dimmed: dict = {}     # ref -> own opacity, while maps trail the model
        self._diff_agreement_gen = -1     # last drag generation the tripwire sampled
        # Live difference map while dragging (see set_live_difference_map): a warm-recompute
        # engine, cached per phased group, fed the latest drag frame off a one-slot queue so
        # only the most recent conformation is ever mapped (older frames are dropped).
        self._live_diff = False
        self._diff_engine: Any = None          # reflections.LiveDifferenceMap
        self._diff_engine_key: Optional[str] = None  # the group id it was built for
        self._diff_queue: Any = None
        self._diff_ctx: Any = None             # (group id, reflection path) for the running drag
        self._diff_atom: Optional[int] = None  # the dragged atom, the window's center
        self._diff_gen = 0                      # bumped on each drag start/clear; drops stale recomputes
        # How many live difference windows have actually reached the viewport. Unlike the
        # state above it is never reset, because it answers a different question: "has the
        # user seen one of these yet?" — which is what the X-ray tutorial waits on, and what
        # distinguishes a drag that recomputed density from one that merely happened.
        self._diff_boxes = 0
        # How much density any map draws around the view center -- None for all of it,
        # which is the default: a map is drawn as it is until asked otherwise.
        # One setting for every map, not a property of each: bounding is a statement
        # about how much density you want to look at, and having it on some maps and
        # not others left contours on screen that behaved differently for no visible
        # reason. Settings changes it; it applies to what is open and to what opens next.
        self.view_radius: Optional[float] = None

        self.bridge = _make_bridge()
        # Workers marshal GUI-thread work (e.g. adding a model) via this signal;
        # emitted from another thread it dispatches as a queued call on the GUI thread.
        self.bridge.run_on_main.connect(lambda fn: fn())
        self.bridge.localres_shown.connect(self._on_localres_shown)

        self._step_view = None  # the last table engaged; Space steps it until another does
        # The viewer host and the controls. A GUI installs real widgets here;
        # headless they are no-op stubs — the semantic call sites that notify
        # them (_update_appearance, _coordinates_moved, ...) become no-ops.
        self._viewport = viewport if viewport is not None else _NullViewport()
        self._controls = _NullControls()
        self._main = None
        self._controls_dock = None
        self._splash = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> int:
        """Start the webapp + MCP endpoint and run the app event loop."""
        self._webapp.start()
        self._start_services()
        print("Press Ctrl-C to stop.", flush=True)
        self._install_sigint_handler()
        try:
            return self._app.exec()
        except KeyboardInterrupt:  # a Ctrl-C that raced the handler being installed
            return 0
        finally:
            self._restore_sigint_handler()

    def _start_services(self) -> None:
        """Everything ``start`` needs before entering the event loop."""
        # Land on an empty viewer: the main screen is "load a file", not a demo.
        self._reload_viewport()  # nothing loaded -> a dummy-backed blank viewer
        self._status(f"Ready — serving {self._webapp.url}")
        print(f"pxviewer running at {self._webapp.url}", flush=True)

        # The MCP tool surface rides along with the app: agents attach to the
        # HTTP endpoint and drive this live session rather than a detached
        # instance. Localhost-only; opt out with PXVIEWER_MCP=0. (An agent that
        # can call save_view can write files, so this is a convenience for the
        # machine's owner, not a network API.)
        import os as _os
        if _os.environ.get("PXVIEWER_MCP", "1") not in ("0", "false", "no"):
            from . import mcp_server
            port = self._mcp_port or int(
                _os.environ.get("PXVIEWER_MCP_PORT") or mcp_server.MCP_PORT)
            mcp_server.attach(self, port=port)

    # -- hooks a GUI overrides ------------------------------------------------

    def _pump_events(self) -> None:
        """Let the GUI paint pending updates before a blocking call.

        A no-op without widgets — a headless status line needs no repaint."""

    def _focused_step_view(self):
        """The table view holding keyboard focus, or None — headless has none."""
        return None

    """Run the pxviewer desktop app with viewport and controls windows."""


    def _install_sigint_handler(self) -> None:
        """Make Ctrl-C quit the Qt event loop instead of raising out of `exec()`.

        Qt's event loop is C++: Python's SIGINT flag is only acted on once the
        interpreter regains control, which surfaces as a KeyboardInterrupt traceback
        thrown from inside `exec()`. So we (a) handle SIGINT by asking Qt to quit,
        and (b) run an idle timer purely to hand the interpreter a slice often
        enough for that handler to actually run.
        """
        from PySide6.QtCore import QTimer

        def _quit(_signum, _frame):
            print("\nstopping…", flush=True)
            self._app.quit()

        try:
            self._prev_sigint = signal.signal(signal.SIGINT, _quit)
        except ValueError:
            return  # not on the main thread; nothing to install
        self._sigint_installed = True

        self._sigint_timer = QTimer()
        self._sigint_timer.start(200)
        self._sigint_timer.timeout.connect(lambda: None)


    def _restore_sigint_handler(self) -> None:
        if self._sigint_timer is not None:
            self._sigint_timer.stop()
            self._sigint_timer = None
        if not self._sigint_installed:
            return
        self._sigint_installed = False

        # By now we are on the way out, so a further Ctrl-C has nothing left to
        # cancel — it can only land mid-teardown or inside an atexit hook and
        # surface as a spurious traceback. Swallow it. A handler the caller
        # installed themselves is theirs to keep, so hand that one back.
        previous = self._prev_sigint
        restore = previous if callable(previous) and previous is not signal.default_int_handler else signal.SIG_IGN
        try:
            signal.signal(signal.SIGINT, restore)
        except ValueError:
            pass
        self._prev_sigint = None


    def stop(self) -> None:
        """Tear down the demo, live session, and webapp. Idempotent.

        Runs on `aboutToQuit` and again from `run_desktop`'s finally, so a second
        call — or one that races a repeated Ctrl-C — must be a no-op.
        """
        if self._stopped:
            return
        self._stopped = True
        if self._drag_map_timer is not None:
            self._drag_map_timer.stop()  # no re-phase into a tree being torn down
        try:
            self._controls.shutdown_console()
        except Exception:  # pragma: no cover - defensive
            pass
        self.stop_demo()
        self._clear_all()  # stops all model sessions, volumes, and the dummy
        # The viewport goes before the server: a browser host may still have a
        # navigation in flight that would otherwise lose its socket mid-goto.
        self._viewport.close()  # release the render process (see close())
        self._webapp.stop()
        self._controls.close()  # release the controls widget tree (see close())
        self._release_shell()


    def _release_shell(self) -> None:
        """Drop the window the viewport and controls were docked into, and the splash.

        Both outlive their own teardown otherwise: the main window is held by this object,
        and the splash by the ``finished`` closure connected to the page-load signal — so
        a run that never loads a page (a test that builds an app but never calls
        :meth:`start`) keeps a splash screen alive with nothing to dismiss it.

        Guarded with ``getattr`` because ``stop`` is reachable before the shell is built:
        an exception during construction runs teardown on a half-built app.
        """
        for name in ("_splash", "_main"):
            widget = getattr(self, name, None)
            setattr(self, name, None)
            if widget is None:
                continue
            try:
                widget.close()
                widget.deleteLater()
            except Exception:  # pragma: no cover - defensive teardown
                pass


    # -- live session ----------------------------------------------------

    # -- registry (models + volumes + groups) ----------------------------

    def _model_entry(self, mid, *, missing_ok: bool = False):
        """The registry entry for a model id — loud on a bad one.

        ``None`` returns ``None`` so the ``mid or self._active_model_id``
        idiom stays short. A kind name ('model', from ``load_file``'s return)
        or an unknown id is a caller bug: it raises, never silently returns
        — the figfuzz error probe found ``set_volume_iso('volume-99')``
        reporting success on nothing. ``missing_ok=True`` is for ids that can
        legitimately be stale — pinned children, guards racing an unload."""
        if mid is None:
            return None
        if mid in ("model", "volume", "reflections", "group"):
            self._warn(f"{mid!r} is a kind, not an object id — "
                       f"status() lists the ids (e.g. 'model-1')")
            raise ValueError(f"{mid!r} is a kind, not an object id — "
                             f"status() lists the ids (e.g. 'model-1')")
        entry = next((m for m in self._models if m["id"] == mid), None)
        if entry is None and not missing_ok:
            raise ValueError(f"no model {mid!r} — "
                             f"status() lists the ids (e.g. 'model-1')")
        return entry


    def run_background(self, work, *, name: str, label: str) -> None:
        """Run ``work`` on a daemon thread with the busy indicator up for its duration.

        Every long operation goes through here so the indicator is genuinely unified: one place
        decides that something is running, and the ``finally`` means no worker can leave the
        indicator spinning by returning early or raising. ``label`` is what the user is told is
        happening ("Finding hotspots"), shown while it runs.

        A worker that raises is reported rather than lost. Without this the traceback goes
        to the interpreter's thread hook -- stderr, which a windowed app has no one reading
        -- and the only thing the user sees is the busy indicator stopping, which is
        exactly what success looks like. Long operations here are the ones most likely to
        fail for reasons outside the app (a download, a missing library, a map that will
        not read), so silence is the worst answer. The message is flashed via
        :meth:`_warn`, the traceback still goes to stderr for a developer, and ``finally``
        continues to guarantee the indicator comes down.

        Not for the silent background work (restraint warm-up) or the continuous interactive
        loops (tug), which the user did not ask for and should not see a spinner for.
        """
        def wrapped() -> None:
            try:
                work()
            except Exception as exc:
                traceback.print_exc()
                with self._busy_lock:
                    self._job_errors.append({"name": name, "label": label,
                                             "error": _first_line(exc),
                                             "when": time.time()})
                    del self._job_errors[:-8]
                self._warn(f"{label} failed: {_first_line(exc)}")
            finally:
                self._end_busy(label)

        self._begin_busy(label)
        threading.Thread(target=wrapped, name=name, daemon=True).start()


    # Both of these emit *while holding the lock*, deliberately. Emitting after releasing it
    # lets two workers finishing at once interleave — the last one to update the list can be
    # the first to emit, so a stale "still running" arrives after the "all done" and the
    # indicator spins forever with nothing behind it. Holding the lock keeps the emission order
    # identical to the state order. Safe because the slot only touches widgets and never calls
    # back in here.
    def _begin_busy(self, label: str) -> None:
        with self._busy_lock:
            self._busy_labels.append(label)
            self._emit_busy()


    def _end_busy(self, label: str) -> None:
        with self._busy_lock:
            if label in self._busy_labels:
                self._busy_labels.remove(label)
            self._emit_busy()


    def _emit_busy(self) -> None:
        """Announce the busy state. Call with ``_busy_lock`` held (see the note above).

        Carries every running label, not just the newest: the controls disable the button that
        started each operation, so they need to know exactly which are in flight rather than
        only that *something* is.
        """
        # Several may overlap (a map rephasing while hotspots run); the indicator stays up
        # until the last finishes, and names whichever is still going.
        running = bool(self._busy_labels)
        current = self._busy_labels[-1] if running else ""
        self.bridge.busy_changed.emit((running, current, tuple(self._busy_labels)))


    def job_state(self) -> dict:
        """Background work as an agent sees it: what is running and what just
        failed. A fetch that timed out or a validation that never landed is
        answered here, not guessed at from the object lists."""
        with self._busy_lock:
            return {"busy": list(self._busy_labels),
                    "errors": list(self._job_errors)}


    def _model_analysis(self, entry):
        """The per-model analysis cache (ramalyze/rotalyze/probe) shared by the Validation tab
        and the Hotspots score, so whichever runs first pays for them and the other reuses it.
        Rebuilt if the model object changed; dropped outright when the atoms move (see
        :meth:`_invalidate_model_state`), so it never serves a stale geometry."""
        from .analysis import ModelAnalysis

        model = getattr(entry["session"], "model", None)
        if model is None:
            return None
        cached = entry.get("analysis")
        if cached is None or cached.model is not model:
            cached = ModelAnalysis(model)
            entry["analysis"] = cached
        return cached


    def _invalidate_model_state(self, entry) -> None:
        """Drop the caches that describe the model's geometry. After the atoms move, the shared
        analysis, the validation results and the hotspot field all describe where the atoms
        *were* — recompute on the next request rather than show a stale fit."""
        entry["_coords_dirty"] = True   # the file no longer describes these sites
        for key in ("analysis", "validation", "hotspots"):
            entry.pop(key, None)
        self._dim_stale_diff(entry.get("group"))
        # The restraint tables' model/delta/residual columns are a snapshot of the
        # sites when the geometry was last built — stale the moment atoms move too.
        mid = entry.get("id")
        if mid is not None:
            self.bridge.run_on_main.emit(
                lambda mid=mid: self._controls._coordinates_moved(mid))


    def _dim_stale_diff(self, gid) -> None:
        """Dim the group's difference map the moment the model outruns its phasing.

        The atoms just moved, so the difference map now answers a question about the
        previous model — true red/green and stale red/green look identical, which is
        exactly how a stale map gets mistaken for a level bug. Dimmed (not hidden):
        readable as "this is old", restored to the entry's own opacity when fresh
        maps land (see update_maps). A render-side change only."""
        control = self._control_session()
        if control is None or gid is None:
            return
        for volume in self._volumes:
            if (volume.get("group") != gid or not volume.get("negative_color")
                    or volume["ref"] in self._stale_dimmed):
                continue
            self._stale_dimmed[volume["ref"]] = volume.get("opacity", 1.0)
            try:
                control.set_volume_opacity(volume["ref"], 0.35)
            except Exception:  # pragma: no cover - defensive
                pass
        if self._stale_dimmed:
            self._status("difference map is behind the model — re-phasing brings it back")


    def _undim_stale_diff(self) -> None:
        """Fresh maps landed: restore every dimmed difference map to its own opacity."""
        control = self._control_session()
        dimmed, self._stale_dimmed = self._stale_dimmed, {}
        if control is None:
            return
        for ref, opacity in dimmed.items():
            if any(v["ref"] == ref for v in self._volumes):
                try:
                    control.set_volume_opacity(ref, opacity)
                except Exception:  # pragma: no cover - defensive
                    pass


    @staticmethod
    def _sites_fingerprint(model):
        """A cheap, exact fingerprint of a model's atomic coordinates.

        Hashing the sites_cart bytes is a few microseconds for a typical model and needs
        no tolerance: any atom that moves at all changes the digest. Used only to tell
        whether the atoms have moved since the model was last validated — not for identity
        or ordering, so a plain hash is enough.
        """
        import hashlib

        try:
            sites = model.get_sites_cart().as_numpy_array()
        except Exception:  # pragma: no cover - defensive: no coordinates to fingerprint
            return None
        return hashlib.blake2b(
            np.ascontiguousarray(sites, dtype="<f8").tobytes(), digest_size=16).digest()


    def _mark_validated(self, entry) -> None:
        """Record the coordinates the just-finished validation describes, so a later move
        can be detected (see :meth:`_refresh_validation_staleness`). Call wherever the
        cached ``validation`` results are (re)written."""
        model = getattr(entry.get("session"), "model", None)
        if model is not None:
            entry["validated_fingerprint"] = self._sites_fingerprint(model)


    def _refresh_validation_staleness(self) -> None:
        """Tell the Validation tab whether the active model has moved since it was last
        validated. Compares the current coordinates against the fingerprint taken at
        validation time; cheap enough to call on every coordinate change and model switch.
        Emits ``False`` when the model was never validated, so the warning stays hidden."""
        entry = self._model_entry(self._active_model_id)
        stale = False
        if entry is not None:
            fingerprint = entry.get("validated_fingerprint")
            if fingerprint is not None:
                model = getattr(entry.get("session"), "model", None)
                if model is not None:
                    stale = self._sites_fingerprint(model) != fingerprint
        self.bridge.validation_stale_changed.emit(stale)


    def _volume_entry(self, vid, *, missing_ok: bool = False):
        if vid is None:
            return None
        if vid in ("model", "volume", "reflections", "group"):
            self._warn(f"{vid!r} is a kind, not an object id — "
                       f"status() lists the ids (e.g. 'volume-1')")
            raise ValueError(f"{vid!r} is a kind, not an object id — "
                             f"status() lists the ids (e.g. 'volume-1')")
        entry = next((v for v in self._volumes if v["id"] == vid), None)
        if entry is None and not missing_ok:
            raise ValueError(f"no volume {vid!r} — "
                             f"status() lists the ids (e.g. 'volume-1')")
        return entry


    def _reflection_entry(self, rid, *, missing_ok: bool = False):
        if rid is None:
            return None
        if rid in ("model", "volume", "reflections", "group"):
            self._warn(f"{rid!r} is a kind, not an object id — "
                       f"status() lists the ids (e.g. 'reflections-1')")
            raise ValueError(f"{rid!r} is a kind, not an object id — "
                             f"status() lists the ids (e.g. 'reflections-1')")
        entry = next((r for r in self._reflections if r["id"] == rid), None)
        if entry is None and not missing_ok:
            raise ValueError(f"no reflections {rid!r} — "
                             f"status() lists the ids (e.g. 'reflections-1')")
        return entry


    @contextmanager
    def _batch_load(self):
        """Defer viewport reload + Loaded-tree signal until a group finishes loading."""
        self._batching = True
        try:
            yield
        finally:
            self._batching = False
            self._reload_viewport()
            self._emit_loaded_changed()


    def _new_group(self, name: str, *, mmm: Any = None, label: str = "map+model group") -> str:
        """Register a group of loaded objects.

        ``mmm`` is the cctbx ``map_model_manager`` the group was built from, when there
        is one. Holding on to it is what makes the group more than a label: it is cctbx's
        record that this model and these maps belong together, in a common frame.
        """
        self._group_counter += 1
        gid = f"group-{self._group_counter}"
        self._groups[gid] = {"name": name, "mmm": mmm, "label": label}
        return gid


    def _object_ids(self) -> set:
        """Every loaded object's id — snapshot it before a load to tell what that load added."""
        return {e["id"] for e in (*self._models, *self._volumes, *self._reflections)}


    def _group_loaded_together(self, before: set, name: str,
                               label: str = "model + data") -> None:
        """Group the objects this load just added (anything new and still ungrouped).

        Files opened as a unit — a model and the reflections to phase it against — belong
        together in the panel from the moment they land, not only once cctbx has a manager
        for them. Scoped to what the load added, so objects already sitting loose from
        earlier are left alone.

        The group carries no ``mmm``, so nothing is treated as paired (see
        :meth:`_is_paired`); Make maps later fills this same group in rather than starting a
        second one.
        """
        members = [e for e in (*self._models, *self._volumes, *self._reflections)
                   if e.get("group") is None and e["id"] not in before]
        if len(members) < 2:
            return  # a lone object is not a group
        gid = self._new_group(name, label=label)
        for entry in members:
            entry["group"] = gid


    def group_mmm(self, gid: Optional[str]) -> Any:
        """The ``map_model_manager`` a group came from, or None if it did not come from one."""
        group = self._groups.get(gid) if gid else None
        return group["mmm"] if group else None


    def _is_paired(self, entry) -> bool:
        """Whether this object is already held by a cctbx ``map_model_manager``.

        Not the same question as "is it in a group". A group is how the Objects panel shows
        things that belong together — a model with its reflections and the maps made from
        them. Only *some* groups carry a manager (a cryo-EM map+model load, or a phasing);
        a model grouped with its reflections before Make maps has none, and is still free to
        be phased or paired. Keying the pairing rules off the manager rather than off group
        membership is what lets the panel group things it would otherwise have to leave loose.
        """
        return self.group_mmm(entry.get("group")) is not None


    def pairable(self) -> tuple:
        """``(models, volumes)`` that are not paired with anything yet.

        Keyed off the manager, not group membership (see :meth:`_is_paired`): an object a
        manager already speaks for cannot be re-paired without moving it out from under
        that, but merely being grouped in the panel does not stop anything.
        """
        models = [m for m in self._models
                  if not self._is_paired(m) and getattr(m["session"], "model", None) is not None]
        volumes = [v for v in self._volumes if not self._is_paired(v)]
        # Slim summaries, not the registry dicts — a full entry carries the
        # model's type-group index arrays and the volume's map object, which
        # dumped pages of repr into MCP output (figfuzz h12).
        return ([self._object_summary(m) for m in models],
                [self._object_summary(v) for v in volumes])


    @staticmethod
    def _object_summary(entry: dict) -> dict:
        """The id/name/group shape a chooser (or an MCP caller) actually needs."""
        return {"id": entry["id"], "name": entry["name"],
                "group": entry.get("group")}


    def alignable(self) -> list:
        """Existing ``(model, volume)`` pairs that can be density-aligned in place —
        returned as slim ``{"id", "name", "group"}`` summaries like :meth:`pairable`."""
        pairs = []
        for model in self._models:
            gid = model.get("group")
            mmm = self.group_mmm(gid)
            if mmm is None or mmm.model() is None:
                continue
            for volume in self._volumes:
                if volume.get("group") != gid:
                    continue
                try:
                    mm = mmm.get_map_manager_by_id(volume["data"].map_id)
                except Exception:
                    mm = None
                if mm is not None:
                    pairs.append((self._object_summary(model),
                                  self._object_summary(volume)))
        return pairs


    @staticmethod
    def _detect_map_model_shift(model: Any, map_manager: Any,
                                initial_shift: Any = None) -> tuple:
        """Find a density-supported translation on a copy, leaving ``model`` untouched.

        cctbx's ``translation_search`` changes its input sites. Running successive coarse and
        fine passes lets the equal-step grid search accumulate a general XYZ translation; the
        displacement of the copy is the one result we retain.
        """
        from mmtbx.refinement.real_space.rigid_body import translation_search

        map_data = map_manager.map_data()
        probe = model.deep_copy()
        before = probe.get_sites_cart().deep_copy()
        if initial_shift is not None and max(abs(float(v)) for v in initial_shift) > 1e-6:
            # An MRC external_origin can be meaningful but off the voxel grid, in which case
            # map_manager refuses to turn it into origin_shift_grid_units. It is still an exact
            # Cartesian starting hypothesis. Apply it to the disposable search model, then let
            # density determine only the residual.
            probe.shift_model_and_set_crystal_symmetry(
                shift_cart=tuple(float(v) for v in initial_shift),
                crystal_symmetry=model.crystal_symmetry())
        else:
            # The coarse pass samples only +/-20 A on an axis, so a map loaded in its
            # own frame (tens of Angstrom from the model) is unreachable without a
            # hypothesis. The map box center against the model's centroid is that
            # hypothesis — density then scores and refines only the residual.
            try:
                n = map_data.all()                    # (nx, ny, nz)
                ucg = map_manager.unit_cell_grid
                frac = [(o + ni / 2.0) / u for o, ni, u
                        in zip(map_data.origin(), n, ucg)]
                center = map_manager.unit_cell().orthogonalize(frac)
                guess = tuple(float(c - m) for c, m
                              in zip(center, before.mean()))
            except Exception:  # pragma: no cover - defensive
                guess = None
            if guess is not None and max(abs(v) for v in guess) > 1e-6:
                probe.shift_model_and_set_crystal_symmetry(
                    shift_cart=guess,
                    crystal_symmetry=model.crystal_symmetry())
        # A broad first pass catches the common missing boxed-map origin; two finer passes
        # settle between its 0.5 A samples. Each pass starts from the previous best position.
        for shifts in (
            [i * 0.5 for i in range(0, 41)],
            [i * 0.1 for i in range(0, 11)],
            [i * 0.02 for i in range(0, 11)],
        ):
            translation_search(model=probe, map_data=map_data, shifts=shifts)
        delta = probe.get_sites_cart() - before
        if not delta.size():
            return (0.0, 0.0, 0.0)
        mean = delta.mean()
        return tuple(float(v) for v in mean)


    def pair_model_with_map(self, mid: str, vid: str, *,
                            detect_shift: bool = False) -> str:
        """Pair a model and a map by building the cctbx manager that joins them.

        This is the explicit answer to the question :meth:`map_for_model` refuses to
        guess at. It is offered as an action rather than inferred because it *is* one:
        cctbx relocates the model, and the map, into a common *grid* frame — a boxed
        map can move a model several angstrom — and that is a change to the data, not
        a label on it. Both objects move into a group holding the manager, which is
        what makes them usable together (minimizing into density, and whatever joint
        work comes later).

        Note what pairing does NOT do: it will not slide an off-origin model into
        the density. The map and model end up in one frame but keep their existing
        offset — the Cartesian move that seats a model in its density is
        ``detect_shift=True`` here or :meth:`align_paired_model_with_map` after.
        """
        from iotbx.map_model_manager import map_model_manager

        mentry = self._model_entry(mid)
        ventry = self._volume_entry(vid)
        if mentry is None or ventry is None:
            raise ValueError("pick a model and a map to pair")
        if mentry.get("group") is not None or ventry.get("group") is not None:
            raise ValueError("those objects are already paired with something")
        model = getattr(mentry["session"], "model", None)
        if model is None:
            raise ValueError("that object has no cctbx model to pair")

        # Preserve a non-grid MRC external origin before map_model_manager calls shift_origin:
        # cctbx otherwise warns, ignores it, and then clears it. It cannot be represented as
        # integer origin_shift_grid_units, but it remains a valid Cartesian shift hypothesis.
        source_map = ventry["data"].map_manager
        external = tuple(
            getattr(source_map, "_pxviewer_external_origin",
                    getattr(source_map, "external_origin", (0, 0, 0))) or (0, 0, 0))
        ventry["external_origin_hint"] = external
        external_hint = None
        if detect_shift and any(abs(float(v)) > 1e-6 for v in external):
            if not source_map.external_origin_is_compatible_with_gridding():
                external_hint = tuple(-float(v) for v in external)
                source_map.external_origin = (0, 0, 0)

        mmm = map_model_manager(
            model=model, map_manager=source_map,
            ignore_symmetry_conflicts=True)

        detected = None
        if detect_shift:
            detected = self._detect_map_model_shift(
                mmm.model(), mmm.map_manager(),
                initial_shift=external_hint)
            if max(abs(v) for v in detected) > 1e-6:
                # This is deliberately not set_sites_cart: the exact Cartesian translation
                # may be sub-voxel and therefore cannot honestly be stored as the map's integer
                # origin_shift_grid_units. Record it on the model with cctbx's shift-aware API;
                # normal model output can undo it and recover the source coordinates.
                mmm.model().shift_model_and_set_crystal_symmetry(
                    shift_cart=detected,
                    crystal_symmetry=mmm.map_manager().crystal_symmetry())
                self._invalidate_model_state(mentry)
            mentry["detected_shift_cart"] = detected
            ventry["external_origin_hint_used"] = True

        # Reuse whichever group these two are already shown in (a model grouped with its
        # reflections, say), so pairing fills that group in rather than starting a second
        # one and stranding the rest of it.
        gid = mentry.get("group") or ventry.get("group")
        if gid is None:
            gid = self._new_group(f"{mentry['name']} + {ventry['name']}")
        self._groups[gid]["mmm"] = mmm
        mentry["group"] = gid
        ventry["group"] = gid
        # cctbx moves the model (and possibly the map) into the shared frame, so show
        # where they now are rather than where they were loaded.
        mentry["session"].push(model.get_sites_cart().as_numpy_array())
        self._write_display_map(vid, ventry["data"])
        self._reload_viewport()
        self._emit_loaded_changed()
        suffix = ""
        if detected is not None:
            suffix = " — detected shift_cart (%+.2f, %+.2f, %+.2f) Å" % detected
        self._status(f"Paired {mentry['name']} with {ventry['name']}{suffix}")
        return gid


    def align_paired_model_with_map(self, mid: str, vid: str, *,
                                    detect_shift: bool = False) -> tuple:
        """Recheck an existing pair for a missing Cartesian origin shift.

        The existing map_model_manager is retained. A preserved non-grid MRC ORIGIN is used
        once as the initial hypothesis; subsequent runs start from the current conformation
        and search only for a residual, so pressing Align twice cannot blindly accumulate it.
        """
        mentry = self._model_entry(mid)
        ventry = self._volume_entry(vid)
        if mentry is None or ventry is None:
            raise ValueError("pick a paired model and map")
        if mentry.get("group") != ventry.get("group"):
            raise ValueError("that model and map are not in the same pair")
        mmm = self.group_mmm(mentry.get("group"))
        if mmm is None or mmm.model() is None:
            raise ValueError("that group has no cctbx map_model_manager")
        try:
            mm = mmm.get_map_manager_by_id(ventry["data"].map_id)
        except Exception:
            mm = None
        if mm is None:
            raise ValueError("that map is not part of the model's cctbx manager")

        if not detect_shift:
            # Still reassert the manager's current conformation; useful after a viewport
            # rebuild, and makes unchecked Align harmless rather than applying hidden work.
            mentry["session"].push(mmm.model().get_sites_cart().as_numpy_array())
            self._status(f"{mentry['name']} + {ventry['name']}: alignment unchanged")
            return (0.0, 0.0, 0.0)

        external_hint = None
        if not ventry.get("external_origin_hint_used"):
            external = tuple(
                ventry.get("external_origin_hint",
                           getattr(mm, "_pxviewer_external_origin", (0, 0, 0)))
                or (0, 0, 0))
            if any(abs(float(v)) > 1e-6 for v in external):
                external_hint = tuple(-float(v) for v in external)

        detected = self._detect_map_model_shift(
            mmm.model(), mm, initial_shift=external_hint)
        if max(abs(v) for v in detected) > 1e-6:
            mmm.model().shift_model_and_set_crystal_symmetry(
                shift_cart=detected, crystal_symmetry=mm.crystal_symmetry())
            self._invalidate_model_state(mentry)
        ventry["external_origin_hint_used"] = True
        mentry["detected_shift_cart"] = detected
        mentry["session"].push(mmm.model().get_sites_cart().as_numpy_array())
        self._emit_loaded_changed()
        self._status(
            f"Aligned {mentry['name']} with {ventry['name']} — "
            "detected shift_cart (%+.2f, %+.2f, %+.2f) Å" % detected)
        return detected


    # -- viewport composition --

    def _visible_model_ws(self) -> List[str]:
        return [f"ws://{self._host}:{m['session'].port}" for m in self._models if m["visible"]]


    def _model_ws(self) -> List[str]:
        """Every model's socket. Hiding is a render skip in place (the model stays connected,
        replaying its own hidden state on reconnect), not a drop — so the page keeps them
        all, and a hidden model reappears the instant it is shown without a reload."""
        return [f"ws://{self._host}:{m['session'].port}" for m in self._models]


    def _ensure_dummy_ws(self) -> str:
        """A persistent 1-atom control session: carries volume commands and keeps the
        page non-blank when no model is visible. Nothing to pick, so no selection."""
        if self._dummy is None:
            self._dummy = _dummy_session()
            self._dummy.start(host=self._host, port=0)
            self._dummy.on_volume_iso(self._on_volume_iso_changed)
            self._dummy.on_localres_shown(self.bridge.localres_shown.emit)
            # So a ligand marker can be placed on a blank canvas (no model), where the
            # dummy is the session the viewport is connected to.
            self._dummy.on_marker(lambda position, atom: self._on_marker(position, atom))
            self._dummy.on_marker_move(lambda mid_, pos, final: self._on_marker_move(mid_, pos, final))
            self._dummy.on_chain_stall(lambda stall: self._status(
                f"Viewer busy applying {stall.get('label')} "
                f"({int((stall.get('ms') or 0) / 1000)}s)…"))
            # Render nothing: an empty `on` set draws no atoms, so an empty scene
            # is truly empty (the dummy only keeps the ws channel open).
            try:
                self._dummy.set_representation("ball-and-stick", on=[])
            except Exception:  # pragma: no cover - defensive
                pass
        return f"ws://{self._host}:{self._dummy.port}"


    def _control_session(self):
        """A session the viewport is connected to, for volume commands. Every model stays
        connected (hiding is a render skip, not a drop), so the active model always carries
        them; the dummy is the fallback only when there is no model at all."""
        entry = self._model_entry(self._active_model_id)
        if entry is not None:
            return entry["session"]
        return self._models[0]["session"] if self._models else self._dummy


    def _write_volume_scene(self) -> Optional[str]:
        """Write an MVSJ composing every volume; return its URL path (or None).

        Hidden volumes stay in the scene (a render skip, re-applied after the reload by
        ``_reassert_hidden_volumes`` and replayed to the reloaded client by the live
        session) so a reload never rebuilds an isosurface from empty.

        A pinned local-resolution map is left out entirely. It is a color source, never a
        surface: it exists to color the map above it, and drawing its own isosurface
        produces a giant featureless blob (a smooth field contoured at its midpoint) on
        top of the data. Keeping it out of the scene makes "never drawn" structural
        instead of a hidden-state race the reload can lose.

        Only the first visible volume is focused, and only when no model is there to center."""
        drawn = [v for v in self._volumes if not v.get("is_resolution")]
        if not drawn:
            return None
        from .volume import Volume, create_volume_view

        focus_first = not self._visible_model_ws()  # center a lone volume; don't fight a model
        first_visible = next((v for v in drawn if v["visible"]), None)
        nodes = []
        for v in drawn:
            factor = int(v.get("localres_downsample") or 1)
            if factor > 1 and (v.get("downsample_urls") or {}).get(factor):
                url = v["downsample_urls"][factor]
            elif v.get("box"):
                url = v["box_url"]
            else:
                url = v["map_url"]
            iso_kind = v.get("iso_kind", "relative")
            iso, neg_iso = v["iso"], v.get("negative_iso")
            if v.get("box") and iso_kind == "relative":
                # A boxed file's sigma is not the map's: hand Mol* the absolute
                # level the sigma value means on the full map, so the pane still
                # reads in map sigma rather than needing a boxed-σ correction.
                mean, std = self._display_map_stats(v)
                neg_iso = (float(v["iso"]) if neg_iso is None else float(neg_iso)) * std - mean
                iso, iso_kind = mean + float(iso) * std, "absolute"
            nodes.append(Volume(
                url=url, ref=v["ref"], format="map",
                # Maps contour in sigma, which is what makes one slider range serve every
                # map. A severity field is the exception: its levels are calibrated (1.0 *is*
                # the outlier cut), so it must contour on the absolute value or the level
                # would mean something different for every structure.
                isosurface_kind=iso_kind, isosurface_value=iso,
                color=v["color"], negative_color=v.get("negative_color"),
                negative_isosurface_value=neg_iso,
                negative_opacity=v.get("negative_opacity"),
                negative_style=v.get("negative_style"),
                # Every map gets a transform node (zero translation when
                # unpositioned) so set_volume_position has a node to update.
                position=v.get("position") or (0.0, 0.0, 0.0),
                opacity=v["opacity"], style=v["style"],
                focus=(focus_first and v is first_visible),
            ))
        self._scene_counter += 1
        scene_dir = self._webapp.volume_dir / "scene" / str(self._scene_counter)
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "scene.mvsj").write_text(create_volume_view(volumes=nodes))
        return f"/scene/{self._scene_counter}/scene.mvsj"


    def _reload_viewport(self) -> None:
        """Compose the models (ws) and volumes (MVSJ) into one viewport URL."""
        if self._batching:
            return
        model_ws = self._model_ws()
        mvsj = self._write_volume_scene()
        ws = list(model_ws)
        if not model_ws:
            # No model to carry volume commands / keep the page alive -> use the dummy.
            ws.append(self._ensure_dummy_ws())
        self._reassert_volume_clips()
        self._reassert_hidden_volumes()
        # The page is about to navigate: tell every session feeding it so a
        # command issued in the same breath (the orient right after a mask call)
        # queues for the new page instead of dying with the old one.
        for m in self._models:
            s = m.get("session")
            if s is not None:
                s.begin_reload()
        if self._dummy is not None:
            self._dummy.begin_reload()
        params = []
        if mvsj:
            params.append(f"mvsj={mvsj}")
        params.append("ws=" + ",".join(ws))
        self._viewport.load(f"{self._webapp.url}index.html?{'&'.join(params)}")
        # Markers ride the control session, which the reload may have changed (e.g. the
        # dummy on a blank canvas -> a model just loaded), so re-assert them onto it.
        if self._markers:
            self._draw_markers()


    def _wire_active(self, session) -> None:
        """Point the active session at ``session`` (the default table model + display target).

        Selection is scene-wide (enabled per model, not tied to the active one), so
        switching the active model just moves which model the atoms table defaults to.
        """
        self._session = session
        self._session_key = None
        self.bridge.structure_changed.emit(session)


    # -- models --

    def _type_groups(self, entry) -> dict:
        """Cached {structure-type -> atom indices} for a model (via cctbx classes)."""
        if entry.get("type_groups") is None:
            entry["type_groups"] = _structure_type_groups(entry["session"])
        return entry["type_groups"]


    def _shown_indices(self, entry) -> Optional[list]:
        """Atom indices to show given the model's hidden types and hidden atoms, or None for
        all."""
        hidden_types = entry.get("hidden_types") or set()
        drop = set(entry.get("hidden_atoms") or set())  # atoms hidden by "hide selected"
        if hidden_types:
            groups = self._type_groups(entry)
            for label in hidden_types:
                drop.update(groups.get(label, []))
        drop.update(self._other_conformer_indices(entry))
        if not drop:
            return None
        mask = np.ones(entry["session"]._n_atoms, dtype=bool)
        mask[list(drop)] = False
        return np.nonzero(mask)[0].tolist()


    def model_conformers(self, mid: str) -> list:
        """The alternate conformations present in a model, e.g. ``["A", "B"]``.

        Empty for the overwhelming majority of structures, which is the signal the
        Appearance pane uses to leave the conformer control out entirely rather than
        offering a choice of one.
        """
        entry = self._model_entry(mid)
        if entry is None:
            return []
        return self._conformers_of(entry)


    def _conformers_of(self, entry) -> list:
        """Sorted altloc labels in a model, cached on the entry (the arrays do not change)."""
        cached = entry.get("conformers")
        if cached is None:
            arrays = entry["session"]._data.arrays
            labels = {alt.strip() for alt in (getattr(arrays, "altloc", None) or []) if alt.strip()}
            cached = entry["conformers"] = sorted(labels)
        return cached


    def _other_conformer_indices(self, entry) -> set:
        """Atoms belonging to a conformer other than the chosen one.

        The atoms with *no* altloc are never dropped: they are the part of the residue
        both conformers share, so hiding them would break the backbone into fragments
        around every alternate side chain.
        """
        chosen = entry.get("conformer")
        if not chosen:
            return set()
        arrays = entry["session"]._data.arrays
        altloc = getattr(arrays, "altloc", None) or []
        return {i for i, alt in enumerate(altloc)
                if alt.strip() and alt.strip() != chosen}


    def set_model_conformer(self, mid: str, conformer: Optional[str]) -> None:
        """Show only one alternate conformation, or ``None`` for all of them.

        Implemented by hiding the other conformers' atoms rather than by selecting the
        chosen one, so it composes with the type and atom hiding already in
        :meth:`_shown_indices` instead of overriding them.
        """
        entry = self._model_entry(mid)
        if entry is None:
            return
        if conformer is not None and not isinstance(conformer, str):
            raise ValueError(
                f"conformer is an altloc letter or None, got {conformer!r}")
        wanted = (conformer or "").strip() or None
        if wanted is not None and wanted not in self._conformers_of(entry):
            raise ValueError(
                "model %s has no conformer %r (has %s)"
                % (mid, wanted, ", ".join(self._conformers_of(entry)) or "none"))
        if entry.get("conformer") == wanted:
            return
        entry["conformer"] = wanted
        self._apply_model_rep(entry)
        self._status("Showing conformer %s" % wanted if wanted
                     else "Showing all conformers")


    def _apply_model_rep(self, entry) -> None:
        session = entry["session"]
        reps = list(entry.get("reps") or [entry["rep"]])
        rep = reps[0]
        on = self._shown_indices(entry)  # restrict to shown structure types
        # While a neighborhood context stands, the main representation steps aside
        # there: the region reads as ball-and-stick alone — the Mol* focus look — not
        # sticks threaded through ribbon. The context layer (below) draws those atoms.
        main_on = on
        context = entry.get("context_on") if rep != "ball-and-stick" else None
        if context:
            exclude = set(context)
            base = on if on is not None else range(session._n_atoms)
            main_on = [i for i in base if i not in exclude]
        attribute = entry.get("attribute")
        if attribute is not None and entry.get("color") == attribute["name"]:
            values = attribute["values"]
            # A ribbon draws no side chains, so per-atom values that live there (hotspot
            # rotamer severity) would simply not be on screen. Where one is supplied, use the
            # residue-broadcast array instead — what the representation can actually show.
            if not _rep_shows_atoms(rep) and attribute.get("residue_values") is not None:
                values = attribute["residue_values"]
            # A computed per-atom quantity (Q-score, hotspot severity), so it colors through
            # the attribute path rather than a Mol* theme: the values go over as an array and
            # the frontend's pxviewer-attribute theme maps them. Same representation type as
            # ever — only what decides each atom's color changes.
            #
            # Registered by name rather than passed as a bare array, so the representation
            # says what it is coloring by instead of "values".
            session.set_attribute(attribute["name"], values)
            session.color_by(attribute["name"], type=rep, palette=attribute["palette"],
                             domain=attribute["domain"], on=main_on)
            # The attribute API replaces the representation list, so the model's other
            # layers are added back — carrying the same coloring. A value coloring is a
            # statement about the atoms, not about one representation of them: drawing a
            # B-factor cartoon threaded with element-colored sticks says two different
            # things about the same atom, and the legend belongs to neither.
            for extra in reps[1:]:
                session.color_by(attribute["name"], type=extra,
                                 palette=attribute["palette"],
                                 domain=attribute["domain"], on=main_on, replace=False)
            self._add_context_layer(entry, session, on)
            return
        for i, layer in enumerate(reps):
            kwargs = self._model_color_kwargs(entry, layer)
            if main_on is not None:
                kwargs["on"] = main_on
            method = session.set_representation if i == 0 else session.add_representation
            method(layer, **kwargs)
        self._add_context_layer(entry, session, on)


    def _add_context_layer(self, entry, session, on) -> None:
        """The selection's neighborhood as an extra ball-and-stick layer (see
        ``_set_context_rep``), restricted to what the structure-type toggles show."""
        # Whatever else happens, never two context layers. Tracking-and-removing the
        # previous id still stacked duplicates under suite load (two adds that never
        # saw each other's tracking), so the layer carries a FIXED id instead: the
        # representation store is id-keyed, and a second add overwrites rather than
        # appends — duplicate-proof against any thread or ordering.
        layer_id = f"context-{entry['id']}"
        context = entry.get("context_on")
        if not context or entry.get("rep") == "ball-and-stick":
            try:
                session.remove_representation(layer_id)  # idempotent
            except Exception:  # pragma: no cover - defensive
                pass
            return
        if on is not None:
            shown = set(on)
            context = [i for i in context if i in shown]
            if not context:
                return
        attribute = entry.get("attribute")
        if entry.get("color") in _MODEL_VALUE_COLORS and attribute is not None:
            # A computed coloring (map-model CC, Q-score, hotspot severity) must not
            # stop at the neighborhood's edge: the sticks around a selection are
            # exactly the atoms being judged, and element colors there made the
            # guided "this residue is painted pink" reading simply false. The values
            # are already registered on the session; add the layer alongside.
            session.color_by(attribute["name"], type="ball-and-stick",
                             palette=attribute["palette"], domain=attribute["domain"],
                             on=context, id=layer_id, replace=False)
            return
        kwargs = self._model_color_kwargs(entry, "ball-and-stick")
        kwargs["on"] = context
        kwargs["id"] = layer_id
        session.add_representation("ball-and-stick", **kwargs)


    def _model_color_kwargs(self, entry, rep: str) -> dict:
        """How to color a model's representation: an explicit user color wins; else the
        palette default (carbon-tint for atoms, uniform for ribbons); else the theme default."""
        explicit = entry.get("color")
        if explicit in _MODEL_VALUE_COLORS:
            # Chosen but not computed yet (or it failed): Mol* has no such theme, so fall
            # through to the default rather than handing it a name it cannot resolve.
            explicit = None
        if explicit:
            return {"color": explicit}  # a color the user set by hand — uniform, as before
        base = entry.get("color_default")  # the random palette default set at creation
        if base:
            return {"carbon_color": base} if _rep_shows_atoms(rep) else {"color": base}
        return {"color": _model_rep_color(rep)}  # no default (fallback): the theme default


    def _default_model_rep(self, session) -> str:
        from . import cctbx_io

        model = getattr(session, "model", None)
        return "cartoon" if model is not None and cctbx_io.model_is_polymer(model) else "ball-and-stick"


    def _add_model(self, session, name: str, *, group: Optional[str] = None,
                   rep: Optional[str] = None) -> str:
        """Register + show a model session (visible + active); returns its id.

        ``rep`` overrides the representation; otherwise cartoon reads better for a
        polymer and ball-and-stick otherwise. The choice is replayed to the viewer
        when it connects and shown in the inline dropdown.
        """
        session.start(host=self._host, port=0)
        self._model_counter += 1
        mid = f"model-{self._model_counter}"
        if rep is not None:
            reps = [rep]
        else:
            reps = list(self._default_model_reps)
            # Cartoon has nothing useful to draw for an isolated ligand/small molecule.
            # Preserve the historical atomistic fallback unless another atom layer was
            # explicitly selected alongside it.
            from . import cctbx_io
            model = getattr(session, "model", None)
            if (model is None or not cctbx_io.model_is_polymer(model)) \
                    and not any(_rep_shows_atoms(value) for value in reps):
                reps = ["ball-and-stick"]
        rep = reps[0]
        type_groups = _structure_type_groups(session)
        hidden_types = {
            label for label in type_groups if label not in self._default_shown_types}
        entry = {"id": mid, "name": name, "session": session, "visible": True, "group": group,
                 "rep": rep, "reps": reps, "color": None,
                 "hidden_types": hidden_types, "hidden_atoms": set(),
                 "type_groups": type_groups, "clip": (0.0, 1.0),
                 # The clip sphere starts in "selection" mode: dormant (no radius)
                 # until a selection lands, then it re-fits each one. The retired
                 # Selection-pane checkbox's saved preference seeds the mode once —
                 # after that the Appearance pane's Clip sphere row is the control.
                 "sphere": self._initial_sphere_mode(),
                 "interactions": self._default_model_interactions, "edits": None,
                 # Opening color: a random pick from the session's current palette group
                 # (see palettes.PaletteCycler). Overridden the moment the user sets one by
                 # hand. Read by _model_color_kwargs.
                 "color_default": self._palettes.next_color()}
        self._models.append(entry)
        self._apply_model_rep(entry)
        if entry["interactions"]:
            session.set_computed_interactions(True)
        self._active_model_id = mid
        self._rebind_kernel_session()
        # Register this model's pick handler once (tagged with its id); the click
        # mode is what actually turns picking on/off. Registering here means a
        # selection can be built in any loaded model, not just the active one.
        session.on_selection(lambda sel, mid=mid: self._on_model_selection(mid, sel))
        # Every atom click reports a pick, whatever the click mode — Pick mode only
        # gates whether an *accumulating* selection is built from it. The pick names
        # the model and the atom; the GUI side decides what the click means (see
        # ControlsWindow._on_atom_picked): with no click mode armed it selects the
        # residue through the same pipeline a typed selection takes, otherwise it
        # just points the panel at the model.
        session.on_pick(lambda info, mid=mid: (
            self.bridge.atom_picked.emit(
                mid, info["index"] if isinstance(info.get("index"), int) else -1,
                bool(info.get("shift")), str(info.get("repr") or ""))
            if info else None))
        # Volume commands ride whichever session is the control session, so contour
        # changes made in the viewport can come back on any of them.
        session.on_volume_iso(self._on_volume_iso_changed)
        session.on_localres_shown(self.bridge.localres_shown.emit)
        session.on_tug(lambda action, atom, target, mid=mid: self._on_tug(mid, action, atom, target))
        # Markers are a scene-level thing, not this model's — but only the armed (control)
        # session's viewport reports one, so wiring every session is harmless.
        session.on_marker(lambda position, atom: self._on_marker(position, atom))
        session.on_marker_move(lambda mid_, pos, final: self._on_marker_move(mid_, pos, final))
        # A chain-stall means the page is alive but one scene update has held its
        # apply queue past the threshold — name the work in the status line so the
        # freeze is diagnosable rather than silent.
        session.on_chain_stall(lambda stall: self._status(
            f"Viewer busy applying {stall.get('label')} "
            f"({int((stall.get('ms') or 0) / 1000)}s)…"))
        if self._selection_enabled:
            session.enable_mouse_selection()  # handler already registered; just arm click mode
        if self._tug_enabled:
            session.set_tug_mode(True)
        if self._focus_surroundings and not self._selection_enabled:
            session.set_focus_surroundings(True)
        self._wire_active(session)
        self._warm_restraints(mid)  # so the first drag does not pay for pdb_interpretation
        self._reload_viewport()
        self._emit_loaded_changed()
        return mid


    def _warm_restraints(self, mid: Optional[str] = None) -> None:
        """Build a model's restraints ahead of time, off any thread the user is waiting on.

        The first drag on a model pays for pdb_interpretation — measured at 0.66 s on
        ubiquitin and 1.35 s on a 2737-atom structure — and it lands *inside* the click, so
        the grab sits dead before anything moves. (Later drags are 22-88 ms, which is the
        real cost of a drag; the rest is this one-off.)

        Called when a model loads, so the build is done long before a drag — and again on
        Shift-keydown, to cover a model that arrived another way or a drag that beat the
        load-time build.

        Runs on a background thread so nothing waits on it, which is only safe because every
        restraint build now funnels through one lock (see ``edits.build_restraints`` and
        ``GeometryRestraints``): cctbx restraint building touches process-global state, and
        two builds at once — the background warm and, say, the restraint tables building
        their own — leave it in a state neither asked for. That race, not any change in
        what the restraints *contain* (the two build paths produce identical proxies), is
        what made a restraint-table test fail intermittently before the paths were unified.
        """
        entry = self._model_entry(mid if mid is not None else self._active_model_id,
                                  missing_ok=True)
        if entry is None:
            return  # a tug arriving for a model already removed
        model = getattr(entry["session"], "model", None)
        if model is None:
            return
        try:
            if model.restraints_manager_available():
                return
        except Exception:  # pragma: no cover - defensive
            return
        from .geometry import monomer_library_available

        if not monomer_library_available():
            return  # nothing to warm; the drag will say so when it is tried

        def work() -> None:
            try:
                from . import edits

                built_now = False
                with self._restraints_lock:
                    if not model.restraints_manager_available():
                        edits.build_restraints(model)
                        built_now = True
                entry.pop("restraints_error", None)
                if built_now:
                    self.bridge.restraints_changed.emit(mid)
            except Exception as exc:  # pragma: no cover - cctbx/runtime errors
                # Recorded rather than swallowed. The warm runs on a thread and cannot ask
                # the user anything, but a later minimize or drag can -- and asking is only
                # cheap if it already knows a build failed, since finding out otherwise
                # costs a full interpretation pass on the GUI thread. See
                # ControlsWindow._offer_restraints_if_blocked.
                entry["restraints_error"] = str(exc)

        threading.Thread(target=work, name="pxviewer-restraints-warm", daemon=True).start()


    def set_model_representation(self, mid: str, rep: str) -> None:
        """Change a model's representation type (from the inline dropdown).

        An explicit choice here also clears the standing neighborhood context layer:
        the user has just said what the model should look like, and keeping a
        selection's ball-and-stick patch on top of it reads as the switch half
        failing. Picking Cartoon therefore yields pure ribbon — until the next
        selection gesture, where the Selection pane's Neighborhood checkbox applies
        as ever. (See ROADMAP.md for the possible future beyond one-rep-per-model.)"""
        entry = self._model_entry(mid)
        if entry is None or entry.get("reps") == [rep]:
            return
        entry["rep"] = rep
        entry["reps"] = [rep]
        entry["context_on"] = None
        self._apply_model_rep(entry)


    def ensure_atoms_shown_once(self, mid: Optional[str] = None) -> None:
        """:meth:`ensure_atoms_shown`, but only the first time per model — the courtesy
        switch for worklist row clicks (Validation, Hotspots). The first click puts a
        ribbon model in ball-and-stick; a user who then deliberately switches back to
        ribbon in the Appearance pane has made a choice, and later clicks respect it
        rather than re-flipping (at the cost of the slower per-click ribbon rebuild)."""
        mid = mid or self._active_model_id
        entry = self._model_entry(mid)
        if entry is None or entry.get("_atoms_shown_once"):
            return
        entry["_atoms_shown_once"] = True
        self.ensure_atoms_shown(mid)


    def ensure_atoms_shown(self, mid: Optional[str] = None) -> None:
        """Switch a model off a ribbon (cartoon) view to ball-and-stick, so atom-precision
        work — measuring, restraint notations, restraint edits, ligand markers, dragging —
        is not invisible or drawn into empty space over the ribbon. A no-op for any rep that
        already shows atoms (ball-and-stick, spacefill, …). Safe to call from any thread."""
        mid = mid or self._active_model_id
        entry = self._model_entry(mid)
        if entry is None or entry.get("rep") != "cartoon":
            return
        entry["rep"] = "ball-and-stick"
        entry["reps"] = ["ball-and-stick"]
        self._apply_model_rep(entry)
        self._emit_loaded_changed()  # reflect the switch in the appearance pane and tree


    def set_model_type_hidden(self, mid: str, label: str, hidden: bool) -> None:
        """Show or hide a structure type on a model.

        ``label`` is one of the display names :meth:`model_structure_types`
        reports ('Protein', 'Water', 'Ligand / other', …), matched
        case-insensitively. A label the model does not actually contain
        raises — storing it would report atoms hidden that do not exist
        (figfuzz j05)."""
        entry = self._model_entry(mid)
        present = {k.lower(): k for k in self._type_groups(entry)}
        key = present.get(str(label).strip().lower())
        if key is None:
            raise ValueError(
                f"no structure type {label!r} on {mid} — has "
                f"{', '.join(present.values()) or 'nothing'}")
        types = entry.setdefault("hidden_types", set())
        if (key in types) == bool(hidden):
            return
        types.add(key) if hidden else types.discard(key)
        self._apply_model_rep(entry)


    def hide_selected(self) -> None:
        """Hide the currently-selected atoms (draw everything else)."""
        self._set_selected_hidden(True)


    def show_selected(self) -> None:
        """Show (un-hide) the currently-selected atoms."""
        self._set_selected_hidden(False)


    def _set_selected_hidden(self, hide: bool) -> None:
        """Add the selection to (or remove it from) each model's hidden-atoms set and
        redraw — the same partial-representation path structure-type hiding uses."""
        changed = False
        for mid, indices in self._scene_selection.items():
            entry = self._model_entry(mid)
            if entry is None or not indices:
                continue
            hidden = entry.setdefault("hidden_atoms", set())
            before = len(hidden)
            if hide:
                hidden.update(indices)
            else:
                hidden.difference_update(indices)
            if len(hidden) != before:
                self._apply_model_rep(entry)
                changed = True
        if not changed:
            self._status("select some atoms first" if not self._scene_selection
                         else ("already hidden" if hide else "already shown"))


    def model_structure_types(self, mid: str) -> list:
        """The structure types present in a model (for the show/hide menu)."""
        entry = self._model_entry(mid)
        return list(self._type_groups(entry).keys()) if entry else []


    def set_model_color(self, mid: str, color: Optional[str]) -> None:
        """Set a model's color theme (None = the representation's default).

        ``'qscore'`` is not a theme the viewer knows: it is computed here against the map and
        pushed as per-atom values (see :meth:`color_model_by_qscore`).
        """
        entry = self._model_entry(mid)
        if entry is None or entry.get("color") == color:
            return
        entry["color"] = color
        info = _MODEL_VALUE_COLORS.get(color)
        if info is not None:
            getattr(self, info["method"])(mid)
            return
        # Leaving a computed color drops the values it colored by: they belong to one model
        # (and for Q-score/hotspots, to one pairing with a map), not to the entry forever.
        entry.pop("attribute", None)
        self._apply_model_rep(entry)


    @staticmethod
    def _opening_domain(entry, name: str, default) -> tuple:
        """``(domain, default_domain)`` for a value coloring about to be applied.

        The range the user last set for *this* coloring on *this* model survives a
        re-apply (a representation change, a recompute); anything else opens at the
        calibrated default, which is also where Reset goes. Shared by all five
        colorings so "the scale sticks" means one thing.
        """
        default = (float(default[0]), float(default[1]))
        previous = entry.get("attribute") or {}
        kept = previous.get("domain") if previous.get("name") == name else None
        return (tuple(float(v) for v in kept) if kept else default), default


    def color_model_by_property(self, mid: str, kind: Optional[str] = None) -> None:
        """Color a model by a refined per-atom number (B-factor or occupancy), through
        the attribute path so the scale is explicit and the user's to set."""
        entry = self._model_entry(mid)
        model = getattr(entry["session"], "model", None) if entry else None
        if entry is None:
            return
        # Dispatched from the registry, which calls every coloring with just the model
        # id; the color already on the entry says which property is meant.
        kind = kind or entry.get("color")
        if model is None:
            self._warn("this model has no atoms to color by")
            return
        atoms = model.get_hierarchy().atoms()
        values = np.asarray(atoms.extract_b() if kind == "bfactor"
                            else atoms.extract_occ(), dtype=float)
        if kind == "occupancy":
            default = (0.0, 1.0)   # its natural scale; percentiles of all-1.0 collapse
        else:
            finite = values[np.isfinite(values)]
            lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
            if hi <= lo:
                hi = lo + 0.1
            default = (round(lo, 2), round(hi, 2))
        domain, default = self._opening_domain(entry, kind, default)
        entry["attribute"] = {
            "name": kind, "values": values,
            "domain": domain, "default_domain": default,
            "palette": _VALUE_PALETTE,
        }
        entry["color"] = kind
        self._apply_model_rep(entry)
        self._emit_loaded_changed()


    def set_model_value_domain(self, mid: str, lo: float, hi: float) -> None:
        """Set the color scale's range for a model's value coloring -- the model-side
        twin of :meth:`set_localres_domain`, stable until the user moves it."""
        entry = self._model_entry(mid)
        attribute = entry.get("attribute") if entry else None
        if attribute is None or attribute["name"] not in _MODEL_VALUE_COLORS:
            return
        lo, hi = float(lo), float(hi)
        if not (hi > lo):
            self._warn("the color range needs max above min")
            return
        attribute["domain"] = (lo, hi)
        self._apply_model_rep(entry)
        self._emit_loaded_changed()


    def fit_model_value_domain(self, mid: str) -> None:
        """Span the model coloring's range over the values actually present."""
        entry = self._model_entry(mid)
        attribute = entry.get("attribute") if entry else None
        if attribute is None or attribute["name"] not in _MODEL_VALUE_COLORS:
            return
        finite = attribute["values"][np.isfinite(attribute["values"])]
        if not finite.size:
            return
        lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
        if hi <= lo:
            hi = lo + 0.1
        self.set_model_value_domain(mid, round(lo, 2), round(hi, 2))


    def reset_model_value_domain(self, mid: str) -> None:
        entry = self._model_entry(mid)
        attribute = entry.get("attribute") if entry else None
        if attribute is None or attribute.get("default_domain") is None:
            return
        self.set_model_value_domain(mid, *attribute["default_domain"])


    def color_model_by_qscore(self, mid: str) -> None:
        """Color a model by per-atom Q-score — its fit to the map, computed by cctbx.

        Needs a map paired with the model: Q-score is a measure of the model *against
        density*, so without one there is nothing to score and the choice is refused rather
        than silently showing something else. Runs on a thread (seconds on a large
        structure) and applies when it lands, if the user has not moved on.
        """
        entry = self._model_entry(mid)
        if entry is None:
            return
        if entry.get("color") != _QSCORE_COLOR:
            # Called directly rather than dispatched from set_model_color — go
            # through it so the entry's color claim, the attribute cleanup, and
            # the apply-time guard all stay in step. The old path computed fine
            # then discarded the values, a silent no-op (figfuzz h04).
            self.set_model_color(mid, _QSCORE_COLOR)
            return
        mmm = self.group_mmm(entry.get("group"))
        if mmm is None:
            self._status("Q-score needs a map paired with this model — pair one, or Make maps")
            entry["color"] = None  # nothing was applied; do not leave the menu claiming it was
            self._apply_model_rep(entry)
            self._emit_loaded_changed()
            return
        self._status(f"computing Q-score for {entry['name']}…")

        def work() -> None:
            try:
                from .qscore import per_atom_qscore

                values = per_atom_qscore(mmm)
            except Exception as exc:  # pragma: no cover - cctbx/runtime errors
                self._status(f"Q-score failed: {exc}")
                return

            def apply_on_main() -> None:
                from .qscore import DOMAIN, PALETTE

                current = self._model_entry(mid)
                # The model may have been unloaded, or the user may have picked another
                # color while this ran — either way these values are no longer wanted.
                if current is None or current.get("color") != _QSCORE_COLOR:
                    return
                domain, default = self._opening_domain(current, _QSCORE_COLOR, DOMAIN)
                current["attribute"] = {"name": _QSCORE_COLOR, "values": values,
                                        "domain": domain, "default_domain": default,
                                        "palette": PALETTE}
                self._apply_model_rep(current)
                finite = values[np.isfinite(values)]
                if finite.size:
                    self._status(f"Q-score for {current['name']}: mean {finite.mean():.2f} "
                                 f"over {finite.size} atoms (red poor, green good)")
                else:  # pragma: no cover - defensive
                    self._status("Q-score produced no values")

            self.bridge.run_on_main.emit(apply_on_main)

        self.run_background(work, name="pxviewer-qscore", label="Computing Q-score")


    #: The px spectrum for map-model CC on atoms, worst to best: hot pink through
    #: purple and blue to calm teal — deliberately unlike Q-score's red-green, the
    #: hotspot yellow-red and the localres blue-red, so no two colorings can be
    #: mistaken for one another. The domain is the correlation's own absolute scale,
    #: so a color means the same thing on every structure.
    _CC_MODEL_PALETTE = ["#ec4899", "#a855f7", "#6366f1", "#3b82f6", "#2dd4bf"]

    _CC_MODEL_DOMAIN = (0.0, 1.0)


    def color_model_by_mapcc(self, mid: str) -> None:
        """Color a model by per-atom map-model correlation — the model's own expected
        density against the real map, computed by cctbx (mmtbx.maps.correlation), atom
        by atom. Pink is an atom the map disagrees with, teal a good fit.

        Needs a paired map, exactly like Q-score, and is computed against the model as
        it stands — re-pick the coloring to recompute after refining or dragging.
        This replaced two volume presentations of the same quantity (a color-by-CC
        map surface and a hotspot-style deficit cloud): both cost far more to draw,
        and neither said *which atoms* to fix, which is the question.
        """
        entry = self._model_entry(mid)
        if entry is None:
            return
        if entry.get("color") != _CC_COLOR:
            # See color_model_by_qscore: route through set_model_color so the
            # computed values are not discarded by the apply-time guard.
            self.set_model_color(mid, _CC_COLOR)
            return
        mmm = self.group_mmm(entry.get("group"))
        if mmm is None or mmm.model() is None:
            self._status("map-model CC needs a map paired with this model — "
                         "pair one, or Make maps")
            entry["color"] = None  # nothing was applied; the menu must not claim it was
            self._apply_model_rep(entry)
            self._emit_loaded_changed()
            return
        self._status(f"computing map-model CC for {entry['name']}…")

        def work() -> None:
            from . import hotspots as hotspots_mod

            try:
                values = hotspots_mod.per_atom_cc(mmm)
            except Exception as exc:  # pragma: no cover - cctbx/runtime errors
                self._status(f"map-model CC failed: {exc}")
                return
            model = getattr(entry["session"], "model", None)
            # The ribbon draws residues, not atoms: broadcast per-residue means so the
            # coloring is visible on a cartoon too, not only in ball-and-stick.
            residue_values = (hotspots_mod.residue_broadcast(model, values, reduce=min)
                              if model is not None else None)

            def apply_on_main() -> None:
                current = self._model_entry(mid)
                # The model may have been unloaded, or the user may have picked another
                # color while this ran — either way these values are no longer wanted.
                if current is None or current.get("color") != _CC_COLOR:
                    return
                domain, default = self._opening_domain(
                    current, _CC_COLOR, self._CC_MODEL_DOMAIN)
                current["attribute"] = {
                    "name": _CC_COLOR, "values": values,
                    "residue_values": residue_values,
                    "domain": domain, "default_domain": default,
                    "palette": self._CC_MODEL_PALETTE}
                self._apply_model_rep(current)
                finite = values[np.isfinite(values)]
                if finite.size:
                    self._status(
                        f"map-model CC for {current['name']}: mean {finite.mean():.2f} "
                        f"over {finite.size} atoms (pink poor, teal good)")
                else:  # pragma: no cover - defensive
                    self._status("map-model CC produced no values")

            self.bridge.run_on_main.emit(apply_on_main)

        self.run_background(work, name="pxviewer-mapcc", label="Computing map-model CC")


    # -- validation hotspots ---------------------------------------------

    def compute_hotspots(self, mid: Optional[str] = None) -> None:
        """Aggregate the validation metrics into one per-atom severity field, and color by it.

        The programmatic entry point. Interactively, hotspots is one more box in the
        Validation tab's run queue — it aggregates the very checks queued beside it, so
        a separate button would have run the same analysis twice and asked the user to
        choose between two spellings of one action.
        """
        entry = self._model_entry(mid or self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        model = getattr(entry["session"], "model", None)
        if model is None:
            raise ValueError("the active object has no cctbx model")
        mid, name = entry["id"], entry["name"]
        analysis = self._model_analysis(entry)  # shared with, and populates, the Validation tab

        def work() -> None:
            self._hotspots_work(mid, name, model, analysis)
            # Finding hotspots already ran Ramachandran and rotamers; spend the little
            # extra to run the remaining validators (reusing that shared analysis) so
            # the Validation tab is populated too.
            self._populate_validation_from_analysis(mid, model, analysis)

        self.run_background(work, name="pxviewer-hotspots", label="Finding hotspots")


    def _hotspots_work(self, mid, name, model, analysis) -> None:
        """Score the severity field and color by it (background thread).

        Shared by :meth:`compute_hotspots` and the validation queue, so ticking hotspots
        beside the per-residue checks costs the aggregation and nothing more: the
        analysis those checks already built is handed straight in.
        """
        entry = self._model_entry(mid)
        if entry is None:
            return
        from . import hotspots

        try:
            self._status(f"finding hotspots in {name}…")
            result = hotspots.score(
                model, fit="none", analysis=analysis,
                use_hydrogens=getattr(self, "_hotspot_hydrogens", False))
            columns = hotspots.residue_columns(result)
            rows = hotspots.residue_rows(model, result)
        except Exception as exc:  # pragma: no cover - validator/runtime errors
            self._status(f"hotspots failed: {exc}")
            return
        # Ask the browser its background so clean atoms can be colored to match and fade
        # into it. Off the GUI thread (it blocks on a round-trip); None -> a light default.
        palette = hotspots.hotspot_palette(entry["session"].background_color())
        residue_values = hotspots.residue_broadcast(model, result.values)

        def apply_on_main() -> None:
            current = self._model_entry(mid)
            if current is None:  # unloaded while we worked
                return
            # A newly computed score supersedes any imported concern field — the mirror
            # of open_hotspot_volume dropping a computed score. Severity and concern are
            # different quantities (see pxviewer.concern); a model shows one or the other.
            self._drop_imported_concern(current)
            self._hotspot_knee = hotspots.FIELD_ISO
            current["hotspots"] = result
            current.pop("hotspot_metric", None)  # a new score starts on Combined
            current["hotspot_palette"] = palette  # kept so a menu re-apply reuses it
            current["color"] = _HOTSPOT_COLOR
            domain, default = self._opening_domain(
                current, _HOTSPOT_COLOR, hotspots.DOMAIN)
            current["attribute"] = {
                "name": _HOTSPOT_COLOR, "values": result.values,
                "residue_values": residue_values,
                "domain": domain, "default_domain": default,
                "palette": palette,
            }
            self._apply_model_rep(current)
            self._emit_loaded_changed()
            self._status(f"{name}: {result.summary}")
            self.bridge.hotspots_ready.emit((mid, result, columns, rows))

        self.bridge.run_on_main.emit(apply_on_main)


    def _populate_validation_from_analysis(self, mid: str, model, analysis) -> None:
        """Fill the Validation tab from a just-finished hotspot run, reusing its shared analysis
        so the Ramachandran and rotamer runs are not repeated — only the validators hotspots
        did not need (cablam, C-beta, omega, Rama-Z) still run.

        Skips when validation is already cached: it is dropped whenever the atoms move, so a
        present cache is current, and re-running would waste those extra validators. Runs on the
        caller's (worker) thread and emits ``validation_ready`` for the GUI to pick up.
        """
        entry = self._model_entry(mid)
        if entry is None or entry.get("validation"):
            return
        from . import validation

        try:
            results = validation.run_all(model, analysis)
        except Exception:  # pragma: no cover - validator/runtime errors
            return
        ventry = self._model_entry(mid)
        if ventry is None:
            return
        ventry["validation"] = {r.key: r for r in results}
        self._mark_validated(ventry)  # fingerprint the coordinates these describe
        # draw_markers=False: fill the tab, but do not draw the markup over the hotspot coloring.
        self.bridge.validation_ready.emit((mid, results, False))
        self._refresh_validation_staleness()  # fresh results: clear any stale warning


    def _populate_hotspots_from_analysis(self, mid: str, model, analysis) -> None:
        """Fill the Hotspots tab from a just-finished validation run, reusing its analysis.

        The counterpart of :meth:`_populate_validation_from_analysis`, so whichever button the
        user pressed, both tabs end up populated and the *other* one is then instant — which
        matters because the two share reduce2 and probe, by far the most expensive steps.

        Deliberately does **not** recolor the model: the user asked for validation, and silently
        repainting the structure by hotspot severity would be a bigger change than they asked
        for. The table fills; choosing the hotspot coloring stays their call. Skips when a score
        is already cached (it is dropped whenever the atoms move, so a present one is current).
        """
        entry = self._model_entry(mid)
        if entry is None or entry.get("hotspots") is not None:
            return
        from . import hotspots

        try:
            result = hotspots.score(
                model, fit="none", analysis=analysis,
                use_hydrogens=getattr(self, "_hotspot_hydrogens", False))
            columns = hotspots.residue_columns(result)
            rows = hotspots.residue_rows(model, result)
        except Exception:  # pragma: no cover - validator/runtime errors
            return
        hentry = self._model_entry(mid)
        if hentry is None:
            return
        hentry["hotspots"] = result
        self.bridge.hotspots_ready.emit((mid, result, columns, rows))


    def color_model_by_hotspots(self, mid: str) -> None:
        """Color a model by a hotspot field already computed for it.

        Nothing is recomputed here — if there is no cached field the choice is refused and
        reverted, because computing a new score here would put a number on screen the user
        never asked for.
        """
        entry = self._model_entry(mid)
        if entry is None:
            return
        if entry.get("color") != _HOTSPOT_COLOR:
            # See color_model_by_qscore: route through set_model_color so the
            # entry's color claim matches what the atoms show.
            self.set_model_color(mid, _HOTSPOT_COLOR)
            return
        from . import hotspots

        result = entry.get("hotspots")
        if result is None:
            self._status("no hotspots computed yet — use Find hotspots on the Hotspots tab")
            entry["color"] = None
            self._apply_model_rep(entry)
            self._emit_loaded_changed()
            return
        model = getattr(entry["session"], "model", None)
        # Reuse the background-matched palette from when it was computed; fall back to the
        # default only if this model was scored before that was recorded.
        palette = entry.get("hotspot_palette") or hotspots.PALETTE
        domain, default = self._opening_domain(entry, _HOTSPOT_COLOR, hotspots.DOMAIN)
        entry["attribute"] = {
            "name": _HOTSPOT_COLOR, "values": result.values,
            "residue_values": (hotspots.residue_broadcast(model, result.values)
                               if model is not None else None),
            "domain": domain, "default_domain": default, "palette": palette,
        }
        self._apply_model_rep(entry)


    def set_hotspot_hydrogens(self, on: bool) -> None:
        """Choose whether the clash component uses hydrogens (MolProbity's definition) or the
        fast heavy-atom-only pass.

        Any cached score describes the *other* setting, so it is dropped — leaving it would show
        a table that quietly disagrees with the checkbox. The shared analysis is kept: it caches
        each probe pass under its own key, and the Ramachandran/rotamer runs are unaffected.
        """
        self._hotspot_hydrogens = bool(on)
        for entry in self._models:
            entry.pop("hotspots", None)
        self._status("clashes will use hydrogens (slower, MolProbity's definition)" if on
                     else "clashes will use the fast heavy-atom pass — re-run Find hotspots")


    def set_cloud_quality(self, quality: str) -> None:
        """Set the cloud quality preset (low/medium/high) and redraw a showing cloud.

        Low stays interactive on a light laptop; high is a clean still-figure render. Only the
        cloud is affected — the contour is a single cheap isosurface.
        """
        from . import hotspots

        if quality not in hotspots.CLOUD_QUALITY:
            return
        self._cloud_quality = quality
        entry = self._model_entry(self._active_model_id)
        if entry is not None and entry.get("hotspot_cloud"):
            self.show_hotspot_field(entry["id"], on=True, style="cloud")


    def open_hotspot_volume(self, path: Any, mid: Optional[str] = None, *,
                            style: str = "cloud") -> None:
        """Import bounded concern fields written by the Hotspots generator.

        A ``*_hotspots.json`` manifest imports every metric it lists and shows ``combined``
        first; a bare concern CCP4 imports on its own. Nothing is computed, and nothing is
        rescaled — the generator's concern values are already the display coordinates, and
        the manifest's own ``primary_display`` block says where yellow, orange and red fall.

        An import **supersedes** any computed severity score on this model, exactly as a
        computed score supersedes an import. The two are different quantities on different
        scales (see :mod:`pxviewer.concern`), so a model shows one or the other and never a
        table from one beside a map from the other.
        """
        from . import concern
        from .volume_io import VolumeData

        entry = self._model_entry(mid or self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        imported = concern.read_fields(path, VolumeData)

        self._clear_hotspot_field(entry)
        self._drop_computed_hotspots(entry)
        entry["concern"] = imported
        entry["concern_metric"] = imported.primary
        # The contract's own knee: below the yellow anchor the field is transparent.
        self._hotspot_knee = imported.anchors["yellow"]
        self.show_hotspot_field(entry["id"], on=True, style=style)
        self._emit_concern_table(entry)

        note = ""
        if imported.omitted_metrics:
            note = (f" — no {', '.join(imported.omitted_metrics)} field in this manifest"
                    f"{': ' + imported.omission_reason if imported.omission_reason else ''}")
        self._status(
            f"imported {len(imported.fields)} concern field"
            f"{'s' if len(imported.fields) != 1 else ''} from {imported.source.name}; "
            f"showing {imported.primary}{note}")


    def _drop_computed_hotspots(self, entry) -> None:
        """Forget a computed severity score, and stop coloring the model by it.

        Severity and concern are not convertible, so an imported field must not leave a
        severity table, a severity-colored model, or a severity-scaled slider behind it —
        that is exactly the mix that lets a table read 0.43 where the map correctly reads 0.
        """
        entry.pop("hotspots", None)
        entry.pop("hotspot_palette", None)
        if entry.get("color") == _HOTSPOT_COLOR:
            entry["color"] = None
            entry.pop("attribute", None)
            self._apply_model_rep(entry)
            self._emit_loaded_changed()


    def _drop_imported_concern(self, entry) -> None:
        """Forget an imported concern field — the mirror of :meth:`_drop_computed_hotspots`."""
        entry.pop("concern", None)
        entry.pop("concern_metric", None)


    def _emit_concern_table(self, entry) -> None:
        """Rebuild the Hotspots table from the imported maps at the current threshold.

        The values are sampled out of the concern grids themselves, so the table lists what
        the viewport is actually drawing; raising the threshold drops rows that have just
        become invisible instead of leaving them on screen with no field beside them.
        """
        from . import concern

        imported = entry.get("concern")
        model = getattr(entry["session"], "model", None)
        if imported is None or model is None:
            return
        metric = entry.get("concern_metric", imported.primary)
        threshold = float(getattr(self, "_hotspot_knee", 0.0))
        metrics = list(imported.fields)
        try:
            rows = concern.residue_rows(model, imported.fields, primary=metric,
                                        threshold=threshold)
        except Exception as exc:  # pragma: no cover - defensive
            self._status(f"could not read concern per residue: {exc}")
            return
        summary = (
            f"{metric} concern from {imported.source.name}: "
            f"{len(rows)} residue{'s' if len(rows) != 1 else ''} at or above {threshold:.2f}, "
            f"ranked by the field on screen. Bounded concern in [0, 1], read from the maps "
            f"themselves — no validation was run. {concern.TABLE_CAVEAT}")
        self.bridge.concern_ready.emit(
            (entry["id"], summary, concern.residue_columns(metrics), rows))


    def set_hotspot_field_metric(self, metric: str, mid: Optional[str] = None) -> None:
        """Switch the displayed field, whichever generation is on screen.

        With imported concern: among the manifest's fields, without re-reading any file.
        With a computed score: between the combined severity (``"combined"``) and its
        per-metric components — nothing is recomputed, because the components are kept
        precisely so the aggregate can be decomposed into the channels that produced
        it. The atom coloring follows (where hotspot coloring is what is on screen),
        and a 3-D field that is up is redrawn from the chosen channel; one that is not
        up stays down.

        Scripting/console only: the tab deliberately shows just the combined value
        (the per-check story is already told by the table's component columns and the
        Validation tab), so no UI drives this — a field-picker dropdown was shipped
        and removed as redundant concept surface.
        """
        entry = self._model_entry(mid or self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        imported = entry.get("concern")
        if imported is not None:
            if metric not in imported.fields:
                raise ValueError(f"concern field is not available: {metric}")
            entry["concern_metric"] = metric
            style = "contour" if entry.get("hotspot_volume") is not None else "cloud"
            self.show_hotspot_field(entry["id"], on=True, style=style)
            self._emit_concern_table(entry)
            return
        result = entry.get("hotspots")
        if result is None:
            raise ValueError("no hotspots yet — use Find hotspots, or Open volume… to import")
        if metric != "combined" and metric not in result.components:
            raise ValueError(f"severity component is not available: {metric}")
        from . import hotspots

        entry["hotspot_metric"] = metric
        values = result.values if metric == "combined" else result.components[metric]
        model = getattr(entry["session"], "model", None)
        # Recolor the atoms only where hotspot coloring is what is on screen; a user
        # who has since chosen another coloring keeps their choice.
        if model is not None and entry.get("color") == _HOTSPOT_COLOR:
            attribute = dict(entry.get("attribute") or {})
            attribute.update({
                "name": _HOTSPOT_COLOR, "values": values,
                "residue_values": hotspots.residue_broadcast(model, values),
                "domain": hotspots.DOMAIN,
                "palette": entry.get("hotspot_palette") or attribute.get("palette"),
            })
            entry["attribute"] = attribute
            self._apply_model_rep(entry)
        if entry.get("hotspot_cloud"):
            self.show_hotspot_field(entry["id"], on=True, style="cloud")
        elif entry.get("hotspot_volume") is not None:
            self.show_hotspot_field(entry["id"], on=True, style="contour")


    def show_hotspot_field(self, mid: Optional[str] = None, *, on: bool = True,
                           style: str = "cloud") -> None:
        """Show/hide the 3-D field for a model — imported concern, or a computed score.

        Per-atom coloring only shows the surface, so a buried hotspot stays hidden until you
        rotate into it, and while it is on you have lost element/chain coloring. A 3-D field is
        visible through the structure and leaves that channel free.

        ``style`` is ``'cloud'`` — every voxel colored by value, transparent where clean
        through yellow to red — or ``'contour'``, a single translucent shell at the current
        threshold. Whichever is showing is torn down before the other is drawn; only one at a
        time. Both styles read the *same* absolute field: hue and opacity follow the value, on
        a fixed domain, so a color means the same thing in every structure and metric.
        """
        entry = self._model_entry(mid or self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        self._clear_hotspot_field(entry)
        if not on:
            return

        from . import concern, hotspots

        result = entry.get("hotspots")
        imported = entry.get("concern")
        if result is None and imported is None:
            self._status("no hotspots yet — use Find hotspots, or Open volume… to import")
            return
        # The cloud's grid is set by the quality preset (it is raymarched, so grid size and
        # step count drive frame rate); the contour keeps the fine grid.
        quality = getattr(self, "_cloud_quality", hotspots.CLOUD_QUALITY_DEFAULT)
        cloud_spacing, cloud_steps = hotspots.CLOUD_QUALITY[quality]

        if imported is not None:
            metric = entry.get("concern_metric", imported.primary)
            selected = imported.fields[metric]
            field = selected.values
            spacing = tuple(float(v) for v in selected.concern.pixel_sizes)
            origin = tuple(int(v) for v in selected.concern.origin)
            threshold = float(getattr(self, "_hotspot_knee", imported.anchors["yellow"]))
            units, label = "concern", f"{metric} concern"
        else:
            model = getattr(entry["session"], "model", None)
            if model is None:  # pragma: no cover - defensive
                return
            grid_spacing = cloud_spacing if style == "cloud" else hotspots.FIELD_SPACING
            metric = entry.get("hotspot_metric", "combined")
            values = (result.values if metric == "combined"
                      else result.components.get(metric, result.values))
            field, spacing, origin = hotspots.severity_field(
                model, values, spacing=grid_spacing)
            threshold = float(getattr(self, "_hotspot_knee", hotspots.FIELD_ISO))
            units = "severity"
            label = "severity" if metric == "combined" else f"{metric} severity"
        if not (field >= threshold).any():
            self._status(f"nothing reaches {units} {threshold:.2f} — nothing to draw in 3-D")
            return

        if style == "cloud":
            # A value-colored raymarched cloud, streamed to the model's own viewer as a
            # direct-volume (MVS has no such node, so it cannot go through the shared scene).
            if imported is not None:
                from .volume_io import encode_hotspot_concern

                # Tell the viewer where the contract puts each color, rather than letting it
                # infer a ramp from the knee: the knee is a user control and moves, the
                # anchors are the generator's fixed scale and must not move with it.
                entry["session"].set_hotspot_anchors(imported.anchors)
                payload = encode_hotspot_concern(
                    selected.concern.map_manager, concern_knee=threshold,
                    steps_per_cell=cloud_steps)
            else:
                entry["session"].set_hotspot_anchors(None)
                payload = hotspots.encode_severity_box(
                    field, spacing, origin, steps_per_cell=cloud_steps)
            entry["session"].show_hotspot_volume(payload)
            entry["hotspot_cloud"] = True
            # A fresh cloud defaults its knee to the cut; reapply the user's setting if any, so
            # redrawing (a recompute, a style toggle) does not silently reset the slider.
            knee = getattr(self, "_hotspot_knee", None)
            if knee is not None:
                self.set_hotspot_opacity(entry["id"], knee)
            self._status(f"{label} density for {entry['name']} (transparent clean → red)")
            return

        from .volume_io import VolumeData

        data = VolumeData.from_numpy(
            field, spacing=spacing, origin=origin, name=f"{entry['name']} hotspots")
        # The shell's color is the color the density shows at that same level — one scale,
        # read two ways, so switching style does not restate the value differently.
        color = (concern.concern_color(threshold, imported.anchors) if imported is not None
                 else hotspots.severity_color(threshold))
        vid = self._add_volume(data, f"{entry['name']} hotspots", group=entry.get("group"),
                               color=color, iso=threshold, iso_kind="absolute")
        # Translucent, so the model stays readable through it — the shell is a pointer, not
        # the thing you are looking at.
        self.set_volume_opacity(vid, 0.45)
        entry["hotspot_volume"] = vid
        self._status(f"{label} contour at {threshold:.2f} for {entry['name']}")


    def set_hotspot_threshold(self, mid: Optional[str], level: float) -> None:
        """Set the absolute threshold driving whichever field is on screen.

        The number is in the units of the field being shown — bounded concern for an import,
        severity for a computed score — and is never converted between them. Only the wire
        format differs: the cloud's opacity knee is a fraction of the streamed grid, and the
        legacy grid is streamed pre-divided by the severity cap.
        """
        entry = self._model_entry(mid or self._active_model_id)
        if entry is None:
            return
        from . import concern, hotspots

        imported = entry.get("concern")
        if imported is not None:
            level = min(1.0, max(0.0, float(level)))
        else:
            level = min(hotspots.SEVERITY_CAP, max(0.0, float(level)))
        self._hotspot_knee = level
        if entry.get("hotspot_cloud"):
            entry["session"].set_hotspot_opacity(
                level if imported is not None else level / hotspots.SEVERITY_CAP)
        else:
            vid = entry.get("hotspot_volume")
            if vid is not None:
                self.set_volume_iso(vid, level)
                self.set_volume_color(
                    vid, concern.concern_color(level, imported.anchors) if imported is not None
                    else hotspots.severity_color(level))
        if imported is not None:
            # The table lists what is visible, so it follows the threshold that decides that.
            self._emit_concern_table(entry)


    def set_hotspot_opacity(self, mid: Optional[str], knee_severity: float) -> None:
        """Backward-compatible name for setting the shared hotspot threshold."""
        self.set_hotspot_threshold(mid, knee_severity)


    def _clear_hotspot_field(self, entry) -> None:
        """Tear down whichever 3-D field a model is showing (cloud or contour)."""
        vid = entry.pop("hotspot_volume", None)
        if vid is not None:
            self.remove_volume(vid)          # the MVS-scene contour
        if entry.pop("hotspot_cloud", None):
            try:
                entry["session"].clear_hotspot_volume()   # the streamed cloud
            except Exception:  # pragma: no cover - defensive
                pass


    def set_model_interactions(self, mid: str, visible: bool) -> None:
        """Show/hide the computed non-covalent interactions overlay for a model."""
        entry = self._model_entry(mid)
        if entry is None or entry.get("interactions", False) == bool(visible):
            return
        entry["interactions"] = bool(visible)
        try:
            entry["session"].set_computed_interactions(bool(visible))
        except Exception:  # pragma: no cover - defensive
            pass


    # -- tools (measure / clashes / display) -----------------------------

    _MEASURE_ARITY = {"distance": 2, "angle": 3, "dihedral": 4}


    def measure_selection(self, kind: str, atoms: Optional[list] = None) -> dict:
        """Draw a distance/angle/dihedral from the active model's selected atoms.

        ``atoms`` may name the vertices explicitly (positional indices, in order —
        vertex order is semantic for angle/dihedral) instead of reading the scene
        selection. Note that ``session.select()``/``session.highlight()`` only
        *draw* — they do not feed the selection this method reads; build it with
        select_picked_atom / select_by_expression / toggle_picked_residue / clicks.

        Returns ``{"kind", "atoms", "value", "unit", "summary"}`` — the measured
        value is computed here from the model's coordinates, since the drawn
        label is the only place the number otherwise exists.
        """
        session = self.active_model_session()
        if session is None:
            raise ValueError("load a model first")
        need = self._MEASURE_ARITY[kind]
        if atoms is None:
            with self._scene_lock:
                atoms = list(self._scene_selection.get(self._active_model_id, []))
        else:
            atoms = [int(a) for a in atoms]
        if len(atoms) != need:
            hint = ("" if atoms else
                    " — session.select()/highlight() only draw; select with "
                    "select_picked_atom, select_by_expression, or clicks, "
                    "or pass atoms=[...]")
            raise ValueError(
                f"{kind} needs exactly {need} atoms (have {len(atoms)}){hint}")
        self.ensure_atoms_shown()  # the measurement marks atoms — show them under the ribbon
        if kind == "distance":
            session.add_distance(atoms[0], atoms[1])
        elif kind == "angle":
            session.add_angle(atoms[0], atoms[1], atoms[2])
        else:
            session.add_dihedral(atoms[0], atoms[1], atoms[2], atoms[3])
        # Compute the value ourselves: it lives only in the viewport's label
        # otherwise — an API measurement no caller could ever read back
        # (figfuzz i04: 'drew angle on 3 atoms' was the whole answer).
        value = self._measure_value(session.model.get_sites_cart(), kind, atoms)
        return {"kind": kind, "atoms": atoms, "value": value,
                "unit": "Å" if kind == "distance" else "°",
                "summary": f"{kind} {value:.2f}"
                           f"{' Å' if kind == 'distance' else '°'} "
                           f"on {need} atoms"}


    @staticmethod
    def _measure_value(sites_cart, kind: str, atoms: list) -> float:
        """The number the viewport's label will draw for these vertices.

        Distance in Angstrom; angle in degrees at the middle vertex; dihedral
        in degrees, signed (the label shows its magnitude)."""
        xyz = sites_cart.as_numpy_array()[list(atoms)]
        if kind == "distance":
            return float(np.linalg.norm(xyz[1] - xyz[0]))
        b1 = xyz[0] - xyz[1]
        b2 = xyz[2] - xyz[1]
        if kind == "angle":
            cos = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        # Delegated to scitbx so the reported value is cctbx's own convention —
        # the one the viewer's label draws. The hand-rolled normals formula this
        # replaced was sign-flipped against both (figfuzz l06).
        from scitbx.array_family import flex
        from scitbx.math import dihedral_angle
        value = dihedral_angle(
            flex.vec3_double([tuple(float(c) for c in p) for p in xyz]),
            deg=True)
        return float(value) if value is not None else float("nan")


    def clear_measurements(self) -> None:
        session = self.active_model_session()
        if session is not None:
            session.clear_primitives()


    def _clashes_work(self, name: str, src_mid, analysis) -> None:
        """The all-atom contacts check (background thread): add hydrogens to the model
        (reduce2), register the result as a new object, hide the original, and draw
        probe2 contacts + clashes as two independently toggleable overlays.

        With real hydrogens probe2 decides overlaps from actual H positions and
        directionality — the MolProbity-approved path — so no heavy-atom heuristics
        are needed. Runs on the validation thread (see :meth:`run_validation`);
        adding the model object is marshalled back to the GUI thread.
        """
        from .hydrogens import hydrogens_available
        from .live import LiveSession, PROBE_CLASHES, PROBE_CONTACTS

        if not hydrogens_available():
            self._status("reduce2 needs the monomer library (set MMTBX_CCP4_MONOMER_LIB)")
            return
        try:
            self._status(f"adding hydrogens to {name} (reduce2)…")
            hmodel = analysis.hydrogenated()
        except Exception as exc:  # pragma: no cover - reduce2/runtime errors
            self._status(f"reduce2 failed: {exc}")
            return

        box: dict = {}
        ready = threading.Event()

        def add_on_main():
            hsession = LiveSession.from_cctbx_model(hmodel)
            # Ball-and-stick so the placed hydrogens and the clash spikes are
            # actually visible (a cartoon ribbon would hide both).
            box["mid"] = self._add_model(hsession, f"{name} + H", rep="ball-and-stick")
            box["session"] = hsession
            self.set_model_visible(src_mid, False)  # hide the H-less original
            # _add_model made the sibling active — but the object the caller
            # validated (and whose entry caches the validation results that
            # set_validation_markers draws) is the original. Keep it active so
            # a check does not silently re-point "the model" out from under
            # whoever asked for it.
            self.set_active_model(src_mid)
            ready.set()

        self.bridge.run_on_main.emit(add_on_main)
        ready.wait()
        hsession, hmid = box["session"], box["mid"]

        try:
            self._status("running probe2 on the hydrogenated model…")
            contacts, clashes = analysis.probe_dots_split()  # cached; free after a hotspot run
        except Exception as exc:  # pragma: no cover - probe/runtime errors
            self._status(f"probe failed: {exc}")
            return

        hentry = self._model_entry(hmid)
        if hentry is not None:  # cache so the toggles redraw without re-running probe
            hentry["probe_dots"] = {PROBE_CONTACTS: contacts, PROBE_CLASHES: clashes}
        hsession.show_probe_dots(contacts, channel=PROBE_CONTACTS)
        hsession.show_probe_dots(clashes, channel=PROBE_CLASHES)
        self._status(f"{name} + H: {len(clashes)} clashes, {len(contacts)} contact dots")
        self.bridge.analysis_ready.emit(hmid)


    def set_probe_channel(self, channel: int, visible: bool) -> None:
        """Toggle a probe overlay (contacts/clashes), redrawing from the dots cached
        by the last analysis (no probe re-run).

        Applies to every model session, not just the active one's: the overlay
        toggle must not go dead because the user activated a different object —
        hiding clears the channel everywhere (a no-op where it was not drawn),
        showing redraws whichever models hold cached dots for it."""
        for entry in self._models:
            dots = (entry.get("probe_dots") or {}).get(channel)
            if visible:
                if dots:
                    entry["session"].show_probe_dots(dots, channel=channel)
            else:
                entry["session"].clear_probe_dots(channel=channel)


    def run_validation(self, keys=None, clashes: bool = False,
                       hotspots: bool = False) -> None:
        """Run the checked MolProbity validations on the active model as one background
        job, handing results to the Validation tab as each lands.

        ``keys`` selects the per-residue validators (``None`` runs every registered
        one; an empty list runs none); ``clashes`` appends the all-atom contact
        analysis (reduce2 + probe2), by far the heaviest check and the one that adds a
        hydrogenated sibling object; ``hotspots`` appends the severity aggregate over
        the checks themselves, which is nearly free once they have run and is why it
        belongs in this queue rather than behind a button of its own. Everything checked runs sequentially on one
        thread because the checks share the per-model analysis cache, whose expensive
        lazy steps (reduce2, probe) are not re-entrant — two concurrent checks would
        race to hydrogenate the same model.
        """
        entry = self._model_entry(self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        model = getattr(entry["session"], "model", None)
        if model is None:
            raise ValueError("the active object has no cctbx model")
        if keys is not None and not keys and not clashes and not hotspots:
            raise ValueError("no validations checked")
        mid, name = entry["id"], entry["name"]
        # reduce2 and probe are the two expensive steps in the whole validation stack, and
        # the hotspot score needs the same two. Everything goes through the shared analysis
        # so whichever check runs first pays and the rest are nearly free.
        analysis = self._model_analysis(entry)

        def work():
            if keys is None or keys:
                self._validation_work(mid, name, model, analysis, keys)
            if hotspots:
                self._hotspots_work(mid, name, model, analysis)
            if clashes:
                self._clashes_work(name, mid, analysis)

        self.run_background(work, name="pxviewer-validation", label="Running validation")
        self._status("validating…")


    def _validation_work(self, mid, name, model, analysis, keys) -> None:
        """The per-residue validator pass (background thread): run the selected
        validators, cache the results on the model entry, emit them to the GUI thread
        via ``validation_ready``, and score hotspots from the same analysis."""
        from . import validation

        try:
            self._status(f"validating {name}…")
            results = validation.run_all(model, analysis, keys=keys)
        except Exception as exc:  # pragma: no cover - validator/runtime errors
            self._status(f"validation failed: {exc}")
            return
        ventry = self._model_entry(mid)
        if ventry is not None:  # cache so marker toggles redraw without re-running
            ventry["validation"] = {r.key: r for r in results}
            self._mark_validated(ventry)  # fingerprint the coordinates these describe
        total = sum(len(r.markup) for r in results)
        self._status(f"{name}: {len(results)} validators, {total} markers")
        self.bridge.validation_ready.emit((mid, results))
        self._refresh_validation_staleness()  # fresh results: clear any stale warning

        # The Hotspots sub-tab is deliberately NOT filled here. It used to be, on the
        # grounds that the analysis was already paid for -- but hotspots is now a box in
        # the same queue, and quietly producing results for a check the user did not
        # tick makes the queue lie about what it ran.

    def set_validation_markers(self, key: str, visible: bool) -> None:
        """Toggle a validator's MolProbity markup on the active model, redrawing from
        the results cached by the last :meth:`run_validation` (no re-run). Each
        validator draws on its own channel (:func:`validation.channel_for`)."""
        from . import validation

        entry = self._model_entry(self._active_model_id)
        if entry is None:
            return
        session = entry["session"]
        channel = validation.channel_for(key)
        result = (entry.get("validation") or {}).get(key)
        if visible and result is not None and result.markup:
            session.show_markup(channel, result.markup)
        else:
            session.clear_markup(channel)
            if visible and (result is None or not result.markup):
                # Distinguish "ran, found nothing" from a broken toggle —
                # otherwise a silent no-op reads the same (figfuzz j07).
                self._status(f"{key} has nothing to draw"
                             + ("" if result is not None else " (not run yet)"))


    def set_tug_into_density(self, enabled: bool) -> None:
        """Whether a drag pulls into the map as well as against the geometry.

        Geometry alone can only deform; the map is what lets a drag *correct* something,
        since it is the only thing that knows where the atoms actually belong. Takes
        effect on the next drag — a drag already running keeps what it started with.
        """
        self._tug_into_density = bool(enabled)


    def set_tug_diagnostics(self, enabled: bool) -> None:
        """Per-gesture reporting in the status line: grabs that found no atom, and
        grab-to-first-motion latency with the zone-build cost — so a session where
        drags feel dead can say WHERE the time goes. Persisted."""
        self._tug_diagnostics = bool(enabled)
        self._settings.setValue("drag/diagnostics", "true" if enabled else "false")


    def set_tug_strength(self, strength: float) -> None:
        """How hard a drag pulls, as a multiple of the default (weight scales with
        ``strength``; the pull restraint's sigma is TUG_SIGMA/sqrt(strength)). Takes
        effect on the next drag — a drag already running keeps what it started with.
        Persisted."""
        self._tug_strength = max(0.05, float(strength))
        self._settings.setValue("drag/pull_strength", self._tug_strength)


    def set_tug_settle_seconds(self, seconds: float) -> None:
        """How long the post-release wind-down plays, in seconds (0 freezes at once).

        The settle minimization itself converges in a fraction of a second; this is the
        watchable decay it is stretched over — the knob for how long the model visibly
        keeps relaxing after a drag lets go. Persisted."""
        self._tug_settle_seconds = max(0.0, float(seconds))
        self._settings.setValue("drag/settle_seconds", self._tug_settle_seconds)


    def set_tug_continuous(self, enabled: bool) -> None:
        """Whether the minimizer keeps running for the whole hold, or settles per move.

        Off, a drag is a series of nudges: each pointer move relaxes the zone toward it
        and stops. On, the minimizer never stops while the button is down — the target
        moves under it and the neighborhood keeps settling even when the pointer is
        still, which is what lets it flow into density rather than only bend. Next drag.
        """
        self._tug_continuous = bool(enabled)


    def set_tug_scope(self, *, mode: Optional[str] = None,
                      radius: Optional[float] = None, flank: Optional[int] = None) -> None:
        """Set what a drag lets move: a ``sphere`` of ``radius`` A, or a residue stretch of
        ``flank`` residues each side (``mode='residues'``, ``flank=0`` for a single residue).
        Takes effect on the next drag; a drag already running keeps the scope it began with.
        """
        if mode is not None:
            self._tug_scope["mode"] = mode
        if radius is not None:
            self._tug_scope["radius"] = float(radius)
        if flank is not None:
            self._tug_scope["flank"] = int(flank)


    def set_update_maps_after_drag(self, enabled: bool) -> None:
        """Whether a settled drag re-phases the whole-structure maps. Takes effect at once.

        Off, the maps stay as they were until **Update maps** on the reflections' Appearance
        pane — which is the old behavior, kept for anyone fitting a long way from the data
        who does not want the recompute between pulls.
        """
        self._maps_after_drag = bool(enabled)
        if not enabled and self._drag_map_timer is not None:
            self._drag_map_timer.stop()  # a pending re-phase is no longer wanted


    def set_live_difference_map(self, enabled: bool) -> None:
        """Whether a drag streams a live mFo-DFc difference map around the dragged atom.

        On: each drag frame triggers a warm recompute of the difference density in a small
        window at the atom (see :class:`reflections.LiveDifferenceMap`), shown as a green/red
        box that flattens as the fit improves — honest feedback, since the difference map is
        about model-vs-data, not the model-echoing 2mFo-DFc. Only the local window updates;
        the whole-structure maps stay put until **Recompute**. Needs reflections phased
        against the model (Make maps first). Takes effect on the next drag.
        """
        self._live_diff = bool(enabled)
        if not enabled:
            self._clear_live_diff()


    def _group_reflection_path(self, gid: Optional[str]) -> Optional[str]:
        """The on-disk reflection file phased into group ``gid``, if any (for re-phasing)."""
        if gid is None:
            return None
        for entry in self._reflections:
            if entry.get("group") == gid and entry.get("r_work") is not None:
                path = entry["data"].path
                if path:
                    return path
        return None


    def _maybe_start_live_diff(self, mid: str, atom: int) -> None:
        """At a drag's start, arm the live difference map if it applies. Worker thread."""
        self._diff_ctx = None
        self._diff_atom = None
        if not self._live_diff:
            return
        entry = self._model_entry(mid)
        gid = entry.get("group") if entry else None
        refl_path = self._group_reflection_path(gid)
        if refl_path is None:
            return  # no phased reflections to recompute from — nothing to show
        self._diff_ctx = (gid, refl_path)
        self._diff_atom = atom
        self._diff_gen += 1
        # The live window is about to replace the STORY the static difference map
        # tells — and that map is stale from the first moved atom, a whole ghost of
        # red/green the user would have to mentally ignore. Suppress it (a viewer
        # render skip; the eye and the entry's visibility are untouched) until fresh
        # maps supersede the window or the live view is turned off.
        self._suppress_static_diff(gid)
        if self._diff_queue is None:
            import queue

            self._diff_queue = queue.Queue()
            threading.Thread(
                target=self._diff_worker, name="pxviewer-diffmap", daemon=True).start()


    def _queue_live_diff(self, coords: Any) -> None:
        """Hand the latest drag frame to the difference-map worker (drops older frames)."""
        if self._diff_ctx is None or self._diff_atom is None or self._diff_queue is None:
            return
        gid, refl_path = self._diff_ctx
        sites = np.ascontiguousarray(np.asarray(coords, dtype="float64")).reshape(-1, 3)
        if self._diff_atom >= len(sites):
            return
        center = tuple(sites[self._diff_atom])
        self._diff_queue.put((sites, center, gid, refl_path, self._diff_gen))


    def _diff_worker(self) -> None:
        """Recompute + stream the local difference map for the latest drag frame.

        Off the tug thread, so a recompute (tens of ms) never slows the drag; runs of frames
        collapse to the newest, and the scaled fmodel is built once per group and reused."""
        import queue

        while not self._stopped:
            try:
                item = self._diff_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            while True:  # collapse to the most recent frame
                try:
                    item = self._diff_queue.get_nowait()
                except queue.Empty:
                    break
            if item is None:
                continue
            sites, center, gid, refl_path, gen = item
            session = self._tug_session
            if session is None or not self._live_diff or gen != self._diff_gen:
                continue  # the drag ended (or a newer one began) before this frame ran
            # Pace the recompute. Each one is a full FFT holding real CPU (and, in
            # bursts, enough GIL to starve the tug worker and the socket threads —
            # grabs went dead while the window chased every frame). The window is
            # feedback, not simulation: a few refreshes a second reads the same.
            since = time.monotonic() - getattr(self, "_diff_last_recompute", 0.0)
            if since < 0.15:
                time.sleep(0.15 - since)
            self._diff_last_recompute = time.monotonic()
            try:
                if self._diff_engine is None or self._diff_engine_key != gid:
                    from .reflections import LiveDifferenceMap

                    mmm = self.group_mmm(gid)
                    model = mmm.model() if mmm is not None else None
                    if model is None:
                        continue
                    self._status("preparing live difference map…")
                    self._diff_engine = LiveDifferenceMap(model, refl_path)
                    self._diff_engine_key = gid
                box = self._diff_engine.recompute_local(center, radius=6.0, sites_cart=sites)
                if gen == self._diff_gen and self._live_diff:  # not superseded during the recompute
                    # The live window wears the paired difference map's own colors, so
                    # a user recoloring that map recolors the recalc too -- never a
                    # third color pair on screen.
                    diff_entry = next((v for v in self._volumes
                                       if v.get("group") == gid
                                       and v.get("negative_color")), None)
                    colors = ((diff_entry["color"], diff_entry["negative_color"])
                              if diff_entry else None)
                    # The window contours at the SAME level as the difference map it
                    # stands in for — both are sigma-scaled, so the slider means one
                    # thing. A hardcoded 3.0 disagreed the moment the user moved it.
                    level = float(diff_entry["iso"]) if diff_entry else 3.0
                    self._check_window_agreement(diff_entry, box, center)
                    session.show_map_box(
                        box, level=level, colors=colors,
                        style=diff_entry.get("style") if diff_entry else None)
                    self._diff_boxes += 1
            except Exception as exc:  # pragma: no cover - cctbx/runtime errors
                self._status(f"live difference map failed: {exc}")
                self._diff_ctx = None  # stop hammering a failing recompute for this drag


    def _stop_live_diff(self) -> None:
        """Stop streaming the live window; a standing window stays on screen."""
        self._diff_ctx = None
        self._diff_atom = None
        self._diff_gen += 1  # invalidate any recompute still in flight


    def _check_window_agreement(self, diff_entry, box, center) -> None:
        """Tripwire: the live window and the static difference map must describe the
        same physics wherever both are CURRENT. Sample both over the window's region
        and warn — with the number — when they disagree, so a real divergence in the
        field names itself instead of reading as a mysterious level change. Once per
        drag (the first frame), a few hundred samples, worker thread."""
        if diff_entry is None or self._diff_agreement_gen == self._diff_gen:
            return
        self._diff_agreement_gen = self._diff_gen
        try:
            from .volume_io import sample_at_sites

            mm = diff_entry["data"].map_manager
            if callable(mm):
                mm = mm()
            rng = np.random.default_rng(0)
            points = np.asarray(center, dtype=float) + rng.uniform(-4.5, 4.5, (400, 3))
            static_v = np.asarray(sample_at_sites(mm, points), dtype=float)
            window_v = np.asarray(sample_at_sites(box, points), dtype=float)
            ok = np.isfinite(static_v) & np.isfinite(window_v)
            if ok.sum() < 50:
                return
            static_v, window_v = static_v[ok], window_v[ok]
            # Only meaningful where there is SIGNAL: for a well-fitted model both
            # difference maps are noise, and two noise realizations decorrelate even
            # when both are honestly quiet — which made this warning fire spuriously.
            strong = (np.abs(static_v) > 2.0) | (np.abs(window_v) > 2.0)
            if strong.sum() < 30:
                return  # both maps are quiet here; nothing to disagree about
            spread = float(np.std(static_v[strong])) * float(np.std(window_v[strong]))
            corr = (float(np.corrcoef(static_v[strong], window_v[strong])[0, 1])
                    if spread > 1e-9 else 1.0)
            if corr < 0.85:
                self._status(
                    f"live window disagrees with the static difference map (r={corr:.2f})"
                    " — the static map is stale or mis-scaled; please report this number")
        except Exception:  # pragma: no cover - a diagnostic must never break the drag
            pass


    def _clear_live_diff(self) -> None:
        """Stop streaming the live difference map and remove the window from the viewport."""
        self._stop_live_diff()
        for entry in self._models:
            # Any session may have shown (and would replay) the window; clear them all.
            try:
                entry["session"].clear_map_box()
            except Exception:  # pragma: no cover - defensive
                pass
        self._restore_static_diff()


    def _suppress_static_diff(self, gid) -> None:
        """Hide the group's static difference map while the live window speaks for it."""
        control = self._control_session()
        if control is None:
            return
        for volume in self._volumes:
            if volume.get("group") != gid or not volume.get("negative_color"):
                continue
            if volume.get("visible") and volume["ref"] not in self._diff_suppressed:
                try:
                    control.set_volume_visible(volume["ref"], False)
                    self._diff_suppressed.append(volume["ref"])
                except Exception:  # pragma: no cover - defensive
                    pass


    def _restore_static_diff(self) -> None:
        """Show again what _suppress_static_diff hid — per the entry's own visibility."""
        control = self._control_session()
        refs, self._diff_suppressed = self._diff_suppressed, []
        if control is None:
            return
        for ref in refs:
            volume = next((v for v in self._volumes if v["ref"] == ref), None)
            if volume is None or not volume.get("visible"):
                continue  # removed, or the user hid it meanwhile: leave it be
            try:
                control.set_volume_visible(ref, True)
            except Exception:  # pragma: no cover - defensive
                pass


    # -- markers ---------------------------------------------------------

    def arm_marker(self) -> None:
        """Arm 'place a marker': the next click in the viewport drops a sphere and reports
        its 3D coordinate (see :meth:`_on_marker`). One-shot — the viewer disarms after
        the click. Rides the control session, so the marker is a scene-level point rather
        than tied to one model."""
        session = self._control_session()
        if session is None:  # blank canvas: fall back to the dummy the viewport connects to
            self._ensure_dummy_ws()
            session = self._control_session()
        if session is None:  # pragma: no cover - defensive
            self._status("could not arm — the viewport has no session")
            return
        self.ensure_atoms_shown()  # snapping to an atom needs the atoms visible, not a ribbon
        session.set_marker_mode(True)
        self._status("Click in the viewport to place a ligand marker…")


    def place_marker(self, position, atom=None) -> str:
        """Drop a marker at ``position`` (3-vector, Angstrom) — the scripting path
        to what :meth:`arm_marker` + a viewport click does. ``atom`` is the atom
        index the marker snapped to, if any (same contract as a click landing on
        one). Returns the new marker id (``'marker-N'``) for
        :meth:`set_marker_visible`/:meth:`remove_marker`."""
        self._on_marker(position, atom)
        return self._markers[-1]["id"]


    def _on_marker(self, position, atom) -> None:
        """A marker was placed: register it (so it appears in the Objects list as a handle)
        and draw it. ``atom`` is the picked atom index if the click landed on one, else
        None. Runs on a session's event-loop thread; the list/markup are cheap and safe."""
        self._marker_counter += 1
        point = [float(c) for c in position]
        # 'name' is what the object list shows: a ligand marker to the user, though the
        # marker machinery itself stays generic for other tools.
        self._markers.append({
            "id": f"marker-{self._marker_counter}",
            "name": f"Ligand marker {self._marker_counter}",
            "position": point, "atom": atom, "visible": True,
        })
        self._draw_markers()
        self._emit_loaded_changed()
        where = f" on atom {atom}" if atom is not None else ""
        self._status(
            f"Ligand marker at ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}){where} "
            f"— {len(self._markers)} placed")


    def _marker_entry(self, mid, *, missing_ok: bool = False):
        entry = next((m for m in self._markers if m["id"] == mid), None)
        if entry is None and mid is not None and not missing_ok:
            raise ValueError(f"no marker {mid!r} — "
                             f"status() lists the ids (e.g. 'marker-1')")
        return entry


    def _draw_markers(self) -> None:
        """(Re)draw every visible marker as a sphere, and tell the viewer where they are so a
        Shift-drag can grab one (see :meth:`_on_marker_move`)."""
        session = self._control_session()
        if session is None:
            return
        visible = [m for m in self._markers if m["visible"]]
        balls = [[m["position"], _MARKER_RADIUS] for m in visible]
        primitive = {"kind": "balls", "color": _MARKER_COLOR, "balls": balls}
        session.show_markup(_MARKER_CHANNEL, [primitive] if balls else [])
        session.set_markers(
            [{"id": m["id"], "position": m["position"]} for m in visible], _MARKER_RADIUS)


    def _on_marker_move(self, mid: str, position, final: bool) -> None:
        """A marker was Shift-dragged in the viewport: move it. Runs on a session thread.

        Moving a marker detaches it from any atom it had snapped to — it now sits wherever it
        was dragged, which is the whole point. The sphere follows every frame (cheap); the
        heavier refresh of the Objects list and Ligand panel waits for the drop (``final``),
        so a drag does not thrash them (and the Appearance pane does not flicker)."""
        entry = self._marker_entry(mid)
        if entry is None:
            return
        entry["position"] = [float(c) for c in position]
        entry["atom"] = None  # dragged off whatever it was snapped to
        self._draw_markers()
        if final:
            self._emit_loaded_changed()  # refresh the marker's coordinate + the ligand target
            p = entry["position"]
            self._status(
                f"Marker moved to ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")


    def set_marker_visible(self, mid: str, visible: bool) -> None:
        """Hide or show a placed marker — it stays in the Objects list, just
        undrawn, until :meth:`remove_marker`."""
        entry = self._marker_entry(mid)
        if entry is None or entry["visible"] == bool(visible):
            return
        entry["visible"] = bool(visible)
        self._draw_markers()
        self._emit_loaded_changed()


    def remove_marker(self, mid: str) -> None:
        """Unload a single marker (from the Objects list or its Remove)."""
        entry = self._marker_entry(mid)
        if entry is None:
            return
        self._markers.remove(entry)
        self._draw_markers()
        self._emit_loaded_changed()


    def clear_markers(self) -> None:
        """Remove every placed marker."""
        had = bool(self._markers)
        self._markers.clear()
        self._draw_markers()  # empty list -> clears the markup channel
        self._emit_loaded_changed()
        if had:
            self._status("Ligand markers cleared")


    def fit_ligand_at_marker(self, mid: str, code: str, *, fit: bool = True,
                             trials: int = 20) -> None:
        """Place a monomer-library ligand at a marker, and — if a map is paired with the
        active model and ``fit`` — settle it into that density with a large radius of
        convergence (explode-and-refine). The result is added as a standalone model, so it
        is fully editable (tug/minimize) and non-destructive. Runs on a background thread
        (the fit takes seconds); adds its result on the GUI thread."""
        from . import ligands

        code = (code or "").strip().upper()
        if not code:
            raise ValueError("enter a monomer code, e.g. GOL")
        if not ligands.available(code):
            raise ValueError(f"no monomer '{code}' in the library")
        self._place_ligand(
            mid, code,
            lambda position, cs: ligands.build_ligand_model(
                code, position, crystal_symmetry=cs),
            fit=fit, trials=trials)


    def fit_ligand_from_smiles_at_marker(self, mid: str, smiles: str, code: str, *,
                                         fit: bool = True, trials: int = 20) -> None:
        """Like :meth:`fit_ligand_at_marker` but for a ligand given as a SMILES string
        rather than a library code — rdkit embeds a 3D conformer and its geometry supplies
        both coordinates and (on-the-fly) restraints. ``code`` is the residue name the new
        object is filed under."""
        from . import ligands

        smiles = (smiles or "").strip()
        if not smiles:
            raise ValueError("enter a SMILES string, e.g. CCO")
        code = (code or "LIG").strip().upper()[:3] or "LIG"
        self._place_ligand(
            mid, code,
            lambda position, cs: ligands.build_ligand_from_smiles(
                smiles, code, position, crystal_symmetry=cs),
            fit=fit, trials=trials)


    def _place_ligand(self, mid: str, code: str, builder, *, fit: bool = True,
                      trials: int = 20) -> None:
        """Shared machinery behind the ligand actions: build the model with ``builder``
        (``builder(position, crystal_symmetry) -> model``) centered on the marker, fit it
        into the active model's density if asked, and add it as a standalone object — all
        off the GUI thread, the add marshalled back onto it."""
        from . import ligands

        marker = self._marker_entry(mid)
        if marker is None:
            return
        position = list(marker["position"])
        # Fitting needs the active model's paired map and its frame, so the ligand is
        # built and refined in the same coordinates the marker was placed in.
        map_data = self.map_for_model() if fit else None
        cs = None
        if map_data is not None:
            entry = self._model_entry(self._active_model_id)
            mmm = self.group_mmm(entry.get("group")) if entry else None
            cs = mmm.crystal_symmetry() if mmm is not None else None

        def work():
            # Failures propagate to run_background's wrapper — it records them
            # in _job_errors and warns, instead of a status line the caller
            # (or an MCP client watching job_errors) could never see (i07).
            self._status(f"building {code}…")
            model = builder(position, cs)
            if map_data is not None:
                self._status(f"fitting {code} into density ({trials} trials)…")
                ligands.fit_into_density(
                    model, map_data, resolution=3.0, number_of_trials=trials)

            def add_on_main():
                from .live import LiveSession
                from . import ligands

                session = LiveSession.from_cctbx_model(model)
                new_mid = self._add_model(session, name=f"{code} (ligand)")
                # Keep the ligand's own geostd restraint CIF on its entry so it can be saved
                # (SMILES ligands carry an rdkit-provenance CIF; library ones the geostd file).
                entry = self._model_entry(new_mid)
                if entry is not None:
                    entry["restraints_cif"] = ligands.restraints_cif_text(model)
                # The marker has done its job — it becomes the ligand object. Consume it so
                # it doesn't linger in the object list alongside the model it produced.
                self.remove_marker(mid)
                self.bridge.ligand_placed.emit()  # so the panel clears its inputs
                self._status(
                    f"placed {code}"
                    + (" and fitted into density" if map_data is not None else "")
                    + f" at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")

            self.bridge.run_on_main.emit(add_on_main)

        self.run_background(work, name="pxviewer-ligand", label="Building ligand")


    def _on_tug(self, mid: str, action: str, atom: int, target) -> None:
        """A drag in the viewport: queued, never served here.

        This runs on the session's event loop — the one thread that both reads the
        viewer's messages and writes coordinates back to it. Building a drag's zone
        takes the better part of a second, and stalling that loop for it means the
        viewer goes deaf and blind exactly as the drag begins.
        """
        if action == "arm":
            # Refine drag enabled: a drag is imminent. Do exactly what Pause does —
            # stop any running minimization (same call, same message) — so by the time the
            # pointer grabs an atom the model is free. Non-blocking (this is the socket
            # thread); the drag's begin waits for the run to actually finish.
            if not self._minimize_idle.is_set():
                self.stop_minimization()
            # And make sure this model's restraints exist before the grab needs them. Nearly
            # always already warm from load; this covers a model that arrived another way.
            self._warm_restraints(mid)
            return
        if self._tug_queue is None:
            import queue

            self._tug_queue = queue.Queue()
            threading.Thread(target=self._tug_worker, name="pxviewer-tug", daemon=True).start()
        self._tug_queue.put((action, mid, atom, target))


    def _tug_worker(self) -> None:
        """Serve drags off the socket's thread.

        When a drag is running in continuous mode the loop does not block: it keeps
        stepping the minimizer between messages, so the model settles even while the
        pointer is still. Otherwise it blocks until the next message, since there is
        nothing to do between them. Runs of pointer targets collapse to the newest — the
        pointer outruns cctbx, and every target but the last is somewhere it has left.
        """
        import queue

        while not self._stopped:
            free_running = self._tug is not None and self._tug_continuous
            waiting = []
            if not free_running:
                try:
                    waiting.append(self._tug_queue.get(timeout=0.2))
                except queue.Empty:
                    continue
            while True:
                try:
                    waiting.append(self._tug_queue.get_nowait())
                except queue.Empty:
                    break
            collapsed = _collapse_moves(_thin_gestures(waiting))
            for i, (action, mid, atom, target) in enumerate(collapsed):
                try:
                    # An 'end' with more messages already drained behind it (the user
                    # grabbed again mid-batch) skips the settle wind-down: the queue
                    # check inside the playback cannot see messages that are in this
                    # batch rather than still queued, and rapid grab/release bursts
                    # serialized into seconds of mandatory playback per release.
                    self._serve_tug(mid, action, atom, target,
                                    settle_animation=(i == len(collapsed) - 1))
                except Exception as exc:  # pragma: no cover - defensive
                    self._status(f"drag failed: {exc}")
                    self._end_tug()
            if self._tug is not None and self._tug_continuous:
                self._tug_relax()


    def _serve_tug(self, mid: str, action: str, atom: int, target,
                   settle_animation: bool = True) -> None:
        """Apply one drag message. On the tug worker's thread."""
        entry = self._model_entry(mid)
        if entry is None:
            # The model was unloaded mid-drag. Nothing left to move — and if this was the
            # model being dragged, close the drag out so its Tug (which holds the now-gone
            # model) is not left dangling for the free-run loop to keep stepping.
            if self._tug is not None and self._tug_model == mid:
                self._end_tug()
            return
        session = entry["session"]
        if action == "begin":
            self._end_tug()  # a drag that never got its mouseup
            model = getattr(session, "model", None)
            if model is None:
                return
            self.ensure_atoms_shown(mid)  # dragging an atom needs to see the atoms, not a ribbon
            from .geometry import monomer_library_available

            if not monomer_library_available():
                self._status("dragging atoms needs the monomer library")
                return
            # The drag takes the model over from any running minimization. Stop it and wait
            # for its thread to let go before we build restraints on the same model, so the
            # two never write the coordinates at once. It converges in well under a second;
            # the timeout is a safety cap, not the expected wait. (Shift-keydown usually
            # started this stop already, in the "arm" branch above.)
            if not self._minimize_idle.is_set():
                self._minimize_stop.set()
                self._minimize_idle.wait(timeout=2.0)
            from .tug import Tug

            # Say so before building, not after: on a cold model this is the one place the
            # user is left waiting, and silence there reads as a dropped click.
            if not model.restraints_manager_available():
                self._status("preparing restraints for dragging…")
            zone_build_started = time.monotonic()
            # A repeat grab in the same neighborhood reuses the last zone: the
            # restraints sub-selection depends only on topology, and rebuilding it
            # is the occasional >1 s a grab goes dead for. The signature pins
            # everything a rebind cannot refresh — the scope, the restraints
            # manager identity (edits rebuild it), and the map object (re-phasing
            # replaces it).
            scope = self._tug_scope
            map_data = self.map_for_model(mid) if self._tug_into_density else None
            selection = None
            if scope["mode"] == "selection":
                with self._scene_lock:
                    selection = list(self._scene_selection.get(mid, []))
            signature = (mid, scope["mode"], scope["radius"], scope["flank"],
                         tuple(selection or ()),
                         id(model.get_restraints_manager()), id(map_data))
            from .tug import TUG_SIGMA as _TUG_SIGMA

            pull_sigma = _TUG_SIGMA / math.sqrt(self._tug_strength)
            if self._tug_cache is not None and self._tug_cache[0] == signature:
                cached = self._tug_cache[1]
                if cached.rebind(atom, pull_sigma=pull_sigma):
                    self._tug = cached
                    self._tug_cache = None
            if self._tug is not None:
                pass  # rebound in ~ms; the timing below reports it honestly
            else:
                self._tug_cache = None
            if scope["mode"] == "selection" and not selection:
                self._status("select atoms first to drag by selection")
                self._tug = None
                return
            try:
                if self._tug is None:
                    # Against the pre-warm (see _warm_restraints): if one is in flight
                    # for this model, wait rather than building alongside it.
                    with self._restraints_lock:
                        self._tug = Tug(
                            model, atom,
                            mode=scope["mode"], radius=scope["radius"],
                            flank=scope["flank"], selection=selection,
                            map_data=map_data,
                            # Strength scales the pull's WEIGHT (weight ~ 1/sigma^2).
                            pull_sigma=pull_sigma)
            except Exception as exc:  # pragma: no cover - restraints/runtime errors
                self._status(f"could not start dragging: {exc}")
                self._tug = None
                return
            self._tug_signature = signature
            self._tug_model = mid
            self._tug_session = session
            self._tug_last = None
            self._tug_last_push = 0.0
            self._tug_zone_build_ms = int((time.monotonic() - zone_build_started) * 1000)
            self._tug_begin_walltime = time.monotonic()
            self._maybe_start_live_diff(mid, atom)  # arm the live difference map, if on
            self._status(f"dragging atom {atom} — {self._tug.zone_size} atoms giving way")
            return

        if action == "miss":
            self._tug_misses += 1
            if self._tug_diagnostics:
                self._status(f"grab found no atom under the pointer "
                             f"({self._tug_misses} missed this session)")
            return
        if self._tug is None or self._tug_model != mid:
            return
        if action == "move" and target is not None:
            if self._tug_continuous:
                self._tug.set_target(target)  # the free-run loop does the stepping
            else:
                self._push_tug(self._tug.move_to(target))
        elif action == "end":
            # Only the LAST release settles. The settle is an LBFGS to convergence —
            # seconds of compute on a big zone — and it used to run once per queued
            # release, serialized on this worker: a burst of quick touches kept it
            # relaxing OLD drags for 10-20 s after the user's last one. A release with
            # more input already waiting (later messages in this batch, or fresh ones
            # in the queue) just lets go where it stands; the next drag's own
            # minimizer takes over immediately.
            rushed = (not settle_animation
                      or (self._tug_queue is not None and not self._tug_queue.empty()))
            if not rushed:
                self._settle_tug()
            # Streaming stops, but the WINDOW stays: the object difference map is stale
            # (it describes the pre-drag model until the maps are recomputed), and the
            # settled window is the freshest local truth. It is replaced by the next
            # drag, and cleared when the maps actually update (see update_maps) or the
            # live-map toggle goes off.
            self._stop_live_diff()
            self._end_tug()
            self._invalidate_model_state(entry)  # stale: the atoms just moved
            self._refresh_validation_staleness()  # warn if this outran a validation run
            self._queue_post_drag_map_update(entry)  # and so is the density, if phased


    def _tug_relax(self) -> None:
        """One free-running step, for continuous mode. On the worker's thread.

        A faint shake as it minimizes, so a held drag stays in motion rather than
        freezing at the first minimum — kept subtle; the clean stop is the settle below.
        """
        from .tug import JIGGLE_AMPLITUDE, JIGGLE_STEPS

        try:
            self._push_tug(self._tug.step(jiggle=JIGGLE_AMPLITUDE, steps=JIGGLE_STEPS))
        except Exception as exc:  # pragma: no cover - runtime errors
            self._status(f"drag failed: {exc}")
            self._end_tug()


    def _settle_tug(self, animate: bool = True) -> None:
        """After release, relax the fragment to rest before letting go of it.

        Fling an atom and let go and it should visibly come to rest, not stop dead
        wherever it happened to be — which is what makes it possible to tell a settled
        fragment from a broken or frozen one. The pull is kept on at its last target so
        the atom stays where you left it while the neighborhood relaxes around it; the
        loop ends when the motion dies away, and the resting frame is forced out past the
        pacing so the final position always shows. On the worker's thread.
        """
        if self._tug is None:
            return
        try:
            settle_started = time.monotonic()
            trajectory: list = []
            self._tug.settle(on_frame=lambda c: trajectory.append(c.copy()))
            compute_took = time.monotonic() - settle_started
        except Exception as exc:  # pragma: no cover - runtime errors
            self._status(f"settle failed: {exc}")
            return
        if not trajectory:
            return
        # The minimization converges in a fraction of a second — far too fast to see. Play
        # it back in real time so the fling visibly winds down to rest, thinning the many
        # optimizer states to what shows at the frame rate. The playback yields to
        # everything more important than a flourish: a new grab, the Pause button, and
        # shutdown — it must never be the thing the user cannot interrupt.
        # The knob is TOTAL time from release to freeze, wall clock. The settle
        # compute above is silent but already spent part of the budget, so the
        # playback gets what remains — and frames are scheduled against the clock
        # rather than accumulating sleeps, because per-frame push overhead on top of
        # a fixed sleep stretched a 0.4 s setting into ~1.2 s of visible motion.
        play = max(0.0, self._tug_settle_seconds - compute_took)
        if animate and play > 0 and not self._minimize_stop.is_set() and not self._stopped:
            shown = min(len(trajectory), max(2, int(play / _TUG_PUSH_INTERVAL)))
            indices = np.linspace(0, len(trajectory) - 1, shown).astype(int)
            playback_started = time.monotonic()
            for k, i in enumerate(indices):
                if self._tug_queue is not None and not self._tug_queue.empty():
                    break  # the user grabbed again; no waiting out the wind-down
                if self._minimize_stop.is_set() or self._stopped:
                    break  # Pause (or shutdown) trumps the flourish
                due = playback_started + play * (k / max(1, shown - 1))
                wait = due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._push_tug(trajectory[i], force=True)
        self._push_tug(trajectory[-1], force=True)  # the resting position, always shown


    def _queue_post_drag_map_update(self, entry) -> None:
        """Ask for a re-phase now that a drag has settled. Worker thread.

        A minimization already chains this (see minimize_model): the moment the model moves
        the maps describe atoms that are no longer there, and the difference map especially
        becomes the answer to a question about the previous model. A drag moves the model
        for exactly the same reason, and leaving it out is what sent the user off to the
        reflections' Appearance pane and back after every pull.

        Only queued here, never run: re-phasing is GUI-thread work, and it is debounced
        (see :meth:`_schedule_map_update`) because drags come in bursts and each recompute
        is two transforms over the whole structure — one per release would spend the
        machine on maps the next pull invalidates before they land.
        """
        if not self._maps_after_drag:
            return
        try:
            reflections = self.reflections_for_model(entry["id"])
        except ValueError:
            return  # the model was removed mid-drag; there is nothing to re-phase
        if reflections is None:
            return  # nothing phased against this model; there are no maps to bring up to date
        self.bridge.run_on_main.emit(
            lambda rid=reflections["id"]: self._schedule_map_update(rid))


    def _schedule_map_update(self, rid: str) -> None:
        """(Re)start the post-drag re-phase timer, so a burst collapses to one. GUI thread."""
        from PySide6.QtCore import QTimer

        if self._stopped:
            return
        if self._drag_map_timer is None:
            # Parentless: DesktopApp is not a QObject; the attribute is what keeps it alive.
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(self._fire_map_update)
            self._drag_map_timer = timer
        self._drag_map_rid = rid
        self._drag_map_timer.start(_DRAG_MAP_UPDATE_MS)


    def _fire_map_update(self) -> None:
        """Re-phase, unless the user has started dragging again. GUI thread.

        A drag running right now means the timer's quiet spell was an illusion — the grab
        landed between the release and this call — so wait out another one rather than
        recomputing against a model that is moving under the recompute.
        """
        if self._stopped or not self._maps_after_drag:
            return
        if self._tug is not None:
            self._drag_map_timer.start(_DRAG_MAP_UPDATE_MS)
            return
        self._update_maps_if_live(self._drag_map_rid)


    def _push_tug(self, coords, force: bool = False) -> None:
        """Stream a drag frame — paced, as a delta, and never a repeat of the last.

        Only the drag's zone can have moved, so the frame is sent as just those atoms and
        the viewer patches its held conformation (see ``LiveSession.push``). That keeps a
        frame's cost proportional to the zone rather than to the structure, which is what
        the zone-limited minimizer already does on this side.

        Still paced: the viewer now drops stale frames rather than queueing them, so
        arriving early is harmless, but computing frames nobody will draw is wasted work
        the drag itself wants. A frame identical to the last is dropped outright — a
        settled geometry drag would otherwise send the same conformation over and over.
        ``force`` bypasses the pacing (not the de-dup) so a final resting frame is never
        dropped for arriving too soon.
        """
        if self._tug_session is None:
            return
        if self._tug_last is not None and np.array_equal(coords, self._tug_last):
            return
        now = time.monotonic()
        if not force and now - self._tug_last_push < _TUG_PUSH_INTERVAL:
            return  # too soon; the next frame supersedes this one
        self._tug_last = coords
        self._tug_last_push = now
        # The zone is the only thing that can have moved; everything else is untouched.
        zone = self._tug.indices if self._tug is not None else None
        self._tug_session.push(coords, changed=zone)
        if self._tug_begin_walltime is not None:
            latency = int((time.monotonic() - self._tug_begin_walltime) * 1000)
            self._tug_begin_walltime = None
            if self._tug_diagnostics:
                self._status(f"grab → first motion {latency} ms "
                             f"(zone build {getattr(self, '_tug_zone_build_ms', 0)} ms)")
        self._queue_live_diff(coords)  # follow the drag with a local difference map, if on


    def _end_tug(self) -> None:
        """Close out a drag, if one is running. Idempotent."""
        if self._tug is None:
            return
        try:
            self._tug.finish()
            # Park the finished zone for the next grab: a rebind in the same
            # neighborhood is milliseconds where a rebuild is the occasional >1 s.
            if getattr(self, "_tug_signature", None) is not None:
                self._tug_cache = (self._tug_signature, self._tug)
        except Exception:  # pragma: no cover - defensive
            pass
        self._tug = None
        self._tug_model = None
        self._tug_session = None
        self._tug_last = None


    def compute_real_space_difference(self, mid: Optional[str] = None) -> None:
        """Subtract the model's predicted density from its map, and add the result.

        The difference map you can have without reflections — which in cryo-EM is the
        only kind there is. Read exactly like mFo-DFc (green: density the model does not
        account for; red: model with nothing under it), and styled as one, so a reader
        who knows the convention needs to learn nothing new.

        Runs on a background thread: generating the model's density and fitting it to
        the map takes seconds on a real box.
        """
        entry = self._model_entry(mid or self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        model = getattr(entry["session"], "model", None)
        mmm = self.model_mmm(entry["id"])
        if model is None or mmm is None or mmm.map_manager() is None:
            raise ValueError("a real-space difference needs a model paired with a map — "
                             "fetch them together, or pair them")
        name, gid = entry["name"], entry.get("group")

        def work():
            from .volume_io import VolumeData, real_space_difference_map

            self._status(f"computing the real-space difference for {name}…")
            try:
                d_min = float(mmm.resolution())
            except Exception:  # pragma: no cover - estimator quirks
                d_min = 3.0
            try:
                data = real_space_difference_map(
                    VolumeData.from_map_manager(mmm.map_manager()), model, d_min=d_min)
            except Exception as exc:  # pragma: no cover - cctbx/runtime errors
                self._status(f"real-space difference failed: {exc}")
                return
            added = threading.Event()

            def add_on_main():
                try:
                    from .reflections import MAP_STYLE

                    color, iso, negative = MAP_STYLE[True]   # a difference map is a
                    self._add_volume(                          # difference map, however made
                        data, f"{name} · real-space difference", group=gid,
                        color=color, iso=iso, negative_color=negative, style="mesh")
                    self._status(f"{name}: real-space difference at ±{iso:g}σ "
                                 "(green unexplained density, red unsupported model)")
                finally:
                    added.set()

            self.bridge.run_on_main.emit(add_on_main)
            added.wait()

        self.run_background(work, name="pxviewer-rs-difference",
                            label="Real-space difference")


    def model_mmm(self, mid: Optional[str] = None) -> Any:
        """The cctbx ``map_model_manager`` a model belongs to, or None (active by default).

        The console's way in to cctbx. :meth:`group_mmm` wants a *group* id, which
        nothing public hands out — so the console's own banner ended up advertising
        ``app.group_mmm(g)`` with no way to obtain ``g``. This asks the question people
        actually have: give me the manager for this model.
        """
        entry = self._model_entry(self._active_model_id if mid is None else mid)
        return self.group_mmm(entry.get("group")) if entry is not None else None


    def map_for_model(self, mid: Optional[str] = None) -> Any:
        """The map this model is paired with — its cctbx ``flex.double`` density
        array, or None when unpaired. (It is the data, not a handle: for the
        paired volume's *id* and metadata, see :meth:`alignable` or the
        ``volumes`` block of ``status()``, matching on ``group``.)

        Whether a model and a map go together is cctbx's call, not ours. They are paired
        exactly when they share a ``map_model_manager``, which is what puts them in a
        common frame — the thing the minimizer's density interpolation assumes. So this
        asks the group for its manager and takes the map from there.

        There is deliberately no logic here that inspects two independently-loaded
        objects and decides they look compatible. Pairing them is a real operation (it
        can shift a model), not an observation, and cctbx's own guess at it —
        ``DataManager.get_map_model_manager``'s ``guess_files`` — is just "one model and
        one map, so probably". Getting it wrong refines a model into someone else's
        density. To pair unpaired objects, build a manager for them explicitly.
        """
        entry = self._model_entry(self._active_model_id if mid is None else mid)
        if entry is None:
            return None
        mmm = self.group_mmm(entry.get("group"))
        if mmm is None:
            return None
        mm = mmm.map_manager()
        return mm.map_data() if mm is not None else None


    def minimize_model(self, *, use_map: bool = False) -> None:
        """Minimize the active model, streaming the run into the viewport.

        Onto its geometry restraints, or with ``use_map`` also into the density of a
        map loaded alongside it (see :meth:`map_for_model`). cctbx hands us every
        intermediate conformation (see :mod:`pxviewer.minimize`), and each one goes
        straight out on the live coordinate wire — so the model is seen relaxing rather
        than jumping to the answer. Runs on a background thread; ``session.push`` is
        thread-safe, and :meth:`stop_minimization` can halt it. The model itself ends up
        minimized, so the tables, validation and Write all see the new coordinates.

        The run is *continuous*: a single convergent minimization is over in ~1 s — too
        fast to watch or interrupt — so once the model reaches its minimum the run stays
        on (the model held there, no CPU spent) until the user ends it with
        :meth:`stop_minimization` or by starting a drag. That keeps a steady window to
        stop it or hand the model to a drag, and keeps the running indicator honest.
        """
        entry = self._model_entry(self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        session = entry["session"]
        model = getattr(session, "model", None)
        if model is None:
            raise ValueError("the active object has no cctbx model")
        name = entry["name"]

        map_data = self.map_for_model(entry["id"]) if use_map else None
        if use_map and map_data is None:
            raise ValueError(
                "this model is not paired with a map — load the two together to pair them")

        self._minimize_stop.clear()
        self._minimize_idle.clear()
        self._minimize_mid = entry["id"]

        def work():
            from .geometry import monomer_library_available
            from .minimize import minimize

            if not monomer_library_available():
                self._status("minimization needs the monomer library (set MMTBX_CCP4_MONOMER_LIB)")
                self._minimize_idle.set()
                self._minimize_mid = None
                self.bridge.minimizing_changed.emit(False)
                return
            first = last = None
            shown = 0
            updated_at_hold = False
            # The static difference map describes the model as it was when last phased;
            # from the first minimization step it is a ghost, exactly as during a drag.
            # Suppressed for the run; the post-run (or at-hold) re-phase restores it.
            reflections = self.reflections_for_model(entry["id"])
            if reflections is not None:
                gid = entry.get("group")
                self.bridge.run_on_main.emit(lambda: self._suppress_static_diff(gid))
            try:
                self._status(
                    f"refining {name}{' into the map' if map_data else ''}… "
                    "(press Stop, or enable Refine drag, to finish)")
                # Pace the frame stream, exactly as a drag does (see _push_tug): cctbx emits
                # states far faster than the viewer can draw them, and pushing every one
                # floods the render so the frames back up and the model keeps *moving after
                # the run has stopped* — which reads as "Stop did nothing". Capping the rate
                # keeps the picture current, so the motion tracks the compute (and the Stop
                # button, which already reflects it). The settled frame of each cycle is
                # force-pushed below, so pacing never drops the resting conformation.
                last_push = [0.0]

                def stream(coords):
                    now = time.monotonic()
                    if now - last_push[0] >= _TUG_PUSH_INTERVAL:
                        last_push[0] = now
                        session.push(coords)

                # Continuous mode: a single convergent run is over in ~1 s — too fast to
                # watch or interrupt. So keep the run alive until the user ends it, giving a
                # steady window to Stop or hand the model to a drag. Refine in cycles (each
                # thins its stream: cctbx emits a state per evaluation, far more than the
                # viewport shows); once a cycle no longer improves the geometry the model is
                # at its minimum, so HOLD the run open rather than spin cctbx on a model that
                # will not move — the run stays "on" (and stoppable) without burning a core.
                while not self._minimize_stop.is_set():
                    stats = minimize(
                        model, map_data=map_data, on_state=stream,
                        should_stop=self._minimize_stop.is_set, stride=4)
                    if first is None:
                        first = stats
                    prev, last = last, stats
                    shown += stats["n_sent"]
                    session.push(model.get_sites_cart().as_numpy_array())  # the resting frame
                    if stats["stopped"]:
                        break
                    # Converged when the model's geometry stops getting better: a cycle that
                    # improves neither bond nor angle rmsd is at the minimum. (A displacement
                    # test is fooled — atoms keep sliding in flat directions as each cycle
                    # rebuilds restraints, without the geometry actually improving.)
                    settled = (stats["bonds_after"] >= (prev["bonds_after"] if prev else 1e9) - 1e-4
                               and stats["angles_after"] >= (prev["angles_after"] if prev else 1e9) - 1e-2)
                    if settled:
                        # The model will not move again: re-phase NOW, not when the hold
                        # ends. A held run used to sit for minutes over maps phased
                        # against the pre-run model — a wall of stale red/green that
                        # silently cleared on the next touch.
                        if reflections is not None:
                            updated_at_hold = True
                            self.bridge.run_on_main.emit(
                                lambda rid=reflections["id"]: self._update_maps_if_live(rid))
                        self._status(
                            f"{name} refined (bond rmsd {stats['bonds_after']:.3f}) — holding; "
                            "press Stop or enable Refine drag to finish")
                        while not self._minimize_stop.is_set():
                            time.sleep(0.1)
                        break
            except Exception as exc:  # pragma: no cover - restraints/runtime errors
                self._status(f"minimization failed: {exc}")
                self.bridge.run_on_main.emit(self._restore_static_diff)
                return
            finally:
                self._minimize_idle.set()  # the model is the drag's to take now
                self._minimize_mid = None
                self.bridge.minimizing_changed.emit(False)
            if first is not None:
                self._status(
                    f"{name}: bond rmsd {first['bonds_before']:.3f} -> {last['bonds_after']:.3f}, "
                    f"angle rmsd {first['angles_before']:.2f} -> {last['angles_after']:.2f} "
                    f"({shown} steps shown)"
                    + (f", map weight {last['weight']:.1f}" if last["weight"] else ""))
            self._invalidate_model_state(entry)  # stale: the coordinates just moved
            self._refresh_validation_staleness()  # warn: results now describe a past geometry
            # So is the density, if this model was phased: it describes where the atoms
            # were. Once per run, never per step — each update is two transforms — and
            # not again if the at-hold update above already re-phased this resting model.
            if reflections is not None and not updated_at_hold:
                self.bridge.run_on_main.emit(
                    lambda rid=reflections["id"]: self._update_maps_if_live(rid))
            elif reflections is None:
                self.bridge.run_on_main.emit(self._restore_static_diff)

        self.bridge.minimizing_changed.emit(True)
        self.run_background(work, name="pxviewer-minimize", label="Minimizing")


    def stop_minimization(self) -> None:
        """Halt a running minimization, returning once the model has stopped moving.

        The flag alone is only a request — the run ends at its next step
        boundary, and until then the coordinates still change. A caller that
        samples geometry right after this returns needs "stopped" to mean
        stopped, so wait out the wind-down (bounded: a wedged run must not
        hang the caller either). The model keeps the progress made so far —
        a stopped run is a shorter run, not a discarded one (figfuzz j03).
        """
        self._minimize_stop.set()
        self._status("stopping minimization…")
        # Bounded to the scale of a hold-loop poll plus a step's wind-down —
        # this can run on the GUI thread, which must not park for long.
        if not self._minimize_idle.wait(timeout=5.0):
            self._status("minimization is still winding down")


    def reset_view(self) -> None:
        """Reframe the viewport camera to fit the whole scene."""
        self._focused_residue = None  # space-bar nav restarts from the top after a reset
        control = self._control_session()
        if control is not None:
            control.reset_view()


    def camera_state(self) -> Optional[dict]:
        """The viewport camera: target, position, up, radius (the near/far depth
        slab, NOT the dolly distance — ``distance`` carries that), radiusMax, fov.

        Scene-level: works in a volume-only scene, where ``session`` is None —
        the camera belongs to the viewport, not to a model. ``radius`` is the
        depth slab the near/far planes come from; it does not zoom. To dolly,
        write ``set_camera`` a new ``position``.
        """
        session = self._control_session()
        return None if session is None else session.camera_state()


    def set_camera(self, state: dict) -> None:
        """Write the viewport camera — the fields ``camera_state()`` reports.

        Only present keys are applied (see :meth:`LiveSession.set_camera`), and
        ``radius`` is the depth slab, not the dolly distance: to zoom, move
        ``position`` toward or away from ``target``.
        """
        session = self._control_session()
        if session is None:
            raise ValueError("no viewport session — is a page connected?")
        session.set_camera(state)


    def orient_camera(self, target, up, direction, radius: float,
                      extents=None, clip: bool = True) -> None:
        """Aim the camera with an explicit orientation — see
        :meth:`LiveSession.orient_camera` for the parameter contract. Scene-level,
        so it works in a volume-only scene."""
        session = self._control_session()
        if session is None:
            raise ValueError("no viewport session — is a page connected?")
        session.orient_camera(target, up, direction, radius,
                              extents=extents, clip=clip)


    #: One press of a zoom button, as a multiple of the camera's current distance.
    #: Big enough that a single click reads as motion, small enough that holding a
    #: course across several presses stays controllable.
    ZOOM_STEP = 1.25


    def zoom_view(self, direction: int) -> None:
        """Zoom the viewport in (``direction`` < 0) or out (> 0) by one step.

        The counterpart to the wheel, which is bound to the contour level here (Coot's
        binding) and so cannot serve as zoom -- and to ctrl+wheel, which a trackpad has
        but a plain mouse makes awkward. The camera keeps its target and orientation.
        """
        if direction > 0:
            self._zoom_out_presses += 1
        else:
            self._zoom_in_presses += 1
        control = self._control_session()
        if control is not None:
            control.zoom_view(self.ZOOM_STEP if direction > 0 else 1.0 / self.ZOOM_STEP)


    # -- performance debugging -------------------------------------------

    def set_perf_prefs(self, **prefs) -> None:
        """Push debug render overrides to the viewer.

        ``overlay``, ``occlusionOff``, ``pixelScale`` (supersampling above 1),
        ``post`` (postprocessing merge: shadow/outline/dof/bloom/…), and
        ``illumination`` (the progressive path-traced-style pass) — see
        ``LiveSession.set_perf_prefs`` for the full key list. Canvas-global —
        sent on whichever session the viewport is connected to.
        """
        control = self._control_session()
        if control is not None:
            control.set_perf_prefs(**prefs)


    def start_perf_capture(self) -> None:
        """Begin recording per-frame drag timings in the viewer (see stop_perf_capture)."""
        run_js = getattr(self._viewport, "run_javascript", None)
        if run_js is None:
            # No page means no capture — claiming one started is the same
            # silent-no-op family as a send to a stopped session.
            self._status("performance capture needs a live viewport")
            return
        run_js("window.__pxviewerPerf && window.__pxviewerPerf.startCapture()")
        self._status("performance capture started — drag now, then stop to save the log")


    def stop_perf_capture(self, on_saved=None) -> None:
        """Stop the capture, write the log to a file, and hand its path to ``on_saved``.

        The samples live in the page, so they are read back through the webview rather than
        the socket; the file lands next to the app's other scratch output.
        """
        import datetime
        import json as _json
        import tempfile

        run_js = getattr(self._viewport, "run_javascript", None)
        if run_js is None:
            return

        def _write(result) -> None:
            if not result:
                self._status("performance capture: nothing recorded")
                return
            try:
                data = _json.loads(result)
            except Exception:
                self._status("performance capture: could not read the log")
                return
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = Path(tempfile.gettempdir()) / f"pxviewer-perf-{stamp}.json"
            path.write_text(_json.dumps(data, indent=2))
            n = data.get("meta", {}).get("samples", 0)
            self._status(f"performance capture saved: {path} ({n} frames)")
            if on_saved is not None:
                on_saved(str(path))

        run_js("window.__pxviewerPerf ? window.__pxviewerPerf.stopCapture() : ''", _write)


    def write_object(self, kind: str, ident: str, path: str) -> None:
        """Write a loaded object to disk: the model's cctbx coordinates, or the map.

        This writes what the DataManager holds (the model's own coordinates, the
        map's own grid), not anything from the viewer — the same bytes cctbx
        would round-trip. Masking, boxing and display decimation are view
        state; none of it reaches the file.

        The extension picks the format and the file must say so honestly —
        ``.pdb``/``.ent`` for PDB, ``.cif``/``.mmcif`` for mmCIF, and
        ``.map``/``.ccp4``/``.mrc`` for a CCP4 map. Anything else is refused
        rather than writing one format under another's name (figfuzz k07).
        """
        p = str(path)
        if kind == "model":
            entry = self._model_entry(ident, missing_ok=True)
            model = entry["session"].model if entry else None
            if model is None:
                raise ValueError("no cctbx model to write")
            lower = p.lower()
            is_pdb = lower.endswith((".pdb", ".ent"))
            if not is_pdb and not lower.endswith((".cif", ".mmcif")):
                raise ValueError(
                    f"cannot tell what format {Path(p).name!r} asks for — "
                    "write models as .pdb/.ent or .cif/.mmcif")
            if is_pdb:
                text = model.model_as_pdb()
            else:
                # Coordinates, not a validation report. model_as_mmcif() computes geometry
                # statistics (clashscore) whenever restraints exist — which every ligand and
                # minimized model here has — and clashscore shells out to Probe, a molprobity
                # binary we neither ship nor need to write a file. The hierarchy's mmCIF is
                # the coordinates + cell, matching the PDB branch, and never invokes Probe.
                text = model.get_hierarchy().as_mmcif_string(
                    crystal_symmetry=model.crystal_symmetry())
            with open(p, "w") as fh:
                fh.write(text)
        elif kind == "volume":
            entry = self._volume_entry(ident, missing_ok=True)
            if entry is None:
                raise ValueError("no such volume")
            # cctbx writes one format — CCP4 — whatever the extension says, so
            # an unrecognized one would name a file it is not.
            if not p.lower().endswith((".map", ".ccp4", ".mrc")):
                raise ValueError(
                    f"cannot tell what format {Path(p).name!r} asks for — "
                    "maps are written as CCP4 (.map/.ccp4/.mrc)")
            entry["data"].write_map(p)
        else:
            raise ValueError("nothing to write")


    def unknown_ligands(self, mid: str) -> list:
        """Residues in a model that cctbx cannot build restraints for.

        One unrecognized ligand costs the whole model its restraints -- pdb_interpretation
        refuses the file rather than the residue -- so minimize, drag, the Geometry tables
        and validation are all unavailable until it is resolved.
        """
        from . import ligands as ligands_mod

        entry = self._model_entry(mid)
        model = entry["session"].model if entry else None
        if model is None:
            return []
        try:
            return ligands_mod.unknown_ligands(model)
        except Exception as exc:
            # "cctbx cannot interpret this model at all" is NOT "nothing
            # unknown" — returning [] for both hid a broken model behind a
            # clean answer (figfuzz i07).
            raise ValueError(
                f"could not check for unknown ligands: {exc}") from exc


    def generate_ligand_restraints(self, mid: str, codes=None) -> dict:
        """Write monomer dictionaries for a model's unknown ligands and rebuild.

        ``codes`` limits it to particular residue names; omit for all of them. Returns
        ``{code: smiles}`` for those generated -- the perceived chemistry, which is what
        the user needs to check, since a dictionary built from a badly modeled ligand
        describes whatever rdkit made of it.

        Failures are per ligand: one residue rdkit cannot read does not prevent the others
        from being given restraints, and the exception for it is raised only if nothing
        succeeded.
        """
        from . import edits as edits_mod
        from . import ligands as ligands_mod

        entry = self._model_entry(mid)
        model = entry["session"].model if entry else None
        if model is None:
            raise ValueError("that object has no cctbx model")

        wanted = set(codes) if codes is not None else None
        generated, failures = {}, []
        smiles_by_code = {}
        for found in ligands_mod.unknown_ligands(model):
            if wanted is not None and found["code"] not in wanted:
                continue
            if found["code"] in generated:
                continue                    # one dictionary serves every copy of a residue
            try:
                cif_text, smiles = ligands_mod.restraints_from_residue(
                    model, found["i_seqs"], found["code"])
            except Exception as exc:
                failures.append("%s: %s" % (found["code"], exc))
                continue
            generated[found["code"]] = cif_text
            smiles_by_code[found["code"]] = smiles

        if not generated:
            raise ValueError("; ".join(failures) or "no ligands needed restraints")

        ligands_mod.apply_generated_restraints(model, generated)
        # Keep the generated dictionaries on the entry too — they ARE this
        # model's restraints now, and save_restraints_cif/export_ligand read
        # the entry field; without it a generated dictionary could never be
        # exported (figfuzz i07).
        entry["restraints_cif"] = "\n".join(
            generated[c] for c in sorted(generated))
        edits_mod.build_restraints(model, force=True)
        self.bridge.restraints_changed.emit(mid)
        if failures:
            self._warn("Restraints made for %d ligand(s); %s"
                       % (len(generated), "; ".join(failures)))
        else:
            self._status("Made restraints for " + ", ".join(sorted(generated)))
        return smiles_by_code


    def save_restraints_cif(self, mid: str, path: str) -> None:
        """Write a ligand's geometry restraints as a geostd-style monomer CIF.

        Only ligands built here carry one (SMILES ligands get an rdkit-provenance CIF;
        library ligands carry the geostd file they came from) — the exact bytes cctbx used,
        so the saved restraints match the model on screen.
        """
        entry = self._model_entry(mid)
        cif = entry.get("restraints_cif") if entry else None
        if not cif:
            raise ValueError("this object has no restraints to save")
        with open(str(path), "w") as fh:
            fh.write(cif)


    def export_ligand(self, mid: str, coord_path: str) -> tuple:
        """Write a ligand as the pair a refinement needs, in one step: its fitted
        coordinates (``coord_path``, mmCIF or PDB by extension) and — saved alongside as
        ``<stem>_restraints.cif`` — its restraints dictionary. The dictionary carries the
        ideal geometry in the monomer frame, so it complements the coordinates rather than
        repeating them. Returns ``(coord_path, restraints_path)``."""
        entry = self._model_entry(mid)
        if not (entry and entry.get("restraints_cif")):
            raise ValueError("this object has no restraints to export")
        p = Path(coord_path)
        restraints_path = str(p.with_name(p.stem + "_restraints.cif"))
        self.write_object("model", mid, str(coord_path))  # fitted coordinates
        self.save_restraints_cif(mid, restraints_path)     # restraints dictionary
        return str(coord_path), restraints_path


    # -- custom geometry restraint edits (phenix/cctbx geometry_restraints.edits) --------

    def model_edits(self, mid: str) -> list:
        """One ``{"kind", "summary", "i_seqs"}`` per custom restraint edit on a model.

        A summary for display and counting. The edits themselves live on the model as
        cctbx's own PHIL scope -- reachable with ``pxviewer.edits.get_edits(model)`` --
        because that is what gets handed to pdb_interpretation.
        """
        entry = self._model_entry(mid)
        if entry is None:
            return []
        return _edit_summaries(entry.get("edits"),
                               getattr(entry["session"], "model", None))


    def _apply_edits(self, entry: dict, scope) -> None:
        """Attach an edits scope to the model and rebuild its restraints.

        Validated: if cctbx rejects them (a bond the library already restrains, a
        selection naming no atom), revert and raise, so a model's stored edits are always
        ones a later minimize or drag can build.
        """
        from . import edits as edits_mod

        model = entry["session"].model
        if model is None:
            raise ValueError("the object has no cctbx model")
        previous = entry.get("edits")
        edits_mod.set_edits(model, scope)
        try:
            edits_mod.build_restraints(model, force=True)
        except Exception as exc:
            edits_mod.set_edits(model, previous)
            try:
                edits_mod.build_restraints(model, force=True)
            except Exception:  # pragma: no cover - revert best effort
                pass
            raise ValueError(str(exc).splitlines()[0] if str(exc) else "cctbx rejected the edit")
        entry["edits"] = scope
        self._emit_loaded_changed()
        # The restraints manager was just rebuilt, so anything cached off the old one --
        # the Geometry tables above all -- is describing restraints that no longer exist.
        self.bridge.restraints_changed.emit(entry["id"])


    def _edits_scope(self, entry):
        """The model's edits scope, copied so a rejected change can be rolled back."""
        import copy

        from . import edits as edits_mod

        existing = entry.get("edits")
        if existing is None:
            return edits_mod.empty_edits(entry["session"].model)
        return copy.deepcopy(existing)


    def add_edit_from_selection(self, mid: str, kind: str, *,
                                atoms: Optional[list] = None,
                                sigma: Optional[float] = None) -> dict:
        """Author a bond/angle/dihedral edit from the model's selected atoms (2/3/4),
        taking the current geometry as the target. ``atoms`` may name the vertices
        explicitly (positional indices, in order — vertex order is semantic for
        angle/dihedral) instead of reading the scene selection. Raises if the wrong
        number of atoms is selected or cctbx rejects it."""
        from . import edits as edits_mod

        entry = self._model_entry(mid)
        model = entry["session"].model
        if model is None:
            raise ValueError("the object has no cctbx model")
        need = edits_mod.KIND_ARITY.get(kind)
        if need is None:
            raise ValueError(f"cannot author a {kind} edit")
        if atoms is None:
            with self._scene_lock:
                atoms = list(self._scene_selection.get(mid, []))
        else:
            atoms = [int(a) for a in atoms]
        if len(atoms) != need:
            hint = ("" if atoms else
                    " — session.select()/highlight() only draw; select with "
                    "select_picked_atom, select_by_expression, or clicks, "
                    "or pass atoms=[...]")
            raise ValueError(
                f"{kind} needs exactly {need} atoms (have {len(atoms)}){hint}")
        sites = model.get_sites_cart().as_numpy_array()
        for i in atoms:
            if not 0 <= i < len(sites):
                raise IndexError(
                    f"atom index {i} is out of range for a "
                    f"{len(sites)}-atom model")
        # Authored here rather than read from a file, so a default weight is the app's to
        # choose — unlike a PHIL, where a missing sigma is refused (see AUTHORING_SIGMA).
        obj = edits_mod.new_entry(
            model, kind,
            [edits_mod.selection_for_atom(model, i) for i in atoms],
            ideal=edits_mod.geometry_value(kind, [sites[i] for i in atoms]),
            sigma=float(sigma) if sigma is not None else edits_mod.AUTHORING_SIGMA[kind])
        scope = self._edits_scope(entry)
        edits_mod.add_entry(scope, obj, kind)
        self._apply_edits(entry, scope)
        self.ensure_atoms_shown(mid)  # an edit is about specific atoms — show them
        return {"kind": kind, "summary": edits_mod.summarize(kind, obj)}


    def remove_edit(self, mid: str, index: int) -> None:
        """Drop the edit at ``index`` and rebuild restraints without it.

        Raises ``IndexError`` if no edit sits at that position — a removal the
        caller *thinks* happened is worse than an error."""
        from . import edits as edits_mod

        entry = self._model_entry(mid)
        scope = self._edits_scope(entry)
        edits_mod.remove(scope, index)
        self._apply_edits(entry, scope)


    def clear_edits(self, mid: str) -> None:
        """Drop every edit and rebuild the model's restraints without them."""
        from . import edits as edits_mod

        entry = self._model_entry(mid)
        self._apply_edits(entry, edits_mod.empty_edits(entry["session"].model))


    def load_edits(self, mid: str, path: str) -> int:
        """Read a geometry_restraints.edits PHIL file and add its edits to the model.

        Returns how many edits were added. Nothing is skipped any more: the file is
        fetched against cctbx's own master and handed to pdb_interpretation whole, so
        planarity and parallelity restraints -- which an earlier version counted and threw
        away -- are applied like the rest.
        """
        from . import edits as edits_mod

        entry = self._model_entry(mid)
        model = entry["session"].model
        with open(str(path)) as fh:
            incoming = edits_mod.edits_from_phil(fh.read(), model)
        if not edits_mod.count(incoming):
            raise ValueError("no restraint edits found in that file")
        scope = self._edits_scope(entry)
        added = edits_mod.merge(scope, incoming)
        self._apply_edits(entry, scope)
        return added


    def save_edits(self, mid: str, path: str) -> None:
        """Write a model's edits as a geometry_restraints.edits PHIL file."""
        from . import edits as edits_mod

        entry = self._model_entry(mid)
        scope = entry.get("edits")
        if scope is None or not edits_mod.count(scope):
            raise ValueError("this object has no edits to save")
        with open(str(path), "w") as fh:
            fh.write(edits_mod.edits_as_phil(scope, entry["session"].model))


    def _volume_command(self, vid: str, key: str, value, send) -> None:
        """Record a volume appearance change and push it to the viewport live.

        The value is kept on the entry so it survives a scene rebuild (which composes
        the MVSJ from these), and sent as a command so nothing has to reload — that is
        what lets a slider drive it while being dragged.
        """
        entry = self._volume_entry(vid)
        if entry is None or entry.get(key) == value:
            return
        entry[key] = value
        control = self._control_session()
        if control is not None:
            try:
                send(control, entry["ref"], value)
            except Exception:  # pragma: no cover - defensive
                pass


    def set_volume_style(self, vid: str, style: str) -> None:
        """Change a volume's isosurface style (surface or mesh) live."""
        self._volume_command(vid, "style", style,
                             lambda c, ref, v: c.set_volume_style(ref, v))


    def set_volume_iso(self, vid: str, value: float) -> None:
        """Set a volume's contour level, in sigma, live.

        Pushed even while the map is hidden. This used to return early, on the reading
        that a hidden map "is parked at an empty contour, so pushing the level would
        bring it back" -- true of the old implementation, but hiding has been a render
        skip on the isosurface for a long time (see :meth:`set_volume_visible`), and
        updating the level of something not being drawn is invisible until it is.

        Holding it back was worse than useless, because of the guard just below: the
        level was stored on the entry but never sent, so the entry and the viewer
        disagreed, and every later attempt to set that same number was swallowed as
        "already there". The Level control then did nothing at all until it was nudged
        to some *other* value -- which is exactly what "the controls stopped responding"
        looks like from the outside.
        """
        entry = self._volume_entry(vid)
        value = float(value)
        if entry is None or entry.get("iso") == value:
            return
        entry["iso"] = value
        if entry.get("color_by_resolution"):
            # The cheap path: the browser retained both grids with the full payload, so a
            # level change is one float over the wire and a client-side re-contour --
            # the same work a plain map's level change costs. _push_localres (which
            # re-encodes and re-streams ~128 MB of unchanged grids) stays for the changes
            # that actually alter a grid: a new mask, a recomputed resolution map.
            session = self._control_session()
            if session is not None:
                surface = self._display_map_data(entry)
                session.set_localres_iso(self._absolute_iso(entry, surface))
            return
        control = self._control_session()
        if control is not None:
            try:
                if entry.get("box"):
                    # The drawn repr is contoured absolutely (the boxed file's
                    # sigma is not the map's) — a sigma message would flip its
                    # kind back to relative and drift the level.
                    pos, kwargs = self._boxed_iso_for_wire(entry, value)
                    control.set_volume_iso(entry["ref"], pos, **kwargs)
                else:
                    control.set_volume_iso(
                        entry["ref"], self._iso_for_wire(entry, value))
            except Exception:  # pragma: no cover - defensive
                pass


    def _boxed_iso_for_wire(self, entry, value: float):
        """``(positive, kwargs)`` — a boxed grid's live contour as absolute levels.

        Same conversion the scene build does, plus the negative contour spelled
        out: ``-(absolute positive)`` is not ``mean - n*sigma``.
        """
        mag = value if entry.get("negative_iso") is None else entry["negative_iso"]
        if entry.get("iso_kind") == "absolute":
            return float(value), {"kind": "absolute",
                                  "negative_value": -float(mag)}
        mean, std = self._display_map_stats(entry)
        return mean + float(value) * std, {"kind": "absolute",
                                           "negative_value": mean - float(mag) * std}


    def _boxed_negative_iso_for_wire(self, entry, value):
        """A boxed difference map's detached negative contour, absolute — None
        re-links it at the positive level's magnitude."""
        if entry.get("iso_kind") == "absolute":
            mag = entry["iso"] if value is None else value
            return -float(mag)
        mean, std = self._display_map_stats(entry)
        mag = entry["iso"] if value is None else value
        return mean - float(mag) * std


    @staticmethod
    def _iso_for_wire(entry, value: float) -> float:
        """A contour level in the sigma units the live command speaks.

        The wire protocol is sigma-only, which is right for maps: it means one slider range
        serves any map without the viewer knowing its absolute scale. A severity field is
        contoured on absolute values instead, because its levels are calibrated — so its
        level has to be converted here, or a live change would land somewhere else entirely
        from where the same number puts it on a scene rebuild.
        """
        if entry.get("iso_kind") != "absolute":
            return float(value)
        stats = entry["data"].stats()
        std = stats["std"] or 1.0
        return (float(value) - stats["mean"]) / std


    def set_volume_opacity(self, vid: str, value: float) -> None:
        """Set a volume's opacity (0-1) live."""
        self._guard_transparent_map_size(vid, value)
        self._volume_command(vid, "opacity", float(value),
                             lambda c, ref, v: c.set_volume_opacity(ref, v))


    def _guard_transparent_map_size(self, vid: str, value: float) -> None:
        """Keep a transparent pass off an oversized grid.

        Transparency lowers the same geometry ceiling the mesh deaths hit — a
        transparent isosurface on a cryo-EM-sized grid killed the render process
        in agent sessions. A full-res oversized map asking for alpha gets the
        auto-downsample first. A boxed map is already small by the caller's
        choice, a caller-set factor stands, and maps under the cap are
        untouched.
        """
        if float(value) >= 1:
            return
        entry = self._volume_entry(vid)
        if (entry is None or entry.get("box")
                or int(entry.get("localres_downsample") or 1) > 1):
            return
        data = entry.get("data")
        if data is None:
            return
        factor = self._auto_downsample_factor(data)
        if factor <= 1:
            return
        self._status(
            f"{entry.get('name', vid)}: transparent pass on a large grid — "
            f"drawing a {factor}x-downsampled copy (Downsample: Full restores)")
        self.set_volume_downsample(vid, factor, _source="auto")


    def set_volume_position(self, vid: str, position) -> None:
        """Translate a volume's display by an Angstrom offset — live and reload-safe.

        A display transform, not a change to the data: the written scene gives
        every map a zero-translation transform node for this to update, and the
        offset is recomposed on reload like the rest of the draw state. Use it to
        bring a map loaded in its own frame over the model — though for a paired
        model+map, :meth:`align_paired_model_with_map` is the density-scored way.
        """
        entry = self._volume_entry(vid)
        if entry is None:
            return
        pos = [float(c) for c in position]
        if len(pos) != 3 or any(not math.isfinite(c) for c in pos):
            raise ValueError("position must be three finite Angstrom offsets")
        if entry.get("position") == pos:
            return
        entry["position"] = pos
        control = self._control_session()
        if control is not None:
            try:
                control.set_volume_position(entry["ref"], pos)
            except Exception:  # pragma: no cover - defensive
                pass


    def set_volume_color(self, vid: str, color: str) -> None:
        """Set a volume's color live."""
        self._volume_command(vid, "color", color,
                             lambda c, ref, v: c.set_volume_color(ref, v))


    def set_volume_negative_color(self, vid: str, color: str) -> None:
        """Set a difference map's negative-contour color live.

        Setting the color *creates* the contour on a map that opened without one
        — naming a color is what makes the second isosurface a thing. The scene
        is recomposed for it: a live command can only reach a node the composed
        scene already has.
        """
        entry = self._volume_entry(vid)
        if entry is None or entry.get("negative_color") == color:
            return
        created = not entry.get("negative_color")
        entry["negative_color"] = color
        if created:
            self._reload_viewport()
            return
        control = self._control_session()
        if control is not None:
            try:
                control.set_volume_negative_color(entry["ref"], color)
            except Exception:  # pragma: no cover - defensive
                pass


    def set_volume_negative_style(self, vid: str, style: str) -> None:
        """Set a difference map's negative-contour style (surface or mesh) live."""
        self._negative_command(vid, "negative_style", str(style),
                               lambda c, ref, v: c.set_volume_negative_style(ref, v))


    def set_volume_negative_opacity(self, vid: str, value: float) -> None:
        """Set a difference map's negative-contour opacity (0-1) live."""
        self._negative_command(vid, "negative_opacity", float(value),
                               lambda c, ref, v: c.set_volume_negative_opacity(ref, v))


    def _negative_command(self, vid: str, key: str, value, send) -> None:
        """Record a negative-contour appearance change and push it live.

        The map-level twin of :meth:`_volume_command`, with one extra guard: a map with
        no negative contour has no such setting to change, and inventing one would put a
        value on the entry that nothing draws.
        """
        entry = self._volume_entry(vid)
        if entry is None or not entry.get("negative_color"):
            return
        if entry.get(key) == value:
            return
        entry[key] = value
        control = self._control_session()
        if control is not None:
            try:
                send(control, entry["ref"], value)
            except Exception:  # pragma: no cover - defensive
                pass


    def set_volume_negative_iso(self, vid: str, value: Optional[float]) -> None:
        """Give a difference map's negative contour its own level, in sigma, live.

        ``value`` is a *magnitude* -- the same number the Level control shows, applied at
        the other sign -- and ``None`` re-links it to the positive level. The two are
        linked until this is called with a number, so the default reading of a difference
        map ("contour at 3 sigma", both ways) is unchanged for anyone who never touches
        the second control.
        """
        entry = self._volume_entry(vid)
        # No contour, no level: the level belongs to a negative contour, which
        # exists once the map is a difference map — opened as one, or made one
        # by set_volume_negative_color. Setting it earlier would invent a
        # contour the map never had (a dead row under every plain 2Fo-Fc map).
        if entry is None or not entry.get("negative_color"):
            return
        value = None if value is None else float(value)
        if entry.get("negative_iso") == value:
            return
        entry["negative_iso"] = value
        control = self._control_session()
        if control is not None:
            try:
                if entry.get("box"):
                    control.set_volume_negative_iso(
                        entry["ref"],
                        self._boxed_negative_iso_for_wire(entry, value),
                        kind="absolute")
                else:
                    control.set_volume_negative_iso(
                        entry["ref"],
                        None if value is None else self._iso_for_wire(entry, value))
            except Exception:  # pragma: no cover - defensive
                pass


    def set_volume_negative_visible(self, vid: str, visible: bool) -> None:
        """Show or hide only a difference map's negative contour.

        Independent of the map's own visibility: "just the green one" is a reading of a
        difference map, not a half-hidden object, so it survives the map being toggled.
        """
        entry = self._volume_entry(vid)
        if entry is None or not entry.get("negative_color"):
            return
        if entry.get("negative_visible", True) == bool(visible):
            return
        entry["negative_visible"] = bool(visible)
        control = self._control_session()
        if control is not None:
            try:
                control.set_volume_negative_visible(entry["ref"], bool(visible))
            except Exception:  # pragma: no cover - defensive
                pass
        # The contour has a row of its own, so its eye has to follow it -- the same
        # refresh a map's own visibility does.
        self._emit_loaded_changed()


    def save_screenshot(self, path: str) -> None:
        """Render the viewport and write it to ``path`` as a PNG.

        The picture is taken in the browser (see LiveSession.screenshot), so this waits
        on a round trip and runs on a background thread. Any connected session can take
        it — the scene is the page's, not one model's.
        """
        session = self._control_session()
        if session is None:
            raise ValueError("nothing is loaded to photograph")
        name = Path(path).name

        def work():
            try:
                png = session.screenshot()
            except Exception as exc:  # pragma: no cover - viewer-side errors
                self._status(f"screenshot failed: {exc}")
                return
            if not png:
                self._status("screenshot failed: the viewport did not answer")
                return
            try:
                Path(path).write_bytes(png)
            except OSError as exc:
                self._status(f"could not write {name}: {exc}")
                return
            self._status(f"Saved {name} ({len(png) // 1024} kB)")

        self.run_background(work, name="pxviewer-screenshot", label="Saving image")
        self._status("taking a picture…")


    def volume_appearance(self, vid: str) -> dict:
        """A volume's current style/color/opacity/level — and what is drawn.

        ``downsample`` is the display decimation actually in effect (1 = full
        grid), ``box`` the carved region if any, and ``max_sigma`` the level
        above which the map empties — the read-back a figure needs before it
        can be trusted to show what was intended.

        The Loaded summary is a snapshot taken when it was emitted, and these can change
        without one — from the console, or by the wheel in the viewport — so the
        Appearance pane reads them from the entry rather than trusting the snapshot.
        """
        entry = self._volume_entry(vid)
        if entry is None:
            return {}
        out = {key: entry.get(key)
               for key in ("style", "color", "opacity", "iso", "clip", "mask_radius",
                           "negative_color", "negative_iso", "position",
                           "negative_opacity", "negative_style", "box")}
        # What grid the surface is actually drawn from. A figure caller must be
        # able to see this: an oversized map opens decimated for interactivity,
        # which is not a publishable contour until set_volume_downsample(vid, 1).
        out["downsample"] = int(entry.get("localres_downsample") or 1)
        # "user" the caller picked the drawn grid; "auto" pxviewer decimated it
        # for interactivity — the distinction a figure workflow must check.
        out["downsample_source"] = entry.get("downsample_source")
        out["sphere"] = self._sphere_state(entry)
        out["visible"] = entry.get("visible", True)
        out["negative_visible"] = entry.get("negative_visible", True)
        # Session-level setters (kernel callers using ``session.set_volume_*``)
        # record the wire value but never touch this entry — the session's replay
        # dicts are the newer of the two stores for anything they carry.
        session = self._control_session()
        if session is not None:
            ref = str(entry["ref"])
            for attr, key in (("_volume_iso", "iso"),
                              ("_volume_opacity", "opacity"),
                              ("_volume_style", "style"),
                              ("_volume_color", "color"),
                              ("_volume_position", "position"),
                              ("_volume_negative_color", "negative_color"),
                              ("_volume_negative_style", "negative_style"),
                              ("_volume_negative_opacity", "negative_opacity"),
                              ("_volume_negative_iso", "negative_iso")):
                recorded = getattr(session, attr, {}).get(ref)
                if recorded is not None:
                    if key in ("iso", "negative_iso") and isinstance(recorded, dict):
                        out[key] = self._iso_from_wire(
                            entry, recorded, negative=(key == "negative_iso"))
                    else:
                        out[key] = (recorded.get("value")
                                    if isinstance(recorded, dict) else recorded)
            hidden = getattr(session, "_volume_negative_hidden", {}).get(ref)
            if hidden is not None:
                out["negative_visible"] = not hidden
        # The map's maximum on the sigma scale -- the level above which nothing is left.
        # The Level slider spans up to here, so its right end genuinely empties the map.
        try:
            stats = entry["data"].stats()
            out["max_sigma"] = (stats["max"] - stats["mean"]) / (stats["std"] or 1.0)
        except Exception:  # pragma: no cover - defensive (a map with no data)
            out["max_sigma"] = None
        return out


    def _iso_from_wire(self, entry: dict, msg: dict, *, negative: bool = False):
        """Inverse of :meth:`_iso_for_wire`/``_boxed_iso_for_wire`` — report a
        recorded wire level in the units ``entry["iso"]`` stores, i.e. the number
        :meth:`set_volume_iso` takes. The wire is sigma for a whole map but
        absolute for a boxed one (the box's sigma is not the map's), so reading
        the recorded value back raw would report the wrong scale (figfuzz h05:
        set 4.5σ on a boxed map, read back 3.59). ``negative`` inverts a negative
        contour's absolute level back into the magnitude the entry stores
        (``mean - n·σ`` on the wire, ``n`` in the entry)."""
        value = msg.get("value")
        if value is None:
            return None
        value = float(value)
        if msg.get("kind") == "absolute":
            if entry.get("iso_kind") == "absolute":
                return -value if negative else value     # absolute magnitude
            try:
                mean, std = self._display_map_stats(entry)
                sigma = (value - mean) / (std or 1.0)
                return -sigma if negative else sigma
            except Exception:  # pragma: no cover - display file gone
                return -value if negative else value
        if entry.get("iso_kind") == "absolute":
            stats = entry["data"].stats()
            return value * (stats["std"] or 1.0) + stats["mean"]
        return value


    def set_volume_clip(self, vid: str, front: float, back: float) -> None:
        """Clip a volume to a front/rear slab (see LiveSession.set_clip)."""
        entry = self._volume_entry(vid)
        clip = (float(front), float(back))
        if entry is None or entry.get("clip") == clip:
            return
        entry["clip"] = clip
        self._send_volume_clip(entry)


    def set_view_radius(self, radius: Optional[float]) -> None:
        """How much density every map draws around the view center (None = all of it).

        One setting for all of them. It used to be per map, set at creation from a
        "for new maps" default that reached nothing already open -- so a map from
        reflections opened bounded, one loaded from a file did not, and a difference map
        made later followed whichever rule was current. The result was contours that
        behaved differently with nothing on screen to say why.

        A crystallographic map fills the unit cell and a cryo-EM box is mostly solvent;
        drawing all of either buries the model. This is the control Coot has for that,
        and it follows the view rather than editing the map (see set_volume_radius, the
        per-map primitive this drives).
        """
        self.view_radius = self._check_sphere_radius(radius)
        for entry in list(self._volumes):
            self.set_volume_radius(entry["id"], self.view_radius)


    def set_default_model_representation(self, rep: str, shown: bool) -> bool:
        """Persist whether newly opened models include a representation layer."""
        reps = list(self._default_model_reps)
        if shown and rep not in reps:
            reps.append(rep)
        elif not shown and rep in reps:
            if len(reps) == 1:
                return False
            reps.remove(rep)
        self._default_model_reps = reps
        self._settings.setValue("defaults/model_representations", json.dumps(reps))
        self._settings.sync()
        return True


    def set_default_model_show(self, label: str, shown: bool) -> None:
        """Persist a structure-type or Mol* interaction default for new models."""
        if label == "Mol* interactions":
            self._default_model_interactions = bool(shown)
            self._settings.setValue(
                "defaults/molstar_interactions", "true" if shown else "false")
            self._settings.sync()
            return
        if label not in _STRUCTURE_TYPE_ORDER:
            return
        if shown:
            self._default_shown_types.add(label)
        else:
            self._default_shown_types.discard(label)
        ordered = [
            value for value in _STRUCTURE_TYPE_ORDER
            if value in self._default_shown_types]
        self._settings.setValue("defaults/shown_structure_types", json.dumps(ordered))
        self._settings.sync()


    def set_volume_radius(self, vid: str, radius: Optional[float]) -> None:
        """Draw only density within ``radius`` A of the view center (None = all of it).

        A crystallographic map fills the unit cell, and contouring the whole thing buries
        the model in density — this is the control Coot has for that, and it follows the
        view. Unlike the mask it edits nothing: the map is whole, just not all drawn.
        It writes the "view" sphere mode on every open map; a map's own Sphere row in
        its Appearance pane is the per-map version of the same control.
        """
        entry = self._volume_entry(vid)
        radius = self._check_sphere_radius(radius)
        want = ({"mode": "view", "radius": radius, "center": None}
                if radius is not None else dict(self._SPHERE_OFF))
        if entry is None or self._sphere_state(entry) == want:
            return
        entry["sphere"] = want
        self._send_volume_clip(entry)


    def _reassert_volume_clips(self) -> None:
        """Re-tell the control session every volume's clip, before the page reloads.

        A clip is worked out from the camera and re-aimed as it moves, so unlike a
        color or a level it cannot be baked into the scene — the session has to replay
        it when the fresh page connects. Both ends of that move underneath it: the
        session carrying volume commands changes (dummy <-> active model), and the page
        is new. So the clips are re-asserted on every reload rather than sent once.
        """
        for entry in self._volumes:
            if (self._sphere_state(entry)["mode"] != "off"
                    or entry.get("clip") != (0.0, 1.0)):
                self._send_volume_clip(entry)


    def _reassert_hidden_volumes(self) -> None:
        """After a reload the fresh scene draws every map; re-hide the ones marked hidden.
        A render skip on the current control session (models replay their own hidden state on
        reconnect, but a volume lives in the shared scene, so the app re-asserts it)."""
        control = self._control_session()
        if control is None:
            return
        for entry in self._volumes:
            if entry.get("is_resolution"):
                continue  # never in the scene (see _write_volume_scene): nothing to hide
            if not entry["visible"] or entry.get("color_by_resolution"):
                # Hidden maps, and the parked plain contour of a colored map -- the
                # colored surface represents that one, whatever the entry's own flag.
                try:
                    control.set_volume_visible(entry["ref"], False)
                except Exception:  # pragma: no cover - defensive
                    pass


    def _send_volume_clip(self, entry) -> None:
        """Push a volume's whole clip: the slab and the sphere are one thing to the
        viewer, so a change to either re-sends both."""
        control = self._control_session()
        if control is None:
            return
        front, back = entry.get("clip") or (0.0, 1.0)
        radius, center = self._resolved_sphere(entry)
        try:
            control.set_clip(front, back, radius=radius, center=center,
                             ref=entry["ref"])
        except Exception:  # pragma: no cover - defensive
            pass


    # -- clip state: slab ∩ sphere ---------------------------------------------------
    #
    # An object's clip is a region: the front/rear depth slab the Appearance pane's
    # Clip slab row drives, INTERSECTED with an optional sphere (a *place* clip).
    # The sphere has three modes — "selection" (center and radius re-fit to each
    # applied selection; the default, and the whole selection-clip policy), "view"
    # (center follows the camera target, Coot-style — what the view-radius setting
    # means for maps) and "point" (frozen at a coordinate). Both halves compose on the
    # wire (LiveSession.set_clip carries front/back + radius/center together), so every
    # send goes through one funnel — a slab change must not wipe a standing sphere and
    # a sphere change must not wipe a standing slab.
    _SPHERE_OFF = {"mode": "off", "radius": None, "center": None}


    def _initial_sphere_mode(self) -> dict:
        """A model's starting clip-sphere state — the armed selection policy.

        "selection" with no radius is dormant: the next selection fits it. The
        long-retired Selection-pane checkbox's preference (clip_on_apply) seeds
        it, so an install that turned clipping off keeps it off."""
        armed = str(self._settings.value(
            "selection/clip_on_apply", "true")).lower() != "false"
        return {"mode": "selection" if armed else "off",
                "radius": None, "center": None}


    @staticmethod
    def _component_bounds(atoms, indices) -> list:
        """``indices`` grown to whole components (residue groups) — for sphere fits.

        The selection keeps its granularity, but a sphere fit to a single picked
        atom clips that atom's own residue in half. The fit therefore measures the
        components the selection stands inside: picking CA of a residue clips to
        the whole residue. (atom → atom_group → residue_group, same parentage the
        residue-granularity pick uses.)
        """
        expanded = set()
        size = atoms.size()
        for i in indices:
            if 0 <= i < size:
                expanded.update(int(a.i_seq)
                                for a in atoms[i].parent().parent().atoms())
        return sorted(expanded)


    @staticmethod
    def _sphere_state(entry) -> dict:
        """An object's clip sphere, normalized to ``{"mode", "radius", "center"}``."""
        sph = entry.get("sphere")
        return sph if isinstance(sph, dict) else dict(AppCore._SPHERE_OFF)


    def _resolved_sphere(self, entry):
        """The sphere's wire values: (radius, center). ``center`` None means the sphere
        follows the camera target — mode "view"."""
        sph = self._sphere_state(entry)
        if sph["mode"] == "off" or sph.get("radius") is None:
            return None, None
        if sph["mode"] in ("selection", "point"):
            return sph["radius"], sph.get("center")
        return sph["radius"], None


    @staticmethod
    def _check_sphere_radius(radius: Optional[float]) -> Optional[float]:
        """A stored sphere radius is a positive finite number of Angstrom, or None.

        A non-positive or NaN/inf value must never reach ``entry["sphere"]``: the
        viewer ignores it, so the app would report a clip it is not drawing —
        the phantom-hidden-types bug again, spatial (figfuzz k06).
        """
        if radius is None:
            return None
        radius = float(radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(
                f"clip sphere radius must be a positive number of Å "
                f"(or None to lift it), got {radius}")
        return radius


    def _send_model_clip(self, entry, *, depth=None) -> None:
        """Push a model's whole clip — slab and sphere together."""
        front, back = entry.get("clip") or (0.0, 1.0)
        radius, center = self._resolved_sphere(entry)
        kw = {} if depth is None else {"depth": depth}
        entry["session"].set_clip(front, back, radius=radius, center=center, **kw)


    def _apply_model_sphere(self, entry, mode, radius=None, center=None,
                            *, depth=None, refresh=True) -> None:
        """Write the model's sphere state and send the composed clip.

        The one funnel for sphere changes: it keeps ``_auto_clip`` (this sphere stands
        on a selection's behalf) and ``_auto_clip_depth`` (the camera slab a clipped
        orient parked) in step with the mode — a hand-set mode drops the auto-clip
        bookkeeping the lift paths check.
        """
        entry["sphere"] = {"mode": mode, "radius": radius,
                           "center": None if center is None else [float(c) for c in center]}
        if mode == "selection" and radius is not None:
            entry["_auto_clip"] = True    # a selection-fitted sphere is standing
        else:
            entry.pop("_auto_clip", None)
            # NB: ``_auto_clip_depth`` is the callers' bookkeeping — a plain lift must
            # keep it, because reapply re-parks that camera slab.
        self._send_model_clip(entry, depth=depth)
        # The sphere row in the Appearance pane is this state's readout — refresh if
        # the pane is showing this object (signature-gated, so a no-op if unchanged).
        # Row-originated changes pass refresh=False: the widget already shows what the
        # user just set, and rebuilding under a still-editing spin would kill it.
        try:
            if (refresh and self._controls._focused == ("model", entry.get("id"))):
                self._controls._update_appearance("model", entry.get("id"))
        except Exception:  # pragma: no cover - defensive
            pass


    def _lift_model_sphere(self, entry, *, depth=None) -> None:
        """Drop an object's sphere, whatever mode put it there (slab untouched)."""
        if self._sphere_state(entry)["mode"] == "off":
            return
        self._apply_model_sphere(entry, "off", depth=depth)


    def _lift_selection_sphere(self, entry) -> None:
        """Drop a standing selection-fitted sphere, keeping "selection" armed.

        Mode is the policy now: the sphere a selection put up comes down with the
        selection, but the mode stays "selection" so the next one re-fits. Leaving
        the policy entirely is the Clip sphere row's job (Off, or another mode)."""
        if not entry.pop("_auto_clip", False):
            return
        entry.pop("_auto_clip_depth", None)
        entry["sphere"] = {"mode": "selection", "radius": None, "center": None}
        self._send_model_clip(entry)
        try:
            if self._controls._focused == ("model", entry.get("id")):
                self._controls._update_appearance("model", entry.get("id"))
        except Exception:  # pragma: no cover - defensive
            pass


    def _selection_clip_armed(self, entry) -> bool:
        """Whether a selection may clip this object — the Clip sphere mode."""
        return (entry is not None
                and self._sphere_state(entry)["mode"] == "selection")


    def _default_sphere_center(self, entry):
        """Where a fresh fixed-point sphere lands: the standing sphere's own spot, then
        the model's selection, then the object's middle (atoms for a model, the grid
        center for a map)."""
        sph = self._sphere_state(entry)
        if sph.get("center") is not None:
            return list(sph["center"])
        if sph.get("mode") == "view":
            # A view sphere's spot is wherever the camera target is — freezing
            # it to a point means keeping exactly that spot, not the object's
            # middle (figfuzz j01).
            control = self._control_session()
            cam = control.camera_state() if control is not None else None
            if cam and cam.get("target"):
                return [float(c) for c in cam["target"]]
        mid = entry.get("id")
        model = getattr(entry.get("session"), "model", None)
        if model is not None:
            with self._scene_lock:
                indices = list(self._scene_selection.get(mid, ()))
            atoms = model.get_hierarchy().atoms()
            xyz = np.array([atoms[i].xyz for i in indices if 0 <= i < len(atoms)],
                           dtype=float)
            if not len(xyz):
                xyz = np.array([a.xyz for a in atoms], dtype=float)
            return list(xyz.mean(axis=0)) if len(xyz) else None
        data = entry.get("data")          # a volume's: the middle of its own grid
        mm = getattr(data, "map_manager", None)
        if mm is None:
            return None
        from .volume_io import grid_affine
        origin, steps = grid_affine(mm)
        dims = np.asarray(mm.map_data().all(), dtype=float)
        return list(origin + steps @ (dims / 2.0))


    def set_model_clip(self, mid: str, front: float, back: float) -> None:
        """Clip a model's representations to a front/rear slab.

        Unlike a volume — whose representation belongs to the shared MVSJ scene, and so
        is addressed by reference — a model is clipped through its own session, which
        owns the representations the viewer built for it.
        """
        entry = self._model_entry(mid)
        clip = (float(front), float(back))
        if entry is None or entry.get("clip") == clip:
            return
        entry["clip"] = clip
        try:
            self._send_model_clip(entry)
        except Exception:  # pragma: no cover - defensive
            pass


    def model_sphere_state(self, mid: str) -> dict:
        """A model's clip sphere for the Appearance pane's Sphere row."""
        entry = self._model_entry(mid)
        return dict(self._sphere_state(entry)) if entry else dict(self._SPHERE_OFF)


    def volume_sphere_state(self, vid: str) -> dict:
        """A volume's clip sphere for the Appearance pane's Sphere row."""
        entry = self._volume_entry(vid)
        return dict(self._sphere_state(entry)) if entry else dict(self._SPHERE_OFF)


    def set_model_sphere(self, mid: str, mode: str, radius: Optional[float] = None,
                         *, refresh=True) -> None:
        """Set a model's clip sphere — the Appearance pane's Sphere row.

        "selection" needs a selection to measure; without one the mode is stored and
        the next selection sizes the sphere. "point" freezes the current sphere's spot
        (or the selection's, or the model's middle). "view" follows the camera target.
        ``refresh`` is False for the radius spin's own ticks — it is the readout, and
        rebuilding the pane under a still-editing widget would kill it.
        """
        entry = self._model_entry(mid)
        if entry is None or mode not in ("off", "selection", "view", "point"):
            return
        radius = self._check_sphere_radius(radius)  # before the swallow-all try
        if mode in ("view", "point"):
            # A user-placed sphere displaces the selection's clip, so the camera
            # slab it parked is no longer ours to restore. "Off" is a pure lift —
            # the parked slab stays, and re-arming "selection" puts the framed
            # view back whole (the old checkbox's lift/re-apply semantics).
            entry.pop("_auto_clip_depth", None)
        try:
            if mode == "point":
                center = self._default_sphere_center(entry)
                if center is None:
                    return
                keep = self._sphere_state(entry).get("radius")
                self._apply_model_sphere(
                    entry, mode,
                    radius=radius if radius is not None else (keep or 10.0),
                    center=center, refresh=refresh)
                return
            if mode == "selection":
                with self._scene_lock:
                    indices = list(self._scene_selection.get(mid, ()))
                model = getattr(entry["session"], "model", None)
                if indices and model is not None:
                    atoms = model.get_hierarchy().atoms()
                    bounds = self._component_bounds(atoms, indices)
                    xyz = np.array([atoms[i].xyz for i in bounds], dtype=float)
                    if len(xyz):
                        center = xyz.mean(axis=0)
                        reach = float(np.linalg.norm(xyz - center, axis=1).max())
                        self._apply_model_sphere(
                            entry, mode, radius=reach + self._clip_padding(),
                            center=center, depth=entry.get("_auto_clip_depth"),
                            refresh=refresh)
                        return
                self._apply_model_sphere(entry, mode, refresh=refresh)
                # dormant until a selection lands
                return
            self._apply_model_sphere(
                entry, mode,
                radius=(radius if radius is not None else 10.0) if mode == "view"
                else None, refresh=refresh)
        except Exception:  # pragma: no cover - defensive
            pass


    def set_volume_sphere(self, vid: str, mode: str, radius: Optional[float] = None,
                          center: Optional[list] = None, *, refresh=True) -> None:
        """Set a volume's clip sphere — "view" is the classic map-around-target bound.

        ``center`` (only read for "point") freezes the sphere at those xyz
        coordinates; without it a point sphere sits at the standing spot,
        the selection, or the grid middle."""
        entry = self._volume_entry(vid)
        if entry is None or mode not in ("off", "view", "point"):
            return
        radius = self._check_sphere_radius(radius)
        if mode == "point":
            if center is None:
                center = self._default_sphere_center(entry)
            if center is None:
                return
            keep = self._sphere_state(entry).get("radius")
            entry["sphere"] = {"mode": mode,
                               "radius": (radius if radius is not None
                                          else (keep or 10.0)),
                               "center": [float(c) for c in center]}
        else:
            entry["sphere"] = {"mode": mode,
                               "radius": ((radius if radius is not None else 10.0)
                                          if mode == "view" else None),
                               "center": None}
        self._send_volume_clip(entry)
        try:
            if (refresh and
                    self._controls._focused in (("volume", vid), ("negative", vid))):
                self._controls._update_appearance(*self._controls._focused)
        except Exception:  # pragma: no cover - defensive
            pass


    def model_appearance(self, mid: str) -> dict:
        """A model's clip state for the Appearance pane (see :meth:`volume_appearance`)."""
        entry = self._model_entry(mid)
        return {} if entry is None else {"clip": entry.get("clip"),
                                         "sphere": self._sphere_state(entry)}


    def set_volume_scroll_target(self, vid: Optional[str]) -> None:
        """Point the scroll wheel's contouring at a volume (None = nothing).

        The wheel adjusts whatever the Appearance pane's Level slider is showing, so
        this follows the focused object rather than the viewport picking for itself.
        (Coot's binding: in map work the contour level is what you reach for most.)
        """
        entry = self._volume_entry(vid, missing_ok=True) if vid else None
        # Always re-assert: the viewport reloads on any scene change, and the session
        # carrying volume commands can switch (dummy <-> active model), so the target
        # has to be told to whoever is carrying them now.
        self._volume_scroll_target = entry["id"] if entry else None
        control = self._control_session()
        if control is not None:
            try:
                control.set_volume_scroll_target(entry["ref"] if entry else None)
            except Exception:  # pragma: no cover - defensive
                pass


    def _on_volume_iso_changed(self, ref: str, value: float) -> None:
        """A contour level changed in the viewport (the wheel): follow it here.

        The viewer has already applied it, so this only records the value and lets the
        controls catch up — sending it back would fight the user's next scroll.
        """
        entry = next((v for v in self._volumes if v["ref"] == ref), None)
        if entry is None:
            return
        entry["iso"] = float(value)
        if entry.get("color_by_resolution"):
            # The viewer just re-leveled the *parked* plain contour (the wheel works on
            # refs); a colored surface is what is on screen, so re-level that too.
            session = self._control_session()
            if session is not None:
                surface = self._display_map_data(entry)
                session.set_localres_iso(self._absolute_iso(entry, surface))
        self.bridge.volume_iso_changed.emit((entry["id"], float(value)))


    def set_active_model(self, mid: str) -> None:
        """Make a loaded model the active one (the atoms table + selection follow it)."""
        entry = self._model_entry(mid)
        if entry is None or self._active_model_id == mid:
            return
        self._active_model_id = mid
        self._rebind_kernel_session()
        self._wire_active(entry["session"])  # no viewport reload: visibility is unchanged
        self._refresh_validation_staleness()  # the warning tracks the now-active model
        self._emit_loaded_changed()


    def set_model_visible(self, mid: str, visible: bool) -> None:
        """Show or hide a loaded model in the viewport, in place.

        Toggles the model's own render visibility — no page reload, so the other objects do
        not flicker and, crucially, the camera does not move: a reload re-runs the scene's
        focus, which reframed the view every time you hid something.

        This was once reload-based, on the belief that "a reload is the only teardown this
        app's WebGL survives; an in-place change segfaults it". That was a misdiagnosis. The
        segfault was never in the renderer — it was a use-after-free in the *object tree*,
        which rebuilt itself from inside ``QTreeWidgetItem::setData`` and freed the item Qt
        was still using (see ``ControlsWindow._on_tree_item_clicked``). Three unrelated hide
        mechanisms crashed with one identical Qt backtrace, on software and hardware alike,
        which is the tell that no GPU was involved.

        Still refused on **software** WebGL (silently — an internal caller, add-hydrogens,
        hides the H-less original, and must not warn or crash); there the tree checkbox is
        non-checkable and a click flashes why. That restriction predates the real diagnosis
        and is probably now unnecessary, but it has not been re-tested on software rendering.
        Maps hide the same way."""
        entry = self._model_entry(mid)
        if entry is None or entry["visible"] == bool(visible):
            return
        if not self._can_hide:
            return  # software: hiding any drawn object segfaults; the model stays shown
        entry["visible"] = bool(visible)
        entry["session"].set_structure_visible(bool(visible))  # a render skip, in place
        self._emit_loaded_changed()


    def remove_model(self, mid: str) -> None:
        """Unload a model: stop its session and drop it from the viewport."""
        entry = self._model_entry(mid)
        if entry is None:
            return
        self._models.remove(entry)
        if self._tug_model == mid:
            # A drag in flight on the departing model: abandon it. Settling is
            # pointless — the model is leaving — and the worker's next
            # _push_tug must not land on a session that is about to stop.
            self._tug = None
            self._tug_model = None
            self._tug_session = None
        if self._tug_cache is not None and self._tug_cache[0][0] == mid:
            self._tug_cache = None  # do not keep the removed model's zone alive
        if self._minimize_mid == mid and not self._minimize_idle.is_set():
            # A run on this model outlives it only by spinning its hold loop
            # forever over a dead session — stop it and let the wind-down's
            # pushes land before the session goes away (figfuzz k05).
            self._minimize_stop.set()
            self._minimize_idle.wait(timeout=5.0)
        try:
            entry["session"].stop()
        except Exception:  # pragma: no cover - defensive
            pass
        with self._scene_lock:
            dropped = self._scene_selection.pop(mid, None) is not None
        if self._active_model_id == mid:
            self._active_model_id = self._models[-1]["id"] if self._models else None
            active = self._model_entry(self._active_model_id) if self._active_model_id else None
            self._wire_active(active["session"] if active else None)
            self._rebind_kernel_session()
        self._prune_group(entry["group"])
        self._reload_viewport()
        self._emit_loaded_changed()
        if dropped:
            self._emit_scene_selection()


    # -- volumes --

    def _write_display_map(self, vid: str, data) -> None:
        """Write the copy of a map the browser fetches, in the frame the viewer draws in.

        Not the frame the map came from: once a map is paired with a model, cctbx has
        shifted both into a common working frame and the model is drawn there, so the
        map has to be written there too (see ``VolumeData.write_map``). Saving a map for
        the user is a different job and keeps the original frame.
        """
        vols_dir = self._webapp.volume_dir / "vols"
        vols_dir.mkdir(parents=True, exist_ok=True)
        data.write_map(str(vols_dir / f"{vid}.map"), working_frame=True)
        # The write can precede registration (_add_volume writes before it
        # appends), so a not-yet-there entry is expected.
        entry = self._volume_entry(vid, missing_ok=True)
        if entry is not None:
            # The file the derived copies were carved from just changed (a mask,
            # a frame shift) — re-derive them or they keep drawing the
            # pre-rewrite density.
            had_derived = bool(entry.pop("downsample_urls", None))
            if had_derived or entry.get("box"):
                if entry.get("box"):
                    self._write_boxed_display_map(entry)
                factor = int(entry.get("localres_downsample") or 1)
                if factor > 1:
                    self._downsample_url(entry, factor)


    def _display_map_data(self, entry):
        """The map the browser should fetch: the real one, or a masked copy of it."""
        radius = entry.get("mask_radius")
        if not radius:
            return entry["data"]
        mmm = self.group_mmm(entry.get("group"))
        if mmm is None or mmm.model() is None:
            return entry["data"]
        from .volume_io import VolumeData, masked_map_copy

        masked = masked_map_copy(mmm, entry["data"].map_id, radius)
        vd = VolumeData.from_map_manager(masked, name=entry["data"].name)
        # Sliced to its nonzero region — the served grid's extent is what the
        # viewport's bounding sphere fits to, so reset_view should frame the
        # density that is actually drawn, not the source grid the mask kept.
        return vd.trimmed()


    def set_volume_mask(self, vid: str, radius: Optional[float]) -> None:
        """Hide density more than ``radius`` A from the model this map is paired with.

        ``None`` turns it off. Unlike the other volume controls this is not a live
        command — masking changes the map itself, so the copy the browser fetches is
        rewritten and the scene reloaded. It masks a copy: the real map keeps its
        density, so minimizing still refines against all of it.

        Needs a paired map, since "away from the molecule" has no meaning without one.
        """
        entry = self._volume_entry(vid)
        if entry is None:
            return
        radius = None if radius is None else float(radius)
        if radius is not None:
            mmm = self.group_mmm(entry.get("group"))
            if mmm is None or mmm.model() is None:
                raise ValueError("masking needs a map paired with a model")
        if entry.get("mask_radius") == radius:
            return
        entry["mask_radius"] = radius
        self._write_display_map(vid, self._display_map_data(entry))
        self._reload_viewport()
        if entry.get("color_by_resolution"):
            self._push_localres(entry)  # re-extract the colored surface from the masked grid
        self._status(
            f"{entry['name']}: masked {radius:g} A around the model" if radius
            else f"{entry['name']}: mask off")


    def can_mask_volume(self, vid: str) -> bool:
        """True when a volume is paired with a model, so masking has a meaning."""
        entry = self._volume_entry(vid)
        if entry is None:
            return False
        mmm = self.group_mmm(entry.get("group"))
        return mmm is not None and mmm.model() is not None


    def _add_volume(self, data, name: str, *, group: Optional[str] = None,
                    color: Optional[str] = None, iso: Optional[float] = None,
                    negative_color: Optional[str] = None,
                    iso_kind: str = "relative", style: str = "surface") -> str:
        """Register + show a volume: write its map (via cctbx) and compose the scene.

        ``color``/``iso`` override the defaults for maps that have a convention — a
        difference map is green at 3 sigma whatever color the palette is up to.
        ``negative_color`` draws a second contour at the negative of the level, which is
        how a difference map is read (see MAP_STYLE).

        Every map opens at the session's view radius (:meth:`set_view_radius`). It is not
        a per-caller choice: bounding is one setting for every map, and letting each
        creation path pick its own is what made a map from reflections open bounded and
        one loaded from a file not.
        """
        self._volume_counter += 1
        vid = f"volume-{self._volume_counter}"
        self._write_display_map(vid, data)
        self._volumes.append({
            "id": vid, "name": name, "data": data, "visible": True, "group": group,
            "ref": vid, "map_url": f"{self._webapp.url}vols/{vid}.map",
            "iso": data.suggested_iso() if iso is None else float(iso),
            # A given color wins (a difference map's green, or a caller's choice); else the
            # map draws a random default from the session's current palette group.
            "color": color or self._palettes.next_color(),
            "opacity": 1.0, "style": style, "clip": (0.0, 1.0), "mask_radius": None,
            # The map opens bounded by the session's view-radius default; its own
            # Appearance pane's Sphere row is the per-map override.
            "sphere": ({"mode": "view", "radius": float(self.view_radius), "center": None}
                       if self.view_radius is not None else dict(self._SPHERE_OFF)),
            "negative_color": negative_color,
            "iso_kind": iso_kind,
            # A difference map's negative contour is an object in its own right: its own
            # row in the panel, its own pane. Only the level has a tie to the map's --
            # None means "linked", which is the classic +/-n sigma reading and the
            # default. The rest start as the map's and go their own way from there.
            "negative_iso": None, "negative_visible": True,
            "negative_opacity": 1.0, "negative_style": style,
        })
        entry = self._volumes[-1]
        auto = self._auto_downsample_factor(data)
        if auto > 1:
            entry["localres_downsample"] = auto
            entry["downsample_source"] = "auto"
            self._downsample_url(entry, auto)
            self._status(
                f"{name}: large grid — drawing a {auto}x-downsampled copy "
                f"(Downsample: Full restores)")
        self._reload_viewport()  # re-asserts the clip; no session exists to tell yet
        self._emit_loaded_changed()
        return vid


    def set_volume_visible(self, vid: str, visible: bool) -> None:
        """Show or hide a volume in place — a render skip on its isosurface, no reload, so
        neither the model nor the other maps flicker and the camera stays put.

        The old "an in-place isosurface change segfaults the renderer" reading was a
        misdiagnosis — see ``set_model_visible`` for what actually crashed (an object-tree
        use-after-free, not the GPU).

        Still refused on **software** WebGL (its checkbox is non-checkable, a click flashes
        why) — a restriction kept only because it has not been re-tested since."""
        entry = self._volume_entry(vid)
        if entry is None or entry["visible"] == bool(visible):
            return
        if not self._can_hide:
            return  # software: hiding a map's isosurface segfaults; it stays shown
        entry["visible"] = bool(visible)
        control = self._control_session()
        if control is not None:
            if entry.get("color_by_resolution"):
                # The colored surface is this map's representation: the one checkbox
                # hides and shows it. The plain contour stays parked regardless.
                control.set_localres_visible(bool(visible))
            else:
                control.set_volume_visible(entry["ref"], bool(visible))  # a render skip
        self._emit_loaded_changed()


    def remove_volume(self, vid: str) -> None:
        entry = self._volume_entry(vid)
        if entry is None:
            return
        if entry.get("color_by_resolution"):
            self.set_color_by_resolution(vid, False)  # tear down the streamed surface first
        res_vid = entry.get("resolution_map")
        if res_vid is not None:
            self._remove_resolution_map(res_vid)  # the pinned resolution map goes with it
        if entry.get("is_resolution") and entry.get("pinned_to"):
            parent = self._volume_entry(entry["pinned_to"], missing_ok=True)
            if parent is not None:
                if parent.get("color_by_resolution"):
                    self.set_color_by_resolution(parent["id"], False)
                parent.pop("resolution_map", None)
        self._volumes.remove(entry)
        self._prune_group(entry["group"])
        self._reload_viewport()
        self._emit_loaded_changed()


    # -- local-resolution surface coloring ------------------------------------

    def colorable_volumes(self) -> list:
        """Maps whose surface local resolution could color — the real maps, not the hidden
        resolution maps pinned under them. ``(vid, name)`` pairs for a picker."""
        return [(v["id"], v["name"]) for v in self._volumes if not v.get("is_resolution")]


    def resolution_map_for(self, full_vid: str) -> Optional[str]:
        """The vid of the resolution map pinned to a full map, if one has been computed."""
        entry = self._volume_entry(full_vid)
        return entry.get("resolution_map") if entry else None


    def is_colored_by_resolution(self, full_vid: str) -> bool:
        """Whether a full map is currently drawn colored by its resolution map."""
        entry = self._volume_entry(full_vid)
        return bool(entry and entry.get("color_by_resolution"))


    def add_map_file(self, path) -> str:
        """Load a map file as a volume and return its id — for the Local resolution wizard,
        which may point at a full map that is not loaded yet."""
        from .volume_io import VolumeData

        return self._add_volume(VolumeData.from_map_file(str(path)), Path(path).name)


    def compute_resolution_map(self, full_vid: str, half1_path, half2_path,
                               *, color: bool = True) -> None:
        """Compute a local-resolution map from two half-maps and pin it (hidden) under a
        loaded full map, then optionally color the full map by it.

        The half-maps (``half1_path``/``half2_path``) are opened in cctbx only and never
        shown — they are inputs to the FSC, not surfaces. The computation runs on a
        background thread; the map is pinned and colored back on the GUI thread.
        """
        full = self._volume_entry(full_vid)
        if full is None:
            raise ValueError("pick a map to color")
        half1_path, half2_path = str(half1_path), str(half2_path)
        for p in (half1_path, half2_path):
            if not Path(p).is_file():
                raise ValueError(f"half-map not found: {p}")

        def work():
            from .volume_io import VolumeData, local_resolution_from_half_maps

            self._status("computing local resolution from half-maps…")
            h1 = VolumeData.from_map_file(half1_path)  # cctbx only — never reaches the viewer
            h2 = VolumeData.from_map_file(half2_path)
            res = local_resolution_from_half_maps(h1, h2, full["data"])

            def apply_on_main():
                self._pin_resolution_map(full_vid, res, color=color)

            self.bridge.run_on_main.emit(apply_on_main)

        self.run_background(work, name="pxviewer-localres", label="Local resolution")


    def _fetch_progress(self):
        """A ``fetch_entry`` progress callback that narrates into the status bar.

        Throttled: the callback fires per 64 KB chunk, which on a fast link is hundreds of
        signals a second for no added information. A quarter-second floor keeps the
        percentage moving visibly while leaving the GUI thread alone.
        """
        from . import fetch as fetchmod

        last = [0.0]

        def report(entity, stage, done, total):
            what = fetchmod.describe(entity)
            if stage == "downloading":
                now = time.time()
                if done and now - last[0] < 0.25:
                    return
                last[0] = now
                if total:
                    self._status("downloading %s… %d%% of %s"
                                 % (what, round(100.0 * done / total),
                                    fetchmod.format_bytes(total)))
                else:
                    self._status("downloading %s… %s"
                                 % (what, fetchmod.format_bytes(done)))
            elif stage == "decompressing":
                self._status("decompressing %s…" % what)
            elif stage == "cached":
                self._status("using the %s already downloaded" % what)

        return report


    @staticmethod
    def _localres_cache_path(work_dir, emdb_number, d_min) -> "Path":
        """Where a computed local-resolution map is saved beside its fetched inputs.

        The d_min it was computed at is part of the name, because it is part of the
        result -- the resolution shells move with it, not just the floor (measured on
        EMD-53478: median 8.3 A at d_min 4.2, 7.3 A at 3.0). A cache that ignored d_min
        would have kept serving results computed against cctbx's bad auto-estimate even
        after the deposited-resolution fix, which is exactly the staleness a cache must
        not add. "auto" marks the estimate fallback, so gaining a real d_min recomputes.
        """
        tag = f"{float(d_min):.2f}A" if d_min else "auto"
        return Path(work_dir) / f"emd_{emdb_number}_local_resolution_{tag}.map"


    @staticmethod
    def _cache_is_fresh(cache_path, *input_paths) -> bool:
        """Whether ``cache_path`` exists and is newer than every input it was built from.

        A plain re-fetch overwrites the half-maps (that is its point -- a deliberate
        refresh), and a resolution map computed from the old ones must not survive it.
        """
        try:
            cache_mtime = Path(cache_path).stat().st_mtime
        except OSError:
            return False
        try:
            return all(cache_mtime >= Path(p).stat().st_mtime for p in input_paths)
        except OSError:
            return False


    @staticmethod
    def _sigma_iso(volume, absolute_level) -> Optional[float]:
        """A deposited absolute contour as sigma on this map's own scale, or None.

        Stored in sigma rather than as an absolute-kind level because the whole level
        pipeline — the appearance slider (labeled σ), the scroll wheel, the live wire
        protocol — speaks sigma. An absolute-kind entry fed to that pipeline reads the
        slider's sigma numbers as absolute values: on EMD-53478 (max 0.092) a 4.5 "σ"
        becomes an absolute 4.5, the contoured surface is empty, and the Level control
        appears dead. Same surface either way; sigma is the representation the controls
        already understand.
        """
        if absolute_level is None:
            return None
        stats = volume.stats()
        std = stats["std"] or 1.0
        return (float(absolute_level) - stats["mean"]) / std


    def fetch_and_compute_resolution(self, *, pdb_id=None, emdb_number=None,
                                     color: bool = True, work_dir=None,
                                     reuse_existing: bool = False,
                                     with_model: bool = False) -> None:
        """Download a map and its two half-maps, then compute local resolution and pin it
        under the (loaded) full map — the fetch counterpart of :meth:`compute_resolution_map`,
        sharing the same computation and the same result. Everything but the loading runs on
        a background thread.

        ``with_model`` also fetches the entry's model and loads it alongside, so a caller
        that wants the whole picture (the local-resolution tutorial) gets it from one
        background job rather than racing two: the second would re-resolve the same EMDB
        number and re-download files the first is mid-way through writing.
        """
        from . import fetch as fetchmod

        target = Path(work_dir) if work_dir else self._work_dir

        def work():
            from .volume_io import VolumeData, local_resolution_from_half_maps

            label = pdb_id or (f"EMD-{emdb_number}" if emdb_number else "")
            self._status(f"fetching maps for {label}…".strip())
            entities = ["map", "half_map_1", "half_map_2"]
            if with_model and pdb_id:
                entities.insert(0, "model")   # the small one first, so something appears early
            paths = fetchmod.fetch_entry(
                entities=entities, work_dir=target,
                pdb_id=pdb_id, emdb_number=emdb_number,
                progress=self._fetch_progress(), reuse_existing=reuse_existing)
            if "model" in paths:
                model_path = str(paths["model"])
                self.bridge.run_on_main.emit(lambda: self.load_file(model_path))

            # Use the deposited resolution as d_min when we can get it. cctbx otherwise
            # derives d_min from its own estimate, which floors the result: on EMD-53478
            # that estimate is 7.7 A against a deposited 4.2 A, and every voxel comes back
            # at 6.4 A or worse. None means "no answer from the API", not "failed" -- the
            # calculation falls back to cctbx's estimate, i.e. the previous behavior.
            d_min = fetchmod.reported_resolution(pdb_id) if pdb_id else None
            # The deposited contour, so the map opens at the level its authors intend
            # rather than at a fixed sigma that is wrong for most cryo-EM maps.
            emdb = emdb_number or (fetchmod.emdb_for_pdb(pdb_id) if pdb_id else None)
            contour = fetchmod.recommended_contour(emdb) if emdb else None
            self._status("reading the maps…")
            full_map = VolumeData.from_map_file(str(paths["map"]))

            # The computed map is saved beside its inputs and reused on the next run --
            # the calculation is deterministic for (half-maps, d_min) and takes a minute
            # or two, which is the whole wait once the downloads are cached. Only under
            # reuse_existing: a plain re-fetch is a deliberate refresh, and recomputes.
            cache = (self._localres_cache_path(target, emdb, d_min)
                     if emdb is not None else None)
            if (reuse_existing and cache is not None and self._cache_is_fresh(
                    cache, paths["half_map_1"], paths["half_map_2"])):
                self._status(f"loading the local-resolution map saved earlier ({cache.name})…")
                res = VolumeData.from_map_file(str(cache))
            else:
                self._status("computing local resolution from half-maps… "
                             "(a minute or two on a full-size map)")
                res = local_resolution_from_half_maps(
                    VolumeData.from_map_file(str(paths["half_map_1"])),
                    VolumeData.from_map_file(str(paths["half_map_2"])),
                    full_map, d_min=d_min)
                if cache is not None:
                    try:
                        res.write_map(str(cache))
                        self._status(f"saved the local-resolution map to {cache}")
                    except Exception as exc:
                        # A failed save costs the next run a recompute, nothing more.
                        self._warn(f"could not save the resolution map: {_first_line(exc)}")

            added = threading.Event()

            def add_on_main():
                try:
                    self._status("adding the map to the viewport…")
                    with self._batch_load():
                        iso_sigma = self._sigma_iso(full_map, contour)
                        full_vid = self._add_volume(
                            full_map, paths["map"].name,
                            **({"iso": iso_sigma} if iso_sigma is not None else {}))
                    self._pin_resolution_map(full_vid, res, color=color)
                finally:
                    added.set()

            self.bridge.run_on_main.emit(add_on_main)
            # Hold this worker's busy label until the main-thread add has run: returning
            # now would drop the indicator while the add, the payload stream and the
            # viewport's build are all still ahead -- the reported dead air. By the time
            # the add finishes, _push_localres has opened the draw hold, which the
            # viewport's ack releases: the indicator is continuous from click to usable.
            added.wait(timeout=300)

        self.run_background(work, name="pxviewer-localres-fetch", label="Local resolution")


    def _pin_resolution_map(self, full_vid: str, res_data, *, color: bool = True) -> None:
        """Add a computed resolution map as a hidden volume pinned under ``full_vid``
        (replacing any previous one), then optionally turn on color-by-resolution."""
        full = self._volume_entry(full_vid, missing_ok=True)
        if full is None:
            return
        old = full.get("resolution_map")
        if old is not None:
            self.set_color_by_resolution(full_vid, False)
            self._remove_resolution_map(old)
        midpoint = sum(self._localres_domain(res_data)) / 2.0
        with self._batch_load():
            res_vid = self._add_volume(
                res_data, f"{full['name']} · local resolution",
                iso=midpoint, iso_kind="absolute")
            res_entry = self._volume_entry(res_vid)
            res_entry["is_resolution"] = True
            res_entry["pinned_to"] = full_vid  # nests under the full map
            # A color source, never a surface: is_resolution keeps it out of the drawn
            # scene altogether (_write_volume_scene) and out of the tree's visibility
            # checkboxes (_loaded_summary). visible=False is kept for the paths that
            # treat "hidden" generically (first-visible focus, render skips).
            res_entry["visible"] = False
        full["resolution_map"] = res_vid
        # Display resolution for the colored surface: contour every nth voxel. x4 by
        # default -- 64^3 on a typical 256^3 map, which re-levels instantly -- and the
        # user turns it up from the map's appearance pane when zoomed in. Explicit
        # rather than adaptive on purpose: what is drawn is what was asked for.
        full.setdefault("localres_downsample", 4)
        full.setdefault("localres_domain", tuple(self._localres_domain(res_data)))
        self._status(f"{full['name']}: local resolution ready")
        if color:
            self.set_color_by_resolution(full_vid, True)
        self._emit_loaded_changed()


    def _remove_resolution_map(self, res_vid: str) -> None:
        """Drop a pinned resolution map (used when replacing one, or removing its full map)."""
        entry = self._volume_entry(res_vid, missing_ok=True)
        if entry is None:
            return
        self._volumes.remove(entry)
        self._prune_group(entry.get("group"))


    def set_color_by_resolution(self, full_vid: str, on: bool) -> None:
        """Toggle coloring a full map by its pinned resolution map. A plain appearance
        option — on streams the value-colored surface (and hides the uniform isosurface it
        stands in for); off restores the ordinary contour."""
        full = self._volume_entry(full_vid)
        if full is None:
            return
        if on:
            if not full.get("resolution_map"):
                raise ValueError("no resolution map computed for this map yet")
            full["color_by_resolution"] = True
            self._push_localres(full)
        else:
            full["color_by_resolution"] = False
            session = self._control_session()
            if session is not None:
                session.clear_localres_grid()
                # The plain contour takes over as the map's representation, at the
                # visibility the map already has -- the checkbox state carries across.
                session.set_volume_visible(full["ref"], bool(full["visible"]))
        self._emit_loaded_changed()


    @staticmethod
    def _localres_domain(color_map) -> tuple:
        """A color-scale range for a local-resolution map: the 2nd–98th percentile of its
        non-zero voxels (zeros are the mask/solvent, and would swamp the scale)."""
        a = color_map.array
        finite = a[np.isfinite(a)]
        inside = finite[finite != 0.0]
        sample = inside if inside.size else finite
        if sample.size == 0:
            return (0.0, 1.0)
        lo, hi = float(np.percentile(sample, 2)), float(np.percentile(sample, 98))
        if hi <= lo:
            hi = lo + 1.0
        return (lo, hi)


    def _absolute_iso(self, entry, surface) -> float:
        """The primary map's display contour as an absolute value on its own scale — what
        marching cubes needs. A relative (sigma) level is resolved against the grid's stats
        the same way Mol*'s ``relative_isovalue`` is (mean + level·sigma)."""
        if entry.get("iso_kind") == "absolute":
            return float(entry["iso"])
        st = surface.stats()
        return float(st["mean"] + float(entry["iso"]) * st["std"])


    def _display_map_stem(self, entry: dict) -> str:
        """The base file the scene draws: the boxed copy when carved, else full."""
        return f"{entry['id']}.box" if entry.get("box") else entry["id"]

    def _display_map_stats(self, entry: dict) -> tuple:
        """(mean, std) of the full display map — what a relative contour level
        means on it. Computed on the flex array, not through numpy: a boxed
        scene write would otherwise materialize a copy of the very grid the
        box exists to avoid shipping."""
        from scitbx.array_family import flex

        from .volume_io import VolumeData

        vols_dir = self._webapp.volume_dir / "vols"
        shown = VolumeData.from_map_file(str(vols_dir / f"{entry['id']}.map"))
        mv = flex.mean_and_variance(shown.map_manager.map_data().as_1d())
        return mv.mean(), mv.unweighted_sample_standard_deviation()

    def _downsample_url(self, entry: dict, factor: int) -> str:
        """The decimated display copy's URL, writing the file on first use.

        The source is the current display base — the box when carved, else the
        full map (already in the working frame either way) — so the coarse map
        lands exactly where the full one draws."""
        urls = entry.setdefault("downsample_urls", {})
        if factor not in urls:
            from .volume_io import VolumeData

            vols_dir = self._webapp.volume_dir / "vols"
            stem = self._display_map_stem(entry)
            shown = VolumeData.from_map_file(str(vols_dir / f"{stem}.map"))
            name = f"{stem}.d{factor}.map"
            shown.decimated(factor).write_map(str(vols_dir / name))
            urls[factor] = f"{self._webapp.url}vols/{name}"
        return urls[factor]

    def _write_boxed_display_map(self, entry: dict) -> None:
        """Carve the display map's box file: the grid points inside the sphere
        ``entry["box"]`` (center in Angstrom, radius in Angstrom + a voxel of
        margin), written beside the full map as ``<id>.box.map``.

        Marching cubes pays per voxel, not per drawn triangle — a clip sphere
        shrinks the render but not the build, which is why a full-fibril mesh
        still dies clipped. Carving the served grid is clip-before-extraction:
        the mesher never sees the outside. The grid is sliced on-lattice off
        the display file, so the box lands exactly where the full map draws.
        """
        from .volume_io import VolumeData, grid_affine

        box = entry["box"]
        vols_dir = self._webapp.volume_dir / "vols"
        shown = VolumeData.from_map_file(str(vols_dir / f"{entry['id']}.map"))
        mm = shown.map_manager
        cell_angles = np.asarray(mm.unit_cell().parameters()[3:], dtype=float)
        if np.any(np.abs(cell_angles - 90.0) > 1e-3):
            self._status(f"{entry['name']}: non-orthogonal grid — cannot box it")
            entry.pop("box", None)
            return
        g0, steps = grid_affine(mm)
        inv = np.linalg.inv(steps)
        center_g = inv @ (np.asarray(box["center"], dtype=float) - g0)
        # Per-axis halfwidth in grid units: radius over the step vector's length,
        # plus one voxel so a contour right at the box face still closes.
        hw = float(box["radius"]) / np.linalg.norm(steps, axis=0) + 1.0
        dims = np.asarray(shown.grid, dtype=float)
        lo = np.maximum(0, np.floor(center_g - hw)).astype(int)
        hi = np.minimum(dims, np.ceil(center_g + hw) + 1).astype(int)
        if np.any(hi - lo < 2):
            self._status(f"{entry['name']}: box is outside the map")
            entry.pop("box", None)
            return
        arr = shown.array
        sliced = np.ascontiguousarray(
            arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
        new_origin = tuple(int(o) + int(l) for o, l in zip(shown.origin, lo))
        (VolumeData.from_numpy(sliced, spacing=shown.pixel_sizes,
                               origin=new_origin, name=f"{entry['name']} box")
         .write_map(str(vols_dir / f"{entry['id']}.box.map")))
        entry["box_url"] = f"{self._webapp.url}vols/{entry['id']}.box.map"

    def set_volume_box(self, vid: str, center=None, radius=None) -> None:
        """Carve a volume's display grid to a box around ``center`` (Angstroms)
        with half-width ``radius`` — clip before extraction.

        A clip sphere shrinks the render but not the marching-cubes build; the
        box shrinks both, which is what makes a clipped mesh on a huge map safe
        under software rendering. Everything outside the box stops existing to
        the viewer. Contour levels stay honest: the scene contours the boxed
        file at the absolute level the sigma value means on the full map, so
        the number in the pane still reads in map sigma — no boxed-statistics
        inflation. ``center=None`` clears the box back to the full grid.

        Display-only: ``entry["data"]`` keeps the full map, so analysis,
        masking and minimization are unaffected. Composes with Downsample —
        the box is what gets decimated."""
        entry = self._volume_entry(vid)
        if entry is None:
            return
        if center is None or radius is None:
            if not entry.pop("box", None):
                return
            entry.pop("box_url", None)
        else:
            entry["box"] = {"center": [float(c) for c in center],
                            "radius": float(radius)}
            self._write_boxed_display_map(entry)
            if entry.get("box") is None:
                return   # unwritable box already reported
        entry.pop("downsample_urls", None)   # the base changed — re-derive
        factor = int(entry.get("localres_downsample") or 1)
        if factor > 1:
            self._downsample_url(entry, factor)
        if not entry.get("color_by_resolution"):
            self._reload_viewport()
        self._emit_loaded_changed()

    #: Voxel count above which a map opens decimated. Marching cubes cost scales
    #: with the grid — a 400^3 map builds ~64x the wire of a 160^3 one — and a
    #: cryo-EM-sized full-res mesh is what kills the renderer process, so an
    #: oversized map opens at the factor that lands it under the cap rather than
    #: dying first and being rescued after. The cap is sized for software GL,
    #: where the margin is thinnest. Display-only: ``entry["data"]`` keeps
    #: the full grid, and Downsample: Full restores it.
    _DOWNSAMPLE_VOXEL_CAP = 2 ** 23

    def _auto_downsample_factor(self, data) -> int:
        """The factor that brings ``data``'s grid under the voxel cap (1 = fine)."""
        voxels = 1
        for n in data.grid:
            voxels *= int(n)
        if voxels <= self._DOWNSAMPLE_VOXEL_CAP:
            return 1
        return max(1, math.ceil((voxels / self._DOWNSAMPLE_VOXEL_CAP) ** (1 / 3)))

    def set_volume_downsample(self, vid: str, factor: int, _source: str = "user") -> None:
        """Set a map's display downsampling: draw every nth voxel (1 = full grid).

        The plain isosurface redraws from an exactly on-lattice decimated copy
        served beside the full map, and the color-by-resolution surface rebuilds
        client-side from the grids the browser already holds. One knob, both
        surfaces — it is the quality control for maps whose full-res mesh the
        renderer cannot carry.
        """
        entry = self._volume_entry(vid)
        if entry is None:
            return
        factor = max(1, int(factor))
        entry["localres_downsample"] = factor
        entry["downsample_source"] = _source
        session = self._control_session()
        if session is not None:
            session.set_localres_downsample(factor)   # the colored surface's path
        if factor > 1:
            self._downsample_url(entry, factor)
        if not entry.get("color_by_resolution"):
            # The plain surface is what is on screen: swap the scene to the coarser
            # (or full) file. Parked-plain maps pick the right file up on any reload.
            self._reload_viewport()
        self._emit_loaded_changed()  # the pane shows the current factor

    def set_localres_downsample(self, full_vid: str, factor: int) -> None:
        """Deprecated name for :meth:`set_volume_downsample` — the key is the wire
        protocol's history; the setting is generic to any map."""
        self.set_volume_downsample(full_vid, factor)


    #: The one busy label spanning "payload streamed" to "the viewport has drawn it".
    #: Streaming finishes long before the surface exists -- the payload is ~128 MB and
    #: the client's marching cubes over it takes seconds -- and this gap was exactly the
    #: reported dead air: model on screen at ~5s, nothing to look at until ~20s, no
    #: indicator up in between.
    _LOCALRES_DRAW_LABEL = "Drawing local resolution"


    def _begin_localres_wait(self) -> None:
        """Hold the busy indicator until the viewport reports the surface drawn."""
        from PySide6.QtCore import QTimer

        if getattr(self, "_localres_wait", False):
            return  # a newer payload supersedes the old ack; one hold covers both
        self._localres_wait = True
        self._begin_busy(self._LOCALRES_DRAW_LABEL)
        # Defensive: a lost ack (a viewport that died mid-build) must not wedge the
        # indicator forever. Generous, because slow machines legitimately take a while.
        timer = QTimer()  # parentless: DesktopApp is not a QObject; the ref below keeps it
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._end_localres_wait(timed_out=True))
        timer.start(90_000)
        self._localres_wait_timer = timer


    def _end_localres_wait(self, *, timed_out: bool = False) -> None:
        if not getattr(self, "_localres_wait", False):
            return
        self._localres_wait = False
        timer = getattr(self, "_localres_wait_timer", None)
        if timer is not None:
            timer.stop()
            self._localres_wait_timer = None
        self._end_busy(self._LOCALRES_DRAW_LABEL)
        if timed_out:
            self._warn("the viewport did not confirm the local-resolution surface")


    def _on_localres_shown(self) -> None:
        """The viewport drew the colored surface: it is on screen and usable now."""
        self._end_localres_wait()
        for entry in self._volumes:
            if entry.get("color_by_resolution"):
                entry["localres_drawn"] = True
                self._status(f"{entry['name']}: local resolution ready — "
                             "the map is colored by it")
        self._emit_loaded_changed()


    def set_localres_domain(self, full_vid: str, lo: float, hi: float) -> None:
        """Set the color ramp's value range (Angstrom): ``lo`` maps to blue, ``hi`` to
        red. Manual and stable -- it never follows the contour on its own, so a figure's
        colors keep their meaning across thresholds and sessions."""
        entry = self._volume_entry(full_vid)
        if entry is None:
            return
        lo, hi = float(lo), float(hi)
        if not (hi > lo):
            self._warn("the color range needs max above min")
            return
        entry["localres_domain"] = (lo, hi)
        session = self._control_session()
        if session is not None:
            session.set_localres_domain(lo, hi)
        self._emit_loaded_changed()


    def fit_localres_domain(self, full_vid: str) -> None:
        """Set the color range from what the current contour actually shows.

        The full-map default spends most of the ramp on solvent-adjacent voxels no
        realistic threshold displays -- on EMD-53478 the whole visible particle sits in
        the blue end. This takes the 2nd-98th percentile of the resolution values inside
        the current contour (density >= the display level), so the ramp spans the values
        on screen. A deliberate action, never automatic: refitting on contour changes
        would re-color a figure under its caption.
        """
        entry = self._volume_entry(full_vid)
        res = self._volume_entry(entry.get("resolution_map")) if entry else None
        if entry is None or res is None:
            return
        surface = self._display_map_data(entry)
        level = self._absolute_iso(entry, surface)
        inside = res["data"].array[surface.array >= level]
        inside = inside[np.isfinite(inside)]
        inside = inside[inside != 0.0]
        if inside.size < 100:
            self._warn("nothing visible at this level to fit the color range to")
            return
        lo, hi = float(np.percentile(inside, 2)), float(np.percentile(inside, 98))
        if hi <= lo:
            hi = lo + 0.1
        self.set_localres_domain(full_vid, round(lo, 2), round(hi, 2))
        self._status(f"color range fitted to the visible surface: {lo:.2f}–{hi:.2f} Å")


    def reset_localres_domain(self, full_vid: str) -> None:
        """Restore the default color range: percentiles of the whole resolution map."""
        entry = self._volume_entry(full_vid)
        res = self._volume_entry(entry.get("resolution_map")) if entry else None
        if entry is None or res is None:
            return
        lo, hi = self._localres_domain(res["data"])
        self.set_localres_domain(full_vid, round(lo, 2), round(hi, 2))


    def _push_localres(self, full) -> None:
        """(Re)stream a full map's color-by-resolution surface and hide its plain isosurface.

        Sourced from the resolution map pinned under the full map. Cheap to re-run whenever
        the surface the browser draws changes (a new contour level, a mask), so the streamed
        surface and its stored replay stay current.
        """
        if not full.get("color_by_resolution"):
            return
        res = self._volume_entry(full.get("resolution_map"))
        if res is None:
            return
        from .volume_io import encode_localres

        surface = self._display_map_data(full)  # the same (masked) grid the browser draws
        # The stored domain, initialized once: recomputing per push would let the color
        # mapping drift on its own, and a mapping that shifts under a figure between
        # sessions or contours is exactly what the explicit Color range control forbids.
        domain = full.get("localres_domain")
        if domain is None:
            domain = self._localres_domain(res["data"])
            full["localres_domain"] = domain
        payload = encode_localres(
            surface.map_manager, res["data"].map_manager,
            iso_level=self._absolute_iso(full, surface), domain=domain)
        session = self._control_session()
        if session is not None:
            # Factor first: it must be in place when the payload's first build runs.
            session.set_localres_downsample(int(full.get("localres_downsample", 4)))
            full["localres_drawn"] = False
            self._begin_localres_wait()   # released by the viewport's localres-shown ack
            session.show_localres_grid(payload)
            # The colored surface *is* the map now: the plain contour steps aside at the
            # ref level (the entry's own visible flag is the map's one checkbox and stays
            # what the user set), and the colored surface takes that visibility over.
            session.set_volume_visible(full["ref"], False)
            session.set_localres_visible(bool(full["visible"]))


    # -- fetch from the PDB / EMDB ---------------------------------------------

    def work_dir(self):
        """The directory fetched files are downloaded into (persisted across runs)."""
        return self._work_dir


    def set_work_dir(self, path) -> None:
        """Set (and persist) the download directory."""
        self._work_dir = Path(path)
        self._settings.setValue("work_dir", str(self._work_dir))
        self._status(f"working directory: {self._work_dir}")


    def fetch_and_load(self, *, pdb_id=None, emdb_number=None, entities, work_dir=None) -> None:
        """Download an entry's model, reflections and/or map into the working directory and
        load them as objects. Runs on a background thread.

        Half-maps are deliberately not a general fetch target — nobody wants two half-maps
        sitting in the viewer; they are inputs to local resolution, which the Local
        resolution wizard fetches and consumes in cctbx without ever showing them.
        """
        from . import fetch as fetchmod

        entities = [e for e in entities if e in ("model", "reflections", "map")]
        target = Path(work_dir) if work_dir else self._work_dir

        def work():
            from .volume_io import VolumeData

            label = pdb_id or (f"EMD-{emdb_number}" if emdb_number else "")
            self._status(f"fetching {label}…".strip())
            paths = fetchmod.fetch_entry(entities=entities, work_dir=target,
                                         pdb_id=pdb_id, emdb_number=emdb_number,
                                         progress=self._fetch_progress())
            full_map = (VolumeData.from_map_file(str(paths["map"]))
                        if "map" in paths else None)
            emdb = emdb_number or (fetchmod.emdb_for_pdb(pdb_id) if pdb_id else None)
            contour = (fetchmod.recommended_contour(emdb)
                       if (full_map is not None and emdb) else None)

            def add_on_main():
                names = []
                with self._batch_load():
                    if full_map is not None:
                        iso_sigma = self._sigma_iso(full_map, contour)
                        vid = self._add_volume(
                            full_map, paths["map"].name,
                            **({"iso": iso_sigma} if iso_sigma is not None else {}))
                        # The archive file IS on disk now — restore reads it as a
                        # plain file rather than re-downloading. The ids stay as
                        # metadata for anything that wants the remote provenance.
                        self._volume_entry(vid)["source"] = {
                            "kind": "file",
                            "path": str(Path(paths["map"]).resolve()),
                            "fetched": {"pdb_id": pdb_id,
                                        "emdb_number": emdb_number}}
                        names.append(paths["map"].name)
                    if "model" in paths:
                        self._load_model_file(str(paths["model"]))
                        names.append(paths["model"].name)
                    if "reflections" in paths:
                        self._load_reflection_file(str(paths["reflections"]))
                        names.append(paths["reflections"].name)
                self._status(f"fetched {', '.join(names)} into {target}")
                self._emit_loaded_changed()

            self.bridge.run_on_main.emit(add_on_main)

        self.run_background(work, name="pxviewer-fetch", label="Fetching")


    def remove_reflections(self, rid: str) -> None:
        """Unload a reflection file. Nothing is drawn from it, so nothing to reload."""
        entry = self._reflection_entry(rid)
        if entry is None:
            return
        self._reflections.remove(entry)
        self._prune_group(entry["group"])
        self._emit_loaded_changed()


    # -- groups --

    def remove_group(self, gid: str) -> None:
        """Unload a whole group (its model + maps + data) at once.

        Every member is named up front: removing them one at a time dissolves the group on
        the way through (see :meth:`_prune_group`), which clears the survivors' group and
        would leave anything enumerated afterwards unowned and un-removed.
        """
        models = [m["id"] for m in self._models if m["group"] == gid]
        volumes = [v["id"] for v in self._volumes if v["group"] == gid]
        reflections = [r["id"] for r in self._reflections if r["group"] == gid]
        with self._batch_load():
            for mid in models:
                self.remove_model(mid)
            for vid in volumes:
                self.remove_volume(vid)
            for rid in reflections:
                self.remove_reflections(rid)


    def _prune_group(self, gid: Optional[str]) -> None:
        """Dissolve a group once it no longer holds more than one object.

        A group says "these belong together". With one member left — its partner unloaded —
        it says nothing, and leaving it puts a lone object under a header naming a pairing
        that no longer exists. So the survivor is set loose and the group goes.
        """
        if gid is None:
            return
        members = [e for e in (*self._models, *self._volumes, *self._reflections)
                   if e.get("group") == gid]
        if len(members) < 2:
            for entry in members:
                entry["group"] = None
            self._groups.pop(gid, None)


    def _clear_all(self) -> None:
        """Stop and drop every model, volume, group, and the dummy control session."""
        # Drain in-flight model work first: a held minimize or a live drag
        # keeps pushing to a session long after its model is gone otherwise.
        self._tug = None
        self._tug_model = None
        self._tug_session = None
        self._tug_cache = None
        if not self._minimize_idle.is_set():
            self._minimize_stop.set()
            self._minimize_idle.wait(timeout=5.0)
        for m in list(self._models):
            try:
                m["session"].stop()
            except Exception:  # pragma: no cover - defensive
                pass
        self._models.clear()
        self._volumes.clear()
        self._reflections.clear()
        self._groups.clear()
        self._active_model_id = None
        self._rebind_kernel_session()
        with self._scene_lock:
            self._scene_selection.clear()
        if self._dummy is not None:
            try:
                self._dummy.stop()
            except Exception:  # pragma: no cover - defensive
                pass
            self._dummy = None
        self._session = None
        self._session_key = None
        try:
            # Without a reload the page keeps painting the last scene forever —
            # dots and all — and nothing the user presses can clear a dead frame.
            # Skipped at teardown: the viewport is being closed next anyway, and a
            # reload on an empty scene lazily starts a new dummy session — a server
            # thread created during shutdown that nothing stops.
            if not self._stopped:
                self._reload_viewport()
        except Exception:  # pragma: no cover - defensive (called during shutdown)
            pass


    def _status(self, text: str) -> None:
        self.bridge.status_changed.emit(text)
        if self._headless:
            # No status line to flash — narrate on stdout instead, which is
            # also where an MCP client's logs land.
            print(f"pxviewer: {text}", flush=True)


    def _warn(self, text: str) -> None:
        """A status message that briefly stands out — for when an action was refused and
        the user needs to know why (not just a quiet nothing-happened)."""
        self.bridge.status_warned.emit(text)
        if self._headless:
            print(f"pxviewer: {text}", flush=True)


    def _on_model_selection(self, mid: str, selection) -> None:
        """A model reported its picked atoms (WS thread). Fold into the scene selection."""
        with self._scene_lock:
            # selection.order carries click order when the frontend supplied it;
            # vertex sequence is semantic for angle/dihedral measures and edits.
            indices = list(getattr(selection, "order", None) or selection.indices)
            if indices:
                self._scene_selection[mid] = indices
            else:
                self._scene_selection.pop(mid, None)
        self._emit_scene_selection(changed=mid)


    def selection_description(self, scene: Optional[dict] = None, *, limit: int = 6) -> str:
        """Describe a selection at its semantic level: complete residues or individual atoms."""
        if scene is None:
            with self._scene_lock:
                scene = {mid: list(indices) for mid, indices in self._scene_selection.items()}
        total = sum(len(indices) for indices in scene.values())
        if not total:
            return "None"

        selected = []
        residue_groups = {}
        residue_order = []
        for mid, indices in scene.items():
            entry = self._model_entry(mid)
            model = getattr(entry["session"], "model", None) if entry else None
            if model is None:
                continue
            atoms = model.get_hierarchy().atoms()
            for index in indices:
                if index < 0 or index >= len(atoms):
                    continue
                atom = atoms[index]
                atom_group = atom.parent()
                residue_group = atom_group.parent()
                chain = residue_group.parent()
                key = (mid, chain.id, residue_group.resseq, residue_group.icode)
                if key not in residue_groups:
                    residue_groups[key] = {
                        "entry": entry, "chain": chain.id.strip() or "—",
                        "resname": atom_group.resname.strip(),
                        "resseq": residue_group.resseq.strip(),
                        "icode": residue_group.icode.strip(),
                        "atoms": [], "size": len(residue_group.atoms()),
                    }
                    residue_order.append(key)
                residue_groups[key]["atoms"].append(atom)
                selected.append((entry, atom, atom_group, residue_group, chain))

        # Ribbon picks arrive as every atom in each residue. Present the chemistry the user
        # selected, not the atom-array representation used to transmit it.
        complete = bool(residue_order) and all(
            len(residue_groups[key]["atoms"]) == residue_groups[key]["size"]
            for key in residue_order)
        if complete:
            n_residues = len(residue_order)
            lines = [
                f"{n_residues} residue{'s' if n_residues != 1 else ''} selected"
                f" · {len(selected)} atoms"
            ]
            by_chain = {}
            chain_order = []
            for key in residue_order:
                residue = residue_groups[key]
                chain_key = (residue["entry"]["name"], residue["chain"])
                if chain_key not in by_chain:
                    by_chain[chain_key] = []
                    chain_order.append(chain_key)
                resid = residue["resseq"] + residue["icode"]
                by_chain[chain_key].append(f"{residue['resname']} {resid}")
            for model_name, chain_id in chain_order:
                residues = by_chain[(model_name, chain_id)]
                shown = ", ".join(residues[:limit])
                if len(residues) > limit:
                    shown += f", … +{len(residues) - limit}"
                lines.append(f"{model_name} · chain {chain_id}: {shown}")

            atoms = [record[1] for record in selected]
            xs, ys, zs = zip(*(atom.xyz for atom in atoms))
            bs = [float(atom.b) for atom in atoms]
            occs = [float(atom.occ) for atom in atoms]
            lines.append(
                f"Center {sum(xs) / len(xs):.2f}, {sum(ys) / len(ys):.2f}, "
                f"{sum(zs) / len(zs):.2f} Å")
            occupancy = (f"{occs[0]:.2f}" if max(occs) - min(occs) < 0.005
                         else f"{min(occs):.2f}–{max(occs):.2f}")
            lines.append(
                f"B-factor mean {sum(bs) / len(bs):.2f} "
                f"({min(bs):.2f}–{max(bs):.2f}) · occupancy {occupancy}")
            return "\n".join(lines)

        lines = [f"{len(selected)} atom{'s' if len(selected) != 1 else ''} selected"]
        shown = 0
        for entry, atom, atom_group, residue_group, chain in selected:
            if shown >= limit:
                break
            residue = f"{atom_group.resname.strip()} {residue_group.resseq.strip()}"
            if residue_group.icode.strip():
                residue += residue_group.icode.strip()
            altloc = f" alt {atom_group.altloc.strip()}" if atom_group.altloc.strip() else ""
            lines.append(
                f"{entry['name']} · chain {chain.id.strip() or '—'} · {residue} · "
                f"{atom.name.strip()}{altloc} ({atom.element.strip() or '—'})")
            x, y, z = atom.xyz
            lines.append(
                f"  xyz {x:.3f}, {y:.3f}, {z:.3f} · occ {atom.occ:.2f} · B {atom.b:.2f}")
            shown += 1
        if len(selected) > shown:
            lines.append(f"…and {len(selected) - shown} more")
        return "\n".join(lines)


    def _emit_scene_selection(self, changed: Optional[str] = None) -> None:
        """Publish the scene selection; ``changed`` names the model whose picks moved."""
        with self._scene_lock:
            snapshot = {k: list(v) for k, v in self._scene_selection.items()}
        self.bridge.scene_selection_changed.emit({"scene": snapshot, "changed": changed})


    def _emitted_items(self) -> list:
        """The Loaded-tree items as published (models, volumes, reflections)."""
        return self._loaded_summary()["items"]


    def _emit_loaded_changed(self) -> None:
        """Publish the Loaded tree: groups + flat items (models, volumes, reflections)."""
        if self._batching:
            return
        self.bridge.loaded_changed.emit(self._loaded_summary())


    def _loaded_summary(self) -> dict:
        items = [
            {"kind": "model", "id": m["id"], "name": m["name"], "visible": m["visible"],
             "active": m["id"] == self._active_model_id, "group": m["group"], "rep": m.get("rep"),
             "color": m.get("color"), "interactions": m.get("interactions", False),
             "types": list(self._type_groups(m).keys()), "hidden_types": sorted(m.get("hidden_types") or []),
             "conformers": self._conformers_of(m), "conformer": m.get("conformer"),
             "has_restraints_cif": bool(m.get("restraints_cif")),
             "restraints_built": _restraints_built(m),
             "value_domain": ((m.get("attribute") or {}).get("domain")
                              if (m.get("attribute") or {}).get("name") in _MODEL_VALUE_COLORS
                              and m.get("color") == (m.get("attribute") or {}).get("name")
                              else None),
             "edits": _edit_summaries(m.get("edits"))}
            for m in self._models
        ] + [
            # One row per map. The resolution dataset is deliberately absent: it is a
            # color source, not a drawable object -- its user-facing existence is the
            # "Color by local resolution" group on its map's pane, and its lifecycle
            # rides its map. (It also lives on disk beside the fetched files, loadable
            # as an ordinary map by anyone who wants to *see* it.)
            {"kind": "volume", "id": v["id"], "name": v["name"], "visible": v["visible"],
             "active": False, "group": v["group"], "style": v.get("style"),
             "color": v.get("color"), "opacity": v.get("opacity"), "iso": v.get("iso"),
             "pinned_to": v.get("pinned_to"), "is_resolution": bool(v.get("is_resolution")),
             "resolution_map": v.get("resolution_map"),
             "color_by_resolution": bool(v.get("color_by_resolution")),
             "localres_downsample": v.get("localres_downsample"),
             "localres_domain": v.get("localres_domain")}
            for v in self._volumes if not v.get("is_resolution")
        ] + [
            # A difference map's negative contour, as its own row nested under its map.
            #
            # It used to have none: a second contour drawn from the map's own row, which
            # left density on screen that no row owned, no eye could hide on its own and
            # no Level slider reached -- "there are maps visible that no level slider
            # affects, and I can't tell which is which". One drawn thing, one row.
            {"kind": "negative", "id": v["id"], "name": "negative contour",
             "visible": v.get("negative_visible", True), "active": False,
             "group": v["group"], "pinned_to": v["id"],
             "style": v.get("negative_style"), "color": v.get("negative_color"),
             "opacity": v.get("negative_opacity"),
             "iso": v["iso"] if v.get("negative_iso") is None else v["negative_iso"],
             "linked": v.get("negative_iso") is None}
            for v in self._volumes
            if v.get("negative_color") and not v.get("is_resolution")
        ] + [
            # visible=None: not drawable, so the tree gives it no visibility box.
            {"kind": "reflections", "id": r["id"], "name": r["name"], "visible": None,
             "active": False, "group": r["group"], "summary": r["data"].summary(),
             "labels": list(r["data"].labels),
             "has_map_coefficients": r["data"].has_map_coefficients,
             "r_work": r.get("r_work"), "r_free": r.get("r_free")}
            for r in self._reflections
        ] + [
            {"kind": "marker", "id": m["id"], "name": m["name"], "visible": m["visible"],
             "active": False, "group": None, "position": m["position"], "atom": m["atom"]}
            for m in self._markers
        ]
        groups = [{"id": gid, "name": g["name"], "label": g.get("label", "")}
                  for gid, g in self._groups.items()]
        return {"groups": groups, "items": items}


    def session_for(self, mid: Optional[str]):
        """The LiveSession for a model id (or None) — used by the atoms table."""
        entry = self._model_entry(mid, missing_ok=True) if mid else None
        return entry["session"] if entry else None


    def active_model_session(self):
        """The active model's LiveSession, or None (e.g. a volume scene)."""
        return self.session_for(self._active_model_id)


    # -- the shared python kernel -----------------------------------------

    @staticmethod
    def _make_gui_invoker():
        """A QObject whose signal runs a callable on the Qt thread.

        Emitting ``run_job`` from any thread lands ``fn`` here via a queued
        connection; the Future carries its result (or exception) back.
        """
        from PySide6.QtCore import QObject, Signal

        class _Invoker(QObject):
            run_job = Signal(object)

            def __init__(self):
                super().__init__()
                self.run_job.connect(self._run)

            def _run(self, job):
                fn, fut = job
                if fut.set_running_or_notify_cancel():
                    try:
                        fut.set_result(fn())
                    except BaseException as exc:
                        fut.set_exception(exc)

        return _Invoker()


    def invoke_on_gui(self, fn):
        """Run ``fn`` on the Qt thread; returns a ``Future`` for its result."""
        from concurrent.futures import Future

        fut = Future()
        self._gui_invoker.run_job.emit((fn, fut))
        return fut


    def ensure_kernel(self):
        """Start (or return) the app's python kernel. Call on the Qt thread.

        The kernel is the shared exec namespace — Console tab, MCP ``python``
        tool and external Jupyter clients all attach to it. ``app`` is bound
        through a GuiThreadProxy so calls marshalled back here stay on the
        Qt thread; ``session`` (thread-safe) is bound raw.
        """
        if self._kernel is not None:
            return self._kernel
        from . import console as console_mod

        if not console_mod.console_available():
            return None
        from .api_guide import ApiGuide

        namespace = {
            "app": console_mod.GuiThreadProxy(self, self.invoke_on_gui),
            "session": self.active_model_session(),
            "api": ApiGuide.full(),
            # Over the builtin, and taking nothing from it: bare help() is the
            # console's guide, help(obj) is still Python's (see ConsoleHelp).
            "help": console_mod.ConsoleHelp(),
        }
        self._kernel = console_mod.AppKernel(namespace)
        print("pxviewer python kernel — attach with:\n"
              f"  jupyter console --existing {self._kernel.connection_file}",
              flush=True)
        return self._kernel


    def kernel_push(self, mapping) -> None:
        """Bind names in the kernel namespace, if a kernel has been started."""
        if self._kernel is not None:
            self._kernel.push(mapping)


    def _rebind_kernel_session(self) -> None:
        """Keep the kernel's ``session`` name tracking the active model.

        The namespace binds ``session`` once at kernel start — usually to None
        before anything is loaded — so every change of ``_active_model_id``
        re-pushes it. Console, MCP ``python(code)`` and attached Jupyter
        clients share the one namespace and must all see the live session.
        """
        self.kernel_push({"session": self.active_model_session()})


    def shutdown_kernel(self) -> None:
        """Detach the app's exec channel from the python kernel.

        Only our client detaches — ``IPKernelApp`` is a process-lifetime
        singleton (its sockets can't init twice), so the kernel keeps running
        and serving other clients, and a later ``python`` exec or
        :meth:`ensure_kernel` re-attaches a fresh client to the same
        namespace (names pushed earlier survive). ``status``'s ``kernel``
        field reports this attachment, not kernel liveness.

        The channel stop is deferred a beat: called through the kernel's own
        exec channel — the MCP ``python`` tool — stopping it synchronously
        severs the very reply the caller is waiting on, and a successful
        detach returns as a timeout."""
        kernel, self._kernel = self._kernel, None
        if kernel is not None:
            threading.Timer(0.25, kernel.shutdown).start()


    # -- loading ---------------------------------------------------------

    def load_file(self, path: str) -> str:
        """Open a single local model or volume file (individually). Returns its kind.

        Everything is read by cctbx: models stream through a live session, maps go
        through cctbx's map_manager. To load a map + model *as a group*, use
        :meth:`load_files` with both paths.
        """
        kind = file_kind(path)
        if kind == "volume":
            return self._load_volume_file(path)
        if kind == "reflections":
            return self._load_reflection_file(path)
        return self._load_model_file(path)


    def load_files(self, paths) -> str:
        """Load one or more files. A single file loads individually; several are
        handed to cctbx as one ``map_model_manager`` and shown as a group."""
        paths = [str(p) for p in paths]
        if len(paths) == 1:
            return self.load_file(paths[0])
        return self._load_group(paths)


    def _model_session(self, model, name: str):
        """Build a live session from a cctbx model (styled by _add_model)."""
        from .live import LiveSession

        return LiveSession.from_cctbx_model(model)


    def models_for_phasing(self) -> list:
        """Models that could phase a set of reflections: unpaired, and really models.

        Unpaired because the maps that come out belong with the model that phased them,
        and they go into a manager together — a model already in one cannot be moved out
        from under it.
        """
        return [m for m in self._models
                if not self._is_paired(m)
                and getattr(m["session"], "model", None) is not None]


    def make_maps(self, rid: str, mid: str) -> None:
        """Compute density from reflections and a model, and pair the result with it.

        The phases come from the model, so these maps and that model are inseparable:
        they go into one map_model_manager, which is also what makes them usable together
        — masking, and minimizing into the density.

        Runs on a background thread (scaling and two transforms), and adds its results on
        the GUI thread.
        """
        rentry = self._reflection_entry(rid)
        mentry = self._model_entry(mid)
        if rentry is None or mentry is None:
            raise ValueError("pick reflections and a model")
        if rentry["data"].has_map_coefficients:
            raise ValueError("this file already carries maps; nothing to phase")
        if rentry["data"].path is None:
            raise ValueError("these reflections did not come from a file")
        if self._is_paired(mentry):
            raise ValueError("that model is already paired with something")
        model = getattr(mentry["session"], "model", None)
        if model is None:
            raise ValueError("that object has no cctbx model to phase with")

        def work():
            from iotbx.map_model_manager import map_model_manager

            from .reflections import DIFFERENCE_MAP_TYPES, MAP_STYLE, phased_maps
            from .volume_io import VolumeData

            try:
                self._status(f"phasing {rentry['name']} against {mentry['name']}…")
                out = phased_maps(model, rentry["data"].path)
            except Exception as exc:  # pragma: no cover - cctbx/data errors
                self._status(f"could not compute maps: {exc}")
                return

            def add_on_main():
                # Phasing took a second on a thread; the model or reflections may have
                # been unloaded in the meantime. The maps belong to objects that no
                # longer exist, so drop them rather than leave a group paired to a model
                # that is gone. (Runs on the GUI thread, as does the unload, so once this
                # starts the check cannot be undercut.)
                if (self._model_entry(mid, missing_ok=True) is None
                        or self._reflection_entry(rid, missing_ok=True) is None):
                    self._status("maps discarded: the model or reflections were unloaded")
                    return
                types = list(out["maps"])
                # The model and its maps in one manager: the phases came from it, so
                # cctbx holds them together and everything that needs a pair — masking,
                # minimizing into density — now works on them. The primary map is the
                # 2mFo-DFc one, which is the map you refine into.
                mmm = map_model_manager(model=model, map_manager=out["maps"][types[0]])
                for map_type in types:
                    # Also by its own id, so recomputing can put each new map back where
                    # the old one was. The primary keeps cctbx's 'map_manager' id too.
                    mmm.add_map_manager_by_id(out["maps"][map_type], map_type)
                # Reuse the group the model and its reflections are already shown in (they
                # are grouped from the moment they load together); phasing just gives that
                # group its manager and its maps. Only make a new one if they were loaded
                # apart and are meeting here for the first time.
                gid = mentry.get("group") or rentry.get("group")
                if gid is None:
                    gid = self._new_group(f"{mentry['name']} + {rentry['name']}")
                self._groups[gid]["mmm"] = mmm
                self._groups[gid]["label"] = "phased from reflections"
                mentry["group"] = gid
                rentry["group"] = gid
                # Before the batch: leaving it opens _emit_loaded_changed publishes the
                # summary, and the pane reads the fit from there.
                rentry["r_work"] = out["r_work"]
                rentry["r_free"] = out["r_free"]
                with self._batch_load():
                    for map_type in types:
                        is_diff = map_type in DIFFERENCE_MAP_TYPES
                        # Canonical colors on purpose (blue 2mFo-DFc, green/red mFo-DFc):
                        # X-ray maps carry Coot-bred meaning, and a random palette color
                        # here clashed with the live recalc's conventions. The color
                        # controls stay authoritative afterwards.
                        color, iso, negative = MAP_STYLE[is_diff]
                        self._add_volume(
                            VolumeData.from_map_manager(
                                mmm.get_map_manager_by_id(map_type),
                                name=map_type, map_id=map_type),
                            map_type, group=gid, color=color, iso=iso,
                            negative_color=negative,
                            style="mesh")  # chickenwire: difference lobes stay visible
                self._status(
                    f"{rentry['name']}: R-work {out['r_work']:.4f}, R-free {out['r_free']:.4f}"
                    f" — maps: {', '.join(types)}")

            self.bridge.run_on_main.emit(add_on_main)

        self.run_background(work, name="pxviewer-phasing", label="Computing maps")


    def reflections_for_model(self, mid: Optional[str] = None) -> Optional[dict]:
        """The reflections this model was phased against, if it was."""
        entry = self._model_entry(self._active_model_id if mid is None else mid)
        if entry is None or entry.get("group") is None:
            return None
        return next((r for r in self._reflections
                     if r["group"] == entry["group"] and r.get("r_work") is not None), None)


    def update_maps(self, rid: str) -> None:
        """Recompute the density from the model as it now stands.

        This is what keeping the reflections is *for*. The moment the model moves — a
        minimization, a flip, anything — the maps describe a model that no longer
        exists. The difference map especially: it is the answer to "what does the
        density have that the model does not", and after the model moves it is the
        answer to that question about the old one.

        The maps are replaced in place, so a level, a color or a radius set on them
        survives. Runs on a background thread.
        """
        rentry = self._reflection_entry(rid)
        if rentry is None or rentry.get("r_work") is None:
            raise ValueError("these reflections have not been phased against a model")
        mmm = self.group_mmm(rentry.get("group"))
        if mmm is None or mmm.model() is None:
            raise ValueError("no model to phase against")
        model = mmm.model()
        volumes = [v for v in self._volumes if v["group"] == rentry["group"]]

        def work():
            from .reflections import PHASED_MAP_TYPES, phased_maps
            from .volume_io import VolumeData

            try:
                self._status(f"recomputing maps from {rentry['name']}…")
                out = phased_maps(model, rentry["data"].path)
            except Exception as exc:  # pragma: no cover - cctbx/data errors
                self._status(f"could not update maps: {exc}")
                return

            def swap():
                refreshed = []
                for entry in volumes:
                    map_type = entry["data"].map_id
                    fresh = out["maps"].get(map_type)
                    if fresh is None:
                        continue
                    mmm.add_map_manager_by_id(fresh, map_type)
                    if map_type == PHASED_MAP_TYPES[0]:
                        # The primary as well: it is what minimizing refines into, and
                        # refining into stale density would undo the point of this.
                        mmm.set_map_manager(fresh)
                    entry["data"] = VolumeData.from_map_manager(
                        fresh, name=map_type, map_id=map_type)
                    self._write_display_map(entry["id"], self._display_map_data(entry))
                    refreshed.append(entry["ref"])
                rentry["r_work"] = out["r_work"]
                rentry["r_free"] = out["r_free"]
                # Fresh maps supersede the drag's standing live window (see the tug
                # 'end' handling): from here the object maps tell the current story.
                self._clear_live_diff()
                self._undim_stale_diff()  # the maps are current again
                # In place, never a page reload: a reload re-runs the scene's camera
                # fit, so the auto re-phase after every settled drag snapped the camera
                # home — the one thing a background refresh must never do. The viewer
                # re-downloads each rewritten map file and swaps the density under a
                # motionless camera (level, colors, style and clip all stand).
                control = self._control_session()
                if control is not None:
                    for ref in refreshed:
                        control.reload_volume(ref)
                else:
                    self._reload_viewport()  # no live client to command; the old way
                self._emit_loaded_changed()
                self._status(
                    f"{rentry['name']}: R-work {out['r_work']:.4f}, "
                    f"R-free {out['r_free']:.4f} — maps updated")

            self.bridge.run_on_main.emit(swap)

        self.run_background(work, name="pxviewer-rephasing", label="Updating maps")


    def _update_maps_if_live(self, rid: str) -> None:
        """Re-phase from the post-minimization auto-chain, but only if it still applies.

        A minimization emits this to run on the GUI thread once it finishes; by the time
        it lands the user may have unloaded the reflections, the model, or the whole
        group. :meth:`update_maps` raises on those (it is written for a direct UI call
        with a live pairing), and an exception here escapes into the event loop. So the
        chain checks the same preconditions first and quietly does nothing if they no
        longer hold. Safe on the GUI thread: the unload runs there too, so this cannot be
        undercut mid-check."""
        entry = self._reflection_entry(rid, missing_ok=True)
        if (entry is None or entry.get("r_work") is None
                or self.group_mmm(entry.get("group")) is None
                or self.group_mmm(entry.get("group")).model() is None):
            # No update will run, so nothing else will lift a standing suppression.
            self._restore_static_diff()
            return
        self.update_maps(rid)


    def _add_reflections(self, data, name: str, *, group: Optional[str] = None) -> str:
        """Register a reflection file. Nothing is drawn: there is nothing drawable yet."""
        self._reflection_counter += 1
        rid = f"reflections-{self._reflection_counter}"
        self._reflections.append({"id": rid, "name": name, "data": data, "group": group})
        self._emit_loaded_changed()
        return rid


    def _load_reflection_file(self, path: str) -> str:
        """Read reflections with cctbx; make their maps when the file already has them.

        A file carrying map coefficients is a refinement result, and the density is what
        it is *for* — so the maps are made on load rather than asked about, which is what
        Coot's Auto Open MTZ gets right and why it is the way most people open one. A
        file of amplitudes cannot do this: its phases have to be computed against a
        model, which is a separate step.
        """
        from .reflections import (
            MAP_STYLE, ReflectionData, is_difference_map, map_from_coefficients,
            root_label,
        )
        from .volume_io import VolumeData

        data = ReflectionData.from_file(path)
        name = Path(path).name
        if not data.has_map_coefficients:
            self._add_reflections(data, name)
            self._status(f"Loaded reflections: {name} — {data.summary()}")
            return "reflections"

        gid = self._new_group(name, label="reflections + maps")
        made = []
        with self._batch_load():
            self._add_reflections(data, name, group=gid)
            for coefficients in data.map_coefficient_arrays():
                label = coefficients.info().label_string()
                is_diff = is_difference_map(label)
                # Canonical colors, as in make_maps: the color controls take over from here.
                color, iso, negative = MAP_STYLE[is_diff]
                volume = VolumeData.from_map_manager(
                    map_from_coefficients(coefficients), name=root_label(label))
                self._add_volume(volume, root_label(label), group=gid,
                                 color=color, iso=iso, negative_color=negative,
                                 style="mesh")  # chickenwire, as Make maps makes them
                made.append(root_label(label))
        self._status(f"Loaded {name} — {data.summary()}; maps: {', '.join(made)}")
        return "reflections"


    def _load_model_file(self, path: str) -> str:
        """Read a model with cctbx and add it to the viewport (alongside any others)."""
        self.stop_demo()
        self._reset_interactions()

        from .live import LiveSession

        session = LiveSession.from_model_file(path)  # _add_model applies the default rep
        mid = self._add_model(session, Path(path).name)
        self._model_entry(mid)["source"] = {
            "kind": "file", "path": str(Path(path).resolve())}
        self._status(f"Loaded model {mid}: {Path(path).name} "
                     f"({session._n_atoms} atoms)")
        return "model"


    def _load_volume_file(self, path: str) -> str:
        """Read a map with cctbx and add it as a volume (alongside any models/maps)."""
        self.stop_demo()
        self._reset_interactions()

        from .reflections import MAP_STYLE
        from .volume_io import VolumeData

        data = VolumeData.from_map_file(path)
        # A file map with a real negative lobe is a difference map by content, whatever
        # it is named — draw it the way difference maps are read (green/red at ±3σ)
        # rather than as a positive map with half its data undrawn.
        color, iso, negative = (
            MAP_STYLE[True] if _bipolar_map(data.stats())
            else (None, None, None))
        vid = self._add_volume(
            data, Path(path).name,
            color=color, iso=iso, negative_color=negative)
        self._volume_entry(vid)["source"] = {
            "kind": "file", "path": str(Path(path).resolve())}
        self._status(f"Loaded volume {vid}: {Path(path).name}")
        return "volume"


    def _load_group(self, paths) -> str:
        """Load several files as one cctbx map_model_manager group (model + maps)."""
        self.stop_demo()
        self._reset_interactions()

        from .volume_io import map_model_manager_from_files, split_map_model_manager

        models = [p for p in paths if file_kind(p) == "model"]
        maps = [p for p in paths if file_kind(p) == "volume"]
        reflections = [p for p in paths if file_kind(p) == "reflections"]
        if len(models) > 1:
            raise ValueError("a group can contain at most one model")
        if not maps and not reflections:
            raise ValueError("a group needs at least one map or reflection file")

        if not maps:
            # A model opened with its reflections: they belong together in the panel even
            # though cctbx has no manager for them until Make maps phases them.
            before = self._object_ids()
            name = Path(models[0]).name if models else Path(reflections[0]).name
            with self._batch_load():
                for path in models:
                    self._load_model_file(path)
                for path in reflections:
                    self._load_reflection_file(path)
                self._group_loaded_together(before, f"{name} + reflections",
                                            label="not yet phased")
            self._status(f"Loaded {name} with {len(reflections)} reflection file(s)")
            return "group"

        group_name = Path(models[0]).name if models else Path(maps[0]).name
        mmm = map_model_manager_from_files(model_file=models[0] if models else None, map_files=maps)
        model_data, volumes = split_map_model_manager(mmm, name=group_name)

        # Keep the manager: it is cctbx's record that these files are paired, and the
        # only place that survives the load (get_map_model_manager empties the
        # DataManager of the model and maps it consumed).
        gid = self._new_group(group_name, mmm=mmm)
        group_source = {
            "kind": "files", "paths": [str(Path(p).resolve()) for p in paths]}
        with self._batch_load():
            if model_data is not None and model_data.model is not None:
                session = self._model_session(model_data.model, group_name)
                mid = self._add_model(session, group_name, group=gid)
                self._model_entry(mid)["source"] = dict(group_source)
            for vd in volumes:
                vid = self._add_volume(vd, vd.name, group=gid)
                self._volume_entry(vid)["source"] = dict(group_source)
        self._status(f"Loaded group: {group_name} ({len(volumes)} map(s), model={'yes' if model_data else 'no'})")
        return "group"


    def load_volume_demo(self, name: str) -> None:
        """Generate a demo map (through cctbx) and add it as a volume."""
        self.stop_demo()
        self._reset_interactions()

        from .volume_demos import make_demo_grids
        from .volume_io import VolumeData

        grids = make_demo_grids(name, shape=(32, 32, 32))
        if len(grids) == 1:
            self._add_volume(VolumeData.from_numpy(grids[0], name=name), f"demo: {name}")
        else:
            with self._batch_load():
                for i, g in enumerate(grids):
                    self._add_volume(VolumeData.from_numpy(g, name=f"{name}-{i}"), f"demo: {name} [{i}]")
        self._status(f"Volume demo: {name}")


    def load_map_model_demo(self, *, d_min: float = 3.0) -> str:
        """Test fixture: the bundled sample model + a cctbx-generated density, as one
        group. Offline and fast, which is why the regression suite builds pairs with
        it — and why no tutorial may: the density is synthetic (computed from the
        model itself, noise-free, near-perfect CC everywhere), and 1UBQ is an X-ray
        structure. Tutorials fetch real deposited pairs (fetch_map_model_pair).
        """
        self.stop_demo()
        self._reset_interactions()

        from iotbx.map_model_manager import map_model_manager

        from .cctbx_io import read_model
        from .volume_io import split_map_model_manager

        sample = sample_structure_path()
        if sample is None:
            raise FileNotFoundError("the bundled sample model is missing")

        mmm = map_model_manager(model=read_model(str(sample)))
        mmm.generate_map(d_min=d_min)  # a density computed from the model

        model_data, volumes = split_map_model_manager(mmm, name=SAMPLE_STRUCTURE[1])
        # generate_map also adds a redundant 'model_map'; keep only the density.
        volumes = [v for v in volumes if v.map_id == "map_manager"] or volumes

        gid = self._new_group(SAMPLE_STRUCTURE[1], mmm=mmm)
        with self._batch_load():
            session = self._model_session(model_data.model, SAMPLE_STRUCTURE[1])
            self._add_model(session, f"{SAMPLE_STRUCTURE[0]} (model)", group=gid)
            for vd in volumes:
                self._add_volume(vd, f"{SAMPLE_STRUCTURE[0]} (density)", group=gid)
        self._status(f"Loaded demo: {SAMPLE_STRUCTURE[1]} — map + model")
        return "group"


    def load_xray_demo(self, *, d_min: float = 2.0) -> str:
        """Demo: the bundled model plus 1ubq's REAL deposited diffraction data.

        The amplitudes are fetched from the PDB once and cached (converted to MTZ in
        the working directory), so the maps behave the way maps behave on an actual
        experiment — in particular the difference map has a real noise floor, so its
        sigma-scaled contours mean what a crystallographer expects. Offline, the demo
        falls back to amplitudes synthesized from the model with realistic noise
        (perfect synthetic data has no floor, and every model twitch went off the
        sigma scale — see _synthetic_demo_mtz). Model and reflections load unpaired:
        pairing them with "Make maps" is the demo.
        """
        self.stop_demo()
        self._reset_interactions()

        sample = sample_structure_path()
        if sample is None:
            raise FileNotFoundError("the bundled sample model is missing")

        try:
            mtz = self._deposited_demo_mtz()
            source = "deposited 1ubq data"
        except Exception as exc:
            self._status(f"deposited 1ubq data unavailable ({_first_line(exc)}) — "
                         "synthesizing amplitudes instead")
            mtz = self._synthetic_demo_mtz(sample, d_min)
            source = "synthesized amplitudes (offline)"

        # Not paired — pairing them is the demo — but loaded in one batch so the viewport
        # reloads once, and shown as one group so the pair reads as the unit it is.
        before = self._object_ids()
        with self._batch_load():
            self._load_model_file(str(sample))
            self._load_reflection_file(mtz)
            self._group_loaded_together(
                before, f"{Path(sample).name} + reflections", label="not yet phased")
        self._status(
            f"X-ray demo: {SAMPLE_STRUCTURE[0]} + {source} — open the reflections and "
            "click Make maps")
        return "xray"


    def _deposited_demo_mtz(self) -> str:
        """1ubq's deposited amplitudes as a cached MTZ in the working directory.

        Fetched once (the PDB's structure-factor CIF), converted once: amplitudes (or
        intensities, converted) plus generated free flags — the 1987 deposition
        predates free-flag bookkeeping — written next to the other fetched files
        where the user can see them. Raises on any problem; the caller falls back."""
        import os

        stem = Path(SAMPLE_STRUCTURE[0]).stem
        out = Path(self.work_dir()) / f"{stem}_data.mtz"
        if out.is_file() and out.stat().st_size > 0:
            return str(out)

        from . import fetch as _fetch

        fetched = _fetch.fetch_entry(
            entities=["reflections"], work_dir=self.work_dir(), pdb_id=stem,
            reuse_existing=True,
            progress=lambda entity, stage, done, total: self._status(
                f"fetching deposited {stem} data: {stage}"))
        from iotbx.reflection_file_reader import any_reflection_file

        arrays = any_reflection_file(str(fetched["reflections"])).as_miller_arrays(
            merge_equivalents=True)
        f_obs = next((a for a in arrays if a.is_xray_amplitude_array()), None)
        if f_obs is None:
            intensities = next((a for a in arrays if a.is_xray_intensity_array()), None)
            if intensities is None:
                raise ValueError("the deposited file carries no amplitudes or intensities")
            f_obs = intensities.f_sq_as_f()
        f_obs = f_obs.set_observation_type_xray_amplitude()
        flags = f_obs.generate_r_free_flags(fraction=0.05)
        dataset = f_obs.as_mtz_dataset(column_root_label="F")
        dataset.add_miller_array(flags, column_root_label="R-free-flags")
        dataset.mtz_object().write(str(out))
        if not (out.is_file() and out.stat().st_size > 0):  # pragma: no cover - io guard
            raise IOError("could not write the converted dataset")
        return str(out)


    def _synthetic_demo_mtz(self, sample, d_min: float) -> str:
        """Amplitudes synthesized from the model — the OFFLINE stand-in.

        With realistic noise, deliberately: perfect synthetic amplitudes give a
        sigma-scaled difference map no noise floor, so any model motion — a 0.1 A
        drag, one settle step — towered over nothing and lit the whole zone at
        3 sigma, which read as the live map being broken. Measured on this demo:
        15% noise puts R-work near 0.12 (an ordinary refinement), a held 0.15 A
        nudge shows a modest local lobe, and the resting difference map stays
        quiet. Seeded, so every run sees the same data."""
        import os
        import tempfile

        from scitbx.array_family import flex as _flex

        from .cctbx_io import read_model

        f_calc = read_model(str(sample)).get_xray_structure().structure_factors(
            d_min=d_min).f_calc()
        f_obs = abs(f_calc).set_observation_type_xray_amplitude()
        noise = np.abs(1.0 + np.random.default_rng(7).normal(0.0, 0.15, f_obs.size()))
        f_obs = f_obs.customized_copy(data=f_obs.data() * _flex.double(noise))
        f_obs = f_obs.customized_copy(sigmas=f_obs.data() * 0.15)  # sigmas to match
        flags = f_obs.generate_r_free_flags(fraction=0.05)

        # A temp dir, not auto-cleaned: make_maps reads this file back, so it has to
        # outlive the load.
        out_dir = tempfile.mkdtemp(prefix="pxviewer-xray-demo-")
        stem = Path(SAMPLE_STRUCTURE[0]).stem
        mtz = os.path.join(out_dir, f"{stem}_data.mtz")
        dataset = f_obs.as_mtz_dataset(column_root_label="F")
        dataset.add_miller_array(flags, column_root_label="R-free-flags")
        dataset.mtz_object().write(mtz)
        return mtz


    # Where the hidden ATP sits, cartesian, in the bundled model's frame — near the surface
    # with clearance from the unit-cell edges, so its difference blob doesn't wrap.
    _LIGAND_FITTING_CENTER = (18.0, 20.0, 15.0)

    _LIGAND_FITTING_CODE = "ATP"


    def load_ligand_fitting_demo(self, *, d_min: float = 2.0) -> str:
        """Demo mirroring Phenix's ligand-fitting tutorial (fit a flexible ligand into a
        difference map), self-contained. Reflections are computed from the bundled protein
        *with* an ATP placed near its surface, but only the ligand-free protein is loaded —
        so Make maps yields an mFo-DFc map with an ATP-shaped blob where the model has
        nothing. Place a marker in the blob and build/fit ATP into it."""
        import os
        import tempfile

        self.stop_demo()
        self._reset_interactions()

        from . import ligands
        from .cctbx_io import read_model

        sample = sample_structure_path()
        if sample is None:
            raise FileNotFoundError("the bundled sample model is missing")

        protein = read_model(str(sample))
        cs = protein.crystal_symmetry()
        ligand = ligands.build_ligand_model(
            self._LIGAND_FITTING_CODE, self._LIGAND_FITTING_CENTER, crystal_symmetry=cs)

        # Amplitudes from protein + ligand together; the model loaded is protein only.
        combined = protein.get_xray_structure().deep_copy_scatterers()
        combined = combined.concatenate(ligand.get_xray_structure())
        f_calc = combined.structure_factors(d_min=d_min).f_calc()
        f_obs = abs(f_calc).set_observation_type_xray_amplitude()
        f_obs = f_obs.customized_copy(sigmas=f_obs.data() * 0.05)
        flags = f_obs.generate_r_free_flags(fraction=0.05)

        out_dir = tempfile.mkdtemp(prefix="pxviewer-ligfit-demo-")  # outlives the load
        mtz = os.path.join(out_dir, "ligand_fitting_data.mtz")
        dataset = f_obs.as_mtz_dataset(column_root_label="F")
        dataset.add_miller_array(flags, column_root_label="R-free-flags")
        dataset.mtz_object().write(mtz)

        before = self._object_ids()
        with self._batch_load():
            self._load_model_file(str(sample))
            self._load_reflection_file(mtz)
            self._group_loaded_together(
                before, f"{Path(sample).name} + reflections", label="not yet phased")
        self._status(
            "Ligand-fitting demo: a ligand-free model + reflections that contain ATP — open "
            "the reflections, Make maps, then fit ATP into the mFo-DFc blob")
        return "ligand-fitting"


    def load_real_space_refinement_demo(self, *, d_min: float = 3.0, shake: float = 0.5) -> str:
        """Test fixture: a model shaken off a density computed from itself, so 'Minimize'
        with 'Into the density' has something to visibly pull back in. Offline and fast
        for the regression suite — and not for tutorials: the density is synthetic, the
        structure is X-ray, and the per-atom shake breaks Mol*'s distance-based bond
        perception (ball-and-stick renders as noise). The tutorial fetches a real
        deposited pair and displaces one segment rigidly instead
        (fetch_map_model_pair)."""
        self.stop_demo()
        self._reset_interactions()

        from iotbx.map_model_manager import map_model_manager

        from .cctbx_io import read_model
        from .volume_io import split_map_model_manager

        sample = sample_structure_path()
        if sample is None:
            raise FileNotFoundError("the bundled sample model is missing")

        mmm = map_model_manager(model=read_model(str(sample)))
        mmm.generate_map(d_min=d_min)  # a cryo-EM-resolution density from the true model
        # Shake the model off the density; putting it back is the refinement's whole job.
        model = mmm.model()
        xrs = model.get_xray_structure().deep_copy_scatterers()
        xrs.shake_sites_in_place(mean_distance=shake)
        model.set_sites_cart(xrs.sites_cart())

        model_data, volumes = split_map_model_manager(mmm, name=SAMPLE_STRUCTURE[1])
        volumes = [v for v in volumes if v.map_id == "map_manager"] or volumes

        gid = self._new_group(SAMPLE_STRUCTURE[1], mmm=mmm)
        with self._batch_load():
            session = self._model_session(model_data.model, SAMPLE_STRUCTURE[1])
            self._add_model(session, f"{SAMPLE_STRUCTURE[0]} (model, shaken)", group=gid)
            for vd in volumes:
                self._add_volume(vd, f"{SAMPLE_STRUCTURE[0]} (synthetic density)", group=gid)
        self._status(
            "Synthetic demo: a shaken model in its density — Minimize with 'Into the "
            "density' to real-space refine it back in")
        return "group"


    def fetch_map_model_pair(self, *, pdb_id: str, emdb_number: Optional[str] = None,
                             displace: Optional[tuple] = None,
                             map_opacity: Optional[float] = None,
                             map_color: Optional[str] = None,
                             reuse_existing: bool = True) -> None:
        """Fetch a deposited cryo-EM map and its model, and load them as one paired
        group — real data, cached in the working directory like every other fetch.
        Runs on a background thread; the loading is marshalled to the GUI thread.

        ``displace``, for the refinement tutorial, is ``(selection, (dx, dy, dz))``:
        the selected atoms are moved **rigidly** — one translation for the whole
        selection — so the model visibly needs refinement while every bond length
        inside the selection stays exact. (A random per-atom shake was tried and
        rejected: Mol* perceives bonds by distance, and a shaken model renders
        ball-and-stick as meaningless noise. Only the two junction bonds stretch,
        which is precisely the honest picture of a segment sitting out of its
        density.)
        """
        from . import fetch as fetchmod

        target = self._work_dir

        def work():
            from .volume_io import (VolumeData, map_model_manager_from_files,
                                    split_map_model_manager)

            label = pdb_id.upper()
            self._status(f"fetching {label} and its map…")
            paths = fetchmod.fetch_entry(
                entities=["model", "map"], work_dir=target, pdb_id=pdb_id,
                emdb_number=emdb_number, progress=self._fetch_progress(),
                reuse_existing=reuse_existing)
            emdb = emdb_number or fetchmod.emdb_for_pdb(pdb_id)
            contour = fetchmod.recommended_contour(emdb) if emdb else None
            self._status(f"pairing {label} with its map…")
            mmm = map_model_manager_from_files(
                model_file=str(paths["model"]), map_files=[str(paths["map"])])
            if displace is not None:
                selection, delta = displace
                from scitbx.array_family import flex

                model = mmm.model()
                chosen = model.selection(selection)
                sites = model.get_sites_cart()
                shift = flex.vec3_double(int(chosen.count(True)),
                                         tuple(float(v) for v in delta))
                sites = sites.set_selected(chosen, sites.select(chosen) + shift)
                model.set_sites_cart(sites)
            name = f"{label} (EMD-{emdb})" if emdb else label
            model_data, volumes = split_map_model_manager(mmm, name=name)
            volumes = [v for v in volumes if v.map_id == "map_manager"] or volumes

            added = threading.Event()

            def add_on_main():
                try:
                    gid = self._new_group(name, mmm=mmm)
                    with self._batch_load():
                        session = self._model_session(model_data.model, name)
                        suffix = " (displaced)" if displace is not None else ""
                        self._add_model(session, f"{label} model{suffix}", group=gid)
                        for vd in volumes:
                            iso_sigma = self._sigma_iso(vd, contour)
                            new_vid = self._add_volume(
                                vd, f"EMD-{emdb} map" if emdb else f"{label} map",
                                group=gid,
                                **({"iso": iso_sigma} if iso_sigma is not None else {}))
                            if map_opacity is not None:
                                # A curated scene (the refinement tutorial): the story
                                # is a model seen against its density, so the map opens
                                # translucent rather than as walls of surface.
                                self.set_volume_opacity(new_vid, float(map_opacity))
                            if map_color is not None:
                                # ...and neutral: the default palette rotates, and a
                                # randomly colored map competes with a colored model.
                                self.set_volume_color(new_vid, map_color)
                    self._status(f"{label}: deposited model and map, loaded as a pair")
                finally:
                    added.set()

            self.bridge.run_on_main.emit(add_on_main)
            added.wait()

        self.run_background(work, name="pxviewer-fetch-pair",
                            label="Fetching map and model")


    def load_model_demo(self, name: str, *, fps: float = 30.0) -> None:
        """Stream an animated model demo into the viewport."""
        demo = DEMOS.get(name)
        if demo is None:
            raise ValueError(f"unknown demo '{name}'. Available: {', '.join(DEMOS)}")

        self.stop_demo()
        self._reset_interactions()

        from . import cctbx_io
        from .live import LiveSession

        sites, labels = demo.make_sites()
        session = LiveSession.from_cctbx_model(cctbx_io.model_from_sites(sites, **labels))
        self._add_model(session, f"demo: {name}")

        base = np.asarray(sites, dtype="<f4")
        player = Player(session, base, fps=fps)
        session.on_pick(player._on_pick)
        self._player = player

        self._status(f"Model demo: {name} — waiting for the viewport…")

        self._demo_thread = threading.Thread(
            target=self._drive_demo,
            args=(demo, player, session),
            name=f"pxviewer-demo-{name}",
            daemon=True,
        )
        self._demo_thread.start()


    def _drive_demo(self, demo, player: Player, session) -> None:
        """Run a demo script once the viewport has connected. Runs off the GUI thread."""
        deadline = time.monotonic() + 30.0
        while not player.stopped and session.client_count == 0:
            if time.monotonic() > deadline:
                self._status(f"Model demo: {demo.name} — no viewport connected")
                return
            time.sleep(0.1)
        if player.stopped:
            return

        self._status(f"Model demo: {demo.name} — running")
        try:
            demo.run(player)
        except Exception as exc:  # a broken demo must not take the app down
            self._status(f"Model demo '{demo.name}' failed: {exc}")
            return
        if not player.stopped:
            self._status(f"Model demo: {demo.name} — finished")


    def stop_demo(self) -> None:
        """Stop any running model demo and wait for its thread to unwind."""
        player, thread = self._player, self._demo_thread
        self._player, self._demo_thread = None, None
        if player is not None:
            player.stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)


    # -- display ---------------------------------------------------------

    def set_computed_interactions(self, visible: bool) -> None:
        """Show or hide Mol*'s computed interaction overlay on the loaded structure."""
        self._computed_interactions_visible = bool(visible)
        if self._session is not None:
            self._session.set_computed_interactions(self._computed_interactions_visible)


    def set_focus_surroundings(self, enabled: bool) -> None:
        """Toggle Mol*'s native click-focus ball-and-stick neighborhood scene-wide."""
        self._focus_surroundings = bool(enabled)
        self._settings.setValue(
            "defaults/focus_surroundings", "true" if enabled else "false")
        self._settings.sync()
        effective = self._focus_surroundings and not self._selection_enabled
        for model in self._models:
            try:
                model["session"].set_focus_surroundings(effective)
            except Exception:  # pragma: no cover - defensive
                pass


    def _reset_interactions(self) -> None:
        """Drop the overlay on load — a freshly loaded structure starts clean."""
        if not self._computed_interactions_visible:
            return
        self._computed_interactions_visible = False
        if self._session is not None:
            self._session.set_computed_interactions(False)
        self.bridge.interactions_changed.emit(False)


    # -- selection -------------------------------------------------------

    def enable_mouse_selection(self) -> None:
        # Selection is scene-wide: arm click mode on every loaded model, so picks
        # can be made in any of them (each already has its pick handler registered).
        # Temporarily suppress click-focus: its neighborhood overlay and camera movement
        # get in the way when Shift-clicking several ribbon residues. This does not alter
        # the saved preference; leaving Pick mode restores it.
        self.set_tug_enabled(False)
        self._selection_enabled = True
        for m in self._models:
            m["session"].set_focus_surroundings(False)
            m["session"].enable_mouse_selection()


    def disable_mouse_selection(self) -> None:
        self._selection_enabled = False
        for m in self._models:
            try:
                m["session"].disable_mouse_selection()
                m["session"].set_focus_surroundings(self._focus_surroundings)
            except Exception:  # pragma: no cover - defensive
                pass


    def set_tug_enabled(self, enabled: bool) -> None:
        """Explicitly arm or disarm coordinate-changing atom drags scene-wide."""
        self._tug_enabled = bool(enabled)
        if enabled:
            self.disable_mouse_selection()
        for model in self._models:
            try:
                model["session"].set_tug_mode(self._tug_enabled)
            except Exception:  # pragma: no cover - defensive
                pass
        self._status("Refine drag enabled — drag an atom to pull and minimize"
                     if enabled else "Refine drag disabled")


    def _selection_flags(self) -> tuple:
        """The persisted selection switches — (focus, context).

        Desktop-side paths read the same truth the widgets persist on every toggle
        (``ControlsWindow._selection_flags`` reads the boxes themselves). Clip is
        absent on purpose: it is a per-object property (the Clip sphere mode), so
        :meth:`_selection_clip_armed` answers it for the model at hand, not a
        setting."""
        return tuple(
            str(self._settings.value(key, "true")).lower() == "true"
            for key in ("selection/focus_on_apply",
                        "selection/context_rep"))


    def _clip_padding(self) -> float:
        """Angstroms of clearance a selection-clip sphere adds past the selection's
        reach — the Settings pane's clip padding, so even a one-atom selection
        keeps readable neighborhood."""
        try:
            return float(self._settings.value("selection/clip_padding", 4.0))
        except (TypeError, ValueError):  # pragma: no cover - a hand-edited ini
            return 4.0


    def reapply_auto_clips(self) -> None:
        """Re-fit every model's standing selection sphere (e.g. the clip padding
        changed in Settings). Models whose Clip sphere mode is not "selection"
        are left alone — that is the user saying no."""
        with self._scene_lock:
            selections = {mid: list(ix) for mid, ix in self._scene_selection.items()}
        for mid, indices in selections.items():
            entry = self._model_entry(mid)
            if not self._selection_clip_armed(entry):
                continue
            session = entry["session"] if entry else None
            model = getattr(session, "model", None) if session else None
            if not indices or model is None:
                continue
            try:
                atoms = model.get_hierarchy().atoms()
                # same sphere _select_fragment draws — fit to the touched
                # components, not the picked atoms alone
                bounds = self._component_bounds(atoms, indices)
                if not bounds:
                    continue
                xyz = np.array([atoms[i].xyz for i in bounds], dtype=float)
                center = xyz.mean(axis=0)
                reach = float(np.linalg.norm(xyz - center, axis=1).max())
                self._apply_model_sphere(
                    entry, "selection", radius=reach + self._clip_padding(),
                    center=center, depth=entry.get("_auto_clip_depth"))
            except Exception:  # pragma: no cover - defensive
                pass


    def clear_selection(self) -> None:
        for m in self._models:
            try:
                m["session"].clear_selection()
                self._lift_selection_sphere(m)
                self._set_context_rep(m, None)
            except Exception:  # pragma: no cover - defensive
                pass
        with self._scene_lock:
            had = bool(self._scene_selection)
            self._scene_selection.clear()
        if had:
            self._emit_scene_selection()


    def highlight_atoms_in(self, mid: Optional[str], indices) -> None:
        """Highlight atoms in one model's viewer (table -> viewer selection sync)."""
        session = self.session_for(mid)
        if session is not None:
            try:
                session.highlight(list(indices))
            except Exception:  # pragma: no cover - defensive (e.g. stale indices)
                pass


    def focus_atoms_in(self, mid: Optional[str], indices) -> None:
        """Aim the viewer camera at atoms in one model (table selection -> focus)."""
        indices = list(indices)
        session = self.session_for(mid)
        if session is not None and indices:
            try:
                session.focus(indices)
            except Exception:  # pragma: no cover - defensive (e.g. stale indices)
                pass


    @staticmethod
    def _build_residue_index(model):
        """(chain id, resid) stripped -> [streamed atom index] over the model."""
        index: dict = {}
        for i, atom in enumerate(model.get_hierarchy().atoms()):
            rg = atom.parent().parent()  # atom -> atom_group -> residue_group
            index.setdefault((rg.parent().id.strip(), rg.resid().strip()), []).append(i)
        return index


    @staticmethod
    def _is_single_residue(model, atom_indices) -> bool:
        """Whether every index belongs to one residue group (altloc copies included)."""
        if model is None or not atom_indices:
            return False
        atoms = model.get_hierarchy().atoms()
        keys = set()
        for i in atom_indices:
            if not (0 <= i < len(atoms)):
                return False
            residue_group = atoms[i].parent().parent()
            chain = residue_group.parent()
            keys.add((chain.id, residue_group.resseq, residue_group.icode))
            if len(keys) > 1:
                return False
        return True


    @staticmethod
    def _frame_extents(xyz, target, up, direction):
        """Half-extents of ``xyz`` about ``target`` along the frame's right/up/view axes,
        padded for atom radii. What the viewer needs to fill the frame with the
        selection (rx, ry) and to clip a slab that just contains it (rz) -- one number
        (a bounding-sphere radius) cannot say both for anything elongated.
        """
        right = np.cross(np.asarray(direction), np.asarray(up))
        centered = np.asarray(xyz, dtype=float) - np.asarray(target, dtype=float)
        # Two pads for two jobs. The framing pad is slim -- it only keeps drawn atoms
        # off the very edge, and generous margin here is exactly "not zoomed enough"
        # (1.8 A per side around a ~6 A residue was a third of the frame). The clip pad
        # stays big enough that no sphere is sliced mid-surface by the near/far planes.
        frame_pad = 0.8
        clip_pad = 1.5
        rx = float(np.abs(centered @ right).max()) + frame_pad
        ry = float(np.abs(centered @ np.asarray(up)).max()) + frame_pad
        rz = float(np.abs(centered @ np.asarray(direction)).max()) + clip_pad
        return (max(rx, 2.0), max(ry, 2.0), max(rz, 2.0))


    @staticmethod
    def _principal_axes_orientation(model, atom_indices):
        """Camera ``(target, up, direction, radius)`` framing a selection by its
        principal axes: longest axis left-to-right, second axis up. Or ``None`` when
        the shape does not define a frame (fewer than three atoms, or a straight line).

        The generic answer for everything that is not a single amino acid -- a ligand
        reads along its length, a helix lies down its axis, a sheet turns face-on --
        close enough to the per-shape heuristics that they are not worth their code.

        Eigenvectors carry an arbitrary sign, so both axes get a deterministic one:
        ``right`` points with the selection's index order (first atom left, last atom
        right -- for a polymer that is N-terminus to C-terminus, matching the
        single-residue convention), ``up`` by its largest component, so re-selecting
        the same atoms always lands the same view.
        """
        atoms = model.get_hierarchy().atoms()
        xyz = np.array([atoms[i].xyz for i in atom_indices
                        if 0 <= i < len(atoms)], dtype=float)
        if len(xyz) < 3:
            return None
        centroid = xyz.mean(axis=0)
        centered = xyz - centroid
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
        if eigenvalues[1] < 1e-6 * max(eigenvalues[2], 1.0):
            return None  # collinear: no second axis to hang "up" on
        right = eigenvectors[:, 2]
        up = eigenvectors[:, 1]
        span = xyz[-1] - xyz[0]
        if abs(np.dot(right, span)) > 1e-6:
            right = right * np.sign(np.dot(right, span))
        elif right[np.argmax(np.abs(right))] < 0:
            right = -right
        if up[np.argmax(np.abs(up))] < 0:
            up = -up
        direction = np.cross(up, right)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        direction = direction / norm
        radius = max(float(np.linalg.norm(centered, axis=1).max()) + 2.0, 4.0)
        extents = AppCore._frame_extents(xyz, centroid, up, direction)
        return centroid, up, direction, radius, extents


    @staticmethod
    def _residue_orientation(model, atom_indices):
        """Camera ``(target, up, direction, radius)`` that shows the residue with its
        N->C backbone left-to-right and side chain up, or ``None`` when the backbone
        atoms are missing (e.g. a non-amino-acid).

        ``right`` = N->C (screen +x); ``up`` = the side-chain (Ca->Cb) component
        perpendicular to it (screen +y); the view axis is up x right.
        """
        atoms = model.get_hierarchy().atoms()
        named: dict = {}
        for i in atom_indices:
            a = atoms[i]
            named[a.name.strip()] = np.array(a.xyz, dtype=float)
        n, ca, c = named.get("N"), named.get("CA"), named.get("C")
        if n is None or ca is None or c is None:
            return None
        right = c - n
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            return None
        right /= rn
        cb = named.get("CB")
        if cb is not None:
            side = cb - ca
        else:  # glycine: approximate where the Cb would sit
            nd, cd = n - ca, c - ca
            side = -(nd / (np.linalg.norm(nd) or 1.0) + cd / (np.linalg.norm(cd) or 1.0))
        up = side - np.dot(side, right) * right
        un = np.linalg.norm(up)
        if un < 1e-6:
            return None
        up /= un
        direction = np.cross(up, right)
        dn = np.linalg.norm(direction)
        if dn < 1e-6:
            return None
        direction /= dn
        # Center on the selection's mass, not on CA: the frame's "up" is the side
        # chain, so CA sits at the *bottom* of the picture, and a CA-centered view rides
        # high by half a side chain (a tyrosine's worth, at full zoom). CA still
        # anchors the axes -- only the camera target moves to the middle of what is
        # actually shown.
        xyz = np.array([atoms[i].xyz for i in atom_indices
                        if 0 <= i < len(atoms)], dtype=float)
        com = xyz.mean(axis=0)
        radius = max(float(np.linalg.norm(xyz - com, axis=1).max()) + 2.0, 4.0)
        extents = AppCore._frame_extents(xyz, com, up, direction)
        return com, up, direction, radius, extents


    def focus_residue(self, chain: str, resid: str, *, focus: bool = True,
                      clip: bool = True, context: bool = False):
        """Select + focus a residue (by chain id and resid, MolProbity's resseq+icode
        string) on the active model — driven by a Validation or Components table
        row or space-bar navigation. Routed through the same fragment pipeline as a
        click or a typed expression, so the caller's flags (the Focus checkbox, or
        the object's Clip sphere mode) govern the framing, isolation clip and
        neighborhood context identically. Space-bar navigation keeps focus on
        and context off. Returns the equivalent selection
        expression (for the selection box), or ``None`` when the residue names no
        atoms. The residue->atom-index map is built once from the model and cached
        on the model entry."""
        entry = self._model_entry(self._active_model_id)
        if entry is None:
            return None
        model = getattr(entry["session"], "model", None)
        if model is None:
            return None
        index = entry.get("_residue_index")
        if index is None:
            index = entry["_residue_index"] = self._build_residue_index(model)
        key = (chain.strip(), resid.strip())
        atoms = index.get(key)
        if not atoms:
            return None
        self._focused_residue = key
        self._select_fragment(entry["id"], entry["session"], entry, list(atoms),
                              focus=focus, clip=clip, context=context)
        return f"chain {key[0]} and resid {key[1]}"


    def step_next(self, step: int = 1) -> None:
        """Space / Shift+Space: the next thing, where "thing" is whatever has focus.

        In a table -- validation outliers, restraints, atoms -- it is the next row, so a
        worklist can be walked from the space bar without moving the hand to the arrow
        keys. Every one of these tables already acts on its *selection* (focus that
        residue, highlight those atoms), so moving the selection is the whole gesture.

        And it stays that table's key once one has been engaged: the table you last
        worked is the thing being stepped, whether or not it still holds focus --
        clicking the viewport or another tab should not silently demote a residue
        worklist to the residue walk. The walk is only the default before any table
        has been touched; switching tables again is deliberate (a sub-tab, a row).

        It stops at the ends of a table rather than wrapping or falling through to the
        residue walk: running off the end of a worklist should be visible, and having
        the model jump at that moment would read as a bug.
        """
        view = self._focused_step_view()
        if view is not None:
            self._controls._engage_step_view(view)  # stepping a focused table engages it
        else:
            view = self._step_view        # the last table engaged, wherever focus is
        model = None
        if view is not None:
            try:
                model = view.model()
            except RuntimeError:            # its page was rebuilt under it (validation)
                self._step_view = view = None
        rows = 0 if model is None else model.rowCount()
        if rows:
            row = view.currentIndex().row()
            if row < 0:
                row = 0
            elif view.selectionModel().hasSelection():
                row = max(0, min(rows - 1, row + step))
            # else: focused but nothing selected yet -- Qt gives a freshly focused table
            # a current row without selecting it, and stepping off it would silently
            # skip the row the user can see is highlighted. Select that one first.
            view.selectRow(row)              # selection is what these tables act on
            view.scrollTo(model.index(row, 0))
            return
        self.advance_residue(step)


    def advance_residue(self, step: int = 1) -> None:
        """Move the focused residue to the next/previous one in its chain (space-bar
        navigation). With nothing focused yet, start at the first residue.

        The walk IS a selection each step, so it takes the object's Clip sphere
        mode — otherwise stepping with it armed would silently differ from
        stepping the Components table. Focus stays on regardless:
        a walk that does not move the camera is no walk. Context stays off:
        rebuilding a ball-and-stick neighborhood every keypress is too heavy
        for rapid stepping."""
        entry = self._model_entry(self._active_model_id)
        if entry is None:
            return
        model = getattr(entry["session"], "model", None)
        if model is None:
            return
        order = entry.get("_chain_order")
        if order is None:
            order = entry["_chain_order"] = self._build_chain_order(model)
        cur = self._focused_residue
        if cur is None:
            for cid, residues in order.items():
                if residues:
                    self.focus_residue(cid, residues[0])
                    return
            return
        chain, resid = cur
        residues = order.get(chain, [])
        if resid not in residues:
            if residues:
                self.focus_residue(chain, residues[0])
            return
        nxt = residues.index(resid) + step
        if 0 <= nxt < len(residues):
            self.focus_residue(chain, residues[nxt])


    @staticmethod
    def _build_chain_order(model):
        """chain id -> ordered list of resid strings, in hierarchy (sequence) order."""
        order: dict = {}
        for chain in model.get_hierarchy().chains():
            residues = order.setdefault(chain.id.strip(), [])
            for rg in chain.residue_groups():
                rid = rg.resid().strip()
                if rid not in residues:
                    residues.append(rid)
        return order


    def _clear_restraint_notations(self) -> None:
        session = self._restraint_prim_session
        if session is not None:
            for pid in self._restraint_prim_ids:
                try:
                    session.remove_primitive(pid)
                except Exception:  # pragma: no cover - defensive
                    pass
        self._restraint_prim_ids = []


    def show_restraint_notations(self, mid: Optional[str], specs) -> None:
        """Show the selected restraint rows in the viewer.

        ``specs`` is a list of ``(kind, i_seqs)``. A restraint row names a collection
        of atoms, so stepping one IS a selection: the participating atoms take the
        same pipeline as a typed or clicked selection — highlighted (so you see
        exactly *which* atoms make up the restraint, not the whole residue),
        framed and context-dressed under the Selection pane's checkboxes,
        clip-sphered while the object's Clip sphere mode is "selection". On top of
        that, bonds/angles/dihedrals get their measurement
        notation drawn (the distance line, angle arc or dihedral fan);
        chirality/planarity have no simple notation, so they show as the highlight
        alone. Multiple rows -> multiple.
        """
        session = self.session_for(mid)
        self._clear_restraint_notations()
        if session is None:
            return
        if specs:
            self.ensure_atoms_shown(mid)  # a ribbon can't show the atoms this notation marks
        self._restraint_prim_session = session
        marked: set = set()
        for i, (kind, iseqs) in enumerate(specs):
            pid = f"geomsel-{i}"
            iseqs = list(iseqs)
            marked.update(iseqs)  # mark every atom in the restraint, whatever the kind
            try:
                if kind == "bond" and len(iseqs) == 2:
                    session.add_distance(iseqs[0], iseqs[1], id=pid)
                elif kind == "angle" and len(iseqs) == 3:
                    session.add_angle(iseqs[0], iseqs[1], iseqs[2], id=pid)
                elif kind == "dihedral" and len(iseqs) == 4:
                    session.add_dihedral(iseqs[0], iseqs[1], iseqs[2], iseqs[3], id=pid)
                else:  # chirality / planarity: the highlight above is the only marking
                    continue
                self._restraint_prim_ids.append(pid)
            except Exception:  # pragma: no cover - defensive (stale indices)
                pass
        entry = self._model_entry(mid)
        if getattr(session, "model", None) is not None:
            focus, context = self._selection_flags()
            self._select_fragment(mid, session, entry, sorted(marked),
                                  focus=focus, context=context)
        else:  # a model-less session keeps the plain highlight-and-aim it had
            try:  # (empty list clears the overlay)
                session.highlight(sorted(marked))
            except Exception:  # pragma: no cover - defensive
                pass
            if marked:
                try:
                    session.focus(sorted(marked))
                except Exception:  # pragma: no cover - defensive
                    pass


    def select_object(self, mid: str) -> int:
        """Select a whole model as an *object*: every atom selected, framed by the
        viewer's own default focus.

        Distinct from :meth:`select_by_expression` on purpose — no oriented
        (principal-axes) framing, no fill-the-frame zoom, no isolation clip sphere.
        Those exist to show a named fragment the standard way; an object select means
        "work on this one", and the right view for that is exactly what Mol* gives a
        whole structure — so double-clicking a freshly opened model's row changes
        nothing visually. A standing isolation sphere from an earlier fragment
        selection is lifted, since the object view must show all of it.
        """
        self.set_active_model(mid)
        session = self.active_model_session()
        if session is None or getattr(session, "model", None) is None:
            raise ValueError("that object has no model to select")
        sel = session.select_by(selection="all")
        session.highlight(sel)
        entry = self._model_entry(mid)
        if entry is not None:
            self._lift_selection_sphere(entry)
        self._set_context_rep(entry, None)        # a whole object needs no context view
        session.focus(list(sel))                  # Mol*'s own whole-object framing
        self._on_model_selection(mid, sel)        # table + label follow
        return len(sel)


    def select_by_expression(self, text: str, *, focus: bool = True,
                             clip: bool = True, context: bool = True) -> int:
        """Resolve a cctbx/Phenix selection string on the active model and select it.

        cctbx's own atom-selection machinery turns the string into atom indices
        (raising on bad syntax); the atoms are highlighted in the viewer and fed
        into the scene selection so the atoms table + count reflect them. Returns
        the number of atoms selected. An empty string clears the model's selection.
        """
        text = (text or "").strip()
        mid = self._active_model_id
        session = self.active_model_session()
        if session is None or getattr(session, "model", None) is None:
            raise ValueError("load a model first, then enter a selection")
        entry = self._model_entry(mid)
        if not text:
            session.clear_selection()
            if entry is not None:
                # The isolation sphere was ours; clearing the selection lifts it
                # (the "selection" mode stays armed for the next one).
                self._lift_selection_sphere(entry)
            self._set_context_rep(entry, None)
            with self._scene_lock:
                dropped = self._scene_selection.pop(mid, None) is not None
            if dropped:
                self._emit_scene_selection()
            return 0
        # select_by can build the model's restraints on first use (the selection
        # grammar's structural/spatial keywords need the processed model) — a
        # real wait on a big structure, so say what it is doing before the GUI
        # sits on it. Usually the load-time warm already did it and this is free.
        built_before = _restraints_built(entry)
        if not built_before:
            self._status(f"Building geometry restraints for {entry['name']}…")
            self._pump_events()  # paint the status before the block
        sel = session.select_by(selection=text)  # cctbx; raises on invalid syntax
        if not built_before and _restraints_built(entry):
            self.bridge.restraints_changed.emit(mid)
            self._status("Ready")
        return self._select_fragment(mid, session, entry, sel,
                                     focus=focus, clip=clip, context=context)


    def select_picked_atom(self, mid: str, atom_index: int, *, focus: bool = True,
                           clip: bool = True, context: bool = True,
                           granularity: str = "atom"):
        """A viewport atom click, unified with the selection box: select the clicked
        atom — or its whole residue at ``granularity="residue"`` — and give it
        exactly the treatment a typed selection gets — same oriented framing,
        same clip sphere, same neighborhood context. One grammar for "show me
        this", however it was indicated. Returns the equivalent selection
        expression (for the selection box), or ``None`` when the index names no
        atom."""
        self.set_active_model(mid)
        entry = self._model_entry(mid)
        session = entry["session"] if entry else None
        if session is None or getattr(session, "model", None) is None:
            return None
        atoms = session.model.get_hierarchy().atoms()
        if not (0 <= int(atom_index) < atoms.size()):
            return None
        if granularity == "atom":
            indices = [int(atom_index)]
            from . import edits as edits_mod
            expression = edits_mod.selection_for_atom(session.model, int(atom_index))
        else:
            residue_group = atoms[int(atom_index)].parent().parent()
            indices = [a.i_seq for a in residue_group.atoms()]
            chain_id = residue_group.parent().id.strip()
            expression = f"chain {chain_id} and resid {residue_group.resid().strip()}"
        self._select_fragment(mid, session, entry, indices,
                              focus=focus, clip=clip, context=context)
        return expression


    def toggle_picked_residue(self, mid: str, atom_index: int, *, clip: bool = True,
                              context: bool = True, granularity: str = "atom"):
        """Shift-click: grow or shrink the selection by the clicked atom — or by
        its whole residue at ``granularity="residue"``.

        The unit joins the selection, or leaves it when it is already entirely
        selected. The camera deliberately stays where it is — re-framing on every
        addition would fight the accumulating gesture — but the isolation clip
        and the neighborhood context re-fit the grown selection, so an addition is
        never clipped out of view. Emptying the selection this way clears everything,
        exactly like an empty expression. Returns the equivalent expression for the
        selection box ('' when emptied or no longer expressible as whole residues),
        or ``None`` when the index names no atom.
        """
        self.set_active_model(mid)
        entry = self._model_entry(mid)
        session = entry["session"] if entry else None
        if session is None or getattr(session, "model", None) is None:
            return None
        atoms = session.model.get_hierarchy().atoms()
        if not (0 <= int(atom_index) < atoms.size()):
            return None
        if granularity == "atom":
            residue = {int(atom_index)}
        else:
            residue = {a.i_seq for a in atoms[int(atom_index)].parent().parent().atoms()}
        with self._scene_lock:
            current_ordered = list(self._scene_selection.get(mid, []))
            current = set(current_ordered)
        removing = residue <= current
        grown = (current - residue) if removing else (current | residue)
        if not grown:
            session.clear_selection()
            self._lift_selection_sphere(entry)
            self._set_context_rep(entry, None)
            with self._scene_lock:
                self._scene_selection.pop(mid, None)
            self._emit_scene_selection(changed=mid)
            return ""
        indices = sorted(grown)
        sel = session.highlight(indices)
        # Click order is semantic (angle/dihedral measures and edits read it):
        # kept atoms stay in their places, a newly toggled-in unit lands last.
        sel.order = ([i for i in current_ordered if i not in residue] if removing
                     else current_ordered + sorted(residue - current))
        if clip and self._selection_clip_armed(entry):
            # The same sphere the focus path fits, re-fit to the grown selection —
            # without it a residue added outside the standing sphere is invisible.
            bounds = self._component_bounds(atoms, indices)
            xyz = np.array([atoms[i].xyz for i in bounds])
            center = xyz.mean(axis=0)
            reach = float(np.linalg.norm(xyz - center, axis=1).max())
            self._apply_model_sphere(entry, "selection",
                                     radius=reach + self._clip_padding(),
                                     center=center)
            entry["_auto_clip_depth"] = None   # the camera never oriented here
        else:
            # A clip-suppressed selection lifts the sphere a clipped one left.
            self._lift_selection_sphere(entry)
        self._set_context_rep(entry, indices if context else None)
        self._on_model_selection(mid, sel)
        return self._selection_expression(entry, indices)


    def _selection_expression(self, entry, indices):
        """The selection as a whole-residue expression ('(chain A and resid 17) or …'),
        or '' when some residue is only partially selected — a box expression that
        does not reproduce the selection would lie. The one exception: a lone atom
        is exactly expressible, so it still echoes."""
        owner, members = self._residue_map(entry)
        chosen = set(indices)
        atoms = entry["session"].model.get_hierarchy().atoms()
        if len(chosen) == 1:
            from . import edits as edits_mod
            return edits_mod.selection_for_atom(entry["session"].model, next(iter(chosen)))
        terms = []
        for group in sorted({int(owner[i]) for i in chosen}):
            group_atoms = members[group]
            if not set(group_atoms) <= chosen:
                return ""
            residue_group = atoms[group_atoms[0]].parent().parent()
            chain_id = residue_group.parent().id.strip()
            terms.append(f"(chain {chain_id} and resid {residue_group.resid().strip()})")
        return " or ".join(terms)


    def _select_fragment(self, mid, session, entry, atoms_or_sel, *,
                         focus: bool, clip: bool = True, context: bool) -> int:
        """The one path every fragment selection takes — typed or clicked."""
        sel = session.highlight(atoms_or_sel)     # show it in the viewer
        if focus and len(sel):
            # Aim the camera at what was just named. On by default: a typed selection
            # is a statement of intent ("show me resseq 29"), unlike a viewport click,
            # which happens where the camera already is and is never auto-focused.
            #
            # A selection that is exactly one amino acid gets the oriented framing the
            # space-bar navigation already uses -- N to the left, C to the right, side
            # chain up -- instead of Mol*'s default (which keeps whatever direction the
            # camera happened to have). A residue always reads the same way on screen,
            # which is the point of orienting at all. Anything else (many residues, a
            # partial residue without its backbone, a ligand) falls back to the plain
            # center-and-frame focus; heuristics for those are a separate question.
            indices = list(sel)
            orientation = None
            if self._is_single_residue(session.model, indices):
                orientation = self._residue_orientation(session.model, indices)
            if orientation is None:
                # Everything else -- many residues, a ligand, a backbone-less fragment
                # -- frames by its principal axes; only shapes with no frame at all
                # (a lone atom, a straight line) fall back to the plain focus.
                orientation = self._principal_axes_orientation(session.model, indices)
            # Whether this selection may isolate: the object's Clip sphere mode is
            # the policy ("On selection"), the caller's flag a per-call override.
            clip = clip and self._selection_clip_armed(entry)
            if orientation is not None:
                session.orient_camera(*orientation, clip=clip)
                center = np.asarray(orientation[0], dtype=float)
            else:
                session.focus(indices)
                atoms = session.model.get_hierarchy().atoms()
                center = np.mean([atoms[i].xyz for i in indices], axis=0)
            if clip:
                # Isolate the neighborhood: a clip sphere around what was selected, so
                # the rest of the structure does not bury it. The same shader-side
                # mechanism as the pane's clipping controls -- it cannot be disturbed by
                # the viewer's own camera management, which is not true of near/far-plane
                # clipping. Centered explicitly on the selection: the clip lands while the
                # camera is still flying there, so the camera-target default would put
                # the sphere on the *old* view and clip out the very thing being framed
                # (worst with several far-apart models loaded -- the whole viewport went
                # blank until a manual clip re-centered it). Sized to the touched
                # components plus enough context to read their surroundings -- a lone
                # atom gets its whole residue, not a ball around itself; lifted when
                # the selection is cleared, and any later manual clipping simply
                # takes over.
                atoms = session.model.get_hierarchy().atoms()
                bounds = self._component_bounds(atoms, indices)
                xyz = np.array([atoms[i].xyz for i in bounds], dtype=float)
                center = xyz.mean(axis=0)
                reach = float(np.linalg.norm(xyz - center, axis=1).max())
                if entry is not None:
                    self._apply_model_sphere(
                        entry, "selection", radius=reach + self._clip_padding(),
                        center=center)
                    # The camera slab orient() parked is part of the clipped view:
                    # remember the depth so lifting and re-applying restore it too.
                    entry["_auto_clip_depth"] = (
                        orientation[4][2] if orientation is not None else None)
            elif entry is not None:
                # Clip is off for this selection, so lift the sphere a previous
                # clipped selection left -- otherwise it keeps cutting the new view.
                self._lift_selection_sphere(entry)
        # The neighborhood context rides the selection, not the camera: it shows (or
        # clears) whether or not the focus checkbox moved the view.
        self._set_context_rep(entry, list(sel) if context else None)
        self._on_model_selection(mid, sel)        # feed the scene selection (table + label)
        return len(sel)


    # A fragment's local context: the selection plus every residue within this many
    # Angstrom drawn ball-and-stick while the selection stands. The size cap keeps the
    # treatment for fragments being *inspected* — a whole chain or model is a region,
    # and dressing thousands of atoms in sticks helps nothing and costs plenty.
    _CONTEXT_RADIUS = 5.0

    _CONTEXT_MAX_ATOMS = 250


    def _set_context_rep(self, entry, indices) -> None:
        """Draw (or clear, with ``indices=None``) the selection's neighborhood context."""
        if entry is None:
            return
        want = None
        if (indices and len(indices) <= self._CONTEXT_MAX_ATOMS
                and entry.get("rep") != "ball-and-stick"):
            want = self._neighborhood_indices(entry, indices)
        if want == entry.get("context_on"):
            return  # unchanged (both None, or the same neighborhood) — no redraw
        entry["context_on"] = want
        self._apply_model_rep(entry)


    def _neighborhood_indices(self, entry, indices) -> list:
        """Atom indices of every residue with an atom within ``_CONTEXT_RADIUS`` of the
        selection — the selection's own residues included."""
        coords = np.array(entry["session"].model.get_hierarchy().atoms().extract_xyz())
        mask = np.zeros(len(coords), dtype=bool)
        r2 = self._CONTEXT_RADIUS ** 2
        # One pass per selected atom keeps peak memory flat on large models.
        for i in indices:
            mask |= ((coords - coords[i]) ** 2).sum(axis=1) <= r2
        owner, members = self._residue_map(entry)
        out: set = set()
        for group in {int(owner[j]) for j in np.nonzero(mask)[0]}:
            out.update(members[group])
        return sorted(out)


    def _residue_map(self, entry):
        """Cached (atom -> residue-group index, residue-group -> atom indices)."""
        cached = entry.get("residue_map")
        if cached is None:
            hierarchy = entry["session"].model.get_hierarchy()
            owner = np.zeros(hierarchy.atoms_size(), dtype=int)
            members: list = []
            for model_ in hierarchy.models():
                for chain in model_.chains():
                    for residue_group in chain.residue_groups():
                        idxs = [a.i_seq for a in residue_group.atoms()]
                        owner[idxs] = len(members)
                        members.append(idxs)
            cached = entry["residue_map"] = (owner, members)
        return cached


def run_core(host: str = "127.0.0.1", port: int = 5173,
             mcp_port: "int | None" = None) -> int:
    """Run the GUI-agnostic core: webapp + MCP + a headless browser if present.

    The headless story for ``python -m pxviewer`` — the same AppCore the
    desktop embeds, hosting the Mol* page in Chromium when playwright is
    installed (screenshots work either way over the wire; without a browser
    host everything but pixels runs).
    """
    from .browser import try_browser_viewport

    app = AppCore(host=host, port=port, viewport=try_browser_viewport(),
                  mcp_port=mcp_port)
    try:
        return app.start()
    finally:
        app.stop()
