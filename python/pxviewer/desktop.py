"""Self-contained PyQt desktop viewer for pxviewer.

The desktop app opens two side-by-side windows:

1. **Viewport** — a `QWebEngineView` that loads the Mol* viewer.
2. **Controls** — a native Qt window whose main screen opens a file from the
   user's filesystem, with the demos tucked behind a second tab.

A `LiveSession` runs in the background so the controls can toggle mouse selection
and receive click-built selections, and so the model demos can stream coordinates
into the viewport. The whole thing is served by the local `Webapp` server, so no
external browser is needed.
"""

from __future__ import annotations

import signal
import json
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
from .palettes import suggested_colors
from .webapp import Webapp

# The swatches offered when coloring an object by hand — hex, drawn from the bundled
# palettes (see palettes.suggested_colors) so a hand-picked color comes from the same
# inventory as the ones objects open in.
_VOLUME_COLORS = suggested_colors()
# Sentinel for the "Custom…" entry in a color dropdown (never a real color value).
_CUSTOM_COLOR = "\x00custom"
# Color a model by its per-atom fit to the map. Not a Mol* theme — the values are computed
# by cctbx and pushed as an attribute (see qscore.py and color_model_by_qscore).
_QSCORE_COLOR = "qscore"
# Color a model by aggregated validation severity (see hotspots.py and the Hotspots tab).
_HOTSPOT_COLOR = "hotspot"
# Colors that are computed per-atom arrays rather than Mol* theme names. They travel on
# entry["attribute"] and are applied by _apply_model_rep's attribute branch.
_ATTRIBUTE_COLORS = frozenset({_QSCORE_COLOR, _HOTSPOT_COLOR})
# Default opacity knee for the severity cloud, in severity units: the outlier cut.
_HOTSPOT_KNEE_DEFAULT = 1.0

# Contour level, in sigma. Mol* does the sigma scaling, so a level means the same thing
# for any map and one fixed slider range serves all of them. The slider covers the range
# people actually work in; the spinbox goes past it, since cryo-EM maps are often
# contoured well above 10 sigma.
_ISO_SLIDER_MAX = 10.0
_ISO_SPIN_MAX = 100.0
_ISO_RESOLUTION = 0.01  # QSlider is integer-only, so the level is stored in steps of this

# Default radius for masking density around a model (A). 3 A is roughly one atom's
# reach, which is what "the density belonging to this model" usually means.
_MASK_RADIUS_DEFAULT = 3.0

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

# How long the post-release wind-down plays for. The minimization itself converges in a
# fraction of a second; this stretches its states over a watchable settle so a released
# fling comes visibly to rest — the clearest signal that the fragment is done, not broken.
_TUG_SETTLE_DURATION = 1.2

# How much density to draw around the view center, for maps that need it. A map made
# from reflections fills the unit cell, so drawing all of it buries the model — those
# open with a radius. A map read from a file is already a box around its subject, so it
# does not. (Coot applies its radius to every map; ours can tell the two apart.)
#
# 15 A is a starting point rather than a considered convention — Coot's own default is
# not something we confirmed — so it is the app's default and adjustable in Settings,
# not a constant. Per-map, the Appearance pane has always been able to change it.
_VIEW_RADIUS_DEFAULT = 15.0

# Markers dropped in the viewport ("place a marker" → click). Drawn as spheres on a
# markup channel well clear of the probe (0, 1) and validation (10+) channels.
_MARKER_CHANNEL = 90
_MARKER_RADIUS = 0.5
_MARKER_COLOR = [255, 90, 90]

# The object list sizes itself to its contents between these. The floor keeps the empty
# state from collapsing to nothing; past the ceiling the list scrolls itself rather than
# taking the whole pane. Roomier now that the mouse reference moved to a popup (its old
# space is the list's), so more objects show at once before it has to scroll.
_TREE_MIN_HEIGHT = 150
_TREE_MAX_HEIGHT = 440

# "Nothing cached yet" for the Appearance pane's rebuild guard — distinct from any real
# signature, including the ``(None, None, None)`` of the empty state.
_UNSET = object()

# Inline representation dropdowns in the Loaded tree (models vs maps differ).
# The model values must be types the LiveSession API accepts (see live.py's
# _STRUCTURE_REPR_TYPES / _REPR_ALIASES) — test_model_rep_options_are_valid guards this.
_MODEL_REP_OPTIONS = [
    ("Cartoon", "cartoon"),
    ("Ball & stick", "ball-and-stick"),
    ("Spacefill", "spacefill"),
    ("Surface", "surface"),
]
# Prefix for a grouped object's name in the Objects tree — see _on_loaded_changed. Qt only
# indents column 0, so the name column needs its own indent to show depth.
_GROUP_MEMBER_INDENT = "     "

_VOLUME_STYLE_OPTIONS = [
    ("Surface", "surface"),
    ("Mesh", "mesh"),  # chickenwire — the crystallographer's map "mesh" (edges only)
]
_MODEL_COLOR_OPTIONS = [
    ("Default", None),
    ("By element", "element-symbol"),
    ("By chain", "chain-id"),
    ("By secondary structure", "secondary-structure"),
    ("By residue", "residue-name"),
    ("By hydrophobicity", "hydrophobicity"),
    # The refined per-atom numbers. Both come straight from the topology, which already
    # carries B_iso_or_equiv and occupancy (see data._atom_site_category), so Mol* colors
    # from them with nothing extra sent. Mol* calls the B-factor theme 'uncertainty'
    # because it also serves pLDDT; to a crystallographer it is the B-factor.
    ("By B-factor", "uncertainty"),
    ("By occupancy", "occupancy"),
    # Not a Mol* theme: computed against the map by cctbx and sent as per-atom values.
    # See DesktopApp.color_model_by_qscore.
    ("By Q-score (fit to map)", _QSCORE_COLOR),
    # Likewise computed, but only re-applies a field the Hotspots tab already produced.
    ("By hotspot severity", _HOTSPOT_COLOR),
]


def _model_rep_color(rep: str) -> str:
    """A sensible default color theme for a representation type (no palette in play)."""
    return "secondary-structure" if rep == "cartoon" else "element-symbol"


# Representations that draw individual atoms: a palette color tints their *carbons* (O/N/S
# keep their standard hues). Ribbons/surfaces have no atoms to tint, so a palette color is
# applied uniformly instead. See _apply_model_rep.
_ATOM_REPS = frozenset({"ball-and-stick", "ball_and_stick", "spacefill", "sphere"})


def _rep_shows_atoms(rep: str) -> bool:
    return rep in _ATOM_REPS


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
_STRUCTURE_TYPE_ORDER = ["Protein", "Nucleic acid", "Sugar", "Ion", "Water", "Ligand / other"]


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


def _check_qt() -> None:
    try:
        import PySide6  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The desktop viewer needs PySide6. Install it with: "
            "pip install 'pxviewer[desktop]'"
        ) from exc


_ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.png"
_ICONS_DIR = Path(__file__).resolve().parent / "assets" / "icons"  # Lucide SVGs (ISC)
_CUSTOM_ICONS_DIR = Path(__file__).resolve().parent / "assets" / "icons_custom"  # our own
_SPLASH_SIDE = 320  # logical px; scaled from the 512px icon for the screen's pixel ratio
_SPLASH_MAX_MS = 15000  # never leave the splash up if the page never reports a load


def _app_icon():
    """The pxviewer window/dock icon as a QIcon, or None if the asset is missing."""
    from PySide6.QtGui import QIcon

    return QIcon(str(_ICON_PATH)) if _ICON_PATH.exists() else None


def _edit_summaries(scope) -> list:
    """``[{"kind", "summary"}]`` for a model's edits scope, for the tree and edits list.

    A read-only projection of cctbx's own scope. It is deliberately not the thing that
    gets applied -- the scope is (see edits.build_restraints) -- so a field this summary
    has no room for cannot go missing from the restraints.
    """
    from . import edits

    if scope is None:
        return []
    return [{"kind": kind, "summary": edits.summarize(kind, obj)}
            for kind, obj in edits.entries(scope)]


_HIGHLIGHT_OVERLAY_CLASS = None


def _highlight_overlay_class():
    """A transparent, click-through widget that paints a rounded ring at a set alpha — used
    to emphasise a button *on top*, so it never touches the button's size or the layout
    around it. Defined lazily (needs Qt) and cached."""
    global _HIGHLIGHT_OVERLAY_CLASS
    if _HIGHLIGHT_OVERLAY_CLASS is None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPainter, QPen
        from PySide6.QtWidgets import QWidget

        class _HighlightOverlay(QWidget):
            def __init__(self, parent):
                super().__init__(parent)
                self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
                self._alpha = 0.0

            def set_alpha(self, a: float) -> None:
                self._alpha = max(0.0, min(1.0, a))
                self.update()

            def paintEvent(self, _event) -> None:
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                color = QColor(255, 152, 0)  # amber, matching the app's accents
                color.setAlphaF(self._alpha)
                painter.setPen(QPen(color, 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 7, 7)

        _HIGHLIGHT_OVERLAY_CLASS = _HighlightOverlay
    return _HIGHLIGHT_OVERLAY_CLASS


def _line_icon(name: str, color, size: int = 20):
    """A monochrome line SVG (``<name>.svg``) tinted to ``color`` as a QIcon.

    Looked up in ``assets/icons_custom`` (our own icons) first, then ``assets/icons`` (the
    Lucide set), so a custom icon can also override a stock name. The icons draw with
    ``stroke="currentColor"``, which Qt's SVG renderer does not resolve on its own, so the
    color is substituted in before rendering. Rendered at 3x the display size so it stays
    crisp on a HiDPI screen. Returns None if the asset is missing."""
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer

    path = _CUSTOM_ICONS_DIR / f"{name}.svg"
    if not path.exists():
        path = _ICONS_DIR / f"{name}.svg"
    if not path.exists():  # pragma: no cover - packaging guard
        return None
    svg = path.read_text().replace("currentColor", QColor(color).name())
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(size * 3, size * 3)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(3.0)
    return QIcon(pm)


# Semantic accent colors, in (light-theme, dark-theme) shades so each reads on its own
# background — dark greens/ambers on a light UI, brighter ones on a dark UI. Everything else
# follows the palette directly (QSS ``palette(...)`` refs); only these carry meaning the
# palette has no role for, so they are chosen by hand and resolved against the live palette.
_ACCENTS = {
    "go": ("#1a7f37", "#2ea043"),     # green — a run is ready / underway (Minimize)
    "stop": ("#b26a00", "#cc8400"),   # amber — stop a run (Pause)
    "warn": ("#b26a00", "#e3a008"),   # amber text — a message to catch the eye
    "error": ("#c0392b", "#f85149"),  # red — an invalid action / bad input
}


def _first_line(exc: Exception) -> str:
    """The first line of an exception's message, for a one-line status report.

    FetchError deliberately carries a multi-line explanation (url, what to try); the
    status bar has room for the first line of it, and the rest is on stderr.
    """
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0].strip()


def _accent(widget, name: str) -> str:
    """The shade of a semantic accent (see :data:`_ACCENTS`) for ``widget``'s current theme."""
    from PySide6.QtGui import QPalette

    light, dark = _ACCENTS[name]
    is_dark = widget.palette().color(QPalette.ColorRole.Window).lightness() < 128
    return dark if is_dark else light


# macOS renders icon-only QPushButtons as tall, near-borderless native bezels, so the glyphs
# read as oversized and unframed (Linux/Adwaita frames them tightly, which looks right). Give
# them a compact, theme-adaptive frame *only on macOS* — palette() colors so it still tracks
# light/dark — and leave the good native look on other platforms untouched.
_IS_MAC = sys.platform == "darwin"
_ICON_BUTTON_QSS = (
    "QPushButton { border: 1px solid palette(mid); border-radius: 5px; padding: 3px;"
    " background-color: palette(button); }"
    "QPushButton:hover { border-color: palette(highlight); }"
    "QPushButton:pressed { background-color: palette(midlight); }"
    "QPushButton:checked { background-color: palette(highlight); }"
)

# macOS's native tab metrics ignore the icon-only size we want; only a stylesheet
# overrides them. A stylesheet also makes Qt paint the bar itself — so paint the bar and
# tabs the panel color (palette(window)) to cover the native grey base, rather than
# leaving it transparent (which shows that grey through). With the metrics overridden,
# setExpanding(True) distributes the tabs evenly across the bar (measured: 7 tabs share
# a 400px bar at 57px each). Applied on macOS only, so the native Linux tabs are untouched.
_TAB_BAR_QSS = (
    "QTabBar { background: palette(window); }"
    "QTabBar::tab { background: palette(window); border: 0; margin: 0;"
    " border-bottom: 2px solid palette(window); padding: 6px 5px; }"
    "QTabBar::tab:selected { border-bottom: 2px solid palette(highlight); }"
    "QTabBar::tab:hover { background: palette(midlight); }"
)


def _icon_button_base_qss() -> str:
    """The base stylesheet for an icon-only button — a macOS frame, or nothing elsewhere."""
    return _ICON_BUTTON_QSS if _IS_MAC else ""


def _tab_hover_filter(tabbar, on_hover):
    """A QObject event filter, parented to ``tabbar``, that calls ``on_hover(index)`` with
    the tab under the pointer (or -1 on leave). Defined lazily so the module imports without
    a running QApplication."""
    from PySide6.QtCore import QEvent, QObject

    class _TabHoverFilter(QObject):
        def eventFilter(self, obj, event):
            etype = event.type()
            if etype == QEvent.Type.MouseMove:
                on_hover(tabbar.tabAt(event.position().toPoint()))
            elif etype in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                on_hover(-1)
            return False

    return _TabHoverFilter(tabbar)


def _palette_watch_filter(widget, on_change):
    """A QObject filter, parented to ``widget``, that calls ``on_change()`` when the palette
    changes — a light/dark switch, or the real theme landing once the window is shown.

    Defer the callback by one event-loop turn. On macOS ``ApplicationPaletteChange`` can be
    delivered while QApplication still exposes parts of the old palette; repainting in the
    event itself creates the half-light/half-dark state this watcher exists to prevent."""
    from PySide6.QtCore import QEvent, QObject, QTimer

    class _PaletteFilter(QObject):
        pending = False

        def refresh(self):
            self.pending = False
            on_change()

        def eventFilter(self, obj, event):
            if event.type() in (QEvent.Type.ApplicationPaletteChange, QEvent.Type.PaletteChange):
                if not self.pending:
                    self.pending = True
                    QTimer.singleShot(0, self.refresh)
            return False

    watcher = _PaletteFilter(widget)
    widget.installEventFilter(watcher)
    return watcher


def install_desktop_entry() -> str:
    """Write a Linux ``.desktop`` file so the launcher shows pxviewer's name and icon.

    The launcher/taskbar (Wayland and X11) finds an app's icon by matching its running
    window to a ``.desktop`` file — ``setWindowIcon`` only covers the title bar. This
    drops one in the user's applications directory, with ``Exec`` pointing at this
    interpreter, ``Icon`` at the bundled icon, and ``StartupWMClass`` matching the app id
    the app sets at startup, so the running window is associated with it. Returns the
    path written. Run once: ``pxviewer install-desktop-entry``.
    """
    apps = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    path = apps / "pxviewer.desktop"
    icon = str(_ICON_PATH) if _ICON_PATH.exists() else "pxviewer"
    path.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=pxviewer\n"
        "GenericName=Molecular viewer\n"
        f"Exec={sys.executable} -m pxviewer desktop\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "Categories=Science;Education;Graphics;\n"
        "StartupWMClass=pxviewer\n"
    )
    return str(path)


def _show_splash():
    """Put the icon on screen before the slow part of starting up.

    Qt's web engine and the Mol* bundle take a few seconds to come up, during which
    nothing is visible and the launch reads as having failed. This goes up as soon as
    there is a QApplication to draw it with — everything expensive happens after.

    Drawn from the full-resolution icon and marked with the screen's pixel ratio, so it
    is crisp rather than an upscaled dock icon.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication, QSplashScreen

    if not _ICON_PATH.exists():
        return None
    pixmap = QPixmap(str(_ICON_PATH))
    if pixmap.isNull():
        return None
    ratio = QApplication.primaryScreen().devicePixelRatio() if QApplication.primaryScreen() else 1.0
    side = int(_SPLASH_SIDE * ratio)
    pixmap = pixmap.scaled(
        side, side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    pixmap.setDevicePixelRatio(ratio)
    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    splash.show()
    # Several round-trips, not one: Wayland maps and paints a surface asynchronously, so a
    # single processEvents can return before the splash is actually on screen — after
    # which the caller blocks on the slow startup and it never appears.
    for _ in range(3):
        QApplication.processEvents()
    return splash


def _make_bridge():
    """A QObject that marshals background-thread events onto the Qt GUI thread.

    Selections and demo status arrive on the WebSocket/demo threads; touching
    widgets from there is not allowed, so everything crosses over as a signal.
    """
    from PySide6.QtCore import QObject, Signal

    class _Bridge(QObject):
        scene_selection_changed = Signal(object)  # {model_id: [atom indices]} across all models
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
        localres_shown = Signal()           # the viewport drew a localres colouring (it is usable now)

    return _Bridge()


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


def _make_range_slider():
    """A slider with two handles, for a front/rear clipping slab (built post-Qt).

    Qt has no two-handle slider. This is the minimum that behaves like one: drag either
    handle, drag the bar between them to move both, and the handles may meet — which is
    not a degenerate case here but the point at which the object is fully clipped.
    """
    from PySide6.QtCore import QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import QPainter, QPalette
    from PySide6.QtWidgets import QSizePolicy, QWidget

    class RangeSlider(QWidget):
        """Two handles on one track. Values are floats in 0..1, front <= back."""

        changed = Signal(float, float)

        _HANDLE = 9.0   # radius, px
        _TRACK = 5.0    # thickness, px

        def __init__(self, parent=None):
            super().__init__(parent)
            self._front = 0.0
            self._back = 1.0
            self._drag = None      # 'front' | 'back' | 'both'
            self._grab_at = 0.0
            self.setMinimumHeight(24)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        def values(self):
            return self._front, self._back

        def set_values(self, front, back, *, notify=False):
            front = min(max(float(front), 0.0), 1.0)
            back = min(max(float(back), 0.0), 1.0)
            if front > back:
                front = back
            if (front, back) == (self._front, self._back):
                return
            self._front, self._back = front, back
            self.update()
            if notify:
                self.changed.emit(self._front, self._back)

        # -- geometry --

        def _span(self):
            return self.width() - 2 * self._HANDLE

        def _x(self, value):
            return self._HANDLE + value * self._span()

        def _value_at(self, x):
            span = self._span()
            return 0.0 if span <= 0 else min(max((x - self._HANDLE) / span, 0.0), 1.0)

        # -- painting --

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            mid = self.height() / 2
            track = QRectF(self._HANDLE, mid - self._TRACK / 2, self._span(), self._TRACK)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.palette().mid())
            painter.drawRoundedRect(track, self._TRACK / 2, self._TRACK / 2)
            # The kept span is what is *shown*, so fill between the handles.
            kept = QRectF(self._x(self._front), track.top(),
                          max(self._x(self._back) - self._x(self._front), 1.0), self._TRACK)
            painter.setBrush(self.palette().highlight())
            painter.drawRoundedRect(kept, self._TRACK / 2, self._TRACK / 2)
            painter.setBrush(self.palette().light())
            painter.setPen(self.palette().color(QPalette.ColorRole.Mid))
            for value in (self._front, self._back):
                painter.drawEllipse(QPointF(self._x(value), mid), self._HANDLE, self._HANDLE)

        # -- interaction --

        def mousePressEvent(self, event):
            x = event.position().x()
            df = abs(x - self._x(self._front))
            db = abs(x - self._x(self._back))
            if min(df, db) <= self._HANDLE + 2:
                # Pick the nearer handle; when they coincide, direction decides, so the
                # slab can always be reopened after being closed.
                if df < db or (df == db and x < self._x(self._front)):
                    self._drag = "front"
                else:
                    self._drag = "back"
            elif self._x(self._front) < x < self._x(self._back):
                self._drag = "both"
                self._grab_at = self._value_at(x)
            else:
                # Clicked off the ends: bring the nearer handle here.
                self._drag = "front" if x < self._x(self._front) else "back"
                self._move_to(self._value_at(x))

        def mouseMoveEvent(self, event):
            if self._drag is None:
                return
            self._move_to(self._value_at(event.position().x()))

        def mouseReleaseEvent(self, _event):
            self._drag = None

        def _move_to(self, value):
            if self._drag == "front":
                self.set_values(min(value, self._back), self._back, notify=True)
            elif self._drag == "back":
                self.set_values(self._front, max(value, self._front), notify=True)
            elif self._drag == "both":
                width = self._back - self._front
                shift = value - self._grab_at
                front = min(max(self._front + shift, 0.0), 1.0 - width)
                self._grab_at = value
                self.set_values(front, front + width, notify=True)

    return RangeSlider


def _runs(indices):
    """Yield contiguous ``(start, end)`` runs over sorted, de-duplicated indices."""
    it = iter(sorted({int(i) for i in indices}))
    try:
        start = prev = next(it)
    except StopIteration:
        return
    for i in it:
        if i == prev + 1:
            prev = i
        else:
            yield (start, prev)
            start = prev = i
    yield (start, prev)


def _make_atom_table_model():
    """A QAbstractTableModel over a session's per-atom columns (built lazily post-Qt).

    Rows are atoms (i_seq order), columns are the structure's per-atom attributes.
    Only the numpy columns are held; values are formatted on demand for the cells the
    view actually paints, so 100k+ atoms stay cheap (QTableView virtualises rendering).
    """
    from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

    class AtomTableModel(QAbstractTableModel):
        def __init__(self):
            super().__init__()
            self._headers: List[str] = []
            self._cols: list = []  # (values_or_None, kind)  kind: idx|str|int|float
            self._n = 0
            self._filter: Optional[list] = None  # None = all rows; else visible atom indices

        def set_session(self, session) -> None:
            self.beginResetModel()
            self._filter = None
            self._headers, self._cols, self._n = [], [], 0
            data = getattr(session, "_data", None)
            arrays = getattr(data, "arrays", None)
            if arrays is not None and len(arrays) > 0:
                self._n = len(arrays)

                def add(header, values, kind):
                    self._headers.append(header)
                    self._cols.append((values, kind))

                add("#", None, "idx")
                add("element", arrays.element, "str")
                add("name", arrays.name, "str")
                add("resname", arrays.resname, "str")
                add("chain", arrays.chain, "str")
                add("resseq", arrays.resseq, "int")
                if arrays.altloc is not None and any(arrays.altloc):
                    add("altloc", arrays.altloc, "str")
                add("x", arrays.x, "float")
                add("y", arrays.y, "float")
                add("z", arrays.z, "float")
                if arrays.b is not None:
                    add("B", arrays.b, "float")
                if arrays.occ is not None:
                    add("occ", arrays.occ, "float")
                for name, values in getattr(session, "_attributes", {}).items():
                    add(name, values, "float")
            self.endResetModel()

        def set_filter(self, indices) -> None:
            """Restrict the visible rows to ``indices`` (atom order preserved); None = all.

            Backs the "show only selected" mode. Only the small selected subset is
            materialised, so the view stays cheap even against 100k+ atoms.
            """
            self.beginResetModel()
            if indices is None:
                self._filter = None
            else:
                self._filter = [i for i in sorted({int(i) for i in indices}) if 0 <= i < self._n]
            self.endResetModel()

        def is_filtered(self) -> bool:
            return self._filter is not None

        def row_atom(self, row: int) -> int:
            """The underlying atom index for a view row (identity unless filtered)."""
            return row if self._filter is None else self._filter[row]

        def atom_row(self, atom: int) -> int:
            """The view row showing a given atom index, or -1 if not visible."""
            if self._filter is None:
                return atom if 0 <= atom < self._n else -1
            try:
                return self._filter.index(atom)
            except ValueError:
                return -1

        def rowCount(self, parent=QModelIndex()):
            if parent.isValid():
                return 0
            return self._n if self._filter is None else len(self._filter)

        def columnCount(self, parent=QModelIndex()):
            return 0 if parent.isValid() else len(self._headers)

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid():
                return None
            values, kind = self._cols[index.column()]
            atom = self.row_atom(index.row())
            if role == Qt.ItemDataRole.DisplayRole:
                if kind == "idx":
                    return str(atom)
                v = values[atom]
                if kind == "float":
                    fv = float(v)
                    return "" if fv != fv else f"{fv:.3f}"  # fv != fv -> NaN
                if kind == "int":
                    return str(int(v))
                return str(v)
            if role == Qt.ItemDataRole.TextAlignmentRole and kind in ("idx", "int", "float"):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return None

        def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
            if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
                return self._headers[section]
            return None

    return AtomTableModel()


def _atom_label_fn(session):
    """A ``i_seq -> "chain/resnameresseq/name"`` labeller from a session's columns."""
    arrays = getattr(getattr(session, "_data", None), "arrays", None)
    if arrays is None:
        return str
    chain, resname, resseq, name = arrays.chain, arrays.resname, arrays.resseq, arrays.name

    def label(i: int) -> str:
        return f"{chain[i]}/{resname[i]}{int(resseq[i])}/{name[i]}"

    return label


def _geostd_source_fn(session):
    """An ``i_seqs -> (text, geostd_path_or_None)`` labeller for a restraint's source.

    Intra-residue restraints come from that monomer's geostd file; a restraint whose
    atoms span residues is defined by a link, not a single monomer file.
    """
    from .geometry import monomer_cif_path

    arrays = getattr(getattr(session, "_data", None), "arrays", None)
    if arrays is None:
        return lambda iseqs: ("", None)
    resname = arrays.resname
    cache: dict = {}

    def source(iseqs):
        names = {resname[i] for i in iseqs}
        if len(names) != 1:
            return ("(link)", None)  # spans residues -> a link, not one monomer file
        rn = next(iter(names))
        if rn not in cache:
            cache[rn] = (rn, monomer_cif_path(rn))
        return cache[rn]

    return source


def _reveal_in_file_manager(path) -> None:
    """Reveal a file in the OS file browser (Finder / Explorer / folder on Linux)."""
    import subprocess

    path = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        elif os.name == "nt":  # noqa: SIM  (explorer wants the odd "/select," token)
            subprocess.Popen(["explorer", "/select,", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except Exception:  # pragma: no cover - platform/tooling dependent
        pass


def _make_restraint_table_model():
    """A QAbstractTableModel over a GeometryRestraints category (built lazily post-Qt).

    Rows are restraint proxies; the first column lists the atoms involved and the
    rest are the restraint's values (ideal/model/delta/…). Values are computed from
    cctbx on demand for the row the view paints — a small one-row memo keeps a row's
    cells from recomputing — so 100k+ restraints stay cheap (QTableView virtualises).
    """
    from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
    from PySide6.QtGui import QColor, QFont

    class RestraintTableModel(QAbstractTableModel):
        def __init__(self):
            super().__init__()
            self._geo = None
            self._category = ""
            self._columns: List[str] = []
            self._label = None
            self._source = None  # i_seqs -> (text, path_or_None); adds a "geostd" link column
            self._n = 0
            self._filter: Optional[list] = None  # None = all rows; else restraint indices
            self._memo_key = -1
            self._memo = None  # (i_seqs, values) for _memo_key (a restraint index)

        def set_source(self, geo, category, columns, label_fn, source_fn=None) -> None:
            self.beginResetModel()
            self._geo, self._category = geo, category
            self._columns = list(columns)
            self._label = label_fn
            self._source = source_fn
            self._n = geo.count(category) if geo is not None else 0
            self._filter = None
            self._memo_key, self._memo = -1, None
            self.endResetModel()

        def source_column(self) -> int:
            """Column index of the geostd link (or -1 when there is none)."""
            return 1 + len(self._columns) if self._source is not None else -1

        def source_for_row(self, row: int):
            """``(text, path_or_None)`` for the geostd file backing a row."""
            if self._source is None:
                return ("", None)
            return self._source(self._rowdata(row)[0])

        def set_filter(self, indices) -> None:
            """Restrict visible rows to ``indices`` (restraint order); None = all."""
            self.beginResetModel()
            self._filter = None if indices is None else list(indices)
            self._memo_key, self._memo = -1, None
            self.endResetModel()

        def is_filtered(self) -> bool:
            return self._filter is not None

        def _restraint_index(self, row: int) -> int:
            return row if self._filter is None else self._filter[row]

        def _rowdata(self, row: int):
            key = self._restraint_index(row)
            if key != self._memo_key:
                self._memo = self._geo.row(self._category, key)
                self._memo_key = key
            return self._memo

        def i_seqs_for_row(self, row: int):
            return self._rowdata(row)[0]

        def rowCount(self, parent=QModelIndex()):
            if parent.isValid():
                return 0
            return self._n if self._filter is None else len(self._filter)

        def columnCount(self, parent=QModelIndex()):
            if parent.isValid() or not self._columns:
                return 0
            extra = 1 if self._source is not None else 0
            return 1 + len(self._columns) + extra  # "atoms" + values [+ "geostd"]

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid():
                return None
            col = index.column()
            src_col = self.source_column()
            if col == src_col:
                text, path = self.source_for_row(index.row())
                if role == Qt.ItemDataRole.DisplayRole:
                    return text
                if path is not None and role == Qt.ItemDataRole.ForegroundRole:
                    from PySide6.QtGui import QPalette
                    from PySide6.QtWidgets import QApplication

                    return QApplication.palette().color(QPalette.ColorRole.Link)  # theme's link color
                if path is not None and role == Qt.ItemDataRole.FontRole:
                    font = QFont()
                    font.setUnderline(True)
                    return font
                return None
            if role == Qt.ItemDataRole.DisplayRole:
                iseqs, vals = self._rowdata(index.row())
                if col == 0:
                    return "  ".join(self._label(i) for i in iseqs) if self._label else str(iseqs)
                v = vals.get(self._columns[col - 1])
                if v is None:
                    return ""
                return "" if v != v else f"{v:.3f}"  # v != v -> NaN
            if role == Qt.ItemDataRole.TextAlignmentRole and col > 0 and col != src_col:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return None

        def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
            if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
                headers = ["atoms"] + self._columns + (["geostd"] if self._source is not None else [])
                return headers[section]
            return None

    return RestraintTableModel()


def _make_checkable_combo():
    """A QComboBox with checkable items — looks like a normal dropdown, but its popup
    is a checklist. The closed control shows a short summary; toggling an item keeps
    the popup open and fires ``on_change(data, checked)``."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QStandardItem, QStandardItemModel
    from PySide6.QtWidgets import QComboBox, QStyle, QStyleOptionComboBox, QStylePainter

    class CheckableComboBox(QComboBox):
        def __init__(self):
            super().__init__()
            self.setModel(QStandardItemModel(self))
            self.view().viewport().installEventFilter(self)
            self.on_change = None  # callback(data, checked)
            self._press_index = None  # where a press landed inside the popup

        def add_checkable(self, text, checked, data):
            item = QStandardItem(text)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
            item.setData(data, Qt.ItemDataRole.UserRole)
            self.model().appendRow(item)

        def _summary(self):
            model = self.model()
            hidden = sum(
                1 for i in range(model.rowCount())
                if model.item(i).checkState() == Qt.CheckState.Unchecked
            )
            return "All shown" if not hidden else f"{hidden} hidden"

        def eventFilter(self, obj, event):
            if obj is self.view().viewport():
                if event.type() == QEvent.Type.MouseButtonPress:
                    # A press inside the open popup arms a toggle; consume it so the
                    # view doesn't start its own selection.
                    self._press_index = self.view().indexAt(event.position().toPoint())
                    return True
                if event.type() == QEvent.Type.MouseButtonRelease:
                    index = self.view().indexAt(event.position().toPoint())
                    pressed = self._press_index
                    self._press_index = None
                    # Toggle only on a real click *inside* the popup (press+release on
                    # the same item). The click that opens the dropdown presses on the
                    # combo, not the viewport, so it never toggles — it just opens.
                    if pressed is not None and pressed.isValid() and pressed == index:
                        item = self.model().itemFromIndex(index)
                        if item is not None and bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                            now = item.checkState() != Qt.CheckState.Checked
                            item.setCheckState(Qt.CheckState.Checked if now else Qt.CheckState.Unchecked)
                            self.update()  # repaint the summary
                            if self.on_change:
                                self.on_change(item.data(Qt.ItemDataRole.UserRole), now)
                    return True  # consume -> popup stays open, never auto-selects/closes
            return super().eventFilter(obj, event)

        def paintEvent(self, _event):
            painter = QStylePainter(self)
            opt = QStyleOptionComboBox()
            self.initStyleOption(opt)
            opt.currentText = self._summary()
            painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
            painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)

    return CheckableComboBox()


def _make_close_filter(on_close):
    """An event filter that reports a window being closed.

    The viewport and controls are two halves of one app, so closing either one
    should bring the whole thing down rather than leaving the other orphaned.
    """
    from PySide6.QtCore import QEvent, QObject

    class _CloseFilter(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.Close:
                on_close()
            return False  # let the widget close normally

    return _CloseFilter()


def _make_dock_close_filter(dock, host_alive):
    """Re-dock the controls when their detached window is closed, rather than lose them.

    When the controls float they get a real window frame with a close button; closing it
    should bring them back into the main window, not hide them. Only while the app is up
    (``host_alive``) — during shutdown the window must be allowed to close so quitting is
    not blocked.
    """
    from PySide6.QtCore import QEvent, QObject

    class _DockCloseFilter(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.Close and dock.isFloating() and host_alive():
                dock.setFloating(False)  # re-dock instead of closing
                event.ignore()
                return True
            return False

    return _DockCloseFilter()


class ViewportWindow:
    """A Qt window wrapping the Mol* viewer in a QWebEngineView."""

    def __init__(self, title: str = "pxviewer — viewport"):
        _check_qt()

        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        self._window = QWidget()
        self._window.setWindowTitle(title)
        icon = _app_icon()
        if icon is not None:
            self._window.setWindowIcon(icon)
        self._window.setMinimumSize(640, 480)

        layout = QVBoxLayout(self._window)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view = QWebEngineView()
        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        # The viewer takes the top; a guided-tutorial 'coach' pane docks under it (hidden
        # until a tutorial runs), so a tutorial never resizes the controls pane. ~4:1.
        layout.addWidget(self._view, stretch=4)
        layout.addWidget(self._build_coach_pane(), stretch=1)
        self._view.loadFinished.connect(self._verify_webgl)

    def _build_coach_pane(self):
        """The guided-tutorial coach: a pane docked below the viewer (hidden until a
        tutorial starts). Built here so it splits the viewport, not the controls pane; the
        ControlsWindow owns the logic and drives these widgets (see its tutorial methods)."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

        bar = QFrame()
        bar.setObjectName("coachPane")
        bar.setStyleSheet(
            "#coachPane { background:palette(alternate-base); "
            "border-top:2px solid palette(mid); }"
            "#coachPane QLabel { color:palette(text); }")
        bar.setMinimumHeight(110)
        v = QVBoxLayout(bar)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        head = QHBoxLayout()
        self.coach_title = QLabel("")
        self.coach_title.setStyleSheet("font-weight:600; color:palette(text);")
        head.addWidget(self.coach_title)
        head.addStretch(1)
        self.coach_progress = QLabel("")
        self.coach_progress.setStyleSheet("color:palette(placeholder-text);")
        head.addWidget(self.coach_progress)
        self.coach_close = QPushButton("✕")
        self.coach_close.setFixedWidth(26)
        self.coach_close.setFlat(True)
        self.coach_close.setToolTip("Exit the tutorial")
        head.addWidget(self.coach_close)
        v.addLayout(head)
        self.coach_text = QLabel("")
        self.coach_text.setWordWrap(True)
        self.coach_text.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.coach_text, stretch=1)
        row = QHBoxLayout()
        self.coach_show = QPushButton("Show me where")
        self.coach_show.setToolTip("Flash the button this step is about — you still click it")
        row.addWidget(self.coach_show)
        row.addStretch(1)
        self.coach_back = QPushButton("Back")
        row.addWidget(self.coach_back)
        self.coach_next = QPushButton("Next")
        row.addWidget(self.coach_next)
        v.addLayout(row)
        bar.setVisible(False)
        self.coach_bar = bar
        return bar

    def load(self, url: str) -> None:
        from PySide6.QtCore import QUrl
        self._view.load(QUrl(url))

    def _verify_webgl(self, ok: bool) -> None:
        """Once the page has loaded, confirm it actually got a WebGL context.

        Only when :mod:`pxviewer.gpu` armed the check (the auto path, outcome unknown):
        if the real viewport has no WebGL, the app restarts on software rendering rather
        than showing a blank viewer; if it does, that is remembered so no future launch
        pays for the check.
        """
        from . import gpu

        if not ok or not gpu.autofix_enabled():
            return
        self._view.page().runJavaScript(
            gpu.webgl_probe_js,
            lambda has_webgl: gpu.mark_hardware_ok() if has_webgl else gpu.on_webgl_missing())

    def show(self) -> None:
        self._window.show()

    def set_geometry(self, rect) -> None:
        self._window.setGeometry(rect)

    def widget(self):
        return self._window

    def close(self) -> None:
        """Tear the viewport down, releasing its QtWebEngine render process.

        A QWebEngineView keeps a Chromium render process alive for as long as it
        exists. Left undisposed — as when many DesktopApps are built and stopped in one
        process, e.g. a test run — those processes pile up, each still churning on the
        Mol* scene. ``setPage(None)`` detaches the page and releases the render process;
        the view and window are then scheduled for deletion. No event loop is pumped
        here: this runs from ``DesktopApp.stop`` (on ``aboutToQuit`` in the real app),
        where re-entering the loop would be unsafe.
        """
        try:
            self._view.stop()
            self._view.setPage(None)   # detaches and tears down the render process
            self._view.deleteLater()
            self._window.close()
            self._window.deleteLater()
        except Exception:  # pragma: no cover - defensive teardown
            pass


class ControlsWindow:
    """Controls for the viewport: open supplied files or get remote/bundled content."""

    # Set here as well as in __init__ so a signal arriving mid-construction cannot find it
    # missing; _UNSET means "no pane built yet", so the first update always builds one.
    _appearance_sig = _UNSET

    def __init__(self, desktop: "DesktopApp", title: str = "pxviewer — controls"):
        _check_qt()

        from PySide6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QProgressBar,
            QPushButton,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        self._desktop = desktop
        self._window = QWidget()
        self._window.setWindowTitle(title)
        icon = _app_icon()
        if icon is not None:
            self._window.setWindowIcon(icon)
        self._window.setMinimumSize(300, 480)  # compact — the viewer takes the space
        from PySide6.QtGui import QPalette

        # Tint icons to WindowText, not ButtonText: on macOS native buttons draw their own
        # text, so ButtonText can stay black and not track dark mode — which would paint every
        # icon dark-on-dark. WindowText follows the system appearance reliably everywhere.
        self._icon_role = QPalette.ColorRole.WindowText
        self._btn_tint = self._window.palette().color(self._icon_role)
        # Icons are baked pixmaps, so (unlike palette() stylesheets) they do not re-color on a
        # theme change on their own. Register each so _retint_icons can rebuild them — needed
        # both for a live light/dark switch and because the real palette may only land once the
        # window is shown (the tint read here can be the pre-show default).
        self._icon_registry: list = []   # (apply_icon(QIcon), name, size)
        self._tab_icon_names: list = []  # (tab index, name)
        self._theme_signature = None

        layout = QVBoxLayout(self._window)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Guided-tutorial state. The coach pane lives on the viewport window (so a tutorial
        # splits the viewer, not this controls pane); this window owns the logic and drives
        # those widgets — wire their buttons to our handlers.
        self._tutorial = None            # active Tutorial (see pxviewer.tutorial)
        self._tutorial_step = 0
        self._tutorial_timer = None      # polls the step's done() predicate while active
        self._hl_timer = None            # pulses a step's highlight target
        self._hl_overlay = None          # a transparent ring drawn over it (no layout effect)
        self._wire_coach()

        self._console = None  # EmbeddedConsole, created lazily on first tab view
        self._console_started = False
        self._items: list = []  # last Loaded-tree items summary (for the appearance pane)
        self._focused: tuple = (None, None)  # (kind, id) currently shown in Appearance
        # Buttons to grey out while the operation they start is running, keyed by its label
        # (see _register_busy_button). Set up before the tabs, which register into it.
        self._busy_buttons: dict = {}

        from PySide6.QtCore import Qt, QSize

        tabs = QTabWidget()
        # The controls pane is narrow — a third of the window, down to 300px, narrower
        # still when floated — so seven text tabs overflow. Use an icon per tab instead (the
        # label becomes its tooltip): icon-only tabs are compact enough that all seven fit at
        # any width, with no scroll arrows hiding any. Document mode drops the heavy frame.
        tabs.setDocumentMode(True)
        tabs.tabBar().setUsesScrollButtons(False)
        # The seven icon tabs share the full bar width evenly -- left-tight tabs left a
        # dead grey field to the right of the last one, most of the bar at panel widths.
        tabs.tabBar().setExpanding(True)
        if _IS_MAC:
            tabs.tabBar().setStyleSheet(_TAB_BAR_QSS)
        tabs.setIconSize(QSize(20, 20))
        # Lucide line icons, tinted to the tab text color so they read in light and dark.
        tint = self._btn_tint
        specs = [
            (self._build_scene_tab(), "Scene", "layers"),
            (self._build_tools_tab(), "Tools", "wrench"),
            (self._build_validation_tab(), "Validation", "award"),
            (self._build_hotspots_tab(), "Hotspots", "flame"),
            (self._build_geometry_tab(), "Geometry", "drafting-compass"),
            (self._build_console_tab(), "Console", "square-terminal"),
            (self._build_settings_tab(), "Settings", "sliders-horizontal"),
        ]
        self._tab_labels: list = []
        for widget, label, icon_name in specs:
            icon = _line_icon(icon_name, tint)
            # Fall back to the text label if the icon asset is somehow missing.
            index = tabs.addTab(widget, icon, "") if icon is not None \
                else tabs.addTab(widget, label)
            tabs.setTabToolTip(index, label)
            self._tab_labels.append(label)
            if icon is not None:
                self._tab_icon_names.append((index, icon_name))
            if label == "Console":
                self._console_tab_index = index
        # The icons need a label too. A tooltip is set above, but it is slow to appear and
        # unreliable on Wayland, so also name the hovered tab in the always-visible status
        # line — immediate and never hidden. See _on_tab_hover / _set_status.
        bar = tabs.tabBar()
        bar.setMouseTracking(True)
        self._tab_hover_filter = _tab_hover_filter(bar, self._on_tab_hover)
        bar.installEventFilter(self._tab_hover_filter)
        # The console spins up an IPython kernel, so defer that cost until the tab
        # is actually opened.
        tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs = tabs  # kept so a tutorial can reveal the tab holding a highlight target
        layout.addWidget(tabs, stretch=1)

        # Keep icon tints tracking the theme: a filter catches a live palette change (or the
        # real appearance landing on show, common on macOS), and a one-shot re-tint covers the
        # case where the palette is already right by the time the loop starts.
        from PySide6.QtCore import QTimer

        self._palette_watcher = _palette_watch_filter(self._window, self._refresh_theme)
        # Startup only needs icon tinting. Rebuilding every stylesheet while native macOS
        # controls are still being mapped is unsafe; full repolishing is reserved for an
        # actual palette-change event after the window is running.
        QTimer.singleShot(0, self._retint_icons)

        # A slim, always-visible status line, with the app icon + Help on the far side.
        # It doubles as the tab labeller on hover, so remember the real status underneath.
        # A busy bar directly above the status line. Some operations run for tens of seconds
        # (probe2, reduce2, a hotspot score), and the status text alone is easy to miss — this
        # is motion, so it reads as "working" at a glance. Indeterminate: none of these report
        # progress, and a fake percentage would be a lie.
        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 0)
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setFixedHeight(6)  # thin, but enough that the motion is unmissable
        self._busy_bar.setVisible(False)
        layout.addWidget(self._busy_bar)

        status_row = QHBoxLayout()
        self._real_status = "Ready"
        self._tab_hover = False
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: palette(placeholder-text);")
        status_row.addWidget(self._status_label, stretch=1)
        self._reset_view_btn = self._make_icon_button(
            "fullscreen", "Reset view",
            "Reset the view — reframe the camera to fit the whole scene")
        self._reset_view_btn.clicked.connect(lambda: self._desktop.reset_view())
        status_row.addWidget(self._reset_view_btn)
        self._picture_btn = self._make_icon_button(
            "camera", "Picture", "Save a picture of the viewport as a PNG")
        self._picture_btn.clicked.connect(self._on_save_picture)
        status_row.addWidget(self._picture_btn)
        # Always-visible dock/detach control: the painted header's button is hidden while
        # the panel floats (it uses the native window frame then), so this is the reliable
        # way back — and the way out, from either state.
        self._dock_btn = self._make_icon_button(
            "maximize-2", "Detach", "Detach the controls to their own window")
        self._dock_btn.clicked.connect(self._desktop.toggle_controls_dock)
        status_row.addWidget(self._dock_btn)
        self._mouse_btn = self._make_icon_button(
            "mouse", "Mouse", "Mouse and keyboard controls for the viewport")
        self._mouse_btn.clicked.connect(self._on_mouse_help)
        status_row.addWidget(self._mouse_btn)
        self._help_btn = self._make_icon_button(
            "circle-question-mark", "Help", "Documentation (coming soon)")
        self._help_btn.clicked.connect(self._on_help)
        status_row.addWidget(self._help_btn)
        layout.addLayout(status_row)

        self._suppress_model_events = False
        # Which model the atoms table shows. Defaults to the active model but the
        # user can pin it to a secondary one via the table's model dropdown.
        self._table_model_id: Optional[str] = None
        self._table_pinned = False
        self._scene_selection: dict = {}  # last {model_id: [indices]} snapshot
        self._models_summary: list = []
        self._suppress_table_model_combo = False
        desktop.bridge.scene_selection_changed.connect(self._on_scene_selection_changed)
        desktop.bridge.status_changed.connect(self._set_status)
        desktop.bridge.status_warned.connect(self._flash_status)
        desktop.bridge.loaded_changed.connect(self._on_loaded_changed)
        desktop.bridge.restraints_changed.connect(self._on_restraints_changed)
        desktop.bridge.analysis_ready.connect(self._on_analysis_ready)
        desktop.bridge.validation_ready.connect(self._on_validation_ready)
        desktop.bridge.validation_stale_changed.connect(self._set_validation_stale)
        desktop.bridge.hotspots_ready.connect(self._on_hotspots_ready)
        desktop.bridge.concern_ready.connect(self._on_concern_ready)
        desktop.bridge.busy_changed.connect(self._on_busy_changed)
        desktop.bridge.minimizing_changed.connect(self._on_minimizing_changed)
        desktop.bridge.ligand_placed.connect(self._on_ligand_placed)
        desktop.bridge.volume_iso_changed.connect(self._on_volume_iso_changed)
        self._update_minimize_map()  # nothing loaded yet, so no map to minimize into
        self._update_tug_density()
        self._update_pair_button()
        self._fit_tree_height()  # the empty list must not reserve space either
        self._appearance_sig = _UNSET  # no pane built yet, so the first update must build one
        self._update_appearance()  # empty-state placeholder

    # -- tabs ------------------------------------------------------------

    def _build_scene_tab(self):
        """Home: open files, the object list, appearance of the focused object, selection."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QButtonGroup,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QPushButton,
            QScrollArea,
            QTreeWidget,
            QVBoxLayout,
            QWidget,
        )

        outer = QWidget()
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(8)

        # -- Objects (the spine), pinned above everything else -------------
        # No forced heights anywhere here: a QPushButton only gets its native macOS
        # chrome at the height the style wants, and overriding it drops the button to a
        # squared-off fallback that looks nothing like the rest of the app.
        ol.addWidget(QLabel("<b>Objects</b>"))
        self._loaded_tree = QTreeWidget()
        # Height follows the contents (see _fit_tree_height): a QTreeWidget's sizeHint is
        # a fixed ~256px whatever it holds, and given a stretch it takes that much and
        # pushes the rest of the pane into a scrollbar. On a 13" screen that space is the
        # difference between the pane fitting and not.
        self._loaded_tree.setMinimumHeight(_TREE_MIN_HEIGHT)
        # Columns: [visible] [active] [name]. Toggles on the left; name last, elides.
        self._loaded_tree.setColumnCount(3)
        self._loaded_tree.setHeaderHidden(True)
        header = self._loaded_tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._loaded_tree.itemChanged.connect(self._on_tree_item_changed)
        self._loaded_tree.currentItemChanged.connect(self._on_tree_current_changed)
        self._loaded_tree.itemClicked.connect(self._on_tree_item_clicked)
        self._active_group = QButtonGroup(self._window)  # exclusive active-model radios
        self._active_group.setExclusive(True)
        self._active_group.buttonClicked.connect(self._on_active_radio)
        ol.addWidget(self._loaded_tree, stretch=1)

        # -- Actions on the objects: a compact icon toolbar -----------------
        # Icon-only (the words move to richer tooltips), one row in two groups: get data in
        # and out | act on what is loaded and the view. Lucide icons tinted to the button
        # text color, with the old label kept as fallback text if an asset is missing.
        def _icon_button(icon_name, label, tooltip, on_click=None):
            b = self._make_icon_button(icon_name, label, tooltip)
            if on_click is not None:
                b.clicked.connect(on_click)
            return b

        self._open_btn = _icon_button(
            "folder-open", "Open",
            "Open a structure or map — models via cctbx, maps as .mrc/.map/.ccp4",
            self._on_open_file)
        self._get_btn = _icon_button(
            "blocks", "Get",
            "Get data from PDB or EMDB, open a bundled example, or start a tutorial")
        self._get_btn.setMenu(self._build_get_menu())
        self._write_btn = _icon_button(
            "save", "Save", "Save the focused object to disk — model coordinates, or a map",
            self._on_write_object)
        self._pair_btn = _icon_button(
            "combine", "Pair",
            "Pair an unpaired model with a map, or check an existing pair for a missing "
            "density-supported origin shift",
            self._on_pair)
        self._remove_model_btn = _icon_button(
            "trash-2", "Remove", "Remove the highlighted object", self._on_remove_selected)

        actions = QHBoxLayout()
        actions.setSpacing(4)
        for button in (self._open_btn, self._get_btn, self._write_btn,
                       self._pair_btn):
            actions.addWidget(button)
        actions.addSpacing(14)  # separate "data in / out" from "act on it"
        actions.addWidget(self._remove_model_btn)
        actions.addStretch(1)  # keep them a compact, left-packed toolbar
        ol.addLayout(actions)

        self._file_label = QLabel("")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: palette(placeholder-text);")
        ol.addWidget(self._file_label)

        # Everything below scrolls, so a busy scene never clips the controls.
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(10)

        # -- Appearance of the focused object ----------------------------
        self._appearance_box = QGroupBox("Appearance")
        self._appearance_layout = QVBoxLayout(self._appearance_box)
        self._appearance_layout.setSpacing(6)
        layout.addWidget(self._appearance_box)

        # -- Selection ---------------------------------------------------
        sel_box = QGroupBox("Selection")
        sl = QVBoxLayout(sel_box)
        sl.setSpacing(6)
        # One row: pick-toggle | selection box | apply | clear. The two jobs are separate
        # controls so neither is ambiguous — pick-mode on the left, apply-the-string on the
        # right of the box it acts on.
        sel_row = QHBoxLayout()
        self._pick_btn = self._make_icon_button(
            "mouse-pointer-click", "Pick",
            "Pick atoms in the 3D view — each click adds to the selection; click empty space "
            "to clear; Shift-click extends the selection", checkable=True)
        self._pick_btn.toggled.connect(self._on_toggle_select)
        sel_row.addWidget(self._pick_btn)

        self._select_expr = QLineEdit()
        self._select_expr.setPlaceholderText("selection, e.g. chain A and resseq 5:14")
        self._select_expr.setToolTip("A cctbx / Phenix selection string on the active model.")
        self._select_expr.returnPressed.connect(self._on_select_expression)
        sel_row.addWidget(self._select_expr, stretch=1)

        apply_btn = self._make_icon_button(
            "arrow-right", "Apply", "Apply the selection string to the active model")
        apply_btn.clicked.connect(self._on_select_expression)
        sel_row.addWidget(apply_btn)

        self._clear_btn = self._make_icon_button("circle-off", "Clear", "Clear the selection")
        self._clear_btn.clicked.connect(self._on_clear_selection)
        sel_row.addWidget(self._clear_btn)
        sl.addLayout(sel_row)

        sl.addWidget(QLabel("Selected:"))
        self._selection_label = QLabel("None")
        self._selection_label.setWordWrap(True)
        self._selection_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._selection_label.setMinimumHeight(52)
        self._selection_label.setStyleSheet("color: palette(placeholder-text);")
        sl.addWidget(self._selection_label)

        # Hide / show the selected atoms (a partial representation, like the type toggles).
        vis_row = QHBoxLayout()
        self._hide_sel_btn = self._make_icon_button(
            "eye-off", "Hide", "Hide the selected atoms")
        self._hide_sel_btn.clicked.connect(lambda: self._desktop.hide_selected())
        vis_row.addWidget(self._hide_sel_btn)
        self._show_sel_btn = self._make_icon_button(
            "eye", "Show", "Show (un-hide) the selected atoms")
        self._show_sel_btn.clicked.connect(lambda: self._desktop.show_selected())
        vis_row.addWidget(self._show_sel_btn)
        vis_row.addStretch(1)
        self._hide_sel_btn.setEnabled(False)  # enabled once there is a selection
        self._show_sel_btn.setEnabled(False)
        sl.addLayout(vis_row)

        layout.addWidget(sel_box)

        # The mouse/key reference used to live here, but it ate the room the object list
        # wants; it's now a popup off the mouse button in the status row (see _on_mouse_help).
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(body)
        ol.addWidget(scroll, stretch=1)
        return outer

    def _build_get_menu(self):
        """Everything pxviewer obtains for the user: remote data, examples and tutorials."""
        from PySide6.QtWidgets import QMenu

        from . import tutorial

        menu = QMenu(self._window)
        menu.setObjectName("getMenu")
        self._add_menu_heading(menu, "Online", first=True)
        menu.addAction("Fetch from PDB / EMDB…", self._on_fetch)
        # Named for what the example is *for*, with the structure in parentheses. A menu
        # of protein names asks the reader to already know which one demonstrates what,
        # and to remember a PDB code to find it again; the task is the thing they came
        # here with.
        self._add_menu_heading(menu, "Examples")
        menu.addAction("Model only (1UBQ)",
                       lambda: self._on_load_sample("1ubq.pdb"))
        menu.addAction("Map + model (1UBQ)",
                       self._on_run_map_model_demo)
        menu.addAction("Validation (1TEC)",
                       lambda: self._on_load_sample("1tec.pdb"))
        menu.addAction("X-ray maps from reflections (1UBQ)",
                       self._on_run_xray_demo)
        menu.addAction("Ligand fitting (ATP into a difference map)",
                       self._on_run_ligand_fitting_demo)
        menu.addAction("Real-space refinement (cryo-EM)",
                       self._on_run_real_space_refinement_demo)
        menu.addAction("Restraint edits (Zn site)",
                       lambda: self._on_load_sample("zn_site.pdb"))
        menu.addAction("Alternate conformations (3NIR)",
                       lambda: self._on_load_sample("3nir.pdb"))
        self._add_menu_heading(menu, "Tutorials")
        for tut in tutorial.all_tutorials():
            menu.addAction(tut.title, lambda _c=False, t=tut: self._start_tutorial(t))
        return menu

    def _add_menu_heading(self, menu, text: str, *, first: bool = False):
        """Add a real, visible section heading to a popup menu.

        Not ``QMenu.addSection``: on macOS the style draws the separator and silently drops
        the label, so the headings were invisible here even though the QAction carried the
        text (which is why a test asserting on the text still passed). A QWidgetAction owns
        its own QLabel, so what is written is what is drawn, on every platform.

        Returns the action, and takes ``first`` to skip the divider above the top heading.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QLabel, QWidgetAction

        if not first:
            menu.addSeparator()
        label = QLabel(text)
        font = QFont(label.font())
        font.setBold(True)
        # Headings label the groups; they must not compete with the items they head.
        font.setPointSizeF(max(9.0, font.pointSizeF() - 1.0))
        label.setFont(font)
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        label.setStyleSheet("color: palette(placeholder-text); padding: 6px 12px 2px 12px;")
        holder = QWidgetAction(menu)
        holder.setDefaultWidget(label)
        holder.setEnabled(False)  # a heading is not a target
        menu.addAction(holder)
        return holder

    def _build_tools_tab(self):
        """Geometry-focused tools: measure from the selection. (Clash/contact analysis
        lives in the Validation tab, alongside the other MolProbity checks.)"""
        from PySide6.QtWidgets import (
            QCheckBox,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        measure = QGroupBox("Measure")
        mg = QVBoxLayout(measure)
        mg.addWidget(QLabel("Select the atoms, then measure:"))
        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        specs = [("Distance", "distance", 2, "ruler"),
                 ("Angle", "angle", 3, "triangle-right"),
                 ("Dihedral", "dihedral", 4, "waypoints")]
        for label, kind, n, icon_name in specs:
            btn = self._make_icon_button(
                icon_name, label, f"Measure the {label.lower()} from {n} selected atoms")
            btn.clicked.connect(lambda _c=False, k=kind: self._on_measure(k))
            mrow.addWidget(btn)
        from PySide6.QtWidgets import QFrame

        sep = QFrame()  # divide the measures from Clear
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        mrow.addWidget(sep)
        clear_m = self._make_icon_button(
            "circle-off", "Clear measurements", "Clear all measurements")
        clear_m.clicked.connect(self._on_clear_measurements)
        mrow.addWidget(clear_m)
        mrow.addStretch(1)
        mg.addLayout(mrow)
        layout.addWidget(measure)

        map_tools = QGroupBox("Map tools")
        map_layout = QVBoxLayout(map_tools)
        map_layout.addWidget(QLabel(
            "Calculate local resolution from two half-maps, then colour the map by it:"))
        self._localres_btn = self._make_icon_button(
            "palette", "Local res",
            "Calculate local resolution from two half-maps and colour the map by it")
        self._localres_btn.clicked.connect(self._on_localres_wizard)
        map_row = QHBoxLayout()
        map_row.addWidget(self._localres_btn)
        map_row.addStretch(1)
        map_layout.addLayout(map_row)
        layout.addWidget(map_tools)

        layout.addWidget(self._build_edits_group())

        layout.addWidget(self._build_ligand_placement_group())

        minimization = QGroupBox("Minimization")
        ming = QVBoxLayout(minimization)
        ming.addWidget(QLabel("Relax the model onto ideal geometry:"))
        self._refine_drag_btn = self._make_icon_button(
            "hand", "Refine drag",
            "Explicitly enable coordinate-changing atom drags. While active, drag an atom "
            "to pull it and locally minimize the model. Mutually exclusive with Pick.",
            checkable=True)
        self._refine_drag_btn.toggled.connect(self._on_toggle_refine_drag)
        ming.addWidget(self._refine_drag_btn)
        self._minimize_map_check = QCheckBox("Use map")
        self._minimize_map_check.setToolTip(
            "Also pull the model into the density. Needs a map loaded together with "
            "the model as a group, so the two share a frame.")
        ming.addWidget(self._minimize_map_check)
        min_row = QHBoxLayout()
        self._minimize_btn = self._make_icon_button(
            "play", "Minimize",
            "Minimize the active model against its geometry restraints (no map), "
            "streaming each step into the viewport as it runs")
        self._minimize_btn.clicked.connect(self._on_minimize)
        min_row.addWidget(self._minimize_btn)
        self._minimize_stop_btn = self._make_icon_button(
            "pause", "Stop", "Halt the run, keeping the progress so far")
        self._minimize_stop_btn.setEnabled(False)
        self._minimize_stop_btn.clicked.connect(lambda: self._desktop.stop_minimization())
        min_row.addWidget(self._minimize_stop_btn)
        self._on_minimizing_changed(False)  # paint the idle look (Minimize green, Stop quiet)
        min_row.addStretch()
        ming.addLayout(min_row)
        layout.addWidget(minimization)

        layout.addStretch()
        # Wrap in a scroll area (like the Scene tab). Without it, when the four groups are
        # taller than the pane the layout compresses them instead — and the icon buttons,
        # whose stylesheet lowers their minimum height, get squashed flat (the Ligand
        # placement row rendered 30x17 rather than 30x26). Scrolling keeps every widget at
        # its natural size.
        from PySide6.QtWidgets import QScrollArea

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(tab)
        return scroll

    def _build_drag_group(self):
        """The 'Drag atoms' options (lives on the Settings tab)."""
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QSpinBox,
            QVBoxLayout,
        )

        dragging = QGroupBox("Drag atoms")
        dg = QVBoxLayout(dragging)
        hint = QLabel(
            "Enable Refine drag on the Tools tab, then drag any atom or bond to pull it; "
            "the model bends to follow.")
        hint.setWordWrap(True)
        dg.addWidget(hint)

        # What a drag lets move — Coot's refine scopes. A sphere (whole residues within a
        # radius), a single residue, or a stretch of residues each side along the chain.
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Moves:"))
        scope_combo = QComboBox()
        scope_combo.addItem("Sphere", "sphere")
        scope_combo.addItem("Single residue", "single")
        scope_combo.addItem("Residue stretch", "stretch")
        scope_combo.addItem("Selection", "selection")
        scope_combo.setToolTip(
            "What a drag is allowed to move:\n"
            "• Sphere — every residue within the radius (the default)\n"
            "• Single residue — only the one you grab\n"
            "• Residue stretch — that residue and a few each side along the chain\n"
            "• Selection — exactly the residues you have picked (select some first)")
        scope_row.addWidget(scope_combo)
        radius_spin = QDoubleSpinBox()
        radius_spin.setRange(2.0, 20.0)
        radius_spin.setDecimals(0)
        radius_spin.setSingleStep(1.0)
        radius_spin.setSuffix(" Å")
        radius_spin.setValue(self._desktop._tug_scope["radius"])
        radius_spin.setToolTip("Radius of the sphere that gives way.")
        scope_row.addWidget(radius_spin)
        flank_spin = QSpinBox()
        flank_spin.setRange(1, 20)
        flank_spin.setPrefix("± ")
        flank_spin.setSuffix(" res")
        flank_spin.setValue(2)
        flank_spin.setToolTip("How many residues each side of the grabbed one also move.")
        scope_row.addWidget(flank_spin)
        scope_row.addStretch()
        dg.addLayout(scope_row)

        def _apply_scope() -> None:
            kind = scope_combo.currentData()
            radius_spin.setVisible(kind == "sphere")
            flank_spin.setVisible(kind == "stretch")
            if kind == "sphere":
                self._safe(lambda: self._desktop.set_tug_scope(
                    mode="sphere", radius=radius_spin.value()))
            elif kind == "single":
                self._safe(lambda: self._desktop.set_tug_scope(mode="residues", flank=0))
            elif kind == "stretch":
                self._safe(lambda: self._desktop.set_tug_scope(
                    mode="residues", flank=flank_spin.value()))
            else:  # selection
                self._safe(lambda: self._desktop.set_tug_scope(mode="selection"))

        scope_combo.currentIndexChanged.connect(lambda _i: _apply_scope())
        radius_spin.valueChanged.connect(lambda _v: _apply_scope())
        flank_spin.valueChanged.connect(lambda _v: _apply_scope())
        _apply_scope()  # set initial visibility (radius shown, flank hidden)

        self._tug_density_check = QCheckBox("Into the density")
        self._tug_density_check.setToolTip(
            "Let the map pull too, so a drag settles the neighborhood into density "
            "rather than only bending it. Needs a map paired with the model.")
        self._tug_density_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_tug_into_density(on)))
        dg.addWidget(self._tug_density_check)
        self._tug_continuous_check = QCheckBox("Keep minimizing while dragging")
        self._tug_continuous_check.setToolTip(
            "While dragging, the model keeps relaxing the whole time — a gentle living "
            "settle that stays in motion even when the pointer is still, rather than "
            "nudging once per move and stopping.")
        self._tug_continuous_check.setChecked(True)  # on by default; connect after, no fire
        self._tug_continuous_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_tug_continuous(on)))
        dg.addWidget(self._tug_continuous_check)
        self._tug_livemap_check = QCheckBox("Live difference map")
        self._tug_livemap_check.setToolTip(
            "While dragging, recompute the mFo-DFc difference map in a small window around "
            "the atom and show it live — green where the data wants density, red where the "
            "model has too much. Honest feedback as you fit (the main 2mFo-DFc map is left "
            "alone to avoid model bias). Needs a map phased from reflections; use Recompute "
            "for the whole-structure maps.")
        self._tug_livemap_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_live_difference_map(on)))
        dg.addWidget(self._tug_livemap_check)
        return dragging

    def _build_ligand_placement_group(self):
        """Permanent 'Ligand placement' panel (Tools tab): drop a ligand marker, then build
        a ligand at it — all in one place, rather than reaching into a marker's Appearance
        pane. It acts on the most recently placed marker."""
        from PySide6.QtWidgets import (
            QCheckBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout)

        box = QGroupBox("Ligand placement")
        lg = QVBoxLayout(box)

        mk_row = QHBoxLayout()
        place_btn = self._make_icon_button(
            "circle-arrow-out-up-left", "Place ligand marker",
            "Arm placement, then click in the viewport: a ligand marker is dropped there — "
            "snapped to the atom under the cursor, or the view plane in empty space.")
        place_btn.clicked.connect(self._desktop.arm_marker)
        self._lig_place_btn = place_btn  # a tutorial highlight target
        mk_row.addWidget(place_btn)
        clear_mk = self._make_icon_button("circle-off", "Clear", "Remove all ligand markers")
        clear_mk.clicked.connect(self._desktop.clear_markers)
        mk_row.addWidget(clear_mk)
        mk_row.addStretch(1)
        lg.addLayout(mk_row)

        self._lig_target_label = QLabel("")
        self._lig_target_label.setWordWrap(True)
        self._lig_target_label.setStyleSheet("color: palette(placeholder-text);")
        lg.addWidget(self._lig_target_label)

        lg.addWidget(QLabel("Ligand (monomer code):"))
        self._lig_code_edit = QLineEdit()
        self._lig_code_edit.setPlaceholderText("e.g. GOL, ATP, NAG")
        self._lig_code_edit.setMaxLength(5)
        self._lig_code_edit.returnPressed.connect(lambda: self._safe(self._on_fit_ligand))
        lg.addWidget(self._lig_code_edit)

        lg.addWidget(QLabel("or SMILES (code above names it):"))
        self._lig_smiles_edit = QLineEdit()
        self._lig_smiles_edit.setPlaceholderText("e.g. CC(=O)Oc1ccccc1C(=O)O")
        self._lig_smiles_edit.returnPressed.connect(lambda: self._safe(self._on_fit_ligand))
        lg.addWidget(self._lig_smiles_edit)

        self._lig_fit_check = QCheckBox("Fit into density (explode-and-refine)")
        self._lig_fit_check.setToolTip(
            "Settle the ligand into the active model's map with a large radius of "
            "convergence. Needs a map paired with the model.")
        lg.addWidget(self._lig_fit_check)

        self._lig_fit_btn = QPushButton("Fit ligand here")
        self._lig_fit_btn.setToolTip(
            "Build the ligand (from the monomer library, or the SMILES string), center it "
            "on the ligand marker, and add it as a new object.")
        self._lig_fit_btn.clicked.connect(lambda: self._safe(self._on_fit_ligand))
        self._register_busy_button(self._lig_fit_btn, "Building ligand")
        lg.addWidget(self._lig_fit_btn)

        self._lig_last_target = None
        self._update_ligand_panel()
        return box

    def _build_edits_group(self):
        """Custom geometry-restraint edits: bond/angle/dihedral restraints added on top of
        the library, authored from the current selection (same as Measure) and saved/loaded
        as a phenix geometry_restraints.edits PHIL file."""
        from PySide6.QtWidgets import (
            QGroupBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout)

        box = QGroupBox("Restraint edits")
        v = QVBoxLayout(box)
        v.addWidget(QLabel(
            "Custom restraints on top of the library — a covalent link, a metal bond.\n"
            "Select the atoms, then add (the current geometry becomes the target):"))
        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        for label, kind, n in [("Bond", "bond", 2), ("Angle", "angle", 3),
                               ("Dihedral", "dihedral", 4)]:
            b = QPushButton(label)
            b.setToolTip(f"Add a custom {kind} restraint from {n} selected atoms of the "
                         "active model, honored by this app's minimize/drag and exportable "
                         "for phenix.refine.")
            b.clicked.connect(lambda _c=False, k=kind: self._on_add_edit(k))
            add_row.addWidget(b)
            if kind == "bond":
                self._edit_bond_btn = b  # a tutorial highlight target
        add_row.addStretch(1)
        v.addLayout(add_row)

        self._edits_list = QListWidget()
        self._edits_list.setToolTip("Restraint edits on the active model. Select one to remove it.")
        self._edits_list.setMaximumHeight(90)
        v.addWidget(self._edits_list)

        file_row = QHBoxLayout()
        file_row.setSpacing(6)
        remove = QPushButton("Remove")
        remove.setToolTip("Remove the selected edit")
        remove.clicked.connect(self._on_remove_edit)
        clear = QPushButton("Clear")
        clear.setToolTip("Remove all edits from the active model")
        clear.clicked.connect(self._on_clear_edits)
        load = QPushButton("Load…")
        load.setToolTip("Read a geometry_restraints.edits PHIL file and add its edits")
        load.clicked.connect(self._on_load_edits)
        self._edit_load_btn = load  # a tutorial highlight target
        save = QPushButton("Save…")
        save.setToolTip("Write the active model's edits as a geometry_restraints.edits PHIL file")
        save.clicked.connect(self._on_save_edits)
        self._edit_save_btn = save  # a tutorial highlight target
        for b in (remove, clear, load, save):
            file_row.addWidget(b)
        file_row.addStretch(1)
        v.addLayout(file_row)
        self._refresh_edits_list()
        return box

    def _refresh_edits_list(self) -> None:
        """Show the active model's edits (called on load/selection changes)."""
        if not hasattr(self, "_edits_list"):
            return
        self._edits_list.clear()
        mid = self._desktop._active_model_id
        item = next((it for it in self._desktop._loaded_summary()["items"]
                     if it["kind"] == "model" and it["id"] == mid), None)
        for e in (item.get("edits") if item else []) or []:
            self._edits_list.addItem(e["summary"])

    def _on_add_edit(self, kind: str) -> None:
        mid = self._desktop._active_model_id
        if mid is None:
            self._set_status("load a model first")
            return
        try:
            self._desktop.add_edit_from_selection(mid, kind)
        except Exception as exc:
            self._flash_status(str(exc))

    def _on_remove_edit(self) -> None:
        row = self._edits_list.currentRow()
        if row < 0:
            self._set_status("select an edit to remove")
            return
        self._safe(lambda: self._desktop.remove_edit(self._desktop._active_model_id, row))

    def _on_clear_edits(self) -> None:
        self._safe(lambda: self._desktop.clear_edits(self._desktop._active_model_id))

    def _on_load_edits(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        mid = self._desktop._active_model_id
        if mid is None:
            self._set_status("load a model first")
            return
        # Open on the bundled sample edits file, so the tutorial's file is right there.
        sample = sample_structure_path("zn_site_edits.phil")
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Load restraint edits", str(sample) if sample else "",
            "Edits PHIL (*.phil *.params *.txt *.eff)")
        if not path:
            return
        try:
            added = self._desktop.load_edits(mid, path)
            self._set_status(
                f"Loaded {added} restraint edit(s) from {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self._window, "Load edits failed", str(exc))

    def _on_save_edits(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        mid = self._desktop._active_model_id
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save restraint edits", "edits.phil", "Edits PHIL (*.phil)")
        if not path:
            return
        try:
            self._desktop.save_edits(mid, path)
            self._set_status(f"Saved edits to {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self._window, "Save edits failed", str(exc))

    def _on_fit_ligand(self) -> None:
        """Build a ligand at the most recently placed ligand marker, from the panel fields."""
        markers = self._desktop._markers
        if not markers:
            self._set_status("place a ligand marker first")
            return
        mid = markers[-1]["id"]
        smiles = self._lig_smiles_edit.text().strip()
        code = self._lig_code_edit.text()
        fit = self._lig_fit_check.isChecked()
        if smiles:  # SMILES wins; the code field just names the residue
            self._desktop.fit_ligand_from_smiles_at_marker(mid, smiles, code or "LIG", fit=fit)
        else:
            self._desktop.fit_ligand_at_marker(mid, code, fit=fit)

    def _on_ligand_placed(self) -> None:
        """A ligand was built and added — clear the code and SMILES inputs so the next one
        starts fresh, rather than silently reusing the last code as its name/restraints."""
        self._lig_code_edit.clear()
        self._lig_smiles_edit.clear()

    def _update_ligand_panel(self) -> None:
        """Enable the ligand fields only with a marker to place at, and reflect the current
        target (the last-placed marker) and whether a map is available to fit into."""
        if not hasattr(self, "_lig_target_label"):
            return  # panel not built yet
        markers = self._desktop._markers
        target = markers[-1] if markers else None
        has_map = self._desktop.map_for_model() is not None
        for w in (self._lig_code_edit, self._lig_smiles_edit, self._lig_fit_btn):
            w.setEnabled(target is not None)
        self._lig_fit_check.setEnabled(target is not None and has_map)
        target_id = target["id"] if target else None
        if target_id != self._lig_last_target:  # a fresh target: default fit on if a map
            self._lig_fit_check.setChecked(target is not None and has_map)
            self._lig_last_target = target_id
        elif not has_map:
            self._lig_fit_check.setChecked(False)
        self._lig_target_label.setText(
            f"Placing at: {target['name']}" if target else
            "Place a ligand marker to build a ligand at it.")

    def _offer_restraints_if_blocked(self, mid=None) -> None:
        """Before an action that needs restraints, offer to fix an unknown ligand.

        Minimize and drag do their restraint building on worker threads, which cannot ask
        the user anything -- so without this the answer to "why will nothing move?" was a
        line of cctbx text, and the offer to do something about it lived only in the
        Geometry tab, where there was no reason to look.

        Costs nothing when all is well: it acts only on a model whose *background* warm
        already failed and recorded why (see ``DesktopApp._prewarm_restraints``). Working
        it out here instead would mean a full interpretation pass on the GUI thread every
        time somebody pressed Minimize.
        """
        mid = mid or self._desktop._active_model_id
        entry = self._desktop._model_entry(mid) if mid else None
        if not (entry and entry.get("restraints_error")):
            return
        model = getattr(entry["session"], "model", None)
        if model is None:
            return
        try:
            if model.restraints_manager_available():
                entry.pop("restraints_error", None)   # something else built them meanwhile
                return
        except Exception:  # pragma: no cover - defensive
            return
        if self._offer_ligand_restraints(mid):
            entry.pop("restraints_error", None)

    def _on_minimize(self) -> None:
        # A model whose restraints will not build cannot be minimized, and the reason is
        # usually a ligand with no dictionary -- fixable, and worth offering here rather
        # than letting the run fail on a thread.
        self._offer_restraints_if_blocked()
        try:
            self._desktop.minimize_model(use_map=self._minimize_map_check.isChecked())
        except Exception as exc:
            self._set_status(str(exc))

    def _on_minimizing_changed(self, running: bool) -> None:
        """Stop is only meaningful while a run is going; Minimize only while one is not.

        Beyond enable/disable, color the live control so a glance tells you the state:
        Minimize glows green (ready to run) while idle, Stop glows amber (a run is going)
        while minimizing. The inactive one stays a plain, quiet button."""
        self._minimize_btn.setEnabled(not running)
        self._minimize_stop_btn.setEnabled(running)
        self._paint_minimize_button(self._minimize_btn, "play", "go", active=not running)
        self._paint_minimize_button(self._minimize_stop_btn, "pause", "stop", active=running)

    def _paint_minimize_button(self, btn, icon_name: str, accent: str, *, active: bool) -> None:
        """Give the active play/pause button a filled accent (white glyph on color); leave
        the inactive one in its default look. ``accent`` is a semantic name (see
        :func:`_accent`) so the fill tracks light/dark. Falls back to the button's text if
        the icon asset is gone (the accent style still applies)."""
        if active:
            color = _accent(btn, accent)
            btn.setStyleSheet(
                f"QPushButton {{ background:{color}; border:1px solid {color}; "
                f"border-radius:4px; }}")
            icon = _line_icon(icon_name, "#ffffff", size=18)
        else:
            btn.setStyleSheet(_icon_button_base_qss())  # back to the plain framed look
            icon = self._icon(icon_name)
        if icon is not None:
            btn.setIcon(icon)

    def _update_tug_density(self) -> None:
        """Tugging into density needs a map paired with the active model."""
        available = self._desktop.map_for_model() is not None
        self._tug_density_check.setEnabled(available)
        if not available:
            self._tug_density_check.setChecked(False)
            self._tug_density_check.setToolTip(
                "Pair the model with a map to let a drag settle it into density.")

    def _update_minimize_map(self) -> None:
        """Offer 'Use map' only when the active model actually has one to use."""
        available = self._desktop.map_for_model() is not None
        self._minimize_map_check.setEnabled(available)
        if not available:
            self._minimize_map_check.setChecked(False)
            self._minimize_map_check.setToolTip(
                "Load a model and a map together to pair them, then minimize into density.")

    def _build_clashes_page(self):
        """The all-atom contacts (probe2) view — shown as a validator sub-tab, so it sits
        beside the per-residue checks rather than in its own panel. It stays a separate run
        from 'Run validation' because it is heavier: it adds hydrogens (a new object) and
        shells out to probe2, so it is opt-in via its own Analyze button. The two overlays
        toggle independently once an analysis has produced dots."""
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

        from .live import PROBE_CLASHES, PROBE_CONTACTS

        page = QWidget()
        ag = QVBoxLayout(page)
        hint = QLabel("MolProbity all-atom contacts. Add hydrogens (reduce2), then run "
                      "probe2, and toggle the overlays:")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        ag.addWidget(hint)
        analyze = self._make_icon_button(
            "heading-1", "Add H + analyze",
            "Add hydrogens with reduce2 as a new object (hiding the original), then run "
            "probe2 for MolProbity contacts and clashes")
        analyze.clicked.connect(self._on_analyze)
        self._register_busy_button(analyze, "Adding hydrogens and running probe")
        self._contacts_toggle = self._make_icon_button(
            "fold-horizontal", "Contacts", "Show/hide the full probe2 contact-dot surface",
            checkable=True)
        self._contacts_toggle.setEnabled(False)
        self._contacts_toggle.toggled.connect(
            lambda on: self._desktop.set_probe_channel(PROBE_CONTACTS, on))
        self._contacts_toggle.toggled.connect(lambda _on: self._sync_all_markup_button())
        self._clashes_toggle = self._make_icon_button(
            "triangle-alert", "Clashes", "Show/hide the bad-overlap (clash) spikes",
            checkable=True)
        self._clashes_toggle.setEnabled(False)
        self._clashes_toggle.toggled.connect(
            lambda on: self._desktop.set_probe_channel(PROBE_CLASHES, on))
        self._clashes_toggle.toggled.connect(lambda _on: self._sync_all_markup_button())

        # One row: add-H (the prerequisite) then the two result toggles.
        prow = QHBoxLayout()
        prow.addWidget(analyze)
        prow.addWidget(self._contacts_toggle)
        prow.addWidget(self._clashes_toggle)
        prow.addStretch(1)
        ag.addLayout(prow)
        ag.addStretch(1)
        return page

    def _build_validation_tab(self):
        """MolProbity validation, all in one place. 'Run validation' runs every registered
        per-residue validator (data-driven from the validation registry) and each result
        becomes a sub-tab. The all-atom contacts analysis (probe2) is a peer sub-tab —
        'Clashes & contacts' — always present as the first tab, so both kinds of check live
        in the same results area rather than in separate panels."""
        from PySide6.QtWidgets import (
            QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        intro = QLabel("MolProbity validation of the active model. Run the per-residue "
                       "checks below; the Clashes & contacts tab runs probe2 separately.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Per-validator Markers checkboxes, rebuilt with the sub-tabs on every run. The probe
        # overlay toggles are separate because they live on the always-present Clashes page and
        # outlive a re-run; both are markup, so the "all" button drives them together.
        self._marker_checks: list = []

        run_row = QHBoxLayout()
        run_btn = QPushButton("Run validation")
        run_btn.setToolTip("Run every MolProbity per-residue validator on the active model "
                           "(background thread); each becomes a sub-tab below.")
        run_btn.clicked.connect(self._on_run_validation)
        self._validate_btn = run_btn  # a tutorial highlight target
        self._register_busy_button(run_btn, "Running validation")
        run_row.addWidget(run_btn, stretch=1)

        # Every validator draws on its own channel, so without this the only way to clear the
        # viewport is to visit each sub-tab and untick it. The label says what the click will
        # do, and tracks the individual boxes.
        self._all_markup_btn = QPushButton("Hide all markers")
        self._all_markup_btn.setToolTip(
            "Show or hide every validation overlay at once — the per-validator markup and the "
            "probe contact/clash dots — without visiting each sub-tab.")
        self._all_markup_btn.clicked.connect(self._on_toggle_all_markup)
        run_row.addWidget(self._all_markup_btn)
        layout.addLayout(run_row)

        # Shown when the atoms move after a run: the tables and markup still on screen
        # describe where the model *was*, so they can no longer be trusted. Hidden until
        # then, and cleared again by the next run. Driven by validation_stale_changed.
        self._stale_warning = QLabel(
            "⚠  The model has moved since this validation was computed — the results below "
            "no longer match the structure. Re-run validation to refresh them.")
        self._stale_warning.setWordWrap(True)
        self._stale_warning.setStyleSheet(
            "background: #B2182B; color: white; padding: 6px 9px; border-radius: 4px;")
        self._stale_warning.setVisible(False)
        layout.addWidget(self._stale_warning)

        # One results area: the always-present Clashes & contacts tab, then a tab per
        # validator, (re)built as runs complete.
        self._validation_subtabs = QTabWidget()
        self._validation_subtabs.setDocumentMode(True)
        self._clashes_page = self._build_clashes_page()
        self._validation_subtabs.addTab(self._clashes_page, "Clashes && contacts")
        layout.addWidget(self._validation_subtabs, stretch=1)
        self._sync_all_markup_button()  # nothing drawn yet, so it starts disabled
        return tab

    def _markup_toggles(self) -> list:
        """Every widget that shows/hides validation markup, across the sub-tabs: the
        per-validator Markers checkboxes and the two probe overlay toggles."""
        toggles = list(self._marker_checks)
        for name in ("_contacts_toggle", "_clashes_toggle"):
            toggle = getattr(self, name, None)
            if toggle is not None:
                toggles.append(toggle)
        return toggles

    def _on_toggle_all_markup(self) -> None:
        """Show everything if nothing is showing, otherwise hide everything.

        Setting each widget (rather than calling the desktop directly) means the individual
        boxes stay truthful about what is drawn, and their own handlers do the drawing.
        """
        live = [t for t in self._markup_toggles() if t.isEnabled()]
        show = not any(t.isChecked() for t in live)
        for toggle in live:
            toggle.setChecked(show)
        self._sync_all_markup_button()

    def _sync_all_markup_button(self) -> None:
        """Keep the label describing the action, and grey it out when there is no markup."""
        button = getattr(self, "_all_markup_btn", None)
        if button is None:  # pragma: no cover - during tab construction
            return
        live = [t for t in self._markup_toggles() if t.isEnabled()]
        button.setEnabled(bool(live))
        button.setText("Hide all markers" if any(t.isChecked() for t in live)
                       else "Show all markers")

    def _build_validation_section(self, mid, result, draw_markers=True):
        """One validator's sub-tab: summary, a Markers checkbox (on by default), and a
        whole-row-selectable table that selects+focuses the residue in the viewport."""
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QCheckBox,
            QLabel,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        page = QWidget()
        v = QVBoxLayout(page)
        summary = QLabel(result.summary)
        summary.setStyleSheet("color: palette(placeholder-text);")
        summary.setWordWrap(True)
        v.addWidget(summary)

        # Above the table and on by default: the markup is the point of the tab, so it
        # shows as soon as the results do. Connected before setChecked so that initial
        # state actually draws it.
        markers = QCheckBox("Markers")
        markers.setToolTip("Show/hide this validator's MolProbity markup in the viewport.")
        markers.setEnabled(bool(result.markup))
        markers.toggled.connect(
            lambda on, k=result.key: self._desktop.set_validation_markers(k, on))
        markers.toggled.connect(lambda _on: self._sync_all_markup_button())
        # On by default when the user ran validation; off when the tab is filled as a side
        # effect of a hotspot run, so the markup does not land on top of the hotspot coloring.
        markers.setChecked(draw_markers and bool(result.markup))
        self._marker_checks.append(markers)
        v.addWidget(markers)

        table = QTableWidget(len(result.rows), len(result.columns))
        table.setHorizontalHeaderLabels(result.columns)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Whole-row selection; picking a row focuses that residue in the viewport.
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for r, row in enumerate(result.rows):
            for c, value in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        table.itemSelectionChanged.connect(
            lambda t=table, res=result: self._on_validation_row_selected(t, res))
        v.addWidget(table)
        return page

    def _on_validation_row_selected(self, table, result) -> None:
        """A validation table row was selected: select + focus that residue. Rows
        carry chain/resid columns (per-residue validators); whole-model results like
        Rama-Z have neither, so there is nothing to focus."""
        cols = result.columns
        if "chain" not in cols or "resid" not in cols:
            return
        row = table.currentRow()
        if row < 0:
            return
        chain = table.item(row, cols.index("chain"))
        resid = table.item(row, cols.index("resid"))
        if chain is None or resid is None:
            return
        self._desktop.focus_residue(chain.text(), resid.text())

    def _on_run_validation(self) -> None:
        try:
            self._desktop.run_validation()
        except Exception as exc:
            self._set_status(str(exc))

    def _set_validation_stale(self, stale: bool) -> None:
        """Show or hide the 'model has moved' warning on the Validation tab. Driven by the
        desktop's ``validation_stale_changed`` signal (emitted on every move and re-run)."""
        label = getattr(self, "_stale_warning", None)
        if label is not None:
            label.setVisible(bool(stale))

    def _on_validation_ready(self, payload) -> None:
        """Validation finished (GUI thread): rebuild a sub-tab per result, keeping the
        always-present Clashes & contacts tab (index 0) in place.

        ``draw_markers`` (default True for an explicit Run validation) is False when the tab is
        being populated as a side effect of a hotspot run — the tables fill in, but the markup
        is left off so it does not pile on top of the hotspot coloring the user is looking at.
        """
        mid, results, draw_markers = (*payload, True)[:3]
        tabs = self._validation_subtabs
        current = tabs.tabText(tabs.currentIndex())  # preserve the selected validator
        while tabs.count() > 1:  # drop the previous run's validator tabs; keep Clashes (0)
            page = tabs.widget(1)
            tabs.removeTab(1)
            page.deleteLater()
        # Those checkboxes went with the pages; drop them before the new ones register, or the
        # "all" button would be reasoning about deleted widgets.
        self._marker_checks.clear()
        for result in results:
            tabs.addTab(self._build_validation_section(mid, result, draw_markers), result.title)
        for i in range(tabs.count()):  # keep the user on the same validator across re-runs
            if tabs.tabText(i) == current:
                tabs.setCurrentIndex(i)
                break
        self._sync_all_markup_button()

    def _build_hotspots_tab(self):
        """Validation hotspots: several metrics aggregated into one per-atom severity field.

        The tab is deliberately the aggregate *and* its parts — the score is only allowed to
        rank, and the per-component columns are what say what is actually wrong. See
        HOTSPOTS.md for why that separation is load-bearing rather than a nicety.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
            QSlider, QTableWidget, QVBoxLayout, QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        intro = QLabel(
            "Find hotspots scores this model itself, on the severity scale where 1.0 is the "
            "outlier cut. Open volume… imports bounded concern fields from Hotspots, where "
            "concern controls both hue and opacity on a fixed 0–1 scale. The combined field "
            "means where to look, not model quality; component fields say which validation "
            "source raised the concern. A model shows one or the other, never both — the two "
            "scales are not interchangeable.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Off by default: adding hydrogens (reduce2) and probing three times as many atoms is
        # by far the slowest thing here, and the score is still useful without it — only the
        # clash component changes. On is what MolProbity's clashscore actually means.
        hydrogens = QCheckBox("Use hydrogens for clashes (accurate, much slower)")
        hydrogens.setToolTip(
            "Clashes are mostly hydrogen-mediated, so adding them (reduce2) finds far more of "
            "them — this is the MolProbity clashscore path.\n"
            "It costs reduce2 plus a probe over three times the atoms, so it is off by default; "
            "the fast pass finds only heavy-atom overlaps.\n"
            "Shared with the Clashes & contacts tab, which always uses hydrogens.")
        hydrogens.toggled.connect(self._on_hotspot_hydrogens)
        self._hotspot_hydrogens = hydrogens
        self._desktop._hotspot_hydrogens = False
        layout.addWidget(hydrogens)

        action_row = QHBoxLayout()
        find = QPushButton("Find hotspots")
        find.setToolTip("Score the active model and color it by severity (background thread).")
        find.clicked.connect(self._on_find_hotspots)
        self._hotspot_btn = find
        self._register_busy_button(find, "Finding hotspots")
        action_row.addWidget(find)
        open_volume = QPushButton("Open volume…")
        open_volume.setToolTip(
            "Open a Hotspots JSON manifest (recommended), or one bounded-concern CCP4 map "
            "with its sibling percentile map. No validation or map-model calculation runs.")
        open_volume.clicked.connect(self._on_open_hotspot_volume)
        self._hotspot_open_btn = open_volume
        action_row.addWidget(open_volume)
        layout.addLayout(action_row)

        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Field:"))
        metric = QComboBox()
        metric.setEnabled(False)
        metric.setToolTip(
            "Choose an imported component field. Combined means where to look; component "
            "fields retain why that region is highlighted.")
        metric.currentIndexChanged.connect(self._on_hotspot_metric_changed)
        self._hotspot_metric_combo = metric
        metric_row.addWidget(metric, stretch=1)
        layout.addLayout(metric_row)

        field_row = QHBoxLayout()
        show3d = QCheckBox("Show in 3-D")
        show3d.setToolTip(
            "Draw the severity field around the model in 3-D.\n"
            "Per-atom color only shows the surface, so a buried hotspot stays hidden; a 3-D "
            "field is visible through the structure.")
        show3d.setEnabled(False)  # nothing to draw until a score exists
        show3d.toggled.connect(self._on_hotspot_field_changed)
        self._hotspot_show3d = show3d
        field_row.addWidget(show3d)

        style = QComboBox()
        style.addItem("Density", "cloud")
        style.addItem("Contour", "contour")
        style.setToolTip(
            "Density: every voxel raymarched, colored and faded by its own absolute value — "
            "transparent where clean, through yellow and orange to red.\n"
            "Contour: a translucent shell at the current threshold, in the color the density "
            "shows at that same level.\n"
            "Both read one absolute scale, so a color means the same thing in every structure.")
        style.setEnabled(False)
        style.currentIndexChanged.connect(self._on_hotspot_field_changed)
        self._hotspot_style = style
        field_row.addWidget(style, stretch=1)

        quality = QComboBox()
        for label, key in (("Low", "low"), ("Medium", "medium"), ("High", "high")):
            quality.addItem(label, key)
        quality.setCurrentIndex(0)  # low: interactive on a light laptop, the floor we target
        quality.setToolTip(
            "Cloud smoothness vs. frame rate.\n"
            "Low: fast — for interacting; some onion-shell facets.\n"
            "High: a clean diffuse render for a still figure — sacrifices interactivity.\n"
            "Bump it up on stronger hardware, or briefly to make a figure.")
        quality.setEnabled(False)
        quality.currentIndexChanged.connect(self._on_hotspot_quality)
        self._hotspot_quality_combo = quality
        field_row.addWidget(quality)
        layout.addLayout(field_row)

        # One absolute threshold controls both looks: the density's opacity knee or the
        # contour's isosurface level. The slider is an integer, so it carries the scale it is
        # currently expressing (see _set_threshold_scale) rather than a fixed 10x.
        knee_row = QHBoxLayout()
        knee_label = QLabel("Severity threshold:")
        knee_row.addWidget(knee_label)
        knee = QSlider(Qt.Orientation.Horizontal)
        knee.setRange(0, 40)
        knee.setValue(int(round(_HOTSPOT_KNEE_DEFAULT * 10)))
        knee.valueChanged.connect(self._on_hotspot_knee)
        self._hotspot_knee_slider = knee
        self._hotspot_knee_label = knee_label
        self._hotspot_knee_scale = 10.0     # slider units per 1.0 of the displayed field
        self._hotspot_knee_digits = 1
        knee_row.addWidget(knee, stretch=1)
        self._hotspot_knee_value = QLabel(f"{_HOTSPOT_KNEE_DEFAULT:.1f}")
        self._hotspot_knee_value.setMinimumWidth(36)
        knee_row.addWidget(self._hotspot_knee_value)
        self._hotspot_knee_widgets = (knee_label, knee, self._hotspot_knee_value)
        for w in self._hotspot_knee_widgets:
            w.setVisible(False)  # shown while either 3-D style is up
        layout.addLayout(knee_row)
        self._set_threshold_scale(concern=False)

        self._hotspot_summary = QLabel("Not computed yet.")
        self._hotspot_summary.setStyleSheet("color: palette(placeholder-text);")
        self._hotspot_summary.setWordWrap(True)
        layout.addWidget(self._hotspot_summary)

        table = QTableWidget(0, 0)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.itemSelectionChanged.connect(self._on_hotspot_row_selected)
        self._hotspot_table = table
        layout.addWidget(table, stretch=1)
        return tab

    def _on_find_hotspots(self) -> None:
        try:
            self._desktop.compute_hotspots()
        except Exception as exc:
            self._set_status(str(exc))

    def _set_threshold_scale(self, *, concern: bool) -> None:
        """Point the threshold slider at one field's units, and say which they are.

        Concern is bounded to [0, 1] and its interesting range is narrow, so it gets 0.01
        steps; severity runs to the display cap in 0.1 steps. The label carries the unit
        because the same widget expresses both, and a slider reading 0.5 means very different
        things on the two scales.
        """
        slider = self._hotspot_knee_slider
        was = slider.value() / self._hotspot_knee_scale
        slider.blockSignals(True)
        if concern:
            self._hotspot_knee_scale, self._hotspot_knee_digits = 100.0, 2
            self._hotspot_knee_label.setText("Concern threshold:")
            slider.setRange(0, 100)
            slider.setToolTip(
                "Density: where the field starts to become visible. Contour: the shell's "
                "level.\nAbsolute bounded concern: 0.5 is yellow, 0.75 orange, 1.0 red, in "
                "every structure and every metric.")
        else:
            self._hotspot_knee_scale, self._hotspot_knee_digits = 10.0, 1
            self._hotspot_knee_label.setText("Severity threshold:")
            slider.setRange(0, 40)
            slider.setToolTip(
                "Density: where opacity starts. Contour: the shell's absolute level.\n"
                "Raise it to keep only worse regions; 1.0 is the outlier threshold.")
        slider.setValue(int(round(was * self._hotspot_knee_scale)))
        slider.blockSignals(False)
        self._hotspot_knee_value.setText(
            f"{slider.value() / self._hotspot_knee_scale:.{self._hotspot_knee_digits}f}")

    def _on_open_hotspot_volume(self) -> None:
        """Import bounded concern fields from the Hotspots generator; nothing is computed."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        if self._desktop._active_model_id is None:
            self._set_status("load a model first")
            return
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Open concern fields", "",
            "Hotspot manifests and maps (*_hotspots.json *.json *.map *.mrc *.ccp4 "
            "*.map.gz *.mrc.gz);;Volume maps (*.map *.mrc *.ccp4 *.map.gz *.mrc.gz);;"
            "All files (*)")
        if not path:
            return
        try:
            self._desktop.open_hotspot_volume(path, style="cloud")
        except Exception as exc:
            QMessageBox.warning(self._window, "Open concern fields failed", str(exc))
            return

        entry = self._desktop._model_entry(self._desktop._active_model_id)
        imported = entry.get("concern") if entry else None
        self._hotspot_show3d.setEnabled(True)
        self._hotspot_show3d.blockSignals(True)
        self._hotspot_show3d.setChecked(True)
        self._hotspot_show3d.blockSignals(False)
        self._hotspot_style.blockSignals(True)
        self._hotspot_style.setCurrentIndex(self._hotspot_style.findData("cloud"))
        self._hotspot_style.blockSignals(False)
        self._hotspot_style.setEnabled(True)
        self._hotspot_quality_combo.setEnabled(True)
        # Match the slider to the imported contract before showing the value it now means.
        self._set_threshold_scale(concern=True)
        if imported is not None:
            self._hotspot_knee_slider.blockSignals(True)
            self._hotspot_knee_slider.setValue(
                int(round(imported.anchors["yellow"] * self._hotspot_knee_scale)))
            self._hotspot_knee_slider.blockSignals(False)
            self._hotspot_knee_value.setText(
                f"{imported.anchors['yellow']:.{self._hotspot_knee_digits}f}")
        fields = list(imported.fields) if imported is not None else []
        self._hotspot_metric_combo.blockSignals(True)
        self._hotspot_metric_combo.clear()
        for name in fields:
            self._hotspot_metric_combo.addItem(name.replace("_", " ").title(), name)
        selected = entry.get("concern_metric") if entry else None
        self._hotspot_metric_combo.setCurrentIndex(
            max(0, self._hotspot_metric_combo.findData(selected)))
        self._hotspot_metric_combo.blockSignals(False)
        self._hotspot_metric_combo.setEnabled(bool(fields))
        for widget in self._hotspot_knee_widgets:
            widget.setVisible(True)

    def _on_hotspot_field_changed(self, *_args) -> None:
        """The 3-D toggle or the cloud/contour selector changed: redraw (or clear)."""
        on = self._hotspot_show3d.isChecked()
        entry = self._desktop._model_entry(self._desktop._active_model_id)
        self._hotspot_style.setEnabled(on)
        # Quality is cloud-only; the absolute threshold controls both cloud and contour.
        is_cloud = on and self._hotspot_style.currentData() == "cloud"
        self._hotspot_quality_combo.setEnabled(is_cloud)
        for w in self._hotspot_knee_widgets:
            w.setVisible(on)
        try:
            self._desktop.show_hotspot_field(
                on=on, style=self._hotspot_style.currentData())
        except Exception as exc:
            self._set_status(str(exc))

    def _on_hotspot_metric_changed(self, *_args) -> None:
        metric = self._hotspot_metric_combo.currentData()
        if metric:
            try:
                self._desktop.set_hotspot_field_metric(str(metric))
            except Exception as exc:
                self._set_status(str(exc))

    def _on_hotspot_hydrogens(self, on: bool) -> None:
        """The hydrogens option changed. It changes what a clash *is*, so a cached score no
        longer describes the chosen setting — drop it, and say so rather than leaving a stale
        table that silently disagrees with the checkbox."""
        self._desktop.set_hotspot_hydrogens(bool(on))

    def _on_hotspot_quality(self, *_args) -> None:
        """The cloud quality preset changed: redraw at the new smoothness/speed."""
        try:
            self._desktop.set_cloud_quality(self._hotspot_quality_combo.currentData())
        except Exception as exc:
            self._set_status(str(exc))

    def _on_hotspot_knee(self, value: int) -> None:
        """The threshold slider moved. Its units are whichever field is on screen."""
        level = value / self._hotspot_knee_scale
        self._hotspot_knee_value.setText(f"{level:.{self._hotspot_knee_digits}f}")
        try:
            self._desktop.set_hotspot_threshold(None, level)
        except Exception as exc:
            self._set_status(str(exc))

    def _fill_hotspot_table(self, columns, rows) -> None:
        from PySide6.QtWidgets import QTableWidgetItem

        table = self._hotspot_table
        table.clearContents()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        self._hotspot_columns = columns

    def _on_hotspots_ready(self, payload) -> None:
        """A computed score finished (GUI thread): fill the residue table, worst first."""
        _mid, result, columns, rows = payload
        # A computed score supersedes any import, so the slider goes back to severity units.
        self._set_threshold_scale(concern=False)
        self._hotspot_metric_combo.blockSignals(True)
        self._hotspot_metric_combo.clear()
        self._hotspot_metric_combo.blockSignals(False)
        self._hotspot_metric_combo.setEnabled(False)
        self._hotspot_summary.setText(result.summary)
        self._hotspot_show3d.setEnabled(True)
        if self._hotspot_show3d.isChecked():
            # A field already up is showing the previous run's scores; redraw it from this one
            # rather than leaving a stale surface next to a fresh table.
            self._on_hotspot_field_changed()
        self._fill_hotspot_table(columns, rows)

    def _on_concern_ready(self, payload) -> None:
        """Imported concern fields were read back per residue: fill the same table.

        Every value here is bounded concern sampled from the maps on screen, so the table and
        the viewport cannot disagree. Validation markers are a separate product with their own
        controls on the Validation tab; they are not filtered by the field selected here.
        """
        _mid, summary, columns, rows = payload
        self._hotspot_summary.setText(summary)
        self._fill_hotspot_table(columns, rows)

    def _on_hotspot_row_selected(self) -> None:
        """Selecting a hotspot focuses that residue — the table is a worklist, so picking a
        row should put you in front of the thing to fix."""
        columns = getattr(self, "_hotspot_columns", None)
        table = self._hotspot_table
        row = table.currentRow()
        if not columns or row < 0:
            return
        chain = table.item(row, columns.index("chain"))
        resid = table.item(row, columns.index("resid"))
        if chain is not None and resid is not None:
            self._desktop.focus_residue(chain.text(), resid.text())

    def _build_settings_tab(self):
        """Second-class settings that don't belong in the everyday workflow."""
        from PySide6.QtWidgets import (
            QCheckBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
            QVBoxLayout, QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        viewer = QGroupBox("Viewer")
        vg = QVBoxLayout(viewer)

        # How much density a map from reflections opens with. 15 A is a starting point,
        # not a convention we can point at, so it is adjustable rather than baked in.
        # Any map's own radius is on its Appearance pane; this is only what new ones get.
        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Map radius for new maps:"))
        radius_spin = QDoubleSpinBox()
        radius_spin.setRange(1.0, 200.0)
        radius_spin.setDecimals(0)
        radius_spin.setSingleStep(5.0)
        radius_spin.setSuffix(" Å")
        radius_spin.setValue(self._desktop.view_radius_default)
        radius_spin.setToolTip(
            "How much density around the view center a map made from reflections opens "
            "with. Each map's own radius is on its Appearance pane.")
        radius_spin.valueChanged.connect(self._desktop.set_view_radius_default)
        radius_row.addWidget(radius_spin)
        radius_row.addStretch()
        vg.addLayout(radius_row)

        focus_surroundings = QCheckBox("Show Mol* focus neighborhood on click")
        focus_surroundings.setChecked(self._desktop._focus_surroundings)
        focus_surroundings.setToolTip(
            "Restore Mol*'s native click-focus display: show the focused residue and its "
            "5 Å surroundings as ball-and-stick. Interactions remain governed separately by "
            "the model's Show → Mol* interactions checkbox. This preference is saved "
            "automatically; camera focusing itself is unchanged.")
        focus_surroundings.toggled.connect(self._desktop.set_focus_surroundings)
        vg.addWidget(focus_surroundings)
        self._focus_surroundings_check = focus_surroundings
        layout.addWidget(viewer)

        defaults = QGroupBox("New model defaults")
        defaults_layout = QVBoxLayout(defaults)
        defaults_layout.addWidget(QLabel("Representations:"))
        rep_grid = QGridLayout()
        rep_checks = {}
        for i, (label, rep) in enumerate(_MODEL_REP_OPTIONS):
            check = QCheckBox(label)
            check.setChecked(rep in self._desktop._default_model_reps)
            rep_grid.addWidget(check, i // 2, i % 2)
            rep_checks[rep] = check

            def rep_changed(on, value=rep, box=check):
                if not self._desktop.set_default_model_representation(value, on):
                    box.blockSignals(True)
                    box.setChecked(True)  # at least one representation must remain
                    box.blockSignals(False)

            check.toggled.connect(rep_changed)
        defaults_layout.addLayout(rep_grid)
        self._default_rep_checks = rep_checks

        defaults_layout.addWidget(QLabel("Show:"))
        show_grid = QGridLayout()
        show_checks = {}
        for i, label in enumerate(_STRUCTURE_TYPE_ORDER + ["Mol* interactions"]):
            check = QCheckBox(label)
            shown = (self._desktop._default_model_interactions
                     if label == "Mol* interactions"
                     else label in self._desktop._default_shown_types)
            check.setChecked(shown)
            check.toggled.connect(
                lambda on, value=label: self._desktop.set_default_model_show(value, on))
            show_grid.addWidget(check, i // 2, i % 2)
            show_checks[label] = check
        defaults_layout.addLayout(show_grid)
        self._default_show_checks = show_checks
        layout.addWidget(defaults)

        layout.addWidget(self._build_drag_group())
        layout.addWidget(self._build_perf_group())

        layout.addStretch()
        return tab

    def _build_perf_group(self):
        """Performance-debugging controls: a live overlay, render overrides to isolate what
        costs, and a capture that saves a per-frame log of a drag."""
        from PySide6.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

        box = QGroupBox("Performance (debug)")
        pg = QVBoxLayout(box)

        hint = QLabel("Tools for chasing lag. The overlay shows live frame timings; the "
                      "toggles isolate one cost at a time; Capture saves a drag's log.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        pg.addWidget(hint)

        overlay = QCheckBox("Show performance overlay  (or press F9 in the viewer)")
        overlay.setToolTip("Live HUD: incoming frame rate, draw rate, state-commit time, and "
                           "frontend latency — so you can see which stage is the bottleneck.")
        overlay.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_perf_prefs(overlay=bool(on))))
        pg.addWidget(overlay)

        ssao = QCheckBox("Disable ambient occlusion")
        ssao.setToolTip("Pin SSAO off (normally it drops only while moving). It costs ~12 ms "
                        "of a frame at Retina resolution; turn it off to feel its share.")
        ssao.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_perf_prefs(occlusionOff=bool(on))))
        pg.addWidget(ssao)

        halfres = QCheckBox("Render at half resolution")
        halfres.setToolTip("Halve the render resolution (softer image). If this makes "
                           "dragging smooth, the bottleneck is GPU fill-rate, not geometry.")
        halfres.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_perf_prefs(pixelScale=0.5 if on else 1.0)))
        pg.addWidget(halfres)

        cap_row = QHBoxLayout()
        self._perf_capture_btn = QPushButton("Start capture")
        self._perf_capture_btn.setToolTip("Record per-frame drag timings, then save them to a "
                                          "JSON file you can inspect or share.")
        self._perf_capturing = False
        self._perf_capture_btn.clicked.connect(self._on_perf_capture)
        cap_row.addWidget(self._perf_capture_btn)
        cap_row.addStretch()
        pg.addLayout(cap_row)

        return box

    def _on_perf_capture(self) -> None:
        if not self._perf_capturing:
            self._perf_capturing = True
            self._perf_capture_btn.setText("Stop capture && save")
            self._desktop.start_perf_capture()
        else:
            self._perf_capturing = False
            self._perf_capture_btn.setText("Start capture")
            self._desktop.stop_perf_capture(on_saved=_reveal_in_file_manager)

    # -- appearance (focused object) -------------------------------------

    def _find_item(self, kind, ident):
        return next((it for it in self._items if it["kind"] == kind and it["id"] == ident), None)

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                # Hide *before* un-parenting: setParent(None) on a still-visible widget
                # turns it into a floating top-level window, which is how a rebuilt
                # Appearance pane spawned stray little combo-box windows. Un-parenting
                # then still removes it from the tree at once (so a rebuild does not see
                # the old widgets), and deleteLater frees it.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
            elif child.layout() is not None:
                self._clear_layout(child.layout())

    def _appearance_signature(self, kind, ident, it):
        """Everything the Appearance pane draws, as a comparable snapshot.

        Deliberately excludes ``visible``. Hiding an object changes nothing in this pane — a
        hidden object is still perfectly editable here — so a visibility toggle must not tear
        the widgets down and rebuild them. That rebuild is what made the pane flicker on every
        show/hide, because ``_on_loaded_changed`` runs on *any* change to the object list.

        Live values are folded in as well as the summary snapshot: a volume's level and color
        can move without a new summary (the wheel, the console), and the pane reads those
        directly, so they have to count as a change.
        """
        if it is None:
            return (kind, ident, None, None)
        live = {}
        extra = None
        if it["kind"] == "model":
            live = self._desktop.model_appearance(it["id"])
        elif it["kind"] == "volume":
            live = self._desktop.volume_appearance(it["id"])
            # Whether a mask is offered depends on the map's pairing, not on the map itself.
            extra = self._desktop.can_mask_volume(it["id"])
        elif it["kind"] == "reflections":
            # The "Make maps" row lists the *other* objects (unpaired models), so it can go
            # stale even when these reflections have not changed at all.
            extra = tuple((m["id"], m["name"]) for m in self._desktop.models_for_phasing())
        merged = {**it, **live}
        merged.pop("visible", None)
        return (kind, ident, tuple(sorted((k, repr(v)) for k, v in merged.items())), repr(extra))

    def _update_appearance(self, kind=None, ident=None, force=False):
        """Rebuild the Appearance box for the focused object (or an empty-state hint).

        Skipped when nothing it displays has changed (see ``_appearance_signature``), so the
        pane survives unrelated object-list churn instead of flickering. ``force`` rebuilds
        regardless, for callers that changed something this snapshot cannot see.
        """
        from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton

        it = self._find_item(kind, ident) if ident else None
        signature = self._appearance_signature(kind, ident, it)
        if not force and signature == self._appearance_sig:
            return  # identical pane; rebuilding it would only flicker
        self._appearance_sig = signature

        self._clear_layout(self._appearance_layout)
        self._focused = (kind, ident) if it else (None, None)
        self._iso_row = None  # rebuilt below only when a volume is focused
        if it is None:
            hint = QLabel("Select an object above to edit how it looks.")
            hint.setStyleSheet("color: palette(placeholder-text);")
            self._appearance_layout.addWidget(hint)
            self._safe(lambda: self._desktop.set_volume_scroll_target(None))
            return

        self._appearance_box.setTitle(f"Appearance · {it['name']}")

        def add_combo(label, options, current, on_pick):
            r = QHBoxLayout()
            lab = QLabel(label)
            lab.setMinimumWidth(80)
            r.addWidget(lab)
            combo = QComboBox()
            # Let the combo shrink and elide instead of forcing a wide panel from a
            # long item like "By secondary structure".
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(6)
            for text, value in options:
                combo.addItem(text, value)
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(lambda _i, c=combo: on_pick(c.currentData()))
            r.addWidget(combo, stretch=1)
            self._appearance_layout.addLayout(r)
            return combo

        if it["kind"] == "marker":
            # Not styled — a marker is a 3D point. The pane just says what it is: its
            # coordinate and what it snapped to. Building a ligand at it lives in the Tools
            # tab's permanent Ligand placement panel, not here.
            pos = it.get("position") or [0.0, 0.0, 0.0]
            coord = QLabel(f"x  {pos[0]:.3f}\ny  {pos[1]:.3f}\nz  {pos[2]:.3f}    Å")
            coord.setStyleSheet("font-family: monospace;")
            self._appearance_layout.addWidget(coord)
            snapped = ("on atom " + str(it["atom"])) if it.get("atom") is not None \
                else "in the view plane (empty space)"
            note = QLabel("Placed " + snapped
                          + ".\nBuild a ligand here from Tools → Ligand placement.")
            note.setWordWrap(True)
            note.setStyleSheet("color: palette(placeholder-text);")
            self._appearance_layout.addWidget(note)
            self._safe(lambda: self._desktop.set_volume_scroll_target(None))
            return
        if it["kind"] == "reflections":
            # Nothing to style — reflections are not drawn. The pane is still where an
            # object says what it is, so it says what the file holds and what that
            # means for getting density out of it.
            summary = QLabel(it.get("summary", ""))
            summary.setWordWrap(True)
            self._appearance_layout.addWidget(summary)
            arrays = QLabel("Arrays: " + ", ".join(it.get("labels") or []))
            arrays.setWordWrap(True)
            arrays.setStyleSheet("color: palette(placeholder-text);")
            self._appearance_layout.addWidget(arrays)
            note = QLabel(
                "Carries map coefficients — density needs no model."
                if it.get("has_map_coefficients")
                else "Amplitudes only — density needs a model to phase against.")
            note.setWordWrap(True)
            note.setStyleSheet("color: palette(placeholder-text);")
            self._appearance_layout.addWidget(note)
            if it.get("r_work") is not None:
                fit = QLabel(f"R-work {it['r_work']:.4f} · R-free {it['r_free']:.4f}")
                self._appearance_layout.addWidget(fit)
            if not it.get("has_map_coefficients"):
                if it.get("r_work") is None:
                    self._add_phasing_row(it["id"])
                else:
                    # Already phased: the useful action is now recomputing, for when the
                    # model has moved by some route that does not do it for you.
                    update = QPushButton("Update maps")
                    update.setToolTip(
                        "Recompute the density from the model as it now stands. "
                        "Minimize does this for you.")
                    update.clicked.connect(
                        lambda _c=False, r=it["id"]:
                        self._safe(lambda: self._desktop.update_maps(r)))
                    self._appearance_layout.addWidget(update)
        elif it["kind"] == "model":
            mid = it["id"]

            def _set_rep(v, it=it):
                it["rep"] = v  # keep this snapshot in step with the backend entry
                self._safe(lambda: self._desktop.set_model_representation(mid, v))

            def _set_color(v, it=it):
                it["color"] = v
                self._safe(lambda: self._desktop.set_model_color(mid, v))

            add_combo("Representation", _MODEL_REP_OPTIONS, it.get("rep"), _set_rep)

            # Only for models that actually have alternate conformations, which is very
            # few of them: offering "Conformer: All" on every ordinary structure would be
            # a control that never does anything.
            conformers = it.get("conformers") or []
            if conformers:
                def _set_conformer(v, it=it):
                    it["conformer"] = v
                    self._safe(lambda: self._desktop.set_model_conformer(mid, v))

                add_combo("Conformer",
                          [("All", None)] + [(label, label) for label in conformers],
                          it.get("conformer"), _set_conformer)
            # Color-by themes *and* flat colors: a model can be colored by a property or
            # just set to one color, and the swatches/wheel are the only way to say the latter.
            self._add_color_row(it.get("color"), _set_color,
                                themes=_MODEL_COLOR_OPTIONS, title="Model color")
            types = it.get("types") or []
            r = QHBoxLayout()
            lab = QLabel("Show")
            lab.setMinimumWidth(80)
            r.addWidget(lab)
            r.addWidget(self._make_type_combo(
                mid, types, set(it.get("hidden_types") or []),
                interactions=bool(it.get("interactions"))), stretch=1)
            self._appearance_layout.addLayout(r)

            def _set_clip(front, back, it=it):
                it["clip"] = (front, back)
                self._safe(lambda: self._desktop.set_model_clip(mid, front, back))

            self._add_clip_row(
                {**it, **self._desktop.model_appearance(mid)}.get("clip"), _set_clip)

            if it.get("has_restraints_cif"):
                # A ligand built here — export the pair a refinement needs: its fitted
                # coordinates and its restraints dictionary, in one action.
                export = QPushButton("Export ligand (coordinates + restraints)…")
                export.setToolTip(
                    "Write the two files a refinement needs together: the fitted coordinates "
                    "(mmCIF or PDB) and, saved alongside as <name>_restraints.cif, the "
                    "restraints dictionary — a geostd-style monomer CIF with the SMILES and "
                    "rdkit-version provenance that built it.")
                export.clicked.connect(
                    lambda _=False, d=mid, nm=it["name"]: self._on_export_ligand(d, nm))
                self._appearance_layout.addWidget(export)
        else:  # volume
            vid = it["id"]
            # Read the live values, not this snapshot: the level in particular can have
            # moved since (the scroll wheel, or the console) without a new summary.
            live = {**it, **self._desktop.volume_appearance(vid)}


            def _set_style(v, it=it):
                it["style"] = v
                self._safe(lambda: self._desktop.set_volume_style(vid, v))

            def _set_color(v, it=it):
                # "Local resolution" rides the same dropdown as the flat colours: both
                # answer "what colours this map", so two controls for one question (a
                # colour row AND a separate checkbox) made each look unrelated to the
                # other. Picking it turns the colouring on; picking any colour returns
                # to a flat surface in that colour.
                if v == "localres":
                    it["color_by_resolution"] = True
                    self._safe(lambda: self._desktop.set_color_by_resolution(vid, True))
                    return
                if it.get("color_by_resolution"):
                    it["color_by_resolution"] = False
                    self._safe(lambda: self._desktop.set_color_by_resolution(vid, False))
                it["color"] = v
                self._safe(lambda: self._desktop.set_volume_color(vid, v))

            add_combo("Style", _VOLUME_STYLE_OPTIONS, live.get("style"), _set_style)
            # Downsample sits with the map's own display controls, not inside the
            # colouring group: it is a property of how this map is drawn. Today its one
            # consumer is the colour-by-resolution surface (the plain isosurface is
            # contoured by the viewer from the full grid), which is why it only appears
            # once a resolution map is pinned; a future progressive-rendering mode for
            # plain maps would claim the same row.
            if it.get("resolution_map"):
                def _set_localres_ds(v, it=it):
                    it["localres_downsample"] = v
                    self._safe(lambda: self._desktop.set_localres_downsample(vid, v))

                add_combo("Downsample",
                          [("Full", 1), ("2×", 2), ("4×", 4), ("8×", 8)],
                          int(it.get("localres_downsample") or 4), _set_localres_ds)
            self._add_color_row(
                "localres" if it.get("color_by_resolution") else live.get("color"),
                _set_color,
                themes=([("Local resolution", "localres")]
                        if it.get("resolution_map") else None),
                title="Map color")

            def _set_opacity(v, it=it):
                it["opacity"] = v
                self._safe(lambda: self._desktop.set_volume_opacity(vid, v))

            self._add_opacity_row(live.get("opacity"), _set_opacity)

            def _set_iso(v, it=it):
                it["iso"] = v
                self._safe(lambda: self._desktop.set_volume_iso(vid, v))

            self._iso_row = self._add_iso_row(live.get("iso"), _set_iso,
                                              max_sigma=live.get("max_sigma"))

            def _set_clip(front, back, it=it):
                it["clip"] = (front, back)
                self._safe(lambda: self._desktop.set_volume_clip(vid, front, back))

            self._add_clip_row(live.get("clip"), _set_clip)

            def _set_radius(radius, it=it):
                it["radius"] = radius
                self._safe(lambda: self._desktop.set_volume_radius(vid, radius))

            self._add_radius_row(live.get("radius"), _set_radius)

            def _set_mask(radius, it=it):
                it["mask_radius"] = radius
                self._safe(lambda: self._desktop.set_volume_mask(vid, radius))

            self._add_mask_row(live.get("mask_radius"),
                               self._desktop.can_mask_volume(vid), _set_mask)

            # The colouring's own settings, shown only while "Local resolution" is the
            # selected colouring (the Color dropdown is the switch; this panel is its
            # detail). A plain framed group -- no checkbox: selection lives above.
            if it.get("color_by_resolution"):
                from PySide6.QtWidgets import QDoubleSpinBox, QGroupBox, QVBoxLayout

                group = QGroupBox("Local resolution")
                gl = QVBoxLayout(group)
                gl.setSpacing(6)

                # The ramp's value range: lo draws blue, hi red, fixed until changed --
                # colours keep their meaning across contour levels, sessions and figures.
                lo0, hi0 = it.get("localres_domain") or (0.0, 1.0)
                rr = QHBoxLayout()
                r_lab = QLabel("Range")
                r_lab.setMinimumWidth(80)
                r_lab.setToolTip(
                    "The resolution values the colour ramp spans; values outside clamp. "
                    "Fixed until you change it.")
                rr.addWidget(r_lab)
                lo_spin, hi_spin = QDoubleSpinBox(), QDoubleSpinBox()
                lo_spin.setObjectName("localres-domain-lo")
                hi_spin.setObjectName("localres-domain-hi")
                for sp in (lo_spin, hi_spin):
                    sp.setRange(0.0, 999.0)
                    sp.setDecimals(2)
                    sp.setSingleStep(0.1)
                    sp.setSuffix(" Å")
                lo_spin.setValue(float(lo0))
                hi_spin.setValue(float(hi0))

                def _apply_domain(_=None, it=it):
                    lo_v, hi_v = lo_spin.value(), hi_spin.value()
                    if hi_v <= lo_v:
                        return  # half-edited state; applied once the other end moves
                    it["localres_domain"] = (lo_v, hi_v)
                    self._safe(lambda: self._desktop.set_localres_domain(vid, lo_v, hi_v))

                lo_spin.valueChanged.connect(_apply_domain)
                hi_spin.valueChanged.connect(_apply_domain)
                rr.addWidget(lo_spin)
                rr.addWidget(QLabel("–"))
                rr.addWidget(hi_spin)
                fit = QPushButton("Fit to surface")
                fit.setToolTip("Span the ramp over the resolution values inside the "
                               "current contour — what is actually on screen.")
                fit.clicked.connect(
                    lambda _=False: self._safe(lambda: self._desktop.fit_localres_domain(vid)))
                reset = QPushButton("Reset")
                reset.setToolTip("Back to the default: percentiles of the whole map.")
                reset.clicked.connect(
                    lambda _=False: self._safe(lambda: self._desktop.reset_localres_domain(vid)))
                rr.addWidget(fit)
                rr.addWidget(reset)
                rr.addStretch(1)
                gl.addLayout(rr)
                self._appearance_layout.addWidget(group)

        # The wheel contours whatever the Level slider above is showing, so the
        # target follows the focused object (and is cleared when it is not a volume).
        self._safe(lambda: self._desktop.set_volume_scroll_target(
            it["id"] if it["kind"] == "volume" else None))

    def _on_volume_iso_changed(self, payload) -> None:
        """A contour level was changed in the viewport (the wheel): show it here.

        The viewer already applied it, so the widgets are moved with their signals
        suppressed — writing it back would round-trip the user's own scroll.
        """
        vid, value = payload
        if self._iso_row is None or self._focused != ("volume", vid):
            return
        item = self._find_item("volume", vid)
        if item is not None and not item.get("visible", True):
            return  # a hidden map is parked at an empty contour; ignore stray wheel echoes
        row = self._iso_row
        row["syncing"]["on"] = True
        try:
            row["slider"].setValue(
                min(row["slider"].maximum(), int(round(value / _ISO_RESOLUTION))))
            row["spin"].setValue(value)
        finally:
            row["syncing"]["on"] = False
        item = self._find_item("volume", vid)
        if item is not None:
            item["iso"] = value

    def _add_phasing_row(self, rid: str) -> None:
        """Offer to compute density from these reflections and a model.

        The model is chosen rather than assumed: it is where the phases come from, so it
        decides what the density says. Only unpaired models are on offer — the maps end
        up in a manager with whichever one phased them.
        """
        from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton

        models = self._desktop.models_for_phasing()
        row = QHBoxLayout()
        button = QPushButton("Make maps")
        combo = QComboBox()
        for m in models:
            combo.addItem(m["name"], m["id"])
        if not models:
            combo.addItem("no unpaired model loaded", None)
            combo.setEnabled(False)
            button.setEnabled(False)
            button.setToolTip("Load a model to phase against, or unpair one.")
        else:
            button.setToolTip(
                "Compute 2mFo-DFc and mFo-DFc from these amplitudes and the chosen "
                "model, and pair the maps with it.")
        button.clicked.connect(
            lambda: self._safe(lambda: self._desktop.make_maps(rid, combo.currentData())))
        row.addWidget(button)
        row.addWidget(combo, stretch=1)
        self._appearance_layout.addLayout(row)

    def _add_radius_row(self, current, on_change):
        """How much density to draw around the view center.

        The map is untouched — this only stops it being drawn everywhere at once, which
        is what Coot's map radius is for. It follows the view, so it is closer to
        clipping than to the mask above it.
        """
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel

        row = QHBoxLayout()
        lab = QLabel("Radius")
        lab.setMinimumWidth(80)
        row.addWidget(lab)
        check = QCheckBox("within")
        check.setToolTip("Draw only the density near the middle of the view.")
        check.setChecked(current is not None)
        spin = QDoubleSpinBox()
        spin.setRange(1.0, 200.0)
        spin.setDecimals(0)
        spin.setSingleStep(5.0)
        spin.setSuffix(" Å")
        spin.setValue(_VIEW_RADIUS_DEFAULT if current is None else float(current))
        spin.setEnabled(current is not None)

        def toggled(on):
            spin.setEnabled(on)
            on_change(spin.value() if on else None)

        check.toggled.connect(toggled)
        spin.valueChanged.connect(
            lambda v: on_change(v) if check.isChecked() else None)
        row.addWidget(check)
        row.addWidget(spin)
        row.addStretch()
        self._appearance_layout.addLayout(row)
        return {"check": check, "spin": spin}

    def _add_mask_row(self, current, enabled, on_change):
        """Hide density away from the model: a switch and the distance.

        Only offered for a paired map — "away from the molecule" needs a molecule, and
        the pairing is what says which one. Applying it rewrites the map the browser
        fetches, so unlike the contour this is a set-and-apply control, not a drag.
        """
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel

        row = QHBoxLayout()
        lab = QLabel("Mask")
        lab.setMinimumWidth(80)
        row.addWidget(lab)
        check = QCheckBox("within")
        check.setChecked(current is not None)
        spin = QDoubleSpinBox()
        spin.setRange(0.5, 50.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setSuffix(" Å")
        spin.setValue(_MASK_RADIUS_DEFAULT if current is None else float(current))
        spin.setEnabled(current is not None)
        for widget in (check, spin):
            widget.setEnabled(widget.isEnabled() and enabled)
        check.setEnabled(enabled)
        if not enabled:
            check.setToolTip("Pair this map with a model to mask around it.")
        else:
            check.setToolTip("Hide density further than this from the model.")

        def toggled(on):
            spin.setEnabled(on)
            on_change(spin.value() if on else None)

        check.toggled.connect(toggled)
        spin.editingFinished.connect(
            lambda: on_change(spin.value()) if check.isChecked() else None)
        row.addWidget(check)
        row.addWidget(spin)
        row.addStretch()
        self._appearance_layout.addLayout(row)
        return {"check": check, "spin": spin}

    def _add_color_row(self, current, on_pick, *, themes=None, title="Color"):
        """A color control: optional color-by themes, then swatches, then a picker.

        Colors are shown rather than named — a swatch says what a hex is and the word does
        not. The picker is the escape hatch, since the wire takes any hex Mol* can decode,
        not just the presets.

        ``themes`` are leading (label, value) entries for color-*by* schemes (by element, by
        chain, …). A model gets them; a volume, whose density has nothing to color by, does
        not — but both get the swatches and the wheel, so either can be set to a flat color.
        """
        from PySide6.QtCore import QSize, Qt
        from PySide6.QtGui import QColor, QIcon, QPixmap
        from PySide6.QtWidgets import QColorDialog, QComboBox, QHBoxLayout, QLabel

        def swatch(name):
            pixmap = QPixmap(28, 14)
            pixmap.fill(QColor(name))
            return QIcon(pixmap)

        themes = list(themes or [])
        theme_values = {value for _label, value in themes}

        row = QHBoxLayout()
        lab = QLabel("Color")
        lab.setMinimumWidth(80)
        row.addWidget(lab)
        combo = QComboBox()
        combo.setIconSize(QSize(28, 14))
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
        for label, value in themes:
            combo.addItem(label, value)
        if themes:
            combo.insertSeparator(combo.count())  # themes above, flat colors below
        for name in _VOLUME_COLORS:
            # The swatch is what identifies it; the text is just the hex, kept as written
            # (capitalize() would lower-case the digits of a color like #FF2D95).
            combo.addItem(swatch(name), name.upper(), name)
        if current and current not in _VOLUME_COLORS and current not in theme_values:
            combo.addItem(swatch(current), current, current)  # a picked color, kept selectable
        combo.addItem("Custom…", _CUSTOM_COLOR)
        idx = combo.findData(current)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

        # The last color actually committed — the target to revert to if a custom pick
        # is cancelled, since by then the live preview has already changed the map.
        committed = {"value": current}

        def picked(_index, combo=combo):
            value = combo.currentData()
            if value != _CUSTOM_COLOR:
                committed["value"] = value
                on_pick(value)
                return
            revert_to = committed["value"]
            # Seed the wheel with the current color — but a theme name ("by chain") is not
            # one, so fall back to a swatch rather than opening on an invalid color.
            seed = revert_to if (isinstance(revert_to, str) and revert_to not in theme_values
                                 and QColor(revert_to).isValid()) else _VOLUME_COLORS[0]
            dialog = QColorDialog(QColor(seed), self._window)
            dialog.setWindowTitle(title)
            # Qt's own dialog, not the native one: the macOS color panel is a shared
            # singleton that emits its live-color signal unreliably, and this preview
            # depends on that signal firing every time.
            dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
            # Apply as the wheel moves, not only on OK — otherwise it looks broken until
            # you close the dialog, which is exactly when you have given up on it.
            dialog.currentColorChanged.connect(
                lambda c: on_pick(c.name()) if c.isValid() else None)
            # Re-index the combo with its signals blocked, then apply once by hand:
            # inserting an item shifts the still-selected "Custom…" entry, which would
            # otherwise re-fire this handler with _CUSTOM_COLOR and reopen the dialog the
            # instant it closed.
            combo.blockSignals(True)
            if dialog.exec() == QColorDialog.DialogCode.Accepted:
                name = dialog.selectedColor().name()  # '#rrggbb', which Mol* decodes
                committed["value"] = name
                at = combo.count() - 1
                combo.insertItem(at, swatch(name), name, name)
                combo.setCurrentIndex(at)
                applied = name
            else:
                # Cancelled: undo the preview and put the selection back where it was.
                back = combo.findData(revert_to)
                combo.setCurrentIndex(back if back >= 0 else 0)
                applied = revert_to
            combo.blockSignals(False)
            on_pick(applied)

        combo.currentIndexChanged.connect(picked)
        row.addWidget(combo, stretch=1)
        self._appearance_layout.addLayout(row)
        return combo

    def _add_clip_row(self, current, on_change):
        """The front/rear clipping slab: one track, two handles.

        Per object, not per scene — cutting the density open while the model inside it
        stays whole is the whole point, and a camera-wide slab cannot do that. Bring the
        handles together and the object is clipped away entirely.
        """
        from PySide6.QtWidgets import QHBoxLayout, QLabel

        front, back = current if current else (0.0, 1.0)
        row = QHBoxLayout()
        lab = QLabel("Clipping")
        lab.setMinimumWidth(80)
        row.addWidget(lab)
        slider = _make_range_slider()()
        slider.setToolTip(
            "Front and rear clipping planes for this object. Drag the handles to slice "
            "into it, or the span between them to move the slab. The slab follows the "
            "camera.")
        slider.set_values(front, back)
        slider.changed.connect(on_change)
        row.addWidget(slider, stretch=1)
        self._appearance_layout.addLayout(row)
        return slider

    def _add_opacity_row(self, current, on_change):
        """Opacity as a slider with its value beside it (QSlider is integer-only)."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider

        value = 1.0 if current is None else float(current)
        row = QHBoxLayout()
        lab = QLabel("Opacity")
        lab.setMinimumWidth(80)
        row.addWidget(lab)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(round(value * 100)))
        readout = QLabel(f"{value:.2f}")
        readout.setMinimumWidth(34)

        def moved(v):
            readout.setText(f"{v / 100:.2f}")
            on_change(v / 100)

        slider.valueChanged.connect(moved)
        row.addWidget(slider, stretch=1)
        row.addWidget(readout)
        self._appearance_layout.addLayout(row)
        return slider

    def _build_mouse_legend(self):
        """A compact reference of the viewport's mouse and key bindings (Coot's layout).

        The gesture on the left, what it does on the right — the same key-cap chips the
        sliders use, so "scroll" beside the Level slider reads as the same thing here.
        """
        from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel

        box = QGroupBox("Mouse")
        grid = QGridLayout(box)
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)
        bindings = [
            ("drag", "Rotate"),
            ("Ctrl + drag", "Pan"),
            ("right-drag", "Zoom"),
            ("Ctrl + scroll", "Zoom"),
            ("scroll", "Contour level"),
            ("Refine drag mode", "Pull and minimize an atom"),
        ]
        for r, (gesture, action) in enumerate(bindings):
            chip = self._gesture_chip(gesture)
            chip.setToolTip("")  # the action label beside it already says what it does
            grid.addWidget(chip, r, 0)
            grid.addWidget(QLabel(action), r, 1)
        return box

    def _gesture_chip(self, text: str):
        """A small key-cap-style badge naming a mouse gesture, for placing beside the
        control it drives — so a slider says how to reach it without opening a manual."""
        from PySide6.QtWidgets import QLabel

        chip = QLabel(text)
        chip.setToolTip("Do this over the viewport to change the control on its right.")
        chip.setStyleSheet(
            "QLabel { border: 1px solid palette(mid); border-radius: 4px;"
            " padding: 1px 5px; color: palette(dark); background: palette(midlight); }")
        return chip

    def _add_iso_row(self, current, on_change, max_sigma=None):
        """Contour level: a slider to hunt with, a spinbox for the exact value.

        Both are wanted. The slider is how you actually find a level — you watch the map,
        not the number — and updates are live, so dragging is the point. The spinbox
        makes a level reproducible ("contour at 1.5 sigma") and reaches past the slider's
        range for maps that need it.

        The slider spans this map's own range: its right end sits just above the map's
        maximum (``max_sigma``), so sliding fully right always empties the map. A fixed
        ceiling could not do that — cryo-EM maps carry long tails (EMD-53478 tops out
        near 28 sigma against the old fixed 10), so full-right left most of the density
        standing, with no way to clear it from the slider.
        """
        import math

        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QSlider

        from .volume_io import DEFAULT_ISO_SIGMA

        ceiling = _ISO_SLIDER_MAX
        if max_sigma is not None and math.isfinite(float(max_sigma)):
            # +0.5 so the top position is strictly above the hottest voxel; floored so a
            # low-contrast map keeps a usable range rather than a hypersensitive one.
            ceiling = max(6.0, math.ceil(float(max_sigma) + 0.5))

        value = DEFAULT_ISO_SIGMA if current is None else float(current)
        row = QHBoxLayout()
        row.addWidget(QLabel("Level"))
        # (The scroll wheel adjusts this — said once in the gesture legend at the bottom
        # and in the spin box's tooltip, rather than a chip that eats space on every map.)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, int(round(ceiling / _ISO_RESOLUTION)))
        slider.setValue(min(slider.maximum(), int(round(value / _ISO_RESOLUTION))))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, _ISO_SPIN_MAX)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setSuffix(" σ")
        spin.setValue(value)
        spin.setToolTip(
            "Contour level in sigma. The scroll wheel over the viewport steps it too.")

        # The two drive each other, so guard against the echo coming back.
        syncing = {"on": False}

        def apply(v):
            on_change(v)

        def from_slider(step):
            if syncing["on"]:
                return
            syncing["on"] = True
            try:
                spin.setValue(step * _ISO_RESOLUTION)
            finally:
                syncing["on"] = False
            apply(step * _ISO_RESOLUTION)

        def from_spin(v):
            if syncing["on"]:
                return
            syncing["on"] = True
            try:
                slider.setValue(min(slider.maximum(), int(round(v / _ISO_RESOLUTION))))
            finally:
                syncing["on"] = False
            apply(v)

        slider.valueChanged.connect(from_slider)
        spin.valueChanged.connect(from_spin)
        row.addWidget(slider, stretch=1)
        row.addWidget(spin)
        self._appearance_layout.addLayout(row)
        return {"slider": slider, "spin": spin, "syncing": syncing}

    def _safe(self, fn):
        try:
            fn()
        except Exception as exc:  # pragma: no cover - defensive
            self._set_status(str(exc))

    def _build_geometry_tab(self):
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget)

        from .geometry import CATEGORIES

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        # Geometry state (restraints follow the same model as the atoms table).
        self._restraint_tabs: dict = {}   # category -> {stack, msg, view, model, columns}
        self._geo_cache: dict = {}        # model_id -> GeometryRestraints
        self._restraints_model_id = None  # model the restraint tables currently show
        self._suppress_restraint_sync = False

        # Shared across every Geometry table: collapse each to the current selection
        # (atoms -> selected atoms; each restraint -> restraints within the selection).
        self._filter_selection_check = QCheckBox("Show only the selection")
        self._filter_selection_check.setToolTip(
            "Collapse every Geometry table to the current selection: the Atoms table "
            "to the selected atoms, and each restraint table to the restraints whose "
            "atoms are all selected."
        )
        self._filter_selection_check.toggled.connect(self._on_filter_toggled)

        # Restraints from one origin only. The reason this is worth a control: a model's
        # restraints are overwhelmingly monomer-library covalent geometry, so the handful
        # the user added themselves through an edits PHIL -- a metal coordination bond, a
        # covalent link the library does not know -- are otherwise a few rows lost among
        # thousands, with nothing in the table to distinguish them.
        self._origin_filter = QComboBox()
        self._origin_filter.setToolTip(
            "Show only restraints from one origin. 'edits' are the restraints supplied "
            "in a geometry_restraints.edits PHIL file; the rest come from the monomer "
            "library or from links cctbx detected.")
        self._origin_filter.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._origin_filter.setMinimumContentsLength(12)
        self._origin_filter.currentIndexChanged.connect(self._on_origin_filter_changed)

        filter_row = QHBoxLayout()
        filter_row.addWidget(self._filter_selection_check)
        filter_row.addStretch(1)
        origin_label = QLabel("Origin:")
        filter_row.addWidget(origin_label)
        filter_row.addWidget(self._origin_filter, stretch=1)
        layout.addLayout(filter_row)

        subtabs = QTabWidget()
        self._geo_subtabs = subtabs
        subtabs.addTab(self._build_atoms_subtab(), "Atoms")
        self._restraint_subtab_start = subtabs.count()
        for key, label, columns in CATEGORIES:
            subtabs.addTab(self._build_restraint_subtab(key, columns), label)
        subtabs.currentChanged.connect(self._on_geometry_subtab_changed)
        layout.addWidget(subtabs)
        return tab

    def _build_restraint_subtab(self, category: str, columns):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QLabel,
            QStackedWidget,
            QTableView,
            QVBoxLayout,
            QWidget,
        )

        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        stack = QStackedWidget()
        msg = QLabel("Open this tab to build geometry restraints.")
        msg.setWordWrap(True)
        msg.setContentsMargins(12, 12, 12, 12)
        msg.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        view = QTableView()
        model = _make_restraint_table_model()
        view.setModel(model)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        view.verticalHeader().setVisible(False)
        view.setAlternatingRowColors(True)
        view.setWordWrap(False)
        view.horizontalHeader().setStretchLastSection(True)
        view.selectionModel().selectionChanged.connect(
            lambda *_, c=category: self._on_restraint_selection(c)
        )
        # Clicking the geostd column reveals that monomer's file in the file browser.
        view.clicked.connect(lambda idx, c=category: self._on_restraint_link_clicked(c, idx))

        stack.addWidget(msg)   # page 0
        stack.addWidget(view)  # page 1
        outer.addWidget(stack)

        self._restraint_tabs[category] = {
            "stack": stack, "msg": msg, "view": view, "model": model, "columns": columns,
        }
        return tab

    def _on_geometry_subtab_changed(self, index: int) -> None:
        if index >= self._restraint_subtab_start:  # a restraint tab
            self._ensure_restraints()

    def _viewing_restraint_tab(self) -> bool:
        return self._geo_subtabs.currentIndex() >= self._restraint_subtab_start

    def _show_restraint_message(self, text: str) -> None:
        self._suppress_restraint_sync = True
        try:
            for info in self._restraint_tabs.values():
                info["msg"].setText(text)
                info["model"].set_source(None, "", info["columns"], None)
                info["stack"].setCurrentWidget(info["msg"])
        finally:
            self._suppress_restraint_sync = False

    def _invalidate_restraints(self) -> None:
        """The geometry model changed; rebuild on next view (now, if one is open)."""
        self._restraints_model_id = None
        if self._viewing_restraint_tab():
            self._ensure_restraints()

    def _ensure_restraints(self) -> None:
        """Build restraints for the current geometry model and fill the tables."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        from . import geometry as geo_mod

        mid = self._table_model_id
        if mid is not None and self._restraints_model_id == mid:
            return  # already showing this model's restraints

        session = self._desktop.session_for(mid)
        if session is None or getattr(session, "model", None) is None:
            self._show_restraint_message("Load a model to see its geometry restraints.")
            self._restraints_model_id = None
            return
        if not geo_mod.monomer_library_available():
            self._show_restraint_message(geo_mod.MONOMER_LIBRARY_HELP)
            self._restraints_model_id = None
            return

        geo = self._geo_cache.get(mid)
        if geo is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                geo = geo_mod.build_geometry(session.model)
            except Exception as exc:  # a malformed model shouldn't take the app down
                QApplication.restoreOverrideCursor()
                # The commonest cause by far is a ligand with no dictionary, and it is one
                # the app can do something about, so offer that before reporting a wall of
                # cctbx text. Retried once if the user accepts.
                if self._offer_ligand_restraints(mid):
                    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                    try:
                        geo = geo_mod.build_geometry(session.model)
                    except Exception as exc2:
                        self._show_restraint_message(
                            f"Could not build restraints:\n{exc2}")
                        self._restraints_model_id = None
                        return
                else:
                    self._show_restraint_message(f"Could not build restraints:\n{exc}")
                    self._restraints_model_id = None
                    return
            finally:
                QApplication.restoreOverrideCursor()
            self._geo_cache[mid] = geo

        label_fn = _atom_label_fn(session)
        source_fn = _geostd_source_fn(session)  # the geostd link column
        self._suppress_restraint_sync = True
        try:
            for cat, info in self._restraint_tabs.items():
                info["model"].set_source(geo, cat, info["columns"], label_fn, source_fn)
                info["stack"].setCurrentWidget(info["view"])
        finally:
            self._suppress_restraint_sync = False
        self._restraints_model_id = mid
        # Which origins exist is a property of this model, so the dropdown is rebuilt
        # before the filter is applied.
        self._refresh_origin_filter(geo)
        self._apply_restraint_filter()  # respect the shared filter on a fresh build

    def _on_restraint_link_clicked(self, category: str, index) -> None:
        """Click on the geostd column -> reveal that monomer's file in the file browser."""
        model = self._restraint_tabs[category]["model"]
        if not index.isValid() or index.column() != model.source_column():
            return
        _text, path = model.source_for_row(index.row())
        if path:
            _reveal_in_file_manager(path)

    def _on_restraint_selection(self, category: str) -> None:
        """Restraint row -> draw its geometry notation (bond/angle/dihedral) in the
        viewer, marking exactly the participating atoms. Multiple rows -> multiple."""
        if self._suppress_restraint_sync:
            return
        info = self._restraint_tabs[category]
        specs = [
            (category, tuple(int(i) for i in info["model"].i_seqs_for_row(idx.row())))
            for idx in info["view"].selectionModel().selectedRows()
        ]
        self._desktop.show_restraint_notations(self._table_model_id, specs)

    def _build_console_tab(self):
        """A live IPython console bound to the API (created on first view)."""
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        tab = QWidget()
        self._console_layout = QVBoxLayout(tab)
        self._console_layout.setContentsMargins(0, 0, 0, 0)
        self._console_placeholder = QLabel("Opening the API console…")
        self._console_placeholder.setWordWrap(True)
        self._console_placeholder.setContentsMargins(12, 12, 12, 12)
        self._console_layout.addWidget(self._console_placeholder)
        return tab

    def _on_tab_changed(self, index: int) -> None:
        if index == self._console_tab_index:
            self._ensure_console()

    def _ensure_console(self) -> None:
        """Build the embedded console the first time its tab is opened."""
        if self._console_started:
            return
        self._console_started = True

        from PySide6.QtWidgets import QLabel

        from . import console as console_mod

        if self._console_placeholder is not None:
            self._console_placeholder.setParent(None)
            self._console_placeholder = None

        if not console_mod.console_available():
            self._console_layout.addWidget(QLabel(console_mod.CONSOLE_MISSING_MESSAGE))
            return
        try:
            import numpy as np

            from .api_guide import ApiGuide
            from .live import LiveSession

            namespace = {
                "app": self._desktop,
                "session": self._desktop.active_model_session(),
                "np": np,
                "api": ApiGuide(LiveSession),
            }
            self._console = console_mod.EmbeddedConsole(
                namespace, banner=console_mod.default_banner()
            )
            self._console_layout.addWidget(self._console.widget)
        except Exception as exc:  # a broken console must not take the app down
            self._console_layout.addWidget(QLabel(f"Console failed to start:\n{exc}"))

    def _refresh_console_session(self) -> None:
        """Keep the console's ``session`` bound to the active model."""
        if self._console is not None:
            self._console.push({"session": self._desktop.active_model_session()})

    def shutdown_console(self) -> None:
        """Tear down the embedded kernel (called on app quit)."""
        if self._console is not None:
            self._console.shutdown()
            self._console = None

    def _build_atoms_subtab(self):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QHBoxLayout,
            QLabel,
            QTableView,
            QVBoxLayout,
            QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        # A selection can span models; the table shows one model at a time. The
        # dropdown picks which — it follows the active model until the user pins it.
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._table_model_combo = QComboBox()
        self._table_model_combo.setToolTip(
            "Which model's atoms this table shows. Follows the active model until you "
            "change it; pick the active model again to resume following."
        )
        self._table_model_combo.currentIndexChanged.connect(self._on_table_model_combo_changed)
        model_row.addWidget(self._table_model_combo, stretch=1)
        layout.addLayout(model_row)

        self._atoms_count = QLabel("No structure loaded")
        layout.addWidget(self._atoms_count)

        self._atom_model = _make_atom_table_model()
        view = QTableView()
        self._atom_view = view
        view.setModel(self._atom_model)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        view.verticalHeader().setVisible(False)  # the "#" column is the atom index
        view.setAlternatingRowColors(True)
        view.setWordWrap(False)
        # ResizeToContents would scan all rows (O(N)); keep interactive + stretch.
        view.horizontalHeader().setStretchLastSection(True)
        view.selectionModel().selectionChanged.connect(lambda *_: self._on_table_selection_changed())
        layout.addWidget(view, stretch=1)

        # Table -> viewer selection is debounced so a drag doesn't flood the socket.
        self._suppress_table_sync = False
        self._table_sync_timer = QTimer()
        self._table_sync_timer.setSingleShot(True)
        self._table_sync_timer.setInterval(60)
        self._table_sync_timer.timeout.connect(self._push_table_selection_to_viewer)
        return tab

    # -- window ----------------------------------------------------------

    def show(self) -> None:
        self._window.show()

    def set_geometry(self, rect) -> None:
        self._window.setGeometry(rect)

    def widget(self):
        return self._window

    def close(self) -> None:
        """Tear the controls down, releasing the several hundred widgets they own.

        The counterpart to :meth:`ViewportWindow.close`, and needed for the same reason:
        a controls window is a tree of ~430 widgets, and nothing else drops the last
        reference to it. Closing the window hides it; ``deleteLater`` is what actually
        frees the tree, once the event loop next delivers deferred deletes.

        Immaterial to a desktop run, which builds one and then exits — but a process that
        builds several, as a test script does, otherwise carries every one of them to the
        end. No event loop is pumped here, for the reason given on the viewport's close.
        """
        try:
            self._window.close()
            self._window.deleteLater()
        except Exception:  # pragma: no cover - defensive teardown
            pass

    _STATUS_STYLE = "color: palette(placeholder-text);"

    def _status_warn_style(self) -> str:
        """The amber flash, in the shade that reads on the current theme (see :func:`_accent`)."""
        return f"color: {_accent(self._window, 'warn')}; font-weight: 600;"

    def _set_status(self, text: str) -> None:
        self._real_status = text
        self._status_label.setStyleSheet(self._STATUS_STYLE)  # a new message clears a flash
        if not self._tab_hover:  # don't stomp a tab label the pointer is showing
            self._status_label.setText(text)

    def _register_busy_button(self, button, label: str) -> None:
        """Disable ``button`` while the operation it starts is running.

        These take tens of seconds, so without this a second click just queues a duplicate of
        work already in flight. Keyed by operation rather than disabling everything, so an
        unrelated action stays available while one runs.

        Only for buttons with no other enabled/disabled logic — re-enabling on completion would
        otherwise stomp it. (Minimize is excluded for exactly that reason: it drives its own
        state from ``minimizing_changed``.)
        """
        self._busy_buttons.setdefault(label, []).append(button)

    def _on_busy_changed(self, payload) -> None:
        """Show or hide the busy bar, and disable the buttons whose operations are running.

        The label is put in the status line only as a fallback: the workers set their own,
        more specific text ("finding hotspots in 1tec…") and that should win. Here it just
        guarantees *something* names the wait if a worker never got round to saying anything.
        """
        running, label, active = payload
        self._busy_bar.setVisible(bool(running))
        for key, buttons in self._busy_buttons.items():
            for button in buttons:
                try:
                    button.setEnabled(key not in active)
                except RuntimeError:  # pragma: no cover - widget torn down under us
                    pass
        if running and label and self._real_status in ("Ready", ""):
            self._set_status(f"{label}…")
        elif not running and self._real_status.endswith("…"):
            # Nothing said anything more specific; do not leave a dangling "working" message.
            self._set_status("Ready")

    def _flash_status(self, text: str) -> None:
        """Show a status message with a brief amber highlight, so a refused action is
        noticed rather than reading as a silent nothing-happened."""
        from PySide6.QtCore import QTimer

        self._real_status = text
        self._status_label.setText(text)
        self._status_label.setStyleSheet(self._status_warn_style())
        # The label is passed as the timer's context object, not captured only by the
        # lambda: four seconds is long enough for the window to be closed first, and
        # without a context Qt still fires the callback onto a deleted C++ object and the
        # process dies with a shiboken "already deleted" error. With one, Qt drops the
        # pending call when the label goes.
        QTimer.singleShot(
            4000, self._status_label,
            lambda: self._status_label.setStyleSheet(self._STATUS_STYLE))

    def _on_tab_hover(self, index: int) -> None:
        """Name the hovered tab in the status line; restore the real status on leave."""
        if 0 <= index < len(self._tab_labels):
            self._tab_hover = True
            self._status_label.setText(self._tab_labels[index])
        elif self._tab_hover:
            self._tab_hover = False
            self._status_label.setText(self._real_status)

    # -- handlers --------------------------------------------------------

    def _on_open_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        # Select several files (a model + its map(s)) to load them as one cctbx
        # map_model_manager group; a single file loads individually.
        paths, _ = QFileDialog.getOpenFileNames(
            self._window, "Open model(s) and/or map(s)", "", FILE_DIALOG_FILTER
        )
        if not paths:
            return
        try:
            kind = self._desktop.load_files(paths)
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not load file", str(exc))
            self._set_status(f"Failed to load {len(paths)} file(s)")
            return
        label = Path(paths[0]).name if len(paths) == 1 else f"{len(paths)} files"
        self._file_label.setText(f"{label}  ({kind})")

    def _on_fetch(self) -> None:
        """Fetch a model, reflections and/or map from the PDB/EMDB into the working
        directory and load them. (Half-maps and local resolution have their own wizard.)"""
        from PySide6.QtWidgets import (
            QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
            QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout)

        dialog = QDialog(self._window)
        dialog.setWindowTitle("Fetch from the PDB / EMDB")
        outer = QVBoxLayout(dialog)
        form = QFormLayout()
        pdb_edit = QLineEdit()
        pdb_edit.setPlaceholderText("e.g. 6nt5")
        form.addRow("PDB id:", pdb_edit)
        emdb_edit = QLineEdit()
        emdb_edit.setPlaceholderText("optional — looked up from the PDB id for the map")
        form.addRow("EMDB number:", emdb_edit)
        outer.addLayout(form)

        checks = {
            "model": QCheckBox("Model"),
            "reflections": QCheckBox("Reflections (structure factors)"),
            "map": QCheckBox("Map"),
        }
        checks["model"].setChecked(True)
        checks["map"].setChecked(True)
        outer.addWidget(QLabel("Fetch:"))
        for cb in checks.values():
            outer.addWidget(cb)

        dir_row = QHBoxLayout()
        dir_label = QLabel(str(self._desktop.work_dir()))
        dir_label.setStyleSheet("color: palette(placeholder-text);")
        browse = QPushButton("Change…")

        def _browse_dir():
            chosen = QFileDialog.getExistingDirectory(
                dialog, "Working directory", str(self._desktop.work_dir()))
            if chosen:
                self._desktop.set_work_dir(chosen)
                dir_label.setText(chosen)
        browse.clicked.connect(_browse_dir)
        dir_row.addWidget(QLabel("Save to:"))
        dir_row.addWidget(dir_label, 1)
        dir_row.addWidget(browse)
        outer.addLayout(dir_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return
        pdb_id = pdb_edit.text().strip() or None
        emdb_number = emdb_edit.text().strip() or None
        entities = [key for key, cb in checks.items() if cb.isChecked()]
        if not entities:
            QMessageBox.information(self._window, "Nothing selected",
                                    "Tick at least one thing to fetch.")
            return
        try:
            self._desktop.fetch_and_load(
                pdb_id=pdb_id, emdb_number=emdb_number, entities=entities)
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not fetch", str(exc))

    def _on_localres_wizard(self) -> None:
        """Colour a map by local resolution computed from its two half-maps.

        The half-maps are read in cctbx only (never shown); the resulting resolution map is
        pinned, hidden, under the full map, and the map is coloured by it — a toggle that
        then lives in the map's own appearance controls. Inputs are either local files or
        fetched from the PDB/EMDB (the same computation either way)."""
        from PySide6.QtWidgets import (
            QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
            QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton,
            QVBoxLayout, QWidget)

        maps_filter = "Maps (*.mrc *.map *.ccp4);;All files (*)"
        dialog = QDialog(self._window)
        dialog.setWindowTitle("Colour by local resolution")
        outer = QVBoxLayout(dialog)
        intro = QLabel("Compute local resolution from two half-maps and colour a map by it.")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        local_radio = QRadioButton("From local files")
        fetch_radio = QRadioButton("Fetch from the PDB / EMDB")
        local_radio.setChecked(True)
        group = QButtonGroup(dialog)
        group.addButton(local_radio)
        group.addButton(fetch_radio)

        # -- local files --
        outer.addWidget(local_radio)
        local_panel = QWidget()
        lp = QFormLayout(local_panel)
        full_combo = QComboBox()
        vols = self._desktop.colorable_volumes()
        for v_id, name in vols:
            full_combo.addItem(name, v_id)
        full_combo.addItem("Browse to a map file…", "__browse__")
        lp.addRow("Full map:", full_combo)

        def _file_row(placeholder):
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            pick = QPushButton("Browse…")

            def _pick():
                p, _ = QFileDialog.getOpenFileName(
                    dialog, "Half-map", str(self._desktop.work_dir()), maps_filter)
                if p:
                    edit.setText(p)
            pick.clicked.connect(_pick)
            row.addWidget(edit, 1)
            row.addWidget(pick)
            holder = QWidget()
            holder.setLayout(row)
            return holder, edit

        half1_w, half1_edit = _file_row("half map 1 (.mrc/.map)")
        half2_w, half2_edit = _file_row("half map 2 (.mrc/.map)")
        lp.addRow("Half map 1:", half1_w)
        lp.addRow("Half map 2:", half2_w)
        outer.addWidget(local_panel)

        # -- fetch --
        outer.addWidget(fetch_radio)
        fetch_panel = QWidget()
        fp = QFormLayout(fetch_panel)
        pdb_edit = QLineEdit()
        pdb_edit.setPlaceholderText("e.g. 6nt5")
        emdb_edit = QLineEdit()
        emdb_edit.setPlaceholderText("optional — looked up from the PDB id")
        fp.addRow("PDB id:", pdb_edit)
        fp.addRow("EMDB number:", emdb_edit)
        outer.addWidget(fetch_panel)

        def _sync_mode():
            local_panel.setEnabled(local_radio.isChecked())
            fetch_panel.setEnabled(fetch_radio.isChecked())
        local_radio.toggled.connect(_sync_mode)
        _sync_mode()

        dir_row = QHBoxLayout()
        dir_label = QLabel(str(self._desktop.work_dir()))
        dir_label.setStyleSheet("color: palette(placeholder-text);")
        change = QPushButton("Change…")

        def _browse_dir():
            chosen = QFileDialog.getExistingDirectory(
                dialog, "Working directory", str(self._desktop.work_dir()))
            if chosen:
                self._desktop.set_work_dir(chosen)
                dir_label.setText(chosen)
        change.clicked.connect(_browse_dir)
        dir_row.addWidget(QLabel("Save to:"))
        dir_row.addWidget(dir_label, 1)
        dir_row.addWidget(change)
        outer.addLayout(dir_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return
        try:
            if fetch_radio.isChecked():
                pdb_id = pdb_edit.text().strip() or None
                emdb_number = emdb_edit.text().strip() or None
                if not (pdb_id or emdb_number):
                    QMessageBox.information(self._window, "Fetch",
                                            "Enter a PDB id or an EMDB number.")
                    return
                self._desktop.fetch_and_compute_resolution(
                    pdb_id=pdb_id, emdb_number=emdb_number)
            else:
                h1, h2 = half1_edit.text().strip(), half2_edit.text().strip()
                if not (h1 and h2):
                    QMessageBox.information(self._window, "Half-maps",
                                            "Choose both half-maps.")
                    return
                full_vid = full_combo.currentData()
                if full_vid == "__browse__":
                    p, _ = QFileDialog.getOpenFileName(
                        dialog, "Full map", str(self._desktop.work_dir()), maps_filter)
                    if not p:
                        return
                    full_vid = self._desktop.add_map_file(p)
                self._desktop.compute_resolution_map(full_vid, h1, h2)
        except Exception as exc:
            QMessageBox.warning(
                self._window, "Could not compute local resolution", str(exc))
            return

    def _on_save_picture(self) -> None:
        """Ask where to put it first, then photograph: the capture is a round trip to
        the viewer, and a file dialog in the middle of it would be a strange pause."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save picture", "pxviewer.png", "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self._desktop.save_screenshot(path)
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not save picture", str(exc))

    def _on_load_sample(self, filename: Optional[str] = None) -> None:
        from PySide6.QtWidgets import QMessageBox

        sample = sample_structure_path(filename)
        if sample is None:
            QMessageBox.warning(self._window, "Sample not available", "The bundled sample file is missing.")
            return
        try:
            kind = self._desktop.load_file(str(sample))
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not load sample", str(exc))
            return
        self._file_label.setText(f"{sample.name}  ({kind})")

    def _on_pair(self) -> None:
        """Pair an unpaired model with an unpaired map, chosen explicitly."""
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
            QMessageBox,
        )

        models, volumes = self._desktop.pairable()
        existing = self._desktop.alignable()
        can_create = bool(models) and bool(volumes)
        if not can_create and not existing:
            QMessageBox.information(
                self._window, "Nothing to pair",
                "Pairing needs an unpaired model and map, or an existing map/model pair "
                "that can be checked for a missing shift.")
            return

        dialog = QDialog(self._window)
        dialog.setWindowTitle("Pair or align model with map")
        form = QFormLayout(dialog)
        note = QLabel(
            "cctbx will move these into a common frame, so the model may shift.\n"
            "That is what makes them usable together — minimizing into density, say.")
        note.setStyleSheet("color: palette(placeholder-text);")
        form.addRow(note)
        operation = QComboBox()
        if can_create:
            operation.addItem("Pair an unpaired model and map", ("pair", None, None))
        for m, v in existing:
            operation.addItem(
                f"Align {m['name']} with {v['name']}", ("align", m["id"], v["id"]))
        if existing:
            form.addRow("Action:", operation)
        model_combo = QComboBox()
        for m in models:
            model_combo.addItem(m["name"], m["id"])
        volume_combo = QComboBox()
        for v in volumes:
            volume_combo.addItem(v["name"], v["id"])
        form.addRow("Model:", model_combo)
        form.addRow("Map:", volume_combo)
        detect_shift = QCheckBox("Detect a missing shift from the map density")
        detect_shift.setChecked(False)
        detect_shift.setToolTip(
            "Search for a translation that puts the model into the density, then apply it "
            "with cctbx's shift-aware model API so it is recorded in shift_cart and is not "
            "baked into the model's original coordinates. Off by default: files with correct "
            "origin metadata need only normal pairing.")
        form.addRow(detect_shift)
        def sync_operation() -> None:
            mode, _mid, _vid = operation.currentData()
            creating = mode == "pair"
            model_combo.setEnabled(creating)
            volume_combo.setEnabled(creating)
            note.setText(
                "cctbx will move these into a common frame, so the model may shift.\n"
                "That is what makes them usable together — minimizing into density, say."
                if creating else
                "Check this existing pair for a missing Cartesian origin shift.\n"
                "The current map_model_manager is retained; no second pair is created.")
        operation.currentIndexChanged.connect(lambda _i: sync_operation())
        sync_operation()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            mode, mid, vid = operation.currentData()
            if mode == "align":
                self._desktop.align_paired_model_with_map(
                    mid, vid, detect_shift=detect_shift.isChecked())
            else:
                self._desktop.pair_model_with_map(
                    model_combo.currentData(), volume_combo.currentData(),
                    detect_shift=detect_shift.isChecked())
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not pair or align", str(exc))

    def _update_pair_button(self) -> None:
        """Enable for either a new pair or alignment of an existing one."""
        models, volumes = self._desktop.pairable()
        self._pair_btn.setEnabled(
            (bool(models) and bool(volumes)) or bool(self._desktop.alignable()))

    def _on_write_object(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        kind, ident = self._focused
        if ident is None:
            self._set_status("select an object to write")
            return
        it = self._find_item(kind, ident)
        default = it["name"] if it else "out"
        if kind == "model":
            # .mmcif marks these as macromolecular coordinates, distinct from a monomer
            # restraint .cif or a small-molecule core CIF; both extensions still write mmCIF.
            fmt = "mmCIF (*.mmcif *.cif);;PDB (*.pdb)"
            default = f"{default}.mmcif"
        else:
            fmt = "CCP4/MRC map (*.mrc *.map *.ccp4)"
        path, _ = QFileDialog.getSaveFileName(self._window, "Write object", default, fmt)
        if not path:
            return
        try:
            self._desktop.write_object(kind, ident, path)
            self._set_status(f"Wrote {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self._window, "Write failed", str(exc))

    def _offer_ligand_restraints(self, mid: str) -> bool:
        """Offer to build dictionaries for a model's unrecognised ligands. True if any were.

        The reason this is a prompt and not automatic: the dictionary is *inferred*. rdkit
        reads the bonds out of the coordinates and the bond orders out of the graph, which
        is a guess about chemistry the file never stated. A well-modelled ligand it gets
        right; a badly modelled one it gets wrong in a way that then looks authoritative,
        which is exactly the situation restraints are usually needed for. So the user is
        shown the ligands, told what the source is, and asked -- one at a time, since a
        model can carry one ligand worth guessing at and another worth fetching properly.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication, QCheckBox, QDialog, QDialogButtonBox, QLabel, QMessageBox,
            QVBoxLayout)

        unknown = self._desktop.unknown_ligands(mid)
        if not unknown:
            return False

        dialog = QDialog(self._window)
        dialog.setWindowTitle("Ligands without restraints")
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "cctbx has no dictionary for these residues, so it will not build restraints "
            "for <i>any</i> of this model — minimize, drag, geometry and validation are "
            "all unavailable until they are resolved.<br><br>"
            "pxviewer can generate one from the coordinates: rdkit works out the bonding, "
            "and the ideal values are measured from a clean conformer of the chemistry it "
            "perceives — <b>not</b> from the geometry as modelled. Check what it perceived "
            "afterwards; on a poorly built ligand the guess can be wrong.")
        intro.setWordWrap(True)
        intro.setMaximumWidth(460)
        layout.addWidget(intro)

        boxes = {}
        for found in unknown:
            code = found["code"]
            if code in boxes:
                continue                    # one dictionary serves every copy
            copies = sum(1 for u in unknown if u["code"] == code)
            label = "%s — %d atoms" % (code, found["n_atoms"])
            if copies > 1:
                label += ", %d copies" % copies
            else:
                label += " (chain %s, residue %s)" % (found["chain"], found["resseq"])
            box = QCheckBox(label)
            box.setChecked(True)
            boxes[code] = box
            layout.addWidget(box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Make restraints")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        chosen = [code for code, box in boxes.items() if box.isChecked()]
        if not chosen:
            return False

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            made = self._desktop.generate_ligand_restraints(mid, chosen)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self._window, "Could not make restraints", str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()

        # What rdkit decided the ligand *is*, which is the thing worth checking.
        QMessageBox.information(
            self._window, "Restraints made",
            "Perceived chemistry:\n\n" + "\n".join(
                "  %s   %s" % (code, smiles) for code, smiles in sorted(made.items()))
            + "\n\nSave the dictionary with “Export ligand” on the model's Appearance "
              "pane if you want it for refinement elsewhere.")
        return True

    def _on_export_ligand(self, mid: str, name: str) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        default = name.split(" (")[0].strip() or "ligand"  # "AIN (ligand)" -> "AIN"
        # .mmcif for the coordinates, so they read as macromolecular mmCIF and are not
        # confused with the monomer restraint .cif (nor a small-molecule COD/core CIF).
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Export ligand (coordinates + restraints)",
            f"{default}.mmcif", "mmCIF coordinates (*.mmcif *.cif);;PDB coordinates (*.pdb)")
        if not path:
            return
        try:
            coord, restraints = self._desktop.export_ligand(mid, path)
            self._set_status(f"Exported {Path(coord).name} + {Path(restraints).name}")
        except Exception as exc:
            QMessageBox.warning(self._window, "Export failed", str(exc))

    def _on_select_expression(self) -> None:
        self._run_selection(self._select_expr.text())

    def _run_selection(self, expr: str) -> None:
        self._select_expr.setText(expr)
        try:
            n = self._desktop.select_by_expression(expr)
        except Exception as exc:  # invalid syntax / no model
            self._selection_label.setText(
                f"<span style='color:{_accent(self._window, 'error')}'>{exc}</span>")
            return
        self._selection_label.setText("selection cleared" if not expr.strip() else f"{n} atom(s) selected")

    def _on_run_map_model_demo(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        try:
            self._desktop.load_map_model_demo()
        except Exception as exc:  # generating the map can fail; don't take the app down
            QMessageBox.warning(self._window, "Map+model demo failed", str(exc))

    def _on_run_xray_demo(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        try:
            self._desktop.load_xray_demo()
        except Exception as exc:  # computing the reflections can fail; keep the app up
            QMessageBox.warning(self._window, "X-ray demo failed", str(exc))

    def _on_run_ligand_fitting_demo(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        try:
            self._desktop.load_ligand_fitting_demo()
        except Exception as exc:  # building the ligand / reflections can fail; keep the app up
            QMessageBox.warning(self._window, "Ligand-fitting demo failed", str(exc))

    def _on_run_real_space_refinement_demo(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        try:
            self._desktop.load_real_space_refinement_demo()
        except Exception as exc:  # generating the map can fail; keep the app up
            QMessageBox.warning(self._window, "Cryo-EM demo failed", str(exc))

    def _on_stop_demo(self) -> None:
        self._desktop.stop_demo()

    def _on_toggle_select(self, checked: bool) -> None:
        if checked:
            self._refine_drag_btn.setChecked(False)
            self._desktop.enable_mouse_selection()
        else:
            self._desktop.disable_mouse_selection()

    def _on_toggle_refine_drag(self, checked: bool) -> None:
        if checked:
            self._pick_btn.setChecked(False)
            # Arming the drag is the moment to sort restraints out: the drag itself builds
            # them on its own thread, where a failure can only be reported, not fixed.
            self._offer_restraints_if_blocked()
        self._desktop.set_tug_enabled(checked)

    def _on_clear_selection(self) -> None:
        self._desktop.clear_selection()

    def _on_measure(self, kind: str) -> None:
        try:
            self._set_status(self._desktop.measure_selection(kind))
        except Exception as exc:
            self._set_status(str(exc))

    def _on_clear_measurements(self) -> None:
        self._desktop.clear_measurements()

    def _on_analyze(self) -> None:
        try:
            self._desktop.analyze_clashes()
        except Exception as exc:
            self._set_status(str(exc))

    def _icon(self, name: str, size: int = 18):
        """A Lucide icon tinted to the button text color (or None if the asset is gone)."""
        return _line_icon(name, self._btn_tint, size=size)

    def _retint_icons(self) -> None:
        """Re-tint every registered icon to the current theme's text color.

        Called on a palette change (a light/dark switch, or the real palette finally landing
        once the window is shown), since baked-pixmap icons — unlike palette() stylesheets —
        do not re-color themselves. Idempotent and cheap; a no-op if the tint is unchanged."""
        from PySide6.QtGui import QPalette

        tint = self._window.palette().color(self._icon_role)
        if tint == self._btn_tint and self._icon_registry:
            return  # nothing changed
        self._btn_tint = tint
        for apply_icon, name, size in self._icon_registry:
            icon = _line_icon(name, tint, size=size)
            if icon is not None:
                apply_icon(icon)
        for index, name in self._tab_icon_names:
            icon = _line_icon(name, tint)
            if icon is not None:
                self._tabs.setTabIcon(index, icon)
        # The Minimize/Stop pair carries a state-dependent look (filled accent + white glyph
        # when active), so repaint it for the current run state rather than a flat re-tint.
        self._on_minimizing_changed(not self._desktop._minimize_idle.is_set())

    def _refresh_theme(self, *, force: bool = False) -> None:
        """Apply one settled application palette to the complete controls subtree.

        Qt normally propagates a system appearance change. On macOS, a window left open
        across a light/dark transition can instead leave resolved ``palette(...)`` QSS
        values behind while native widgets adopt the new appearance. Repolish only styled
        widgets, then rebuild the pixmap-backed icons and semantic accent buttons. Native
        widget palettes remain Qt's responsibility: forcing them recursively can crash
        PySide while WebEngine windows are being mapped.
        """
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication, QWidget

        palette = QApplication.palette()
        roles = (
            QPalette.ColorRole.Window, QPalette.ColorRole.WindowText,
            QPalette.ColorRole.Base, QPalette.ColorRole.Text,
            QPalette.ColorRole.Button, QPalette.ColorRole.ButtonText,
            QPalette.ColorRole.Highlight,
        )
        signature = tuple(palette.color(role).rgba() for role in roles)
        if not force and signature == self._theme_signature:
            return
        self._theme_signature = signature

        widgets = [self._window, *self._window.findChildren(QWidget)]
        # Qt resolves palette() references when a stylesheet is polished. Reassigning the
        # same string is optimized away, so clear it first before restoring it. Do not call
        # setPalette on every descendant: during native macOS window mapping that recursively
        # invalidates controls and can crash Qt. Unstyled widgets already follow QApplication.
        for widget in widgets:
            qss = widget.styleSheet()
            if qss:
                widget.setStyleSheet("")
                widget.setStyleSheet(qss)
        self._retint_icons()

    def _make_icon_button(self, icon_name, fallback_text, tooltip, *, checkable=False,
                          icon_size=18, square=False):
        """An icon-only button (tinted SVG), falling back to text if the asset is gone.

        ``icon_size`` is the glyph size; ``square`` fixes the button to a square just larger
        than the glyph — for the detailed geometry icons that need room to read."""
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QPushButton

        b = QPushButton()
        b.setCheckable(checkable)
        b.setStyleSheet(_icon_button_base_qss())  # a tidy frame on macOS, native elsewhere
        icon = self._icon(icon_name, size=icon_size)
        if icon is not None:
            b.setIcon(icon)
            b.setIconSize(QSize(icon_size, icon_size))
            self._icon_registry.append((b.setIcon, icon_name, icon_size))
        else:
            b.setText(fallback_text)
        b.setToolTip(tooltip)
        if square:
            side = icon_size + 16
            b.setFixedSize(side, side)
        return b

    def _on_help(self) -> None:
        # Placeholder until the documentation is linked. Guided tutorials live under Get.
        self._set_status("Documentation coming soon. Guided tutorials are under Get.")

    def _on_mouse_help(self) -> None:
        """Pop up the mouse/keyboard reference above the mouse button (built once, reused).

        A ``Qt.Popup`` so it dismisses on a click away, like a menu — the reference is a
        glance, not a window to manage."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QFrame, QVBoxLayout

        popup = getattr(self, "_mouse_popup", None)
        if popup is None:
            popup = QFrame(self._window, Qt.WindowType.Popup)
            popup.setFrameShape(QFrame.Shape.StyledPanel)
            v = QVBoxLayout(popup)
            v.setContentsMargins(8, 8, 8, 8)
            v.addWidget(self._build_mouse_legend())
            self._mouse_popup = popup
        popup.adjustSize()
        # The button sits at the bottom of the pane, so open the popup above it, right-aligned.
        corner = self._mouse_btn.mapToGlobal(self._mouse_btn.rect().topRight())
        popup.move(corner.x() - popup.width(), corner.y() - popup.height() - 4)
        popup.show()

    # -- guided tutorials: a non-modal coach that advances when a step is actually done ----

    def _wire_coach(self) -> None:
        """Connect the viewport window's coach buttons to this window's tutorial logic. The
        pane and its widgets are built on the viewport (see ViewportWindow._build_coach_pane)
        so a tutorial splits the viewer, not this controls pane."""
        vp = self._desktop._viewport
        vp.coach_close.clicked.connect(lambda: self._tutorial_exit())
        vp.coach_show.clicked.connect(self._on_coach_show_me)
        vp.coach_back.clicked.connect(self._tutorial_back)
        vp.coach_next.clicked.connect(self._tutorial_next)

    @staticmethod
    def _coach_markup(text: str) -> str:
        """Tiny markdown → HTML for the coach: **bold**, `code`, and blank-line breaks."""
        import html
        import re

        t = html.escape(text)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        return t.replace("\n\n", "<br><br>").replace("\n", "<br>")

    def _load_tutorial_data(self, tutorial_obj) -> bool:
        """Put the tutorial's example on screen. False means the user backed out.

        Asked about only when there is something to lose. On an empty scene this is
        silent -- the common case, and a dialog there would be friction for nothing. With
        objects already loaded it is a real choice, because a tutorial reasons about the
        scene: its steps say things like "select the reflections", and a second set of
        reflections makes that instruction ambiguous. Replacing is offered first for that
        reason, but keeping the user's work is never done without asking.
        """
        from PySide6.QtWidgets import QMessageBox

        loader = getattr(tutorial_obj, "loader", None)
        if loader is None:
            return True

        occupied = bool(self._desktop._models or self._desktop._volumes
                        or self._desktop._reflections)
        if occupied:
            box = QMessageBox(self._window)
            box.setWindowTitle("Start tutorial")
            box.setText(f"“{tutorial_obj.title}” loads its own example.")
            box.setInformativeText(
                "You have objects loaded already. Clear them first, or keep them and add "
                "the example alongside?")
            clear = box.addButton("Clear and load", QMessageBox.ButtonRole.AcceptRole)
            keep = box.addButton("Keep mine", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(clear)
            box.exec()
            clicked = box.clickedButton()
            if clicked is clear:
                self._desktop._clear_all()
                self._desktop._emit_loaded_changed()
            elif clicked is not keep:
                return False

        try:
            loader(self._desktop)
        except Exception as exc:  # a demo that computes a map can fail; keep the app up
            QMessageBox.warning(self._window, "Could not load the tutorial's example",
                                str(exc))
            return False
        return True

    def _start_tutorial(self, tutorial_obj) -> None:
        from PySide6.QtCore import QTimer

        # The example first: a tutorial that had to ask for its data could not tell
        # whether it had arrived (see Tutorial.loader), so it opened on a step about a
        # structure that might not be the one on screen.
        if not self._load_tutorial_data(tutorial_obj):
            return

        self._tutorial = tutorial_obj
        self._tutorial_step = 0
        self._desktop._viewport.coach_bar.setVisible(True)
        if self._tutorial_timer is None:
            self._tutorial_timer = QTimer(self._window)
            self._tutorial_timer.setInterval(400)  # poll the step's done() predicate
            self._tutorial_timer.timeout.connect(self._maybe_advance_tutorial)
        self._tutorial_timer.start()
        self._show_tutorial_step()

    def _show_tutorial_step(self) -> None:
        self._stop_highlight()  # clear any flash from the previous step
        tut = self._tutorial
        if tut is None:
            return
        step = tut.steps[self._tutorial_step]
        vp = self._desktop._viewport
        vp.coach_title.setText(tut.title)
        vp.coach_progress.setText(f"Step {self._tutorial_step + 1} / {len(tut.steps)}")
        vp.coach_text.setText(self._coach_markup(step.text))
        vp.coach_show.setVisible(step.target is not None)  # "Show me where" only if targeted
        vp.coach_back.setEnabled(self._tutorial_step > 0)
        last = self._tutorial_step == len(tut.steps) - 1
        vp.coach_next.setText("Finish" if last else ("Skip" if step.done else "Next"))

    def _maybe_advance_tutorial(self) -> None:
        if self._tutorial is None:
            return
        step = self._tutorial.steps[self._tutorial_step]
        if step.done is None:
            return
        try:
            satisfied = bool(step.done(self))
        except Exception:  # pragma: no cover - a predicate touching not-yet-ready state
            satisfied = False
        if satisfied:
            self._advance_tutorial(auto=True)

    def _advance_tutorial(self, *, auto: bool) -> None:
        if self._tutorial is None:
            return
        if self._tutorial_step >= len(self._tutorial.steps) - 1:
            self._tutorial_exit(finished=True)
            return
        self._tutorial_step += 1
        if auto:
            self._flash_status("✓ step done")
        self._show_tutorial_step()

    def _tutorial_next(self) -> None:
        self._advance_tutorial(auto=False)

    def _tutorial_back(self) -> None:
        if self._tutorial is not None and self._tutorial_step > 0:
            self._tutorial_step -= 1
            self._show_tutorial_step()

    def _tutorial_exit(self, finished: bool = False) -> None:
        self._tutorial = None
        if self._tutorial_timer is not None:
            self._tutorial_timer.stop()
        self._stop_highlight()
        self._desktop._viewport.coach_bar.setVisible(False)
        self._set_status("Tutorial complete — nicely done." if finished else "Tutorial closed.")

    def _on_coach_show_me(self) -> None:
        """Flash the control the current step is about — the coach points, it never acts."""
        if self._tutorial is None:
            return
        step = self._tutorial.steps[self._tutorial_step]
        if step.target is None:
            return
        try:
            widget = step.target(self)
        except Exception:  # pragma: no cover - a target touching not-yet-built UI
            widget = None
        self._highlight_widget(widget)

    def _highlight_widget(self, widget) -> None:
        """Pulse a fading ring *over* ``widget`` (revealing its tab first). An overlay, so it
        never changes the button's size or nudges the widgets around it."""
        from PySide6.QtCore import QPoint, QTimer

        if widget is None:
            return
        self._stop_highlight()
        self._reveal_widget_tab(widget)
        window = widget.window()
        if self._hl_overlay is None or self._hl_overlay.parent() is not window:
            self._hl_overlay = _highlight_overlay_class()(window)
        pad = 5
        top_left = widget.mapTo(window, QPoint(0, 0))
        self._hl_overlay.setGeometry(
            top_left.x() - pad, top_left.y() - pad,
            widget.width() + 2 * pad, widget.height() + 2 * pad)
        self._hl_overlay.set_alpha(1.0)
        self._hl_overlay.show()
        self._hl_overlay.raise_()
        self._hl_phase = 0
        if self._hl_timer is None:
            self._hl_timer = QTimer(self._window)
            self._hl_timer.setInterval(30)
            self._hl_timer.timeout.connect(self._highlight_tick)
        self._hl_timer.start()

    def _highlight_tick(self) -> None:
        import math

        self._hl_phase += 1
        total = 50  # ~1.5 s at 30 ms: a couple of pulses that fade out
        if self._hl_phase > total or self._hl_overlay is None:
            self._stop_highlight()
            return
        envelope = 1.0 - self._hl_phase / total          # fade away over the run
        pulse = 0.5 + 0.5 * math.sin(self._hl_phase * 0.45)  # ...while pulsing
        self._hl_overlay.set_alpha(envelope * pulse)

    def _stop_highlight(self) -> None:
        if self._hl_timer is not None:
            self._hl_timer.stop()
        if self._hl_overlay is not None:
            self._hl_overlay.hide()

    def _reveal_widget_tab(self, widget) -> None:
        """If ``widget`` lives on one of the tabs, switch to that tab so it is visible."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        for i in range(tabs.count()):
            page = tabs.widget(i)
            if page is widget or page.isAncestorOf(widget):
                tabs.setCurrentIndex(i)
                return

    def reflect_dock_state(self, floating: bool) -> None:
        """Keep the dock/detach button in step with the panel's state: maximize-2 to detach
        while docked, minimize-2 to re-dock while floating."""
        icon = self._icon("minimize-2" if floating else "maximize-2")
        if icon is not None:
            self._dock_btn.setIcon(icon)
        else:
            self._dock_btn.setText("Dock" if floating else "Detach")
        self._dock_btn.setToolTip(
            "Re-dock the controls" if floating else "Detach the controls to their own window")

    def _on_analysis_ready(self, mid) -> None:
        """Analysis finished: enable and check both overlay toggles (both drawn)."""
        for toggle in (self._contacts_toggle, self._clashes_toggle):
            toggle.setEnabled(True)
            toggle.blockSignals(True)
            toggle.setChecked(True)
            toggle.blockSignals(False)
        self._sync_all_markup_button()  # these are markup too, and now there is some

    def _on_scene_selection_changed(self, scene) -> None:
        """A model's picks changed. Refresh the aggregate label + the atoms table."""
        self._scene_selection = scene or {}
        total = sum(len(v) for v in self._scene_selection.values())
        # Hide/show-selected only make sense with a selection.
        self._hide_sel_btn.setEnabled(total > 0)
        self._show_sel_btn.setEnabled(total > 0)
        self._selection_label.setText(self._desktop.selection_description(self._scene_selection))
        # Viewer -> Geometry: reflect the picks in the atoms + restraint tables.
        self._apply_geometry_filter()

    @contextmanager
    def _table_sync_suppressed(self):
        """Suppress table -> viewer echo while we mutate the model programmatically.

        Resetting the model (set_session/set_filter) or setting rows emits
        selectionChanged; without this guard that would bounce straight back to the
        viewer as a spurious highlight.
        """
        prev = self._suppress_table_sync
        self._suppress_table_sync = True
        try:
            yield
        finally:
            self._suppress_table_sync = prev

    def _table_selection_indices(self):
        """The current selection restricted to the model the table is showing."""
        return self._scene_selection.get(self._table_model_id, [])

    def _apply_table_selection(self) -> None:
        """Reflect the table model's selection: filter the rows, or highlight them."""
        indices = self._table_selection_indices()
        if self._filter_selection_check.isChecked():
            with self._table_sync_suppressed():
                self._atom_model.set_filter(indices)
            self._update_atoms_count()
        else:
            if self._atom_model.is_filtered():
                with self._table_sync_suppressed():
                    self._atom_model.set_filter(None)
                self._update_atoms_count()
            self._select_table_rows(indices)

    def _on_filter_toggled(self, _checked: bool) -> None:
        self._apply_geometry_filter()

    def _apply_geometry_filter(self) -> None:
        """Apply the shared 'show only the selection' state to every Geometry table."""
        self._apply_table_selection()   # the Atoms table
        self._apply_restraint_filter()  # Bonds / Angles / Dihedrals / Chirality / Planarity

    def _on_restraints_changed(self, mid) -> None:
        """A model's restraints were rebuilt: drop the cache and refill the tables.

        Without this an added restraint is real -- minimize honours it, and it is in the
        file the user saves -- but invisible: ``_geo_cache`` is keyed by model id and the
        entry it holds wraps the *previous* restraints manager, so the Bonds table goes on
        listing the restraints from before the edit.

        Only rebuilt when that model's tables are the ones on screen. For any other model
        dropping the cache is enough; it will be rebuilt when the user looks at it.

        Gated on ``_table_model_id`` rather than ``_restraints_model_id``: the loaded-tree
        update runs first and has already cleared the latter, so comparing against it here
        never matches and the tables would keep their stale contents.
        """
        self._geo_cache.pop(mid, None)
        if self._table_model_id == mid:
            self._restraints_model_id = None      # force _ensure_restraints to refill
            self._ensure_restraints()

    def _on_origin_filter_changed(self, _index: int) -> None:
        self._apply_restraint_filter()

    def _selected_origin_id(self):
        """The chosen origin id, or None for "all origins"."""
        combo = getattr(self, "_origin_filter", None)
        return combo.currentData() if combo is not None else None

    def _refresh_origin_filter(self, geo) -> None:
        """Repopulate the origin dropdown from the origins this model actually has.

        Rebuilt per model rather than listed once: which origins exist is a property of
        the structure, and an entry for an origin with no restraints behind it would
        filter every table to nothing.
        """
        combo = getattr(self, "_origin_filter", None)
        if combo is None:
            return
        from .geometry import CATEGORIES

        present = {}
        for cat, _label, _cols in CATEGORIES:
            for oid, name, count in (geo.origins(cat) if geo is not None else []):
                entry = present.setdefault(oid, [name, 0])
                entry[1] += count

        previous = combo.currentData()
        blocked = combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("All", None)
            for oid in sorted(present):
                name, count = present[oid]
                combo.addItem(f"{oid}: {name} ({count})", oid)
            index = combo.findData(previous)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(blocked)
        # Nothing to choose between when a model has one origin, which is the common case.
        combo.setEnabled(len(present) > 1)

    def _apply_restraint_filter(self) -> None:
        """Filter each built restraint table by selection and by origin.

        The two are independent reasons to hide a restraint, so they intersect: asking for
        the user's own edits *within* the current selection is the question worth being
        able to ask, and either filter alone still works.
        """
        if self._restraints_model_id is None:
            return  # restraints not built yet — _ensure_restraints will apply on build
        geo = self._geo_cache.get(self._restraints_model_id)
        if geo is None:
            return
        on = self._filter_selection_check.isChecked()
        selected = set(self._table_selection_indices()) if on else None
        origin_id = self._selected_origin_id()
        self._suppress_restraint_sync = True
        try:
            for cat, info in self._restraint_tabs.items():
                keep = None
                if on:
                    keep = geo.indices_within(cat, selected)
                by_origin = geo.indices_with_origin(cat, origin_id)
                if by_origin is not None:
                    keep = by_origin if keep is None else sorted(set(keep) & set(by_origin))
                info["model"].set_filter(keep)
        finally:
            self._suppress_restraint_sync = False

    def _update_atoms_count(self) -> None:
        n = self._atom_model.rowCount()
        if self._atom_model.is_filtered():
            self._atoms_count.setText(f"{n} selected atom(s)")
        else:
            self._atoms_count.setText(f"{n} atoms" if n else "No structure loaded")

    # -- geometry / atoms table ------------------------------------------

    def _set_table_model(self, mid) -> None:
        """Point the atoms table at model ``mid`` (or None) and reflect its selection."""
        self._table_model_id = mid
        session = self._desktop.session_for(mid)
        with self._table_sync_suppressed():
            self._atom_model.set_session(session)  # clears any filter
        self._update_atoms_count()
        self._apply_table_selection()
        self._invalidate_restraints()  # geometry follows the same model

    def _on_table_model_combo_changed(self, _index: int) -> None:
        if self._suppress_table_model_combo:
            return
        from PySide6.QtCore import Qt

        mid = self._table_model_combo.currentData(Qt.ItemDataRole.UserRole)
        # Picking the active model again resumes auto-follow; any other choice pins.
        active = next((m["id"] for m in self._models_summary if m["active"]), None)
        self._table_pinned = mid is not None and mid != active
        self._set_table_model(mid)

    # -- loaded tree (models + volumes + groups) -------------------------

    def _fit_checkbox_column(self) -> None:
        """Widen the visibility column so a nested row's checkbox is not clipped.

        Column 0 is ResizeToContents, and Qt measures that from the items' contents while
        the tree's own indentation is drawn *inside* the same column. A child row therefore
        gets its parent's width minus one indent for the checkbox: measured at 49 px for a
        root row and 29 px for the resolution map pinned under its full map, which leaves a
        sliver of checkbox that is nearly impossible to hit and reads as a broken widget.

        Qt has no mode for "contents plus the deepest indent", so the column is set by hand
        once the rows are in: the width contents asked for, plus an indent for every level
        below the root that actually exists.
        """
        from PySide6.QtWidgets import QHeaderView, QTreeWidgetItemIterator

        tree = self._loaded_tree
        deepest = 0
        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value():
            node, depth = iterator.value(), 0
            parent = node.parent()
            while parent is not None:
                depth += 1
                parent = parent.parent()
            deepest = max(deepest, depth)
            iterator += 1

        header = tree.header()
        if not deepest:
            # Nothing nested: let Qt size it, which is right and stays right as rows change.
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            return
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        needed = tree.columnWidth(0) + deepest * tree.indentation()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        tree.setColumnWidth(0, needed)

    def _fit_tree_height(self) -> None:
        """Make the object list exactly as tall as what it holds, within limits.

        A list holding two objects should not reserve room for ten: on a small screen
        that space is what decides whether the rest of the pane fits without scrolling.
        Past the ceiling the list keeps its own scrollbar, so nothing is unreachable.
        """
        tree = self._loaded_tree
        rows = 0
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            rows += 1
            if item.isExpanded():
                rows += item.childCount()
        row_height = tree.sizeHintForRow(0) if rows else 0
        wanted = rows * row_height + 2 * tree.frameWidth() + 4
        tree.setMaximumHeight(
            max(_TREE_MIN_HEIGHT, min(wanted, _TREE_MAX_HEIGHT)))

    def _on_loaded_changed(self, summary) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QRadioButton, QTreeWidgetItem

        groups = {g["id"]: g for g in summary.get("groups", [])}
        items = summary.get("items", [])
        self._items = items
        model_items = [it for it in items if it["kind"] == "model"]
        self._models_summary = model_items

        self._suppress_model_events = True
        try:
            self._loaded_tree.clear()
            for button in self._active_group.buttons():
                self._active_group.removeButton(button)  # radios are rebuilt below
            group_nodes: dict = {}
            active_item = None
            # Group parent nodes first (plain headers — membership is from cctbx).
            for it in items:
                gid = it["group"]
                if gid and gid not in group_nodes:
                    g = groups.get(gid) or {}
                    heading = g.get("name", gid)
                    if g.get("label"):
                        heading += f"  ({g['label']})"
                    node = QTreeWidgetItem(self._loaded_tree, [heading])
                    node.setData(0, Qt.ItemDataRole.UserRole, ("group", gid))
                    node.setFirstColumnSpanned(True)  # the header spans the whole row
                    # Bold, so a header reads as one. Without it a group's members and the
                    # loose top-level objects below them look like one flat run, and the
                    # loose ones appear to belong to the group above.
                    font = node.font(0)
                    font.setBold(True)
                    node.setFont(0, font)
                    node.setExpanded(True)
                    group_nodes[gid] = node
            vol_nodes: dict = {}  # vid -> node, so a pinned map can nest under its full map
            for it in items:
                # A resolution map pinned to a full map nests under that map's node (the
                # full map precedes it in the list, so its node already exists); everything
                # else sits under its group header, or at the root.
                pin = it.get("pinned_to")
                parent = vol_nodes.get(pin) if pin else None
                if parent is None:
                    parent = group_nodes.get(it["group"], self._loaded_tree)
                elif isinstance(parent, QTreeWidgetItem):
                    parent.setExpanded(True)
                # [visible check] col 0, [active radio] col 1, [name] col 2 (elides).
                node = QTreeWidgetItem(parent)
                if it["kind"] == "volume":
                    vol_nodes[it["id"]] = node
                node.setData(0, Qt.ItemDataRole.UserRole, (it["kind"], it["id"]))
                if it["visible"] is None:
                    # Reflections: nothing drawable, so nothing to show or hide.
                    node.setFlags(node.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                elif it["kind"] in ("model", "volume") and not self._desktop._can_hide:
                    # Hiding is disabled on software WebGL (this VM's SwiftShader). The
                    # original reason — "hiding segfaults the software renderer" — turned out
                    # to be a misread of the object-tree use-after-free fixed in
                    # _on_tree_item_changed, which crashed on every renderer because no
                    # renderer was involved. The block stays only because software rendering
                    # has not been re-tested since; it is likely safe to lift. Not a dead
                    # control: a click flashes why (see _on_tree_item_clicked).
                    node.setFlags(node.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    node.setCheckState(0, Qt.CheckState.Checked)
                    node.setToolTip(0, "Hiding needs hardware WebGL "
                                       "(not available on software rendering)")
                else:
                    node.setFlags(node.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    node.setToolTip(0, "Visible")
                    node.setCheckState(
                        0, Qt.CheckState.Checked if it["visible"] else Qt.CheckState.Unchecked)
                if it["kind"] == "model":
                    radio = QRadioButton()
                    radio.setToolTip("Active model — drives the atoms table, geometry and selection.")
                    radio.setProperty("mid", it["id"])
                    self._active_group.addButton(radio)
                    radio.setChecked(bool(it.get("active")))  # won't fire buttonClicked
                    self._loaded_tree.setItemWidget(node, 1, radio)
                # No marker suffix: its name ("Ligand marker N") already says what it is.
                suffix = {"volume": "   [map]", "reflections": "   [data]"}
                # Indent a group member's name by hand. Qt applies tree indentation to
                # column 0 only, so the checkbox shifts with depth but the name — the part
                # you actually read — sits at the same x whether the object is in a group or
                # standing alone, which made every object look like a group member. Indent
                # the name to match, and an object at the root reads as standing alone.
                indent = _GROUP_MEMBER_INDENT if (it["group"] or it.get("pinned_to")) else ""
                node.setText(2, indent + it["name"] + suffix.get(it["kind"], ""))
                node.setToolTip(2, it["name"])  # full name on hover when elided
                if it.get("active"):
                    active_item = node
            if active_item is not None:
                self._loaded_tree.setCurrentItem(active_item)
        finally:
            self._suppress_model_events = False
        self._fit_checkbox_column()
        self._fit_tree_height()
        self._sync_table_model_combo(model_items)
        self._refresh_console_session()
        self._update_minimize_map()  # the active model may now have (or have lost) a map
        self._update_tug_density()
        self._update_pair_button()
        # Point the Appearance pane at the focused object. Focusing a model activates
        # it, so a focused *model* must always be the active one — if the active model
        # changed underneath us (a new model, a radio click, hydrogenate+analyze),
        # follow it. A focused volume is left alone while it still exists.
        kind, ident = self._focused
        active = next((m for m in model_items if m["active"]), None)
        active_ref = ("model", active["id"]) if active else (None, None)
        if self._find_item(kind, ident) is None:
            kind, ident = active_ref
        elif kind == "model" and active and ident != active["id"]:
            kind, ident = active_ref
        self._update_appearance(kind, ident)
        self._update_ligand_panel()  # markers/maps may have changed
        self._refresh_edits_list()   # active model or its edits may have changed

    def _on_tree_current_changed(self, current, _previous) -> None:
        if self._suppress_model_events or current is None:
            return
        from PySide6.QtCore import Qt, QTimer

        kind, ident = current.data(0, Qt.ItemDataRole.UserRole)
        if kind == "group":
            self._update_appearance()  # a group header has nothing to edit
            return
        self._update_appearance(kind, ident)  # master -> detail; touches no tree item
        if kind == "model":
            # Activating rebuilds the tree, and this signal runs inside the tree's own
            # selection handling — rebuilding here would free the item Qt is still using
            # (see _on_tree_item_changed). Defer past the signal.
            QTimer.singleShot(0, lambda: self._desktop.set_active_model(ident))

    def _make_type_combo(self, mid, types, hidden, *, interactions=False):
        """The model's authoritative Show menu: structure types plus Mol* interactions."""
        combo = _make_checkable_combo()
        combo.setToolTip(
            "Show or hide structure types and Mol*-computed interactions in this model.")
        for label in types:
            combo.add_checkable(label, label not in hidden, label)  # before on_change
        combo.add_checkable("Mol* interactions", interactions, "Mol* interactions")
        combo.on_change = lambda label, shown, d=mid: self._on_type_toggle(d, label, shown)
        return combo

    def _on_type_toggle(self, mid: str, label: str, shown: bool) -> None:
        if self._suppress_model_events:
            return
        if label == "Mol* interactions":
            self._desktop.set_model_interactions(mid, shown)
        else:
            self._desktop.set_model_type_hidden(mid, label, not shown)  # checked = shown

    def _sync_table_model_combo(self, model_items) -> None:
        """Rebuild the table's model dropdown, following the active model unless pinned."""
        active = next((m["id"] for m in model_items if m["active"]), None)
        ids = {m["id"] for m in model_items}
        if not self._table_pinned or self._table_model_id not in ids:
            self._table_pinned = False
            target = active
        else:
            target = self._table_model_id

        self._suppress_table_model_combo = True
        try:
            self._table_model_combo.clear()
            for m in model_items:
                self._table_model_combo.addItem(m["name"], m["id"])
            idx = next((i for i, m in enumerate(model_items) if m["id"] == target), -1)
            if idx >= 0:
                self._table_model_combo.setCurrentIndex(idx)
        finally:
            self._suppress_table_model_combo = False
        self._set_table_model(target)

    def _on_tree_item_changed(self, item, _column=0) -> None:
        """A visibility box was toggled -> apply it, but never from inside this signal.

        ``itemChanged`` is emitted *synchronously* from inside ``QTreeWidgetItem::setData``,
        which Qt is running from the tree's own mouse-release/edit stack. Applying the change
        here would run ``_emit_loaded_changed`` -> ``_on_loaded_changed`` -> ``tree.clear()``,
        destroying the very item whose ``setData`` is on the stack; Qt then keeps using that
        freed item as the stack unwinds and the process dies with SIGSEGV. That — not the
        GPU — is what every past "hiding segfaults" report actually was, which is why it
        reproduced identically on software and hardware WebGL and on three unrelated hide
        mechanisms.

        So read the plain values off the item now and do the work on the next event-loop
        turn, once Qt has finished with the item. Nothing here may touch the tree.
        """
        from PySide6.QtCore import Qt, QTimer

        if self._suppress_model_events:
            return
        kind, ident = item.data(0, Qt.ItemDataRole.UserRole)
        visible = item.checkState(0) == Qt.CheckState.Checked
        QTimer.singleShot(0, lambda: self._apply_visibility(kind, ident, visible))

    def _apply_visibility(self, kind: str, ident: str, visible: bool) -> None:
        """Apply a visibility toggle, off the tree's signal stack (see above)."""
        if kind == "model":
            self._desktop.set_model_visible(ident, visible)
        elif kind == "volume":
            self._desktop.set_volume_visible(ident, visible)
        elif kind == "marker":
            self._desktop.set_marker_visible(ident, visible)
        # reflections have no visibility to change

    def _on_tree_item_clicked(self, item, column: int) -> None:
        """A model's or map's visibility box is non-checkable on software (hiding either
        segfaults the renderer), so a click there does nothing — say why. Only the check
        column, and only a pure status flash: no viewer message, so it cannot itself
        crash."""
        from PySide6.QtCore import Qt

        if column != 0:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, _ident = data
        if kind in ("model", "volume") and not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            self._desktop._warn("Hiding needs hardware WebGL — not available on software "
                                "rendering.")

    def _on_active_radio(self, button) -> None:
        """A model's active radio was clicked -> make it the active model."""
        from PySide6.QtCore import QTimer

        if self._suppress_model_events:
            return
        mid = button.property("mid")
        if mid:
            # set_active_model refreshes the Loaded tree; _on_loaded_changed then points
            # Appearance at the newly active model (a focused model tracks the active one).
            # The radio lives *in* the tree (setItemWidget), so that refresh deletes this
            # very button while it is still delivering its own click — defer past it.
            QTimer.singleShot(0, lambda: self._desktop.set_active_model(mid))

    def _on_remove_selected(self) -> None:
        from PySide6.QtCore import Qt

        item = self._loaded_tree.currentItem()
        if item is None:
            return
        kind, ident = item.data(0, Qt.ItemDataRole.UserRole)
        if kind == "model":
            self._desktop.remove_model(ident)
        elif kind == "volume":
            self._desktop.remove_volume(ident)
        elif kind == "reflections":
            self._desktop.remove_reflections(ident)
        elif kind == "marker":
            self._desktop.remove_marker(ident)
        elif kind == "group":
            self._desktop.remove_group(ident)

    def _on_table_selection_changed(self) -> None:
        if not self._suppress_table_sync:
            self._table_sync_timer.start()  # debounce a drag-select

    def _push_table_selection_to_viewer(self) -> None:
        rows = [idx.row() for idx in self._atom_view.selectionModel().selectedRows()]
        atoms = [self._atom_model.row_atom(r) for r in rows]
        self._desktop.highlight_atoms_in(self._table_model_id, atoms)
        self._desktop.focus_atoms_in(self._table_model_id, atoms)

    def _select_table_rows(self, indices) -> None:
        """Select the given atom rows in the table without echoing back to the viewer."""
        from PySide6.QtCore import QItemSelection, QItemSelectionModel

        model = self._atom_model
        view = self._atom_view
        sm = view.selectionModel()
        with self._table_sync_suppressed():
            sm.clearSelection()
            ncols = model.columnCount()
            # Map atom indices to view rows (identity unless the table is filtered).
            rows = sorted(r for r in (model.atom_row(int(i)) for i in indices) if r >= 0)
            if rows and ncols:
                selection = QItemSelection()
                last = ncols - 1
                for start, end in _runs(rows):  # contiguous ranges keep this cheap
                    selection.select(model.index(start, 0), model.index(end, last))
                sm.select(selection, QItemSelectionModel.SelectionFlag.Select)
                view.scrollTo(model.index(rows[0], 0))


class DesktopApp:
    """Run the pxviewer desktop app with viewport and controls windows."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5173,
                 can_hide: bool = False):
        _check_qt()

        from PySide6.QtWidgets import QApplication

        self._host = host
        self._port = port
        # Whether objects can be hidden at all. Hiding recomposes the scene without the
        # hidden object and reloads the page — a clean context teardown that never disposes
        # anything mid-frame. A software renderer (SwiftShader) segfaults even on that, so
        # hiding is refused there and the tree checkboxes are non-checkable. run_desktop sets
        # this from the GL backend (hardware only). See pxviewer.gpu and [[gpu-webgl]].
        self._can_hide = bool(can_hide)

        # Qt must be initialized before any widgets are created.
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv[:1])
        # App identity. On Linux — Wayland especially — the launcher/taskbar finds an
        # app's icon by matching its running window to a .desktop file of this name;
        # setWindowIcon alone only covers the title bar. `pxviewer install-desktop-entry`
        # writes the matching file (StartupWMClass=pxviewer).
        self._app.setApplicationName("pxviewer")
        self._app.setApplicationDisplayName("pxviewer")
        self._app.setDesktopFileName("pxviewer")
        icon = _app_icon()  # dock/taskbar icon for the whole app
        if icon is not None:
            self._app.setWindowIcon(icon)
        # Before anything slow: the web engine and the Mol* bundle take seconds, and an
        # empty screen for that long looks like a launch that failed.
        self._splash = _show_splash()

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
        self._focus_surroundings = (
            str(self._settings.value("defaults/focus_surroundings", "true")).lower()
            in ("1", "true", "yes"))
        self._scene_counter = 0  # cache-buster for the composed volume MVSJ
        self._dummy: Optional[Any] = None  # persistent control ws when no model is visible
        self._batching = False  # defer viewport reload / signals during a group load
        # Scene-level selection: {model_id: [atom indices]}. Each model reports its
        # own picks independently (a selection may span models — e.g. protein +
        # ligand); the union across models is the scene selection. Mutated on the
        # WebSocket threads, read on the GUI thread, so guard it.
        self._scene_selection: dict = {}
        self._scene_lock = threading.Lock()
        # Serialises restraint builds. The pre-warm below runs on its own thread while a
        # drag can start on the tug worker, and both would otherwise process the same
        # model at once. Held only around the build, which is rare and per-model.
        self._restraints_lock = threading.Lock()
        # Labels of the long operations currently running, newest last — drives the busy
        # indicator (see run_background). A list, not a flag, so overlapping operations keep
        # it up until the last one finishes.
        self._busy_labels: list = []
        self._busy_lock = threading.Lock()
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
        self._sigint_installed = False
        self._sigint_timer = None
        self._minimize_stop = threading.Event()  # set to halt a running minimization
        # Set while no minimization is running. A drag waits on this before it builds
        # restraints on the model, so the minimizer's thread and the drag's never write the
        # same coordinates at once — the drag takes the model over from a clean stop.
        self._minimize_idle = threading.Event()
        self._minimize_idle.set()
        self._volume_scroll_target: Optional[str] = None  # volume the wheel contours
        # Dragging atoms: explicitly armed, one drag at a time (there is one pointer). Continuous
        # relaxation is on by default — a drag settles as a living motion, which reads better
        # than a nudge-and-stop; the checkbox in Settings mirrors this.
        self._tug_into_density = False
        self._tug_continuous = True
        # What a drag lets move (see tug.Tug): a sphere of a given radius, or a stretch of
        # residues (flank each side; 0 == single residue). Default is the sphere.
        self._tug_scope = {"mode": "sphere", "radius": 8.0, "flank": 0}
        self._tug: Any = None
        self._tug_model: Optional[str] = None
        self._tug_session: Any = None
        self._tug_last: Any = None
        self._tug_last_push: float = 0.0
        self._tug_queue: Any = None  # made with its worker on the first drag
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
        # The radius new maps from reflections open with (Settings changes it).
        self.view_radius_default: float = _VIEW_RADIUS_DEFAULT

        self.bridge = _make_bridge()
        # Workers marshal GUI-thread work (e.g. adding a model) via this signal;
        # emitted from another thread it dispatches as a queued call on the GUI thread.
        self.bridge.run_on_main.connect(lambda fn: fn())
        self.bridge.localres_shown.connect(self._on_localres_shown)
        self._viewport = ViewportWindow()
        self._controls = ControlsWindow(self)

        # One coherent window: the viewport fills it, the controls ride in a dock on the
        # right. Wayland forbids a client from positioning its own top-level windows, so
        # two side-by-side windows cannot be arranged — but a single maximized window with
        # a dock gives the same 2/3 + 1/3 default (see start()), and the dock's float
        # button pops the controls out to their own window for a second monitor (drag it
        # across, maximize the viewport). Re-dock with the float button or by dragging back.
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

        self._main = QMainWindow()
        self._main.setWindowTitle("pxviewer")
        if icon is not None:
            self._main.setWindowIcon(icon)
        self._main.setCentralWidget(self._viewport.widget())

        self._controls_dock = QDockWidget("Controls", self._main)
        self._controls_dock.setObjectName("pxviewer-controls")
        self._controls_dock.setWidget(self._controls.widget())
        # Movable + floatable (detach for a second screen). Not closable via the dock;
        # closing the *detached* window re-docks it instead, so it is never lost.
        self._controls_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        # No title bar when docked — the panel is obviously the controls, and the
        # status-row Dock/Detach button toggles it (an empty widget hides the title bar).
        # When floated, _reframe_dock promotes it to a normal Window so Wayland/Qt draw a
        # real frame (a bare floated dock has none).
        self._controls_dock.setTitleBarWidget(QWidget())
        self._reframing_dock = False
        self._controls_dock.topLevelChanged.connect(self._reframe_dock)
        self._controls_dock.topLevelChanged.connect(self._controls.reflect_dock_state)
        # "Is the app still up?" — a torn-down app answers no rather than raising, since
        # stop() drops the window and this filter can outlive it.
        self._dock_close_filter = _make_dock_close_filter(
            self._controls_dock,
            lambda: self._main is not None and self._main.isVisible())
        self._controls_dock.installEventFilter(self._dock_close_filter)
        self._main.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._controls_dock)

        # Closing the window quits the app; tear the backend down on the way out so
        # background threads stop before Qt destroys the widgets they signal.
        self._close_filter = _make_close_filter(self._app.quit)
        self._main.installEventFilter(self._close_filter)
        self._app.aboutToQuit.connect(self.stop)

        # Space / Shift+Space step the focused residue forward / back along its chain,
        # from either window.
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeySequence, QShortcut

        for _w in (self._viewport.widget(), self._controls.widget()):
            nxt = QShortcut(QKeySequence(Qt.Key.Key_Space), _w)
            nxt.setContext(Qt.ShortcutContext.WindowShortcut)
            nxt.activated.connect(lambda: self.advance_residue(1))
            prv = QShortcut(QKeySequence("Shift+Space"), _w)
            prv.setContext(Qt.ShortcutContext.WindowShortcut)
            prv.activated.connect(lambda: self.advance_residue(-1))

    # -- lifecycle -------------------------------------------------------

    def start(self) -> int:
        self._webapp.start()

        # Fill the screen. showMaximized() is enough on X11 and most compositors, but some
        # Wayland compositors drop a maximized state requested before the surface is mapped,
        # leaving the window at its default size. So also size it to the available screen
        # area (a size request Wayland does honor) as a floor, and re-assert the maximized
        # state once the window is really up — hence the delayed pass as well as the
        # immediate one. Re-applying is idempotent where the first request already took.
        screen = self._app.primaryScreen()
        if screen is not None:
            self._main.resize(screen.availableGeometry().size())
        self._main.showMaximized()

        from PySide6.QtCore import QTimer, Qt

        def _fill_screen() -> None:
            if not (self._main.windowState() & Qt.WindowState.WindowMaximized):
                self._main.showMaximized()
            self._size_controls_dock()  # ~1/3, worked out from the now-known width

        QTimer.singleShot(0, _fill_screen)
        QTimer.singleShot(250, _fill_screen)  # after a slow compositor has mapped the surface

        # Land on an empty viewer: the main screen is "load a file", not a demo.
        self._reload_viewport()  # nothing loaded -> a dummy-backed blank viewer
        self._dismiss_splash()
        self._status(f"Ready — serving {self._webapp.url}")
        print(f"pxviewer desktop viewer running at {self._webapp.url}", flush=True)
        print("Press Ctrl-C (or close a window) to stop.", flush=True)

        self._install_sigint_handler()
        try:
            return self._app.exec()
        except KeyboardInterrupt:  # a Ctrl-C that raced the handler being installed
            return 0
        finally:
            self._restore_sigint_handler()

    def _dismiss_splash(self) -> None:
        """Take the splash down once the viewport has really loaded.

        Tied to the page load rather than to the windows appearing: the window exists
        long before Mol* is up, and closing on that would just move the blank wait.
        """
        splash = getattr(self, "_splash", None)
        if splash is None:
            return
        self._splash = None

        def finished(_ok=True):
            # The page can finish loading after stop() has already dropped the window and
            # the splash with it; there is then nothing left to dismiss.
            if self._main is not None:
                splash.finish(self._main)

        view = getattr(self._viewport, "_view", None)
        if view is None:
            finished()
            return
        view.loadFinished.connect(finished)
        # ...but never leave it up if the page never reports back.
        from PySide6.QtCore import QTimer

        QTimer.singleShot(_SPLASH_MAX_MS, finished)

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
        try:
            self._controls.shutdown_console()
        except Exception:  # pragma: no cover - defensive
            pass
        self.stop_demo()
        self._clear_all()  # stops all model sessions, volumes, and the dummy
        self._webapp.stop()
        self._viewport.close()  # release the QtWebEngine render process (see close())
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

    def _size_controls_dock(self) -> None:
        """Give the controls dock ~1/3 of the width, the viewport the rest.

        Run after the window is up (a queued call) so the maximized width is known; the
        separator stays draggable, so this is only the starting proportion.
        """
        from PySide6.QtCore import Qt

        if self._main is None:      # queued, so it can arrive after stop()
            return
        width = self._main.width() or 1600
        self._main.resizeDocks(
            [self._controls_dock], [max(320, width // 3)], Qt.Orientation.Horizontal)

    def toggle_controls_dock(self) -> None:
        """Detach the controls to their own window, or re-dock them. Bound to the
        always-visible Dock/Detach button, so it works from either state — including when
        the floated window's native frame provides no re-dock control."""
        dock = self._controls_dock
        dock.setFloating(not dock.isFloating())

    def _reframe_dock(self, floating: bool) -> None:
        """Give the detached controls a native window frame.

        A floated ``QDockWidget`` is a decorationless Tool window on Wayland — nothing to
        grab. Promoting it to a normal ``Qt.Window`` gets a real compositor/Qt title bar.
        Docked, the dock has no title bar at all; re-docking (via the Dock button or the
        native close) restores that. Reentrancy-guarded, since ``setWindowFlags`` hides and
        reshows the window.
        """
        if not floating or self._reframing_dock:
            return
        from PySide6.QtCore import Qt

        dock = self._controls_dock
        self._reframing_dock = True
        try:
            dock.setWindowTitle("pxviewer — Controls")
            dock.setWindowFlags(Qt.WindowType.Window)
            dock.show()  # setWindowFlags hid it
        finally:
            self._reframing_dock = False

    # -- live session ----------------------------------------------------

    # -- registry (models + volumes + groups) ----------------------------

    def _model_entry(self, mid):
        return next((m for m in self._models if m["id"] == mid), None)

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
        for key in ("analysis", "validation", "hotspots"):
            entry.pop(key, None)

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

    def _volume_entry(self, vid):
        return next((v for v in self._volumes if v["id"] == vid), None)

    def _reflection_entry(self, rid):
        return next((r for r in self._reflections if r["id"] == rid), None)

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
        return models, volumes

    def alignable(self) -> list:
        """Existing ``(model, volume)`` pairs that can be density-aligned in place."""
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
                    pairs.append((model, volume))
        return pairs

    @staticmethod
    def _detect_map_model_shift(model: Any, map_data: Any,
                                initial_shift: Any = None) -> tuple:
        """Find a density-supported translation on a copy, leaving ``model`` untouched.

        cctbx's ``translation_search`` changes its input sites. Running successive coarse and
        fine passes lets the equal-step grid search accumulate a general XYZ translation; the
        displacement of the copy is the one result we retain.
        """
        from mmtbx.refinement.real_space.rigid_body import translation_search

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
        cctbx relocates the model, and the map, into a common frame — a boxed map can
        move a model several angstrom — and that is a change to the data, not a label on
        it. Both objects move into a group holding the manager, which is what makes them
        usable together (minimizing into density, and whatever joint work comes later).
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
                mmm.model(), mmm.map_manager().map_data(),
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
            mmm.model(), mm.map_data(), initial_shift=external_hint)
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

        A pinned local-resolution map is left out entirely. It is a colour source, never a
        surface: it exists to colour the map above it, and drawing its own isosurface
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
            nodes.append(Volume(
                url=v["map_url"], ref=v["ref"], format="map",
                # Maps contour in sigma, which is what makes one slider range serve every
                # map. A severity field is the exception: its levels are calibrated (1.0 *is*
                # the outlier cut), so it must contour on the absolute value or the level
                # would mean something different for every structure.
                isosurface_kind=v.get("iso_kind", "relative"), isosurface_value=v["iso"],
                color=v["color"], negative_color=v.get("negative_color"),
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
                             domain=attribute["domain"], on=on)
            # The attribute API replaces the representation list. Additional default layers
            # remain visible with their ordinary coloring; the primary layer carries the
            # computed scalar coloring.
            for extra in reps[1:]:
                kwargs = self._model_color_kwargs(entry, extra)
                if on is not None:
                    kwargs["on"] = on
                session.add_representation(extra, **kwargs)
            return
        for i, layer in enumerate(reps):
            kwargs = self._model_color_kwargs(entry, layer)
            if on is not None:
                kwargs["on"] = on
            method = session.set_representation if i == 0 else session.add_representation
            method(layer, **kwargs)

    def _model_color_kwargs(self, entry, rep: str) -> dict:
        """How to color a model's representation: an explicit user color wins; else the
        palette default (carbon-tint for atoms, uniform for ribbons); else the theme default."""
        explicit = entry.get("color")
        if explicit in _ATTRIBUTE_COLORS:
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
        # Register this model's pick handler once (tagged with its id); the click
        # mode is what actually turns picking on/off. Registering here means a
        # selection can be built in any loaded model, not just the active one.
        session.on_selection(lambda sel, mid=mid: self._on_model_selection(mid, sel))
        # Volume commands ride whichever session is the control session, so contour
        # changes made in the viewport can come back on any of them.
        session.on_volume_iso(self._on_volume_iso_changed)
        session.on_localres_shown(self.bridge.localres_shown.emit)
        session.on_tug(lambda action, atom, target, mid=mid: self._on_tug(mid, action, atom, target))
        # Markers are a scene-level thing, not this model's — but only the armed (control)
        # session's viewport reports one, so wiring every session is harmless.
        session.on_marker(lambda position, atom: self._on_marker(position, atom))
        session.on_marker_move(lambda mid_, pos, final: self._on_marker_move(mid_, pos, final))
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
        entry = self._model_entry(mid if mid is not None else self._active_model_id)
        if entry is None:
            return
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

                with self._restraints_lock:
                    if not model.restraints_manager_available():
                        edits.build_restraints(model)
                entry.pop("restraints_error", None)
            except Exception as exc:  # pragma: no cover - cctbx/runtime errors
                # Recorded rather than swallowed. The warm runs on a thread and cannot ask
                # the user anything, but a later minimize or drag can -- and asking is only
                # cheap if it already knows a build failed, since finding out otherwise
                # costs a full interpretation pass on the GUI thread. See
                # ControlsWindow._offer_restraints_if_blocked.
                entry["restraints_error"] = str(exc)

        threading.Thread(target=work, name="pxviewer-restraints-warm", daemon=True).start()

    def set_model_representation(self, mid: str, rep: str) -> None:
        """Change a model's representation type (from the inline dropdown)."""
        entry = self._model_entry(mid)
        if entry is None or entry.get("reps") == [rep]:
            return
        entry["rep"] = rep
        entry["reps"] = [rep]
        self._apply_model_rep(entry)

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
        """Show or hide a structure type (protein/water/…) on a model."""
        entry = self._model_entry(mid)
        if entry is None:
            return
        types = entry.setdefault("hidden_types", set())
        if (label in types) == bool(hidden):
            return
        types.add(label) if hidden else types.discard(label)
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
        if color == _QSCORE_COLOR:
            self.color_model_by_qscore(mid)
            return
        if color == _HOTSPOT_COLOR:
            self.color_model_by_hotspots(mid)
            return
        # Leaving a computed color drops the values it colored by: they belong to one model
        # (and for Q-score/hotspots, to one pairing with a map), not to the entry forever.
        entry.pop("attribute", None)
        self._apply_model_rep(entry)

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
                current["attribute"] = {"name": _QSCORE_COLOR, "values": values,
                                        "domain": DOMAIN, "palette": PALETTE}
                self._apply_model_rep(current)
                finite = values[np.isfinite(values)]
                if finite.size:
                    self._status(f"Q-score for {current['name']}: mean {finite.mean():.2f} "
                                 f"over {finite.size} atoms (red poor, green good)")
                else:  # pragma: no cover - defensive
                    self._status("Q-score produced no values")

            self.bridge.run_on_main.emit(apply_on_main)

        self.run_background(work, name="pxviewer-qscore", label="Computing Q-score")

    # -- validation hotspots ---------------------------------------------

    def compute_hotspots(self, mid: Optional[str] = None) -> None:
        """Aggregate the validation metrics into one per-atom severity field, and color by it.

        Hotspots are geometry-only: Ramachandran, rotamer, and clash severity. Runs on a
        thread because probe2 plus the two mmtbx validators takes seconds.
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
                current["hotspot_palette"] = palette  # kept so a menu re-apply reuses it
                current["color"] = _HOTSPOT_COLOR
                current["attribute"] = {
                    "name": _HOTSPOT_COLOR, "values": result.values,
                    "residue_values": residue_values,
                    "domain": hotspots.DOMAIN, "palette": palette,
                }
                self._apply_model_rep(current)
                self._emit_loaded_changed()
                self._status(f"{name}: {result.summary}")
                self.bridge.hotspots_ready.emit((mid, result, columns, rows))

            self.bridge.run_on_main.emit(apply_on_main)
            # Finding hotspots already ran Ramachandran and rotamers; spend the little extra to
            # run the remaining validators (reusing that shared analysis) so the Validation tab
            # is populated too — the user asked for one, they get both.
            self._populate_validation_from_analysis(mid, model, analysis)

        self.run_background(work, name="pxviewer-hotspots", label="Finding hotspots")

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
        entry["attribute"] = {
            "name": _HOTSPOT_COLOR, "values": result.values,
            "residue_values": (hotspots.residue_broadcast(model, result.values)
                               if model is not None else None),
            "domain": hotspots.DOMAIN, "palette": palette,
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
        """Forget a computed severity score, and stop colouring the model by it.

        Severity and concern are not convertible, so an imported field must not leave a
        severity table, a severity-coloured model, or a severity-scaled slider behind it —
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
        """Switch among the imported fields without re-reading any file."""
        entry = self._model_entry(mid or self._active_model_id)
        imported = entry.get("concern") if entry else None
        if imported is None or metric not in imported.fields:
            raise ValueError(f"concern field is not available: {metric}")
        entry["concern_metric"] = metric
        style = "contour" if entry.get("hotspot_volume") is not None else "cloud"
        self.show_hotspot_field(entry["id"], on=True, style=style)
        self._emit_concern_table(entry)

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
        a fixed domain, so a colour means the same thing in every structure and metric.
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
            field, spacing, origin = hotspots.severity_field(
                model, result.values, spacing=grid_spacing)
            threshold = float(getattr(self, "_hotspot_knee", hotspots.FIELD_ISO))
            units, label = "severity", "severity"
        if not (field >= threshold).any():
            self._status(f"nothing reaches {units} {threshold:.2f} — nothing to draw in 3-D")
            return

        if style == "cloud":
            # A value-colored raymarched cloud, streamed to the model's own viewer as a
            # direct-volume (MVS has no such node, so it cannot go through the shared scene).
            if imported is not None:
                from .volume_io import encode_hotspot_concern

                # Tell the viewer where the contract puts each colour, rather than letting it
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
        # The shell's colour is the colour the density shows at that same level — one scale,
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

    def measure_selection(self, kind: str) -> str:
        """Draw a distance/angle/dihedral from the active model's selected atoms."""
        session = self.active_model_session()
        if session is None:
            raise ValueError("load a model first")
        need = self._MEASURE_ARITY[kind]
        with self._scene_lock:
            atoms = list(self._scene_selection.get(self._active_model_id, []))
        if len(atoms) != need:
            raise ValueError(f"select exactly {need} atoms for a {kind} (have {len(atoms)})")
        self.ensure_atoms_shown()  # the measurement marks atoms — show them under the ribbon
        if kind == "distance":
            session.add_distance(atoms[0], atoms[1])
        elif kind == "angle":
            session.add_angle(atoms[0], atoms[1], atoms[2])
        else:
            session.add_dihedral(atoms[0], atoms[1], atoms[2], atoms[3])
        return f"drew {kind} on {need} atoms"

    def clear_measurements(self) -> None:
        session = self.active_model_session()
        if session is not None:
            session.clear_primitives()

    def analyze_clashes(self) -> None:
        """Add hydrogens to the active model (reduce2), register the result as a new
        object, hide the original, and draw probe2 contacts + clashes as two
        independently toggleable overlays.

        With real hydrogens probe2 decides overlaps from actual H positions and
        directionality — the MolProbity-approved path — so no heavy-atom heuristics
        are needed. reduce2 + probe2 are slow, so this runs on a background thread;
        adding the model object is marshalled back to the GUI thread.
        """
        entry = self._model_entry(self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        model = getattr(entry["session"], "model", None)
        if model is None:
            raise ValueError("the active object has no cctbx model")
        name, src_mid = entry["name"], entry["id"]
        # reduce2 and probe are the two expensive steps in the whole validation stack, and the
        # hotspot score needs the same two. Go through the shared analysis so whichever feature
        # runs first pays and the other is nearly free.
        analysis = self._model_analysis(entry)

        def work():
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

        self.run_background(work, name="pxviewer-reduce2", label="Adding hydrogens and running probe")
        self._status("adding hydrogens with reduce2…")

    def set_probe_channel(self, channel: int, visible: bool) -> None:
        """Toggle a probe overlay (contacts/clashes) on the active model, redrawing
        from the dots cached by the last analysis (no probe re-run)."""
        entry = self._model_entry(self._active_model_id)
        if entry is None:
            return
        session = entry["session"]
        dots = (entry.get("probe_dots") or {}).get(channel)
        if visible and dots:
            session.show_probe_dots(dots, channel=channel)
        else:
            session.clear_probe_dots(channel=channel)

    def run_validation(self) -> None:
        """Run every registered MolProbity validator on the active model and hand the
        results to the Validation tab. Validators can be slow (they build restraints
        and run mmtbx analyses), so this runs on a background thread; the results are
        cached on the model entry and emitted to the GUI thread via ``validation_ready``.
        """
        entry = self._model_entry(self._active_model_id)
        if entry is None:
            raise ValueError("load a model first")
        model = getattr(entry["session"], "model", None)
        if model is None:
            raise ValueError("the active object has no cctbx model")
        mid, name = entry["id"], entry["name"]
        analysis = self._model_analysis(entry)  # reused by a later hotspot score

        def work():
            from . import validation

            try:
                self._status(f"validating {name}…")
                results = validation.run_all(model, analysis)
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
            # Fill the Hotspots tab too, on the same shared analysis. Both features need
            # reduce2 and probe — the expensive part — so doing it here means the user pays
            # once whichever button they pressed, instead of again on the next tab.
            self._status(f"{name}: scoring hotspots from the same analysis…")
            self._populate_hotspots_from_analysis(mid, model, analysis)

        self.run_background(work, name="pxviewer-validation", label="Running validation")
        self._status("validating…")

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

    def set_tug_into_density(self, enabled: bool) -> None:
        """Whether a drag pulls into the map as well as against the geometry.

        Geometry alone can only deform; the map is what lets a drag *correct* something,
        since it is the only thing that knows where the atoms actually belong. Takes
        effect on the next drag — a drag already running keeps what it started with.
        """
        self._tug_into_density = bool(enabled)

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
                    session.show_map_box(box, level=3.0)
                    self._diff_boxes += 1
            except Exception as exc:  # pragma: no cover - cctbx/runtime errors
                self._status(f"live difference map failed: {exc}")
                self._diff_ctx = None  # stop hammering a failing recompute for this drag

    def _clear_live_diff(self) -> None:
        """Stop streaming the live difference map and remove the window from the viewport."""
        self._diff_ctx = None
        self._diff_atom = None
        self._diff_gen += 1  # invalidate any recompute still in flight
        session = self._tug_session
        if session is not None:
            try:
                session.clear_map_box()
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

    def _marker_entry(self, mid):
        return next((m for m in self._markers if m["id"] == mid), None)

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
            self._status("enter a monomer code, e.g. GOL")
            return
        if not ligands.available(code):
            self._status(f"no monomer '{code}' in the library")
            return
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
            self._status("enter a SMILES string, e.g. CCO")
            return
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
            try:
                self._status(f"building {code}…")
                model = builder(position, cs)
                if map_data is not None:
                    self._status(f"fitting {code} into density ({trials} trials)…")
                    ligands.fit_into_density(
                        model, map_data, resolution=3.0, number_of_trials=trials)
            except Exception as exc:  # pragma: no cover - cctbx/runtime errors
                self._status(f"could not place {code}: {exc}")
                return

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
            for action, mid, atom, target in _collapse_moves(waiting):
                try:
                    self._serve_tug(mid, action, atom, target)
                except Exception as exc:  # pragma: no cover - defensive
                    self._status(f"drag failed: {exc}")
                    self._end_tug()
            if self._tug is not None and self._tug_continuous:
                self._tug_relax()

    def _serve_tug(self, mid: str, action: str, atom: int, target) -> None:
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
            try:
                # Against the pre-warm (see _warm_restraints): if one is in flight for this
                # model, wait for it rather than building the same thing alongside it.
                scope = self._tug_scope
                selection = None
                if scope["mode"] == "selection":
                    with self._scene_lock:
                        selection = list(self._scene_selection.get(mid, []))
                    if not selection:
                        self._status("select atoms first to drag by selection")
                        self._tug = None
                        return
                with self._restraints_lock:
                    self._tug = Tug(
                        model, atom,
                        mode=scope["mode"], radius=scope["radius"], flank=scope["flank"],
                        selection=selection,
                        map_data=self.map_for_model(mid) if self._tug_into_density else None)
            except Exception as exc:  # pragma: no cover - restraints/runtime errors
                self._status(f"could not start dragging: {exc}")
                self._tug = None
                return
            self._tug_model = mid
            self._tug_session = session
            self._tug_last = None
            self._tug_last_push = 0.0
            self._maybe_start_live_diff(mid, atom)  # arm the live difference map, if on
            self._status(f"dragging atom {atom} — {self._tug.zone_size} atoms giving way")
            return

        if self._tug is None or self._tug_model != mid:
            return
        if action == "move" and target is not None:
            if self._tug_continuous:
                self._tug.set_target(target)  # the free-run loop does the stepping
            else:
                self._push_tug(self._tug.move_to(target))
        elif action == "end":
            self._settle_tug()   # let go, and watch it come to rest
            self._clear_live_diff()  # remove the live window (while the session is still known)
            self._end_tug()
            self._invalidate_model_state(entry)  # stale: the atoms just moved
            self._refresh_validation_staleness()  # warn if this outran a validation run

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

    def _settle_tug(self) -> None:
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
            trajectory: list = []
            self._tug.settle(on_frame=lambda c: trajectory.append(c.copy()))
        except Exception as exc:  # pragma: no cover - runtime errors
            self._status(f"settle failed: {exc}")
            return
        if not trajectory:
            return
        # The minimization converges in a fraction of a second — far too fast to see. Play
        # it back in real time so the fling visibly winds down to rest, thinning the many
        # optimizer states to what shows at the frame rate. A new grab aborts it.
        shown = min(len(trajectory), max(1, int(_TUG_SETTLE_DURATION / _TUG_PUSH_INTERVAL)))
        for i in np.linspace(0, len(trajectory) - 1, shown).astype(int):
            if self._tug_queue is not None and not self._tug_queue.empty():
                break  # the user grabbed again; do not make them wait out the wind-down
            self._push_tug(trajectory[i], force=True)
            time.sleep(_TUG_PUSH_INTERVAL)
        self._push_tug(trajectory[-1], force=True)  # the resting position, always shown

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
        self._queue_live_diff(coords)  # follow the drag with a local difference map, if on

    def _end_tug(self) -> None:
        """Close out a drag, if one is running. Idempotent."""
        if self._tug is None:
            return
        try:
            self._tug.finish()
        except Exception:  # pragma: no cover - defensive
            pass
        self._tug = None
        self._tug_model = None
        self._tug_session = None
        self._tug_last = None

    def map_for_model(self, mid: Optional[str] = None) -> Any:
        """The map this model is paired with, or None.

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

        def work():
            from .geometry import monomer_library_available
            from .minimize import minimize

            if not monomer_library_available():
                self._status("minimization needs the monomer library (set MMTBX_CCP4_MONOMER_LIB)")
                self._minimize_idle.set()
                self.bridge.minimizing_changed.emit(False)
                return
            first = last = None
            shown = 0
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
                        self._status(
                            f"{name} refined (bond rmsd {stats['bonds_after']:.3f}) — holding; "
                            "press Stop or enable Refine drag to finish")
                        while not self._minimize_stop.is_set():
                            time.sleep(0.1)
                        break
            except Exception as exc:  # pragma: no cover - restraints/runtime errors
                self._status(f"minimization failed: {exc}")
                return
            finally:
                self._minimize_idle.set()  # the model is the drag's to take now
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
            # were. Once per run, never per step — each update is two transforms.
            reflections = self.reflections_for_model(entry["id"])
            if reflections is not None:
                self.bridge.run_on_main.emit(
                    lambda rid=reflections["id"]: self._update_maps_if_live(rid))

        self.bridge.minimizing_changed.emit(True)
        self.run_background(work, name="pxviewer-minimize", label="Minimizing")

    def stop_minimization(self) -> None:
        """Halt a running minimization at its next step.

        The model keeps the progress made so far — a stopped run is a shorter run, not
        a discarded one.
        """
        self._minimize_stop.set()
        self._status("stopping minimization…")

    def reset_view(self) -> None:
        """Reframe the viewport camera to fit the whole scene."""
        self._focused_residue = None  # space-bar nav restarts from the top after a reset
        control = self._control_session()
        if control is not None:
            control.reset_view()

    # -- performance debugging -------------------------------------------

    def set_perf_prefs(self, **prefs) -> None:
        """Push debug render overrides (overlay, occlusionOff, pixelScale) to the viewer.

        Canvas-global — sent on whichever session the viewport is connected to.
        """
        control = self._control_session()
        if control is not None:
            control.set_perf_prefs(**prefs)

    def start_perf_capture(self) -> None:
        """Begin recording per-frame drag timings in the viewer (see stop_perf_capture)."""
        page = getattr(self._viewport, "_view", None)
        if page is not None:
            page.page().runJavaScript(
                "window.__pxviewerPerf && window.__pxviewerPerf.startCapture()")
        self._status("performance capture started — drag now, then stop to save the log")

    def stop_perf_capture(self, on_saved=None) -> None:
        """Stop the capture, write the log to a file, and hand its path to ``on_saved``.

        The samples live in the page, so they are read back through the webview rather than
        the socket; the file lands next to the app's other scratch output.
        """
        import datetime
        import json as _json
        import tempfile

        page = getattr(self._viewport, "_view", None)
        if page is None:
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

        page.page().runJavaScript(
            "window.__pxviewerPerf ? window.__pxviewerPerf.stopCapture() : ''", _write)

    def write_object(self, kind: str, ident: str, path: str) -> None:
        """Write a loaded object to disk: the model's cctbx coordinates, or the map.

        This writes what the DataManager holds (the model's own coordinates), not
        anything from the viewer — the same bytes cctbx would round-trip.
        """
        p = str(path)
        if kind == "model":
            entry = self._model_entry(ident)
            model = entry["session"].model if entry else None
            if model is None:
                raise ValueError("no cctbx model to write")
            is_pdb = p.lower().endswith((".pdb", ".ent"))
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
            entry = self._volume_entry(ident)
            if entry is None:
                raise ValueError("no such volume")
            entry["data"].write_map(p)  # cctbx writes the map
        else:
            raise ValueError("nothing to write")

    def unknown_ligands(self, mid: str) -> list:
        """Residues in a model that cctbx cannot build restraints for.

        One unrecognised ligand costs the whole model its restraints -- pdb_interpretation
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
        except Exception:  # pragma: no cover - a model cctbx cannot interpret at all
            return []

    def generate_ligand_restraints(self, mid: str, codes=None) -> dict:
        """Write monomer dictionaries for a model's unknown ligands and rebuild.

        ``codes`` limits it to particular residue names; omit for all of them. Returns
        ``{code: smiles}`` for those generated -- the perceived chemistry, which is what
        the user needs to check, since a dictionary built from a badly modelled ligand
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
        """One ``{"kind", "summary"}`` per custom restraint edit carried on a model.

        A summary for display and counting. The edits themselves live on the model as
        cctbx's own PHIL scope -- reachable with ``pxviewer.edits.get_edits(model)`` --
        because that is what gets handed to pdb_interpretation.
        """
        entry = self._model_entry(mid)
        return _edit_summaries(entry.get("edits")) if entry else []

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

    def add_edit_from_selection(self, mid: str, kind: str, *, sigma: Optional[float] = None) -> dict:
        """Author a bond/angle/dihedral edit from the active model's selected atoms (2/3/4),
        taking the current geometry as the target. Order follows the selection, same as the
        measurement tools. Raises if the wrong number of atoms is selected or cctbx rejects it."""
        from . import edits as edits_mod

        entry = self._model_entry(mid)
        if entry is None:
            raise ValueError("no such model")
        model = entry["session"].model
        if model is None:
            raise ValueError("the object has no cctbx model")
        need = edits_mod.KIND_ARITY.get(kind)
        if need is None:
            raise ValueError(f"cannot author a {kind} edit")
        with self._scene_lock:
            atoms = list(self._scene_selection.get(mid, []))
        if len(atoms) != need:
            raise ValueError(f"select exactly {need} atoms for a {kind} (have {len(atoms)})")
        sites = model.get_sites_cart().as_numpy_array()
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
        """Drop the edit at ``index`` and rebuild restraints without it."""
        entry = self._model_entry(mid)
        if entry is None:
            return
        from . import edits as edits_mod

        scope = self._edits_scope(entry)
        try:
            edits_mod.remove(scope, index)
        except IndexError:
            return
        self._apply_edits(entry, scope)

    def clear_edits(self, mid: str) -> None:
        """Drop every edit and rebuild the model's restraints without them."""
        from . import edits as edits_mod

        entry = self._model_entry(mid)
        if entry is None:
            return
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
        if entry is None:
            raise ValueError("no such model")
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
        scope = entry.get("edits") if entry else None
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

        A hidden map is parked at an empty contour (that is how it hides), so a level
        change is stored but not pushed — pushing it would bring the map back. It takes
        effect when the map is shown again."""
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
        if not entry["visible"]:
            return
        control = self._control_session()
        if control is not None:
            try:
                control.set_volume_iso(entry["ref"], self._iso_for_wire(entry, value))
            except Exception:  # pragma: no cover - defensive
                pass

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
        self._volume_command(vid, "opacity", float(value),
                             lambda c, ref, v: c.set_volume_opacity(ref, v))

    def set_volume_color(self, vid: str, color: str) -> None:
        """Set a volume's color live."""
        self._volume_command(vid, "color", color,
                             lambda c, ref, v: c.set_volume_color(ref, v))

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
        """A volume's current style/color/opacity/level.

        The Loaded summary is a snapshot taken when it was emitted, and these can change
        without one — from the console, or by the wheel in the viewport — so the
        Appearance pane reads them from the entry rather than trusting the snapshot.
        """
        entry = self._volume_entry(vid)
        if entry is None:
            return {}
        out = {key: entry.get(key)
               for key in ("style", "color", "opacity", "iso", "clip", "mask_radius",
                           "radius")}
        # The map's maximum on the sigma scale -- the level above which nothing is left.
        # The Level slider spans up to here, so its right end genuinely empties the map.
        try:
            stats = entry["data"].stats()
            out["max_sigma"] = (stats["max"] - stats["mean"]) / (stats["std"] or 1.0)
        except Exception:  # pragma: no cover - defensive (a map with no data)
            out["max_sigma"] = None
        return out

    def set_volume_clip(self, vid: str, front: float, back: float) -> None:
        """Clip a volume to a front/rear slab (see LiveSession.set_clip)."""
        entry = self._volume_entry(vid)
        clip = (float(front), float(back))
        if entry is None or entry.get("clip") == clip:
            return
        entry["clip"] = clip
        self._send_volume_clip(entry)

    def set_view_radius_default(self, radius: float) -> None:
        """How much density a map made from reflections opens with.

        Only what *new* maps get: a map already on screen has its own radius, which the
        user may have set, and reaching in to change it would be presumptuous.
        """
        self.view_radius_default = float(radius)

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
        """
        entry = self._volume_entry(vid)
        radius = None if radius is None else float(radius)
        if entry is None or entry.get("radius") == radius:
            return
        entry["radius"] = radius
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
            if entry.get("radius") is not None or entry.get("clip") != (0.0, 1.0):
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
                # Hidden maps, and the parked plain contour of a coloured map -- the
                # coloured surface represents that one, whatever the entry's own flag.
                try:
                    control.set_volume_visible(entry["ref"], False)
                except Exception:  # pragma: no cover - defensive
                    pass

    def _send_volume_clip(self, entry) -> None:
        """Push a volume's whole clip: the slab and the radius are one thing to the
        viewer, so a change to either re-sends both."""
        control = self._control_session()
        if control is None:
            return
        front, back = entry.get("clip") or (0.0, 1.0)
        try:
            control.set_clip(front, back, radius=entry.get("radius"), ref=entry["ref"])
        except Exception:  # pragma: no cover - defensive
            pass

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
            entry["session"].set_clip(front, back)
        except Exception:  # pragma: no cover - defensive
            pass

    def model_appearance(self, mid: str) -> dict:
        """A model's current clip slab (see :meth:`volume_appearance`)."""
        entry = self._model_entry(mid)
        return {} if entry is None else {"clip": entry.get("clip")}

    def set_volume_scroll_target(self, vid: Optional[str]) -> None:
        """Point the scroll wheel's contouring at a volume (None = nothing).

        The wheel adjusts whatever the Appearance pane's Level slider is showing, so
        this follows the focused object rather than the viewport picking for itself.
        (Coot's binding: in map work the contour level is what you reach for most.)
        """
        entry = self._volume_entry(vid) if vid else None
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
        self.bridge.volume_iso_changed.emit((entry["id"], float(value)))

    def set_active_model(self, mid: str) -> None:
        """Make a loaded model the active one (the atoms table + selection follow it)."""
        entry = self._model_entry(mid)
        if entry is None or self._active_model_id == mid:
            return
        self._active_model_id = mid
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
        was still using (see ``ControlsWindow._on_tree_item_changed``). Three unrelated hide
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
        return VolumeData.from_map_manager(masked, name=entry["data"].name)

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
            self._push_localres(entry)  # re-extract the coloured surface from the masked grid
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
                    radius: Optional[float] = None,
                    negative_color: Optional[str] = None,
                    iso_kind: str = "relative") -> str:
        """Register + show a volume: write its map (via cctbx) and compose the scene.

        ``color``/``iso`` override the defaults for maps that have a convention — a
        difference map is green at 3 sigma whatever color the palette is up to.
        ``radius`` limits drawing to near the view center (see :meth:`set_volume_radius`).
        ``negative_color`` draws a second contour at the negative of the level, which is
        how a difference map is read (see MAP_STYLE).
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
            "opacity": 1.0, "style": "surface", "clip": (0.0, 1.0), "mask_radius": None,
            "radius": radius, "negative_color": negative_color, "iso_kind": iso_kind,
        })
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
                # The coloured surface is this map's representation: the one checkbox
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
            parent = self._volume_entry(entry["pinned_to"])
            if parent is not None:
                if parent.get("color_by_resolution"):
                    self.set_color_by_resolution(parent["id"], False)
                parent.pop("resolution_map", None)
        self._volumes.remove(entry)
        self._prune_group(entry["group"])
        self._reload_viewport()
        self._emit_loaded_changed()

    # -- local-resolution surface colouring ------------------------------------

    def colorable_volumes(self) -> list:
        """Maps whose surface local resolution could colour — the real maps, not the hidden
        resolution maps pinned under them. ``(vid, name)`` pairs for a picker."""
        return [(v["id"], v["name"]) for v in self._volumes if not v.get("is_resolution")]

    def resolution_map_for(self, full_vid: str) -> Optional[str]:
        """The vid of the resolution map pinned to a full map, if one has been computed."""
        entry = self._volume_entry(full_vid)
        return entry.get("resolution_map") if entry else None

    def is_colored_by_resolution(self, full_vid: str) -> bool:
        """Whether a full map is currently drawn coloured by its resolution map."""
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
        loaded full map, then optionally colour the full map by it.

        The half-maps (``half1_path``/``half2_path``) are opened in cctbx only and never
        shown — they are inputs to the FSC, not surfaces. The computation runs on a
        background thread; the map is pinned and coloured back on the GUI thread.
        """
        full = self._volume_entry(full_vid)
        if full is None:
            raise ValueError("pick a map to colour")
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
        pipeline — the appearance slider (labelled σ), the scroll wheel, the live wire
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
            # calculation falls back to cctbx's estimate, i.e. the previous behaviour.
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
        (replacing any previous one), then optionally turn on colour-by-resolution."""
        full = self._volume_entry(full_vid)
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
            # A colour source, never a surface: is_resolution keeps it out of the drawn
            # scene altogether (_write_volume_scene) and out of the tree's visibility
            # checkboxes (_loaded_summary). visible=False is kept for the paths that
            # treat "hidden" generically (first-visible focus, render skips).
            res_entry["visible"] = False
        full["resolution_map"] = res_vid
        # Display resolution for the coloured surface: contour every nth voxel. x4 by
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
        entry = self._volume_entry(res_vid)
        if entry is None:
            return
        self._volumes.remove(entry)
        self._prune_group(entry.get("group"))

    def set_color_by_resolution(self, full_vid: str, on: bool) -> None:
        """Toggle colouring a full map by its pinned resolution map. A plain appearance
        option — on streams the value-coloured surface (and hides the uniform isosurface it
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
        """A colour-scale range for a local-resolution map: the 2nd–98th percentile of its
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

    def set_localres_downsample(self, full_vid: str, factor: int) -> None:
        """Set how finely a map's colour-by-resolution surface is contoured (1 = every
        voxel, n = every nth). A client-side rebuild from grids the browser holds."""
        entry = self._volume_entry(full_vid)
        if entry is None:
            return
        entry["localres_downsample"] = max(1, int(factor))
        session = self._control_session()
        if session is not None:
            session.set_localres_downsample(entry["localres_downsample"])
        self._emit_loaded_changed()  # the pane shows the current factor

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
        """The viewport drew the coloured surface: it is on screen and usable now."""
        self._end_localres_wait()
        for entry in self._volumes:
            if entry.get("color_by_resolution"):
                entry["localres_drawn"] = True
                self._status(f"{entry['name']}: local resolution ready — "
                             "the map is coloured by it")
        self._emit_loaded_changed()

    def set_localres_domain(self, full_vid: str, lo: float, hi: float) -> None:
        """Set the colour ramp's value range (Angstrom): ``lo`` maps to blue, ``hi`` to
        red. Manual and stable -- it never follows the contour on its own, so a figure's
        colours keep their meaning across thresholds and sessions."""
        entry = self._volume_entry(full_vid)
        if entry is None:
            return
        lo, hi = float(lo), float(hi)
        if not (hi > lo):
            self._warn("the colour range needs max above min")
            return
        entry["localres_domain"] = (lo, hi)
        session = self._control_session()
        if session is not None:
            session.set_localres_domain(lo, hi)
        self._emit_loaded_changed()

    def fit_localres_domain(self, full_vid: str) -> None:
        """Set the colour range from what the current contour actually shows.

        The full-map default spends most of the ramp on solvent-adjacent voxels no
        realistic threshold displays -- on EMD-53478 the whole visible particle sits in
        the blue end. This takes the 2nd-98th percentile of the resolution values inside
        the current contour (density >= the display level), so the ramp spans the values
        on screen. A deliberate action, never automatic: refitting on contour changes
        would re-colour a figure under its caption.
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
            self._warn("nothing visible at this level to fit the colour range to")
            return
        lo, hi = float(np.percentile(inside, 2)), float(np.percentile(inside, 98))
        if hi <= lo:
            hi = lo + 0.1
        self.set_localres_domain(full_vid, round(lo, 2), round(hi, 2))
        self._status(f"colour range fitted to the visible surface: {lo:.2f}–{hi:.2f} Å")

    def reset_localres_domain(self, full_vid: str) -> None:
        """Restore the default colour range: percentiles of the whole resolution map."""
        entry = self._volume_entry(full_vid)
        res = self._volume_entry(entry.get("resolution_map")) if entry else None
        if entry is None or res is None:
            return
        lo, hi = self._localres_domain(res["data"])
        self.set_localres_domain(full_vid, round(lo, 2), round(hi, 2))

    def _push_localres(self, full) -> None:
        """(Re)stream a full map's colour-by-resolution surface and hide its plain isosurface.

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
        # The stored domain, initialised once: recomputing per push would let the colour
        # mapping drift on its own, and a mapping that shifts under a figure between
        # sessions or contours is exactly what the explicit Colour range control forbids.
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
            # The coloured surface *is* the map now: the plain contour steps aside at the
            # ref level (the entry's own visible flag is the map's one checkbox and stays
            # what the user set), and the coloured surface takes that visibility over.
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
                        self._add_volume(
                            full_map, paths["map"].name,
                            **({"iso": iso_sigma} if iso_sigma is not None else {}))
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

    def _status(self, text: str) -> None:
        self.bridge.status_changed.emit(text)

    def _warn(self, text: str) -> None:
        """A status message that briefly stands out — for when an action was refused and
        the user needs to know why (not just a quiet nothing-happened)."""
        self.bridge.status_warned.emit(text)

    def _on_model_selection(self, mid: str, selection) -> None:
        """A model reported its picked atoms (WS thread). Fold into the scene selection."""
        with self._scene_lock:
            indices = list(selection.indices)
            if indices:
                self._scene_selection[mid] = indices
            else:
                self._scene_selection.pop(mid, None)
        self._emit_scene_selection()

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

    def _emit_scene_selection(self) -> None:
        with self._scene_lock:
            snapshot = {k: list(v) for k, v in self._scene_selection.items()}
        self.bridge.scene_selection_changed.emit(snapshot)

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
             "edits": _edit_summaries(m.get("edits"))}
            for m in self._models
        ] + [
            # One row per map. The resolution dataset is deliberately absent: it is a
            # colour source, not a drawable object -- its user-facing existence is the
            # "Colour by local resolution" group on its map's pane, and its lifecycle
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
        entry = self._model_entry(mid) if mid else None
        return entry["session"] if entry else None

    def active_model_session(self):
        """The active model's LiveSession, or None (e.g. a volume scene)."""
        return self.session_for(self._active_model_id)

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
                if self._model_entry(mid) is None or self._reflection_entry(rid) is None:
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
                        color, iso, negative = MAP_STYLE[is_diff]
                        # Difference maps keep green/red (Coot semantics); the 2mFo-DFc map
                        # takes a random color from the session's current palette group.
                        if not is_diff:
                            color = self._palettes.next_color()
                        self._add_volume(
                            VolumeData.from_map_manager(
                                mmm.get_map_manager_by_id(map_type),
                                name=map_type, map_id=map_type),
                            map_type, group=gid, color=color, iso=iso,
                            radius=self.view_radius_default, negative_color=negative)
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
                rentry["r_work"] = out["r_work"]
                rentry["r_free"] = out["r_free"]
                self._reload_viewport()
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
        entry = self._reflection_entry(rid)
        if entry is None or entry.get("r_work") is None:
            return
        mmm = self.group_mmm(entry.get("group"))
        if mmm is None or mmm.model() is None:
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
                color, iso, negative = MAP_STYLE[is_diff]
                if not is_diff:  # a random palette color; difference maps keep green/red
                    color = self._palettes.next_color()
                volume = VolumeData.from_map_manager(
                    map_from_coefficients(coefficients), name=root_label(label))
                # A map from reflections fills the unit cell: open it with a radius,
                # or the model is lost inside a wall of density.
                self._add_volume(volume, root_label(label), group=gid,
                                 color=color, iso=iso, radius=self.view_radius_default,
                                 negative_color=negative)
                made.append(root_label(label))
        self._status(f"Loaded {name} — {data.summary()}; maps: {', '.join(made)}")
        return "reflections"

    def _load_model_file(self, path: str) -> str:
        """Read a model with cctbx and add it to the viewport (alongside any others)."""
        self.stop_demo()
        self._reset_interactions()

        from .live import LiveSession

        session = LiveSession.from_model_file(path)  # _add_model applies the default rep
        self._add_model(session, Path(path).name)
        self._status(f"Loaded model: {Path(path).name} ({session._n_atoms} atoms)")
        return "model"

    def _load_volume_file(self, path: str) -> str:
        """Read a map with cctbx and add it as a volume (alongside any models/maps)."""
        self.stop_demo()
        self._reset_interactions()

        from .volume_io import VolumeData

        self._add_volume(VolumeData.from_map_file(path), Path(path).name)
        self._status(f"Loaded volume: {Path(path).name}")
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
        with self._batch_load():
            if model_data is not None and model_data.model is not None:
                session = self._model_session(model_data.model, group_name)
                self._add_model(session, group_name, group=gid)
            for vd in volumes:
                self._add_volume(vd, vd.name, group=gid)
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
        """Demo: the bundled sample model + a cctbx-generated density, as one group.

        The map is computed from the model (no large file to ship, no network), and
        because it comes back as a cctbx map_model_manager it loads as a real group.
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
        """Demo: the bundled model plus reflections computed from it.

        The point is to show the density-from-data path without shipping a real dataset:
        amplitudes (and free flags) are generated from the model and written to an MTZ,
        then the model and the reflections are loaded side by side — unpaired, so the
        Reflections pane offers "Make maps" and you can watch 2mFo-DFc and mFo-DFc get
        computed from them.
        """
        import os
        import tempfile

        self.stop_demo()
        self._reset_interactions()

        from .cctbx_io import read_model

        sample = sample_structure_path()
        if sample is None:
            raise FileNotFoundError("the bundled sample model is missing")

        f_calc = read_model(str(sample)).get_xray_structure().structure_factors(
            d_min=d_min).f_calc()
        f_obs = abs(f_calc).set_observation_type_xray_amplitude()
        f_obs = f_obs.customized_copy(sigmas=f_obs.data() * 0.05)  # plausible sigmas
        flags = f_obs.generate_r_free_flags(fraction=0.05)

        # A temp dir, not auto-cleaned: make_maps reads this file back, so it has to
        # outlive the load.
        out_dir = tempfile.mkdtemp(prefix="pxviewer-xray-demo-")
        stem = Path(SAMPLE_STRUCTURE[0]).stem
        mtz = os.path.join(out_dir, f"{stem}_data.mtz")
        dataset = f_obs.as_mtz_dataset(column_root_label="F")
        dataset.add_miller_array(flags, column_root_label="R-free-flags")
        dataset.mtz_object().write(mtz)

        # Not paired — pairing them is the demo — but loaded in one batch so the viewport
        # reloads once, and shown as one group so the pair reads as the unit it is.
        before = self._object_ids()
        with self._batch_load():
            self._load_model_file(str(sample))
            self._load_reflection_file(mtz)
            self._group_loaded_together(
                before, f"{Path(sample).name} + reflections", label="not yet phased")
        self._status(
            f"X-ray demo: {SAMPLE_STRUCTURE[0]} + reflections — open the reflections and "
            "click Make maps")
        return "xray"

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
        """Demo mirroring Phenix's cryo-EM real-space refinement: a model sitting slightly off
        a density map, to be refined back into it. A cryo-EM-resolution map is computed from
        the bundled model, then the model is *shaken* off it — so 'Minimize' with 'Use map'
        (gradient-driven real-space refinement, exactly what phenix.real_space_refine does)
        pulls it back into the density. Self-contained (no phenix, no dataset)."""
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
                self._add_volume(vd, f"{SAMPLE_STRUCTURE[0]} (cryo-EM density)", group=gid)
        self._status(
            "Cryo-EM demo: a shaken model in its density — Minimize with 'Use map' to "
            "real-space refine it back in")
        return "group"

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

    def clear_selection(self) -> None:
        for m in self._models:
            try:
                m["session"].clear_selection()
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
        radius = max(float(max(np.linalg.norm(v - ca) for v in named.values())) + 2.0, 4.0)
        return ca, up, direction, radius

    def focus_residue(self, chain: str, resid: str) -> None:
        """Select + focus a residue (by chain id and resid, MolProbity's resseq+icode
        string) on the active model — driven by a Validation table row or space-bar
        navigation. The residue is framed N->C left-to-right with its side chain up
        (falling back to a plain focus for non-amino-acids). The residue->atom-index
        map is built once from the model and cached on the model entry."""
        entry = self._model_entry(self._active_model_id)
        if entry is None:
            return
        model = getattr(entry["session"], "model", None)
        if model is None:
            return
        index = entry.get("_residue_index")
        if index is None:
            index = entry["_residue_index"] = self._build_residue_index(model)
        key = (chain.strip(), resid.strip())
        atoms = index.get(key)
        if not atoms:
            return
        self._focused_residue = key
        self.highlight_atoms_in(entry["id"], atoms)
        orient = self._residue_orientation(model, atoms)
        if orient is None:
            self.focus_atoms_in(entry["id"], atoms)
        else:
            entry["session"].orient_camera(*orient)

    def advance_residue(self, step: int = 1) -> None:
        """Move the focused residue to the next/previous one in its chain (space-bar
        navigation). With nothing focused yet, start at the first residue."""
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

        ``specs`` is a list of ``(kind, i_seqs)``. Every participating atom is highlighted —
        so you see exactly *which* atoms make up the restraint, not the whole residue — and
        bonds/angles/dihedrals also get their measurement notation drawn (the distance line,
        angle arc or dihedral fan). Chirality/planarity have no simple notation, so they show
        as the highlight alone. Multiple rows -> multiple. The camera frames them all.
        """
        session = self.session_for(mid)
        self._clear_restraint_notations()
        if session is None:
            return
        if specs:
            self.ensure_atoms_shown(mid)  # a ribbon can't show the atoms this notation marks
        self._restraint_prim_session = session
        highlight: set = set()
        focus_atoms: set = set()
        for i, (kind, iseqs) in enumerate(specs):
            pid = f"geomsel-{i}"
            iseqs = list(iseqs)
            focus_atoms.update(iseqs)
            highlight.update(iseqs)  # mark every atom in the restraint, whatever the kind
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
        try:  # (empty list clears the overlay)
            session.highlight(sorted(highlight))
        except Exception:  # pragma: no cover - defensive
            pass
        if focus_atoms:  # aim the camera at the selected restraint's atoms
            try:
                session.focus(sorted(focus_atoms))
            except Exception:  # pragma: no cover - defensive
                pass

    def select_by_expression(self, text: str) -> int:
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
        if not text:
            session.clear_selection()
            with self._scene_lock:
                dropped = self._scene_selection.pop(mid, None) is not None
            if dropped:
                self._emit_scene_selection()
            return 0
        sel = session.select_by(selection=text)  # cctbx; raises on invalid syntax
        session.highlight(sel)                    # show it in the viewer
        self._on_model_selection(mid, sel)        # feed the scene selection (table + label)
        return len(sel)


def run_desktop(host: str = "127.0.0.1", port: int = 5173,
                gpu: Optional[str] = None) -> int:
    """Start the desktop app with viewport and controls windows."""
    from . import gpu as gpu_backend

    # Before any QtWebEngine/QApplication exists: choose the GL backend. On a machine
    # whose GPU cannot provide WebGL (common on VMs) this arms a one-time restart into
    # software rendering rather than leaving the viewport blank. See pxviewer.gpu.
    mode = gpu_backend.configure(gpu)
    _check_qt()

    # Hide/show in place only on hardware WebGL: it is a live GPU state change, which a
    # software renderer (SwiftShader) segfaults on. Software and user-custom flags fall
    # back to the reload-based hide, which is slower to the eye but never disposes
    # anything mid-frame. See DesktopApp.__init__.
    desktop = DesktopApp(host=host, port=port, can_hide=(mode == "hardware"))
    try:
        return desktop.start()
    finally:
        desktop.stop()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(run_desktop())
