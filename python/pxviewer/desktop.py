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
from .palettes import suggested_colors
from .webapp import Webapp

# Shared with the core (single definition lives in appcore).
from .appcore import (
    AppCore,
    _CC_COLOR,
    _DRAG_MAP_UPDATE_MS,
    _HOTSPOT_COLOR,
    _MARKER_CHANNEL,
    _MARKER_COLOR,
    _MARKER_RADIUS,
    _MODEL_REP_OPTIONS,
    _MODEL_VALUE_COLORS,
    _QSCORE_COLOR,
    _STRUCTURE_TYPE_ORDER,
    _TUG_PUSH_INTERVAL,
    _TUG_SETTLE_DURATION,
    _VALUE_PALETTE,
    _make_bridge,
    _rep_shows_atoms,
)

# The swatches offered when coloring an object by hand — hex, drawn from the bundled
# palettes (see palettes.suggested_colors) so a hand-picked color comes from the same
# inventory as the ones objects open in.
_VOLUME_COLORS = suggested_colors()
# Sentinel for the "Custom…" entry in a color dropdown (never a real color value).
_CUSTOM_COLOR = "\x00custom"



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




# How much density to draw around the view center, for maps that need it. A map made
# from reflections fills the unit cell, so drawing all of it buries the model — those
# open with a radius. A map read from a file is already a box around its subject, so it
# does not. (Coot applies its radius to every map; ours can tell the two apart.)
#
# 15 A is a starting point rather than a considered convention — Coot's own default is
# not something we confirmed — so it is the app's default and adjustable in Settings,
# not a constant. Per-map, the Appearance pane has always been able to change it.
_VIEW_RADIUS_DEFAULT = 15.0


# The object list sizes itself to its contents between these. The floor keeps the empty
# state from collapsing to nothing; past the ceiling the list scrolls itself rather than
# taking the whole pane. Roomier now that the mouse reference moved to a popup (its old
# space is the list's), so more objects show at once before it has to scroll.
_TREE_MIN_HEIGHT = 150
_TREE_MAX_HEIGHT = 440

# "Nothing cached yet" for the Appearance pane's rebuild guard — distinct from any real
# signature, including the ``(None, None, None)`` of the empty state.
_UNSET = object()

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
    # The refined per-atom numbers, colored through the attribute path rather than
    # Mol*'s fixed 'uncertainty'/'occupancy' themes: the attribute theme takes an
    # explicit domain and palette, which is what gives these a user-settable Range
    # (shared with the map's local-resolution scale) where the built-in themes have
    # a hard-wired one.
    ("By B-factor", "bfactor"),
    ("By occupancy", "occupancy"),
    # Not a Mol* theme: computed against the map by cctbx and sent as per-atom values.
    # See DesktopApp.color_model_by_qscore.
    ("By Q-score (fit to map)", _QSCORE_COLOR),
    # See DesktopApp.color_model_by_mapcc.
    ("By map-model CC", _CC_COLOR),
    # Likewise computed, but only re-applies a field the Hotspots tab already produced.
    ("By hotspot severity", _HOTSPOT_COLOR),
]














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


def _set_macos_process_name(name: str = "pxviewer") -> None:
    """Rename the process so the macOS menubar app menu and Dock label read
    ``pxviewer`` rather than ``python``.

    Qt's ``applicationName``/``displayName`` cover the window title and about
    box, but the bold app-menu name (and the Dock label for an unbundled
    process) come from ``NSProcessInfo.processName``, which for a bare
    interpreter is ``python``. Renamed via the Objective-C runtime (no pyobjc
    dependency) right after QApplication exists, before the menubar is
    assembled. A no-op off darwin or if the runtime is unavailable.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        libobjc = ctypes.CDLL(ctypes.util.find_library("libobjc"))
        sel = libobjc.sel_registerName
        sel.restype = ctypes.c_void_p
        sel.argtypes = [ctypes.c_char_p]

        def send(obj, selector: bytes, *args):
            """objc_msgSend with an explicit per-call prototype — passing
            varargs beyond ``argtypes`` crashes on ARM64, so each call declares
            the full signature."""
            fn = libobjc.objc_msgSend
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [
                type(a) for a in args]
            return fn(ctypes.c_void_p(obj), sel(selector), *args)

        get_class = libobjc.objc_getClass
        get_class.restype = ctypes.c_void_p
        get_class.argtypes = [ctypes.c_char_p]

        info = send(get_class(b"NSProcessInfo"), b"processInfo")
        ns_name = send(get_class(b"NSString"), b"stringWithUTF8String:",
                       ctypes.c_char_p(name.encode("utf-8")))
        send(info, b"setProcessName:", ctypes.c_void_p(ns_name))
    except Exception:
        pass






_HIGHLIGHT_OVERLAY_CLASS = None


def _highlight_overlay_class():
    """A transparent, click-through widget that paints a rounded ring at a set alpha — used
    to emphasize a button *on top*, so it never touches the button's size or the layout
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


def _line_icon(name: str, color, size: int = 20, selected_color=None):
    """A monochrome line SVG (``<name>.svg``) tinted to ``color`` as a QIcon.

    Looked up in ``assets/icons_custom`` (our own icons) first, then ``assets/icons`` (the
    Lucide set), so a custom icon can also override a stock name. The icons draw with
    ``stroke="currentColor"``, which Qt's SVG renderer does not resolve on its own, so the
    color is substituted in before rendering. Rendered at 3x the display size so it stays
    crisp on a HiDPI screen. Returns None if the asset is missing.

    ``selected_color`` adds a QIcon Selected-mode rendering — item views use it
    automatically on selected rows, where the normal tint would sit dark-on-dark
    against the highlight."""
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer

    path = _CUSTOM_ICONS_DIR / f"{name}.svg"
    if not path.exists():
        path = _ICONS_DIR / f"{name}.svg"
    if not path.exists():  # pragma: no cover - packaging guard
        return None
    svg = path.read_text()

    def _render(tint):
        renderer = QSvgRenderer(QByteArray(
            svg.replace("currentColor", QColor(tint).name()).encode("utf-8")))
        pm = QPixmap(size * 3, size * 3)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        renderer.render(painter)
        painter.end()
        pm.setDevicePixelRatio(3.0)
        return pm

    icon = QIcon(_render(color))
    if selected_color is not None:
        icon.addPixmap(_render(selected_color), QIcon.Mode.Selected)
    return icon


def _icon_img_tag(label, name: str) -> str:
    """An ``<img>`` tag for icon ``name`` — rendered into the label's document as a
    resource, tinted to its text color, sized to cap alongside the line. Lets coach
    text show the actual button glyph it is naming (``{icon:eye-off}``)."""
    from PySide6.QtCore import QByteArray, Qt, QUrl
    from PySide6.QtGui import QImage, QPainter, QTextDocument
    from PySide6.QtSvg import QSvgRenderer

    path = _CUSTOM_ICONS_DIR / f"{name}.svg"
    if not path.exists():
        path = _ICONS_DIR / f"{name}.svg"
    if not path.exists():  # pragma: no cover - packaging guard
        return ""
    px = label.fontMetrics().height() + 2
    tint = label.palette().windowText().color().name()
    svg = path.read_text().replace("currentColor", tint)
    img = QImage(px * 3, px * 3, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter)
    painter.end()
    img.setDevicePixelRatio(3.0)
    label.document().addResource(
        QTextDocument.ResourceType.ImageResource, QUrl(f"pxicon:{name}"), img)
    return f'<img src="pxicon:{name}" width="{px}" height="{px}">'


# One idea, one name. "Let the map pull" was offered twice under two labels — "Use map"
# beside Minimize, "Into the density" beside the drag — which read as two unrelated
# features rather than the same switch on two operations.
_INTO_DENSITY_TIP = (
    "Let the map pull too, so the model settles into the density rather than only onto "
    "ideal geometry. Needs a map paired with the model, so the two share a frame.")

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
# tabs the panel color (palette(window)) to cover the native gray base, rather than
# leaving it transparent (which shows that gray through). With the metrics overridden,
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


def _viewport_dblclick_filter(tree, on_dblclick):
    """A QObject filter on ``tree``'s viewport calling ``on_dblclick(item, column)``
    for a left double-click. Installed at the event level because the item-signal
    route (``itemDoubleClicked``) dies whenever the first click's side effects rebuild
    the tree mid-gesture — see the caller. ``item`` may be None (empty area). Returns
    the filter (parented to the viewport, so it lives as long as the tree)."""
    from PySide6.QtCore import QEvent, QObject, Qt

    class _DblClickFilter(QObject):
        def eventFilter(self, obj, event):
            if (event.type() == QEvent.Type.MouseButtonDblClick
                    and event.button() == Qt.MouseButton.LeftButton):
                point = event.position().toPoint()
                on_dblclick(tree.itemAt(point), tree.columnAt(point.x()))
            return False  # never consumed: Qt's own handling still runs

    filt = _DblClickFilter(tree.viewport())
    tree.viewport().installEventFilter(filt)
    return filt


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
    view actually paints, so 100k+ atoms stay cheap (QTableView virtualizes rendering).
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
            materialized, so the view stays cheap even against 100k+ atoms.
            """
            new = (None if indices is None else
                   [i for i in sorted({int(i) for i in indices}) if 0 <= i < self._n])
            if new == self._filter:
                return  # a reset clears the view's selection — only pay it on change
            self.beginResetModel()
            self._filter = new
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


def _make_component_table_model():
    """A QAbstractTableModel over a session's components (built lazily post-Qt).

    Rows are residue groups in hierarchy order -- protein residues, ligands,
    waters, ions: the unit the residue walk steps in, as a list it can be
    stepped through like every other table here. Each row keeps its atom
    indices so a selection pushes straight to the viewer, and an atom->row
    owner map so a viewer selection can mark the rows it touches.
    """
    from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

    class ComponentTableModel(QAbstractTableModel):
        def __init__(self):
            super().__init__()
            self._headers = ["#", "chain", "resname", "resid", "atoms"]
            self._rows: list = []   # (chain, resname, resid, [atom indices])
            self._owner = None      # atom index -> component index (numpy, -1 = none)
            self._filter: Optional[list] = None  # None = all rows; else visible components

        def set_session(self, session) -> None:
            self.beginResetModel()
            self._filter = None
            self._rows = []
            self._owner = None
            model = getattr(session, "model", None)
            if model is not None:
                hierarchy = model.get_hierarchy()
                self._owner = np.full(hierarchy.atoms_size(), -1, dtype=np.int64)
                for model_ in hierarchy.models():
                    for chain in model_.chains():
                        cid = chain.id.strip()
                        for rg in chain.residue_groups():
                            idxs = [int(a.i_seq) for a in rg.atoms()]
                            if not idxs:
                                continue
                            resname = rg.atom_groups()[0].resname.strip()
                            self._owner[idxs] = len(self._rows)
                            self._rows.append((cid, resname, rg.resid().strip(), idxs))
            else:
                arrays = getattr(getattr(session, "_data", None), "arrays", None)
                if arrays is not None and len(arrays) > 0:
                    # No hierarchy: group consecutive atoms by their residue labels.
                    self._owner = np.full(len(arrays), -1, dtype=np.int64)
                    prev = object()
                    for i in range(len(arrays)):
                        key = (arrays.chain[i], int(arrays.resseq[i]), arrays.resname[i])
                        if key != prev:
                            prev = key
                            self._rows.append(
                                (arrays.chain[i], arrays.resname[i],
                                 str(int(arrays.resseq[i])), [i]))
                        else:
                            self._rows[-1][3].append(i)
                        self._owner[i] = len(self._rows) - 1
            self.endResetModel()

        def set_filter(self, components) -> None:
            """Restrict the visible rows to these component indices; None = all."""
            new = (None if components is None else
                   [c for c in sorted({int(c) for c in components})
                    if 0 <= c < len(self._rows)])
            if new == self._filter:
                return  # a reset clears the view's selection — only pay it on change
            self.beginResetModel()
            self._filter = new
            self.endResetModel()

        def is_filtered(self) -> bool:
            return self._filter is not None

        def row_component(self, row: int) -> int:
            """The underlying component index for a view row (identity unless filtered)."""
            return row if self._filter is None else self._filter[row]

        def row_atoms(self, row: int) -> list:
            return self._rows[self.row_component(row)][3]

        def row_label(self, row: int):
            """The ``(chain, resid)`` a view row names -- the residue-walk key."""
            chain, _resname, resid, _atoms = self._rows[self.row_component(row)]
            return chain, resid

        def components_with_atoms(self, atoms) -> list:
            """Component indices (not view rows) containing any of ``atoms``."""
            if self._owner is None:
                return []
            return sorted({int(self._owner[a]) for a in atoms
                           if 0 <= a < len(self._owner) and self._owner[a] >= 0})

        def rows_with_atoms(self, atoms) -> list:
            """View rows whose component contains any of ``atoms``."""
            comps = self.components_with_atoms(atoms)
            if self._filter is None:
                return comps
            pos = {c: r for r, c in enumerate(self._filter)}
            return sorted(pos[c] for c in comps if c in pos)

        def rowCount(self, parent=QModelIndex()):
            if parent.isValid():
                return 0
            return len(self._rows) if self._filter is None else len(self._filter)

        def columnCount(self, parent=QModelIndex()):
            return 0 if parent.isValid() else len(self._headers)

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid():
                return None
            comp = self.row_component(index.row())
            chain, resname, resid, atoms = self._rows[comp]
            col = index.column()
            if role == Qt.ItemDataRole.DisplayRole:
                if col == 0:
                    return str(comp)
                if col == 1:
                    return chain
                if col == 2:
                    return resname
                if col == 3:
                    return resid
                return str(len(atoms))
            if role == Qt.ItemDataRole.TextAlignmentRole and col in (0, 3, 4):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return None

        def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
            if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
                return self._headers[section]
            return None

    return ComponentTableModel()


def _atom_label_fn(session):
    """A ``i_seq -> "chain/resnameresseq/name"`` labeler from a session's columns."""
    arrays = getattr(getattr(session, "_data", None), "arrays", None)
    if arrays is None:
        return str
    chain, resname, resseq, name = arrays.chain, arrays.resname, arrays.resseq, arrays.name

    def label(i: int) -> str:
        return f"{chain[i]}/{resname[i]}{int(resseq[i])}/{name[i]}"

    return label


def _geostd_source_fn(session):
    """An ``i_seqs -> (text, geostd_path_or_None)`` labeler for a restraint's source.

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
    cells from recomputing — so 100k+ restraints stay cheap (QTableView virtualizes).
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
            new = None if indices is None else sorted({int(i) for i in indices})
            if new == self._filter:
                return  # a reset clears the view's selection — only pay it on change
            self.beginResetModel()
            self._filter = new
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
        self._load_done = True     # a watchdog reads this; set False by load()
        self._closed = False
        # A renderer that cannot spawn at all — the Chromium sandbox refuses on some
        # Linux kernels (aarch64 among them) — never delivers loadFinished, so the
        # WebGL check alone can never see that failure. The signal below covers a
        # spawned-then-killed renderer; the watchdog in load() covers one that never
        # registered a process at all. Both feed pxviewer.gpu's one-shot restart.
        self._view.page().renderProcessTerminated.connect(self._renderer_gone)

    def _build_coach_pane(self):
        """The guided-tutorial coach: a pane docked below the viewer (hidden until a
        tutorial starts). Built here so it splits the viewport, not the controls pane; the
        ControlsWindow owns the logic and drives these widgets (see its tutorial methods)."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QFrame, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout)

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
        # QTextBrowser rather than QLabel: the coach embeds the buttons' own icons
        # inline ({icon:name} — see _coach_markup), which needs a real document to
        # hold the image resources; QLabel keeps its document private.
        self.coach_text = QTextBrowser()
        self.coach_text.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        self.coach_text.setFrameShape(QFrame.Shape.NoFrame)
        self.coach_text.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.coach_text.setOpenLinks(False)
        self.coach_text.setStyleSheet("background:transparent; border:none;")
        self.coach_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.coach_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.coach_text.document().setDocumentMargin(0)
        v.addWidget(self.coach_text, stretch=1)
        row = QHBoxLayout()
        self.coach_show = QPushButton("Show me where")
        self.coach_show.setToolTip("Flash the button this step is about; you still click it")
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

    #: How long a load may run before a never-spawned renderer is assumed. The
    #: frontend is served from localhost, so even a slow machine answers in seconds.
    _RENDERER_WATCHDOG_MS = 15000

    def load(self, url: str) -> None:
        from PySide6.QtCore import QTimer, QUrl
        from . import gpu
        self._load_done = False
        self._view.load(QUrl(url))
        if gpu.sandbox_autofix_enabled():
            QTimer.singleShot(self._RENDERER_WATCHDOG_MS, self._watchdog)

    def _renderer_gone(self, status, exit_code) -> None:
        """A spawned renderer died (or was killed). Normal teardown is ignored —
        close() releases the page deliberately."""
        if self._closed:
            return
        from PySide6.QtWebEngineCore import QWebEnginePage
        if status != QWebEnginePage.RenderProcessTerminationStatus.NormalTerminationStatus:
            from . import gpu
            gpu.on_renderer_unavailable()

    def _watchdog(self) -> None:
        """A load that never finished means no renderer ever ran the page — the
        signal-free form of the same failure _renderer_gone sees."""
        if not self._closed and not self._load_done:
            from . import gpu
            gpu.on_renderer_unavailable()

    def _verify_webgl(self, ok: bool) -> None:
        """Once the page has loaded, confirm it actually got a WebGL context.

        Only when :mod:`pxviewer.gpu` armed the check (the auto path, outcome unknown):
        if the real viewport has no WebGL, the app restarts on software rendering rather
        than showing a blank viewer; if it does, that is remembered so no future launch
        pays for the check.
        """
        from . import gpu

        self._load_done = True
        # A finished load — even a failed one, ok=False — proves a renderer is
        # running inside the sandbox, so the spawn watch can stand down.
        gpu.mark_sandbox_ok()
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
        self._closed = True          # stand down the renderer watchdog for teardown
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
        self._tutorial_step_satisfied = False  # edge-latch for the done() poll
        self._tutorial_timer = None      # polls the step's done() predicate while active
        self._hl_timer = None            # pulses a step's highlight target
        self._hl_overlay = None          # a transparent ring drawn over it (no layout effect)
        self._wire_coach()

        self._console = None  # EmbeddedConsole, created lazily on first tab view
        self._console_started = False
        self._items: list = []  # last Loaded-tree items summary (for the appearance pane)
        self._focused: tuple = (None, None)  # (kind, id) currently shown in Appearance
        # Buttons to gray out while the operation they start is running, keyed by its label
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
        # dead gray field to the right of the last one, most of the bar at panel widths.
        tabs.tabBar().setExpanding(True)
        if _IS_MAC:
            tabs.tabBar().setStyleSheet(_TAB_BAR_QSS)
        tabs.setIconSize(QSize(20, 20))
        # Lucide line icons, tinted to the tab text color so they read in light and dark.
        tint = self._btn_tint
        specs = [
            (self._build_scene_tab(), "Scene", "layers"),
            (self._build_geometry_tab(), "Geometry", "drafting-compass"),
            (self._build_validation_tab(), "Validation", "award"),
            (self._build_tools_tab(), "Tools", "wrench"),
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
        # It doubles as the tab labeler on hover, so remember the real status underneath.
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
        # Which mouse mode is armed (see _update_mode_chip), beside the buttons rather
        # than in the message: a mode is a standing state, not an event, and the status
        # text is overwritten by the next thing that happens.
        self._mode_chip = QLabel("")
        self._mode_chip.setToolTip("The armed mouse mode — what a click in the viewport does")
        self._mode_chip.setVisible(False)
        status_row.addWidget(self._mode_chip)
        # The camera trio, together: step out, step in, then reframe. The wheel is the
        # contour level here (Coot's binding), so zoom otherwise needs ctrl+wheel or a
        # right-drag -- fine on a trackpad, awkward on a plain mouse, and invisible to
        # anyone who has not read the mouse help.
        self._zoom_out_btn = self._make_icon_button(
            "zoom-out", "Zoom out", "Zoom out one step (the camera keeps its target)")
        self._zoom_out_btn.clicked.connect(lambda: self._desktop.zoom_view(1))
        status_row.addWidget(self._zoom_out_btn)
        self._zoom_in_btn = self._make_icon_button(
            "zoom-in", "Zoom in", "Zoom in one step (the camera keeps its target)")
        self._zoom_in_btn.clicked.connect(lambda: self._desktop.zoom_view(-1))
        status_row.addWidget(self._zoom_in_btn)
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
            "picture-in-picture-2", "Detach",
            "Detach the controls to their own window")
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
        desktop.bridge.atom_picked.connect(self._on_atom_picked)
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
        self._update_tug_maps_after()
        self._update_pair_button()
        self._fit_tree_height()  # the empty list must not reserve space either
        self._appearance_sig = _UNSET  # no pane built yet, so the first update must build one
        self._update_appearance()  # empty-state placeholder

    # -- tabs ------------------------------------------------------------

    def _build_scene_tab(self):
        """Home: open files, the object list, appearance of the focused object, selection."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QFrame,
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
        # Columns: [eye] [name] [restraints]. One affordance each: the eye
        # shows/hides the object, clicking anywhere on a row makes it the current
        # object (the highlight) — the Appearance pane edits the current object,
        # and a current *model* is also the active one (selection box, atoms
        # table, geometry). Column 2 is state, not a control: the "restraints"
        # box on a model row ticks when its geometry restraints have been
        # processed (the load-time warm, or on demand) — so a user can tell a
        # model that is drag/selection-ready from one whose first use still pays
        # the pdb_interpretation pass.
        self._loaded_tree.setColumnCount(3)
        self._loaded_tree.setHeaderHidden(True)
        # Pin the selection paint. Native macOS uses the vivid accent with white text
        # (which drowned the dark eye icon) when focused, and a pale gray when not. A
        # fixed pale blue with the normal dark text works for both icon and name, and
        # is deterministic across machines and system accent colors — safe to
        # hardcode because the app is pinned to the light theme. One rule, no :active
        # variant, so focused and unfocused rows paint identically.
        self._loaded_tree.setStyleSheet(
            "QTreeWidget::item:selected { background: #b3d7ff;"
            " color: palette(text); }")
        header = self._loaded_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._loaded_tree.currentItemChanged.connect(self._on_tree_current_changed)
        self._loaded_tree.itemClicked.connect(self._on_tree_item_clicked)
        # Double-clicks are caught at the raw event level, not via itemDoubleClicked:
        # the first click on a non-active model activates it, which rebuilds the tree,
        # and Qt then swallows the whole gesture — mouseDoubleClickEvent requires its
        # remembered pressed index to still match, and the rebuild invalidated it, so
        # neither itemDoubleClicked nor a second itemClicked ever fires (measured with
        # QTest-synthesized events). The MouseButtonDblClick *event* is synthesized
        # from click timing and position on the window, independent of items, so a
        # viewport filter sees the gesture whatever the rebuild did.
        self._tree_dblclick_filter = _viewport_dblclick_filter(
            self._loaded_tree, self._on_tree_row_double_clicked)
        ol.addWidget(self._loaded_tree, stretch=1)

        # -- Actions on the objects: a compact icon toolbar -----------------
        # Icon-only (the words move to richer tooltips), one row in three groups:
        # data in (open, tutorials) | data out (save, remove) | other (pair).
        # Lucide icons tinted to the button text color, with the old label kept as
        # fallback text if an asset is missing.
        def _icon_button(icon_name, label, tooltip, on_click=None):
            b = self._make_icon_button(icon_name, label, tooltip)
            if on_click is not None:
                b.clicked.connect(on_click)
            return b

        self._open_btn = _icon_button(
            "folder-open", "Open",
            "Open a structure or map — local files, or fetched from the PDB / EMDB")
        self._open_btn.setMenu(self._build_open_menu())
        # No menu-indicator dot: at icon-button size it reads as a smudge next to the
        # glyph, and the tooltip already says a menu opens.
        for _menu_btn in (self._open_btn,):
            _menu_btn.setStyleSheet(_menu_btn.styleSheet()
                                    + " QPushButton::menu-indicator { image: none; width: 0px; }")
        self._get_btn = _icon_button(
            "graduation-cap", "Tutorials",
            "Guided tutorials: each loads its own example and walks one skill")
        self._get_btn.setMenu(self._build_get_menu())
        self._get_btn.setStyleSheet(self._get_btn.styleSheet()
                                    + " QPushButton::menu-indicator { image: none; width: 0px; }")
        self._write_btn = _icon_button(
            "save", "Save",
            "Write the focused object to disk, or save the scene as a view")
        self._write_btn.setMenu(self._build_save_menu())
        for _menu_btn in (self._write_btn,):
            _menu_btn.setStyleSheet(_menu_btn.styleSheet()
                                    + " QPushButton::menu-indicator { image: none; width: 0px; }")
        self._pair_btn = _icon_button(
            "combine", "Pair",
            "Pair an unpaired model with a map, or check an existing pair for a missing "
            "density-supported origin shift",
            self._on_pair)
        self._remove_model_btn = _icon_button(
            "trash-2", "Remove", "Remove the highlighted object", self._on_remove_selected)

        def _vsep():
            line = QFrame()
            line.setFrameShape(QFrame.Shape.VLine)
            line.setStyleSheet("color: palette(mid);")
            line.setFixedWidth(10)
            return line

        actions = QHBoxLayout()
        actions.setSpacing(4)
        for button in (self._open_btn, self._get_btn):        # data in
            actions.addWidget(button)
        actions.addWidget(_vsep())
        for button in (self._write_btn, self._remove_model_btn):  # data out
            actions.addWidget(button)
        actions.addWidget(_vsep())
        actions.addWidget(self._pair_btn)                     # other
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
        self._select_expr_model = None  # which model the box text describes (see _run_selection)
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

        # Whether applying a typed selection also aims the camera at it. On by default:
        # typing "resseq 29" is a statement of intent, and a selection that highlights
        # something off-screen looks like nothing happened. Lives here rather than in
        # Settings -- the option belongs beside the control it modifies. (Viewport picks
        # are never auto-focused; the camera is already where the user is clicking.)
        self._focus_on_select = QCheckBox("Focus on selection")
        self._focus_on_select.setToolTip(
            "Aim the camera at the atoms a typed selection matches.")
        self._focus_on_select.setChecked(
            str(self._desktop._settings.value("selection/focus_on_apply", "true")).lower()
            != "false")
        self._focus_on_select.toggled.connect(
            lambda on: self._desktop._settings.setValue(
                "selection/focus_on_apply", "true" if on else "false"))
        sl.addWidget(self._focus_on_select)

        # Selection clipping is a property of the object, so it lives with the rest
        # of the object's look: the Appearance pane's Clip sphere row ("On
        # selection" = selections isolate, the default). The neighborhood
        # context is likewise a standing style choice, in the Settings tab's
        # Selection box (_build_settings_tab), where _context_on_select is built.
        # What a viewport click selects. Atom is the default — a click takes exactly
        # the atom under the cursor — while a residue-level worklist (Components, a
        # validation table) flips it to Residue, since a click there wants the whole
        # residue (_engage_step_view). The control is also the readout: it always
        # says what the next click will do, and a hand-set choice holds until the
        # next table engagement. Deliberately NOT persisted: it moves with table
        # engagement all session, so a stored value is stale the moment it is read —
        # every session starts at Atom.
        gran_row = QHBoxLayout()
        gran_row.addWidget(QLabel("Click selects:"))
        self._pick_granularity = QComboBox()
        self._pick_granularity.addItem("Atom", "atom")
        self._pick_granularity.addItem("Residue", "residue")
        self._pick_granularity.setToolTip(
            "What a click in the viewport selects — just the atom (the default), "
            "or the atom's whole residue. Follows the table you last engaged (the "
            "Atoms table and restraint tables imply atoms, residue-level tables "
            "imply residues); changing it by hand holds until the next table "
            "engagement. A ribbon face has no atoms to aim at, so a click on "
            "cartoon still takes the residue — clicks on ball-and-stick "
            "(including the neighborhood layer) stay atom-precise.")
        gran_row.addWidget(self._pick_granularity)
        gran_row.addStretch(1)
        sl.addLayout(gran_row)

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
        # Tutorials, and nothing else: fetching lives under Open with the other ways of
        # getting data on screen (where a reader looks for it), and there is no separate
        # examples list -- every bundled example is the opening scene of the tutorial
        # that explains it. One list, no headings to explain.
        for tut in tutorial.all_tutorials():
            menu.addAction(tut.title, lambda _c=False, t=tut: self._start_tutorial(t))
        return menu

    def _build_open_menu(self):
        """The two ways data gets on screen: a local file, or an entry fetched live."""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self._window)
        menu.setObjectName("openMenu")
        menu.addAction("Open file(s)…", self._on_open_file)
        menu.addAction("Fetch from PDB / EMDB…", self._on_fetch)
        menu.addSeparator()
        menu.addAction("Open saved view…", self._on_open_view)
        return menu

    def _build_save_menu(self):
        """The two things Save can produce: the focused object's data, or the
        whole scene as a restorable view."""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self._window)
        menu.setObjectName("saveMenu")
        menu.addAction("Write object…", self._on_write_object)
        menu.addSeparator()
        menu.addAction("Save view as…", self._on_save_view)
        return menu

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
        map_layout.addWidget(QLabel("Maps computed from what you already have:"))
        self._localres_btn = self._make_text_button(
            "Local resolution",
            "Calculate local resolution from two half-maps and color the map by it")
        self._localres_btn.clicked.connect(self._on_localres_wizard)
        map_row = QHBoxLayout()
        map_row.addWidget(self._localres_btn)
        self._rs_diff_btn = self._make_text_button(
            "Difference",
            "Real-space difference map: the density the model predicts, subtracted from "
            "the map. Green is density the model does not account for, red is model with "
            "no map under it — the mFo-DFc reading, but computed from a map and a model "
            "alone, with no reflections to phase.")
        self._rs_diff_btn.clicked.connect(self._on_real_space_difference)
        self._register_busy_button(self._rs_diff_btn, "Real-space difference")
        map_row.addWidget(self._rs_diff_btn)
        map_row.addStretch(1)
        map_layout.addLayout(map_row)
        layout.addWidget(map_tools)

        layout.addWidget(self._build_refine_group())

        # Last, and in this order: these two are tall (a list, four inputs) and reached
        # occasionally, where the boxes above are the everyday ones. Refine drag in
        # particular is a mouse mode toggled constantly while fitting, and it must not sit
        # below a scroll of panels nobody has opened.
        layout.addWidget(self._build_edits_group())

        layout.addWidget(self._build_ligand_placement_group())

        layout.addStretch()
        # Wrap in a scroll area (like the Scene tab). Without it, when the groups are
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

    #: Width of the driver labels in the Refine box, so the two action rows line up
    #: under one another and read as two ways of driving one thing.
    _DRIVER_LABEL_WIDTH = 86

    def _driver_label(self, text: str):
        """The left-hand name of a Refine driver: fixed width, so the two rows align and
        neither label swallows the space a hidden summary leaves behind."""
        from PySide6.QtWidgets import QLabel, QSizePolicy

        label = QLabel(text)
        label.setFixedWidth(self._DRIVER_LABEL_WIDTH)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, label.sizePolicy().verticalPolicy())
        return label

    def _build_refine_group(self):
        """Refine — one box, because it is one engine driven two ways.

        Minimize relaxes the whole model; a drag relaxes the neighborhood under the
        pointer. They were two boxes, which is how "Into the density" came to be offered
        twice as two unrelated-looking checkboxes, and how "does the map answer back"
        (the live difference map) ended up filed under one of them. The shared question
        is asked once, at the top; then a row per driver, each with a summary of what it
        would do; then the drag's own knobs, folded until it is armed.

        Two rules from elsewhere in the panel survive the merge: an option belongs beside
        the control it modifies (these knobs were once on Settings, in a group whose hint
        had to open by giving directions to its own switch), and a summary line is what a
        folded box owes the reader.
        """
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGroupBox, QHBoxLayout, QLabel,
            QSpinBox, QVBoxLayout, QWidget,
        )

        box = QGroupBox("Refine")
        dg = QVBoxLayout(box)
        dg.addWidget(QLabel("Relax the model onto ideal geometry — all of it, or the "
                            "neighborhood under the pointer."))

        # The shared question, asked once. Two boxes rather than one switch because the
        # two operations really can differ -- a predictable geometry-only tug while the
        # whole-model relax uses the map is a reasonable way to work -- but they are one
        # idea, so they share a label and a tooltip and sit on one row.
        density_row = QHBoxLayout()
        density_label = QLabel("Into the density:")
        density_label.setToolTip(_INTO_DENSITY_TIP)
        density_row.addWidget(density_label)
        self._minimize_map_check = QCheckBox("whole model")
        self._minimize_map_check.setToolTip(_INTO_DENSITY_TIP)
        self._minimize_map_check.toggled.connect(
            lambda _on: self._refresh_minimize_summary())
        density_row.addWidget(self._minimize_map_check)
        self._tug_density_check = QCheckBox("drag")
        self._tug_density_check.setToolTip(_INTO_DENSITY_TIP)
        self._tug_density_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_tug_into_density(on)))
        density_row.addWidget(self._tug_density_check)
        density_row.addStretch()
        dg.addLayout(density_row)

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)
        dg.addWidget(rule)

        whole_row = QHBoxLayout()
        whole_row.addWidget(self._driver_label("Whole model"))
        self._minimize_btn = self._make_icon_button(
            "play", "Minimize",
            "Minimize the active model against its geometry restraints (no map), "
            "streaming each step into the viewport as it runs")
        self._minimize_btn.clicked.connect(self._on_minimize)
        whole_row.addWidget(self._minimize_btn)
        self._minimize_stop_btn = self._make_icon_button(
            "pause", "Stop", "Halt the run, keeping the progress so far")
        self._minimize_stop_btn.setEnabled(False)
        self._minimize_stop_btn.clicked.connect(lambda: self._desktop.stop_minimization())
        whole_row.addWidget(self._minimize_stop_btn)
        self._on_minimizing_changed(False)  # paint the idle look (Minimize green, Stop quiet)
        # The drag has always said what it would do; the other driver owes the same.
        self._minimize_summary = QLabel("")
        self._minimize_summary.setWordWrap(True)
        self._minimize_summary.setStyleSheet("color: palette(placeholder-text);")
        whole_row.addWidget(self._minimize_summary, stretch=1)
        whole_row.addStretch()
        dg.addLayout(whole_row)
        self._refresh_minimize_summary()

        arm_row = QHBoxLayout()
        arm_row.addWidget(self._driver_label("Drag"))
        self._refine_drag_btn = self._make_icon_button(
            "hand", "Refine drag",
            "Drag any atom or bond to pull it and the model bends to follow — a local "
            "minimization under the pointer. Mutually exclusive with Pick.  (Ctrl+R)",
            checkable=True)
        self._refine_drag_btn.toggled.connect(self._on_toggle_refine_drag)
        # Arming hides the summary beside it (the unfolded options say the same thing),
        # and a QHBoxLayout hands a hidden widget's space to whatever can grow — which
        # stretched this button across the row. It is a button; keep it button-sized.
        from PySide6.QtWidgets import QSizePolicy

        self._refine_drag_btn.setSizePolicy(QSizePolicy.Policy.Fixed,
                                            self._refine_drag_btn.sizePolicy().verticalPolicy())
        arm_row.addWidget(self._refine_drag_btn)
        # What a drag would do, in one line, for when the controls below are folded away.
        self._tug_summary = QLabel("")
        self._tug_summary.setWordWrap(True)
        self._tug_summary.setStyleSheet("color: palette(placeholder-text);")
        arm_row.addWidget(self._tug_summary, stretch=1)
        arm_row.addStretch()
        dg.addLayout(arm_row)

        self._tug_options = QWidget()
        og = QVBoxLayout(self._tug_options)
        og.setContentsMargins(self._DRIVER_LABEL_WIDTH, 0, 0, 0)
        dg.addWidget(self._tug_options)

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
        og.addLayout(scope_row)

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
            self._refresh_tug_summary()

        scope_combo.currentIndexChanged.connect(lambda _i: _apply_scope())
        radius_spin.valueChanged.connect(lambda _v: _apply_scope())
        flank_spin.valueChanged.connect(lambda _v: _apply_scope())

        self._tug_continuous_check = QCheckBox("Keep minimizing while dragging")
        self._tug_continuous_check.setToolTip(
            "While dragging, the model keeps relaxing the whole time — a gentle living "
            "settle that stays in motion even when the pointer is still, rather than "
            "nudging once per move and stopping.")
        self._tug_continuous_check.setChecked(True)  # on by default; connect after, no fire
        self._tug_continuous_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_tug_continuous(on)))
        og.addWidget(self._tug_continuous_check)
        self._tug_diag_check = QCheckBox("Drag diagnostics")
        self._tug_diag_check.setToolTip(
            "Report each gesture in the status line: grabs that found no atom under "
            "the pointer, and grab-to-first-motion latency with the zone-build cost. "
            "For chasing dead-feeling drags; harmless to leave on.")
        self._tug_diag_check.setChecked(self._desktop._tug_diagnostics)
        self._tug_diag_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_tug_diagnostics(on)))
        og.addWidget(self._tug_diag_check)
        strength_row = QHBoxLayout()
        strength_label = QLabel("Pull strength")
        strength_row.addWidget(strength_label)
        self._tug_strength_spin = QDoubleSpinBox()
        self._tug_strength_spin.setRange(0.1, 20.0)
        self._tug_strength_spin.setSingleStep(0.5)
        self._tug_strength_spin.setDecimals(1)
        self._tug_strength_spin.setSuffix("×")
        self._tug_strength_spin.setValue(self._desktop._tug_strength)
        strength_tip = ("How hard a drag pulls the grabbed atom toward the pointer, as a "
                        "multiple of the default. Raise it if drags feel weak against the "
                        "geometry (or the density term); takes effect on the next grab.")
        strength_label.setToolTip(strength_tip)
        self._tug_strength_spin.setToolTip(strength_tip)
        self._tug_strength_spin.valueChanged.connect(lambda v: self._safe(
            lambda: self._desktop.set_tug_strength(v)))
        strength_row.addWidget(self._tug_strength_spin)
        strength_row.addStretch()
        og.addLayout(strength_row)
        settle_row = QHBoxLayout()
        settle_label = QLabel("Settle wind-down")
        settle_row.addWidget(settle_label)
        self._tug_settle_spin = QDoubleSpinBox()
        self._tug_settle_spin.setRange(0.0, 5.0)
        self._tug_settle_spin.setSingleStep(0.2)
        self._tug_settle_spin.setDecimals(1)
        self._tug_settle_spin.setSuffix(" s")
        self._tug_settle_spin.setValue(self._desktop._tug_settle_seconds)
        tip = ("How long the model visibly keeps relaxing after you let go, before it "
               "freezes. The physics finishes either way — this stretches the decay so "
               "a fling reads as settling, not stopping dead. 0 freezes immediately.")
        settle_label.setToolTip(tip)
        self._tug_settle_spin.setToolTip(tip)
        self._tug_settle_spin.valueChanged.connect(lambda v: self._safe(
            lambda: self._desktop.set_tug_settle_seconds(v)))
        settle_row.addWidget(self._tug_settle_spin)
        settle_row.addStretch()
        og.addLayout(settle_row)
        self._tug_livemap_check = QCheckBox("Live difference map")
        self._tug_livemap_check.setToolTip(
            "While dragging, recompute the mFo-DFc difference map in a small window around "
            "the atom and show it live — green where the data wants density, red where the "
            "model has too much. Honest feedback as you fit (the main 2mFo-DFc map is left "
            "alone to avoid model bias). Needs a map phased from reflections.")
        self._tug_livemap_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_live_difference_map(on)))
        og.addWidget(self._tug_livemap_check)
        self._tug_maps_check = QCheckBox("Update maps after each drag")
        self._tug_maps_check.setToolTip(
            "Re-phase the whole-structure maps once a drag settles, the way Minimize "
            "already does — so the density stops describing where the atoms used to be, "
            "with no trip to the reflections' Appearance pane. Debounced, so a burst of "
            "drags costs one recompute. Needs reflections phased against the model.")
        self._tug_maps_check.setChecked(True)  # on by default; connect after, no fire
        self._tug_maps_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_update_maps_after_drag(on)))
        og.addWidget(self._tug_maps_check)

        for check in (self._tug_density_check, self._tug_livemap_check,
                      self._tug_maps_check, self._tug_continuous_check):
            check.toggled.connect(lambda _on: self._refresh_tug_summary())

        _apply_scope()             # initial visibility (radius shown, flank hidden) + summary
        self._tug_options.setVisible(False)   # folded until the drag is armed
        return box

    def _refresh_tug_summary(self) -> None:
        """One line saying what a drag would do — the folded box's whole content.

        Only what is actually in force: an option that is switched off, or unavailable
        because nothing is loaded to make it work, would be a promise the drag cannot keep.
        """
        scope = self._desktop._tug_scope
        if scope["mode"] == "sphere":
            bits = ["sphere %.0f Å" % scope["radius"]]
        elif scope["mode"] == "selection":
            bits = ["selection"]
        elif scope.get("flank"):
            bits = ["± %d residues" % scope["flank"]]
        else:
            bits = ["single residue"]
        for check, word in ((self._tug_density_check, "into density"),
                            (self._tug_livemap_check, "live difference map"),
                            (self._tug_maps_check, "maps update")):
            if check.isChecked() and check.isEnabled():
                bits.append(word)
        if not self._tug_continuous_check.isChecked():
            bits.append("step per move")
        self._tug_summary.setText(" · ".join(bits))

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
        self._edit_remove_btn = remove  # a tutorial highlight target
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
        else:
            self._tug_density_check.setToolTip(_INTO_DENSITY_TIP)  # a map arrived; say so
        self._refresh_tug_summary()

    def _update_minimize_map(self) -> None:
        """Offer the density pull only when the active model actually has a map to use."""
        available = self._desktop.map_for_model() is not None
        self._minimize_map_check.setEnabled(available)
        if not available:
            self._minimize_map_check.setChecked(False)
            self._minimize_map_check.setToolTip(
                "Load a model and a map together to pair them, then minimize into density.")
        else:
            self._minimize_map_check.setToolTip(_INTO_DENSITY_TIP)
        self._refresh_minimize_summary()

    def _refresh_minimize_summary(self) -> None:
        """One line saying what Minimize would do — the drag's summary, for the other
        driver. Only what is actually in force: the density pull is reported when it is
        both asked for and available, since an unavailable option is a promise the run
        cannot keep."""
        summary = getattr(self, "_minimize_summary", None)
        if summary is None:      # during construction, before the row exists
            return
        check = self._minimize_map_check
        bits = ["ideal geometry"]
        if check.isChecked() and check.isEnabled():
            bits.append("into density")
        summary.setText(" · ".join(bits))

    def _update_tug_maps_after(self) -> None:
        """Re-phasing after a drag needs reflections already phased against the model.

        Left checked when unavailable, unlike the density boxes: it is a standing
        preference for a thing that costs nothing when there is nothing to re-phase (the
        backend simply finds no reflections), so unchecking it here would silently discard
        the user's choice the moment a model loaded without data.
        """
        available = self._desktop.reflections_for_model() is not None
        self._tug_maps_check.setEnabled(available)
        if not available:
            self._tug_maps_check.setToolTip(
                "Phase reflections against the model (Make maps) and each drag can "
                "re-phase them as it settles.")
        else:
            self._tug_maps_check.setToolTip(
                "Re-phase the whole-structure maps once a drag settles, the way Minimize "
                "already does — so the density stops describing where the atoms used to "
                "be, with no trip to the reflections' Appearance pane. Debounced, so a "
                "burst of drags costs one recompute.")
        self._refresh_tug_summary()

    def _build_clashes_page(self):
        """The all-atom contacts (probe2) results view — a validator sub-tab, so it sits
        beside the per-residue checks rather than in its own panel. The analysis itself
        is queued from the checks row above (it is one of the checkboxes); this page
        is just the legend — the overlay toggles live on the tab itself so they are
        reachable without opening a second tab."""
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        page = QWidget()
        ag = QVBoxLayout(page)
        hint = QLabel(
            "<b>Clashes &amp; contacts</b> — tick it in the checks above and press play: it "
            "adds hydrogens (reduce2) as a new object, runs probe2, and the overlay "
            "toggles above light up. There is no table here because probe's answer is "
            "the dots themselves: green and blue are favorable contacts, and the red "
            "spikes are <b>overlaps</b>, drawn where two atoms interpenetrate by more "
            "than 0.4 Å — the length of a spike is that overlap.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        ag.addWidget(hint)
        ag.addStretch(1)
        return page

    def _build_validation_tab(self):
        """MolProbity validation, all in one place: tick the checks to queue, press one
        run button. The checks are data-driven from the validation registry, plus the
        all-atom contacts analysis (reduce2 + probe2) as one more — heavier — check.
        Every result becomes a sub-tab in the shared results area below; contacts keeps
        its always-present 'Clashes & contacts' sub-tab for the overlay toggles."""
        from PySide6.QtWidgets import (
            QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget,
            QVBoxLayout, QWidget,
        )

        from . import validation

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        intro = QLabel("MolProbity validation of the active model: tick the checks to "
                       "run, then press play. Each result becomes a sub-tab below.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # The queue: one checkbox per registered validator, all on by default -- they
        # share one cheap pass. Clashes & contacts is the odd one out (reduce2 + probe2,
        # and it adds a hydrogenated sibling object), so it starts unticked.
        checks = QGridLayout()
        checks.setHorizontalSpacing(16)
        checks.setVerticalSpacing(4)
        self._check_boxes: list = []   # (validator key, QCheckBox)
        for i, spec in enumerate(validation.validators()):
            box = QCheckBox(spec.title)
            box.setChecked(True)
            box.setToolTip(f"Queue the {spec.title} check for the next run")
            checks.addWidget(box, i // 2, i % 2)
            self._check_boxes.append((spec.key, box))
        self._clashes_check = QCheckBox(
            "All-atom contacts && clashes (adds hydrogens — slower)")
        self._clashes_check.setToolTip(
            "Queue the all-atom contact analysis: add hydrogens with reduce2 as a new "
            "object (hiding the original), then run probe2 for MolProbity contacts and "
            "clashes. The heaviest check, so it starts unticked.")
        checks.addWidget(self._clashes_check, (len(self._check_boxes) + 1) // 2, 0, 1, 2)
        self._hotspots_check = QCheckBox("Hotspot severity (aggregate of the above)")
        self._hotspots_check.setToolTip(
            "Queue the severity aggregate: the ticked checks combined into one per-atom "
            "field, coloring the model and drawing where they agree in 3-D.\n"
            "It reuses the analysis the other checks already build, so ticking it beside "
            "them costs the aggregation and little else. Results land in the Hotspots "
            "sub-tab.")
        checks.addWidget(self._hotspots_check,
                         (len(self._check_boxes) + 1) // 2 + 1, 0, 1, 2)
        layout.addLayout(checks)

        # Per-validator Markers checkboxes, rebuilt with the sub-tabs on every run. The probe
        # overlay toggles are separate because they outlive a re-run; both are markup, so the
        # "all" button drives them together.
        self._marker_checks: list = []

        # The two probe overlay toggles. They sit on the tab itself — not inside the
        # Clashes sub-tab — because a user looking for "hide the dots" must not have to
        # open a second tab; a buried toggle reads as "there is no way to hide this".
        # Disabled until a contacts run produces dots (see _sync_probe_toggles).
        from .live import PROBE_CLASHES, PROBE_CONTACTS

        self._contacts_toggle = self._make_icon_button(
            "fold-horizontal", "Contacts", "Show/hide the probe2 contact-dot surface",
            checkable=True)
        self._contacts_toggle.setEnabled(False)
        self._contacts_toggle.toggled.connect(
            lambda on: self._desktop.set_probe_channel(PROBE_CONTACTS, on))
        self._contacts_toggle.toggled.connect(lambda _on: self._sync_all_markup_button())
        self._clashes_toggle = self._make_icon_button(
            "triangle-alert", "Clashes",
            "Show/hide the bad-overlap (clash) dots and spikes",
            checkable=True)
        self._clashes_toggle.setEnabled(False)
        self._clashes_toggle.toggled.connect(
            lambda on: self._desktop.set_probe_channel(PROBE_CLASHES, on))
        self._clashes_toggle.toggled.connect(lambda _on: self._sync_all_markup_button())

        run_row = QHBoxLayout()
        run_btn = self._make_icon_button(
            "play", "Run",
            "Run the ticked checks on the active model (background); each result "
            "becomes a sub-tab below.")
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

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Probe overlays:"))
        overlay_row.addWidget(self._contacts_toggle)
        overlay_row.addWidget(self._clashes_toggle)
        overlay_row.addStretch(1)
        layout.addLayout(overlay_row)

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

        # One results area: a tab per validator, (re)built as runs complete, then two
        # always-present tabs at the end — Clashes & contacts (a legend for the dot
        # overlays, whose toggles sit in the row above) and Hotspots (the aggregate
        # over these same checks, demoted from a top-level tab: it is a *view* of
        # validation, sharing its expensive analysis, and did not earn a place
        # beside Scene and Tools).
        self._validation_subtabs = QTabWidget()
        self._validation_subtabs.setDocumentMode(True)
        self._clashes_page = self._build_clashes_page()
        # Short labels: eight tabs across this panel leaves little room each, and an
        # elided "Clashes & cont..." names nothing. Each page opens with its full name.
        self._validation_subtabs.addTab(self._clashes_page, "Clashes")
        self._validation_subtabs.addTab(self._build_hotspots_tab(), "Hot")
        layout.addWidget(self._validation_subtabs, stretch=1)
        self._sync_probe_toggles()      # reflect any dots a run already produced
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
        """Keep the label describing the action, and gray it out when there is no markup."""
        button = getattr(self, "_all_markup_btn", None)
        if button is None:  # pragma: no cover - during tab construction
            return
        live = [t for t in self._markup_toggles() if t.isEnabled()]
        button.setEnabled(bool(live))
        button.setText("Hide all markers" if any(t.isChecked() for t in live)
                       else "Show all markers")

    def _sync_probe_toggles(self) -> None:
        """Keep the probe overlay toggles honest about what is drawn.

        Enabled while some model holds cached probe dots, checked only while a
        session is actually drawing the channel. Without this a removed model (or
        a cleared scene) leaves the toggles enabled and checked, commanding a
        session that no longer exists — the toggle looks armed and does nothing."""
        from .live import PROBE_CLASHES, PROBE_CONTACTS

        shown = {PROBE_CONTACTS: False, PROBE_CLASHES: False}
        available = False
        for m in self._desktop._models:
            if m.get("probe_dots"):
                available = True
                payloads = getattr(m["session"], "_probe_dots_payloads", {})
                for ch in shown:
                    shown[ch] = shown[ch] or ch in payloads
        for name, ch in (("_contacts_toggle", PROBE_CONTACTS),
                         ("_clashes_toggle", PROBE_CLASHES)):
            toggle = getattr(self, name, None)
            if toggle is None:
                continue
            toggle.blockSignals(True)
            toggle.setEnabled(available)
            toggle.setChecked(shown[ch])
            toggle.blockSignals(False)

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
        # The full name first: the tab above is abbreviated to fit eight of them across
        # the panel, so this is where "Cbeta" is spelled out.
        heading = self._wrapped_label(f"<b>{result.title}</b> — {result.summary}")
        heading.setStyleSheet("color: palette(placeholder-text);")
        v.addWidget(heading)
        if result.notes:
            # What the numbers in this table are. A column of "0.041" or "26.39" is
            # unreadable without knowing whether it is Angstrom, degrees, a percentage
            # of a reference distribution or a standard deviation -- and this tab is
            # where the reader is standing when the question occurs to them.
            notes = self._wrapped_label(result.notes)
            notes.setStyleSheet("color: palette(placeholder-text);")
            v.addWidget(notes)

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
        """A validation table row was selected: select + focus that residue through
        the same pipeline a click or a typed expression takes, under the Selection
        pane's checkboxes, with the selection box showing the equivalent expression.
        Rows carry chain/resid columns (per-residue validators); whole-model results
        like Rama-Z have neither, so there is nothing to focus.

        The first click switches a ribbon model to ball-and-stick (the standing rule
        for atom-precision work — measuring and dragging do the same). Judging an
        outlier needs its atoms anyway, and it makes every later click nearly free:
        with the whole model in sticks the per-selection neighborhood layer has
        nothing to add, so clicking through a worklist re-sends no representations —
        without this, each click rebuilt the entire cartoon (the ribbon steps aside
        around the neighborhood), which was sluggish. Once per model only: a user
        who switches back to ribbon afterwards keeps ribbon (see
        :meth:`~DesktopApp.ensure_atoms_shown_once`)."""
        cols = result.columns
        if "chain" not in cols or "resid" not in cols:
            return
        self._engage_step_view(table)
        row = table.currentRow()
        if row < 0:
            return
        chain = table.item(row, cols.index("chain"))
        resid = table.item(row, cols.index("resid"))
        if chain is None or resid is None:
            return
        self._desktop.ensure_atoms_shown_once()
        expression = self._desktop.focus_residue(
            chain.text(), resid.text(),
            focus=self._focus_on_select.isChecked(),
            context=self._context_on_select.isChecked())
        if expression is not None:
            self._select_expr.setText(expression)
            self._select_expr_model = self._desktop._active_model_id

    def _on_run_validation(self) -> None:
        try:
            self._desktop.run_validation(
                keys=[key for key, box in self._check_boxes if box.isChecked()],
                clashes=self._clashes_check.isChecked(),
                hotspots=self._hotspots_check.isChecked())
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
        always-present tail — Clashes & contacts, then Hotspots — in place.

        ``draw_markers`` (default True for an explicit Run validation) is False when the tab is
        being populated as a side effect of a hotspot run — the tables fill in, but the markup
        is left off so it does not pile on top of the hotspot coloring the user is looking at.
        """
        mid, results, draw_markers = (*payload, True)[:3]
        tabs = self._validation_subtabs
        current = tabs.tabText(tabs.currentIndex())  # preserve the selected validator
        while tabs.count() > 2:  # drop the run's validator tabs; keep Clashes + Hotspots
            page = tabs.widget(0)
            tabs.removeTab(0)
            page.deleteLater()
        # Those checkboxes went with the pages; drop them before the new ones register, or the
        # "all" button would be reasoning about deleted widgets.
        self._marker_checks.clear()
        for pos, result in enumerate(results):
            tabs.insertTab(pos, self._build_validation_section(mid, result, draw_markers),
                           result.tab or result.title)
        # Land on the freshest thing: the same validator the user was reading when it is
        # part of this run, otherwise the first result — never the Clashes toggles page,
        # which a run of per-residue checks says nothing about.
        if results:
            tabs.setCurrentIndex(0)
            for i in range(tabs.count() - 2):
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

        # One paragraph, like every other sub-tab's: a longer one was clipped by the
        # panel, and what belongs to the import button now lives on the import button.
        intro = self._wrapped_label(
            "<b>severity</b> is scaled so <b>1.0 is exactly the outlier cut</b> of "
            "whichever check raised it (Ramachandran 0.05%, rotamer 0.3%, clash 0.4 Å "
            "overlap), and each step of 1.0 is a decade less likely again. The component "
            "columns carry each check's own severity on that same scale — the combined "
            "value only ranks, and is not a quality score.")
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

        # No run button here: hotspots is queued with the checks it aggregates, on the
        # run row above. Import is not a check, so it keeps its button.
        action_row = QHBoxLayout()
        open_volume = QPushButton("Open volume…")
        open_volume.setToolTip(
            "Open a Hotspots JSON manifest (recommended), or one bounded-concern CCP4 map "
            "with its sibling percentile map. No validation or map-model calculation "
            "runs.\n\nThis imports bounded *concern* from the Hotspots generator: a "
            "different quantity from the severity computed here, on a fixed 0–1 scale. A "
            "model shows one or the other, never both — the two scales are not "
            "interchangeable.")
        open_volume.clicked.connect(self._on_open_hotspot_volume)
        self._hotspot_open_btn = open_volume
        action_row.addWidget(open_volume)
        layout.addLayout(action_row)

        # Deliberately no field picker. Only the combined value is shown: it is the one
        # thing this tab adds (where to look), and the per-check story is already told
        # twice — the table's component columns and the Validation tab. A menu of
        # per-component fields was built, shipped, and removed as concept surface the
        # user should not have to ingest; the backend switch survives for scripting
        # (see set_hotspot_field_metric).
        field_row = QHBoxLayout()
        show3d = QCheckBox("Show in 3-D")
        show3d.setToolTip(
            "Draw the severity field around the model in 3-D.\n"
            "Per-atom color only shows the surface, so a buried hotspot stays hidden; a 3-D "
            "field is visible through the structure.")
        show3d.setEnabled(False)  # nothing to draw until a score exists
        # On by default: the field is the thing per-atom color cannot show (a buried
        # hotspot sits behind the atoms in front of it), and leaving it off meant the
        # first run showed only the half of the answer that was already visible. Set
        # before the signal is connected -- the handler reads the style and quality
        # widgets, which are built further down this method.
        show3d.setChecked(True)
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
        self._set_threshold_scale(mode="severity")

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
        # Scrolled, like the Tools and Scene tabs: this page is a paragraph, three
        # rows of controls, a slider, a summary and a table, and it is now a sub-tab
        # inside a tab. When the panel is shorter than that, a plain layout compresses
        # its widgets into each other -- the units paragraph rendered over the
        # hydrogens checkbox -- where scrolling keeps every one at its natural size.
        from PySide6.QtWidgets import QScrollArea

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(tab)
        return scroll

    def _set_threshold_scale(self, *, mode: str = "severity") -> None:
        """Point the threshold slider at one field's units, and say which they are.

        Concern is bounded to [0, 1] and its interesting range is narrow, so it gets 0.01
        steps; severity runs to the display cap in 0.1 steps. The label carries the unit
        because the same widget expresses both, and a slider reading 0.5 means very different
        things on the two scales.
        """
        slider = self._hotspot_knee_slider
        was = slider.value() / self._hotspot_knee_scale
        slider.blockSignals(True)
        if mode == "concern":
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
        self._set_threshold_scale(mode="concern")
        if imported is not None:
            self._hotspot_knee_slider.blockSignals(True)
            self._hotspot_knee_slider.setValue(
                int(round(imported.anchors["yellow"] * self._hotspot_knee_scale)))
            self._hotspot_knee_slider.blockSignals(False)
            self._hotspot_knee_value.setText(
                f"{imported.anchors['yellow']:.{self._hotspot_knee_digits}f}")
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
        self._set_threshold_scale(mode="severity")
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
        row should put you in front of the thing to fix. Same unified pipeline,
        Selection-pane checkboxes and first-click switch to ball-and-stick as a
        Validation row (see :meth:`_on_validation_row_selected` for why)."""
        columns = getattr(self, "_hotspot_columns", None)
        table = self._hotspot_table
        row = table.currentRow()
        if not columns or row < 0:
            return
        self._engage_step_view(table)
        chain = table.item(row, columns.index("chain"))
        resid = table.item(row, columns.index("resid"))
        if chain is not None and resid is not None:
            self._desktop.ensure_atoms_shown_once()
            expression = self._desktop.focus_residue(
                chain.text(), resid.text(),
                focus=self._focus_on_select.isChecked(),
                context=self._context_on_select.isChecked())
            if expression is not None:
                self._select_expr.setText(expression)
                self._select_expr_model = self._desktop._active_model_id

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

        # How much density any map draws around the view center. One default for all of
        # them: bounding is a statement about how much density you want to look at,
        # and having it on some maps and not others left contours behaving differently
        # with nothing on screen to say why. 15 A is a starting point, not a convention
        # we can point at, so it is adjustable rather than baked in. This is the default
        # — each map's own Sphere row (its Appearance pane, "Around view center") is the
        # per-map override.
        from PySide6.QtWidgets import QCheckBox

        current = self._desktop.view_radius
        radius_row = QHBoxLayout()
        radius_check = QCheckBox("Draw map density within")
        radius_check.setChecked(current is not None)
        radius_spin = QDoubleSpinBox()
        radius_spin.setRange(1.0, 200.0)
        radius_spin.setDecimals(0)
        radius_spin.setSingleStep(5.0)
        radius_spin.setSuffix(" Å")
        radius_spin.setValue(_VIEW_RADIUS_DEFAULT if current is None else float(current))
        radius_spin.setEnabled(current is not None)
        radius_tail = QLabel("of the view center")
        tip = ("How much density every map draws around the middle of the view. It "
               "follows the view and edits nothing — the maps stay whole, they are "
               "just not all drawn at once. Unticked, every map is drawn in full.")
        radius_check.setToolTip(tip)
        radius_spin.setToolTip(tip)
        radius_tail.setToolTip(tip)

        def _apply_radius():
            self._safe(lambda: self._desktop.set_view_radius(
                radius_spin.value() if radius_check.isChecked() else None))

        def _radius_toggled(on):
            radius_spin.setEnabled(bool(on))
            radius_tail.setEnabled(bool(on))
            _apply_radius()

        radius_check.toggled.connect(_radius_toggled)
        radius_spin.valueChanged.connect(lambda _v: _apply_radius())
        radius_row.addWidget(radius_check)
        radius_row.addWidget(radius_spin)
        # The units belong in the box; the rest of the sentence does not — words inside
        # an editable spin read as part of the value.
        radius_tail.setEnabled(current is not None)
        radius_row.addWidget(radius_tail)
        radius_row.addStretch()
        vg.addLayout(radius_row)

        # (The old "Show Mol* focus neighborhood on click" toggle is gone: a click now
        # runs pxviewer's own selection pipeline, and the neighborhood context is the
        # "Neighborhood in ball-and-stick" checkbox in the Selection box below — one
        # treatment for clicked and typed selections alike.)
        layout.addWidget(viewer)

        selection = QGroupBox("Selection")
        sg = QVBoxLayout(selection)

        # The neighborhood context around a selection: a style of looking at
        # selections, not a per-selection choice, so it lives here rather than in the
        # Selection pane where Focus/Clip sit. On by default — the sticks around a
        # selection are the context that makes a residue-level rep readable up close.
        self._context_on_select = QCheckBox("Neighborhood in ball-and-stick")
        self._context_on_select.setToolTip(
            "Draw the selected atoms and every residue within 5 Å of them in "
            "ball-and-stick while the selection stands — the same local context for a "
            "typed selection and a clicked atom; the main representation steps aside "
            "there, so the region reads as sticks alone. Restored when the selection "
            "clears. Skipped for large selections and when the model already shows "
            "ball-and-stick.")
        self._context_on_select.setChecked(
            str(self._desktop._settings.value("selection/context_rep", "true")).lower()
            != "false")
        self._context_on_select.toggled.connect(
            lambda on: self._desktop._settings.setValue(
                "selection/context_rep", "true" if on else "false"))
        sg.addWidget(self._context_on_select)

        # The clip sphere is reach + this much: how much neighborhood survives
        # around a clipped selection is a taste setting, not a property of any one
        # selection, so it lives here once. 4 A keeps a lone atom readable; a bond's
        # own reach already covers its two atoms.
        pad_row = QHBoxLayout()
        pad_label = QLabel("Pad the clip sphere by")
        pad_spin = QDoubleSpinBox()
        pad_spin.setRange(0.0, 50.0)
        pad_spin.setDecimals(1)
        pad_spin.setSingleStep(1.0)
        pad_spin.setSuffix(" Å")
        pad_spin.setValue(float(self._desktop._settings.value(
            "selection/clip_padding", 4.0)))
        pad_tail = QLabel("past the selected atoms")
        pad_tip = ("The sphere a clipped selection draws is the selection's reach "
                   "plus this much clearance, so even a single atom keeps its "
                   "neighborhood in view. Applies to the next selection and "
                   "re-fits a sphere that is standing now.")
        pad_label.setToolTip(pad_tip)
        pad_spin.setToolTip(pad_tip)
        pad_tail.setToolTip(pad_tip)

        def _pad_changed(value):
            self._desktop._settings.setValue("selection/clip_padding", float(value))
            if any(m.get("_auto_clip") for m in self._desktop._models):
                self._safe(self._desktop.reapply_auto_clips)

        pad_spin.valueChanged.connect(_pad_changed)
        pad_row.addWidget(pad_label)
        pad_row.addWidget(pad_spin)
        pad_row.addWidget(pad_tail)
        pad_row.addStretch()
        sg.addLayout(pad_row)
        layout.addWidget(selection)

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

        # (The "Drag atoms" group used to sit here. It is now the Refine drag box on the
        # Tools tab, wrapped around the switch it was giving directions to.)
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
        elif it["kind"] == "negative":
            # Same source: the negative contour's settings live on its map's entry. The
            # map's level counts too, because a linked contour shows it.
            live = self._desktop.volume_appearance(it["id"])
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
        self._neg_iso_row = None  # ...and only a difference map gets this one
        if it is None:
            # Reset the title too: leaving the old object's name over an empty-state hint
            # made the pane look like it was still editing something.
            self._appearance_box.setTitle("Appearance")
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
            # A value coloring's scale, the same framed Range group the map's
            # local-resolution coloring gets -- one widget, one meaning, and the same
            # blue->red ramp on both, so a model and a map coloring can share a legend.
            if it.get("value_domain") is not None:
                info = _MODEL_VALUE_COLORS[it["color"]]
                self._add_value_scale_group(
                    info["label"], it["value_domain"], info["unit"],
                    lambda lo_v, hi_v: self._safe(
                        lambda: self._desktop.set_model_value_domain(mid, lo_v, hi_v)),
                    lambda: self._safe(lambda: self._desktop.fit_model_value_domain(mid)),
                    lambda: self._safe(lambda: self._desktop.reset_model_value_domain(mid)),
                    fit_label="Fit to values",
                    fit_tip="Span the ramp over the 2nd–98th percentile of this "
                            "model's values.")
            types = it.get("types") or []
            r = QHBoxLayout()
            lab = QLabel("Show")
            lab.setMinimumWidth(80)
            r.addWidget(lab)
            r.addWidget(self._make_type_combo(
                mid, types, set(it.get("hidden_types") or []),
                interactions=bool(it.get("interactions"))), stretch=1)
            self._appearance_layout.addLayout(r)

            _model_live = self._desktop.model_appearance(mid)

            def _set_clip(front, back, it=it):
                it["clip"] = (front, back)
                self._safe(lambda: self._desktop.set_model_clip(mid, front, back))

            self._add_clip_row({**it, **_model_live}.get("clip"), _set_clip)
            self._add_sphere_row(
                _model_live.get("sphere"), allow_selection=True,
                on_pick=lambda mode, r: self._safe(
                    lambda: self._desktop.set_model_sphere(mid, mode, r)),
                on_radius=lambda r: self._safe(
                    lambda: self._desktop.set_model_sphere(
                        mid, (self._desktop.model_sphere_state(mid).get("mode")
                              or "view"), r, refresh=False)))

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
        elif it["kind"] == "negative":
            self._build_negative_pane(it["id"], add_combo)
        else:  # volume
            vid = it["id"]
            # Read the live values, not this snapshot: the level in particular can have
            # moved since (the scroll wheel, or the console) without a new summary.
            live = {**it, **self._desktop.volume_appearance(vid)}


            def _set_style(v, it=it):
                it["style"] = v
                self._safe(lambda: self._desktop.set_volume_style(vid, v))

            def _set_color(v, it=it):
                # "Local resolution" rides the same dropdown as the flat colors: both
                # answer "what colors this map", so two controls for one question (a
                # color row AND a separate checkbox) made each look unrelated to the
                # other. Picking it turns the coloring on; picking any color returns
                # to a flat surface in that color.
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
            # Downsample is a property of how any map is drawn: the plain isosurface
            # redraws from an on-lattice decimated copy, and the color-by-resolution
            # surface (when active) rebuilds from its held grids at the same factor.
            # Plain maps default to Full; the localres wizard defaults its map to 4x.
            def _set_localres_ds(v, it=it):
                it["localres_downsample"] = v
                self._safe(lambda: self._desktop.set_volume_downsample(vid, v))

            add_combo("Downsample",
                      [("Full", 1), ("2×", 2), ("4×", 4), ("8×", 8)],
                      int(it.get("localres_downsample") or 1), _set_localres_ds)
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
            self._add_sphere_row(
                live.get("sphere"), allow_selection=False,
                on_pick=lambda mode, r: self._safe(
                    lambda: self._desktop.set_volume_sphere(vid, mode, r)),
                on_radius=lambda r: self._safe(
                    lambda: self._desktop.set_volume_sphere(
                        vid, (self._desktop.volume_sphere_state(vid).get("mode")
                              or "view"), r, refresh=False)))

            def _set_mask(radius, it=it):
                it["mask_radius"] = radius
                self._safe(lambda: self._desktop.set_volume_mask(vid, radius))

            self._add_mask_row(live.get("mask_radius"),
                               self._desktop.can_mask_volume(vid), _set_mask)

            # The coloring's own settings, shown only while "Local resolution" is the
            # selected coloring (the Color dropdown is the switch; this panel is its
            # detail). Built by the shared value-scale builder, so the map's range
            # control is the identical widget a model's B-factor coloring gets.
            if it.get("color_by_resolution"):
                def _set_map_domain(lo_v, hi_v, it=it):
                    it["localres_domain"] = (lo_v, hi_v)
                    self._safe(lambda: self._desktop.set_localres_domain(vid, lo_v, hi_v))

                self._add_value_scale_group(
                    "Local resolution", it.get("localres_domain"), " Å",
                    _set_map_domain,
                    lambda: self._safe(lambda: self._desktop.fit_localres_domain(vid)),
                    lambda: self._safe(lambda: self._desktop.reset_localres_domain(vid)),
                    fit_label="Fit to surface",
                    fit_tip="Span the ramp over the resolution values inside the "
                            "current contour — what is actually on screen.")

        # The wheel contours whatever the Level slider above is showing, so the
        # target follows the focused object (and is cleared when it is not a volume).
        # Deliberately not aimed at a focused *negative contour*: the wheel drives the
        # map's own level, and turning it while that pane is open would move the other
        # contour -- exactly the "something moved and I cannot tell what" this row exists
        # to end. Its level has a slider; the wheel stays with the map.
        self._safe(lambda: self._desktop.set_volume_scroll_target(
            it["id"] if it["kind"] == "volume" else None))

    @staticmethod
    def _show_iso_value(row, value) -> None:
        """Move a level row's widgets to ``value`` without emitting a change.

        Used wherever the level moved somewhere else and the panel is only catching up —
        the viewport wheel, or a linked negative contour following the positive one.
        Writing it back would round-trip the change that caused it.
        """
        row["syncing"]["on"] = True
        try:
            row["slider"].setValue(
                min(row["slider"].maximum(), int(round(value / _ISO_RESOLUTION))))
            row["spin"].setValue(value)
        finally:
            row["syncing"]["on"] = False

    def _build_negative_pane(self, vid, add_combo):
        """The pane for a difference map's negative contour, which is its own object.

        It has a row of its own in the panel, so it gets the tools of any drawn thing:
        style, color, opacity, and a level. Only what is genuinely one map stays on the
        map's own pane -- which part of the box is drawn, and at what sampling.

        The level is the one tie. "Link level with positive map" is on by default, which
        is the classic +/-n sigma reading of a difference map; unticking it lets the two
        contours sit at strengths of their own, because the features they answer to --
        density the model does not account for, and a model sitting on nothing -- are
        not obliged to be equally strong.
        """
        from PySide6.QtWidgets import QCheckBox

        live = self._desktop.volume_appearance(vid)
        linked = live.get("negative_iso") is None

        add_combo("Style", _VOLUME_STYLE_OPTIONS, live.get("negative_style"),
                  lambda v: self._safe(
                      lambda: self._desktop.set_volume_negative_style(vid, v)))
        self._add_color_row(
            live.get("negative_color"),
            lambda c: self._safe(
                lambda: self._desktop.set_volume_negative_color(vid, c)),
            title="Negative-contour color")
        self._add_opacity_row(
            live.get("negative_opacity"),
            lambda v: self._safe(
                lambda: self._desktop.set_volume_negative_opacity(vid, v)))

        check = QCheckBox("Link level with positive map")
        check.setChecked(linked)
        check.setToolTip(
            "Contour both halves of the difference at the same strength — the usual "
            "reading. Untick to give this contour a level of its own.")
        self._appearance_layout.addWidget(check)

        def _set_neg_iso(v):
            if check.isChecked():
                return      # linked: this row is only reporting the map's level
            self._safe(lambda: self._desktop.set_volume_negative_iso(vid, float(v)))

        row = self._add_iso_row(live.get("iso") if linked else live.get("negative_iso"),
                                _set_neg_iso, max_sigma=live.get("max_sigma"))
        row["slider"].setEnabled(not linked)
        row["spin"].setEnabled(not linked)

        def _toggle(state):
            on = bool(state)
            row["slider"].setEnabled(not on)
            row["spin"].setEnabled(not on)
            if on:
                # Back to the map's level, and show the number it is actually going to.
                self._show_iso_value(row, self._desktop.volume_appearance(vid).get("iso")
                                     or 0.0)
                self._safe(lambda: self._desktop.set_volume_negative_iso(vid, None))
            else:
                self._safe(lambda: self._desktop.set_volume_negative_iso(
                    vid, float(row["spin"].value())))

        check.toggled.connect(_toggle)
        self._neg_iso_row = row

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
        self._show_iso_value(self._iso_row, value)
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

    def _add_color_row(self, current, on_pick, *, themes=None, title="Color",
                       label="Color"):
        """A color control: optional color-by themes, then swatches, then a picker.

        Colors are shown rather than named — a swatch says what a hex is and the word does
        not. The picker is the escape hatch, since the wire takes any hex Mol* can decode,
        not just the presets.

        ``themes`` are leading (label, value) entries for color-*by* schemes (by element, by
        chain, …). A model gets them; a volume, whose density has nothing to color by, does
        not — but both get the swatches and the wheel, so either can be set to a flat color.
        """
        from PySide6.QtCore import QSize, Qt, QTimer
        from PySide6.QtGui import QColor, QIcon, QPixmap
        from PySide6.QtWidgets import QColorDialog, QComboBox, QHBoxLayout, QLabel

        def swatch(name):
            pixmap = QPixmap(28, 14)
            pixmap.fill(QColor(name))
            return QIcon(pixmap)

        themes = list(themes or [])
        theme_values = {value for _label, value in themes}

        row = QHBoxLayout()
        lab = QLabel(label)
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
        # is canceled, since by then the live preview has already changed the map.
        committed = {"value": current}

        def picked(_index, combo=combo):
            value = combo.currentData()
            if value != _CUSTOM_COLOR:
                committed["value"] = value
                on_pick(value)
                return
            # Open the picker only once this signal emission has unwound. exec() runs a
            # nested event loop, which processes deferred deletions -- and the live
            # preview below rebuilds the appearance pane, deleting this very combo. Doing
            # that *inside* the emission leaves QComboBox's own C++ frame standing on
            # freed memory when the handler returns: an intermittent segfault (measured
            # here as one crash in three runs of the exercise below), and the other half
            # of the crash report that produced the guard further down. Returning first
            # means nothing is inside the combo when it goes.
            QTimer.singleShot(0, open_picker)

        def open_picker(combo=combo):
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
            accepted = dialog.exec() == QColorDialog.DialogCode.Accepted
            name = dialog.selectedColor().name() if accepted else revert_to
            if accepted:
                committed["value"] = name             # '#rrggbb', which Mol* decodes
            import shiboken6

            if not shiboken6.isValid(combo):
                # The live preview can rebuild the appearance pane under this handler:
                # applying a color while a color-by-CC/resolution theme is on turns
                # that flag off, which emits loaded-changed, which rebuilds the pane
                # and deletes the combo mid-``exec()``. The rebuilt pane already shows
                # the entry's current state; only the final apply/undo remains to do.
                on_pick(name)
                return
            if accepted:
                at = combo.count() - 1
                combo.insertItem(at, swatch(name), name, name)
                combo.setCurrentIndex(at)
            else:
                # Canceled: undo the preview and put the selection back where it was.
                back = combo.findData(revert_to)
                combo.setCurrentIndex(back if back >= 0 else 0)
            combo.blockSignals(False)
            on_pick(name)

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
        lab = QLabel("Clip slab")
        lab.setMinimumWidth(80)
        row.addWidget(lab)
        slider = _make_range_slider()()
        slider.setToolTip(
            "Front and rear clipping planes for this object. Drag the handles to slice "
            "into it, or the span between them to move the slab. The slab follows the "
            "camera and composes with the Clip sphere below — slicing the depth does "
            "not lift a sphere, and lifting a sphere does not reopen the slab.")
        slider.set_values(front, back)
        slider.changed.connect(on_change)
        row.addWidget(slider, stretch=1)
        self._appearance_layout.addLayout(row)
        return slider

    def _add_sphere_row(self, sphere, allow_selection, on_pick, on_radius):
        """The clip sphere — the *place* clip, complementing the Clip slab row's depth.

        Where the sphere sits is the mode. "On selection" is both the policy and
        the readout: armed, every selection fits the sphere to its components;
        lifted by a clear, re-fitted by the next selection. "Around view center"
        is the classic map bound (it follows the camera); "Fixed point" freezes
        it. The radius is the sphere's reach in Å — for a selection it is fitted
        (components + the clip padding), so it reads rather than edits there.
        """
        from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel

        sphere = dict(sphere or {})
        mode = sphere.get("mode") or "off"
        row = QHBoxLayout()
        lab = QLabel("Clip sphere")
        lab.setMinimumWidth(80)
        row.addWidget(lab)
        combo = QComboBox()
        options = [("Off", "off")]
        if allow_selection:
            options.append(("On selection", "selection"))
        options += [("Around view center", "view"), ("Fixed point", "point")]
        for text, value in options:
            combo.addItem(text, value)
        idx = combo.findData(mode)
        if idx < 0:
            idx = 0   # a mode this row doesn't offer (shouldn't happen)
        combo.setCurrentIndex(idx)
        radius = QDoubleSpinBox()
        radius.setRange(0.5, 500.0)
        radius.setDecimals(1)
        radius.setSingleStep(1.0)
        radius.setSuffix(" Å")
        radius.setValue(float(sphere.get("radius") or 10.0))
        radius.setEnabled(mode in ("view", "point"))
        radius.setToolTip(
            "The sphere's reach. 'On selection' sizes it to the selection plus the "
            "clip padding (Settings ▸ Selection); edit it under 'Around view center' "
            "or 'Fixed point'.")
        combo.setToolTip(
            "Draw only what is inside a sphere: 'On selection' fits each new "
            "selection's components (the default — uncheck by picking Off), "
            "'Around view center' follows the camera's target like Coot's map "
            "radius, 'Fixed point' freezes the sphere where it is. The Clip slab "
            "above still applies on top.")
        combo.currentIndexChanged.connect(
            lambda _i, c=combo, r=radius: (r.setEnabled(c.currentData() in ("view", "point")),
                                           on_pick(c.currentData(),
                                                   r.value() if c.currentData() != "off" else None)))
        radius.valueChanged.connect(
            lambda v, c=combo: on_radius(float(v)) if c.currentData() in ("view", "point")
            else None)
        row.addWidget(combo)
        row.addWidget(radius)
        self._appearance_layout.addLayout(row)
        return combo, radius

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
            ("Ctrl + R", "Arm / disarm Refine drag"),
            ("Refine drag mode", "Pull and minimize an atom"),
            # Space did not appear here at all, and now that what it steps depends on
            # what has focus it needs saying (see DesktopApp.step_next).
            ("space", "Next residue — or next row, in a table"),
            ("shift + space", "The previous one"),
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

    def _add_value_scale_group(self, title, domain, unit, on_domain, on_fit, on_reset,
                               fit_label="Fit to values", fit_tip=""):
        """One framed value-scale panel: Range spins, Fit, Reset. The single control for
        every user-scaled coloring -- the map's local-resolution surface and a model's
        B-factor/occupancy coloring build the identical group, so a range means the
        same thing wherever it appears (lo draws blue, hi draws red, values clamp,
        fixed until the user moves it)."""
        from PySide6.QtWidgets import (QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
                                       QPushButton, QVBoxLayout)

        group = QGroupBox(title)
        gl = QVBoxLayout(group)
        gl.setSpacing(6)
        rr = QHBoxLayout()
        lab = QLabel("Range")
        lab.setMinimumWidth(80)
        lab.setToolTip("The values the color ramp spans; values outside clamp. "
                       "Fixed until you change it.")
        rr.addWidget(lab)
        lo_spin, hi_spin = QDoubleSpinBox(), QDoubleSpinBox()
        lo_spin.setObjectName("value-domain-lo")
        hi_spin.setObjectName("value-domain-hi")
        lo0, hi0 = domain or (0.0, 1.0)
        # A correlation runs to -1, so the floor cannot be zero the way a B-factor's or a
        # resolution's may as well be; and a quantity living inside a couple of units
        # (occupancy, Q-score, CC) needs a finer wheel than one spanning tens.
        step = 0.01 if (float(hi0) - float(lo0)) <= 2.0 else 0.1
        for sp in (lo_spin, hi_spin):
            sp.setRange(-9999.0, 9999.0)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(unit)
        lo_spin.setValue(float(lo0))
        hi_spin.setValue(float(hi0))

        def _apply(_=None):
            lo_v, hi_v = lo_spin.value(), hi_spin.value()
            if hi_v <= lo_v:
                return  # half-edited state; applied once the other end moves
            on_domain(lo_v, hi_v)

        lo_spin.valueChanged.connect(_apply)
        hi_spin.valueChanged.connect(_apply)
        rr.addWidget(lo_spin)
        rr.addWidget(QLabel("–"))
        rr.addWidget(hi_spin)
        fit = QPushButton(fit_label)
        if fit_tip:
            fit.setToolTip(fit_tip)
        fit.clicked.connect(lambda _=False: on_fit())
        reset = QPushButton("Reset")
        reset.setToolTip("Back to the default range.")
        reset.clicked.connect(lambda _=False: on_reset())
        rr.addWidget(fit)
        rr.addWidget(reset)
        rr.addStretch(1)
        gl.addLayout(rr)
        self._appearance_layout.addWidget(group)
        return group

    def _add_iso_row(self, current, on_change, max_sigma=None, label="Level",
                     tail=None):
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
        row.addWidget(QLabel(label))
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
        if tail is not None:
            row.addWidget(tail)
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
            "Collapse every Geometry table to the current selection: the Components "
            "table to the residues it touches, the Atoms table to the selected atoms, "
            "and each restraint table to the restraints whose atoms are all selected."
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
        subtabs.addTab(self._build_components_subtab(), "Components")
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

    def _engage_step_view(self, view) -> None:
        """Remember a table as Space's target: the last one the user engaged keeps
        the key, so leaving its tab (or clicking the viewport) does not demote the
        worklist back to the residue walk.

        The engaged table is also the selection's granularity: an atom-level
        worklist (the Atoms table, or a restraint table — a restraint row's unit
        IS its participating atoms) implies the next viewport click wants one
        atom, a residue-level one the whole residue — so the pick-granularity
        control follows it (and doubles as the readout of which the next click
        will do). Atom is the default before any table has been engaged."""
        self._desktop._step_view = view
        combo = getattr(self, "_pick_granularity", None)
        if combo is not None:
            atom_level = (view is self._atom_view or
                          any(view is info["view"]
                              for info in getattr(self, "_restraint_tabs", {}).values()))
            combo.setCurrentIndex(0 if atom_level else 1)

    def _on_geometry_subtab_changed(self, index: int) -> None:
        if index >= self._restraint_subtab_start:  # a restraint tab
            self._ensure_restraints()
        # Space steps the focused table; the table you can see is the one that
        # should answer, without having to click into it first. Switching the
        # sub-tab is itself the deliberate engagement that moves the key to it.
        view = self._geometry_subtab_view(index)
        if view is not None:
            self._engage_step_view(view)
            view.setFocus()

    def _geometry_subtab_view(self, index: int):
        """The table a Geometry sub-tab is about, if it has one."""
        if index == 0:
            return self._component_view
        if index == 1:
            return self._atom_view
        infos = list(self._restraint_tabs.values())
        restraint = index - self._restraint_subtab_start
        return infos[restraint]["view"] if 0 <= restraint < len(infos) else None

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

    def _coordinates_moved(self, mid) -> None:
        """A model's atoms moved: the cached geometry describes the old sites.

        Unlike :meth:`_on_restraints_changed` the restraint *set* did not change —
        only the measured values — so the cache drop is what matters; the tables
        refill now if one is on screen, or on next view either way.
        """
        self._geo_cache.pop(mid, None)
        if mid == self._table_model_id:
            self._invalidate_restraints()

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
        self._engage_step_view(info["view"])
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
            kernel = self._desktop.ensure_kernel()
            self._console = console_mod.EmbeddedConsole(
                kernel, banner=console_mod.default_banner()
            )
            self._console_layout.addWidget(self._console.widget)
        except Exception as exc:  # a broken console must not take the app down
            self._console_layout.addWidget(QLabel(f"Console failed to start:\n{exc}"))

    def _refresh_console_session(self) -> None:
        """Keep the kernel's ``session`` bound to the active model.

        Pushes whether or not the Console tab was ever opened — the kernel is
        the shared namespace for the console, MCP exec and attached Jupyter
        clients alike.
        """
        self._desktop.kernel_push({"session": self._desktop.active_model_session()})

    def shutdown_console(self) -> None:
        """Detach this client and stop the shared kernel (called on app quit)."""
        if self._console is not None:
            self._console.shutdown()
            self._console = None
        self._desktop.shutdown_kernel()

    def _build_components_subtab(self):
        """The Components table: every residue group in the model, one row each.

        The residue walk rendered as a list -- stepping it with Space is the
        same gesture as anywhere else, and a picked or typed residue lights its
        row here. Follows the same model as the Atoms table (the Model dropdown
        there governs every Geometry table).
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QLabel,
            QTableView,
            QVBoxLayout,
            QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        self._components_count = QLabel("No structure loaded")
        layout.addWidget(self._components_count)

        self._component_model = _make_component_table_model()
        view = QTableView()
        self._component_view = view
        view.setModel(self._component_model)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        view.verticalHeader().setVisible(False)  # the "#" column is the component index
        view.setAlternatingRowColors(True)
        view.setWordWrap(False)
        view.horizontalHeader().setStretchLastSection(True)
        view.selectionModel().selectionChanged.connect(
            lambda *_: self._on_component_selection_changed())
        layout.addWidget(view, stretch=1)

        # Table -> viewer selection is debounced so a drag doesn't flood the socket.
        self._component_sync_timer = QTimer()
        self._component_sync_timer.setSingleShot(True)
        self._component_sync_timer.setInterval(60)
        self._component_sync_timer.timeout.connect(self._push_component_selection_to_viewer)
        return tab

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
        # Model only by default: most fetches are "show me this structure", and a
        # ticked-by-default map turns a crystal-entry fetch into an error and an EM
        # fetch into an unasked-for 50+ MB download.
        checks["model"].setChecked(True)
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

    def _on_real_space_difference(self) -> None:
        try:
            self._desktop.compute_real_space_difference()
        except Exception as exc:
            self._set_status(str(exc))

    def _on_localres_wizard(self) -> None:
        """Color a map by local resolution computed from its two half-maps.

        The half-maps are read in cctbx only (never shown); the resulting resolution map is
        pinned, hidden, under the full map, and the map is colored by it — a toggle that
        then lives in the map's own appearance controls. Inputs are either local files or
        fetched from the PDB/EMDB (the same computation either way)."""
        from PySide6.QtWidgets import (
            QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
            QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton,
            QVBoxLayout, QWidget)

        maps_filter = "Maps (*.mrc *.map *.ccp4);;All files (*)"
        dialog = QDialog(self._window)
        dialog.setWindowTitle("Color by local resolution")
        outer = QVBoxLayout(dialog)
        intro = QLabel("Compute local resolution from two half-maps and color a map by it.")
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

    def _on_save_view(self) -> None:
        """Write the scene as a view file: JSON naming everything on screen, plus
        a data directory the checkbox controls — bundled (self-contained, for
        archiving and figures) by default, or references to the files objects
        came from (light, when one dataset backs many views)."""
        from PySide6.QtWidgets import (
            QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
            QMessageBox)

        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save view", "view.pxview.json",
            "pxviewer view (*.pxview.json)")
        if not path:
            return
        dialog = QDialog(self._window)
        dialog.setWindowTitle("Save view")
        form = QFormLayout(dialog)
        bundle = QCheckBox(
            "Bundle all data files into the view (self-contained)")
        bundle.setChecked(str(self._desktop._settings.value(
            "view/bundle_data", "true")).lower() != "false")
        bundle.setToolTip(
            "Checked: every object's coordinates and maps are written beside the "
            "file, so the view reproduces exactly anywhere — the figure case.\n"
            "Unchecked: objects loaded from files are referenced by path instead "
            "— light, when many views share one dataset that stays put.")
        form.addRow(bundle)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._desktop._settings.setValue(
            "view/bundle_data", "true" if bundle.isChecked() else "false")
        try:
            from . import viewstate
            viewstate.save_view(self._desktop, path,
                                bundle_data=bundle.isChecked())
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not save view", str(exc))

    def _on_open_view(self) -> None:
        """Restore a saved view file — its objects, their appearance, selections,
        clip state and camera."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getOpenFileName(
            self._window, "Open saved view", "",
            "pxviewer view (*.pxview.json *.json);;All files (*)")
        if not path:
            return
        d = self._desktop
        if d._models or d._volumes or d._reflections:
            # Restoring replaces the scene (a view is a whole-scene snapshot),
            # and what it replaces may hold unsaved edits — ask first.
            answer = QMessageBox.question(
                self._window, "Open saved view",
                "Opening a view replaces everything in the current scene. Continue?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            from . import viewstate
            viewstate.load_view(self._desktop, path)
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not open view", str(exc))
            return
        self._file_label.setText(f"{Path(path).name}  (view)")

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
        """Offer to build dictionaries for a model's unrecognized ligands. True if any were.

        The reason this is a prompt and not automatic: the dictionary is *inferred*. rdkit
        reads the bonds out of the coordinates and the bond orders out of the graph, which
        is a guess about chemistry the file never stated. A well-modeled ligand it gets
        right; a badly modeled one it gets wrong in a way that then looks authoritative,
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
            "perceives — <b>not</b> from the geometry as modeled. Check what it perceived "
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
            n = self._desktop.select_by_expression(
                expr, focus=self._focus_on_select.isChecked(),
                context=self._context_on_select.isChecked())
        except Exception as exc:  # invalid syntax / no model
            self._selection_label.setText(
                f"<span style='color:{_accent(self._window, 'error')}'>{exc}</span>")
            return
        # Which model the box text describes — so removing that model can clear it
        # (a leftover expression about a gone model reads as a live selection).
        self._select_expr_model = (
            self._desktop._active_model_id if expr.strip() else None)
        self._selection_label.setText("selection cleared" if not expr.strip() else f"{n} atom(s) selected")

    def _on_stop_demo(self) -> None:
        self._desktop.stop_demo()

    def _on_toggle_select(self, checked: bool) -> None:
        if checked:
            self._refine_drag_btn.setChecked(False)
            self._desktop.enable_mouse_selection()
        else:
            self._desktop.disable_mouse_selection()
        self._update_mode_chip()

    def _on_toggle_refine_drag(self, checked: bool) -> None:
        if checked:
            self._pick_btn.setChecked(False)
            # Arming the drag is the moment to sort restraints out: the drag itself builds
            # them on its own thread, where a failure can only be reported, not fixed.
            self._offer_restraints_if_blocked()
        self._desktop.set_tug_enabled(checked)
        # The options matter while the mode is on; the summary is what stands in for them
        # when it is off, so exactly one of the two is ever showing.
        self._tug_options.setVisible(checked)
        self._tug_summary.setVisible(not checked)
        self._update_mode_chip()

    def _update_mode_chip(self) -> None:
        """Name the armed mouse mode in the status row, or hide the chip if none is.

        Pick and Refine drag are set on a tab and then used in the viewport, with the tab
        long since switched away from — so until now nothing on screen said what a click
        would do. The status line is the one strip visible from every tab.
        """
        if self._refine_drag_btn.isChecked():
            text, accent = "Refine drag", "stop"
        elif self._pick_btn.isChecked():
            text, accent = "Pick", "go"
        else:
            self._mode_chip.setVisible(False)
            return
        color = _accent(self._mode_chip, accent)
        self._mode_chip.setText(text)
        self._mode_chip.setStyleSheet(
            "QLabel { border: 1px solid %s; border-radius: 4px; padding: 1px 6px;"
            " color: %s; }" % (color, color))
        self._mode_chip.setVisible(True)

    def _on_clear_selection(self) -> None:
        self._desktop.clear_selection()

    def _on_measure(self, kind: str) -> None:
        try:
            self._set_status(self._desktop.measure_selection(kind))
        except Exception as exc:
            self._set_status(str(exc))

    def _on_clear_measurements(self) -> None:
        self._desktop.clear_measurements()

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
        self._update_mode_chip()  # likewise a baked accent, not a palette() stylesheet

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

    @staticmethod
    def _wrapped_label(text: str):
        """A word-wrapped QLabel that is actually given the height its text needs.

        ``QLabel.heightForWidth`` under-reports a wrapped paragraph by a line at some
        widths — measured here at 410px: it returns 80 where laying the same text out
        needs 88 — so the label is sized a line short and the last sentence is
        silently clipped. Nothing warns; the paragraph simply ends mid-word, which is
        how the hotspot units note lost its conclusion. So the height is measured from
        the laid-out document and kept in step with whatever width we are given.
        """
        import math

        from PySide6.QtGui import QTextDocument
        from PySide6.QtWidgets import QLabel, QSizePolicy

        class _Wrapped(QLabel):
            def resizeEvent(self, event):
                super().resizeEvent(event)
                width = max(1, self.contentsRect().width())
                doc = QTextDocument()
                doc.setDefaultFont(self.font())
                # Qt::mightBeRichText is not exposed in PySide6; these labels are
                # authored here, so a tag is the whole test.
                content = self.text()
                if "<" in content and ">" in content:
                    doc.setHtml(content)
                else:
                    doc.setPlainText(content)
                doc.setTextWidth(width)
                needed = int(math.ceil(doc.size().height()))
                if needed != self.minimumHeight():
                    self.setMinimumHeight(needed)   # guarded: a relayout loop otherwise

        label = _Wrapped(text)
        label.setWordWrap(True)
        policy = label.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        policy.setHeightForWidth(True)
        label.setSizePolicy(policy)
        return label

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

    def _make_text_button(self, text, tooltip, *, checkable=False):
        """A button that says what it does, styled like the icon buttons beside it.

        For actions no glyph names honestly. "Difference" and "Local resolution" are
        whole operations with wordy results; a picture of either is a guess the reader
        has to decode, and the two we had were a merge symbol and a palette. Deliberate
        text, rather than the text :meth:`_make_icon_button` falls back to when an icon
        asset happens to be missing -- which is how this row ended up with one worded
        button beside one glyph.
        """
        from PySide6.QtWidgets import QPushButton

        b = QPushButton(text)
        b.setCheckable(checkable)
        b.setStyleSheet(_icon_button_base_qss())
        b.setToolTip(tooltip)
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
    def _coach_markup(text: str, label) -> str:
        """Tiny markdown → HTML for the coach: **bold**, `code`, blank-line breaks,
        and ``{icon:name}`` — the icon the named button wears, embedded inline."""
        import html
        import re

        t = html.escape(text)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        t = re.sub(r"\{icon:([\w-]+)\}",
                   lambda m: _icon_img_tag(label, m.group(1)), t)
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
        self._fix_coach_height(tutorial_obj)
        self._desktop._viewport.coach_bar.setVisible(True)
        if self._tutorial_timer is None:
            self._tutorial_timer = QTimer(self._window)
            self._tutorial_timer.setInterval(400)  # poll the step's done() predicate
            self._tutorial_timer.timeout.connect(self._poll_tutorial_done)
        self._tutorial_timer.start()
        self._show_tutorial_step()

    def _fix_coach_height(self, tutorial_obj) -> None:
        """Reserve the tallest step's text height up front, so the coach pane keeps one
        height for the whole tutorial instead of resizing -- and jiggling the viewer
        above it -- on every Next. Measured once at start against the current window
        width; a mid-tutorial window resize re-flows text but does not re-reserve.
        """
        vp = self._desktop._viewport
        label = vp.coach_text
        # The bar is still hidden, so its own geometry is stale; the pane spans the
        # viewport window minus the layout's 14px side margins. QTextBrowser has no
        # heightForWidth, so measure the document itself at that width.
        width = max(vp.coach_bar.parentWidget().width() - 28, 200)
        doc = label.document()
        tallest = 0
        for step in tutorial_obj.steps:
            label.setHtml(self._coach_markup(step.text, label))
            doc.setTextWidth(width)
            tallest = max(tallest, int(math.ceil(doc.size().height())))
        label.setMinimumHeight(tallest)

    def _show_tutorial_step(self) -> None:
        self._stop_highlight()  # clear any flash from the previous step
        tut = self._tutorial
        if tut is None:
            return
        step = tut.steps[self._tutorial_step]
        vp = self._desktop._viewport
        vp.coach_title.setText(tut.title)
        vp.coach_progress.setText(f"Step {self._tutorial_step + 1} / {len(tut.steps)}")
        vp.coach_text.setHtml(self._coach_markup(step.text, vp.coach_text))
        vp.coach_show.setVisible(step.target is not None)  # "Show me where" only if targeted
        vp.coach_back.setEnabled(self._tutorial_step > 0)
        self._tutorial_step_satisfied = False
        self._sync_step_button()

    def _sync_step_button(self) -> None:
        """Name the right-side button for the current step: Next (or Finish on the
        last) once the step's task is done — or when there is nothing to wait for —
        and Skip while a doable predicate is still unsatisfied. The name is the
        acknowledgement the poll gives: it never moves the coach itself."""
        if self._tutorial is None:
            return
        step = self._tutorial.steps[self._tutorial_step]
        last = self._tutorial_step == len(self._tutorial.steps) - 1
        skip = step.done is not None and not self._tutorial_step_satisfied
        self._desktop._viewport.coach_next.setText(
            "Skip" if skip else ("Finish" if last else "Next"))

    def _poll_tutorial_done(self) -> None:
        """Poll the step's done() predicate. The predicate *acknowledges* — flash the
        status and swap Skip for Next on the false->true edge — but the coach never
        advances itself: pressing Enter in a neighboring field, for instance, must
        not turn a finished step into a surprise page turn. Next is always the
        user's click."""
        if self._tutorial is None:
            return
        step = self._tutorial.steps[self._tutorial_step]
        if step.done is None:
            return
        try:
            satisfied = bool(step.done(self))
        except Exception:  # pragma: no cover - a predicate touching not-yet-ready state
            satisfied = False
        if satisfied == self._tutorial_step_satisfied:
            return
        self._tutorial_step_satisfied = satisfied
        self._sync_step_button()
        if satisfied:
            self._flash_status("✓ step done")

    def _advance_tutorial(self) -> None:
        if self._tutorial is None:
            return
        if self._tutorial_step >= len(self._tutorial.steps) - 1:
            self._tutorial_exit(finished=True)
            return
        self._tutorial_step += 1
        self._show_tutorial_step()

    def _tutorial_next(self) -> None:
        self._advance_tutorial()

    def _tutorial_back(self) -> None:
        """Step back for a re-read. The poll is edge-triggered, and this step is
        latched to its predicate's *present* value: a still-done step keeps reading
        Next rather than flipping back to Skip, and one not yet done goes back to
        acknowledging when it genuinely becomes done."""
        if self._tutorial is not None and self._tutorial_step > 0:
            self._tutorial_step -= 1
            self._show_tutorial_step()
            step = self._tutorial.steps[self._tutorial_step]
            try:
                self._tutorial_step_satisfied = (
                    bool(step.done(self)) if step.done is not None else False)
            except Exception:
                self._tutorial_step_satisfied = False
            self._sync_step_button()

    def _tutorial_exit(self, finished: bool = False) -> None:
        self._tutorial = None
        if self._tutorial_timer is not None:
            self._tutorial_timer.stop()
        self._stop_highlight()
        self._desktop._viewport.coach_bar.setVisible(False)
        self._desktop._viewport.coach_text.setMinimumHeight(0)  # release the reserve
        self._set_status("Tutorial complete. Nicely done." if finished else "Tutorial closed.")

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
        """Switch every tab bar between ``widget`` and the panel so it is visible.

        Walked from the widget upward rather than reading self._tabs: a coach target
        can sit inside a *nested* tab row (the Hotspots controls live in a Validation
        sub-tab), and revealing only the top level would flash a control on a page
        that is not showing."""
        from PySide6.QtWidgets import QTabWidget

        node = widget.parentWidget() if widget is not None else None
        while node is not None:
            if isinstance(node, QTabWidget):
                for i in range(node.count()):
                    page = node.widget(i)
                    if page is widget or page.isAncestorOf(widget):
                        node.setCurrentIndex(i)
                        break
            node = node.parentWidget()

    def reflect_dock_state(self, floating: bool) -> None:
        """Keep the dock/detach button in step with the panel's state:
        picture-in-picture-2 to detach while docked, minimize-2 to re-dock
        while floating."""
        icon = self._icon(
            "minimize-2" if floating else "picture-in-picture-2")
        if icon is not None:
            self._dock_btn.setIcon(icon)
        else:
            self._dock_btn.setText("Dock" if floating else "Detach")
        self._dock_btn.setToolTip(
            "Re-dock the controls" if floating else "Detach the controls to their own window")

    def _on_analysis_ready(self, mid) -> None:
        """Analysis finished: the dots are drawn, so the toggles enable and check."""
        self._sync_probe_toggles()  # reads the sessions' payloads: both just drew
        self._sync_all_markup_button()  # these are markup too, and now there is some

    def _on_scene_selection_changed(self, payload) -> None:
        """A model's picks changed. Refresh the aggregate label + the atoms table.

        ``payload`` is ``{"scene": {model_id: [indices]}, "changed": model_id}`` —
        ``changed`` names the model whose picks just changed. A bare scene dict is
        also accepted (older callers); then only a sole owner can be inferred."""
        if isinstance(payload, dict) and "scene" in payload:
            scene, changed = payload.get("scene") or {}, payload.get("changed")
        else:
            scene, changed = payload or {}, None
        self._scene_selection = scene
        total = sum(len(v) for v in scene.values())
        # Hide/show-selected only make sense with a selection.
        self._hide_sel_btn.setEnabled(total > 0)
        self._show_sel_btn.setEnabled(total > 0)
        self._selection_label.setText(self._desktop.selection_description(scene))
        # Viewer -> Objects list: follow the model whose picks just changed, making its
        # row current (and it the active model) — the panel follows the user's
        # attention. Judged by WHO CHANGED, not by counting owners: sessions keep
        # independent selections, so a standing selection in one model must not veto
        # the sync when atoms are picked in another (it did — clicking the non-active
        # model moved nothing whenever the active one still had picks). A model whose
        # picks were just *cleared* moves nothing.
        if changed is None:
            owners = [mid for mid, indices in scene.items() if indices]
            changed = owners[0] if len(owners) == 1 else None
        if changed is not None and scene.get(changed):
            self._set_current_tree_row("model", changed)
        # Viewer -> Geometry: reflect the picks in the atoms + restraint tables.
        self._apply_geometry_filter()

    def _on_atom_picked(self, mid: str, index: int, shift: bool,
                        rep: str = "") -> None:
        """An atom of ``mid`` was clicked. With no click mode armed this IS a
        selection: a plain click replaces it through the same pipeline a typed
        expression takes — oriented framing, clip, neighborhood context, all
        under the Selection pane's checkboxes — and a SHIFT-click grows or
        shrinks it, camera left where it is. How much each click takes — the
        atom's whole residue, or just the atom — is the Selection pane's
        granularity control, which the last engaged table already set. Either way
        the box shows the equivalent expression. While Pick mode accumulates,
        refine-drag tugs, or a measurement is being placed, the click belongs to
        that tool, and the panel just follows the model.

        ``rep`` is the representation the click landed on. Atom granularity is
        only aimable where atoms are drawn: a ribbon face resolves to the
        residue's anchor atom whatever Mol* granularity says, so a cartoon hit
        promotes to the residue — the unit the click could actually have meant.
        A hit on the ball-and-stick neighborhood layer stays atom-precise."""
        desktop = self._desktop
        entry = desktop._model_entry(mid)
        tool_owns_click = (
            desktop._selection_enabled or desktop._tug_enabled
            or (entry is not None
                and getattr(entry["session"], "_click_mode", "off") != "off"))
        if tool_owns_click or index < 0:
            self._set_current_tree_row("model", mid)  # follow attention, touch nothing
            return
        granularity = self._pick_granularity.currentData()
        if granularity == "atom" and rep and not _rep_shows_atoms(rep):
            granularity = "residue"
        try:
            if shift:
                expression = desktop.toggle_picked_residue(
                    mid, index, granularity=granularity,
                    context=self._context_on_select.isChecked())
            else:
                expression = desktop.select_picked_atom(
                    mid, index, granularity=granularity,
                    focus=self._focus_on_select.isChecked(),
                    context=self._context_on_select.isChecked())
        except Exception:  # a click must never surface a selection error dialogically
            self._set_current_tree_row("model", mid)
            return
        # None: the click named no atom. "" is meaningful — the selection emptied, or
        # it is no longer expressible as whole residues; either way a stale expression
        # would lie, so the box follows.
        if expression is not None:
            self._select_expr.setText(expression)
            self._select_expr_model = mid if expression else None

    def _set_current_tree_row(self, kind: str, ident: str) -> None:
        """Point the object list's highlight at (kind, ident), if it has a row."""
        from PySide6.QtCore import Qt

        tree = self._loaded_tree
        current = tree.currentItem()
        if current is not None and current.data(0, Qt.ItemDataRole.UserRole) == (kind, ident):
            return
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            node = stack.pop()
            if node.data(0, Qt.ItemDataRole.UserRole) == (kind, ident):
                tree.setCurrentItem(node)  # _on_tree_current_changed does the rest
                return
            stack.extend(node.child(i) for i in range(node.childCount()))

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

    def _apply_component_selection(self) -> None:
        """Reflect the table model's selection in the Components list: filter the
        rows, or mark the components the selection touches."""
        indices = self._table_selection_indices()
        if self._filter_selection_check.isChecked():
            with self._table_sync_suppressed():
                self._component_model.set_filter(
                    self._component_model.components_with_atoms(indices))
            self._update_components_count()
        else:
            if self._component_model.is_filtered():
                with self._table_sync_suppressed():
                    self._component_model.set_filter(None)
                self._update_components_count()
            self._select_component_rows(indices)

    def _select_component_rows(self, atom_indices) -> None:
        """Mark the components the atoms belong to, without echoing back to the viewer."""
        from PySide6.QtCore import QItemSelection, QItemSelectionModel

        model = self._component_model
        view = self._component_view
        sm = view.selectionModel()
        with self._table_sync_suppressed():
            sm.clearSelection()
            ncols = model.columnCount()
            rows = model.rows_with_atoms(atom_indices)
            if rows and ncols:
                selection = QItemSelection()
                last = ncols - 1
                for start, end in _runs(rows):  # contiguous ranges keep this cheap
                    selection.select(model.index(start, 0), model.index(end, last))
                sm.select(selection, QItemSelectionModel.SelectionFlag.Select)
                view.scrollTo(model.index(rows[0], 0))

    def _update_components_count(self) -> None:
        n = self._component_model.rowCount()
        if self._component_model.is_filtered():
            self._components_count.setText(f"{n} selected component(s)")
        else:
            self._components_count.setText(f"{n} components" if n else "No structure loaded")

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
        self._apply_component_selection()  # the Components table
        self._apply_table_selection()      # the Atoms table
        self._apply_restraint_filter()     # Bonds / Angles / Dihedrals / Chirality / Planarity

    def _on_restraints_changed(self, mid) -> None:
        """A model's restraints were rebuilt: drop the cache and refill the tables.

        Without this an added restraint is real -- minimize honors it, and it is in the
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
        # Tick the row's "restraints" box without a whole-tree rebuild — and
        # keep the items list agreeing, since a later rebuild reads it for
        # tooltips/state outside _loaded_summary.
        for it in self._items:
            if it["kind"] == "model" and it["id"] == mid:
                it["restraints_built"] = True
        self._tick_restraints_box(mid, True)

    def _tick_restraints_box(self, mid, built: bool) -> None:
        """Set one model row's 'restraints' checkmark to the live build state."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QTreeWidgetItemIterator

        iterator = QTreeWidgetItemIterator(self._loaded_tree)
        while iterator.value():
            node = iterator.value()
            if node.data(0, Qt.ItemDataRole.UserRole) == ("model", mid):
                node.setCheckState(2, Qt.CheckState.Checked if built
                                   else Qt.CheckState.Unchecked)
                node.setToolTip(2,
                    "Geometry restraints built — bonds, angles and the "
                    "structural selection keywords are ready"
                    if built else
                    "Geometry restraints not built yet — they process on "
                    "first use (a drag, minimize, or a selection that needs "
                    "them)")
                return
            iterator += 1

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
        if mid == self._table_model_id:
            # Every loaded_changed funnels here; when the model did not actually
            # change (a rep switch, an eye toggle, ensure_atoms_shown's courtesy
            # flip on the first restraint click) the rebuild below would only
            # rebind the same session and reset the restraint tables' rows --
            # which discards the very row selection that triggered it.
            return
        self._table_model_id = mid
        session = self._desktop.session_for(mid)
        with self._table_sync_suppressed():
            self._atom_model.set_session(session)  # clears any filter
            self._component_model.set_session(session)
        self._update_atoms_count()
        self._update_components_count()
        self._apply_component_selection()
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
        """Widen the visibility column so a nested row's eye icon is not clipped.

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
        from PySide6.QtWidgets import QTreeWidgetItem

        groups = {g["id"]: g for g in summary.get("groups", [])}
        items = summary.get("items", [])
        self._items = items
        model_items = [it for it in items if it["kind"] == "model"]
        self._models_summary = model_items

        self._suppress_model_events = True
        try:
            self._loaded_tree.clear()
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
            # Eye / eye-off, tinted like the surrounding text (dimmed when hidden), so the
            # column reads as "visibility" without a legend — a checkbox only said
            # "enabled at something".
            from PySide6.QtGui import QPalette

            palette = self._loaded_tree.palette()
            # A selected row keeps the same dark tints: the tree's stylesheet paints
            # its selection in the pale highlight (see the tree setup), on which the
            # text-colored icon reads fine — Selected-mode pixmaps are passed
            # explicitly so the view never swaps in a faint variant of its own.
            text = palette.color(QPalette.ColorRole.Text)
            dim = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            eye_shown = _line_icon("eye", text, size=16, selected_color=text)
            eye_hidden = _line_icon("eye-off", dim, size=16, selected_color=dim)
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
                # [eye] col 0, [name] col 1 (elides).
                node = QTreeWidgetItem(parent)
                if it["kind"] == "volume":
                    vol_nodes[it["id"]] = node
                node.setData(0, Qt.ItemDataRole.UserRole, (it["kind"], it["id"]))
                if it["visible"] is None:
                    pass  # reflections: nothing drawable, so no eye to click
                elif it["kind"] in ("model", "volume", "negative") and not self._desktop._can_hide:
                    # Hiding is disabled on software WebGL (this VM's SwiftShader). The
                    # original reason — "hiding segfaults the software renderer" — turned out
                    # to be a misread of the object-tree use-after-free (the tree rebuilding
                    # itself from inside its own item signal), which crashed on every
                    # renderer because no renderer was involved. The block stays only
                    # because software rendering has not been re-tested since; it is likely
                    # safe to lift. Not a dead control: a click flashes why
                    # (see _on_tree_item_clicked).
                    if eye_shown:
                        node.setIcon(0, eye_shown)
                    node.setToolTip(0, "Hiding needs hardware WebGL "
                                       "(not available on software rendering)")
                else:
                    icon = eye_shown if it["visible"] else eye_hidden
                    if icon:
                        node.setIcon(0, icon)
                    node.setToolTip(0, "Shown — click the eye to hide" if it["visible"]
                                    else "Hidden — click the eye to show")
                # No marker suffix: its name ("Ligand marker N") already says what it is.
                suffix = {"volume": "   [map]", "reflections": "   [data]"}
                # Indent a group member's name by hand. Qt applies tree indentation to
                # column 0 only, so the checkbox shifts with depth but the name — the part
                # you actually read — sits at the same x whether the object is in a group or
                # standing alone, which made every object look like a group member. Indent
                # the name to match, and an object at the root reads as standing alone.
                indent = _GROUP_MEMBER_INDENT if (it["group"] or it.get("pinned_to")) else ""
                node.setText(1, indent + it["name"] + suffix.get(it["kind"], ""))
                node.setToolTip(1, it["name"])  # full name on hover when elided
                if it["kind"] == "model":
                    # [restraints] — a state readout, not a toggle (the
                    # checkable flag is stripped, so a click cannot lie about
                    # whether the build has run). Ticked once the model's
                    # restraints exist — usually seconds after load via the
                    # background warm, or after whichever use first needed them.
                    built = bool(it.get("restraints_built"))
                    node.setCheckState(
                        2, Qt.CheckState.Checked if built
                        else Qt.CheckState.Unchecked)
                    node.setText(2, "restraints")
                    node.setFlags(
                        node.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    node.setToolTip(2,
                        "Geometry restraints built — bonds, angles and the "
                        "structural selection keywords are ready"
                        if built else
                        "Geometry restraints not built yet — they process on "
                        "first use (a drag, minimize, or a selection that needs "
                        "them)")
                    # The active model is the last model row you touched; the selection
                    # box, atoms table and geometry act on it. Bold — a foreground accent
                    # would vanish against the row-highlight background when the active
                    # row is also the current one, which it usually is.
                    node.setToolTip(1, it["name"] + (
                        "\nActive model — the selection box, atoms table and geometry "
                        "act on it." if it.get("active")
                        else "\nClick to make this the active model."))
                    if it.get("active"):
                        font = node.font(1)
                        font.setBold(True)
                        node.setFont(1, font)
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
        self._update_tug_maps_after()
        self._update_pair_button()
        # Point the Appearance pane at the focused object. Focusing a model activates
        # it, so a focused *model* must always be the active one — if the active model
        # changed underneath us (a new model, a row click, hydrogenate+analyze),
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
        # The selection box describes one model's selection; when that model was
        # removed, the text would read as a live selection of nothing. Clear it and
        # let the label state what is actually selected now.
        if (self._select_expr_model is not None
                and self._select_expr_model not in {m["id"] for m in model_items}):
            self._select_expr_model = None
            self._select_expr.setText("")
            self._selection_label.setText(
                self._desktop.selection_description(self._scene_selection))
        self._sync_probe_toggles()  # the dots-owning model may have been removed

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
            # (see _on_tree_item_clicked). Defer past the signal.
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

    def _apply_visibility(self, kind: str, ident: str, visible: bool) -> None:
        """Apply a visibility toggle, off the tree's signal stack (see below)."""
        if kind == "model":
            self._desktop.set_model_visible(ident, visible)
        elif kind == "volume":
            self._desktop.set_volume_visible(ident, visible)
        elif kind == "negative":
            # Its own eye, independent of its map's: "just the green one" is a reading of
            # a difference map, and it survives the map being toggled.
            self._desktop.set_volume_negative_visible(ident, visible)
        elif kind == "marker":
            self._desktop.set_marker_visible(ident, visible)
        # reflections have no visibility to change

    def _on_tree_item_clicked(self, item, column: int) -> None:
        """A click on the eye column shows/hides the object — applied on the *next*
        event-loop turn, never inside this signal. Applying here would run
        ``_emit_loaded_changed`` -> ``_on_loaded_changed`` -> ``tree.clear()``, destroying
        the very item Qt is still delivering this click through; Qt then keeps using the
        freed item as the stack unwinds and the process dies with SIGSEGV. That — not the
        GPU — is what every past "hiding segfaults" report actually was, which is why it
        reproduced identically on software and hardware WebGL.

        On software WebGL hiding models and maps is refused (see _on_loaded_changed), so
        there the eye is not a dead control: a click flashes why, as a pure status flash —
        no viewer message, so it cannot itself crash."""
        from PySide6.QtCore import Qt, QTimer

        if column != 0 or self._suppress_model_events:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, ident = data
        it = self._find_item(kind, ident)
        if it is None or it.get("visible") is None:
            return  # group header, or reflections: nothing drawable to toggle
        if kind in ("model", "volume", "negative") and not self._desktop._can_hide:
            self._desktop._warn("Hiding needs hardware WebGL — not available on software "
                                "rendering.")
            return
        visible = not it["visible"]
        QTimer.singleShot(0, lambda: self._apply_visibility(kind, ident, visible))

    def _on_tree_row_double_clicked(self, item, column: int) -> None:
        """A double-click on a model's name is an *object select*: every atom selected,
        the camera on Mol*'s own whole-object framing — deliberately not the oriented
        fill-the-frame treatment a typed selection gets, so double-clicking a freshly
        opened model changes nothing visually. Called from the viewport event filter
        (see the tree setup), so it fires even when the first click's activation
        rebuilt the tree mid-gesture. Not on the eye column: a double-click there is
        two visibility toggles, not a select-all."""
        from PySide6.QtCore import Qt, QTimer

        if item is None or column == 0 or self._suppress_model_events:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, ident = data
        if kind != "model":
            return

        def _select_object(mid=ident):
            try:
                n = self._desktop.select_object(mid)
            except Exception as exc:
                self._selection_label.setText(
                    f"<span style='color:{_accent(self._window, 'error')}'>{exc}</span>")
                return
            self._select_expr.setText("all")
            self._select_expr_model = mid
            self._selection_label.setText(f"{n} atom(s) selected")
        QTimer.singleShot(0, _select_object)     # off the event, like every tree action

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

    def _on_component_selection_changed(self) -> None:
        if not self._suppress_table_sync:
            self._component_sync_timer.start()  # debounce a drag-select

    def _selection_flags(self):
        """The selection behavior switches, as (focus, context). Clip is not a
        pane switch any more — it is the object's Clip sphere mode, read at the
        selection sites themselves."""
        return (self._focus_on_select.isChecked(),
                self._context_on_select.isChecked())

    def _push_component_selection_to_viewer(self) -> None:
        """Component rows -> the viewer: select what the rows name.

        One row IS a residue, so it goes through the same pipeline as a
        space-bar step or a clicked residue -- the focused-residue marker, the
        scene selection, the framing -- and the atoms table echoes the same
        atoms. Several rows select their atoms together. Like a Validation row,
        the framing, clip sphere and neighborhood context answer to the
        Selection pane's checkboxes: stepping a list is still just selecting.
        """
        rows = [idx.row() for idx in self._component_view.selectionModel().selectedRows()]
        mid = self._table_model_id
        desktop = self._desktop
        self._engage_step_view(self._component_view)
        active = mid is not None and mid == desktop._active_model_id
        focus, context = self._selection_flags()
        if len(rows) == 1 and active:
            chain, resid = self._component_model.row_label(rows[0])
            if desktop.focus_residue(chain, resid,
                                     focus=focus, context=context) is not None:
                return
        atoms = [i for r in rows for i in self._component_model.row_atoms(r)]
        entry = desktop._model_entry(mid)
        session = entry["session"] if entry else None
        if session is not None and getattr(session, "model", None) is not None:
            desktop._select_fragment(mid, session, entry, atoms,
                                     focus=focus, context=context)
            if rows and active:
                # Keep the residue walk anchored at the last row chosen, so an
                # unfocused Space continues from here rather than an older pick.
                chain, resid = self._component_model.row_label(rows[-1])
                desktop._focused_residue = (chain.strip(), resid.strip())
        else:
            desktop.highlight_atoms_in(mid, atoms)
            desktop.focus_atoms_in(mid, atoms)

    def _push_table_selection_to_viewer(self) -> None:
        """Atom rows -> the viewer: the same selection a clicked atom makes.

        Model-backed sessions go through the fragment pipeline, so the rows
        honor the Selection pane's checkboxes like every other selection;
        model-less ones get the plain highlight-and-aim they always had.
        """
        rows = [idx.row() for idx in self._atom_view.selectionModel().selectedRows()]
        atoms = [self._atom_model.row_atom(r) for r in rows]
        mid = self._table_model_id
        self._engage_step_view(self._atom_view)
        entry = self._desktop._model_entry(mid)
        session = entry["session"] if entry else None
        if session is not None and getattr(session, "model", None) is not None:
            focus, context = self._selection_flags()
            self._desktop._select_fragment(mid, session, entry, atoms,
                                           focus=focus, context=context)
        else:
            self._desktop.highlight_atoms_in(mid, atoms)
            self._desktop.focus_atoms_in(mid, atoms)

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



class DesktopApp(AppCore):
    """The desktop shell: Qt widgets attached to the GUI-agnostic AppCore.

    This adds the QApplication, the viewport widget, the controls window and
    the dock/shortcut wiring. All semantic state and operations live in
    ``AppCore`` — methods here only build, arrange and tear down widgets.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5173,
                 can_hide: bool = False, mcp_port: "int | None" = None):
        _check_qt()

        from PySide6.QtWidgets import QApplication

        # Qt must be initialized before any widgets are created — and the
        # process must be renamed before QApplication/NSApplication init, which
        # snapshots the name the macOS menubar app menu will show.
        _set_macos_process_name()
        app = QApplication.instance()
        if app is None:
            # argv[0] of "pxviewer": an unbundled process derives its app name
            # from it — leaving the interpreter's argv would say "python" (or
            # "__main__" under -m) anywhere Qt falls back to the executable.
            app = QApplication(["pxviewer"])
        # Always the light theme, whatever the OS is set to. Dark mode never rendered
        # well here (half-dark panels, unreadable accents) and is not worth maintaining.
        # Pinned before any widget exists so nothing ever paints from a dark palette;
        # the palette-watch machinery stays for the one real event left, the initial
        # palette landing after show.
        from PySide6.QtCore import Qt as _Qt

        app.styleHints().setColorScheme(_Qt.ColorScheme.Light)
        # App identity. On Linux — Wayland especially — the launcher/taskbar finds an
        # app's icon by matching its running window to a .desktop file of this name;
        # setWindowIcon alone only covers the title bar. `pxviewer install-desktop-entry`
        # writes the matching file (StartupWMClass=pxviewer).
        app.setApplicationName("pxviewer")
        app.setApplicationDisplayName("pxviewer")
        app.setDesktopFileName("pxviewer")
        icon = _app_icon()  # dock/taskbar icon for the whole app
        if icon is not None:
            app.setWindowIcon(icon)
        # Before anything slow: the web engine and the Mol* bundle take seconds, and an
        # empty screen for that long looks like a launch that failed.
        self._splash = _show_splash()

        super().__init__(host=host, port=port, can_hide=can_hide,
                         mcp_port=mcp_port)

        self._viewport = ViewportWindow()
        self._controls = ControlsWindow(self)

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

        # Space / Shift+Space step to the next / previous one, from either window --
        # where "one" is whatever you are working in (see step_next).
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeySequence, QShortcut

        # One set per *window*, not per pane. The controls are a dock inside the main
        # window, so both panes live in the same window, and two WindowShortcuts on the
        # same sequence there are AMBIGUOUS: Qt emits activatedAmbiguously and fires
        # neither. Every one of these keys was dead -- space, shift+space and Ctrl+R --
        # from the moment the controls pane became a dock. Deduplicated by window, and
        # parented to the window itself so the context says what it means.
        windows = []
        for _w in (self._viewport.widget(), self._controls.widget()):
            window = _w.window()
            if any(w is window for w in windows):
                continue
            windows.append(window)
            nxt = QShortcut(QKeySequence(Qt.Key.Key_Space), window)
            nxt.setContext(Qt.ShortcutContext.WindowShortcut)
            nxt.activated.connect(lambda: self.step_next(1))
            prv = QShortcut(QKeySequence("Shift+Space"), window)
            prv.setContext(Qt.ShortcutContext.WindowShortcut)
            prv.activated.connect(lambda: self.step_next(-1))
            # Refine drag is a mode you leave and re-enter constantly while fitting, and
            # its switch is a tab away from the viewport where it is used.
            tug = QShortcut(QKeySequence("Ctrl+R"), window)
            tug.setContext(Qt.ShortcutContext.WindowShortcut)
            tug.activated.connect(self._controls._refine_drag_btn.click)

    def _pump_events(self) -> None:
        """Paint pending updates (status messages) before a blocking call."""
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def _focused_step_view(self):
        """The focused widget, if it is one of the stepping tables."""
        from PySide6.QtWidgets import QTableView
        view = self._app.focusWidget() if self._app is not None else None
        return view if isinstance(view, QTableView) else None

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

        self._start_services()
        self._dismiss_splash()
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




def run_desktop(host: str = "127.0.0.1", port: int = 5173,
                gpu: Optional[str] = None,
                mcp_port: "int | None" = None) -> int:
    """Start the desktop app with viewport and controls windows.

    With no display — or ``PXVIEWER_HEADLESS=1`` — there is no desktop to
    start: run the GUI-agnostic core instead (same AppCore, a headless
    Chromium for the Mol* page when playwright is installed). Nobody watches
    a spawned Xvfb, so headless never spins one up.
    """
    from . import gpu as gpu_backend

    headless_env = os.environ.get("PXVIEWER_HEADLESS", "").lower() in (
        "1", "true", "yes")
    # DISPLAY/WAYLAND_DISPLAY are X11/Wayland notions — macOS and Windows
    # always have a windowing system.
    have_display = sys.platform in ("darwin", "win32") or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless_env or not have_display:
        from .appcore import run_core
        return run_core(host=host, port=port, mcp_port=mcp_port)

    # Before any QtWebEngine exists: choose the GL backend. On a machine whose GPU
    # cannot provide WebGL (common on VMs) this arms a one-time restart into
    # software rendering rather than leaving the viewport blank. See pxviewer.gpu.
    mode = gpu_backend.configure(gpu)
    _check_qt()

    # Hide/show in place only on hardware WebGL: it is a live GPU state change, which a
    # software renderer (SwiftShader) segfaults on. Software and user-custom flags fall
    # back to the reload-based hide, which is slower to the eye but never disposes
    # anything mid-frame. See DesktopApp.__init__.
    desktop = DesktopApp(host=host, port=port, can_hide=(mode == "hardware"),
                         mcp_port=mcp_port)
    try:
        return desktop.start()
    finally:
        desktop.stop()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(run_desktop())
