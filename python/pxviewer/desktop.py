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
import os
import sys
import threading
import time
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

# Distinct default isosurface colours so overlaid volumes read apart.
_VOLUME_COLORS = ["gold", "dodgerblue", "salmon", "mediumseagreen", "orchid", "orange"]
# Sentinel for the "Custom…" entry in a colour dropdown (never a real colour value).
_CUSTOM_COLOR = "\x00custom"

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

# Least time between coordinate frames pushed during a drag. A drag can compute frames
# faster than the viewer renders them; unpaced, they back up and read as lag, so this
# caps the rate (~20 fps) and keeps the picture current rather than queued.
_TUG_PUSH_INTERVAL = 0.05

# How long the post-release wind-down plays for. The minimization itself converges in a
# fraction of a second; this stretches its states over a watchable settle so a released
# fling comes visibly to rest — the clearest signal that the fragment is done, not broken.
_TUG_SETTLE_DURATION = 1.2

# How much density to draw around the view centre, for maps that need it. A map made
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
# taking the whole pane.
_TREE_MIN_HEIGHT = 66
_TREE_MAX_HEIGHT = 320

# Inline representation dropdowns in the Loaded tree (models vs maps differ).
# The model values must be types the LiveSession API accepts (see live.py's
# _STRUCTURE_REPR_TYPES / _REPR_ALIASES) — test_model_rep_options_are_valid guards this.
_MODEL_REP_OPTIONS = [
    ("Cartoon", "cartoon"),
    ("Ball & stick", "ball-and-stick"),
    ("Spacefill", "spacefill"),
    ("Surface", "surface"),
]
_VOLUME_STYLE_OPTIONS = [
    ("Surface", "surface"),
    ("Wireframe", "wireframe"),
    ("Mesh", "mesh"),
]
_MODEL_COLOR_OPTIONS = [
    ("Default", None),
    ("By element", "element-symbol"),
    ("By chain", "chain-id"),
    ("By secondary structure", "secondary-structure"),
    ("By residue", "residue-name"),
    ("By hydrophobicity", "hydrophobicity"),
]


def _model_rep_color(rep: str) -> str:
    """A sensible default colour theme for a representation type."""
    return "secondary-structure" if rep == "cartoon" else "element-symbol"


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
_SPLASH_SIDE = 320  # logical px; scaled from the 512px icon for the screen's pixel ratio
_SPLASH_MAX_MS = 15000  # never leave the splash up if the page never reports a load


def _app_icon():
    """The pxviewer window/dock icon as a QIcon, or None if the asset is missing."""
    from PySide6.QtGui import QIcon

    return QIcon(str(_ICON_PATH)) if _ICON_PATH.exists() else None


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
        interactions_changed = Signal(bool)
        structure_changed = Signal(object)  # the active LiveSession (or None)
        loaded_changed = Signal(object)     # {groups, items} for the Loaded tree
        run_on_main = Signal(object)        # call a thunk on the GUI thread
        analysis_ready = Signal(object)     # clash/contact analysis finished (model id)
        validation_ready = Signal(object)   # validation finished: (model id, [ValidationResult])
        minimizing_changed = Signal(bool)   # a minimization started (True) / finished (False)
        volume_iso_changed = Signal(object)  # (volume id, level) changed in the viewport

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
    from .geometry import geostd_monomer_path, monomer_library_root

    arrays = getattr(getattr(session, "_data", None), "arrays", None)
    if arrays is None:
        return lambda iseqs: ("", None)
    resname = arrays.resname
    root = monomer_library_root()
    cache: dict = {}

    def source(iseqs):
        names = {resname[i] for i in iseqs}
        if len(names) != 1:
            return ("(link)", None)  # spans residues -> a link, not one monomer file
        rn = next(iter(names))
        if rn not in cache:
            cache[rn] = (rn, geostd_monomer_path(root, rn))
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
                    return QColor("#2563eb")  # link blue
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
        layout.addWidget(self._view)
        self._view.loadFinished.connect(self._verify_webgl)

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
    """Controls for the viewport: open a file, or run a demo from the Demos tab."""

    def __init__(self, desktop: "DesktopApp", title: str = "pxviewer — controls"):
        _check_qt()

        from PySide6.QtWidgets import (
            QHBoxLayout,
            QLabel,
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

        layout = QVBoxLayout(self._window)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        self._console = None  # EmbeddedConsole, created lazily on first tab view
        self._console_started = False
        self._items: list = []  # last Loaded-tree items summary (for the appearance pane)
        self._focused: tuple = (None, None)  # (kind, id) currently shown in Appearance

        tabs = QTabWidget()
        tabs.addTab(self._build_scene_tab(), "Scene")
        tabs.addTab(self._build_tools_tab(), "Tools")
        tabs.addTab(self._build_validation_tab(), "Validation")
        tabs.addTab(self._build_geometry_tab(), "Geometry")
        console_tab = self._build_console_tab()
        self._console_tab_index = tabs.addTab(console_tab, "Console")
        tabs.addTab(self._build_settings_tab(), "Settings")
        # The console spins up an IPython kernel, so defer that cost until the tab
        # is actually opened.
        tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(tabs, stretch=1)

        # A slim, always-visible status line, with the app icon + Help on the far side.
        status_row = QHBoxLayout()
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #666;")
        status_row.addWidget(self._status_label, stretch=1)
        # Always-visible dock/detach control: the painted header's button is hidden while
        # the panel floats (it uses the native window frame then), so this is the reliable
        # way back — and the way out, from either state.
        self._dock_btn = QPushButton("Detach")
        self._dock_btn.setToolTip("Detach the controls to their own window, or re-dock them")
        self._dock_btn.clicked.connect(self._desktop.toggle_controls_dock)
        status_row.addWidget(self._dock_btn)
        help_btn = QPushButton("Help…")
        help_btn.setToolTip("Documentation (coming soon)")
        help_btn.clicked.connect(self._on_help)
        status_row.addWidget(help_btn)
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
        desktop.bridge.loaded_changed.connect(self._on_loaded_changed)
        desktop.bridge.analysis_ready.connect(self._on_analysis_ready)
        desktop.bridge.validation_ready.connect(self._on_validation_ready)
        desktop.bridge.minimizing_changed.connect(self._on_minimizing_changed)
        desktop.bridge.volume_iso_changed.connect(self._on_volume_iso_changed)
        self._update_minimize_map()  # nothing loaded yet, so no map to minimize into
        self._update_tug_density()
        self._update_pair_button()
        self._fit_tree_height()  # the empty list must not reserve space either
        self._update_appearance()  # empty-state placeholder

    # -- tabs ------------------------------------------------------------

    def _build_scene_tab(self):
        """Home: open files, the object list, appearance of the focused object, selection."""
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
        self._active_group = QButtonGroup(self._window)  # exclusive active-model radios
        self._active_group.setExclusive(True)
        self._active_group.buttonClicked.connect(self._on_active_radio)
        ol.addWidget(self._loaded_tree, stretch=1)

        # -- Actions on the objects: one grid, two rows of four ------------
        # Row 1 gets data in and out; row 2 acts on what is loaded and on the view.
        self._open_btn = QPushButton("Open…")
        self._open_btn.setToolTip("Open a structure or map (models via cctbx; maps as .mrc/.map/.ccp4)")
        self._open_btn.clicked.connect(self._on_open_file)
        demos_btn = QPushButton("Demos")  # a menu button (native dropdown arrow)
        demos_btn.setToolTip("Load a bundled example to try things out")
        demos_btn.setMenu(self._build_demos_menu())
        self._write_btn = QPushButton("Write…")
        self._write_btn.setToolTip("Write the focused object to disk (model coordinates or map).")
        self._write_btn.clicked.connect(self._on_write_object)
        self._pair_btn = QPushButton("Pair…")
        self._pair_btn.setToolTip(
            "Pair a model with a map so they can be used together. cctbx moves them into "
            "a common frame, which is what a map+model group already has from loading.")
        self._pair_btn.clicked.connect(self._on_pair)
        self._remove_model_btn = QPushButton("Remove")
        self._remove_model_btn.setToolTip("Remove the highlighted object")
        self._remove_model_btn.clicked.connect(self._on_remove_selected)
        reset_btn = QPushButton("Reset view")
        reset_btn.setToolTip("Reframe the camera to fit the whole scene.")
        reset_btn.clicked.connect(lambda: self._desktop.reset_view())
        picture_btn = QPushButton("Picture…")  # short: the grid's columns are equal
        picture_btn.setToolTip("Save a picture of the viewport as a PNG.")
        picture_btn.clicked.connect(self._on_save_picture)

        actions = QGridLayout()
        actions.setSpacing(6)
        # Row 1 gets data in and out; row 2 acts on what is loaded and on the view.
        rows = (
            (self._open_btn, demos_btn, self._write_btn, self._pair_btn),
            (self._remove_model_btn, reset_btn, picture_btn),
        )
        for r, row in enumerate(rows):
            for c, button in enumerate(row):
                # The shorter second row spans its last button so both rows fill the width.
                span = 2 if (r == 1 and c == len(row) - 1) else 1
                actions.addWidget(button, r, c, 1, span)
        for c in range(4):
            actions.setColumnStretch(c, 1)  # equal columns, so it reads as a grid
        ol.addLayout(actions)

        self._file_label = QLabel("")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: #888;")
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
        pick_row = QHBoxLayout()
        self._pick_btn = QPushButton("Pick atoms")
        self._pick_btn.setCheckable(True)
        self._pick_btn.setToolTip("Click atoms in the 3D view to build a selection.")
        self._pick_btn.toggled.connect(self._on_toggle_select)
        pick_row.addWidget(self._pick_btn, stretch=1)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_selection)
        pick_row.addWidget(self._clear_btn)
        sl.addLayout(pick_row)

        expr_row = QHBoxLayout()
        self._select_expr = QLineEdit()
        self._select_expr.setPlaceholderText("selection, e.g. chain A and resseq 5:14")
        self._select_expr.setToolTip("A cctbx / Phenix selection string on the active model.")
        self._select_expr.returnPressed.connect(self._on_select_expression)
        expr_row.addWidget(self._select_expr, stretch=1)
        self._select_expr_btn = QPushButton("Select")
        self._select_expr_btn.clicked.connect(self._on_select_expression)
        expr_row.addWidget(self._select_expr_btn)
        sl.addLayout(expr_row)

        from PySide6.QtWidgets import QSizePolicy

        sl.addWidget(QLabel("Quick select:"))
        chips = QGridLayout()
        chips.setSpacing(4)
        self._sel_chips = []  # (button, expr); checkable, highlighted when active
        self._chip_selecting = False
        specs = [("Protein", "protein"), ("Ligands", "hetero and not water"),
                 ("Water", "water"), ("Backbone", "protein and name CA")]
        for i, (label, expr) in enumerate(specs):
            chip = QPushButton(label)
            chip.setCheckable(True)  # native checked state (accent tint), no stylesheet
            chip.setToolTip(expr)
            chip.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            chip.clicked.connect(lambda _c=False, b=chip, e=expr: self._on_chip(b, e))
            chips.addWidget(chip, i // 2, i % 2)  # two columns
            self._sel_chips.append((chip, expr))
        sl.addLayout(chips)

        self._selection_label = QLabel("none selected")
        self._selection_label.setWordWrap(True)
        self._selection_label.setStyleSheet("color: #666;")
        sl.addWidget(self._selection_label)
        layout.addWidget(sel_box)

        # -- Mouse reference ---------------------------------------------
        # Zoom moved off the scroll wheel (which now contours) when the bindings went
        # Coot-style, and an invisible zoom is a real trap — so the whole map is spelled
        # out here on the home tab.
        layout.addWidget(self._build_mouse_legend())

        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(body)
        ol.addWidget(scroll, stretch=1)
        return outer

    def _build_demos_menu(self):
        """A short list of bundled examples, each showing off one thing the app does."""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self._window)
        menu.addAction("Ubiquitin (1UBQ)",
                       lambda: self._on_load_sample("1ubq.pdb"))
        menu.addAction("Ubiquitin — with density (map + model)",
                       self._on_run_map_model_demo)
        menu.addAction("Thermitase-eglin (1TEC) — validation demo",
                       lambda: self._on_load_sample("1tec.pdb"))
        menu.addAction("X-ray — model + reflections (make density)",
                       self._on_run_xray_demo)
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
        grid = QGridLayout()
        grid.setSpacing(6)
        specs = [("Distance", "distance", 2), ("Angle", "angle", 3), ("Dihedral", "dihedral", 4)]
        for i, (label, kind, n) in enumerate(specs):
            btn = QPushButton(label)
            btn.setToolTip(f"Measure a {kind} from {n} selected atoms.")
            btn.clicked.connect(lambda _c=False, k=kind: self._on_measure(k))
            grid.addWidget(btn, 0, i)
        mg.addLayout(grid)
        clear_m = QPushButton("Clear measurements")
        clear_m.clicked.connect(self._on_clear_measurements)
        mg.addWidget(clear_m)
        layout.addWidget(measure)

        marker = QGroupBox("Marker")
        mkg = QVBoxLayout(marker)
        mkg.addWidget(QLabel("Drop a marker at a 3D point in the viewport:"))
        mk_row = QHBoxLayout()
        place_btn = QPushButton("Place marker")
        place_btn.setToolTip(
            "Arm placement, then click in the viewport: a sphere is dropped there — "
            "snapped to the atom under the cursor, or the view plane in empty space — and "
            "its 3D coordinate is sent back.")
        place_btn.clicked.connect(self._desktop.arm_marker)
        mk_row.addWidget(place_btn)
        clear_mk = QPushButton("Clear")
        clear_mk.setToolTip("Remove all placed markers.")
        clear_mk.clicked.connect(self._desktop.clear_markers)
        mk_row.addWidget(clear_mk)
        mkg.addLayout(mk_row)
        layout.addWidget(marker)

        minimization = QGroupBox("Minimization")
        ming = QVBoxLayout(minimization)
        ming.addWidget(QLabel("Relax the model onto ideal geometry:"))
        self._minimize_map_check = QCheckBox("Use map")
        self._minimize_map_check.setToolTip(
            "Also pull the model into the density. Needs a map loaded together with "
            "the model as a group, so the two share a frame.")
        ming.addWidget(self._minimize_map_check)
        min_row = QHBoxLayout()
        self._minimize_btn = QPushButton("Minimize")
        self._minimize_btn.setToolTip(
            "Minimize the active model against its geometry restraints (no map), "
            "streaming each step into the viewport as it runs.")
        self._minimize_btn.clicked.connect(self._on_minimize)
        min_row.addWidget(self._minimize_btn)
        self._minimize_stop_btn = QPushButton("Stop")
        self._minimize_stop_btn.setToolTip("Halt the run, keeping the progress so far.")
        self._minimize_stop_btn.setEnabled(False)
        self._minimize_stop_btn.clicked.connect(lambda: self._desktop.stop_minimization())
        min_row.addWidget(self._minimize_stop_btn)
        min_row.addStretch()
        ming.addLayout(min_row)
        layout.addWidget(minimization)

        # -- Dragging atoms ------------------------------------------------
        # Armed deliberately: while it is on, a left-drag onto an atom pulls the model
        # about instead of turning the view.
        dragging = QGroupBox("Drag atoms")
        dg = QVBoxLayout(dragging)
        hint = QLabel("Shift-drag any atom or bond to pull it; the model bends to follow.")
        hint.setWordWrap(True)
        dg.addWidget(hint)
        self._tug_density_check = QCheckBox("Into the density")
        self._tug_density_check.setToolTip(
            "Let the map pull too, so a drag settles the neighbourhood into density "
            "rather than only bending it. Needs a map paired with the model.")
        self._tug_density_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_tug_into_density(on)))
        dg.addWidget(self._tug_density_check)
        self._tug_continuous_check = QCheckBox("Keep minimizing while dragging")
        self._tug_continuous_check.setToolTip(
            "Hold Shift and the model keeps relaxing the whole time — a gentle living "
            "settle that stays in motion even when the pointer is still, rather than "
            "nudging once per move and stopping.")
        self._tug_continuous_check.toggled.connect(lambda on: self._safe(
            lambda: self._desktop.set_tug_continuous(on)))
        dg.addWidget(self._tug_continuous_check)
        layout.addWidget(dragging)

        layout.addStretch()
        return tab

    def _on_minimize(self) -> None:
        try:
            self._desktop.minimize_model(use_map=self._minimize_map_check.isChecked())
        except Exception as exc:
            self._set_status(str(exc))

    def _on_minimizing_changed(self, running: bool) -> None:
        """Stop is only meaningful while a run is going; Minimize only while one is not."""
        self._minimize_btn.setEnabled(not running)
        self._minimize_stop_btn.setEnabled(running)

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

    def _build_clashes_group(self):
        """All-atom contacts: add hydrogens with reduce2, then run probe2. Its two
        overlays toggle independently once an analysis has produced dots."""
        from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

        from .live import PROBE_CLASHES, PROBE_CONTACTS

        box = QGroupBox("Clashes && contacts (probe2)")
        ag = QVBoxLayout(box)
        ag.addWidget(QLabel("Add hydrogens, then run MolProbity probe2:"))
        analyze = QPushButton("Add H + analyze")
        analyze.setToolTip(
            "Add hydrogens with reduce2 as a new object (hiding the original), then run "
            "probe2 for MolProbity contacts and clashes.")
        analyze.clicked.connect(self._on_analyze)
        ag.addWidget(analyze)

        toggles = QHBoxLayout()
        self._contacts_toggle = QPushButton("Contacts")
        self._contacts_toggle.setToolTip("Show/hide the full probe2 contact-dot surface.")
        self._contacts_toggle.setCheckable(True)
        self._contacts_toggle.setEnabled(False)
        self._contacts_toggle.toggled.connect(
            lambda on: self._desktop.set_probe_channel(PROBE_CONTACTS, on))
        toggles.addWidget(self._contacts_toggle)
        self._clashes_toggle = QPushButton("Clashes")
        self._clashes_toggle.setToolTip("Show/hide the bad-overlap (clash) spikes.")
        self._clashes_toggle.setCheckable(True)
        self._clashes_toggle.setEnabled(False)
        self._clashes_toggle.toggled.connect(
            lambda on: self._desktop.set_probe_channel(PROBE_CLASHES, on))
        toggles.addWidget(self._clashes_toggle)
        toggles.addStretch()
        ag.addLayout(toggles)
        return box

    def _build_validation_tab(self):
        """MolProbity validation: the hydrogen-based all-atom contact analysis, plus
        the per-residue validators. The latter is data-driven from the validation
        registry — one "Run validation" button runs every registered validator on the
        active model and each result becomes its own sub-tab, so new validators appear
        here automatically with no changes to this tab."""
        from PySide6.QtWidgets import (
            QLabel,
            QPushButton,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        layout.addWidget(self._build_clashes_group())

        run_btn = QPushButton("Run validation")
        run_btn.setToolTip("Run every MolProbity validator on the active model (background thread).")
        run_btn.clicked.connect(self._on_run_validation)
        layout.addWidget(run_btn)

        # One sub-tab per validator, (re)built as runs complete.
        self._validation_subtabs = QTabWidget()
        self._validation_subtabs.setDocumentMode(True)
        placeholder = QWidget()
        pl = QVBoxLayout(placeholder)
        hint = QLabel("Run validation to see results.")
        hint.setStyleSheet("color: #666;")
        pl.addWidget(hint)
        pl.addStretch()
        self._validation_subtabs.addTab(placeholder, "—")
        layout.addWidget(self._validation_subtabs, stretch=1)
        return tab

    def _build_validation_section(self, mid, result):
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
        summary.setStyleSheet("color: #666;")
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
        markers.setChecked(bool(result.markup))
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

    def _on_validation_ready(self, payload) -> None:
        """Validation finished (GUI thread): rebuild one sub-tab per result."""
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        mid, results = payload
        tabs = self._validation_subtabs
        current = tabs.tabText(tabs.currentIndex())  # preserve the selected validator
        tabs.clear()
        if not results:
            empty = QWidget()
            el = QVBoxLayout(empty)
            el.addWidget(QLabel("No validators registered."))
            el.addStretch()
            tabs.addTab(empty, "—")
            return
        for result in results:
            tabs.addTab(self._build_validation_section(mid, result), result.title)
        for i in range(tabs.count()):  # keep the user on the same validator across re-runs
            if tabs.tabText(i) == current:
                tabs.setCurrentIndex(i)
                break

    def _build_settings_tab(self):
        """Second-class settings that don't belong in the everyday workflow."""
        from PySide6.QtWidgets import (
            QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        viewer = QGroupBox("Viewer")
        vg = QVBoxLayout(viewer)
        self._axis_check = QCheckBox("Show XYZ axes")
        self._axis_check.setChecked(False)  # the viewer hides them by default
        self._axis_check.toggled.connect(lambda on: self._desktop.set_axis(on))
        vg.addWidget(self._axis_check)

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
            "How much density around the view centre a map made from reflections opens "
            "with. Each map's own radius is on its Appearance pane.")
        radius_spin.valueChanged.connect(self._desktop.set_view_radius_default)
        radius_row.addWidget(radius_spin)
        radius_row.addStretch()
        vg.addLayout(radius_row)
        layout.addWidget(viewer)

        layout.addStretch()
        return tab

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

    def _update_appearance(self, kind=None, ident=None):
        """Rebuild the Appearance box for the focused object (or an empty-state hint)."""
        from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton

        self._clear_layout(self._appearance_layout)
        it = self._find_item(kind, ident) if ident else None
        self._focused = (kind, ident) if it else (None, None)
        self._iso_row = None  # rebuilt below only when a volume is focused
        if it is None:
            hint = QLabel("Select an object above to edit how it looks.")
            hint.setStyleSheet("color: #999;")
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
            # Not styled — a marker is a 3D point. The pane is where it says what it is:
            # its coordinate (the handle to act on) and what it snapped to.
            pos = it.get("position") or [0.0, 0.0, 0.0]
            coord = QLabel(f"x  {pos[0]:.3f}\ny  {pos[1]:.3f}\nz  {pos[2]:.3f}    Å")
            coord.setStyleSheet("font-family: monospace;")
            self._appearance_layout.addWidget(coord)
            snapped = ("on atom " + str(it["atom"])) if it.get("atom") is not None \
                else "in the view plane (empty space)"
            note = QLabel("Placed " + snapped)
            note.setWordWrap(True)
            note.setStyleSheet("color: #888;")
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
            arrays.setStyleSheet("color: #888;")
            self._appearance_layout.addWidget(arrays)
            note = QLabel(
                "Carries map coefficients — density needs no model."
                if it.get("has_map_coefficients")
                else "Amplitudes only — density needs a model to phase against.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #888;")
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
            add_combo("Colour", _MODEL_COLOR_OPTIONS, it.get("color"), _set_color)
            types = it.get("types") or []
            if len(types) > 1:
                r = QHBoxLayout()
                lab = QLabel("Show")
                lab.setMinimumWidth(80)
                r.addWidget(lab)
                r.addWidget(self._make_type_combo(mid, types, set(it.get("hidden_types") or [])), stretch=1)
                self._appearance_layout.addLayout(r)
            inter = QCheckBox("Computed interactions")
            inter.setToolTip("Overlay Mol*-computed non-covalent contacts (H-bonds, salt bridges, …).")
            inter.setChecked(bool(it.get("interactions")))
            inter.toggled.connect(lambda on, d=mid: self._desktop.set_model_interactions(d, on))
            self._appearance_layout.addWidget(inter)

            def _set_clip(front, back, it=it):
                it["clip"] = (front, back)
                self._safe(lambda: self._desktop.set_model_clip(mid, front, back))

            self._add_clip_row(
                {**it, **self._desktop.model_appearance(mid)}.get("clip"), _set_clip)
        else:  # volume
            vid = it["id"]
            # Read the live values, not this snapshot: the level in particular can have
            # moved since (the scroll wheel, or the console) without a new summary.
            live = {**it, **self._desktop.volume_appearance(vid)}

            def _set_style(v, it=it):
                it["style"] = v
                self._safe(lambda: self._desktop.set_volume_style(vid, v))

            def _set_color(v, it=it):
                it["color"] = v
                self._safe(lambda: self._desktop.set_volume_color(vid, v))

            add_combo("Style", _VOLUME_STYLE_OPTIONS, live.get("style"), _set_style)
            self._add_color_row(live.get("color"), _set_color)

            def _set_opacity(v, it=it):
                it["opacity"] = v
                self._safe(lambda: self._desktop.set_volume_opacity(vid, v))

            self._add_opacity_row(live.get("opacity"), _set_opacity)

            def _set_iso(v, it=it):
                it["iso"] = v
                self._safe(lambda: self._desktop.set_volume_iso(vid, v))

            self._iso_row = self._add_iso_row(live.get("iso"), _set_iso)

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
        """How much density to draw around the view centre.

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

    def _add_color_row(self, current, on_pick):
        """A volume's colour: swatches, and a picker for anything else.

        Colours are shown rather than named — a swatch says what "orchid" is and a word
        does not. The picker is the escape hatch, since the wire takes any hex Mol* can
        decode, not just the presets.
        """
        from PySide6.QtCore import QSize, Qt
        from PySide6.QtGui import QColor, QIcon, QPixmap
        from PySide6.QtWidgets import QColorDialog, QComboBox, QHBoxLayout, QLabel

        def swatch(name):
            pixmap = QPixmap(28, 14)
            pixmap.fill(QColor(name))
            return QIcon(pixmap)

        row = QHBoxLayout()
        lab = QLabel("Colour")
        lab.setMinimumWidth(80)
        row.addWidget(lab)
        combo = QComboBox()
        combo.setIconSize(QSize(28, 14))
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
        for name in _VOLUME_COLORS:
            combo.addItem(swatch(name), name.capitalize(), name)
        custom = None
        if current and current not in _VOLUME_COLORS:
            custom = current  # a picked colour: keep it on the list so it stays selected
            combo.addItem(swatch(current), current, current)
        combo.addItem("Custom…", _CUSTOM_COLOR)
        idx = combo.findData(current)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

        # The last colour actually committed — the target to revert to if a custom pick
        # is cancelled, since by then the live preview has already changed the map.
        committed = {"value": current}

        def picked(_index, combo=combo):
            value = combo.currentData()
            if value != _CUSTOM_COLOR:
                committed["value"] = value
                on_pick(value)
                return
            revert_to = committed["value"]
            dialog = QColorDialog(QColor(revert_to or _VOLUME_COLORS[0]), self._window)
            dialog.setWindowTitle("Volume colour")
            # Qt's own dialog, not the native one: the macOS colour panel is a shared
            # singleton that emits its live-colour signal unreliably, and this preview
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
            ("Shift + drag", "Pull an atom"),
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

    def _add_iso_row(self, current, on_change):
        """Contour level: a slider to hunt with, a spinbox for the exact value.

        Both are wanted. The slider is how you actually find a level — you watch the map,
        not the number — and updates are live, so dragging is the point. The spinbox
        makes a level reproducible ("contour at 1.5 sigma") and reaches past the slider's
        range for maps that need it.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QSlider

        from .volume_io import DEFAULT_ISO_SIGMA

        value = DEFAULT_ISO_SIGMA if current is None else float(current)
        row = QHBoxLayout()
        lab = QLabel("Level")
        lab.setMinimumWidth(44)  # room for the gesture chip beside it, same total width
        row.addWidget(lab)
        # The bare scroll wheel adjusts this — say so right where the control is, since
        # that is the one gesture people miss (it is also why scroll no longer zooms).
        row.addWidget(self._gesture_chip("scroll"))

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, int(round(_ISO_SLIDER_MAX / _ISO_RESOLUTION)))
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
        from PySide6.QtWidgets import QCheckBox, QTabWidget, QVBoxLayout, QWidget

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
        layout.addWidget(self._filter_selection_check)

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

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

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
            QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox,
        )

        models, volumes = self._desktop.pairable()
        if not models or not volumes:
            QMessageBox.information(
                self._window, "Nothing to pair",
                "Pairing needs a model and a map that are not already paired with "
                "something.\n\nLoading a model and a map together pairs them for you.")
            return

        dialog = QDialog(self._window)
        dialog.setWindowTitle("Pair model with map")
        form = QFormLayout(dialog)
        note = QLabel(
            "cctbx will move these into a common frame, so the model may shift.\n"
            "That is what makes them usable together — minimizing into density, say.")
        note.setStyleSheet("color: #888;")
        form.addRow(note)
        model_combo = QComboBox()
        for m in models:
            model_combo.addItem(m["name"], m["id"])
        volume_combo = QComboBox()
        for v in volumes:
            volume_combo.addItem(v["name"], v["id"])
        form.addRow("Model:", model_combo)
        form.addRow("Map:", volume_combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self._desktop.pair_model_with_map(
                model_combo.currentData(), volume_combo.currentData())
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not pair", str(exc))

    def _update_pair_button(self) -> None:
        """Pairing needs something unpaired on both sides."""
        models, volumes = self._desktop.pairable()
        self._pair_btn.setEnabled(bool(models) and bool(volumes))

    def _on_write_object(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        kind, ident = self._focused
        if ident is None:
            self._set_status("select an object to write")
            return
        it = self._find_item(kind, ident)
        default = it["name"] if it else "out"
        if kind == "model":
            fmt = "mmCIF (*.cif);;PDB (*.pdb)"
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

    def _on_chip(self, button, expr: str) -> None:
        for other, _ in self._sel_chips:
            if other is not button:
                other.setChecked(False)
        self._chip_selecting = True  # so the resulting change doesn't clear this chip
        try:
            if button.isChecked():
                self._run_selection(expr)
            else:  # clicking the active chip again clears the selection
                self._desktop.clear_selection()
                self._selection_label.setText("none selected")
        finally:
            self._chip_selecting = False

    def _on_select_expression(self) -> None:
        self._run_selection(self._select_expr.text())

    def _run_selection(self, expr: str) -> None:
        self._select_expr.setText(expr)
        try:
            n = self._desktop.select_by_expression(expr)
        except Exception as exc:  # invalid syntax / no model
            self._selection_label.setText(f"<span style='color:#c0392b'>{exc}</span>")
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

    def _on_stop_demo(self) -> None:
        self._desktop.stop_demo()

    def _on_toggle_select(self, checked: bool) -> None:
        if checked:
            self._desktop.enable_mouse_selection()
        else:
            self._desktop.disable_mouse_selection()

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

    def _on_help(self) -> None:
        # Placeholder until the documentation is linked.
        self._set_status("Documentation coming soon.")

    def reflect_dock_state(self, floating: bool) -> None:
        """Keep the dock/detach button's label in step with the panel's state."""
        self._dock_btn.setText("Dock" if floating else "Detach")

    def _on_analysis_ready(self, mid) -> None:
        """Analysis finished: enable and check both overlay toggles (both drawn)."""
        for toggle in (self._contacts_toggle, self._clashes_toggle):
            toggle.setEnabled(True)
            toggle.blockSignals(True)
            toggle.setChecked(True)
            toggle.blockSignals(False)

    def _on_scene_selection_changed(self, scene) -> None:
        """A model's picks changed. Refresh the aggregate label + the atoms table."""
        # A selection from anywhere but a chip click no longer matches a preset.
        if not self._chip_selecting:
            for chip, _ in getattr(self, "_sel_chips", []):
                chip.setChecked(False)
        self._scene_selection = scene or {}
        total = sum(len(v) for v in self._scene_selection.values())
        n_models = len(self._scene_selection)
        if total:
            across = f" across {n_models} models" if n_models > 1 else ""
            self._selection_label.setText(f"{total} atom(s) selected{across}")
        else:
            self._selection_label.setText("none selected")
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

    def _apply_restraint_filter(self) -> None:
        """Filter each built restraint table to restraints within the selection (or all)."""
        if self._restraints_model_id is None:
            return  # restraints not built yet — _ensure_restraints will apply on build
        geo = self._geo_cache.get(self._restraints_model_id)
        if geo is None:
            return
        on = self._filter_selection_check.isChecked()
        selected = set(self._table_selection_indices()) if on else None
        self._suppress_restraint_sync = True
        try:
            for cat, info in self._restraint_tabs.items():
                info["model"].set_filter(geo.indices_within(cat, selected) if on else None)
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
                    node.setExpanded(True)
                    group_nodes[gid] = node
            for it in items:
                parent = group_nodes.get(it["group"], self._loaded_tree)
                # [visible check] col 0, [active radio] col 1, [name] col 2 (elides).
                node = QTreeWidgetItem(parent)
                node.setData(0, Qt.ItemDataRole.UserRole, (it["kind"], it["id"]))
                if it["visible"] is None:
                    # Reflections: nothing drawable, so nothing to show or hide.
                    node.setFlags(node.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
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
                suffix = {"volume": "   [map]", "reflections": "   [data]", "marker": "   [marker]"}
                name = it["name"] + suffix.get(it["kind"], "")
                node.setText(2, name)
                node.setToolTip(2, it["name"])  # full name on hover when elided
                if it.get("active"):
                    active_item = node
            if active_item is not None:
                self._loaded_tree.setCurrentItem(active_item)
        finally:
            self._suppress_model_events = False
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

    def _on_tree_current_changed(self, current, _previous) -> None:
        if self._suppress_model_events or current is None:
            return
        from PySide6.QtCore import Qt

        kind, ident = current.data(0, Qt.ItemDataRole.UserRole)
        if kind == "group":
            self._update_appearance()  # a group header has nothing to edit
            return
        self._update_appearance(kind, ident)  # master -> detail
        if kind == "model":
            self._desktop.set_active_model(ident)  # focusing a model activates it

    def _make_type_combo(self, mid, types, hidden):
        """A checkable dropdown of structure types (checked = shown)."""
        combo = _make_checkable_combo()
        combo.setToolTip("Show or hide structure types (protein, water, …) in this model.")
        for label in types:
            combo.add_checkable(label, label not in hidden, label)  # before on_change
        combo.on_change = lambda label, shown, d=mid: self._on_type_toggle(d, label, shown)
        return combo

    def _on_type_toggle(self, mid: str, label: str, shown: bool) -> None:
        if self._suppress_model_events:
            return
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
        from PySide6.QtCore import Qt

        if self._suppress_model_events:
            return
        kind, ident = item.data(0, Qt.ItemDataRole.UserRole)
        visible = item.checkState(0) == Qt.CheckState.Checked
        if kind == "model":
            self._desktop.set_model_visible(ident, visible)
        elif kind == "volume":
            self._desktop.set_volume_visible(ident, visible)
        elif kind == "marker":
            self._desktop.set_marker_visible(ident, visible)
        # reflections have no visibility to change

    def _on_active_radio(self, button) -> None:
        """A model's active radio was clicked -> make it the active model."""
        if self._suppress_model_events:
            return
        mid = button.property("mid")
        if mid:
            # set_active_model refreshes the Loaded tree; _on_loaded_changed then points
            # Appearance at the newly active model (a focused model tracks the active one).
            self._desktop.set_active_model(mid)

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

    def __init__(self, host: str = "127.0.0.1", port: int = 5173):
        _check_qt()

        from PySide6.QtWidgets import QApplication

        self._host = host
        self._port = port

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
        self._scene_counter = 0  # cache-buster for the composed volume MVSJ
        self._dummy: Optional[Any] = None  # persistent control ws when no model is visible
        self._batching = False  # defer viewport reload / signals during a group load
        # Scene-level selection: {model_id: [atom indices]}. Each model reports its
        # own picks independently (a selection may span models — e.g. protein +
        # ligand); the union across models is the scene selection. Mutated on the
        # WebSocket threads, read on the GUI thread, so guard it.
        self._scene_selection: dict = {}
        self._scene_lock = threading.Lock()
        # Restraint-notation primitives currently drawn for the selected geometry rows.
        self._restraint_prim_ids: list = []
        self._restraint_prim_session = None
        self._markers: list = []  # placed markers: {id, name, position, atom, visible}
        self._marker_counter = 0
        self._player: Optional[Player] = None
        self._demo_thread: Optional[threading.Thread] = None
        self._selection_enabled = False
        self._computed_interactions_visible = False
        self._load_counter = 0

        self._stopped = False
        self._prev_sigint = None
        self._sigint_installed = False
        self._sigint_timer = None
        self._minimize_stop = threading.Event()  # set to halt a running minimization
        self._volume_scroll_target: Optional[str] = None  # volume the wheel contours
        # Dragging atoms: Shift-drag, one drag at a time (there is one pointer).
        self._tug_into_density = False
        self._tug_continuous = False
        self._tug: Any = None
        self._tug_model: Optional[str] = None
        self._tug_session: Any = None
        self._tug_last: Any = None
        self._tug_last_push: float = 0.0
        self._tug_queue: Any = None  # made with its worker on the first drag
        # The radius new maps from reflections open with (Settings changes it).
        self.view_radius_default: float = _VIEW_RADIUS_DEFAULT

        self.bridge = _make_bridge()
        # Workers marshal GUI-thread work (e.g. adding a model) via this signal;
        # emitted from another thread it dispatches as a queued call on the GUI thread.
        self.bridge.run_on_main.connect(lambda fn: fn())
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
        self._dock_close_filter = _make_dock_close_filter(
            self._controls_dock, lambda: self._main.isVisible())
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

        # Maximize is honoured on Wayland (unlike window positioning), so this fills the
        # screen; the dock is sized to ~1/3 once the maximized width is known.
        self._main.showMaximized()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._size_controls_dock)

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

    def _size_controls_dock(self) -> None:
        """Give the controls dock ~1/3 of the width, the viewport the rest.

        Run after the window is up (a queued call) so the maximized width is known; the
        separator stays draggable, so this is only the starting proportion.
        """
        from PySide6.QtCore import Qt

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

    def group_mmm(self, gid: Optional[str]) -> Any:
        """The ``map_model_manager`` a group came from, or None if it did not come from one."""
        group = self._groups.get(gid) if gid else None
        return group["mmm"] if group else None

    def pairable(self) -> tuple:
        """``(models, volumes)`` that are not paired with anything yet.

        Only ungrouped objects: an object already in a group has a manager speaking for
        it, and re-pairing it would move it out from under that.
        """
        models = [m for m in self._models
                  if m.get("group") is None and getattr(m["session"], "model", None) is not None]
        volumes = [v for v in self._volumes if v.get("group") is None]
        return models, volumes

    def pair_model_with_map(self, mid: str, vid: str) -> str:
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

        mmm = map_model_manager(
            model=model, map_manager=ventry["data"].map_manager,
            ignore_symmetry_conflicts=True)

        gid = self._new_group(f"{mentry['name']} + {ventry['name']}", mmm=mmm)
        mentry["group"] = gid
        ventry["group"] = gid
        # cctbx moves the model (and possibly the map) into the shared frame, so show
        # where they now are rather than where they were loaded.
        mentry["session"].push(model.get_sites_cart().as_numpy_array())
        self._write_display_map(vid, ventry["data"])
        self._reload_viewport()
        self._emit_loaded_changed()
        self._status(f"Paired {mentry['name']} with {ventry['name']}")
        return gid

    # -- viewport composition --

    def _visible_model_ws(self) -> List[str]:
        return [f"ws://{self._host}:{m['session'].port}" for m in self._models if m["visible"]]

    def _ensure_dummy_ws(self) -> str:
        """A persistent 1-atom control session: carries volume commands and keeps the
        page non-blank when no model is visible. Nothing to pick, so no selection."""
        if self._dummy is None:
            self._dummy = _dummy_session()
            self._dummy.start(host=self._host, port=0)
            self._dummy.on_volume_iso(self._on_volume_iso_changed)
            # Render nothing: an empty `on` set draws no atoms, so an empty scene
            # is truly empty (the dummy only keeps the ws channel open).
            try:
                self._dummy.set_representation("ball-and-stick", on=[])
            except Exception:  # pragma: no cover - defensive
                pass
        return f"ws://{self._host}:{self._dummy.port}"

    def _control_session(self):
        """A session the viewport is actually connected to, for volume commands.

        It has to be one of the sockets ``_reload_viewport`` put in the page: the visible
        models', or the dummy when no model is visible. The *active* model is the wrong
        answer when it is hidden — commands would be broadcast to a session with no
        clients and vanish, so every volume control would quietly stop working.
        """
        entry = self._model_entry(self._active_model_id)
        if entry is not None and entry["visible"]:
            return entry["session"]
        visible = next((m["session"] for m in self._models if m["visible"]), None)
        if visible is not None:
            return visible
        return self._dummy

    def _write_volume_scene(self) -> Optional[str]:
        """Write an MVSJ composing every visible volume; return its URL path (or None)."""
        visible = [v for v in self._volumes if v["visible"]]
        if not visible:
            return None
        from .volume import Volume, create_volume_view

        focus_first = not self._visible_model_ws()  # centre a lone volume; don't fight a model
        nodes = []
        for i, v in enumerate(visible):
            nodes.append(Volume(
                url=v["map_url"], ref=v["ref"], format="map",
                isosurface_kind="relative", isosurface_value=v["iso"],
                color=v["color"], negative_color=v.get("negative_color"),
                opacity=v["opacity"], style=v["style"],
                focus=(focus_first and i == 0),
            ))
        self._scene_counter += 1
        scene_dir = self._webapp.volume_dir / "scene" / str(self._scene_counter)
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "scene.mvsj").write_text(create_volume_view(volumes=nodes))
        return f"/scene/{self._scene_counter}/scene.mvsj"

    def _reload_viewport(self) -> None:
        """Compose the visible models (ws) and volumes (MVSJ) into one viewport URL."""
        if self._batching:
            return
        model_ws = self._visible_model_ws()
        mvsj = self._write_volume_scene()
        ws = list(model_ws)
        if not model_ws:
            # No model to carry volume commands / keep the page alive -> use the dummy.
            ws.append(self._ensure_dummy_ws())
        self._reassert_volume_clips()
        params = []
        if mvsj:
            params.append(f"mvsj={mvsj}")
        params.append("ws=" + ",".join(ws))
        self._viewport.load(f"{self._webapp.url}index.html?{'&'.join(params)}")

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
        """Atom indices to show given the model's hidden types, or None for all."""
        hidden = entry.get("hidden_types") or set()
        if not hidden:
            return None
        groups = self._type_groups(entry)
        drop = set()
        for label in hidden:
            drop.update(groups.get(label, []))
        if not drop:
            return None
        mask = np.ones(entry["session"]._n_atoms, dtype=bool)
        mask[list(drop)] = False
        return np.nonzero(mask)[0].tolist()

    def _apply_model_rep(self, entry) -> None:
        session, rep = entry["session"], entry["rep"]
        color = entry.get("color") or _model_rep_color(rep)  # explicit colour overrides the default
        on = self._shown_indices(entry)  # restrict to shown structure types
        if on is not None:
            session.set_representation(rep, color=color, on=on)
        else:
            session.set_representation(rep, color=color)

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
        rep = rep or self._default_model_rep(session)
        entry = {"id": mid, "name": name, "session": session, "visible": True, "group": group,
                 "rep": rep, "color": None, "hidden_types": set(), "type_groups": None,
                 "clip": (0.0, 1.0),
                 "interactions": False}
        self._models.append(entry)
        self._apply_model_rep(entry)
        self._active_model_id = mid
        # Register this model's pick handler once (tagged with its id); the click
        # mode is what actually turns picking on/off. Registering here means a
        # selection can be built in any loaded model, not just the active one.
        session.on_selection(lambda sel, mid=mid: self._on_model_selection(mid, sel))
        # Volume commands ride whichever session is the control session, so contour
        # changes made in the viewport can come back on any of them.
        session.on_volume_iso(self._on_volume_iso_changed)
        session.on_tug(lambda action, atom, target, mid=mid: self._on_tug(mid, action, atom, target))
        # Markers are a scene-level thing, not this model's — but only the armed (control)
        # session's viewport reports one, so wiring every session is harmless.
        session.on_marker(lambda position, atom: self._on_marker(position, atom))
        if self._selection_enabled:
            session.enable_mouse_selection()  # handler already registered; just arm click mode
        self._wire_active(session)
        self._reload_viewport()
        self._emit_loaded_changed()
        return mid

    def set_model_representation(self, mid: str, rep: str) -> None:
        """Change a model's representation type (from the inline dropdown)."""
        entry = self._model_entry(mid)
        if entry is None or entry.get("rep") == rep:
            return
        entry["rep"] = rep
        self._apply_model_rep(entry)

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

    def model_structure_types(self, mid: str) -> list:
        """The structure types present in a model (for the show/hide menu)."""
        entry = self._model_entry(mid)
        return list(self._type_groups(entry).keys()) if entry else []

    def set_model_color(self, mid: str, color: Optional[str]) -> None:
        """Set a model's colour theme (None = the representation's default)."""
        entry = self._model_entry(mid)
        if entry is None or entry.get("color") == color:
            return
        entry["color"] = color
        self._apply_model_rep(entry)

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

        def work():
            from .hydrogens import add_hydrogens, hydrogens_available
            from .live import LiveSession, PROBE_CLASHES, PROBE_CONTACTS
            from .probe import probe_dots_split

            if not hydrogens_available():
                self._status("reduce2 needs the monomer library (set MMTBX_CCP4_MONOMER_LIB)")
                return
            try:
                self._status(f"adding hydrogens to {name} (reduce2)…")
                hmodel = add_hydrogens(model)
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
                contacts, clashes = probe_dots_split(hmodel)
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

        threading.Thread(target=work, name="pxviewer-reduce2", daemon=True).start()
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

        def work():
            from . import validation

            try:
                self._status(f"validating {name}…")
                results = validation.run_all(model)
            except Exception as exc:  # pragma: no cover - validator/runtime errors
                self._status(f"validation failed: {exc}")
                return
            ventry = self._model_entry(mid)
            if ventry is not None:  # cache so marker toggles redraw without re-running
                ventry["validation"] = {r.key: r for r in results}
            total = sum(len(r.markup) for r in results)
            self._status(f"{name}: {len(results)} validators, {total} markers")
            self.bridge.validation_ready.emit((mid, results))

        threading.Thread(target=work, name="pxviewer-validation", daemon=True).start()
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
        moves under it and the neighbourhood keeps settling even when the pointer is
        still, which is what lets it flow into density rather than only bend. Next drag.
        """
        self._tug_continuous = bool(enabled)

    # -- markers ---------------------------------------------------------

    def arm_marker(self) -> None:
        """Arm 'place a marker': the next click in the viewport drops a sphere and reports
        its 3D coordinate (see :meth:`_on_marker`). One-shot — the viewer disarms after
        the click. Rides the control session, so the marker is a scene-level point rather
        than tied to one model."""
        session = self._control_session()
        if session is None:
            self._status("load a model first to place a marker")
            return
        session.set_marker_mode(True)
        self._status("Click in the viewport to place a marker…")

    def _on_marker(self, position, atom) -> None:
        """A marker was placed: register it (so it appears in the Objects list as a handle)
        and draw it. ``atom`` is the picked atom index if the click landed on one, else
        None. Runs on a session's event-loop thread; the list/markup are cheap and safe."""
        self._marker_counter += 1
        point = [float(c) for c in position]
        self._markers.append({
            "id": f"marker-{self._marker_counter}", "name": f"Marker {self._marker_counter}",
            "position": point, "atom": atom, "visible": True,
        })
        self._draw_markers()
        self._emit_loaded_changed()
        where = f" on atom {atom}" if atom is not None else ""
        self._status(
            f"Marker at ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}){where} "
            f"— {len(self._markers)} placed")

    def _marker_entry(self, mid):
        return next((m for m in self._markers if m["id"] == mid), None)

    def _draw_markers(self) -> None:
        """(Re)draw every visible marker as a sphere on the marker channel."""
        session = self._control_session()
        if session is None:
            return
        balls = [[m["position"], _MARKER_RADIUS] for m in self._markers if m["visible"]]
        primitive = {"kind": "balls", "color": _MARKER_COLOR, "balls": balls}
        session.show_markup(_MARKER_CHANNEL, [primitive] if balls else [])

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
            self._status("Markers cleared")

    def _on_tug(self, mid: str, action: str, atom: int, target) -> None:
        """A drag in the viewport: queued, never served here.

        This runs on the session's event loop — the one thread that both reads the
        viewer's messages and writes coordinates back to it. Building a drag's zone
        takes the better part of a second, and stalling that loop for it means the
        viewer goes deaf and blind exactly as the drag begins.
        """
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
            from .geometry import monomer_library_available

            if not monomer_library_available():
                self._status("dragging atoms needs the monomer library")
                return
            from .tug import Tug

            try:
                self._tug = Tug(model, atom, map_data=self.map_for_model(mid)
                                if self._tug_into_density else None)
            except Exception as exc:  # pragma: no cover - restraints/runtime errors
                self._status(f"could not start dragging: {exc}")
                self._tug = None
                return
            self._tug_model = mid
            self._tug_session = session
            self._tug_last = None
            self._tug_last_push = 0.0
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
            self._end_tug()
            entry.pop("validation", None)  # stale: the atoms just moved

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
        the atom stays where you left it while the neighbourhood relaxes around it; the
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
        """Stream a drag frame — paced, and never a repeat of the last.

        Each frame is a whole set of coordinates and the viewer re-derives geometry from
        it; pushing them as fast as cctbx computes floods the render, and the frames back
        up into visible lag. Capping the rate keeps the picture current instead. A frame
        identical to the last is dropped outright — a settled geometry drag would
        otherwise send the same conformation over and over. ``force`` bypasses the pacing
        (not the de-dup) so a final resting frame is never dropped for arriving too soon.
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
        self._tug_session.push(coords)

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

        def work():
            from .geometry import monomer_library_available
            from .minimize import minimize

            if not monomer_library_available():
                self._status("minimization needs the monomer library (set MMTBX_CCP4_MONOMER_LIB)")
                self.bridge.minimizing_changed.emit(False)
                return
            try:
                self._status(f"minimizing {name}{' into the map' if map_data else ''}…")
                # Thin the stream: cctbx emits a state per function evaluation, far
                # more than the viewport can show.
                stats = minimize(
                    model, map_data=map_data, on_state=session.push,
                    should_stop=self._minimize_stop.is_set, stride=4)
            except Exception as exc:  # pragma: no cover - restraints/runtime errors
                self._status(f"minimization failed: {exc}")
                return
            finally:
                self.bridge.minimizing_changed.emit(False)
            self._status(
                f"{name}: bond rmsd {stats['bonds_before']:.3f} -> {stats['bonds_after']:.3f}, "
                f"angle rmsd {stats['angles_before']:.2f} -> {stats['angles_after']:.2f} "
                f"({stats['n_sent']} of {stats['n_states']} steps shown)"
                + (f", map weight {stats['weight']:.1f}" if stats["weight"] else "")
                + (" — stopped" if stats["stopped"] else ""))
            entry.pop("validation", None)  # stale: the coordinates just moved
            # So is the density, if this model was phased: it describes where the atoms
            # were. Once per run, never per step — each update is two transforms.
            reflections = self.reflections_for_model(entry["id"])
            if reflections is not None:
                self.bridge.run_on_main.emit(
                    lambda rid=reflections["id"]: self._update_maps_if_live(rid))

        self.bridge.minimizing_changed.emit(True)
        threading.Thread(target=work, name="pxviewer-minimize", daemon=True).start()

    def stop_minimization(self) -> None:
        """Halt a running minimization at its next step.

        The model keeps the progress made so far — a stopped run is a shorter run, not
        a discarded one.
        """
        self._minimize_stop.set()
        self._status("stopping minimization…")
        self._status("minimizing…")

    def set_axis(self, visible: bool) -> None:
        control = self._control_session()
        if control is not None:
            control.set_axis(bool(visible))

    def reset_view(self) -> None:
        """Reframe the viewport camera to fit the whole scene."""
        self._focused_residue = None  # space-bar nav restarts from the top after a reset
        control = self._control_session()
        if control is not None:
            control.reset_view()

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
            text = model.model_as_pdb() if is_pdb else model.model_as_mmcif()
            with open(p, "w") as fh:
                fh.write(text)
        elif kind == "volume":
            entry = self._volume_entry(ident)
            if entry is None:
                raise ValueError("no such volume")
            entry["data"].write_map(p)  # cctbx writes the map
        else:
            raise ValueError("nothing to write")

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
        """Change a volume's isosurface style (surface/wireframe/mesh) live."""
        self._volume_command(vid, "style", style,
                             lambda c, ref, v: c.set_volume_style(ref, v))

    def set_volume_iso(self, vid: str, value: float) -> None:
        """Set a volume's contour level, in sigma, live."""
        self._volume_command(vid, "iso", float(value),
                             lambda c, ref, v: c.set_volume_iso(ref, v))

    def set_volume_opacity(self, vid: str, value: float) -> None:
        """Set a volume's opacity (0-1) live."""
        self._volume_command(vid, "opacity", float(value),
                             lambda c, ref, v: c.set_volume_opacity(ref, v))

    def set_volume_color(self, vid: str, color: str) -> None:
        """Set a volume's colour live."""
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

        threading.Thread(target=work, name="pxviewer-screenshot", daemon=True).start()
        self._status("taking a picture…")

    def volume_appearance(self, vid: str) -> dict:
        """A volume's current style/colour/opacity/level.

        The Loaded summary is a snapshot taken when it was emitted, and these can change
        without one — from the console, or by the wheel in the viewport — so the
        Appearance pane reads them from the entry rather than trusting the snapshot.
        """
        entry = self._volume_entry(vid)
        if entry is None:
            return {}
        return {key: entry.get(key)
                for key in ("style", "color", "opacity", "iso", "clip", "mask_radius",
                            "radius")}

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

    def set_volume_radius(self, vid: str, radius: Optional[float]) -> None:
        """Draw only density within ``radius`` A of the view centre (None = all of it).

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
        colour or a level it cannot be baked into the scene — the session has to replay
        it when the fresh page connects. Both ends of that move underneath it: the
        session carrying volume commands changes (dummy <-> active model), and the page
        is new. So the clips are re-asserted on every reload rather than sent once.
        """
        for entry in self._volumes:
            if entry.get("radius") is not None or entry.get("clip") != (0.0, 1.0):
                self._send_volume_clip(entry)

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
        self._emit_loaded_changed()

    def set_model_visible(self, mid: str, visible: bool) -> None:
        """Show or hide a loaded model in the viewport."""
        entry = self._model_entry(mid)
        if entry is None or entry["visible"] == bool(visible):
            return
        entry["visible"] = bool(visible)
        self._reload_viewport()
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
                    negative_color: Optional[str] = None) -> str:
        """Register + show a volume: write its map (via cctbx) and compose the scene.

        ``color``/``iso`` override the defaults for maps that have a convention — a
        difference map is green at 3 sigma whatever colour the palette is up to.
        ``radius`` limits drawing to near the view centre (see :meth:`set_volume_radius`).
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
            "color": color or _VOLUME_COLORS[self._volume_counter % len(_VOLUME_COLORS)],
            "opacity": 1.0, "style": "surface", "clip": (0.0, 1.0), "mask_radius": None,
            "radius": radius, "negative_color": negative_color,
        })
        self._reload_viewport()  # re-asserts the clip; no session exists to tell yet
        self._emit_loaded_changed()
        return vid

    def set_volume_visible(self, vid: str, visible: bool) -> None:
        entry = self._volume_entry(vid)
        if entry is None or entry["visible"] == bool(visible):
            return
        entry["visible"] = bool(visible)
        self._reload_viewport()
        self._emit_loaded_changed()

    def remove_volume(self, vid: str) -> None:
        entry = self._volume_entry(vid)
        if entry is None:
            return
        self._volumes.remove(entry)
        self._prune_group(entry["group"])
        self._reload_viewport()
        self._emit_loaded_changed()

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
        """Unload a whole group (its model + maps) at once."""
        with self._batch_load():
            for m in [m for m in self._models if m["group"] == gid]:
                self.remove_model(m["id"])
            for v in [v for v in self._volumes if v["group"] == gid]:
                self.remove_volume(v["id"])
            for r in [r for r in self._reflections if r["group"] == gid]:
                self.remove_reflections(r["id"])

    def _prune_group(self, gid: Optional[str]) -> None:
        """Drop a group's name once it has no members left."""
        if gid is None:
            return
        members = (
            any(m["group"] == gid for m in self._models)
            or any(v["group"] == gid for v in self._volumes)
            or any(r["group"] == gid for r in self._reflections)
        )
        if not members:
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

    def _on_model_selection(self, mid: str, selection) -> None:
        """A model reported its picked atoms (WS thread). Fold into the scene selection."""
        with self._scene_lock:
            indices = list(selection.indices)
            if indices:
                self._scene_selection[mid] = indices
            else:
                self._scene_selection.pop(mid, None)
        self._emit_scene_selection()

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
             "types": list(self._type_groups(m).keys()), "hidden_types": sorted(m.get("hidden_types") or [])}
            for m in self._models
        ] + [
            {"kind": "volume", "id": v["id"], "name": v["name"], "visible": v["visible"],
             "active": False, "group": v["group"], "style": v.get("style"),
             "color": v.get("color"), "opacity": v.get("opacity"), "iso": v.get("iso")}
            for v in self._volumes
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
                if m.get("group") is None
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
        if mentry.get("group") is not None:
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
                gid = self._new_group(f"{mentry['name']} + {rentry['name']}",
                                      mmm=mmm, label="phased from reflections")
                mentry["group"] = gid
                rentry["group"] = gid
                # Before the batch: leaving it opens _emit_loaded_changed publishes the
                # summary, and the pane reads the fit from there.
                rentry["r_work"] = out["r_work"]
                rentry["r_free"] = out["r_free"]
                with self._batch_load():
                    for map_type in types:
                        colour, iso, negative = MAP_STYLE[map_type in DIFFERENCE_MAP_TYPES]
                        self._add_volume(
                            VolumeData.from_map_manager(
                                mmm.get_map_manager_by_id(map_type),
                                name=map_type, map_id=map_type),
                            map_type, group=gid, color=colour, iso=iso,
                            radius=self.view_radius_default, negative_color=negative)
                self._status(
                    f"{rentry['name']}: R-work {out['r_work']:.4f}, R-free {out['r_free']:.4f}"
                    f" — maps: {', '.join(types)}")

            self.bridge.run_on_main.emit(add_on_main)

        threading.Thread(target=work, name="pxviewer-phasing", daemon=True).start()

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

        The maps are replaced in place, so a level, a colour or a radius set on them
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

        threading.Thread(target=work, name="pxviewer-rephasing", daemon=True).start()

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
                colour, iso, negative = MAP_STYLE[is_difference_map(label)]
                volume = VolumeData.from_map_manager(
                    map_from_coefficients(coefficients), name=root_label(label))
                # A map from reflections fills the unit cell: open it with a radius,
                # or the model is lost inside a wall of density.
                self._add_volume(volume, root_label(label), group=gid,
                                 color=colour, iso=iso, radius=self.view_radius_default,
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
        if len(models) > 1:
            raise ValueError("a group can contain at most one model")
        if not maps:
            raise ValueError("a group needs at least one map file")

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

        # Loaded separately (not paired — pairing them is the demo), but in one batch so
        # the viewport reloads and the pane rebuilds once rather than twice.
        with self._batch_load():
            self._load_model_file(str(sample))
            self._load_reflection_file(mtz)
        self._status(
            f"X-ray demo: {SAMPLE_STRUCTURE[0]} + reflections — open the reflections and "
            "click Make maps")
        return "xray"

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
        self._selection_enabled = True
        for m in self._models:
            m["session"].enable_mouse_selection()

    def disable_mouse_selection(self) -> None:
        self._selection_enabled = False
        for m in self._models:
            try:
                m["session"].disable_mouse_selection()
            except Exception:  # pragma: no cover - defensive
                pass

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
        """Draw geometry notations for the selected restraint rows.

        ``specs`` is a list of ``(kind, i_seqs)``. Bonds/angles/dihedrals get their
        measurement notation (so exactly the participating atoms are marked, not the
        whole residue); chirality/planarity have no simple notation, so their atoms
        are highlighted instead. Multiple rows -> multiple notations.
        """
        session = self.session_for(mid)
        self._clear_restraint_notations()
        if session is None:
            return
        self._restraint_prim_session = session
        highlight: set = set()
        focus_atoms: set = set()
        for i, (kind, iseqs) in enumerate(specs):
            pid = f"geomsel-{i}"
            iseqs = list(iseqs)
            focus_atoms.update(iseqs)
            try:
                if kind == "bond" and len(iseqs) == 2:
                    session.add_distance(iseqs[0], iseqs[1], id=pid)
                elif kind == "angle" and len(iseqs) == 3:
                    session.add_angle(iseqs[0], iseqs[1], iseqs[2], id=pid)
                elif kind == "dihedral" and len(iseqs) == 4:
                    session.add_dihedral(iseqs[0], iseqs[1], iseqs[2], iseqs[3], id=pid)
                else:  # chirality / planarity: no notation, just mark the atoms
                    highlight.update(iseqs)
                    continue
                self._restraint_prim_ids.append(pid)
            except Exception:  # pragma: no cover - defensive (stale indices)
                highlight.update(iseqs)
        # Highlight only the atoms without a notation (empty list clears the overlay,
        # so a pure bond/angle/dihedral selection shows just the notation).
        try:
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
    gpu_backend.configure(gpu)
    _check_qt()

    desktop = DesktopApp(host=host, port=port)
    try:
        return desktop.start()
    finally:
        desktop.stop()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(run_desktop())
