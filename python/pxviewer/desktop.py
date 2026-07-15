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
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .demos import DEMOS, Player, list_demos
from .loader import (
    FILE_DIALOG_FILTER,
    SAMPLE_STRUCTURE,
    create_volume_file_view,
    file_kind,
    sample_structure_path,
)
from .volume_demos import create_volume_demo, list_volume_demos
from .webapp import Webapp

# A single off-screen atom is enough to keep the LiveSession WebSocket open and
# route click-mode / mouse-selection messages for scenes that carry no live model
# (e.g. a volume). Built through cctbx like every other session.
_DUMMY_KEY = "__dummy__"


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
        models_changed = Signal(object)     # [{id, name, visible, active}] for the Models list

    return _Bridge()


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


class ViewportWindow:
    """A Qt window wrapping the Mol* viewer in a QWebEngineView."""

    def __init__(self, title: str = "pxviewer — viewport"):
        _check_qt()

        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        self._window = QWidget()
        self._window.setWindowTitle(title)
        self._window.setMinimumSize(640, 480)

        layout = QVBoxLayout(self._window)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view = QWebEngineView()
        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        layout.addWidget(self._view)

    def load(self, url: str) -> None:
        from PySide6.QtCore import QUrl
        self._view.load(QUrl(url))

    def show(self) -> None:
        self._window.show()

    def set_geometry(self, rect) -> None:
        self._window.setGeometry(rect)

    def widget(self):
        return self._window


class ControlsWindow:
    """Controls for the viewport: open a file, or run a demo from the Demos tab."""

    def __init__(self, desktop: "DesktopApp", title: str = "pxviewer — controls"):
        _check_qt()

        from PySide6.QtWidgets import (
            QLabel,
            QPushButton,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        self._desktop = desktop
        self._window = QWidget()
        self._window.setWindowTitle(title)
        self._window.setMinimumSize(360, 520)

        layout = QVBoxLayout(self._window)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("<h2>pxviewer</h2>"))

        self._console = None  # EmbeddedConsole, created lazily on first tab view
        self._console_started = False

        tabs = QTabWidget()
        tabs.addTab(self._build_file_tab(), "File")
        tabs.addTab(self._build_geometry_tab(), "Geometry")
        console_tab = self._build_console_tab()
        self._console_tab_index = tabs.addTab(console_tab, "Console")
        tabs.addTab(self._build_demos_tab(), "Demos")
        # The console spins up an IPython kernel, so defer that cost until the tab
        # is actually opened.
        tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(tabs, stretch=1)

        # Display / Selection controls apply to whatever is loaded, so they sit
        # below the tabs — but they are hidden on the Console tab so the console
        # fills the whole pane.
        self._bottom_controls = QWidget()
        bottom = QVBoxLayout(self._bottom_controls)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(6)
        layout.addWidget(self._bottom_controls)

        bottom.addWidget(QLabel("<b>Display</b>"))

        self._interactions_btn = QPushButton("Show computed interactions")
        self._interactions_btn.setCheckable(True)
        self._interactions_btn.setToolTip(
            "Overlay Mol*-computed non-covalent contacts (hydrogen bonds, salt "
            "bridges, pi-stacking, hydrophobic) as dashed cylinders. For explicit, "
            "user-defined contacts, use LiveSession.set_interactions() from Python."
        )
        self._interactions_btn.clicked.connect(self._on_toggle_interactions)
        bottom.addWidget(self._interactions_btn)

        bottom.addWidget(QLabel("<b>Selection</b>"))

        self._select_btn = QPushButton("Enable selection mode")
        self._select_btn.setCheckable(True)
        self._select_btn.clicked.connect(self._on_toggle_select)
        bottom.addWidget(self._select_btn)

        self._clear_btn = QPushButton("Clear selection")
        self._clear_btn.clicked.connect(self._on_clear_selection)
        bottom.addWidget(self._clear_btn)

        self._selection_label = QLabel("none")
        self._selection_label.setWordWrap(True)
        bottom.addWidget(self._selection_label)

        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        bottom.addWidget(self._status_label)

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
        desktop.bridge.interactions_changed.connect(self._on_interactions_reset)
        desktop.bridge.models_changed.connect(self._on_models_changed)

    # -- tabs ------------------------------------------------------------

    def _build_file_tab(self):
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        blurb = QLabel("Open a model or volume from your computer. Models are read by cctbx.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self._open_btn = QPushButton("Load file…")
        self._open_btn.setMinimumHeight(44)
        self._open_btn.clicked.connect(self._on_open_file)
        layout.addWidget(self._open_btn)

        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(True)
        layout.addWidget(self._file_label)

        formats = QLabel("Models (cctbx): .pdb .ent .cif .mmcif\nVolumes: .mrc .map .ccp4")
        formats.setWordWrap(True)
        layout.addWidget(formats)

        # Loaded models: check to show/hide (one -> switch, several -> overlay); the
        # highlighted row is the active model (drives the atoms table + selection).
        layout.addSpacing(12)
        layout.addWidget(QLabel("<b>Loaded models</b>  (check = shown, selected row = active)"))
        self._models_list = QListWidget()
        self._models_list.setMaximumHeight(120)
        self._models_list.itemChanged.connect(self._on_model_item_toggled)
        self._models_list.currentRowChanged.connect(self._on_model_active_row)
        layout.addWidget(self._models_list)
        row = QHBoxLayout()
        self._remove_model_btn = QPushButton("Remove selected")
        self._remove_model_btn.clicked.connect(self._on_remove_model)
        row.addWidget(self._remove_model_btn)
        row.addStretch()
        layout.addLayout(row)

        layout.addSpacing(12)
        layout.addWidget(QLabel("<b>Sample</b>"))

        sample = sample_structure_path()
        self._sample_btn = QPushButton(f"Load {SAMPLE_STRUCTURE[1]}")
        self._sample_btn.clicked.connect(self._on_load_sample)
        if sample is None:
            # Shipped under tests/data, so it is missing from an installed wheel.
            self._sample_btn.setEnabled(False)
            self._sample_btn.setToolTip("The bundled sample is only present in a source checkout.")
        else:
            self._sample_btn.setToolTip(str(sample))
        layout.addWidget(self._sample_btn)

        layout.addStretch()
        return tab

    def _build_demos_tab(self):
        from PySide6.QtWidgets import (
            QComboBox,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        layout.addWidget(QLabel("<b>Model demos</b>"))
        model_blurb = QLabel("Animated coordinate streams from Python.")
        model_blurb.setWordWrap(True)
        layout.addWidget(model_blurb)

        self._model_select = QComboBox()
        for name, _ in list_demos():
            self._model_select.addItem(name, name)
        layout.addWidget(self._model_select)

        self._model_desc = QLabel("")
        self._model_desc.setWordWrap(True)
        layout.addWidget(self._model_desc)
        self._model_select.currentIndexChanged.connect(self._on_model_demo_changed)
        self._on_model_demo_changed()

        run_model = QPushButton("Run model demo")
        run_model.clicked.connect(self._on_run_model_demo)
        layout.addWidget(run_model)

        layout.addSpacing(8)
        layout.addWidget(QLabel("<b>Volume demos</b>"))

        self._volume_select = QComboBox()
        for name, _ in list_volume_demos():
            self._volume_select.addItem(name, name)
        layout.addWidget(self._volume_select)

        self._volume_desc = QLabel("")
        self._volume_desc.setWordWrap(True)
        layout.addWidget(self._volume_desc)
        self._volume_select.currentIndexChanged.connect(self._on_volume_demo_changed)
        self._on_volume_demo_changed()

        run_volume = QPushButton("Run volume demo")
        run_volume.clicked.connect(self._on_run_volume_demo)
        layout.addWidget(run_volume)

        layout.addSpacing(8)
        self._stop_btn = QPushButton("Stop demo")
        self._stop_btn.clicked.connect(self._on_stop_demo)
        layout.addWidget(self._stop_btn)

        layout.addStretch()
        return tab

    def _build_geometry_tab(self):
        from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        subtabs = QTabWidget()
        subtabs.addTab(self._build_atoms_subtab(), "Atoms")
        layout.addWidget(subtabs)
        return tab

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
        on_console = index == self._console_tab_index
        # Give the console the whole pane; the Display/Selection controls only
        # make sense for the other tabs anyway.
        self._bottom_controls.setVisible(not on_console)
        if on_console:
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

        self._filter_selection_check = QCheckBox("Show only selected atoms")
        self._filter_selection_check.setToolTip(
            "Collapse the table to just the atoms selected in this model — handy when "
            "the selection is scattered across chains."
        )
        self._filter_selection_check.toggled.connect(self._on_filter_toggled)
        layout.addWidget(self._filter_selection_check)

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

        path, _ = QFileDialog.getOpenFileName(
            self._window, "Open model or volume", "", FILE_DIALOG_FILTER
        )
        if not path:
            return
        try:
            kind = self._desktop.load_file(path)
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not load file", str(exc))
            self._set_status(f"Failed to load {Path(path).name}")
            return
        self._file_label.setText(f"{Path(path).name}  ({kind})")

    def _on_load_sample(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        sample = sample_structure_path()
        if sample is None:
            QMessageBox.warning(self._window, "Sample not available", "The bundled sample file is missing.")
            return
        try:
            kind = self._desktop.load_file(str(sample))
        except Exception as exc:
            QMessageBox.warning(self._window, "Could not load sample", str(exc))
            return
        self._file_label.setText(f"{sample.name}  ({kind})")

    def _on_model_demo_changed(self) -> None:
        name = self._model_select.currentData()
        demo = DEMOS.get(name)
        self._model_desc.setText(demo.description if demo else "")

    def _on_volume_demo_changed(self) -> None:
        name = self._volume_select.currentData()
        descriptions = dict(list_volume_demos())
        self._volume_desc.setText(descriptions.get(name, ""))

    def _on_run_model_demo(self) -> None:
        name = self._model_select.currentData()
        if name:
            self._desktop.load_model_demo(name)

    def _on_run_volume_demo(self) -> None:
        name = self._volume_select.currentData()
        if name:
            self._desktop.load_volume_demo(name)

    def _on_stop_demo(self) -> None:
        self._desktop.stop_demo()

    def _on_toggle_interactions(self, checked: bool) -> None:
        self._desktop.set_computed_interactions(checked)
        self._interactions_btn.setText(
            "Hide computed interactions" if checked else "Show computed interactions"
        )

    def _on_interactions_reset(self, visible: bool) -> None:
        # A fresh load clears the overlay; keep the button in sync with that.
        self._interactions_btn.setChecked(visible)
        self._interactions_btn.setText(
            "Hide computed interactions" if visible else "Show computed interactions"
        )

    def _on_toggle_select(self, checked: bool) -> None:
        if checked:
            self._desktop.enable_mouse_selection()
            self._select_btn.setText("Disable selection mode")
        else:
            self._desktop.disable_mouse_selection()
            self._select_btn.setText("Enable selection mode")

    def _on_clear_selection(self) -> None:
        self._desktop.clear_selection()

    def _on_scene_selection_changed(self, scene) -> None:
        """A model's picks changed. Refresh the aggregate label + the atoms table."""
        self._scene_selection = scene or {}
        total = sum(len(v) for v in self._scene_selection.values())
        n_models = len(self._scene_selection)
        if total:
            across = f" across {n_models} models" if n_models > 1 else ""
            self._selection_label.setText(f"{total} atom(s){across}")
        else:
            self._selection_label.setText("none")
        # Viewer -> table: reflect the picks belonging to the table's model.
        self._apply_table_selection()

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
        self._apply_table_selection()

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

    def _on_table_model_combo_changed(self, _index: int) -> None:
        if self._suppress_table_model_combo:
            return
        from PySide6.QtCore import Qt

        mid = self._table_model_combo.currentData(Qt.ItemDataRole.UserRole)
        # Picking the active model again resumes auto-follow; any other choice pins.
        active = next((m["id"] for m in self._models_summary if m["active"]), None)
        self._table_pinned = mid is not None and mid != active
        self._set_table_model(mid)

    # -- loaded-models list ----------------------------------------------

    def _on_models_changed(self, summary) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem

        self._models_summary = summary
        self._suppress_model_events = True
        try:
            self._models_list.clear()
            active_row = -1
            for i, m in enumerate(summary):
                item = QListWidgetItem(m["name"])
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if m["visible"] else Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, m["id"])
                self._models_list.addItem(item)
                if m["active"]:
                    active_row = i
            if active_row >= 0:
                self._models_list.setCurrentRow(active_row)
        finally:
            self._suppress_model_events = False
        self._sync_table_model_combo(summary)
        self._refresh_console_session()

    def _sync_table_model_combo(self, summary) -> None:
        """Rebuild the table's model dropdown, following the active model unless pinned."""
        from PySide6.QtCore import Qt

        active = next((m["id"] for m in summary if m["active"]), None)
        ids = {m["id"] for m in summary}
        if not self._table_pinned or self._table_model_id not in ids:
            self._table_pinned = False
            target = active
        else:
            target = self._table_model_id

        self._suppress_table_model_combo = True
        try:
            self._table_model_combo.clear()
            for m in summary:
                self._table_model_combo.addItem(m["name"], m["id"])
            idx = next((i for i, m in enumerate(summary) if m["id"] == target), -1)
            if idx >= 0:
                self._table_model_combo.setCurrentIndex(idx)
        finally:
            self._suppress_table_model_combo = False
        self._set_table_model(target)

    def _on_model_item_toggled(self, item) -> None:
        from PySide6.QtCore import Qt

        if self._suppress_model_events:
            return
        self._desktop.set_model_visible(
            item.data(Qt.ItemDataRole.UserRole), item.checkState() == Qt.CheckState.Checked
        )

    def _on_model_active_row(self, row) -> None:
        from PySide6.QtCore import Qt

        if self._suppress_model_events or row < 0:
            return
        item = self._models_list.item(row)
        if item is not None:
            self._desktop.set_active_model(item.data(Qt.ItemDataRole.UserRole))

    def _on_remove_model(self) -> None:
        from PySide6.QtCore import Qt

        item = self._models_list.currentItem()
        if item is not None:
            self._desktop.remove_model(item.data(Qt.ItemDataRole.UserRole))

    def _on_table_selection_changed(self) -> None:
        if not self._suppress_table_sync:
            self._table_sync_timer.start()  # debounce a drag-select

    def _push_table_selection_to_viewer(self) -> None:
        rows = [idx.row() for idx in self._atom_view.selectionModel().selectedRows()]
        atoms = [self._atom_model.row_atom(r) for r in rows]
        self._desktop.highlight_atoms_in(self._table_model_id, atoms)

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

        self._webapp = Webapp(host=host, port=port)

        self._session: Optional[Any] = None  # the ACTIVE session (a model, or a dummy)
        self._session_key: Optional[str] = None
        # Loaded models: list of {id, name, session, visible}. The viewport shows the
        # visible ones (one -> switch, several -> simultaneous). ``_session`` points at
        # the active model (drives the atoms table + selection sync).
        self._models: List[dict] = []
        self._model_counter = 0
        self._active_model_id: Optional[str] = None
        # Scene-level selection: {model_id: [atom indices]}. Each model reports its
        # own picks independently (a selection may span models — e.g. protein +
        # ligand); the union across models is the scene selection. Mutated on the
        # WebSocket threads, read on the GUI thread, so guard it.
        self._scene_selection: dict = {}
        self._scene_lock = threading.Lock()
        self._player: Optional[Player] = None
        self._demo_thread: Optional[threading.Thread] = None
        self._selection_enabled = False
        self._computed_interactions_visible = False
        self._load_counter = 0

        self._stopped = False
        self._prev_sigint = None
        self._sigint_installed = False
        self._sigint_timer = None

        self.bridge = _make_bridge()
        self._viewport = ViewportWindow()
        self._controls = ControlsWindow(self)

        # Closing either window quits the app; tear the backend down on the way out
        # so background threads stop before Qt destroys the widgets they signal.
        self._close_filter = _make_close_filter(self._app.quit)
        self._viewport.widget().installEventFilter(self._close_filter)
        self._controls.widget().installEventFilter(self._close_filter)
        self._app.aboutToQuit.connect(self.stop)

    # -- lifecycle -------------------------------------------------------

    def start(self) -> int:
        self._webapp.start()

        self._viewport.show()
        self._controls.show()
        self._arrange_windows()

        # Land on an empty viewer: the main screen is "load a file", not a demo.
        ws_url = self._start_dummy()
        self._viewport.load(f"{self._webapp.url}index.html?ws={ws_url}")
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
        self._clear_models()  # stops all model sessions and the active/dummy one
        self._webapp.stop()

    def _arrange_windows(self) -> None:
        """Place the two windows side by side on the primary screen."""
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        screen = app.primaryScreen()
        if screen is None:
            # Fallback geometry if no screen is reported.
            rect = QRect(0, 0, 1920, 1080)
        else:
            rect = screen.availableGeometry()

        x, y, total_width, total_height = rect.x(), rect.y(), rect.width(), rect.height()
        half_width = total_width // 2

        self._viewport.set_geometry(QRect(x, y, half_width, total_height))
        self._controls.set_geometry(QRect(x + half_width, y, total_width - half_width, total_height))

    # -- live session ----------------------------------------------------

    # -- model registry (multi-model) ------------------------------------

    def _model_entry(self, mid):
        return next((m for m in self._models if m["id"] == mid), None)

    def _visible_model_ws(self) -> str:
        """Comma-separated ws URLs of the visible models (one -> switch, many -> overlay)."""
        return ",".join(
            f"ws://{self._host}:{m['session'].port}" for m in self._models if m["visible"]
        )

    def _emit_models_changed(self) -> None:
        self.bridge.models_changed.emit([
            {"id": m["id"], "name": m["name"], "visible": m["visible"], "active": m["id"] == self._active_model_id}
            for m in self._models
        ])

    def _wire_active(self, session) -> None:
        """Point the active session at ``session`` (the default table model + display target).

        Selection is scene-wide (enabled per model, not tied to the active one), so
        switching the active model no longer re-wires any picking — it just moves
        which model the atoms table defaults to.
        """
        self._session = session
        self._session_key = None
        self.bridge.structure_changed.emit(session)

    def _reload_model_viewport(self) -> None:
        ws = self._visible_model_ws()
        if ws:
            self._viewport.load(f"{self._webapp.url}index.html?ws={ws}")

    def _add_model(self, session, name: str) -> str:
        """Register + show a model session (visible + active); returns its id."""
        session.start(host=self._host, port=0)
        self._model_counter += 1
        mid = f"model-{self._model_counter}"
        self._models.append({"id": mid, "name": name, "session": session, "visible": True})
        self._active_model_id = mid
        # Register this model's pick handler once (tagged with its id); the click
        # mode is what actually turns picking on/off. Registering here means a
        # selection can be built in any loaded model, not just the active one.
        session.on_selection(lambda sel, mid=mid: self._on_model_selection(mid, sel))
        if self._selection_enabled:
            session.enable_mouse_selection()  # handler already registered; just arm click mode
        self._wire_active(session)
        self._reload_model_viewport()
        self._emit_models_changed()
        return mid

    def set_active_model(self, mid: str) -> None:
        """Make a loaded model the active one (the atoms table + selection follow it)."""
        entry = self._model_entry(mid)
        if entry is None or self._active_model_id == mid:
            return
        self._active_model_id = mid
        self._wire_active(entry["session"])  # no viewport reload: visibility is unchanged
        self._emit_models_changed()

    def set_model_visible(self, mid: str, visible: bool) -> None:
        """Show or hide a loaded model in the viewport."""
        entry = self._model_entry(mid)
        if entry is None or entry["visible"] == bool(visible):
            return
        entry["visible"] = bool(visible)
        self._reload_model_viewport()
        self._emit_models_changed()

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
        self._reload_model_viewport()
        self._emit_models_changed()
        if dropped:
            self._emit_scene_selection()

    def _clear_models(self) -> None:
        """Stop and drop every loaded model, plus the active/dummy session."""
        for m in list(self._models):
            try:
                m["session"].stop()
            except Exception:  # pragma: no cover - defensive
                pass
        self._models.clear()
        self._active_model_id = None
        with self._scene_lock:
            self._scene_selection.clear()
        if self._session is not None and self._session_key == _DUMMY_KEY:
            try:
                self._session.stop()
            except Exception:  # pragma: no cover - defensive
                pass
        self._session = None
        self._session_key = None

    def _start_dummy(self) -> str:
        """A 1-atom session that keeps the WS channel open for a volume scene."""
        session = _dummy_session()
        session.start(host=self._host, port=0)
        self._session = session
        self._session_key = _DUMMY_KEY
        # The dummy carries a single off-screen atom (a volume scene has no model),
        # so there is nothing to pick — selection stays a model-only affair.
        return f"ws://{self._host}:{session.port}"

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

    def session_for(self, mid: Optional[str]):
        """The LiveSession for a model id (or None) — used by the atoms table."""
        entry = self._model_entry(mid) if mid else None
        return entry["session"] if entry else None

    def active_model_session(self):
        """The active model's LiveSession, or None (e.g. a volume scene)."""
        return self.session_for(self._active_model_id)

    # -- loading ---------------------------------------------------------

    def load_file(self, path: str) -> str:
        """Open a local model or volume file in the viewport. Returns its kind.

        Atomic models are read by cctbx and streamed through a live session (no
        browser parsing); volumes are still staged as an MVSJ scene.
        """
        if file_kind(path) == "volume":
            return self._load_volume_file(path)
        return self._load_model_file(path)

    def _load_model_file(self, path: str) -> str:
        """Read a model with cctbx and add it to the viewport (alongside any others)."""
        self.stop_demo()
        self._reset_interactions()

        from . import cctbx_io
        from .live import LiveSession

        # DataManager -> model -> hierarchy; the native model is retained on the
        # session so selection uses cctbx's machinery.
        session = LiveSession.from_model_file(path)
        # The topology BinaryCIF already carries the coordinates, so the structure
        # appears without pushing a frame. Cartoon reads better than ball-and-stick
        # for a polymer; the choice is replayed to the viewer when it connects.
        if cctbx_io.model_is_polymer(session.model):
            session.set_representation("cartoon", color="secondary-structure")
        self._add_model(session, Path(path).name)
        self._status(f"Loaded model: {Path(path).name} ({session._n_atoms} atoms)")
        return "model"

    def _load_volume_file(self, path: str) -> str:
        """Stage a volume file as an MVSJ scene and load it in the viewport."""
        self.stop_demo()
        self._reset_interactions()
        self._clear_models()  # a volume is its own view, not a model

        # Each load gets a fresh directory, so a reloaded page can never pick up
        # a cached scene or data file from the previous one.
        self._load_counter += 1
        out_dir = self._webapp.volume_dir / "file" / str(self._load_counter)
        mvsj_path = create_volume_file_view(path, out_dir=out_dir)

        ws_url = self._start_dummy()
        mvsj_url = f"/file/{self._load_counter}/{mvsj_path.name}"
        self._viewport.load(f"{self._webapp.url}index.html?mvsj={mvsj_url}&ws={ws_url}")
        self.bridge.structure_changed.emit(None)  # no atoms table for a volume
        self._emit_models_changed()  # empty models list
        self._emit_scene_selection()  # dropped any prior model selection
        self._status(f"Loaded volume: {Path(path).name}")
        return "volume"

    def load_volume_demo(self, name: str) -> None:
        """Generate a volume demo and load its static scene in the viewport."""
        self.stop_demo()
        self._reset_interactions()
        self._clear_models()

        demo_dir = self._webapp.volume_dir / name
        demo_dir.mkdir(parents=True, exist_ok=True)
        create_volume_demo(
            name,
            mrc_path=demo_dir / "volume.mrc",
            mvsj_path=demo_dir / "volume.mvsj",
            shape=(32, 32, 32),
        )

        ws_url = self._start_dummy()
        mvsj_url = f"/demo/{name}/volume.mvsj"
        self._viewport.load(f"{self._webapp.url}index.html?mvsj={mvsj_url}&ws={ws_url}")
        self.bridge.structure_changed.emit(None)
        self._emit_models_changed()
        self._emit_scene_selection()
        self._status(f"Volume demo: {name}")

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
        self._clear_models()  # a demo is a single animated model
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


def run_desktop(host: str = "127.0.0.1", port: int = 5173) -> int:
    """Start the desktop app with viewport and controls windows."""
    _check_qt()

    desktop = DesktopApp(host=host, port=port)
    try:
        return desktop.start()
    finally:
        desktop.stop()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(run_desktop())
