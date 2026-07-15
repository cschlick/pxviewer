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
        selection_changed = Signal(object)
        status_changed = Signal(str)
        interactions_changed = Signal(bool)
        structure_changed = Signal(object)  # the newly installed LiveSession (or None)

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

        def set_session(self, session) -> None:
            self.beginResetModel()
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

        def rowCount(self, parent=QModelIndex()):
            return 0 if parent.isValid() else self._n

        def columnCount(self, parent=QModelIndex()):
            return 0 if parent.isValid() else len(self._headers)

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid():
                return None
            values, kind = self._cols[index.column()]
            if role == Qt.ItemDataRole.DisplayRole:
                if kind == "idx":
                    return str(index.row())
                v = values[index.row()]
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

        tabs = QTabWidget()
        tabs.addTab(self._build_file_tab(), "File")
        tabs.addTab(self._build_geometry_tab(), "Geometry")
        tabs.addTab(self._build_demos_tab(), "Demos")
        layout.addWidget(tabs, stretch=1)

        # These apply to whatever is loaded, so they sit below the tabs.
        layout.addWidget(QLabel("<b>Display</b>"))

        self._interactions_btn = QPushButton("Show computed interactions")
        self._interactions_btn.setCheckable(True)
        self._interactions_btn.setToolTip(
            "Overlay Mol*-computed non-covalent contacts (hydrogen bonds, salt "
            "bridges, pi-stacking, hydrophobic) as dashed cylinders. For explicit, "
            "user-defined contacts, use LiveSession.set_interactions() from Python."
        )
        self._interactions_btn.clicked.connect(self._on_toggle_interactions)
        layout.addWidget(self._interactions_btn)

        layout.addWidget(QLabel("<b>Selection</b>"))

        self._select_btn = QPushButton("Enable selection mode")
        self._select_btn.setCheckable(True)
        self._select_btn.clicked.connect(self._on_toggle_select)
        layout.addWidget(self._select_btn)

        self._clear_btn = QPushButton("Clear selection")
        self._clear_btn.clicked.connect(self._on_clear_selection)
        layout.addWidget(self._clear_btn)

        self._selection_label = QLabel("none")
        self._selection_label.setWordWrap(True)
        layout.addWidget(self._selection_label)

        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        desktop.bridge.selection_changed.connect(self._on_selection_changed)
        desktop.bridge.status_changed.connect(self._set_status)
        desktop.bridge.interactions_changed.connect(self._on_interactions_reset)
        desktop.bridge.structure_changed.connect(self._on_structure_changed)

    # -- tabs ------------------------------------------------------------

    def _build_file_tab(self):
        from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

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

    def _build_atoms_subtab(self):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QAbstractItemView, QLabel, QTableView, QVBoxLayout, QWidget

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

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
        self._selection_label.setText("none")

    def _on_selection_changed(self, selection) -> None:
        indices = list(selection.indices)
        if indices:
            shown = indices[:12]
            suffix = "…" if len(indices) > len(shown) else ""
            self._selection_label.setText(f"{len(indices)} atom(s): {shown}{suffix}")
        else:
            self._selection_label.setText("none")
        # Viewer -> table: reflect the picked atoms as selected rows.
        self._select_table_rows(indices)

    # -- geometry / atoms table ------------------------------------------

    def _on_structure_changed(self, session) -> None:
        self._atom_model.set_session(session)
        n = self._atom_model.rowCount()
        self._atoms_count.setText(f"{n} atoms" if n else "No structure loaded")

    def _on_table_selection_changed(self) -> None:
        if not self._suppress_table_sync:
            self._table_sync_timer.start()  # debounce a drag-select

    def _push_table_selection_to_viewer(self) -> None:
        rows = [idx.row() for idx in self._atom_view.selectionModel().selectedRows()]
        self._desktop.highlight_atoms(rows)

    def _select_table_rows(self, indices) -> None:
        """Select the given atom rows in the table without echoing back to the viewer."""
        from PySide6.QtCore import QItemSelection, QItemSelectionModel

        model = self._atom_model
        view = self._atom_view
        sm = view.selectionModel()
        self._suppress_table_sync = True
        try:
            sm.clearSelection()
            ncols = model.columnCount()
            valid = [i for i in indices if 0 <= i < model.rowCount()]
            if valid and ncols:
                selection = QItemSelection()
                last = ncols - 1
                for start, end in _runs(valid):  # contiguous ranges keep this cheap
                    selection.select(model.index(start, 0), model.index(end, last))
                sm.select(selection, QItemSelectionModel.SelectionFlag.Select)
                view.scrollTo(model.index(valid[0], 0))
        finally:
            self._suppress_table_sync = False


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

        self._session: Optional[Any] = None
        self._session_key: Optional[str] = None
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
        ws_url = self._ensure_session(_DUMMY_KEY, _dummy_session)
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
        self.stop_demo()
        if self._session is not None:
            self._session.stop()
            self._session = None
            self._session_key = None
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

    def _install_session(self, session, key: str) -> str:
        """Swap in a prebuilt session, start it, and return its WebSocket URL.

        Replaces the current session (a session's atom set is fixed for life, so a
        different model means a new session) and rewires mouse selection if it was
        enabled.
        """
        if self._session is not None:
            self._session.stop()
        self._session = session
        self._session_key = key
        session.start(host=self._host, port=0)
        if self._selection_enabled:
            session.enable_mouse_selection(self._emit_selection)
        self.bridge.structure_changed.emit(session)  # refresh the atoms table
        return f"ws://{self._host}:{session.port}"

    def _ensure_session(self, key: str, make_session) -> str:
        """Return the WebSocket URL for a session keyed by ``key``.

        Same key -> reuse, which keeps page reloads cheap; otherwise build a new
        session via ``make_session()`` and install it.
        """
        if self._session is not None and self._session_key == key:
            return f"ws://{self._host}:{self._session.port}"
        return self._install_session(make_session(), key)

    def _status(self, text: str) -> None:
        self.bridge.status_changed.emit(text)

    def _emit_selection(self, selection) -> None:
        # Called on the WebSocket thread; hop to the GUI thread via the bridge.
        self.bridge.selection_changed.emit(selection)

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
        """Read a model with cctbx and stream it through a fresh live session."""
        self.stop_demo()
        self._reset_interactions()

        from . import cctbx_io
        from .live import LiveSession

        # DataManager -> model -> hierarchy; the native model is retained on the
        # session so selection uses cctbx's machinery.
        session = LiveSession.from_model_file(path)
        self._load_counter += 1
        ws_url = self._install_session(session, key=f"model:{self._load_counter}")
        # The topology BinaryCIF already carries the coordinates, so the structure
        # appears without pushing a frame. Cartoon reads better than ball-and-stick
        # for a polymer; the choice is replayed to the viewer when it connects.
        if cctbx_io.model_is_polymer(session.model):
            session.set_representation("cartoon", color="secondary-structure")
        self._viewport.load(f"{self._webapp.url}index.html?ws={ws_url}")
        self._status(f"Loaded model: {Path(path).name} ({session._n_atoms} atoms)")
        return "model"

    def _load_volume_file(self, path: str) -> str:
        """Stage a volume file as an MVSJ scene and load it in the viewport."""
        self.stop_demo()
        self._reset_interactions()

        # Each load gets a fresh directory, so a reloaded page can never pick up
        # a cached scene or data file from the previous one.
        self._load_counter += 1
        out_dir = self._webapp.volume_dir / "file" / str(self._load_counter)
        mvsj_path = create_volume_file_view(path, out_dir=out_dir)

        ws_url = self._ensure_session(_DUMMY_KEY, _dummy_session)
        mvsj_url = f"/file/{self._load_counter}/{mvsj_path.name}"
        self._viewport.load(f"{self._webapp.url}index.html?mvsj={mvsj_url}&ws={ws_url}")
        self._status(f"Loaded volume: {Path(path).name}")
        return "volume"

    def load_volume_demo(self, name: str) -> None:
        """Generate a volume demo and load its static scene in the viewport."""
        self.stop_demo()
        self._reset_interactions()

        demo_dir = self._webapp.volume_dir / name
        demo_dir.mkdir(parents=True, exist_ok=True)
        create_volume_demo(
            name,
            mrc_path=demo_dir / "volume.mrc",
            mvsj_path=demo_dir / "volume.mvsj",
            shape=(32, 32, 32),
        )

        ws_url = self._ensure_session(_DUMMY_KEY, _dummy_session)
        mvsj_url = f"/demo/{name}/volume.mvsj"
        self._viewport.load(f"{self._webapp.url}index.html?mvsj={mvsj_url}&ws={ws_url}")
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
        ws_url = self._install_session(session, key=f"demo:{name}")

        base = np.asarray(sites, dtype="<f4")
        player = Player(session, base, fps=fps)
        session.on_pick(player._on_pick)
        self._player = player

        self._viewport.load(f"{self._webapp.url}index.html?ws={ws_url}")
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
        self._selection_enabled = True
        if self._session is not None:
            self._session.enable_mouse_selection(self._emit_selection)

    def disable_mouse_selection(self) -> None:
        self._selection_enabled = False
        if self._session is not None:
            self._session.disable_mouse_selection()

    def clear_selection(self) -> None:
        if self._session is not None:
            self._session.clear_selection()

    def highlight_atoms(self, indices) -> None:
        """Highlight atoms in the viewer (table -> viewer selection sync)."""
        if self._session is not None:
            try:
                self._session.highlight(list(indices))
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
