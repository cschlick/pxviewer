"""Live coordinate streaming between Python and the pxviewer Mol* frontend.

`LiveSession` runs a small WebSocket server that streams a *fixed topology* once
and then streams coordinate frames for that same topology, driving Mol*'s
in-place (Level 1) trajectory update path in the browser. It is duplex: the
frontend reports pick events (clicks) back over the same socket.

## Wire protocol (pxviewer-live/1)

The topology (which atoms exist, their names/elements/residues/bonds) is sent
*once*; every subsequent frame carries only coordinates, positionally aligned to
that topology. See `ATOM_IDENTITY_CONTRACT` below.

Server -> client (binary, little-endian; first uint32 is a tag):
  - tag 0 TOPOLOGY : [u32 tag=0][BinaryCIF bytes]
  - tag 1 FRAME    : [u32 tag=1][u32 frameIndex][f32 * 3N]  (x0,y0,z0,x1,y1,z1,...)

Server -> client (UTF-8 JSON text control messages):
  - {"type": "axis", "visible": bool}                          toggle the axis helper
  - {"type": "highlight", "atoms": <index-set>}                show selection overlay
  - {"type": "focus", "atoms": <index-set>}                    aim the camera
  - {"type": "primitive", "action": "add",                     draw a measurement:
     "kind": "angle"|"distance"|"dihedral"|"label",             angle/distance/
     "id": str, "groups": [[int,...],...], "options": {...}}    dihedral/label
  - {"type": "primitive", "action": "remove", "id": str}       remove one primitive
  - {"type": "primitive", "action": "clear"}                   remove all primitives
  - {"type": "representations", "reprs": [{...}, ...]}          declarative repr list
  - {"type": "click-mode", "mode": str}                        'select'|measure|'off'

Client -> server (UTF-8 JSON text):
  - {"type": "ready"}                              after topology is parsed
  - {"type": "pick", "empty": bool, "atom": {...}} on click (atom omitted if empty)
  - {"type": "mouse-selection", "indices": [int]} click-built selection ('select')
  - {"type": "measure", "kind": str, "atoms": [int]} click-built measurement

All atoms are addressed by *positional index* (row in the topology's _atom_site
table). Selections are resolved to indices entirely on the Python side; the wire
never carries a query language. An <index-set> is either {"list": [int,...]} or a
run-length {"runs": [[start,end],...]} (see `_encode_index_set`), and a highlight
with an empty set clears the selection.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import struct
import threading
from http import HTTPStatus
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np

from .data import Atom, encode_bcif

__all__ = ["LiveSession", "Selection", "Primitive", "ATOM_IDENTITY_CONTRACT"]

_TAG_TOPOLOGY = 0
_TAG_FRAME = 1

ATOM_IDENTITY_CONTRACT = """\
pxviewer atom-identity contract
-------------------------------
The topology sent at connect time defines the atom set and its order. Every
coordinate frame is *positional*: value triple i (x,y,z) always refers to the
same atom as row i of the topology's _atom_site table. Consequences:

  * The atom count is fixed for the lifetime of a session. A frame whose length
    does not match the topology is rejected (Mol* raises on element-count
    mismatch), rather than silently mis-assigning coordinates.
  * You may not add, remove, or reorder atoms mid-stream. To change the atom set,
    start a new session (new topology).
  * Per-atom identity (id/name/resname/resseq/chain) lives only in the topology;
    frames never resend it. Pick events reference atoms by that stable identity.

This is what makes the update "in-place": the browser reuses the parsed topology
(hierarchy + bonds) and swaps only the conformation for each frame.
"""


def _encode_index_set(indices: Iterable[int]) -> dict:
    """Compactly encode a set of atom indices for the wire.

    Selections are often contiguous (a chain, a residue range), so run-length
    encoding collapses them to a few ``[start, end]`` pairs. When runs would be
    larger than the explicit list (a scattered set), the plain list is sent
    instead. Indices are sorted and de-duplicated.
    """
    idx = sorted({int(i) for i in indices})
    runs: List[List[int]] = []
    for i in idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    if len(runs) * 2 <= len(idx):
        return {"runs": runs}
    return {"list": idx}


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """Angle a-b-c at vertex ``b``, in degrees (None if a ray has zero length)."""
    va, vc = a - b, c - b
    na, nc = np.linalg.norm(va), np.linalg.norm(vc)
    if na == 0 or nc == 0:
        return None
    cosang = float(np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    """Signed dihedral across the p1-p2 axis, in degrees on (-180, 180]."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    b1n = np.linalg.norm(b1)
    if np.linalg.norm(n1) == 0 or np.linalg.norm(n2) == 0 or b1n == 0:
        return None
    m1 = np.cross(n1, b1 / b1n)
    return float(np.degrees(np.arctan2(float(np.dot(m1, n2)), float(np.dot(n1, n2)))))


def _coords_to_f32(coords: Any, n_atoms: int) -> np.ndarray:
    """Normalise an (N,3) coordinate input to a contiguous little-endian f32 array."""
    arr = np.ascontiguousarray(np.asarray(coords, dtype="<f4"))
    if arr.ndim == 2 and arr.shape == (n_atoms, 3):
        return arr
    flat = arr.reshape(-1)
    if flat.size != n_atoms * 3:
        raise ValueError(
            f"frame has {flat.size} values but topology has {n_atoms} atoms "
            f"({n_atoms * 3} values expected); see ATOM_IDENTITY_CONTRACT"
        )
    return flat.reshape(n_atoms, 3)


@dataclasses.dataclass
class Selection:
    """A set of atoms addressed by positional index.

    ``indices`` are positional (0-based) rows into the topology's _atom_site
    table — the same stable identity the rest of the live protocol uses (see
    ``ATOM_IDENTITY_CONTRACT``); they are kept sorted and de-duplicated. ``atoms``
    are the corresponding ``Atom`` objects; ``ids`` and ``mask`` are derived views.
    Build one with :meth:`LiveSession.select_by`.
    """

    indices: List[int]
    atoms: List[Atom]
    n_total: int

    @property
    def ids(self) -> List[int]:
        """The ``_atom_site.id`` of each matched atom."""
        return [a.id for a in self.atoms]

    @property
    def mask(self) -> np.ndarray:
        """A boolean array of length ``n_total``, True at matched positions."""
        m = np.zeros(self.n_total, dtype=bool)
        if self.indices:
            m[self.indices] = True
        return m

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __repr__(self) -> str:
        head = self.indices[:8]
        tail = "..." if len(self.indices) > 8 else ""
        return f"Selection({len(self.indices)} atoms, indices={head}{tail})"


@dataclasses.dataclass
class Primitive:
    """Handle to a measurement primitive drawn in the viewer.

    ``id`` removes it via :meth:`LiveSession.remove_primitive`. ``kind`` is one of
    ``"angle"``, ``"distance"``, ``"dihedral"``, ``"label"``. ``value`` is the
    quantity measured in Python from the latest known coordinates — degrees for
    ``angle``/``dihedral``, Ångström for ``distance``, ``None`` for ``label`` (or
    when it can't be computed, e.g. coincident points). ``selections`` are the
    atom groups the primitive spans (1 for a label, 2 distance, 3 angle, 4
    dihedral); each group's centroid is used when it holds more than one atom.
    """

    id: str
    kind: str
    value: Optional[float]
    selections: List["Selection"]
    text: Optional[str] = None

    @property
    def degrees(self) -> Optional[float]:
        """The measured angle, for ``angle``/``dihedral`` primitives."""
        return self.value if self.kind in ("angle", "dihedral") else None

    @property
    def distance(self) -> Optional[float]:
        """The measured distance in Ångström, for ``distance`` primitives."""
        return self.value if self.kind == "distance" else None


class LiveSession:
    """Serve a fixed topology and stream coordinate frames to Mol* clients.

    Example:
        atoms = [Atom(id=1, element="C", x=0, y=0, z=0), ...]
        session = LiveSession(atoms)
        session.on_pick(lambda info: print("picked", info))
        session.start()                       # background thread, ws://127.0.0.1:8787
        for frame in trajectory:              # frame: (N,3) array-like
            session.push(frame)
    """

    def __init__(
        self,
        atoms: Iterable[Atom],
        *,
        polymer: bool = False,
        secondary_structure: Optional[Any] = None,
    ):
        self.atoms: List[Atom] = list(atoms)
        if not self.atoms:
            raise ValueError("LiveSession requires at least one atom")
        self._topology: bytes = encode_bcif(
            self.atoms, polymer=polymer, secondary_structure=secondary_structure
        )
        self._n_atoms = len(self.atoms)

        self._frame_index = 0
        self._last_frame: Optional[bytes] = None
        self._pick_handlers: List[Callable[[Optional[dict]], None]] = []

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Any = None
        self._clients: set = set()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._client_ready = threading.Event()

        self._lock = threading.Lock()  # guards the primitive counter

        # Atom-id -> positional index, for building selections by id.
        self._id_to_index = {atom.id: i for i, atom in enumerate(self.atoms)}
        # Active highlight (positional indices), replayed to clients that connect later.
        self._last_highlight_indices: List[int] = []
        # Active drawing primitives (id -> the "add" message), replayed to late clients.
        self._primitives: dict = {}
        self._primitive_counter = 0
        # Representations (id -> spec), sent declaratively; replayed to late clients.
        self._representations: dict = {}
        self._representation_counter = 0

        # Click interaction mode: 'off' | 'select' | 'distance' | 'angle' | 'dihedral'
        # | 'label'. In 'select' the user builds a selection streamed back here; in a
        # measure mode they click N atoms and the primitive is drawn.
        self._click_mode = "off"
        self._mouse_selection_indices: List[int] = []
        self._selection_handlers: List[Callable[[Selection], None]] = []
        self._measure_handlers: List[Callable[[Primitive], None]] = []
        self._selection_changed = threading.Event()
        # Per-connection send lock: websockets forbids concurrent send() on one
        # connection, so serialize sends (this also preserves message order).
        self._send_locks: dict = {}

        self.host = "127.0.0.1"
        self.port = 8787

    # -- scene -> python -------------------------------------------------

    def on_pick(self, handler: Callable[[Optional[dict]], None]) -> None:
        """Register a callback invoked (on the server's loop thread) for each pick."""
        self._pick_handlers.append(handler)

    def wait_for_client(self, timeout: Optional[float] = None) -> bool:
        """Block until a client has connected and reported it parsed the topology.

        Returns True if a client became ready within ``timeout`` seconds.
        """
        return self._client_ready.wait(timeout)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # -- lifecycle -------------------------------------------------------

    def start(self, host: str = "127.0.0.1", port: int = 8787) -> "LiveSession":
        """Start the server in a background daemon thread and return self.

        Pass ``port=0`` to bind an ephemeral port; the chosen port is available as
        ``session.port`` once this call returns.
        """
        if self._thread is not None:
            raise RuntimeError("LiveSession already started")
        self.host = host
        self.port = port
        self._thread = threading.Thread(target=self._run, name="pxviewer-live", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        return self

    def stop(self) -> None:
        """Stop the server and join the background thread. Idempotent."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass  # loop already stopped/closed (e.g. teardown interrupted then retried)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        self._thread = None
        self._loop = None

    def __enter__(self) -> "LiveSession":
        if self._thread is None:
            self.start(self.host, self.port)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- python -> scene -------------------------------------------------

    def push(self, coords: Any) -> int:
        """Broadcast a coordinate frame to all connected clients.

        ``coords`` is any (N,3) array-like in the topology's atom order. Returns the
        frame index. Thread-safe: may be called from any thread.
        """
        arr = _coords_to_f32(coords, self._n_atoms)
        index = self._frame_index
        self._frame_index += 1
        payload = struct.pack("<II", _TAG_FRAME, index) + arr.tobytes()
        self._last_frame = payload
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast, payload)
        return index

    def set_axis(self, visible: bool) -> None:
        """Broadcast a command to show or hide the camera XYZ axis helper.

        Thread-safe: may be called from any thread.
        """
        message = json.dumps({"type": "axis", "visible": bool(visible)})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def set_volume_color(self, ref: str, color: str) -> None:
        """Broadcast a command to change the color of a volume by reference.

        The ``ref`` is the volume reference used when the scene was built (e.g.
        ``volume-0`` or a custom :class:`Volume` ref). Thread-safe.
        """
        message = json.dumps({"type": "volume_color", "ref": str(ref), "color": str(color)})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def set_volume_opacity(self, ref: str, opacity: float) -> None:
        """Broadcast a command to change the opacity of a volume by reference.

        Thread-safe: may be called from any thread.
        """
        message = json.dumps({"type": "volume_opacity", "ref": str(ref), "opacity": float(opacity)})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    # -- selection (all index-based) -------------------------------------

    def select(self, atoms: Any, *, highlight: bool = True, focus: bool = True) -> Selection:
        """Show a set of atoms in the viewer: highlight and/or focus.

        ``atoms`` is a :class:`Selection` or anything coercible to one — an atom
        index, an iterable of indices, or a boolean mask of length N. Composes
        :meth:`highlight` and :meth:`focus`. Resolution is entirely Python-side, so
        this is fire-and-forget: no viewer round-trip. Returns the Selection.
        """
        sel = self._as_selection(atoms)
        if highlight:
            self.highlight(sel)
        if focus:
            self.focus(sel)
        return sel

    def select_by(
        self,
        *,
        indices: Optional[Iterable[int]] = None,
        ids: Optional[Iterable[int]] = None,
        mask: Optional[Any] = None,
    ) -> Selection:
        """Build a :class:`Selection` from positional indices, atom ids, or a mask.

        Pure Python; no viewer needed. Pass exactly one of ``indices``, ``ids``, or
        ``mask`` (a boolean array of length N). The resulting indices are stored
        sorted and de-duplicated.
        """
        if sum(x is not None for x in (indices, ids, mask)) != 1:
            raise ValueError("pass exactly one of indices=, ids=, or mask=")
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != (self._n_atoms,):
                raise ValueError(f"mask must have shape ({self._n_atoms},), got {tuple(m.shape)}")
            idx = [int(i) for i in np.nonzero(m)[0]]
        elif ids is not None:
            try:
                idx = [self._id_to_index[int(i)] for i in ids]
            except KeyError as exc:
                raise ValueError(f"unknown atom id {exc.args[0]}") from None
        else:
            idx = [int(i) for i in indices]  # type: ignore[union-attr]
        for i in idx:
            if not 0 <= i < self._n_atoms:
                raise ValueError(f"atom index {i} out of range [0, {self._n_atoms})")
        idx = sorted(set(idx))
        return Selection(idx, [self.atoms[i] for i in idx], self._n_atoms)

    # -- graphics primitives ---------------------------------------------

    def add_distance(self, a: Any, b: Any, *, label: bool = True, id: Optional[str] = None) -> Primitive:
        """Draw a distance line between two atom groups. See :meth:`add_angle`."""
        return self._add_primitive("distance", [a, b], {"label": bool(label)}, id)

    def add_angle(
        self, a: Any, b: Any, c: Any, *, opacity: float = 0.35, label: bool = True, id: Optional[str] = None
    ) -> Primitive:
        """Draw a pie-shaped angle wedge at the three atom groups ``a``-``b``-``c``.

        Each argument is a :class:`Selection` or anything coercible to one: an atom
        index, an iterable of indices, or a boolean mask of length N. A multi-atom
        group uses its centroid, so this also draws the angle between three groups.
        The wedge tracks the atoms as they move.

        ``opacity`` sets how translucent the wedge is; ``label`` toggles the degree
        text. Returns a :class:`Primitive` whose ``id`` removes it later and whose
        ``value``/``degrees`` is the current angle computed in Python.
        """
        return self._add_primitive("angle", [a, b, c], {"opacity": float(opacity), "label": bool(label)}, id)

    def add_dihedral(
        self, a: Any, b: Any, c: Any, d: Any, *, opacity: float = 0.35, label: bool = True, id: Optional[str] = None
    ) -> Primitive:
        """Draw a dihedral (torsion) between four atom groups. See :meth:`add_angle`."""
        return self._add_primitive(
            "dihedral", [a, b, c, d], {"opacity": float(opacity), "label": bool(label)}, id
        )

    def add_label(self, a: Any, text: str, *, id: Optional[str] = None) -> Primitive:
        """Draw a floating text label at an atom group's position."""
        return self._add_primitive("label", [a], {"text": str(text)}, id)

    def remove_primitive(self, primitive_id: str) -> None:
        """Remove a primitive (angle/distance/dihedral/label) by its id. Thread-safe."""
        self._primitives.pop(primitive_id, None)
        self._send_control({"type": "primitive", "action": "remove", "id": str(primitive_id)})

    def clear_primitives(self) -> None:
        """Remove all primitives from the viewer. Thread-safe."""
        self._primitives.clear()
        self._send_control({"type": "primitive", "action": "clear"})

    # -- representations -------------------------------------------------

    _REPR_ALIASES = {"sphere": "spacefill", "ribbon": "cartoon", "surface": "molecular-surface"}

    def set_representation(self, type: str, **kwargs: Any) -> str:
        """Replace all representations with a single one. See :meth:`add_representation`."""
        self._representations.clear()
        return self.add_representation(type, **kwargs)

    def add_representation(
        self,
        type: str,
        *,
        color: Optional[str] = None,
        color_value: Optional[str] = None,
        on: Any = None,
        opacity: Optional[float] = None,
        params: Optional[dict] = None,
        id: Optional[str] = None,
    ) -> str:
        """Add a representation of the structure (or a subset).

        ``type`` is a Mol* representation — ``'ball-and-stick'``, ``'spacefill'``
        (alias ``'sphere'``), ``'cartoon'`` (alias ``'ribbon'``),
        ``'molecular-surface'``, ``'gaussian-surface'``, ``'point'``, ``'line'``,
        ``'putty'``, ``'backbone'``, ``'ellipsoid'``, … ``color`` is a color-theme
        name (``'element-symbol'``, ``'chain-id'``, ``'secondary-structure'``,
        ``'residue-name'``, ``'hydrophobicity'``, ``'molecule-type'``, …); for a flat
        colour pass ``color_value`` (or a ``'#rrggbb'`` ``color``). ``on`` restricts
        it to a subset (a :class:`Selection` or coercible); omit for the whole
        structure. ``opacity`` sets transparency and ``params`` passes type-specific
        options. Returns the id; representations track streamed coordinates.
        """
        spec = self._make_repr_spec(type, color, color_value, on, opacity, params, id)
        self._representations[spec["id"]] = spec
        self._send_representations()
        return spec["id"]

    def remove_representation(self, representation_id: str) -> None:
        """Remove a representation by id. Thread-safe."""
        self._representations.pop(representation_id, None)
        self._send_representations()

    def clear_representations(self) -> None:
        """Remove all representations (restoring the default ball-and-stick). Thread-safe."""
        self._representations.clear()
        self._send_representations()

    def highlight(self, atoms: Any) -> Selection:
        """Show the selection overlay on the given atoms (:class:`Selection` or coercible)."""
        sel = self._as_selection(atoms)
        self._last_highlight_indices = sel.indices
        self._send_control({"type": "highlight", "atoms": _encode_index_set(sel.indices)})
        return sel

    def focus(self, atoms: Any) -> Selection:
        """Aim the viewer camera at the given atoms (:class:`Selection` or coercible)."""
        sel = self._as_selection(atoms)
        self._send_control({"type": "focus", "atoms": _encode_index_set(sel.indices)})
        return sel

    def clear_selection(self) -> None:
        """Clear any highlighted selection in the viewer. Thread-safe."""
        self._last_highlight_indices = []
        self._send_control({"type": "highlight", "atoms": _encode_index_set([])})

    # -- click interaction (scene -> python) -----------------------------

    _MEASURE_ARITY = {"distance": 2, "angle": 3, "dihedral": 4, "label": 1}

    def enable_mouse_selection(self, on_change: Optional[Callable[[Selection], None]] = None) -> None:
        """Select mode: let the user pick atoms by clicking (shift-click adds/removes).

        This is the primary mode — the picked set is reported back to Python: read
        the :attr:`mouse_selection` property, register ``on_change`` (or
        :meth:`on_selection`) for a callback, or block with
        :meth:`wait_for_selection`. Thread-safe.
        """
        if on_change is not None:
            self.on_selection(on_change)
        self._set_click_mode("select")

    def enable_measure_mode(
        self, kind: str, on_measure: Optional[Callable[["Primitive"], None]] = None
    ) -> None:
        """Measure mode: let the user click atoms to draw a measurement.

        A distinct mode from :meth:`enable_mouse_selection`. ``kind`` is
        ``"distance"`` (2 clicks), ``"angle"`` (3), ``"dihedral"`` (4), or
        ``"label"`` (1). Each completed set is drawn as a primitive that tracks the
        atoms; ``on_measure`` (or :meth:`on_measurement`) is called with the
        resulting :class:`Primitive`. Thread-safe.
        """
        if kind not in self._MEASURE_ARITY:
            raise ValueError(f"unknown measure kind {kind!r}; use one of {list(self._MEASURE_ARITY)}")
        if on_measure is not None:
            self.on_measurement(on_measure)
        self._set_click_mode(kind)

    def disable_mouse_selection(self) -> None:
        """Turn off any click interaction (select or measure mode). Thread-safe."""
        self._set_click_mode("off")

    def _set_click_mode(self, mode: str) -> None:
        self._click_mode = mode
        self._send_control({"type": "click-mode", "mode": mode})

    def on_selection(self, handler: Callable[[Selection], None]) -> None:
        """Register a callback invoked with a :class:`Selection` when the mouse selection changes."""
        self._selection_handlers.append(handler)

    def on_measurement(self, handler: Callable[["Primitive"], None]) -> None:
        """Register a callback invoked with a :class:`Primitive` when the user draws a measurement."""
        self._measure_handlers.append(handler)

    @property
    def mouse_selection(self) -> Selection:
        """The current mouse selection — the atoms the user has clicked in the viewer."""
        idx = list(self._mouse_selection_indices)
        return Selection(idx, [self.atoms[i] for i in idx], self._n_atoms)

    def wait_for_selection(self, timeout: Optional[float] = None) -> Optional[Selection]:
        """Block until the mouse selection next changes, then return it.

        Returns the new :class:`Selection`, or ``None`` if ``timeout`` elapses first.
        """
        self._selection_changed.clear()
        if self._selection_changed.wait(timeout):
            return self.mouse_selection
        return None

    # -- internals -------------------------------------------------------

    def _send_control(self, message: dict) -> None:
        """Broadcast a JSON control message to all clients (thread-safe)."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, json.dumps(message))

    def _next_primitive_id(self, kind: str) -> str:
        with self._lock:
            self._primitive_counter += 1
            return f"{kind}-{self._primitive_counter}"

    def _next_representation_id(self) -> str:
        with self._lock:
            self._representation_counter += 1
            return f"repr-{self._representation_counter}"

    def _make_repr_spec(self, type, color, color_value, on, opacity, params, id) -> dict:
        theme, value = color, None
        if color_value is not None:
            theme, value = "uniform", color_value
        elif color is not None and color.startswith("#"):
            theme, value = "uniform", color
        spec: dict = {
            "id": id if id is not None else self._next_representation_id(),
            "type": self._REPR_ALIASES.get(type, type),
        }
        if theme is not None:
            spec["color"] = theme
        if value is not None:
            spec["colorValue"] = value
        if on is not None:
            spec["on"] = _encode_index_set(self._as_selection(on).indices)
        if opacity is not None:
            spec["opacity"] = float(opacity)
        if params:
            spec["params"] = params
        return spec

    def _send_representations(self) -> None:
        self._send_control({"type": "representations", "reprs": list(self._representations.values())})

    def _as_selection(self, spec: Any) -> Selection:
        """Coerce a Selection / index / iterable-of-indices / boolean mask to a Selection."""
        if isinstance(spec, Selection):
            return spec
        if isinstance(spec, bool):  # bool is an int subclass; reject to avoid surprises
            raise TypeError("bool is not a valid atom specifier")
        if isinstance(spec, (int, np.integer)):
            return self.select_by(indices=[int(spec)])
        arr = np.asarray(spec)
        if arr.dtype == bool:
            return self.select_by(mask=arr)
        try:
            idx = [int(i) for i in arr.reshape(-1)]
        except (TypeError, ValueError):
            raise TypeError(f"cannot interpret {spec!r} as atom(s)") from None
        return self.select_by(indices=idx)

    def _add_primitive(self, kind: str, specs: List[Any], options: dict, primitive_id: Optional[str]) -> Primitive:
        selections = [self._as_selection(s) for s in specs]
        groups = [s.indices for s in selections]
        for group in groups:
            if not group:
                raise ValueError(f"each {kind} vertex needs at least one atom")
        pid = primitive_id if primitive_id is not None else self._next_primitive_id(kind)
        message = {
            "type": "primitive",
            "action": "add",
            "kind": kind,
            "id": pid,
            "groups": groups,
            "options": options,
        }
        self._primitives[pid] = message
        self._send_control(message)
        return Primitive(
            id=pid,
            kind=kind,
            value=self._measure(kind, groups),
            selections=selections,
            text=options.get("text"),
        )

    def _on_measure(self, event: dict) -> None:
        """Draw a primitive the user built by clicking atoms, and fire callbacks."""
        kind = event.get("kind")
        arity = self._MEASURE_ARITY.get(kind)
        raw = event.get("atoms") or []
        atoms = [i for i in raw if isinstance(i, int) and 0 <= i < self._n_atoms]
        if arity is None or len(atoms) != arity:
            return  # ignore malformed / incomplete requests
        options = self._default_measure_options(kind, atoms)
        try:
            primitive = self._add_primitive(kind, atoms, options, None)
        except Exception:  # pragma: no cover - defensive
            return
        for handler in self._measure_handlers:
            try:
                handler(primitive)
            except Exception:  # pragma: no cover - user callback errors
                pass

    def _default_measure_options(self, kind: str, atoms: List[int]) -> dict:
        if kind == "label":
            atom = self.atoms[atoms[0]]
            return {"text": f"{atom.name}{atom.resseq}"}
        options: dict = {"label": True}
        if kind in ("angle", "dihedral"):
            options["opacity"] = 0.35
        return options

    def _current_coords(self) -> np.ndarray:
        """The most recent coordinates (last streamed frame, else the topology's)."""
        if self._last_frame is not None:
            arr = np.frombuffer(self._last_frame[8:], dtype="<f4")
            return arr.reshape(self._n_atoms, 3).astype(float)
        return np.array([[a.x, a.y, a.z] for a in self.atoms], dtype=float)

    def _measure(self, kind: str, groups: List[List[int]]) -> Optional[float]:
        if kind == "label":
            return None
        coords = self._current_coords()
        pts = [coords[group].mean(axis=0) for group in groups]
        if kind == "distance":
            return float(np.linalg.norm(pts[1] - pts[0]))
        if kind == "angle":
            return _angle_deg(pts[0], pts[1], pts[2])
        if kind == "dihedral":
            return _dihedral_deg(pts[0], pts[1], pts[2], pts[3])
        return None

    def _run(self) -> None:
        try:
            import websockets  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "LiveSession needs the 'websockets' package. Install pxviewer[live]."
            ) from exc

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_until_complete(self._serve())
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(self._shutdown())
            loop.close()

    async def _serve(self) -> None:
        import websockets

        self._server = await websockets.serve(
            self._handler, self.host, self.port, process_request=self._process_request
        )
        # Resolve the actual bound port (matters when port=0 was requested).
        for sock in self._server.sockets or []:
            self.port = sock.getsockname()[1]
            break
        self._ready.set()

    async def _shutdown(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def _process_request(self, connection: Any, request: Any) -> Any:
        """Answer plain HTTP requests with a helpful page instead of failing the
        handshake (which otherwise logs a traceback for every stray browser hit)."""
        upgrade = request.headers.get("Upgrade", "") or ""
        if "websocket" in upgrade.lower():
            return None  # genuine WebSocket handshake — let it proceed
        body = (
            "pxviewer live WebSocket endpoint\n\n"
            "This address speaks the WebSocket protocol, not HTTP web pages, so it\n"
            "cannot be opened directly in a browser. Open the pxviewer frontend page\n"
            "instead and pass this address to it as ?ws=... (the demo command prints\n"
            "a ready-to-click http:// URL that does this for you).\n"
        )
        return connection.respond(HTTPStatus.UPGRADE_REQUIRED, body)

    async def _handler(self, websocket: Any) -> None:
        self._send_locks[websocket] = asyncio.Lock()
        self._clients.add(websocket)
        try:
            await self._locked_send(websocket, struct.pack("<I", _TAG_TOPOLOGY) + self._topology)
            if self._last_frame is not None:
                await self._locked_send(websocket, self._last_frame)
            if self._last_highlight_indices:
                # Bring a late-joining viewer up to the current highlight.
                await self._locked_send(
                    websocket,
                    json.dumps({"type": "highlight", "atoms": _encode_index_set(self._last_highlight_indices)}),
                )
            for message in list(self._primitives.values()):
                # Bring a late-joining viewer up to the active drawing primitives.
                await self._locked_send(websocket, json.dumps(message))
            if self._representations:
                await self._locked_send(
                    websocket, json.dumps({"type": "representations", "reprs": list(self._representations.values())})
                )
            if self._click_mode != "off":
                await self._locked_send(websocket, json.dumps({"type": "click-mode", "mode": self._click_mode}))
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    continue
                self._on_message(message)
        except Exception:  # pragma: no cover - client disconnects are routine
            pass
        finally:
            self._clients.discard(websocket)
            self._send_locks.pop(websocket, None)

    def _on_message(self, message: str) -> None:
        try:
            event = json.loads(message)
        except (ValueError, TypeError):
            return
        etype = event.get("type")
        if etype == "ready":
            self._client_ready.set()
        elif etype == "pick":
            info = None if event.get("empty") else event.get("atom")
            for handler in self._pick_handlers:
                try:
                    handler(info)
                except Exception:  # pragma: no cover - user callback errors
                    pass
        elif etype == "mouse-selection":
            raw = event.get("indices") or []
            indices = sorted({i for i in raw if isinstance(i, int) and 0 <= i < self._n_atoms})
            self._mouse_selection_indices = indices
            selection = Selection(indices, [self.atoms[i] for i in indices], self._n_atoms)
            self._selection_changed.set()
            for handler in self._selection_handlers:
                try:
                    handler(selection)
                except Exception:  # pragma: no cover - user callback errors
                    pass
        elif etype == "measure":
            self._on_measure(event)

    def _broadcast(self, payload: bytes) -> None:
        for websocket in list(self._clients):
            # Fire-and-forget: scheduling the coroutine keeps push() non-blocking.
            asyncio.ensure_future(self._safe_send(websocket, payload))

    def _broadcast_text(self, message: str) -> None:
        for websocket in list(self._clients):
            asyncio.ensure_future(self._safe_send_text(websocket, message))

    async def _locked_send(self, websocket: Any, data: Any) -> None:
        """Send holding the per-connection lock so sends never overlap or reorder."""
        lock = self._send_locks.get(websocket)
        if lock is None:
            await websocket.send(data)
        else:
            async with lock:
                await websocket.send(data)

    async def _safe_send(self, websocket: Any, payload: bytes) -> None:
        try:
            await self._locked_send(websocket, payload)
        except Exception:  # pragma: no cover - drop on closed sockets
            self._clients.discard(websocket)
            self._send_locks.pop(websocket, None)

    async def _safe_send_text(self, websocket: Any, message: str) -> None:
        try:
            await self._locked_send(websocket, message)
        except Exception:  # pragma: no cover - drop on closed sockets
            self._clients.discard(websocket)
            self._send_locks.pop(websocket, None)


def oscillating_frames(
    atoms: Sequence[Atom],
    *,
    steps: int = 240,
    amplitude: float = 3.0,
    wavelength: float = 4.0,
) -> Iterable[np.ndarray]:
    """Yield a looping demo trajectory: a travelling sine wave along +y.

    Topology is unchanged; only y is displaced per frame. Useful for exercising
    the live path without a real simulation.
    """
    base = np.array([[a.x, a.y, a.z] for a in atoms], dtype="<f4")
    n = len(atoms)
    step = 0
    while True:
        phase = 2.0 * np.pi * (step / steps)
        offsets = amplitude * np.sin(np.arange(n) / wavelength + phase)
        frame = base.copy()
        frame[:, 1] = base[:, 1] + offsets
        yield frame
        step = (step + 1) % steps
