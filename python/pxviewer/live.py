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
  - {"type": "volume-iso-changed", "ref": str, "value": float}  wheel contouring
  - {"type": "tug", "action": str, "atom": int, "target": [x,y,z]}  Shift-drag of an atom
  - {"type": "screenshot-result", "reqId": int, "dataUri": str}  rendered viewport

All atoms are addressed by *positional index* (row in the topology's _atom_site
table). Selections are resolved to indices entirely on the Python side; the wire
never carries a query language. An <index-set> is either {"list": [int,...]} or a
run-length {"runs": [[start,end],...]} (see `_encode_index_set`), and a highlight
with an empty set clears the selection.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import struct
import threading
from http import HTTPStatus
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, get_args

import numpy as np
from molviewspec.nodes import ColorNamesT, ComponentExpression, RepresentationTypeT

from .data import AtomArrays, encode_bcif_arrays
from .cctbx_io import ModelData

__all__ = ["LiveSession", "Selection", "Primitive", "ComponentExpression", "ATOM_IDENTITY_CONTRACT"]

# Representation vocabulary is MVS's RepresentationTypeT (structure subset), mapped
# to Mol*'s internal representation names. Colours are MVS ColorNamesT (uniform)
# or a Mol* colour theme layered on top (e.g. 'element-symbol').
_SVG_COLOR_NAMES = set(get_args(ColorNamesT))
_STRUCTURE_REPR_TYPES = set(get_args(RepresentationTypeT)) - {"isosurface"}  # isosurface is for volumes
_REPR_ALIASES = {
    "sphere": "spacefill",
    "ribbon": "cartoon",
    "ball-and-stick": "ball_and_stick",
    "molecular-surface": "surface",
    "gaussian-surface": "surface",
}
_MVS_TO_MOLSTAR_REPR = {
    "ball_and_stick": "ball-and-stick",
    "spacefill": "spacefill",
    "cartoon": "cartoon",
    "surface": "molecular-surface",
    "carbohydrate": "carbohydrate",
}


# Cap on an inbound message. Everything the viewer sends back is small except a
# screenshot, which scales with the window: generous enough for any real one, but still
# a bound rather than None.
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024

_TAG_TOPOLOGY = 0
_TAG_FRAME = 1
_TAG_ATTRIBUTE = 2  # per-atom scalar values for colour-by-attribute
_TAG_DOTS = 3       # probe2 contact-dot surface (positions + spikes + colours)

# probe2 dot overlay channels — independently toggleable (full surface vs clashes).
PROBE_CONTACTS = 0
PROBE_CLASHES = 1

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


# The interaction kinds Mol*'s custom-interactions extension understands.
_INTERACTION_KINDS = frozenset({
    "unknown",
    "ionic",
    "pi-stacking",
    "cation-pi",
    "halogen-bond",
    "hydrogen-bond",
    "weak-hydrogen-bond",
    "hydrophobic",
    "metal-coordination",
    "water-bridge",
    "covalent",
})

# Friendly spellings people actually type -> the canonical kind.
_INTERACTION_ALIASES = {
    "h-bond": "hydrogen-bond",
    "hbond": "hydrogen-bond",
    "hydrogen bond": "hydrogen-bond",
    "weak-h-bond": "weak-hydrogen-bond",
    "weak hydrogen bond": "weak-hydrogen-bond",
    "salt-bridge": "ionic",
    "saltbridge": "ionic",
    "salt bridge": "ionic",
    "ionic-bond": "ionic",
    "pi-pi": "pi-stacking",
    "pi stacking": "pi-stacking",
    "pi-cation": "cation-pi",
    "cation pi": "cation-pi",
    "halogen": "halogen-bond",
    "metal": "metal-coordination",
    "metal coordination": "metal-coordination",
    "water bridge": "water-bridge",
}


def _normalize_kind(kind: Any) -> str:
    """Map a user-supplied interaction kind to a canonical Mol* kind."""
    key = str(kind).strip().lower().replace("_", "-")
    key = _INTERACTION_ALIASES.get(key, key)
    if key not in _INTERACTION_KINDS:
        raise ValueError(
            f"unknown interaction kind {kind!r}. Known kinds: {', '.join(sorted(_INTERACTION_KINDS))}"
        )
    return key


def _normalize_interactions(interactions: Any, n_atoms: int) -> List[dict]:
    """Coerce a user interaction table into a flat list of contact dicts.

    Accepts a ``{kind: pairs}`` mapping, an iterable of ``(kind, a, b)`` tuples,
    or an iterable of ``{kind, a, b, description?}`` dicts. Validates every atom
    index against ``n_atoms`` and every kind against the known set.
    """
    def _atom(idx: Any) -> int:
        i = int(idx)
        if not 0 <= i < n_atoms:
            raise ValueError(f"atom index {i} out of range [0, {n_atoms})")
        return i

    def _contact(kind: Any, a: Any, b: Any, description: Any = None) -> dict:
        c = {"kind": _normalize_kind(kind), "a": _atom(a), "b": _atom(b)}
        if description is not None:
            c["description"] = str(description)
        return c

    if interactions is None:
        return []

    contacts: List[dict] = []
    if isinstance(interactions, dict):
        for kind, pairs in interactions.items():
            for pair in pairs:
                a, b, *rest = pair
                contacts.append(_contact(kind, a, b, rest[0] if rest else None))
        return contacts

    for item in interactions:
        if isinstance(item, dict):
            contacts.append(
                _contact(item["kind"], item["a"], item["b"], item.get("description"))
            )
        else:
            kind, a, b, *rest = item
            contacts.append(_contact(kind, a, b, rest[0] if rest else None))
    return contacts


_MISSING_TOKENS = {"", ".", "nan", "na", "n/a", "none", "null"}


def _read_value_column(path: Any) -> np.ndarray:
    """Read a one-value-per-line text file into a float array.

    Blank lines and ``#`` comments are ignored; ``nan``/``.``/``na`` (any case) become
    NaN. Raises ``ValueError`` on a line that isn't a single number.
    """
    values: List[float] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower() in _MISSING_TOKENS:
                values.append(float("nan"))
                continue
            try:
                values.append(float(s))
            except ValueError:
                raise ValueError(f"{path}:{lineno}: expected one number per line, got {s!r}") from None
    return np.array(values, dtype=float)


def _normalize_pairs(pairs: Any, n_atoms: int) -> List[dict]:
    """Coerce clash pairs (tuples or dicts) into validated ``{a, b}`` dicts."""
    def _atom(idx: Any) -> int:
        i = int(idx)
        if not 0 <= i < n_atoms:
            raise ValueError(f"atom index {i} out of range [0, {n_atoms})")
        return i

    out: List[dict] = []
    seen: set = set()
    for pair in pairs or []:
        if isinstance(pair, dict):
            a, b = _atom(pair["a"]), _atom(pair["b"])
        else:
            a, b, *_ = pair
            a, b = _atom(a), _atom(b)
        if a == b:
            raise ValueError(f"a clash needs two distinct atoms, got ({a}, {a})")
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append({"a": key[0], "b": key[1]})
    return out


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


class Selection:
    """A set of atoms addressed by positional index (i_seq).

    ``indices`` are positional (0-based) rows into the topology's _atom_site table —
    the same stable identity the whole live protocol uses (see
    ``ATOM_IDENTITY_CONTRACT``); kept sorted and de-duplicated. Per-atom fields are
    exposed as **columnar lists** read from the session's columns on demand
    (``ids``, ``names``, ``resnames``, ``chains``, ``resseqs``, ``elements``) — no
    per-atom objects are built. Construct via :meth:`LiveSession.select_by`.
    """

    def __init__(self, indices: Iterable[int], n_total: int, arrays: "AtomArrays"):
        self.indices: List[int] = list(indices)
        self.n_total = n_total
        self._arrays = arrays  # shared reference to the session's columns (not copied)

    @property
    def ids(self) -> List[int]:
        """The ``_atom_site.id`` of each matched atom."""
        return [int(self._arrays.id[i]) for i in self.indices]

    @property
    def names(self) -> List[str]:
        """The atom name of each matched atom."""
        return [self._arrays.name[i] for i in self.indices]

    @property
    def resnames(self) -> List[str]:
        """The residue name of each matched atom."""
        return [self._arrays.resname[i] for i in self.indices]

    @property
    def chains(self) -> List[str]:
        """The chain id of each matched atom."""
        return [self._arrays.chain[i] for i in self.indices]

    @property
    def resseqs(self) -> List[int]:
        """The residue number of each matched atom."""
        return [int(self._arrays.resseq[i]) for i in self.indices]

    @property
    def elements(self) -> List[str]:
        """The element symbol of each matched atom."""
        return [self._arrays.element[i] for i in self.indices]

    @property
    def mask(self) -> np.ndarray:
        """A boolean array of length ``n_total``, True at matched positions."""
        m = np.zeros(self.n_total, dtype=bool)
        if self.indices:
            m[self.indices] = True
        return m

    def to_component_expression(self) -> List[ComponentExpression]:
        """Express this selection as MVS ``ComponentExpression`` objects (one per atom)."""
        return [ComponentExpression(atom_index=i) for i in self.indices]

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
        session = LiveSession.from_model_file("model.pdb")   # or from_cctbx_model / from_sites
        session.on_pick(lambda info: print("picked", info))
        session.start()                       # background thread, ws://127.0.0.1:8787
        for frame in trajectory:              # frame: (N,3) array-like
            session.push(frame)
    """

    def __init__(self, data: ModelData, *, topology: Optional[bytes] = None):
        # The session's atom source is a columnar :class:`ModelData` (numpy columns
        # plus the native cctbx model, carrying its own polymer/secondary-structure
        # flags). Build sessions via the classmethods — from_model_file /
        # from_cctbx_model / from_sites — which always route through cctbx; this
        # low-level constructor just wires up a prepared ModelData.
        if not isinstance(data, ModelData):
            raise TypeError(
                "LiveSession(data) takes a ModelData; build a session via "
                "LiveSession.from_model_file / from_cctbx_model / from_sites"
            )
        self._data = data
        if self._data.n_atoms == 0:
            raise ValueError("LiveSession requires at least one atom")
        self._topology: bytes = topology if topology is not None else encode_bcif_arrays(
            data.arrays, polymer=data.polymer, secondary_structure=data.secondary_structure
        )
        self._n_atoms = self._data.n_atoms

        self._frame_index = 0
        self._last_frame: Optional[bytes] = None
        self._pick_handlers: List[Callable[[Optional[dict]], None]] = []

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Any = None
        self._clients: set = set()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._started_or_error = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._client_ready = threading.Event()

        self._lock = threading.Lock()  # guards the primitive counter

        # Atom-id -> positional index, for building selections by id. Built from the
        # id column (no per-atom objects). Identity is otherwise i_seq (positional).
        self._id_to_index = {int(v): i for i, v in enumerate(self._data.arrays.id)}
        # Active highlight (positional indices), replayed to clients that connect later.
        self._last_highlight_indices: List[int] = []
        # Active highlight (PyMOL expression), replayed to clients that connect later.
        self._last_highlight_expression: Optional[str] = None
        # PyMOL selections are evaluated in the browser and echoed back; each
        # request waits on an Event keyed by a monotonic id.
        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        self._req_counter = 0
        # Active drawing primitives (id -> the "add" message), replayed to late clients.
        self._primitives: dict = {}
        self._primitive_counter = 0
        # Representations (id -> spec), sent declaratively; replayed to late clients.
        self._representations: dict = {}
        self._representation_counter = 0
        # Named per-atom scalar attributes (name -> float array of length N), for
        # colour-by-attribute. Custom _atom_site columns from the model's mmCIF are
        # seeded here; b-factor / occupancy are always available from the topology.
        self._attributes: dict = {k: np.asarray(v, dtype=float) for k, v in self._data.attributes.items()}
        # Binary payloads (wire key -> bytes) for attributes referenced by a current
        # representation; sent as binary (not JSON) and replayed to late clients.
        self._attribute_payloads: dict = {}
        self._attribute_counter = 0
        # Explicit, Python-supplied interaction contacts (list of dicts); replayed.
        self._interactions_contacts: List[dict] = []
        # Whether Mol*'s *computed* interaction overlay is on; replayed to late clients.
        self._computed_interactions_visible = False
        # probe2 dot overlays, keyed by channel (contacts / clashes) so they toggle
        # independently; each value is a binary payload replayed to late clients.
        self._probe_dots_payloads: Dict[int, bytes] = {}
        # Explicit clash-marker pairs drawn by set_clashes (list of {a, b} dicts);
        # replayed to late clients. Used by the synthetic streaming demo.
        self._clashes: List[dict] = []

        # Click interaction mode: 'off' | 'select' | 'distance' | 'angle' | 'dihedral'
        # | 'label'. In 'select' the user builds a selection streamed back here; in a
        # measure mode they click N atoms and the primitive is drawn.
        self._click_mode = "off"
        self._mouse_selection_indices: List[int] = []
        self._volume_iso_handlers: List[Callable[[str, float], None]] = []
        self._tug_handlers: List[Callable[[str, int, Optional[list]], None]] = []
        self._marker_handlers: List[Callable[[list, Optional[int]], None]] = []
        # Which volume the scroll wheel contours. Not part of the MVSJ scene (unlike a
        # volume's style/colour/level, which a rebuild restores), so it has to be
        # replayed to late clients or the wheel goes dead after every scene reload.
        self._volume_scroll_target: Optional[str] = None
        # Clips, keyed by ref (None = this session's model). Replayed to late clients for
        # the same reason: a clip is worked out from the camera and re-aimed as it moves,
        # so it cannot be baked into the scene — a viewport reload would drop it.
        self._clips: dict = {}
        # Volumes hidden from the shared scene, by ref. Replayed to late clients: hiding is
        # a state-cell toggle over the MVSJ scene, which a reload rebuilds fresh (every
        # volume visible), so a hidden one has to be re-hidden when the new page connects.
        self._hidden_volumes: set = set()
        self._selection_handlers: List[Callable[[Selection], None]] = []
        self._measure_handlers: List[Callable[[Primitive], None]] = []
        self._selection_changed = threading.Event()
        # Per-connection send lock: websockets forbids concurrent send() on one
        # connection, so serialize sends (this also preserves message order).
        self._send_locks: dict = {}

        self.host = "127.0.0.1"
        self.port = 8787

    # -- construction (everything routes through cctbx) ------------------

    @classmethod
    def from_cctbx_model(cls, model: Any) -> "LiveSession":
        """Build a session from an ``mmtbx.model.manager`` (cctbx model object).

        Reads coordinates, labels and secondary structure from the model's
        ``pdb_hierarchy``; a multi-MODEL ensemble is reduced to model 1. The native
        model is retained, so selection uses cctbx's own machinery and polymer
        models render as cartoon.
        """
        return cls(ModelData.from_model(model))

    @classmethod
    def from_model_file(cls, path: Any) -> "LiveSession":
        """Read a model file (PDB/mmCIF) with cctbx and build a session from it."""
        from . import cctbx_io

        return cls(cctbx_io.load_model(path))

    @classmethod
    def from_sites(cls, sites: Any, **labels: Any) -> "LiveSession":
        """Build a session from raw coordinates (+ optional label columns), via cctbx.

        The synthetic-data counterpart to :meth:`from_model_file`: ``sites`` is an
        ``(N, 3)`` array and ``labels`` are the ``model_from_sites`` overrides
        (``elements``, ``names``, ``chains``, ``resseqs``, ``resnames``); it builds
        a real cctbx model so identity and selection go through cctbx like any other
        session. Defaults give a chain of carbons, one per residue.
        """
        from . import cctbx_io

        return cls.from_cctbx_model(cctbx_io.model_from_sites(sites, **labels))

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
        self._started_or_error.wait(timeout=10)
        if self._start_error is not None:
            raise RuntimeError(
                f"LiveSession server failed to start on {host}:{port}"
            ) from self._start_error
        if not self._ready.is_set():
            raise RuntimeError(
                f"LiveSession server failed to start on {host}:{port} (timeout)"
            )
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

    def reset_view(self) -> None:
        """Reframe the camera to fit the whole scene at its default orientation.
        Thread-safe: may be called from any thread."""
        message = json.dumps({"type": "reset-view"})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def set_interactions(self, interactions: Any) -> List[dict]:
        """Draw an explicit set of non-covalent interactions between atom pairs.

        You supply the contacts; nothing is inferred. Each is a typed pair of
        positional atom indices (the same 0-based identity the rest of the live
        protocol uses). They are drawn as Mol*'s non-covalent interaction notation
        (dashed cylinders, coloured by kind) and — because they reference atoms,
        not fixed positions — their endpoints track streamed coordinates.

        ``interactions`` may be:

        - a mapping of ``kind -> pairs``, e.g.::

              session.set_interactions({
                  "hydrogen-bond": [(0, 1), (5, 6)],
                  "salt-bridge":   [(3, 8)],
              })

        - an iterable of ``(kind, a, b)`` tuples, e.g. ``[("h-bond", 0, 1), ...]``
        - an iterable of dicts ``{"kind", "a", "b", "description"?}``

        Kinds are ``hydrogen-bond``, ``weak-hydrogen-bond``, ``ionic``,
        ``hydrophobic``, ``pi-stacking``, ``cation-pi``, ``halogen-bond``,
        ``metal-coordination``, ``water-bridge``, ``covalent`` and ``unknown``;
        common aliases like ``h-bond`` and ``salt-bridge`` are accepted.

        Returns the normalised contacts. Raises ``ValueError`` for an unknown kind
        or an out-of-range atom index (fail loud, per the identity contract).
        Thread-safe; the set is replayed to viewers that connect later. Pass an
        empty collection (or call :meth:`clear_interactions`) to remove them.
        """
        contacts = _normalize_interactions(interactions, self._n_atoms)
        self._interactions_contacts = contacts
        if contacts:
            message = json.dumps({"type": "interactions", "action": "set", "contacts": contacts})
        else:
            message = json.dumps({"type": "interactions", "action": "clear"})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)
        return contacts

    def clear_interactions(self) -> None:
        """Remove all explicit interactions. See :meth:`set_interactions`."""
        self._interactions_contacts = []
        message = json.dumps({"type": "interactions", "action": "clear"})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def set_computed_interactions(self, visible: bool) -> None:
        """Show or hide Mol*'s *computed* non-covalent interaction overlay.

        Unlike :meth:`set_interactions`, the contacts are inferred by Mol* from
        the geometry, on every structure in the scene (file/MVSJ or live). Useful
        when Python doesn't have an explicit contact table — e.g. a structure
        loaded and parsed entirely in the browser. Thread-safe; replayed to late
        viewers.
        """
        visible = bool(visible)
        self._computed_interactions_visible = visible
        message = json.dumps({"type": "computed-interactions", "visible": visible})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def show_computed_interactions(self) -> None:
        """Show the computed overlay. See :meth:`set_computed_interactions`."""
        self.set_computed_interactions(True)

    def hide_computed_interactions(self) -> None:
        """Hide the computed overlay. See :meth:`set_computed_interactions`."""
        self.set_computed_interactions(False)

    def set_clashes(self, pairs: Any) -> List[tuple]:
        """Draw red clash markers between the given atom-index pairs.

        A low-level drawing primitive: each pair becomes a distinct red marker
        (visually separate from the interaction notation). ``pairs`` is an iterable
        of ``(i, j)`` tuples or ``{"a", "b"}`` dicts; indices are positional and
        validated against the atom count, self-pairs are rejected, and duplicates are
        collapsed. Because the markers reference atoms, they track streamed
        coordinates. Returns the normalised ``(a, b)`` pairs. Thread-safe; replayed to
        late viewers. Pass an empty iterable (or call :meth:`clear_clashes`) to remove
        them. (Rigorous clash analysis is done with hydrogens via the probe2 dot
        overlay — see :meth:`show_probe_dots`.)
        """
        clashes = _normalize_pairs(pairs, self._n_atoms)
        self._clashes = clashes
        if clashes:
            message = json.dumps({"type": "clashes", "action": "set", "pairs": clashes})
        else:
            message = json.dumps({"type": "clashes", "action": "clear"})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)
        return [(c["a"], c["b"]) for c in clashes]

    def clear_clashes(self) -> None:
        """Remove all clash markers. See :meth:`set_clashes`."""
        self._clashes = []
        message = json.dumps({"type": "clashes", "action": "clear"})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def show_probe_dots(self, dots: Any, *, channel: int = PROBE_CONTACTS) -> int:
        """Draw a probe2 dot overlay: ``dots`` is ``[(loc, spike, rgb), …]``.

        ``channel`` selects an independently toggleable overlay (:data:`PROBE_CONTACTS`
        for the full surface, :data:`PROBE_CLASHES` for a clash-only overlay). Sent as
        one binary payload (positions + spike tips + colours) and drawn as a Mol*
        point cloud plus clash spikes. Thread-safe; replayed to late viewers. Returns
        the number of dots. Pass an empty list (or :meth:`clear_probe_dots`) to remove
        that channel.
        """
        from .probe import encode_dots

        dots = list(dots)
        if not dots:
            self.clear_probe_dots(channel=channel)
            return 0
        payload = struct.pack("<II", _TAG_DOTS, channel) + encode_dots(dots)
        self._probe_dots_payloads[channel] = payload
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast, payload)
        return len(dots)

    def clear_probe_dots(self, *, channel: Optional[int] = None) -> None:
        """Remove a probe dot overlay (a single ``channel``, or all). See
        :meth:`show_probe_dots`."""
        if channel is None:
            self._probe_dots_payloads.clear()
        else:
            self._probe_dots_payloads.pop(channel, None)
        message = json.dumps({"type": "dots", "action": "clear", "channel": channel})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def show_markup(self, channel: int, primitives: Any) -> int:
        """Draw MolProbity validation markup on ``channel``: ``primitives`` is a list
        of kinemage primitives (see :mod:`pxviewer.kinemage`) — vectors/dots/balls/
        triangles — rendered as a Mesh. Pass an empty list to clear. Thread-safe."""
        primitives = list(primitives)
        message = json.dumps({"type": "markup", "channel": int(channel), "primitives": primitives})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)
        return len(primitives)

    def clear_markup(self, channel: int) -> None:
        """Remove a validation markup overlay. See :meth:`show_markup`."""
        self.show_markup(channel, [])

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

    def set_volume_visible(self, ref: str, visible: bool) -> None:
        """Broadcast a command to show or hide a volume by reference.

        A state-cell toggle over the shared scene, not a rebuild: hiding a map leaves
        every other object on screen untouched. The hidden set is remembered and replayed
        to late clients, so a scene reload (which rebuilds every volume visible) re-hides
        what should stay hidden. Thread-safe: may be called from any thread.
        """
        key = str(ref)
        if visible:
            self._hidden_volumes.discard(key)
        else:
            self._hidden_volumes.add(key)
        message = json.dumps({"type": "volume_visible", "ref": key, "visible": bool(visible)})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def set_volume_style(self, ref: str, style: str) -> None:
        """Broadcast a command to change the isosurface style of a volume by reference.

        ``style`` is one of ``'surface'``, ``'wireframe'`` or ``'mesh'``.
        Thread-safe: may be called from any thread.
        """
        message = json.dumps({"type": "volume_style", "ref": str(ref), "style": str(style)})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def set_volume_iso(self, ref: str, value: float) -> None:
        """Broadcast a command to set a volume's contour level, in sigma.

        The level is relative (sigma), so it means the same thing for any map — the
        viewer never has to know a map's absolute scale to contour it sensibly.
        Thread-safe: may be called from any thread.
        """
        message = json.dumps({"type": "volume_iso", "ref": str(ref), "value": float(value)})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def screenshot(self, *, timeout: float = 20.0) -> Optional[bytes]:
        """Render the viewport and return the PNG bytes (None if nobody answers).

        The picture is taken in the browser, which is the only place the scene exists,
        and comes back over the wire — so this works for a remote viewer as much as the
        desktop one. Blocks until the image arrives; call it off the GUI thread.
        """
        req_id = self._next_req()
        slot: dict = {"event": threading.Event(), "uri": None, "error": None}
        with self._pending_lock:
            self._pending[req_id] = slot
        self._send_control({"type": "screenshot", "reqId": req_id})
        answered = slot["event"].wait(timeout)
        with self._pending_lock:
            self._pending.pop(req_id, None)
        if not answered:
            return None
        if slot["error"]:
            raise RuntimeError(f"screenshot failed: {slot['error']}")
        uri = slot["uri"] or ""
        if "," not in uri:
            return None
        return base64.b64decode(uri.split(",", 1)[1])

    def set_clip(
        self,
        front: float,
        back: float,
        *,
        radius: Optional[float] = None,
        ref: Optional[str] = None,
    ) -> None:
        """Clip a representation: a front/rear slab, a radius around the view, or both.

        ``front`` and ``back`` run 0..1 across the scene's depth: ``(0, 1)`` clips
        nothing, and when the two meet everything is clipped and the object disappears.
        ``radius`` (Angstrom) draws only what is near the view centre; None draws it all.
        ``ref`` names a volume; without one this session's own model is clipped.

        Both follow the camera, and both are per representation deliberately — it is
        what lets density be cut open, or thinned out, while the model inside stays
        whole. Thread-safe.
        """
        key = None if ref is None else str(ref)
        clip = {
            "type": "clip", "ref": key,
            "front": float(front), "back": float(back),
            "radius": None if radius is None else float(radius),
        }
        if clip["front"] <= 0 and clip["back"] >= 1 and clip["radius"] is None:
            self._clips.pop(key, None)  # nothing to restore
        else:
            self._clips[key] = clip
        message = json.dumps(clip)
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def on_tug(self, handler: Callable[[str, int, Optional[list]], None]) -> None:
        """Register a callback for atom drags: ``(action, atom, target)``.

        ``action`` is 'begin', 'move' or 'end'; ``target`` is the pointer in space for
        'move' and None otherwise. Dragging is Shift + left-drag in the viewport, gated
        by the modifier rather than a mode — the browser says which atom and where the
        pointer is, and what the model does about it is cctbx's business.
        """
        self._tug_handlers.append(handler)

    def on_marker(self, handler: Callable[[list, Optional[int]], None]) -> None:
        """Register a callback for a marker placed in the viewport: ``handler(position,
        atom)``, where ``position`` is a world-space ``[x, y, z]`` and ``atom`` is the
        picked atom index if the click landed on one, else ``None``. Armed one-shot with
        :meth:`set_marker_mode`; the click reports a point rather than rotating.
        """
        self._marker_handlers.append(handler)

    def set_marker_mode(self, on: bool) -> None:
        """Arm (or disarm) 'place a marker' mode in the viewport. While on, the next
        click reports a 3D point back via :meth:`on_marker` instead of rotating, then the
        viewer disarms itself. The point snaps to the atom under the cursor, or falls to
        the view-plane depth for a click in empty space. Thread-safe.
        """
        message = json.dumps({"type": "marker-mode", "on": bool(on)})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def set_volume_scroll_target(self, ref: Optional[str]) -> None:
        """Name the volume the viewport's scroll wheel contours (None = nothing).

        The wheel is a shortcut for the contour control the user is looking at, so the
        target is whichever volume the controls are pointing at rather than anything the
        viewport decides for itself. Thread-safe: may be called from any thread.
        """
        self._volume_scroll_target = None if ref is None else str(ref)
        message = json.dumps(
            {"type": "volume_scroll_target", "ref": self._volume_scroll_target})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    def on_volume_iso(self, handler: Callable[[str, float], None]) -> None:
        """Register a callback for contour levels changed in the viewport (the wheel).

        Called with ``(ref, value)``. The viewer applies the change itself; this is how
        the controls hear about it, so the slider keeps telling the truth.
        """
        self._volume_iso_handlers.append(handler)

    def set_volume_position(self, ref: str, position: Any) -> None:
        """Broadcast a command to translate a volume by reference.

        ``position`` is a 3-element sequence of Angstrom offsets.
        Thread-safe: may be called from any thread.
        """
        x, y, z = position
        message = json.dumps({"type": "volume_position", "ref": str(ref), "position": [float(x), float(y), float(z)]})
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, message)

    # -- selection (python -> scene -> python) ---------------------------

    def select(
        self,
        atoms_or_expression: Any,
        *,
        highlight: bool = True,
        focus: bool = True,
        timeout: float = 5.0,
    ) -> Optional[Selection]:
        """Show a set of atoms in the viewer: highlight and/or focus.

        If ``atoms_or_expression`` is a string, it is treated as a PyMOL selection
        expression, evaluated by the browser, and the matched atoms are returned.
        Otherwise it is coerced to a :class:`Selection` — an atom index, an
        iterable of indices, or a boolean mask of length N. In the latter case
        resolution is entirely Python-side and the selection is returned.
        """
        if isinstance(atoms_or_expression, str):
            return self._selection_request(atoms_or_expression, highlight=highlight, focus=focus, timeout=timeout)
        sel = self._as_selection(atoms_or_expression)
        if highlight:
            self.highlight(sel)
        if focus:
            self.focus(sel)
        return sel

    @property
    def model(self) -> Any:
        """The native cctbx ``mmtbx.model.manager`` backing this session, or ``None``.

        Present when built via :meth:`from_model_file` / :meth:`from_cctbx_model`.
        """
        return self._data.model

    def diff(self) -> Optional[str]:
        """Check the cached columns still match the cctbx model; see ``ModelData.diff``.

        Returns ``None`` when in sync (or no model is attached), else a drift message.
        """
        return self._data.diff()

    def _make_selection(self, indices: Iterable[int]) -> Selection:
        """Build a :class:`Selection` from indices, validating and de-duplicating.

        The Selection holds only the indices plus a reference to the session's
        columns — per-atom fields are read columnarly on access, nothing is
        materialised per atom.
        """
        idx = sorted({int(i) for i in indices})
        for i in idx:
            if not 0 <= i < self._n_atoms:
                raise ValueError(f"atom index {i} out of range [0, {self._n_atoms})")
        return Selection(idx, self._n_atoms, self._data.arrays)

    def select_by(
        self,
        *,
        indices: Optional[Iterable[int]] = None,
        ids: Optional[Iterable[int]] = None,
        mask: Optional[Any] = None,
        selection: Optional[str] = None,
    ) -> Selection:
        """Build a :class:`Selection` from positional indices, atom ids, a mask, or a
        cctbx atom-selection string.

        Pass exactly one of ``indices``, ``ids``, ``mask`` (a boolean array of length
        N), or ``selection`` (a cctbx/Phenix selection string, e.g.
        ``"chain A and resseq 5:14 and name CA"``, resolved by cctbx's own machinery —
        requires a model-backed session). Pure Python; no viewer needed.
        """
        if sum(x is not None for x in (indices, ids, mask, selection)) != 1:
            raise ValueError("pass exactly one of indices=, ids=, mask=, or selection=")
        if selection is not None:
            idx = [int(i) for i in self._data.selection_indices(selection)]
        elif mask is not None:
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
        return self._make_selection(idx)

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

        ``type`` is an MVS representation — ``'ball_and_stick'``, ``'spacefill'``
        (alias ``'sphere'``), ``'cartoon'`` (alias ``'ribbon'``), ``'surface'``, or
        ``'carbohydrate'``. ``color`` is either a **uniform** colour (an SVG name
        such as ``'orange'``, or ``'#ff8800'``) or a Mol* **colour theme** name
        (``'element-symbol'``, ``'chain-id'``, ``'secondary-structure'``,
        ``'residue-name'``, ``'hydrophobicity'``, …); ``color_value`` also forces a
        uniform colour. ``on`` restricts it to a subset (a :class:`Selection`, an MVS
        ``ComponentExpression``, or anything coercible); omit for the whole
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

    # -- colour by per-atom attribute ------------------------------------

    def set_attribute(self, name: str, values: Any) -> None:
        """Register a named per-atom scalar attribute (a length-N array).

        Once registered it can be used with :meth:`color_by`. Values are floats
        (use ``nan`` for "missing" — those atoms take the theme's missing colour).
        ``bfactor`` and ``occupancy`` are always available from the topology and
        need not be registered.
        """
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != self._n_atoms:
            raise ValueError(
                f"attribute {name!r} has {arr.shape[0]} values but the structure has {self._n_atoms} atoms"
            )
        self._attributes[str(name)] = arr

    def attributes(self) -> List[str]:
        """The attributes available to :meth:`color_by` (registered + built-in)."""
        names = list(self._attributes)
        if self._data.arrays.b is not None:
            names.append("bfactor")
        if self._data.arrays.occ is not None:
            names.append("occupancy")
        return names

    def load_attributes(self, path: Any) -> List[str]:
        """Read custom ``_atom_site`` columns from an mmCIF file and register them.

        The file is matched to this session's model **by atom identity** (chain,
        residue, insertion code, altloc, atom name), so it need not be in the same
        order; atoms absent from the file get ``nan``. Returns the attribute names
        loaded. Requires a model-backed session.
        """
        if self._data.model is None:
            raise ValueError("load_attributes needs a model-backed session")
        from . import cctbx_io

        loaded = cctbx_io.attributes_from_cif(path, self._data.model)
        for name, values in loaded.items():
            self.set_attribute(name, values)
        return list(loaded)

    def load_attribute_text(self, name: str, path: Any) -> str:
        """Register a per-atom attribute from a plain one-value-per-line text file.

        Values align to atoms **by position** — line *i* is atom *i* (i_seq order) —
        so the file must have exactly one value per atom (blank lines and ``#``
        comments are ignored; ``nan``/``.``/``na`` mark missing). Simpler than an
        mmCIF column when you just have a column of numbers in atom order. Returns
        the attribute name.
        """
        values = _read_value_column(path)
        if values.shape[0] != self._n_atoms:
            raise ValueError(
                f"{path} has {values.shape[0]} values but the structure has {self._n_atoms} atoms"
            )
        self.set_attribute(name, values)
        return str(name)

    def write_cif(self, path: Any, *, attributes: Any = None) -> None:
        """Write the model, plus per-atom attributes, to an mmCIF file.

        Each attribute becomes a custom ``_atom_site.<name>`` column. ``attributes``
        is a list of attribute names (defaults to every registered custom attribute)
        or a ``{name: values}`` mapping; ``'bfactor'`` / ``'occupancy'`` and raw
        arrays are resolved like :meth:`color_by`. Requires a model-backed session.
        """
        if self._data.model is None:
            raise ValueError("write_cif needs a model-backed session")
        from . import cctbx_io

        if attributes is None:
            resolved = {n: self._resolve_attribute(n) for n in self._attributes}
        elif isinstance(attributes, dict):
            resolved = {str(n): self._resolve_attribute(v) for n, v in attributes.items()}
        else:
            resolved = {str(n): self._resolve_attribute(n) for n in attributes}
        cctbx_io.write_model_with_attributes(self._data.model, resolved, path)

    def _resolve_attribute(self, attribute: Any) -> np.ndarray:
        """Resolve an attribute name (or a raw length-N array) to a float array."""
        if isinstance(attribute, str):
            if attribute in self._attributes:
                return self._attributes[attribute]
            if attribute in ("bfactor", "b_factor", "b", "b_iso"):
                if self._data.arrays.b is None:
                    raise ValueError("this model has no B-factors")
                return np.asarray(self._data.arrays.b, dtype=float)
            if attribute in ("occupancy", "occ"):
                if self._data.arrays.occ is None:
                    raise ValueError("this model has no occupancies")
                return np.asarray(self._data.arrays.occ, dtype=float)
            raise ValueError(
                f"unknown attribute {attribute!r}; register it with set_attribute() "
                f"or use 'bfactor' / 'occupancy'"
            )
        arr = np.asarray(attribute, dtype=float).reshape(-1)
        if arr.shape[0] != self._n_atoms:
            raise ValueError(f"attribute has {arr.shape[0]} values but the structure has {self._n_atoms} atoms")
        return arr

    def color_by(
        self,
        attribute: Any,
        *,
        type: str = "ball_and_stick",
        palette: Any = "turbo",
        domain: Optional[tuple] = None,
        on: Any = None,
        id: Optional[str] = None,
    ) -> str:
        """Colour atoms by a per-atom attribute, mapped through a colour scale.

        ``attribute`` is ``'bfactor'``, ``'occupancy'``, the name of an attribute
        registered with :meth:`set_attribute`, or a raw length-N array of values.
        The values are mapped onto ``palette`` (a Mol* colour-list name such as
        ``'turbo'``, ``'viridis'``, ``'spectral'``, or an explicit list of colours)
        over ``domain`` (``(min, max)``; taken from the finite values when omitted).
        Non-finite values render in the theme's missing colour.

        This sets a single representation of ``type`` (optionally limited to ``on``),
        replacing any current ones — like :meth:`set_representation`. Returns the
        representation id. The colouring is replayed to viewers that connect later.
        """
        values = self._resolve_attribute(attribute)
        if domain is None:
            finite = values[np.isfinite(values)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            if lo == hi:
                hi = lo + 1.0
            domain = (lo, hi)

        # The per-atom values go over the wire as a compact binary message (f32,
        # NaN = missing), keyed so the representation JSON can stay small — this is
        # what makes colouring very large structures cheap. See _TAG_ATTRIBUTE.
        self._attribute_counter += 1
        key = f"attr-{self._attribute_counter}"
        key_bytes = key.encode("utf-8")
        f32 = values.astype("<f4", copy=False)
        # Pad the key so the f32 block is 4-byte aligned -> the client reads it as a
        # zero-copy Float32Array view (header is [u32 tag][u32 keyLen][key][pad]).
        pad = (-len(key_bytes)) % 4
        payload = (
            struct.pack("<II", _TAG_ATTRIBUTE, len(key_bytes))
            + key_bytes + b"\x00" * pad + f32.tobytes()
        )

        self._representations.clear()
        spec = self._make_repr_spec(type, None, None, on, None, None, id)
        spec["color"] = "attribute"
        spec["attribute"] = {
            "name": attribute if isinstance(attribute, str) else "values",
            "key": key,
            "domain": [float(domain[0]), float(domain[1])],
            "palette": palette,
        }
        self._representations[spec["id"]] = spec
        self._attribute_payloads[key] = payload

        # Send the values (binary) before the representation (JSON) that names them.
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast, payload)
        self._send_representations()
        return spec["id"]

    def _prune_attributes(self) -> None:
        """Drop attribute payloads no longer referenced by any representation."""
        keys = {
            r["attribute"]["key"]
            for r in self._representations.values()
            if r.get("color") == "attribute" and "attribute" in r
        }
        self._attribute_payloads = {k: v for k, v in self._attribute_payloads.items() if k in keys}

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

    def orient_camera(self, target: Any, up: Any, direction: Any, radius: float) -> None:
        """Aim the camera at ``target`` with an explicit orientation. ``up`` is
        screen-up, ``direction`` is the view axis (eye -> target), ``radius`` frames
        the view. Thread-safe. Used to show a residue N->C left-to-right, side chain up.
        """
        self._send_control({
            "type": "orient",
            "target": [float(c) for c in target],
            "up": [float(c) for c in up],
            "direction": [float(c) for c in direction],
            "radius": float(radius),
        })

    def clear_selection(self) -> None:
        """Clear any highlighted selection in the viewer. Thread-safe."""
        self._last_highlight_indices = []
        self._last_highlight_expression = None
        self._send_control({"type": "highlight", "atoms": _encode_index_set([])})
        self._send_control(
            {"type": "select", "reqId": -1, "expression": "", "highlight": True, "focus": False}
        )

    def _next_req(self) -> int:
        with self._pending_lock:
            self._req_counter += 1
            return self._req_counter

    def _selection_request(
        self, expression: str, *, highlight: bool, focus: bool, timeout: float
    ) -> Optional[Selection]:
        """Send a PyMOL expression to the viewer and wait for the matched indices."""
        req_id = self._next_req()
        slot: dict = {"event": threading.Event(), "indices": None, "error": None}
        with self._pending_lock:
            self._pending[req_id] = slot
        if highlight:
            self._last_highlight_expression = expression
        self._send_control(
            {
                "type": "select",
                "reqId": req_id,
                "expression": expression,
                "highlight": highlight,
                "focus": focus,
            }
        )
        answered = slot["event"].wait(timeout)
        with self._pending_lock:
            self._pending.pop(req_id, None)
        if not answered:
            return None
        if slot["error"]:
            raise ValueError(f"invalid selection {expression!r}: {slot['error']}")
        raw = slot["indices"] or []
        indices = [i for i in raw if isinstance(i, int) and 0 <= i < self._n_atoms]
        return self._make_selection(indices)

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
        return self._make_selection(self._mouse_selection_indices)

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

    def _normalize_repr_type(self, type: str) -> str:
        """Validate an MVS representation type and map it to Mol*'s internal name."""
        mvs = _REPR_ALIASES.get(type, type)
        if mvs not in _STRUCTURE_REPR_TYPES:
            raise ValueError(
                f"unknown representation type {type!r}; use one of "
                f"{sorted(_STRUCTURE_REPR_TYPES)} (or aliases 'sphere', 'ribbon')"
            )
        return _MVS_TO_MOLSTAR_REPR[mvs]

    def _make_repr_spec(self, type, color, color_value, on, opacity, params, id) -> dict:
        spec: dict = {
            "id": id if id is not None else self._next_representation_id(),
            "type": self._normalize_repr_type(type),
        }
        # Colour: a uniform SVG name / hex (MVS ColorT), else a Mol* colour theme.
        if color_value is not None:
            spec["color"], spec["colorValue"] = "uniform", color_value
        elif color is not None and (color.startswith("#") or color in _SVG_COLOR_NAMES):
            spec["color"], spec["colorValue"] = "uniform", color
        elif color is not None:
            spec["color"] = color  # a Mol* colour theme, e.g. 'element-symbol'
        if on is not None:
            spec["on"] = _encode_index_set(self._as_selection(on).indices)
        if opacity is not None:
            spec["opacity"] = float(opacity)
        if params:
            spec["params"] = params
        return spec

    def _send_representations(self) -> None:
        self._prune_attributes()  # drop payloads no representation references anymore
        self._send_control({"type": "representations", "reprs": list(self._representations.values())})

    def _as_selection(self, spec: Any) -> Selection:
        """Coerce a Selection / index / indices / mask / cctbx string to a Selection."""
        if isinstance(spec, Selection):
            return spec
        if isinstance(spec, str):
            return self.select_by(selection=spec)
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
            arr, i = self._data.arrays, atoms[0]
            return {"text": f"{arr.name[i]}{int(arr.resseq[i])}"}
        options: dict = {"label": True}
        if kind in ("angle", "dihedral"):
            options["opacity"] = 0.35
        return options

    def _current_coords(self) -> np.ndarray:
        """The most recent coordinates (last streamed frame, else the topology's)."""
        if self._last_frame is not None:
            arr = np.frombuffer(self._last_frame[8:], dtype="<f4")
            return arr.reshape(self._n_atoms, 3).astype(float)
        return self._data.coords.astype(float)

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
        try:
            loop.run_until_complete(self._serve())
            loop.run_forever()
        except Exception as exc:
            self._start_error = exc
            self._started_or_error.set()
        finally:
            loop.run_until_complete(self._shutdown())
            loop.close()

    async def _serve(self) -> None:
        import websockets

        self._server = await websockets.serve(
            self._handler, self.host, self.port, process_request=self._process_request,
            # Screenshots come back this way and are genuinely large — a 1640x1280 PNG
            # is ~700 kB before base64 adds a third. The default 1 MiB cap does not just
            # drop an oversized message, it closes the connection (1009), so one picture
            # would take the whole live session down with it.
            max_size=_MAX_MESSAGE_BYTES,
        )
        # Resolve the actual bound port (matters when port=0 was requested).
        for sock in self._server.sockets or []:
            self.port = sock.getsockname()[1]
            break
        self._ready.set()
        self._started_or_error.set()

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
            if self._last_highlight_expression is not None:
                # Bring a late-joining viewer up to the current PyMOL highlight.
                await self._locked_send(
                    websocket,
                    json.dumps(
                        {
                            "type": "select",
                            "reqId": -1,
                            "expression": self._last_highlight_expression,
                            "highlight": True,
                            "focus": False,
                        }
                    ),
                )
            for message in list(self._primitives.values()):
                # Bring a late-joining viewer up to the active drawing primitives.
                await self._locked_send(websocket, json.dumps(message))
            # Attribute values (binary) must precede the representations that
            # reference them by key, so the client has them when it colours.
            for payload in self._attribute_payloads.values():
                await self._locked_send(websocket, payload)
            if self._representations:
                await self._locked_send(
                    websocket, json.dumps({"type": "representations", "reprs": list(self._representations.values())})
                )
            if self._interactions_contacts:
                await self._locked_send(
                    websocket,
                    json.dumps({"type": "interactions", "action": "set", "contacts": self._interactions_contacts}),
                )
            if self._computed_interactions_visible:
                await self._locked_send(websocket, json.dumps({"type": "computed-interactions", "visible": True}))
            if self._volume_scroll_target is not None:
                await self._locked_send(
                    websocket,
                    json.dumps({"type": "volume_scroll_target", "ref": self._volume_scroll_target}),
                )
            for clip in list(self._clips.values()):
                await self._locked_send(websocket, json.dumps(clip))
            for ref in list(self._hidden_volumes):
                # The rebuilt scene draws every volume; re-hide the ones that were hidden.
                await self._locked_send(
                    websocket, json.dumps({"type": "volume_visible", "ref": ref, "visible": False}))
            if self._clashes:
                await self._locked_send(
                    websocket, json.dumps({"type": "clashes", "action": "set", "pairs": self._clashes})
                )
            for payload in self._probe_dots_payloads.values():
                await self._locked_send(websocket, payload)
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
            selection = self._make_selection(indices)
            self._selection_changed.set()
            for handler in self._selection_handlers:
                try:
                    handler(selection)
                except Exception:  # pragma: no cover - user callback errors
                    pass
        elif etype == "measure":
            self._on_measure(event)
        elif etype == "tug":
            action, atom = event.get("action"), event.get("atom")
            if action in ("begin", "move", "end") and isinstance(atom, int):
                target = event.get("target") if action == "move" else None
                for handler in self._tug_handlers:
                    try:
                        handler(action, atom, target)
                    except Exception:  # pragma: no cover - user callback errors
                        pass
        elif etype == "marker":
            position = event.get("position")
            atom = event.get("atom")
            if isinstance(position, list) and len(position) == 3:
                pos = [float(c) for c in position]
                idx = atom if isinstance(atom, int) else None
                for handler in self._marker_handlers:
                    try:
                        handler(pos, idx)
                    except Exception:  # pragma: no cover - user callback errors
                        pass
        elif etype == "volume-iso-changed":
            ref, value = event.get("ref"), event.get("value")
            if isinstance(ref, str) and isinstance(value, (int, float)):
                for handler in self._volume_iso_handlers:
                    try:
                        handler(ref, float(value))
                    except Exception:  # pragma: no cover - user callback errors
                        pass
        elif etype == "screenshot-result":
            req_id = event.get("reqId")
            with self._pending_lock:
                slot = self._pending.get(req_id)
            if slot is not None:
                slot["uri"] = event.get("dataUri")
                slot["error"] = event.get("error")
                slot["event"].set()
        elif etype == "selection-result":
            req_id = event.get("reqId")
            with self._pending_lock:
                slot = self._pending.get(req_id)
            if slot is not None:
                # First responder wins (harmless with multiple viewers connected).
                slot["indices"] = event.get("indices")
                slot["error"] = event.get("error")
                slot["event"].set()

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
    sites: Any,
    *,
    steps: int = 240,
    amplitude: float = 3.0,
    wavelength: float = 4.0,
) -> Iterable[np.ndarray]:
    """Yield a looping demo trajectory: a travelling sine wave along +y.

    ``sites`` is the base ``(N, 3)`` coordinates. Topology is unchanged; only y is
    displaced per frame. Useful for exercising the live path without a simulation.
    """
    base = np.asarray(sites, dtype="<f4").reshape(-1, 3)
    n = base.shape[0]
    step = 0
    while True:
        phase = 2.0 * np.pi * (step / steps)
        offsets = amplitude * np.sin(np.arange(n) / wavelength + phase)
        frame = base.copy()
        frame[:, 1] = base[:, 1] + offsets
        yield frame
        step = (step + 1) % steps
