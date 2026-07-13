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
  - {"type": "select", "reqId": int, "expression": str,        evaluate a PyMOL
     "highlight": bool, "focus": bool}                          selection and show it

Client -> server (UTF-8 JSON text):
  - {"type": "ready"}                              after topology is parsed
  - {"type": "pick", "empty": bool, "atom": {...}} on click (atom omitted if empty)
  - {"type": "selection-result", "reqId": int,     echo of a "select" request;
     "indices": [int, ...], "error": str|None}      indices are positional atom rows

The PyMOL expression is parsed and evaluated by Mol* in the browser (via its
`mol-script` pymol transpiler); Python sends the string and gets back the matched
positional atom indices. See `Selection` and `LiveSession.select`.
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

__all__ = ["LiveSession", "Selection", "ATOM_IDENTITY_CONTRACT"]

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
    """Atoms matched by a PyMOL selection, resolved by the connected viewer.

    ``indices`` are positional (0-based) rows into the topology's _atom_site
    table — the same stable identity the rest of the live protocol uses (see
    ``ATOM_IDENTITY_CONTRACT``). ``atoms`` are the matching ``Atom`` objects;
    ``ids`` and ``mask`` are derived views.
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

    def __init__(self, atoms: Iterable[Atom]):
        self.atoms: List[Atom] = list(atoms)
        if not self.atoms:
            raise ValueError("LiveSession requires at least one atom")
        self._topology: bytes = encode_bcif(self.atoms)
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

        # PyMOL selections are evaluated in the browser and echoed back; each
        # request waits on an Event keyed by a monotonic id.
        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        self._req_counter = 0
        self._last_highlight: Optional[str] = None  # replayed to clients that connect later

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

    # -- selection (python -> scene -> python) ---------------------------

    def select(
        self,
        expression: str,
        *,
        highlight: bool = True,
        focus: bool = True,
        timeout: float = 5.0,
    ) -> Optional[Selection]:
        """Select atoms by PyMOL syntax and show them in the viewer.

        Composes :meth:`highlight` and :meth:`focus`: with ``highlight`` the
        matched atoms get the selection overlay; with ``focus`` the camera zooms
        to them. Mol* evaluates the expression in the browser and echoes the
        matched atoms back.

        Returns a :class:`Selection` of the matched atoms, or ``None`` if no
        viewer answered within ``timeout`` seconds. Raises ``ValueError`` if the
        viewer rejects the expression as invalid PyMOL syntax.
        """
        return self._selection_request(expression, highlight=highlight, focus=focus, timeout=timeout)

    def highlight(self, expression: str, *, timeout: float = 5.0) -> Optional[Selection]:
        """Highlight atoms matching a PyMOL selection (viewer selection overlay)."""
        return self._selection_request(expression, highlight=True, focus=False, timeout=timeout)

    def focus(self, expression: str, *, timeout: float = 5.0) -> Optional[Selection]:
        """Aim the viewer camera at atoms matching a PyMOL selection."""
        return self._selection_request(expression, highlight=False, focus=True, timeout=timeout)

    def clear_selection(self) -> None:
        """Clear any highlighted selection in the viewer. Thread-safe."""
        self._last_highlight = None
        self._send_control(
            {"type": "select", "reqId": self._next_req(), "expression": "", "highlight": True, "focus": False}
        )

    # -- internals -------------------------------------------------------

    def _next_req(self) -> int:
        with self._pending_lock:
            self._req_counter += 1
            return self._req_counter

    def _send_control(self, message: dict) -> None:
        """Broadcast a JSON control message to all clients (thread-safe)."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast_text, json.dumps(message))

    def _selection_request(
        self, expression: str, *, highlight: bool, focus: bool, timeout: float
    ) -> Optional[Selection]:
        req_id = self._next_req()
        slot: dict = {"event": threading.Event(), "indices": None, "error": None}
        with self._pending_lock:
            self._pending[req_id] = slot
        if highlight:
            self._last_highlight = expression
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
            return None  # no viewer connected, or it did not respond in time
        if slot["error"]:
            raise ValueError(f"invalid selection {expression!r}: {slot['error']}")
        raw = slot["indices"] or []
        indices = [i for i in raw if isinstance(i, int) and 0 <= i < self._n_atoms]
        return Selection(indices, [self.atoms[i] for i in indices], self._n_atoms)

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
        self._clients.add(websocket)
        try:
            await websocket.send(struct.pack("<I", _TAG_TOPOLOGY) + self._topology)
            if self._last_frame is not None:
                await websocket.send(self._last_frame)
            if self._last_highlight is not None:
                # Bring a late-joining viewer up to the current highlight.
                await websocket.send(
                    json.dumps(
                        {
                            "type": "select",
                            "reqId": self._next_req(),
                            "expression": self._last_highlight,
                            "highlight": True,
                            "focus": False,
                        }
                    )
                )
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    continue
                self._on_message(message)
        except Exception:  # pragma: no cover - client disconnects are routine
            pass
        finally:
            self._clients.discard(websocket)

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

    async def _safe_send(self, websocket: Any, payload: bytes) -> None:
        try:
            await websocket.send(payload)
        except Exception:  # pragma: no cover - drop on closed sockets
            self._clients.discard(websocket)

    async def _safe_send_text(self, websocket: Any, message: str) -> None:
        try:
            await websocket.send(message)
        except Exception:  # pragma: no cover - drop on closed sockets
            self._clients.discard(websocket)


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
