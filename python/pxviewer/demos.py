"""Human-facing demos for the pxviewer live bridge.

These are like tests, but meant to be *watched*: each drives a `LiveSession` on
a slow, narrated schedule so a human with the frontend open can see the
coordinate stream animate step by step.

Run one with:

    python -m pxviewer demo wave

and open the frontend at the printed `index.html?ws=...` URL.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from typing import Any, Callable, List, Optional

import numpy as np

from .appserver import announce_viewer, stop_all, stop_frontend
from .data import Atom
from .live import LiveSession

__all__ = ["DEMOS", "Player", "run_demo", "list_demos"]


# -- geometry helpers ----------------------------------------------------

def _smooth(t: float) -> float:
    """Smoothstep easing so motions start and stop gently."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _line(n: int, spacing: float = 1.4) -> np.ndarray:
    x = (np.arange(n) - (n - 1) / 2.0) * spacing
    return np.stack([x, np.zeros(n), np.zeros(n)], axis=1).astype("<f4")


def _ring(n: int, radius: float = 6.0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return radius * np.stack([np.cos(t), np.sin(t), np.zeros(n)], axis=1).astype("<f4")


def _sphere(n: int, radius: float = 6.0) -> np.ndarray:
    """Roughly even points on a sphere (Fibonacci lattice)."""
    idx = np.arange(n)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (idx + 0.5) / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * idx
    xyz = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)
    return (radius * xyz).astype("<f4")


def _helix(n: int, radius: float = 4.0, pitch: float = 6.0, turns: float = 2.0) -> np.ndarray:
    t = np.linspace(0.0, turns * 2.0 * math.pi, n)
    coords = np.stack([radius * np.cos(t), pitch * t / (2.0 * math.pi), radius * np.sin(t)], axis=1)
    coords[:, 1] -= coords[:, 1].mean()
    return coords.astype("<f4")


def _atoms(coords: np.ndarray, element: str = "C") -> List[Atom]:
    return [
        Atom(
            id=i + 1,
            element=element,
            name=element,
            resname="UNL",
            resseq=1,
            chain="A",
            x=float(c[0]),
            y=float(c[1]),
            z=float(c[2]),
        )
        for i, c in enumerate(coords)
    ]


# -- playback ------------------------------------------------------------

class Player:
    """Slow, narrated driver around a `LiveSession` (or any object with `push`)."""

    def __init__(self, session: Any, base: np.ndarray, fps: float = 30.0):
        self.session = session
        self.base = np.asarray(base, dtype="<f4")
        self.fps = fps
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._picks: List[Optional[dict]] = []

    def set_fps(self, fps: float) -> None:
        self.fps = fps

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def step(self, label: Any, message: str) -> None:
        print(f"  → [{label}] {message}", flush=True)

    def hold(self, seconds: float) -> None:
        """Pause, but wake immediately on stop."""
        self._stop.wait(seconds)

    def push(self, coords: np.ndarray) -> None:
        self.session.push(coords)

    def play(self, frame_fn: Callable[[float], np.ndarray], *, seconds: float, fps: Optional[float] = None) -> None:
        """Stream frames of ``frame_fn(t)`` for ``t`` in [0, 1] over ``seconds``."""
        fps = fps or self.fps
        n = max(1, int(round(seconds * fps)))
        dt = 1.0 / fps
        for i in range(n + 1):
            if self._stop.is_set():
                return
            self.session.push(frame_fn(i / n))
            time.sleep(dt)

    # pick plumbing (used by the interactive demo)
    def _on_pick(self, info: Optional[dict]) -> None:
        with self._lock:
            self._picks.append(info)
        if info:
            print(
                f"  ● clicked atom {info.get('id')} "
                f"({info.get('name')} {info.get('resname')}{info.get('resseq')})",
                flush=True,
            )
        else:
            print("  ○ clicked empty space", flush=True)

    def drain_picks(self) -> List[Optional[dict]]:
        with self._lock:
            picks, self._picks = self._picks, []
        return picks


# -- demo scripts --------------------------------------------------------

def _run_wave(p: Player) -> None:
    base = p.base
    n = len(base)
    idx = np.arange(n)

    def frame(amp: float, travel: float) -> np.ndarray:
        c = base.copy()
        c[:, 1] = base[:, 1] + amp * np.sin(2.0 * math.pi * (2.0 * idx / n - travel))
        return c

    while not p.stopped:
        p.step(1, "Flat resting chain.")
        p.push(base)
        p.hold(1.5)
        p.step(2, "A gentle travelling wave builds up.")
        p.play(lambda t: frame(2.0 * t, 1.5 * t), seconds=4.0)
        p.step(3, "The wave grows taller and travels faster.")
        p.play(lambda t: frame(2.0 + 2.0 * t, 1.5 + 4.0 * t), seconds=5.0)
        p.step(4, "Damping smoothly back to rest.")
        p.play(lambda t: frame(4.0 * (1.0 - t), 5.5 + 1.5 * t), seconds=4.0)
        p.hold(1.0)


def _run_breathe(p: Player) -> None:
    base = p.base
    while not p.stopped:
        p.step(0, "Sphere at rest.")
        p.push(base)
        p.hold(1.0)
        for cycle in range(1, 4):
            p.step(f"inhale {cycle}", "expanding outward.")
            p.play(lambda t: base * (1.0 + 0.6 * _smooth(t)), seconds=2.0)
            p.step(f"exhale {cycle}", "contracting back in.")
            p.play(lambda t: base * (1.6 - 0.6 * _smooth(t)), seconds=2.0)


def _run_orbit(p: Player) -> None:
    base = p.base
    offset = np.zeros(3, dtype="<f4")
    legs = [
        ("right", np.array([8, 0, 0], dtype="<f4")),
        ("up", np.array([0, 8, 0], dtype="<f4")),
        ("left", np.array([-8, 0, 0], dtype="<f4")),
        ("down", np.array([0, -8, 0], dtype="<f4")),
    ]
    while not p.stopped:
        for name, delta in legs:
            if p.stopped:
                return
            p.step(name, f"the whole body glides {name} (rigid translation).")
            start = offset.copy()
            p.play(lambda t: base + start + delta * _smooth(t), seconds=2.5)
            offset = start + delta
            p.hold(0.4)


def _run_morph(p: Player) -> None:
    a = p.base
    b = _helix(len(a), radius=4.0, pitch=6.0, turns=2.0)
    while not p.stopped:
        p.step(1, "Extended chain.")
        p.push(a)
        p.hold(1.0)
        p.step(2, "Folding into a helix…")
        p.play(lambda t: a + (b - a) * _smooth(t), seconds=4.0)
        p.step(3, "Holding the folded state.")
        p.hold(1.2)
        p.step(4, "Unfolding back to extended…")
        p.play(lambda t: b + (a - b) * _smooth(t), seconds=4.0)
        p.hold(1.0)


def _run_pick(p: Player) -> None:
    base = p.base
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    dirs = base / np.clip(norms, 1e-6, None)
    print("  Click any atom in the viewer — it will pulse outward and back.", flush=True)
    active: List[List[float]] = []  # [atom_index, start_time]
    dt = 1.0 / 30.0
    while not p.stopped:
        for info in p.drain_picks():
            if info and info.get("id") is not None:
                i = int(info["id"]) - 1
                if 0 <= i < len(base):
                    active.append([i, time.monotonic()])
        now = time.monotonic()
        active = [a for a in active if now - a[1] < 1.0]
        coords = base.copy()
        for i, t0 in active:
            amp = 3.0 * math.sin(math.pi * (now - t0))  # 0 → 3 → 0 over 1s
            coords[int(i)] = base[int(i)] + dirs[int(i)] * amp
        p.push(coords)
        time.sleep(dt)


def _labeled_chain(per_chain: int = 10, chains: int = 3, spacing: float = 1.4) -> List[Atom]:
    """A straight run of atoms split into named chains, one residue per atom.

    The distinct chain/residue labels just make the highlighted index ranges read
    as visibly different subsets.
    """
    n = per_chain * chains
    coords = _line(n, spacing)
    letters = "ABCDEFGH"
    return [
        Atom(
            id=i + 1,
            element="C",
            name="CA",
            resname="ALA",
            resseq=i + 1,
            chain=letters[i // per_chain],
            x=float(c[0]),
            y=float(c[1]),
            z=float(c[2]),
        )
        for i, c in enumerate(coords)
    ]


def _run_select(p: Player) -> None:
    session = p.session
    base = p.base
    n = len(base)
    steps = [
        ("first chain", list(range(0, 10)), "the first chain"),
        ("a band", list(range(4, 16)), "a residue band spanning chains"),
        ("scattered", list(range(24, 30, 2)), "a scattered subset"),
        ("leading stretch", list(range(0, 8)), "the leading stretch"),
        ("one atom", [min(14, n - 1)], "one atom — the camera zooms in"),
    ]
    while not p.stopped:
        p.push(base)  # stream the resting frame so a late viewer stays live
        for label, indices, desc in steps:
            if p.stopped:
                return
            sel = session.select(indices)  # highlight + focus, by positional index
            p.step(label, f"{desc} — {len(sel)} atom{'' if len(sel) == 1 else 's'} highlighted.")
            p.hold(2.5)
        p.step("clear", "clearing the selection.")
        session.clear_selection()
        p.hold(1.5)


def _bent_chain() -> List[Atom]:
    """A short zig-zag chain — enough elbows for a visible angle and dihedral."""
    pts = np.array(
        [[-5, 0, 0], [-3, 0, 0], [-1, 1.5, 0], [1, 1.5, 0], [3, 0, 0], [5, 0, 0]],
        dtype="<f4",
    )
    return _atoms(pts)


def _run_primitives(p: Player) -> None:
    session = p.session
    base = p.base
    n = len(base)

    def flex(t: float) -> np.ndarray:
        c = base.copy()
        ang = 0.9 * math.sin(2.0 * math.pi * t)
        ca, sa = math.cos(ang), math.sin(ang)
        pivot = base[2]
        for i in range(3, n):  # swing the tail about atom 2
            dx, dy = base[i, 0] - pivot[0], base[i, 1] - pivot[1]
            c[i, 0] = pivot[0] + ca * dx - sa * dy
            c[i, 1] = pivot[1] + sa * dx + ca * dy
        c[3, 2] = base[3, 2] + 1.5 * math.sin(2.0 * math.pi * t)  # out-of-plane -> dihedral moves
        return c

    while not p.stopped:
        p.push(base)
        p.step(1, "Distance between the two ends.")
        session.add_distance(0, n - 1, id="dist")
        p.hold(1.5)
        p.step(2, "Angle at the first elbow.")
        session.add_angle(1, 2, 3, id="ang")
        p.hold(1.5)
        p.step(3, "Dihedral across the middle four atoms.")
        session.add_dihedral(1, 2, 3, 4, id="dih")
        p.hold(1.5)
        p.step(4, "A label on the far end.")
        session.add_label(n - 1, "tip", id="lbl")
        p.hold(1.5)
        p.step(5, "Flexing — every measurement tracks the motion.")
        p.play(flex, seconds=5.0)
        p.step(6, "Clearing all primitives.")
        session.clear_primitives()
        p.hold(1.2)


def _interaction_pair() -> List[Atom]:
    """Two short parallel strands whose facing atoms make a few contacts."""
    top = np.stack([np.linspace(-4, 4, 5), np.full(5, 1.6), np.zeros(5)], axis=1)
    bot = np.stack([np.linspace(-4, 4, 5), np.full(5, -1.6), np.zeros(5)], axis=1)
    return _atoms(np.concatenate([top, bot]).astype("<f4"))


def _run_interactions(p: Player) -> None:
    session = p.session
    base = p.base  # atoms 0..4 = top strand, 5..9 = bottom strand

    # An explicit, typed contact table — exactly what Python declares, nothing
    # inferred. Facing atoms across the two strands.
    session.set_interactions({
        "hydrogen-bond": [(0, 5), (2, 7), (4, 9)],
        "hydrophobic": [(1, 6), (3, 8)],
    })

    def breathe(sep: float) -> np.ndarray:
        c = base.copy()
        c[:5, 1] = sep       # top strand y
        c[5:, 1] = -sep      # bottom strand y
        return c

    while not p.stopped:
        p.step(1, "Two strands with declared H-bonds and hydrophobic contacts.")
        p.push(base)
        p.hold(1.5)
        p.step(2, "Pulling the strands apart — the interactions stretch with them.")
        p.play(lambda t: breathe(1.6 + 2.0 * _smooth(t)), seconds=3.5)
        p.step(3, "Bringing them back together.")
        p.play(lambda t: breathe(3.6 - 2.0 * _smooth(t)), seconds=3.5)
        p.hold(1.0)


def _spin(base: np.ndarray, a: float) -> np.ndarray:
    """Rotate coordinates about the y axis by angle ``a`` (radians)."""
    ca, sa = math.cos(a), math.sin(a)
    c = base.copy()
    c[:, 0] = ca * base[:, 0] + sa * base[:, 2]
    c[:, 2] = -sa * base[:, 0] + ca * base[:, 2]
    return c


def _run_measure(p: Player) -> None:
    session = p.session
    base = p.base
    pending: List[Any] = []
    lock = threading.Lock()

    def on_measure(prim: Any) -> None:  # runs on the server loop thread
        with lock:
            pending.append(prim)

    def drain() -> None:
        with lock:
            items, pending[:] = list(pending), []
        for prim in items:
            value = "—" if prim.value is None else f"{prim.value:.1f}"
            unit = "°" if prim.kind in ("angle", "dihedral") else (" Å" if prim.kind == "distance" else "")
            p.step(prim.kind, f"drew {prim.kind} = {value}{unit}")

    modes = [
        ("distance", "Click 2 atoms to measure a distance."),
        ("angle", "Click 3 atoms to measure an angle."),
        ("dihedral", "Click 4 atoms to measure a dihedral."),
    ]
    dt = 1.0 / 30.0
    angle = 0.0
    while not p.stopped:
        for kind, prompt in modes:
            if p.stopped:
                return
            session.enable_measure_mode(kind, on_measure=on_measure)
            p.step(kind, prompt)
            for _ in range(270):  # ~9 s per mode; the structure turns slowly so drawn measures track
                if p.stopped:
                    return
                angle += 0.008
                p.push(_spin(base, angle))
                drain()
                p.hold(dt)
        p.step("reset", "Clearing the measurements.")
        session.clear_primitives()
        p.hold(1.0)


@dataclasses.dataclass
class Demo:
    name: str
    description: str
    make_atoms: Callable[[], List[Atom]]
    run: Callable[[Player], None]


DEMOS: dict[str, Demo] = {
    d.name: d
    for d in [
        Demo("wave", "A chain rippling with a growing travelling wave.", lambda: _atoms(_line(24)), _run_wave),
        Demo("breathe", "A sphere of atoms expanding and contracting.", lambda: _atoms(_sphere(64)), _run_breathe),
        Demo("orbit", "A rigid body gliding around a square path.", lambda: _atoms(_helix(20, radius=2.5, pitch=3.0)), _run_orbit),
        Demo("morph", "A chain folding into a helix and back.", lambda: _atoms(_line(30, spacing=1.2)), _run_morph),
        Demo("pick", "Interactive: click atoms to make them pulse (scene → Python).", lambda: _atoms(_ring(16)), _run_pick),
        Demo("select", "Highlight atoms by index, cycling through subsets.", _labeled_chain, _run_select),
        Demo("primitives", "Angle/distance/dihedral/label measurements that track motion.", _bent_chain, _run_primitives),
        Demo("interactions", "Explicit typed non-covalent contacts that track motion.", _interaction_pair, _run_interactions),
        Demo("measure", "Interactive: click atoms to measure distances/angles/dihedrals (scene → Python).", lambda: _atoms(_helix(12, radius=4.0, pitch=3.0)), _run_measure),
    ]
}


def list_demos() -> List[tuple[str, str]]:
    return [(d.name, d.description) for d in DEMOS.values()]


def _wait_for_client(session: LiveSession, *, timeout: float, tick: float = 0.25) -> bool:
    """Wait for a viewer, polling in short ticks so Ctrl-C stays responsive.

    A single long Event.wait() on the main thread defers a pending KeyboardInterrupt
    until it returns; short ticks surface it within ``tick`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.wait_for_client(tick):
            return True
    return session.wait_for_client(0)


def run_demo(
    name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    fps: float = 30.0,
    http_port: int = 5173,
    serve_frontend: bool = True,
) -> None:
    demo = DEMOS.get(name)
    if demo is None:
        available = ", ".join(DEMOS)
        raise SystemExit(f"unknown demo '{name}'. Available: {available}")

    atoms = demo.make_atoms()
    base = np.array([[a.x, a.y, a.z] for a in atoms], dtype="<f4")
    session = LiveSession(atoms)
    player = Player(session, base, fps=fps)
    session.on_pick(player._on_pick)
    session.start(host=host, port=port)

    ws_url = f"ws://{host}:{session.port}"
    print(f"\n{demo.name}: {demo.description}")
    httpd = announce_viewer(host, ws_url, http_port=http_port, serve=serve_frontend)
    print("Waiting for the viewer to connect…  (Ctrl-C to stop)\n", flush=True)

    try:
        if _wait_for_client(session, timeout=120):
            print("Viewer connected — starting.\n", flush=True)
            player.hold(0.5)
        else:
            print("No viewer connected within 2 min; starting anyway.\n", flush=True)
        demo.run(player)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        # Idempotent, interrupt-proof: a repeated Ctrl-C can't leave threads/ports dangling.
        stop_all(player.stop, session.stop, lambda: stop_frontend(httpd))
