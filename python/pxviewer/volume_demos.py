"""Static volume demos for pxviewer.

Each demo generates a 3D scalar field and produces an MRC/MAP file plus an MVSJ
scene that loads it. A small bundled server can serve the generated files together
with the built frontend so the demo is viewable in a browser.
"""

from __future__ import annotations

import dataclasses
import functools
import http.server
import os
import socketserver
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .data import Atom
from .live import LiveSession
from .volume import Volume, create_volume_view, write_volume

__all__ = [
    "VolumeDemo",
    "VOLUME_DEMOS",
    "list_volume_demos",
    "create_volume_demo",
    "run_volume_demo",
    "run_live_volume_demo",
]


def _grid(shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a normalised [x, y, z] meshgrid for volume data of the given shape.

    The returned arrays have shape ``(nx, ny, nz)`` and are indexed in ``xyz``
    order, matching the ``data_order='xyz'`` convention used by ``write_volume``.
    """
    x = np.linspace(-1.0, 1.0, shape[0])
    y = np.linspace(-1.0, 1.0, shape[1])
    z = np.linspace(-1.0, 1.0, shape[2])
    return np.meshgrid(x, y, z, indexing="ij")


def _gaussian_blob(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    X, Y, Z = _grid(shape)
    dx = X - center[0]
    dy = Y - center[1]
    dz = Z - center[2]
    return amplitude * np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma))


def _make_gaussian(shape: Tuple[int, int, int]) -> np.ndarray:
    return _gaussian_blob(shape, (0.0, 0.0, 0.0), 0.4, 5.0).astype(np.float32)


def _make_two_blobs(shape: Tuple[int, int, int]) -> np.ndarray:
    a = _gaussian_blob(shape, (-0.5, 0.0, 0.0), 0.25, 4.0)
    b = _gaussian_blob(shape, (0.5, 0.0, 0.0), 0.25, 4.0)
    return (a + b).astype(np.float32)


def _make_two_blobs_multi(shape: Tuple[int, int, int]) -> List[np.ndarray]:
    """Two separate Gaussian blobs, one per volume."""
    a = _gaussian_blob(shape, (-0.5, 0.0, 0.0), 0.25, 4.0).astype(np.float32)
    b = _gaussian_blob(shape, (0.5, 0.0, 0.0), 0.25, 4.0).astype(np.float32)
    return [a, b]


def _make_shell(shape: Tuple[int, int, int]) -> np.ndarray:
    X, Y, Z = _grid(shape)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    return (5.0 * np.exp(-((r - 0.7) ** 2) / 0.008)).astype(np.float32)


def _make_lattice(shape: Tuple[int, int, int]) -> np.ndarray:
    data = np.zeros(shape, dtype=np.float32)
    spacing = max(2, min(shape) // 4)
    for ix in range(0, shape[0], spacing):
        for iy in range(0, shape[1], spacing):
            for iz in range(0, shape[2], spacing):
                cx = 2.0 * (ix / max(1, shape[0] - 1)) - 1.0
                cy = 2.0 * (iy / max(1, shape[1] - 1)) - 1.0
                cz = 2.0 * (iz / max(1, shape[2] - 1)) - 1.0
                data += _gaussian_blob(shape, (cx, cy, cz), 0.12, 1.5)
    return data


def _make_ripple(shape: Tuple[int, int, int]) -> np.ndarray:
    X, Y, Z = _grid(shape)
    return (2.5 * (1.0 + np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y) * np.exp(-Z * Z))).astype(np.float32)


@dataclasses.dataclass
class VolumeDemo:
    name: str
    description: str
    make_data: Callable[[Tuple[int, int, int]], np.ndarray | List[np.ndarray]]
    view_kwargs: dict | List[dict]


VOLUME_DEMOS: Dict[str, VolumeDemo] = {
    d.name: d
    for d in [
        VolumeDemo(
            "gaussian",
            "A single Gaussian density blob centred at the origin.",
            _make_gaussian,
            {"isosurface_value": 2.0, "isosurface_kind": "relative", "color": "gold"},
        ),
        VolumeDemo(
            "two_blobs",
            "Two separated Gaussian density blobs.",
            _make_two_blobs,
            {"isosurface_value": 2.0, "isosurface_kind": "relative", "color": "teal"},
        ),
        VolumeDemo(
            "shell",
            "A thin spherical shell.",
            _make_shell,
            {"isosurface_value": 2.0, "isosurface_kind": "absolute", "color": "red"},
        ),
        VolumeDemo(
            "lattice",
            "A 3D lattice of small Gaussian blobs.",
            _make_lattice,
            {"isosurface_value": 1.5, "isosurface_kind": "relative", "color": "blue"},
        ),
        VolumeDemo(
            "ripple",
            "A sinusoidal ripple pattern.",
            _make_ripple,
            {"isosurface_value": 2.0, "isosurface_kind": "relative", "color": "purple"},
        ),
        VolumeDemo(
            "multi_volume",
            "Two addressable volumes with different initial colors and opacities.",
            _make_two_blobs_multi,
            [
                {"isosurface_value": 2.0, "isosurface_kind": "relative", "color": "red", "opacity": 0.8},
                {"isosurface_value": 2.0, "isosurface_kind": "relative", "color": "blue", "opacity": 0.8},
            ],
        ),
    ]
}


def list_volume_demos() -> List[Tuple[str, str]]:
    return [(d.name, d.description) for d in VOLUME_DEMOS.values()]


def create_volume_demo(
    name: str,
    *,
    mrc_path: str | os.PathLike,
    mvsj_path: str | os.PathLike,
    voxel_size: float = 1.0,
    shape: Tuple[int, int, int] = (32, 32, 32),
    write_kwargs: Optional[dict] = None,
    view_kwargs: Optional[dict] = None,
) -> str:
    """Generate a volume demo and write the MRC and MVSJ files.

    Returns the generated MVSJ string.
    """
    demo = VOLUME_DEMOS.get(name)
    if demo is None:
        available = ", ".join(VOLUME_DEMOS)
        raise ValueError(f"unknown volume demo '{name}'. Available: {available}")

    data = demo.make_data(shape)
    if not isinstance(data, list):
        data = [data]

    demo_view_kwargs = demo.view_kwargs
    if isinstance(demo_view_kwargs, list):
        per_volume_demo_kwargs = demo_view_kwargs
    else:
        per_volume_demo_kwargs = [demo_view_kwargs] * len(data)

    if view_kwargs is None:
        view_kwargs = {}

    volumes = []
    mrc_path_obj = Path(mrc_path)
    base_name = mrc_path_obj.name
    for i, vol_data in enumerate(data):
        mrc_name = base_name if i == 0 else f"{i}-{base_name}"
        mrc_out = mrc_path_obj.with_name(mrc_name)
        vol_kwargs = dict(per_volume_demo_kwargs[i])
        vol_kwargs.update(view_kwargs)
        write_volume(
            vol_data,
            mrc_out,
            voxel_size=voxel_size,
            data_order="xyz",
        )
        volumes.append(
            Volume(
                url=mrc_out.name,
                ref=f"volume-{i}",
                **vol_kwargs,
            )
        )

    mvsj = create_volume_view(volumes=volumes)
    with open(mvsj_path, "w") as f:
        f.write(mvsj)
    return mvsj


class _VolumeDemoHandler(http.server.SimpleHTTPRequestHandler):
    """Serve generated volume files from ``volume_dir`` with the frontend as fallback."""

    def __init__(self, *args, volume_dir: str, frontend_dir: str, mvsj_url: str, ws_url: Optional[str] = None, **kwargs):
        self.volume_dir = Path(volume_dir)
        self.frontend_dir = Path(frontend_dir)
        self.mvsj_url = mvsj_url
        self.ws_url = ws_url
        super().__init__(*args, directory=str(volume_dir), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (name required by base class)
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            location = f"/index.html?mvsj={self.mvsj_url}"
            if self.ws_url:
                location += f"&ws={self.ws_url}"
            self.send_header("Location", location)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, *args) -> None:  # keep the console focused on the demo
        pass

    def translate_path(self, path: str) -> str:
        """Resolve against the volume directory first, then the frontend directory."""
        # Reproduce SimpleHTTPRequestHandler's path normalisation, then try each base.
        path = path.split("?", 1)[0]
        path = path.split("#", 1)[0]
        path = urllib.parse.unquote(path)
        path = os.path.normpath(path)
        words = [w for w in path.split("/") if w and w not in (os.curdir, os.pardir)]
        for base in (self.volume_dir, self.frontend_dir):
            candidate = base
            for word in words:
                candidate = candidate / word
            if candidate.exists():
                return str(candidate)
        # Default to the volume directory; this will produce a 404 for missing files.
        return str(self.volume_dir.joinpath(*words))


class _VolumeDemoServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _find_frontend_dir() -> Optional[Path]:
    from . import appserver

    return appserver.find_frontend_dir()


def run_volume_demo(
    name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 5173,
    voxel_size: float = 1.0,
    shape: Tuple[int, int, int] = (32, 32, 32),
    serve: bool = True,
) -> None:
    """Generate a volume demo and optionally serve it with the built frontend."""
    demo = VOLUME_DEMOS.get(name)
    if demo is None:
        available = ", ".join(VOLUME_DEMOS)
        raise SystemExit(f"unknown volume demo '{name}'. Available: {available}")

    if not serve:
        create_volume_demo(
            name,
            mrc_path="volume.mrc",
            mvsj_path="volume.mvsj",
            voxel_size=voxel_size,
            shape=shape,
        )
        print(f"\n{demo.name}: {demo.description}")
        print("Wrote volume.mrc and volume.mvsj")
        return

    frontend_dir = _find_frontend_dir()
    if frontend_dir is None or not (frontend_dir / "build" / "index.js").exists():
        raise SystemExit(
            "frontend not built. Run `cd frontend && npm install && npm run build`"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        volume_dir = Path(tmpdir)
        mrc_path = volume_dir / "volume.mrc"
        mvsj_path = volume_dir / "volume.mvsj"
        create_volume_demo(
            name,
            mrc_path=mrc_path,
            mvsj_path=mvsj_path,
            voxel_size=voxel_size,
            shape=shape,
        )

        print(f"\n{demo.name}: {demo.description}")
        handler = functools.partial(
            _VolumeDemoHandler,
            volume_dir=str(volume_dir),
            frontend_dir=str(frontend_dir),
            mvsj_url="volume.mvsj",
        )
        try:
            httpd = _VolumeDemoServer((host, port), handler)
        except OSError:
            httpd = _VolumeDemoServer((host, 0), handler)
        actual_port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, name="pxviewer-volume-demo", daemon=True)
        thread.start()

        print(f"Open the viewer in your browser: http://{host}:{actual_port}/", flush=True)
        print("Press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nstopping...")
        finally:
            httpd.shutdown()
            httpd.server_close()


def run_live_volume_demo(
    *,
    host: str = "127.0.0.1",
    http_port: int = 5173,
    ws_port: int = 8787,
    voxel_size: float = 1.0,
    shape: Tuple[int, int, int] = (32, 32, 32),
    period: float = 2.0,
) -> None:
    """Serve a two-volume scene and live-cycle colors and opacities over it.

    This opens the frontend at ``http://host:http_port/?mvsj=volume.mvsj&ws=...``
    and starts a `LiveSession` (using a single off-screen atom for the WebSocket
    channel). The main thread then animates the two volumes, demonstrating
    multi-volume addressing, live color changes and live opacity changes.
    """
    frontend_dir = _find_frontend_dir()
    if frontend_dir is None or not (frontend_dir / "build" / "index.js").exists():
        raise SystemExit(
            "frontend not built. Run `cd frontend && npm install && npm run build`"
        )

    x = np.linspace(-1.0, 1.0, shape[0])
    y = np.linspace(-1.0, 1.0, shape[1])
    z = np.linspace(-1.0, 1.0, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    left = (4.0 * np.exp(-((X + 0.5) ** 2 + Y * Y + Z * Z) / (2.0 * 0.25 * 0.25))).astype(np.float32)
    right = (4.0 * np.exp(-((X - 0.5) ** 2 + Y * Y + Z * Z) / (2.0 * 0.25 * 0.25))).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        volume_dir = Path(tmpdir)
        write_volume(left, volume_dir / "0-volume.mrc", voxel_size=voxel_size, data_order="xyz")
        write_volume(right, volume_dir / "1-volume.mrc", voxel_size=voxel_size, data_order="xyz")
        mvsj = create_volume_view(
            volumes=[
                Volume(url="0-volume.mrc", ref="volume-0", color="red", opacity=0.8, isosurface_value=2.0, isosurface_kind="relative"),
                Volume(url="1-volume.mrc", ref="volume-1", color="blue", opacity=0.8, isosurface_value=2.0, isosurface_kind="relative"),
            ]
        )
        with open(volume_dir / "volume.mvsj", "w") as f:
            f.write(mvsj)

        # A single off-screen atom is enough to keep the LiveSession WebSocket open.
        dummy_atom = Atom(id=1, element="C", name="C", resname="UNL", resseq=1, chain="A", x=100.0, y=0.0, z=0.0)
        session = LiveSession([dummy_atom])
        session.start(host=host, port=ws_port)
        ws_url = f"ws://{host}:{session.port}"

        handler = functools.partial(
            _VolumeDemoHandler,
            volume_dir=str(volume_dir),
            frontend_dir=str(frontend_dir),
            mvsj_url="volume.mvsj",
            ws_url=ws_url,
        )
        try:
            httpd = _VolumeDemoServer((host, http_port), handler)
        except OSError:
            httpd = _VolumeDemoServer((host, 0), handler)
        actual_port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, name="pxviewer-live-volume-demo", daemon=True)
        thread.start()

        print(f"\nLive two-volume demo: cycling colors and opacities over a {shape[0]}x{shape[1]}x{shape[2]} grid")
        print(f"Open the viewer in your browser: http://{host}:{actual_port}/", flush=True)
        print("Press Ctrl-C to stop.")

        colors = ["red", "green", "blue", "purple", "gold"]
        opacities = [1.0, 0.6, 0.3]
        step = 0
        try:
            print("Waiting for a viewer to connect...", flush=True)
            while session.client_count == 0:
                time.sleep(0.25)
            print("Viewer connected — animating.", flush=True)
            while True:
                session.set_volume_color("volume-0", colors[step % len(colors)])
                session.set_volume_opacity("volume-0", opacities[step % len(opacities)])
                session.set_volume_color("volume-1", colors[(step + 2) % len(colors)])
                session.set_volume_opacity("volume-1", opacities[(step + 1) % len(opacities)])
                time.sleep(period)
                step += 1
        except KeyboardInterrupt:
            print("\nstopping...")
        finally:
            session.stop()
            httpd.shutdown()
            httpd.server_close()
