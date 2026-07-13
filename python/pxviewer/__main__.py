"""CLI entry point for pxviewer."""

import argparse
import time

import numpy as np

from .api import create_example_view, create_volume_view_from_data
from .appserver import announce_viewer, stop_all, stop_frontend
from .data import Atom
from .demos import DEMOS, list_demos, run_demo
from .live import LiveSession, oscillating_frames
from .volume_demos import create_volume_demo, list_volume_demos, run_volume_demo


def _demo_atoms(n: int) -> list[Atom]:
    """A simple linear chain of carbons, one per angstrom along +x."""
    return [
        Atom(id=i + 1, element="C", name="C", resname="UNL", resseq=1, chain="A", x=float(i), y=0.0, z=0.0)
        for i in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="pxviewer Python API CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    view = subparsers.add_parser(
        "create-example-view",
        help="Create an example MVSJ scene from a structure URL",
    )
    view.add_argument(
        "--url",
        default="https://www.ebi.ac.uk/pdbe/entry-files/1cbs.bcif",
        help="URL to a BCIF structure file",
    )
    view.add_argument("--output", "-o", default="scene.mvsj", help="Output MVSJ file path")

    vol = subparsers.add_parser(
        "create-volume-view",
        help="Create an example MVSJ scene for a generated volume",
    )
    vol.add_argument("--output-mrc", default="volume.mrc", help="Output MRC file path")
    vol.add_argument("--output-mvsj", default="volume.mvsj", help="Output MVSJ file path")
    vol.add_argument("--voxel-size", type=float, default=1.0, help="Isotropic voxel size in Angstroms")
    vol.add_argument("--isovalue", type=float, default=2.0, help="Isovalue for the isosurface")
    vol.add_argument("--isovalue-kind", choices=["absolute", "relative"], default="relative", help="Whether the isovalue is absolute or relative (sigma)")

    demo = subparsers.add_parser(
        "serve-demo",
        help="Stream an oscillating demo structure over WebSocket for the live frontend",
    )
    demo.add_argument("--host", default="127.0.0.1", help="Host to bind")
    demo.add_argument("--port", type=int, default=8787, help="Port to bind")
    demo.add_argument("--atoms", type=int, default=24, help="Number of atoms in the demo chain")
    demo.add_argument("--fps", type=float, default=30.0, help="Frames per second to stream")
    demo.add_argument("--http-port", type=int, default=5173, help="Port for the bundled frontend server")
    demo.add_argument("--no-frontend", action="store_true", help="Do not serve the frontend (connect manually)")

    run = subparsers.add_parser(
        "demo",
        help="Run a narrated, slowed-down demo for the live frontend",
    )
    run.add_argument("name", nargs="?", help="Demo to run; omit to list available demos")
    run.add_argument("--host", default="127.0.0.1", help="Host to bind")
    run.add_argument("--port", type=int, default=8787, help="Port to bind")
    run.add_argument("--fps", type=float, default=30.0, help="Frames per second within each motion")
    run.add_argument("--http-port", type=int, default=5173, help="Port for the bundled frontend server")
    run.add_argument("--no-frontend", action="store_true", help="Do not serve the frontend (connect manually)")

    vol = subparsers.add_parser(
        "volume-demo",
        help="Generate and serve a static volume demo",
    )
    vol.add_argument("name", nargs="?", help="Demo to run; omit to list available demos")
    vol.add_argument("--host", default="127.0.0.1", help="Host to bind")
    vol.add_argument("--port", type=int, default=5173, help="Port for the bundled frontend server")
    vol.add_argument("--voxel-size", type=float, default=1.0, help="Isotropic voxel size in Angstroms")
    vol.add_argument("--shape", type=int, nargs=3, default=[32, 32, 32], help="Volume grid shape (x y z)")
    vol.add_argument("--output-dir", help="Write volume.mrc and volume.mvsj here and exit without serving")
    vol.add_argument("--no-serve", action="store_true", help="Write files and exit without serving")

    args = parser.parse_args()

    if args.command == "create-example-view":
        mvsj = create_example_view(args.url)
        with open(args.output, "w") as f:
            f.write(mvsj)
        print(f"Wrote {args.output}")

    elif args.command == "create-volume-view":
        shape = (32, 32, 32)
        x = np.linspace(-1.5, 1.5, shape[2])
        y = np.linspace(-1.5, 1.5, shape[1])
        z = np.linspace(-1.5, 1.5, shape[0])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        data = np.exp(-(X**2 + Y**2 + Z**2)) * 5.0
        create_volume_view_from_data(
            data,
            mrc_path=args.output_mrc,
            mvsj_path=args.output_mvsj,
            write_kwargs={"voxel_size": args.voxel_size, "data_order": "xyz"},
            view_kwargs={"isosurface_value": args.isovalue, "isosurface_kind": args.isovalue_kind, "color": "gold"},
        )
        print(f"Wrote {args.output_mrc} and {args.output_mvsj}")

    elif args.command == "serve-demo":
        atoms = _demo_atoms(args.atoms)
        session = LiveSession(atoms)
        session.on_pick(lambda info: print(f"picked: {info}"))
        session.start(host=args.host, port=args.port)
        url = f"ws://{args.host}:{session.port}"
        print(f"pxviewer live demo streaming at {url}")
        httpd = announce_viewer(args.host, url, http_port=args.http_port, serve=not args.no_frontend)
        print("Press Ctrl-C to stop.")
        delay = 1.0 / args.fps if args.fps > 0 else 0.0
        try:
            for frame in oscillating_frames(atoms):
                session.push(frame)
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\nstopping...")
        finally:
            stop_all(session.stop, lambda: stop_frontend(httpd))

    elif args.command == "demo":
        if not args.name:
            print("Available demos:\n")
            for name, description in list_demos():
                print(f"  {name:10s} {description}")
            print("\nRun one with:  python -m pxviewer demo <name>")
            return
        run_demo(
            args.name,
            host=args.host,
            port=args.port,
            fps=args.fps,
            http_port=args.http_port,
            serve_frontend=not args.no_frontend,
        )

    elif args.command == "volume-demo":
        if not args.name:
            print("Available volume demos:\n")
            for name, description in list_volume_demos():
                print(f"  {name:10s} {description}")
            print("\nRun one with:  python -m pxviewer volume-demo <name>")
            return

        if args.output_dir:
            from pathlib import Path

            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            create_volume_demo(
                args.name,
                mrc_path=out / "volume.mrc",
                mvsj_path=out / "volume.mvsj",
                voxel_size=args.voxel_size,
                shape=tuple(args.shape),
            )
            print(f"Wrote {out / 'volume.mrc'} and {out / 'volume.mvsj'}")
            return

        run_volume_demo(
            args.name,
            host=args.host,
            port=args.port,
            voxel_size=args.voxel_size,
            shape=tuple(args.shape),
            serve=not args.no_serve,
        )


if __name__ == "__main__":
    main()
