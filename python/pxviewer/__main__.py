"""CLI entry point for pxviewer."""

import argparse
import time

from .api import create_example_view
from .data import Atom
from .demos import DEMOS, list_demos, run_demo
from .live import LiveSession, oscillating_frames


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

    demo = subparsers.add_parser(
        "serve-demo",
        help="Stream an oscillating demo structure over WebSocket for the live frontend",
    )
    demo.add_argument("--host", default="127.0.0.1", help="Host to bind")
    demo.add_argument("--port", type=int, default=8787, help="Port to bind")
    demo.add_argument("--atoms", type=int, default=24, help="Number of atoms in the demo chain")
    demo.add_argument("--fps", type=float, default=30.0, help="Frames per second to stream")

    run = subparsers.add_parser(
        "demo",
        help="Run a narrated, slowed-down demo for the live frontend",
    )
    run.add_argument("name", nargs="?", help="Demo to run; omit to list available demos")
    run.add_argument("--host", default="127.0.0.1", help="Host to bind")
    run.add_argument("--port", type=int, default=8787, help="Port to bind")
    run.add_argument("--fps", type=float, default=30.0, help="Frames per second within each motion")

    args = parser.parse_args()

    if args.command == "create-example-view":
        mvsj = create_example_view(args.url)
        with open(args.output, "w") as f:
            f.write(mvsj)
        print(f"Wrote {args.output}")

    elif args.command == "serve-demo":
        atoms = _demo_atoms(args.atoms)
        session = LiveSession(atoms)
        session.on_pick(lambda info: print(f"picked: {info}"))
        session.start(host=args.host, port=args.port)
        url = f"ws://{args.host}:{session.port}"
        print(f"pxviewer live demo streaming at {url}")
        print("Open the frontend with ?ws=" + url + " and press Ctrl-C to stop.")
        delay = 1.0 / args.fps if args.fps > 0 else 0.0
        try:
            for frame in oscillating_frames(atoms):
                session.push(frame)
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\nstopping...")
        finally:
            session.stop()

    elif args.command == "demo":
        if not args.name:
            print("Available demos:\n")
            for name, description in list_demos():
                print(f"  {name:10s} {description}")
            print("\nRun one with:  python -m pxviewer demo <name>")
            return
        run_demo(args.name, host=args.host, port=args.port, fps=args.fps)


if __name__ == "__main__":
    main()
