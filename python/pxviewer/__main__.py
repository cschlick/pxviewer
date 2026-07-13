"""CLI entry point for pxviewer."""

import argparse

from .api import create_example_view


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
    view.add_argument(
        "--output",
        "-o",
        default="scene.mvsj",
        help="Output MVSJ file path",
    )

    args = parser.parse_args()
    if args.command == "create-example-view":
        mvsj = create_example_view(args.url)
        with open(args.output, "w") as f:
            f.write(mvsj)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
