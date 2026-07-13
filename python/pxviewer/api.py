"""High-level Python API for building pxviewer scenes and data."""

import os
from typing import List

import molviewspec as mvs

from .data import Atom, encode_bcif, read_atoms, write_bcif
from .volume import (
    Volume,
    create_volume_view,
    create_volume_view_from_data,
    read_volume,
    set_volume_color,
    set_volume_opacity,
    write_volume,
)

__all__ = [
    "create_view",
    "create_example_view",
    "create_fragment_view",
    "create_volume_view",
    "create_volume_view_from_data",
    "set_volume_color",
    "set_volume_opacity",
    "Volume",
    "Atom",
    "write_bcif",
    "encode_bcif",
    "read_atoms",
    "write_volume",
    "read_volume",
]


def create_view(
    structure_url: str,
    *,
    title: str | None = None,
    components: List[dict] | None = None,
    format: str = "bcif",
) -> str:
    """Build an MVSJ scene from a structure URL.

    Example:
        mvsj = create_view(
            "https://www.ebi.ac.uk/pdbe/entry-files/1cbs.bcif",
            components=[
                {"selector": "polymer", "representation": "cartoon", "color": "green"},
                {"selector": "ligand", "representation": "ball_and_stick", "color": "#cc3399"},
            ],
        )
    """
    builder = mvs.create_builder()
    structure = builder.download(url=structure_url).parse(format=format).model_structure()
    for comp in components or []:
        component = structure.component(selector=comp.get("selector", "all"))
        repr = component.representation(type=comp.get("representation", "cartoon"))
        repr.color(color=comp.get("color", "white"))
    return builder.get_state(title=title).model_dump_json(exclude_none=True)


def create_example_view(
    url: str = "https://www.ebi.ac.uk/pdbe/entry-files/1cbs.bcif",
) -> str:
    """Build an example MVSJ scene with polymer and ligand components."""
    return create_view(
        url,
        title="Example pxviewer scene",
        components=[
            {"selector": "polymer", "representation": "cartoon", "color": "green"},
            {"selector": "ligand", "representation": "ball_and_stick", "color": "#cc3399"},
        ],
    )


def create_fragment_view(
    atoms: List[Atom],
    *,
    bcif_path: str | os.PathLike,
    mvsj_path: str | os.PathLike | None = None,
    title: str | None = None,
) -> str:
    """Write a small atom model to BCIF and return an MVSJ scene that loads it.

    The MVSJ uses the BCIF filename as a relative URL, so both files should be
    served from the same directory.
    """
    write_bcif(atoms, bcif_path)
    bcif_url = os.path.basename(str(bcif_path))
    mvsj = create_view(bcif_url, title=title)
    if mvsj_path is not None:
        with open(mvsj_path, "w") as f:
            f.write(mvsj)
    return mvsj
