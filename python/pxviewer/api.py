"""High-level Python API for building pxviewer scenes and data."""

from typing import List

import molviewspec as mvs

from .data import AtomArrays, encode_bcif_arrays
from .volume import (
    Volume,
    VolumeStyle,
    create_volume_view,
    create_volume_view_from_data,
    read_volume,
    set_volume_color,
    set_volume_opacity,
    set_volume_style,
    write_volume,
)

__all__ = [
    "create_view",
    "create_example_view",
    "create_volume_view",
    "create_volume_view_from_data",
    "set_volume_color",
    "set_volume_opacity",
    "set_volume_style",
    "Volume",
    "VolumeStyle",
    "AtomArrays",
    "encode_bcif_arrays",
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


