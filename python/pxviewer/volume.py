"""Volumetric data helpers for writing MRC/MAP files and building MVS scenes."""

import dataclasses
import json
import os
from typing import Any, List, Literal

import mrcfile
import numpy as np

__all__ = [
    "Volume",
    "write_volume",
    "read_volume",
    "create_volume_view",
    "create_volume_view_from_data",
    "set_volume_color",
    "set_volume_opacity",
    "set_volume_style",
]


VolumeStyle = Literal["surface", "wireframe", "mesh"]


@dataclasses.dataclass
class Volume:
    """A single volume and how it should be rendered in an MVSJ scene."""

    url: str
    ref: str | None = None
    isosurface_value: float | None = None
    isosurface_kind: Literal["absolute", "relative"] = "relative"
    color: str | None = "gold"
    opacity: float | None = 1.0
    style: VolumeStyle | None = "surface"


def _normalize_volume_data(data: np.ndarray) -> np.ndarray:
    """Convert a numpy array to a dtype Mol*/mrcfile both support."""
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Volume data must be 3D, got shape {data.shape}")

    if np.issubdtype(data.dtype, np.floating):
        if data.dtype == np.float32:
            return data
        return data.astype(np.float32, copy=False)

    if data.dtype in (np.int8, np.int16, np.uint8, np.uint16):
        return data

    raise ValueError(
        f"Unsupported volume data dtype {data.dtype!r}. "
        "Use float32, int8, int16, uint8 or uint16."
    )


def write_volume(
    data: np.ndarray,
    path: str | os.PathLike,
    *,
    voxel_size: float | tuple[float, float, float] | None = None,
    origin: tuple[float, float, float] | None = None,
    origin_units: Literal["angstrom", "grid"] = "angstrom",
    data_order: Literal["mrc", "xyz"] = "mrc",
    overwrite: bool = True,
) -> None:
    """Write a 3D numpy array to an MRC/MAP file that Mol* can load.

    Parameters
    ----------
    data
        3D array of density values. With ``data_order='mrc'`` (default) the
        array is indexed ``data[z, y, x]`` and has shape ``(nz, ny, nx)``,
        matching the MRC2014 convention. With ``data_order='xyz'`` the array
        is indexed ``data[x, y, z]`` and will be transposed before writing.
    path
        Output file path. Mol* recognizes ``.mrc``, ``.map`` and ``.ccp4``
        extensions.
    voxel_size
        Voxel size in Angstroms. A single float for isotropic, or a 3-tuple
        ``(x, y, z)`` for anisotropic data.
    origin
        Origin of the volume. If ``origin_units='angstrom'`` (default) this is
        the physical origin in Angstroms. If ``origin_units='grid'`` this is
        the grid offset and is written to ``nxstart/nystart/nzstart``.
    origin_units
        Units for the ``origin`` argument.
    data_order
        Ordering of the input array axes.
    overwrite
        Whether to overwrite an existing file.
    """
    data = _normalize_volume_data(data)

    if data_order == "xyz":
        # data[x, y, z] -> data[z, y, x]
        data = np.transpose(data, (2, 1, 0))
    elif data_order != "mrc":
        raise ValueError(f"data_order must be 'mrc' or 'xyz', got {data_order!r}")

    with mrcfile.new(str(path), overwrite=overwrite) as mrc:
        mrc.set_data(data)

        if voxel_size is not None:
            mrc.voxel_size = voxel_size

        if origin is not None:
            if origin_units == "angstrom":
                mrc.header["origin"] = (float(origin[0]), float(origin[1]), float(origin[2]))
            elif origin_units == "grid":
                mrc.header["nxstart"] = int(round(origin[0]))
                mrc.header["nystart"] = int(round(origin[1]))
                mrc.header["nzstart"] = int(round(origin[2]))
            else:
                raise ValueError(f"origin_units must be 'angstrom' or 'grid', got {origin_units!r}")


def read_volume(path: str | os.PathLike) -> dict:
    """Read an MRC/MAP file and return its data plus key metadata.

    Returns
    -------
    dict with keys ``data`` (``nz, ny, nx``), ``voxel_size`` (tuple),
    ``origin`` (tuple) and ``shape`` (tuple ``(nz, ny, nx)``).
    """
    with mrcfile.open(str(path)) as mrc:
        origin = mrc.header["origin"]
        return {
            "data": mrc.data,
            "voxel_size": (float(mrc.voxel_size.x), float(mrc.voxel_size.y), float(mrc.voxel_size.z)),
            "origin": (float(origin["x"]), float(origin["y"]), float(origin["z"])),
            "shape": mrc.data.shape,
        }


def _normalize_volume(v: str | Volume | dict) -> Volume:
    """Convert a string URL or dict into a Volume object."""
    if isinstance(v, Volume):
        return v
    if isinstance(v, str):
        return Volume(url=v)
    return Volume(**v)


def _build_volume(builder: Any, volume: Volume, ref: str) -> str:
    """Add one volume branch to the MVS builder and return the volume ref."""
    import molviewspec as mvs

    mvs_volume = builder.download(url=volume.url).parse(format="map").volume(ref=ref)

    repr_kwargs: dict = {"type": "isosurface"}
    if volume.isosurface_value is not None:
        if volume.isosurface_kind == "absolute":
            repr_kwargs["absolute_isovalue"] = volume.isosurface_value
        elif volume.isosurface_kind == "relative":
            repr_kwargs["relative_isovalue"] = volume.isosurface_value
        else:
            raise ValueError(f"isosurface_kind must be 'absolute' or 'relative', got {volume.isosurface_kind!r}")

    if volume.style == "surface":
        repr_kwargs["show_wireframe"] = False
        repr_kwargs["show_faces"] = True
    elif volume.style == "wireframe":
        repr_kwargs["show_wireframe"] = True
        repr_kwargs["show_faces"] = False
    elif volume.style == "mesh":
        repr_kwargs["show_wireframe"] = True
        repr_kwargs["show_faces"] = True
    elif volume.style is not None:
        raise ValueError(f"style must be 'surface', 'wireframe' or 'mesh', got {volume.style!r}")

    repr = mvs_volume.representation(**repr_kwargs, ref=f"{ref}-repr")
    if volume.color is not None:
        repr = repr.color(color=volume.color)
    if volume.opacity is not None:
        repr = repr.opacity(opacity=volume.opacity)

    mvs_volume.focus()
    return ref


def create_volume_view(
    volume_url: str | None = None,
    *,
    volumes: List[str | Volume | dict] | None = None,
    isosurface_value: float | None = None,
    isosurface_kind: Literal["absolute", "relative"] = "relative",
    color: str | None = "gold",
    opacity: float | None = 1.0,
    style: VolumeStyle | None = "surface",
    title: str | None = None,
) -> str:
    """Build an MVSJ scene that loads one or more MRC/MAP volumes from URLs.

    A single volume may be passed as ``volume_url`` (or with the convenience
    keyword arguments). For multiple volumes, pass ``volumes`` as a list of
    strings, dicts, or :class:`Volume` objects.

    Each volume may be addressed by its ``ref`` (auto-generated if not given)
    so that color, opacity and style can be changed later with
    :func:`set_volume_color`, :func:`set_volume_opacity` and
    :func:`set_volume_style`.
    """
    import molviewspec as mvs

    builder = mvs.create_builder()

    if volumes is not None:
        volume_list = [_normalize_volume(v) for v in volumes]
    elif volume_url is not None:
        volume_list = [
            Volume(
                url=volume_url,
                isosurface_value=isosurface_value,
                isosurface_kind=isosurface_kind,
                color=color,
                opacity=opacity,
                style=style,
            )
        ]
    else:
        raise ValueError("create_volume_view requires volume_url or volumes")

    for i, volume in enumerate(volume_list):
        ref = volume.ref or f"volume-{i}"
        _build_volume(builder, volume, ref)

    return builder.get_state(title=title).model_dump_json(exclude_none=True)


def create_volume_view_from_data(
    data: np.ndarray,
    *,
    mrc_path: str | os.PathLike,
    mvsj_path: str | os.PathLike | None = None,
    title: str | None = None,
    write_kwargs: dict | None = None,
    view_kwargs: dict | None = None,
) -> str:
    """Write a volume to MRC and return an MVSJ scene that loads it.

    The MVSJ uses the MRC filename as a relative URL, so both files should be
    served from the same directory.
    """
    write_kwargs = write_kwargs or {}
    view_kwargs = view_kwargs or {}

    write_volume(data, mrc_path, **write_kwargs)
    mvsj = create_volume_view(str(os.path.basename(str(mrc_path))), title=title, **view_kwargs)

    if mvsj_path is not None:
        with open(mvsj_path, "w") as f:
            f.write(mvsj)

    return mvsj


def _find_volume_node(root: dict, ref: str) -> dict | None:
    """Locate a volume node with the given ref in an MVS tree."""
    for download in root.get("children", []):
        if download.get("kind") != "download":
            continue
        for parse in download.get("children", []):
            if parse.get("kind") != "parse":
                continue
            for volume in parse.get("children", []):
                if volume.get("kind") == "volume" and volume.get("ref") == ref:
                    return volume
    return None


def _upsert_child_node(parent: dict, kind: str, params: dict) -> None:
    """Replace a child node of ``kind`` with ``params`` or append a new one."""
    children = parent.setdefault("children", [])
    for child in children:
        if child.get("kind") == kind:
            child["params"] = params
            return
    children.append({"kind": kind, "params": params})


def set_volume_color(mvsj: str, ref: str, color: str) -> str:
    """Set the color of a specific volume in an MVSJ string.

    ``ref`` is the volume reference used when the scene was built (e.g.
    ``volume-0`` or a custom value passed to :class:`Volume`).
    """
    state = json.loads(mvsj)
    root = state["root"]
    volume = _find_volume_node(root, ref)
    if volume is None:
        raise ValueError(f"volume with ref '{ref}' not found in MVSJ")
    for repr_node in volume.get("children", []):
        if repr_node.get("kind") == "volume_representation":
            _upsert_child_node(repr_node, "color", {"color": color})
    return json.dumps(state, separators=(",", ":"))


def set_volume_opacity(mvsj: str, ref: str, opacity: float) -> str:
    """Set the opacity of a specific volume in an MVSJ string.

    ``ref`` is the volume reference used when the scene was built.
    """
    state = json.loads(mvsj)
    root = state["root"]
    volume = _find_volume_node(root, ref)
    if volume is None:
        raise ValueError(f"volume with ref '{ref}' not found in MVSJ")
    for repr_node in volume.get("children", []):
        if repr_node.get("kind") == "volume_representation":
            _upsert_child_node(repr_node, "opacity", {"opacity": opacity})
    return json.dumps(state, separators=(",", ":"))


def _style_to_show_flags(style: VolumeStyle) -> tuple[bool, bool]:
    """Return (show_wireframe, show_faces) for a given volume style."""
    if style == "surface":
        return False, True
    if style == "wireframe":
        return True, False
    if style == "mesh":
        return True, True
    raise ValueError(f"style must be 'surface', 'wireframe' or 'mesh', got {style!r}")


def set_volume_style(mvsj: str, ref: str, style: VolumeStyle) -> str:
    """Set the isosurface style of a specific volume in an MVSJ string.

    ``style`` is one of ``'surface'`` (filled triangles), ``'wireframe'`` (edges
    only), or ``'mesh'`` (filled triangles with wireframe overlay).
    """
    show_wireframe, show_faces = _style_to_show_flags(style)
    state = json.loads(mvsj)
    root = state["root"]
    volume = _find_volume_node(root, ref)
    if volume is None:
        raise ValueError(f"volume with ref '{ref}' not found in MVSJ")
    for repr_node in volume.get("children", []):
        if repr_node.get("kind") == "volume_representation":
            params = repr_node.setdefault("params", {})
            params["show_wireframe"] = show_wireframe
            params["show_faces"] = show_faces
    return json.dumps(state, separators=(",", ":"))
