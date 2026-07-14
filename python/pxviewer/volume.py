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

VolumeFormatT = Literal["map", "dx", "dxbin", "bcif"]
VolumeRepresentationT = Literal["isosurface", "grid_slice"]


@dataclasses.dataclass
class Volume:
    """A single volume and how it should be rendered in an MVSJ scene.

    This maps onto the MVS ``volume``/``volume_representation`` tree nodes
    supported by MolViewSpec. Most fields are optional and omitted from the
    MVSJ JSON when not set.
    """

    url: str
    ref: str | None = None
    format: VolumeFormatT = "map"
    channel_id: str | None = None
    isosurface_value: float | None = None
    isosurface_kind: Literal["absolute", "relative"] = "relative"
    representation: VolumeRepresentationT = "isosurface"
    grid_slice_dimension: Literal["x", "y", "z"] | None = None
    grid_slice_index: float | None = None
    grid_slice_index_kind: Literal["absolute", "relative"] = "relative"
    color: str | None = "gold"
    opacity: float | None = 1.0
    style: VolumeStyle | None = "surface"
    position: tuple[float, float, float] | None = None
    rotation: tuple[float, ...] | None = None
    rotation_center: tuple[float, float, float] | str | None = None
    matrix: tuple[float, ...] | None = None
    instances: list[dict] | None = None
    clip: dict | None = None
    focus: bool = True


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

    mvs_volume = builder.download(url=volume.url).parse(format=volume.format).volume(
        ref=ref, channel_id=volume.channel_id
    )

    # MVS transform node (rotation, translation, matrix)
    transform_args = _make_transform_args(volume)
    if transform_args:
        mvs_volume = mvs_volume.transform(**transform_args)

    # MVS instance nodes
    if volume.instances:
        for inst in volume.instances:
            mvs_volume = mvs_volume.instance(**inst)

    # Build volume_representation kwargs
    repr_kwargs: dict = {"type": volume.representation}
    if volume.isosurface_value is not None:
        if volume.isosurface_kind == "absolute":
            repr_kwargs["absolute_isovalue"] = volume.isosurface_value
        elif volume.isosurface_kind == "relative":
            repr_kwargs["relative_isovalue"] = volume.isosurface_value
        else:
            raise ValueError(f"isosurface_kind must be 'absolute' or 'relative', got {volume.isosurface_kind!r}")

    if volume.representation == "isosurface" and volume.style is not None:
        if volume.style == "surface":
            repr_kwargs["show_wireframe"] = False
            repr_kwargs["show_faces"] = True
        elif volume.style == "wireframe":
            repr_kwargs["show_wireframe"] = True
            repr_kwargs["show_faces"] = False
        elif volume.style == "mesh":
            repr_kwargs["show_wireframe"] = True
            repr_kwargs["show_faces"] = True
        else:
            raise ValueError(f"style must be 'surface', 'wireframe' or 'mesh', got {volume.style!r}")

    if volume.representation == "grid_slice":
        if volume.grid_slice_dimension is not None:
            repr_kwargs["dimension"] = volume.grid_slice_dimension
        if volume.grid_slice_index is not None:
            if volume.grid_slice_index_kind == "absolute":
                repr_kwargs["absolute_index"] = int(volume.grid_slice_index)
            elif volume.grid_slice_index_kind == "relative":
                repr_kwargs["relative_index"] = float(volume.grid_slice_index)
            else:
                raise ValueError(f"grid_slice_index_kind must be 'absolute' or 'relative', got {volume.grid_slice_index_kind!r}")

    repr = mvs_volume.representation(**repr_kwargs, ref=f"{ref}-repr")

    if volume.color is not None:
        repr = repr.color(color=volume.color)
    if volume.opacity is not None:
        repr = repr.opacity(opacity=volume.opacity)
    if volume.clip is not None:
        repr = repr.clip(**volume.clip)

    if volume.focus:
        mvs_volume.focus()

    return ref


def _make_transform_args(volume: Volume) -> dict:
    """Build MVS transform/instance kwargs from Volume fields."""
    args: dict = {}
    if volume.matrix is not None:
        if volume.rotation is not None or volume.position is not None or volume.rotation_center is not None:
            raise ValueError("matrix cannot be used together with rotation, position or rotation_center")
        args["matrix"] = volume.matrix
    else:
        if volume.rotation is not None:
            args["rotation"] = volume.rotation
        if volume.position is not None:
            args["translation"] = volume.position
        if volume.rotation_center is not None:
            args["rotation_center"] = volume.rotation_center
    return args


def create_volume_view(
    volume_url: str | None = None,
    *,
    volumes: List[str | Volume | dict] | None = None,
    format: VolumeFormatT = "map",
    channel_id: str | None = None,
    isosurface_value: float | None = None,
    isosurface_kind: Literal["absolute", "relative"] = "relative",
    representation: VolumeRepresentationT = "isosurface",
    grid_slice_dimension: Literal["x", "y", "z"] | None = None,
    grid_slice_index: float | None = None,
    grid_slice_index_kind: Literal["absolute", "relative"] = "relative",
    color: str | None = "gold",
    opacity: float | None = 1.0,
    style: VolumeStyle | None = "surface",
    position: tuple[float, float, float] | None = None,
    rotation: tuple[float, ...] | None = None,
    rotation_center: tuple[float, float, float] | str | None = None,
    matrix: tuple[float, ...] | None = None,
    instances: list[dict] | None = None,
    clip: dict | None = None,
    focus: bool = True,
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
                format=format,
                channel_id=channel_id,
                isosurface_value=isosurface_value,
                isosurface_kind=isosurface_kind,
                representation=representation,
                grid_slice_dimension=grid_slice_dimension,
                grid_slice_index=grid_slice_index,
                grid_slice_index_kind=grid_slice_index_kind,
                color=color,
                opacity=opacity,
                style=style,
                position=position,
                rotation=rotation,
                rotation_center=rotation_center,
                matrix=matrix,
                instances=instances,
                clip=clip,
                focus=focus,
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
    voxel_size: float | tuple[float, float, float] | None = None,
    origin: tuple[float, float, float] | None = None,
    origin_units: Literal["angstrom", "grid"] = "angstrom",
    position: tuple[float, float, float] | None = None,
) -> str:
    """Write a volume to MRC and return an MVSJ scene that loads it.

    The MVSJ uses the MRC filename as a relative URL, so both files should be
    served from the same directory.

    ``voxel_size`` and ``origin`` set the MRC header coordinate system (in
    Angstroms by default). ``position`` applies an MVS transform translation to
    the volume after loading.

    Any additional ``create_volume_view`` keyword arguments (``rotation``,
    ``matrix``, ``clip``, ``grid_slice_dimension``, etc.) can be passed via
    ``view_kwargs``.
    """
    write_kwargs = dict(write_kwargs or {})
    view_kwargs = dict(view_kwargs or {})

    if voxel_size is not None and "voxel_size" not in write_kwargs:
        write_kwargs["voxel_size"] = voxel_size
    if origin is not None and "origin" not in write_kwargs:
        write_kwargs["origin"] = origin
        write_kwargs["origin_units"] = origin_units
    if position is not None and "position" not in view_kwargs:
        view_kwargs["position"] = position

    # create_volume_view_from_data always writes MRC/MAP, so parsing must be map.
    view_kwargs["format"] = "map"
    view_kwargs.pop("channel_id", None)

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
