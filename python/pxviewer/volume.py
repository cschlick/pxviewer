"""Volumetric data helpers for writing MRC/MAP files and building MVS scenes."""

import os
from typing import Literal

import mrcfile
import numpy as np

__all__ = [
    "write_volume",
    "read_volume",
    "create_volume_view",
    "create_volume_view_from_data",
]


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


def create_volume_view(
    volume_url: str,
    *,
    isosurface_value: float | None = None,
    isosurface_kind: Literal["absolute", "relative"] = "relative",
    color: str | None = "gold",
    opacity: float | None = 1.0,
    title: str | None = None,
) -> str:
    """Build an MVSJ scene that loads an MRC/MAP volume from a URL.

    The ``volume_url`` should be the URL (or relative path) that the Mol*
    frontend will use to fetch the volume. If the file is local, use the
    filename relative to the MVSJ file.
    """
    import molviewspec as mvs

    builder = mvs.create_builder()
    volume = builder.download(url=volume_url).parse(format="map").volume()

    repr_kwargs: dict = {"type": "isosurface"}
    if isosurface_value is not None:
        if isosurface_kind == "absolute":
            repr_kwargs["absolute_isovalue"] = isosurface_value
        elif isosurface_kind == "relative":
            repr_kwargs["relative_isovalue"] = isosurface_value
        else:
            raise ValueError(f"isosurface_kind must be 'absolute' or 'relative', got {isosurface_kind!r}")

    repr = volume.representation(**repr_kwargs)
    if color is not None:
        repr = repr.color(color=color)
    if opacity is not None:
        repr = repr.opacity(opacity=opacity)

    volume.focus()

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
    mvsj = create_volume_view(os.path.basename(str(mrc_path)), title=title, **view_kwargs)

    if mvsj_path is not None:
        with open(mvsj_path, "w") as f:
            f.write(mvsj)

    return mvsj
