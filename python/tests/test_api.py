"""Tests and examples for the pxviewer Python API."""

import json

import pytest

from pxviewer import (
    Volume,
    create_example_view,
    create_view,
    create_volume_view,
    create_volume_view_from_data,
    set_volume_color,
    set_volume_opacity,
    set_volume_style,
)
import numpy as np


def test_create_view_builds_download_node():
    """The simplest example: create an MVSJ scene from a structure URL."""
    url = "https://www.ebi.ac.uk/pdbe/entry-files/1cbs.bcif"
    mvsj = create_view(url)

    state = json.loads(mvsj)
    assert state["kind"] == "single"
    root = state["root"]
    assert root["kind"] == "root"

    download = root["children"][0]
    assert download["kind"] == "download"
    assert download["params"]["url"] == url


def test_create_example_view_has_polymer_and_ligand():
    """Example of the built-in demo scene."""
    mvsj = create_example_view()
    state = json.loads(mvsj)
    root = state["root"]

    # Walk into the structure node and collect component selectors
    download = root["children"][0]
    parse_node = download["children"][0]
    structure = parse_node["children"][0]

    selectors = [c["params"]["selector"] for c in structure["children"]]
    assert "polymer" in selectors
    assert "ligand" in selectors


def test_create_volume_view_builds_map_node():
    """Build an MVSJ scene that loads a volume from a URL."""
    mvsj = create_volume_view(
        "density.mrc",
        isosurface_value=3.0,
        isosurface_kind="absolute",
        color="red",
        opacity=0.5,
    )

    state = json.loads(mvsj)
    assert state["kind"] == "single"
    root = state["root"]
    download = root["children"][0]
    assert download["kind"] == "download"
    assert download["params"]["url"] == "density.mrc"

    parse_node = download["children"][0]
    assert parse_node["kind"] == "parse"
    assert parse_node["params"]["format"] == "map"

    volume = parse_node["children"][0]
    assert volume["kind"] == "volume"

    repr = volume["children"][0]
    assert repr["kind"] == "volume_representation"
    assert repr["params"]["type"] == "isosurface"
    assert repr["params"]["absolute_isovalue"] == pytest.approx(3.0)


def test_create_volume_view_from_data(tmp_path):
    """Write a volume and MVSJ in one call."""
    import numpy as np

    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[5, 5, 5] = 10.0
    mrc_path = tmp_path / "model.mrc"
    mvsj_path = tmp_path / "model.mvsj"

    mvsj = create_volume_view_from_data(
        data,
        mrc_path=mrc_path,
        mvsj_path=mvsj_path,
        write_kwargs={"voxel_size": 2.0},
        view_kwargs={"isosurface_value": 2.0, "isosurface_kind": "relative"},
    )

    assert mrc_path.exists()
    assert mvsj_path.exists()

    state = json.loads(mvsj)
    assert "model.mrc" in state["root"]["children"][0]["params"]["url"]


def test_create_volume_view_default_isosurface():
    """When no isosurface is supplied, the MVSJ omits the value and Mol* uses its default."""
    mvsj = create_volume_view("density.mrc")
    state = json.loads(mvsj)
    repr = state["root"]["children"][0]["children"][0]["children"][0]["children"][0]
    assert repr["kind"] == "volume_representation"
    assert repr["params"]["type"] == "isosurface"
    assert "absolute_isovalue" not in repr["params"]
    assert "relative_isovalue" not in repr["params"]


def test_create_volume_view_multiple_volumes():
    """Build an MVSJ with several independently addressable volumes."""
    mvsj = create_volume_view(
        volumes=[
            Volume(url="a.mrc", ref="vol1", color="red", opacity=0.5),
            Volume(url="b.mrc", ref="vol2", color="blue", opacity=0.8, isosurface_value=2.0, isosurface_kind="absolute"),
        ]
    )
    state = json.loads(mvsj)
    root = state["root"]
    assert len(root["children"]) == 2

    urls = [d["params"]["url"] for d in root["children"]]
    assert urls == ["a.mrc", "b.mrc"]

    volumes = [d["children"][0]["children"][0] for d in root["children"]]
    assert [v["ref"] for v in volumes] == ["vol1", "vol2"]

    repr2 = volumes[1]["children"][0]
    assert repr2["params"]["absolute_isovalue"] == pytest.approx(2.0)


def test_set_volume_color_and_opacity():
    """Update color and opacity of a specific volume by ref."""
    mvsj = create_volume_view(
        volumes=[
            Volume(url="a.mrc", ref="vol1", color="red"),
            Volume(url="b.mrc", ref="vol2", color="blue"),
        ]
    )

    mvsj = set_volume_color(mvsj, "vol1", "green")
    mvsj = set_volume_opacity(mvsj, "vol2", 0.25)

    state = json.loads(mvsj)
    volumes = [d["children"][0]["children"][0] for d in state["root"]["children"]]
    by_ref = {v["ref"]: v for v in volumes}

    repr1 = by_ref["vol1"]["children"][0]
    color_nodes = [c for c in repr1["children"] if c["kind"] == "color"]
    assert color_nodes[0]["params"]["color"] == "green"

    repr2 = by_ref["vol2"]["children"][0]
    opacity_nodes = [c for c in repr2["children"] if c["kind"] == "opacity"]
    assert opacity_nodes[0]["params"]["opacity"] == pytest.approx(0.25)


def test_create_volume_view_style():
    """Build an MVSJ with a wireframe isosurface style."""
    mvsj = create_volume_view("density.mrc", style="wireframe")
    state = json.loads(mvsj)
    repr = state["root"]["children"][0]["children"][0]["children"][0]["children"][0]
    assert repr["params"]["show_wireframe"] is True
    assert repr["params"]["show_faces"] is False


def test_set_volume_style():
    """Update the isosurface style of a specific volume by ref."""
    mvsj = create_volume_view(
        volumes=[
            Volume(url="a.mrc", ref="vol1", style="surface"),
            Volume(url="b.mrc", ref="vol2", style="wireframe"),
        ]
    )

    mvsj = set_volume_style(mvsj, "vol1", "mesh")

    state = json.loads(mvsj)
    volumes = [d["children"][0]["children"][0] for d in state["root"]["children"]]
    by_ref = {v["ref"]: v for v in volumes}

    repr1 = by_ref["vol1"]["children"][0]
    assert repr1["params"]["show_wireframe"] is True
    assert repr1["params"]["show_faces"] is True

    repr2 = by_ref["vol2"]["children"][0]
    assert repr2["params"]["show_wireframe"] is True
    assert repr2["params"]["show_faces"] is False


def test_create_volume_view_position():
    """Build an MVSJ with a volume transform position."""
    mvsj = create_volume_view("density.mrc", position=(10.0, 0.0, -5.0))
    state = json.loads(mvsj)
    volume = state["root"]["children"][0]["children"][0]["children"][0]
    transforms = [c for c in volume.get("children", []) if c["kind"] == "transform"]
    assert len(transforms) == 1
    assert transforms[0]["params"]["translation"] == [10.0, 0.0, -5.0]


def test_create_volume_view_from_data_origin_and_position(tmp_path):
    """Write an MRC with origin/voxel_size and build an MVSJ with a position transform."""
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[4, 4, 4] = 10.0
    mrc_path = tmp_path / "model.mrc"
    mvsj_path = tmp_path / "model.mvsj"

    mvsj = create_volume_view_from_data(
        data,
        mrc_path=mrc_path,
        mvsj_path=mvsj_path,
        voxel_size=2.0,
        origin=(4.0, 0.0, 0.0),  # grid-aligned (cctbx snaps origin to whole voxels)
        position=(1.0, 2.0, 3.0),
    )

    assert mrc_path.exists()
    assert mvsj_path.exists()

    from pxviewer import read_volume
    read = read_volume(mrc_path)
    assert read["voxel_size"] == pytest.approx((2.0, 2.0, 2.0))
    assert read["origin"] == pytest.approx((4.0, 0.0, 0.0))

    state = json.loads(mvsj)
    volume = state["root"]["children"][0]["children"][0]["children"][0]
    transforms = [c for c in volume.get("children", []) if c["kind"] == "transform"]
    assert len(transforms) == 1
    assert transforms[0]["params"]["translation"] == [1.0, 2.0, 3.0]


def test_create_volume_view_all_mvs_features():
    """Use Volume to emit channel_id, rotation, matrix, instances, clip and grid_slice."""
    mvsj = create_volume_view(
        volumes=[
            Volume(
                url="em.mrc",
                ref="em",
                format="map",
                isosurface_value=0.5,
                isosurface_kind="absolute",
                rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                rotation_center=[0, 0, 0],
                position=[1, 2, 3],
                instances=[{"translation": [4, 0, 0]}],
                color="blue",
                opacity=0.6,
                clip={"type": "sphere", "center": [0, 0, 0], "radius": 5.0},
            ),
            Volume(
                url="bcif",
                ref="slice",
                format="bcif",
                channel_id="2FO-FC",
                representation="grid_slice",
                grid_slice_dimension="z",
                grid_slice_index=0.5,
                grid_slice_index_kind="relative",
                color="green",
                focus=False,
            ),
        ]
    )
    state = json.loads(mvsj)

    by_ref = {}
    for download in state["root"]["children"]:
        if download.get("kind") != "download":
            continue
        for parse in download.get("children", []):
            for node in parse.get("children", []):
                if node.get("kind") == "volume":
                    by_ref[node["ref"]] = node
    assert by_ref["em"]["params"].get("channel_id") is None
    assert by_ref["em"]["children"][0]["kind"] == "transform"
    assert by_ref["em"]["children"][0]["params"]["rotation"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert by_ref["em"]["children"][1]["kind"] == "instance"
    repr_node = by_ref["em"]["children"][2]
    assert repr_node["params"]["type"] == "isosurface"
    clip_node = [c for c in repr_node.get("children", []) if c["kind"] == "clip"][0]
    assert clip_node["params"]["type"] == "sphere"

    assert by_ref["slice"]["params"]["channel_id"] == "2FO-FC"
    assert by_ref["slice"]["children"][0]["params"]["type"] == "grid_slice"
    assert by_ref["slice"]["children"][0]["params"]["dimension"] == "z"
    assert by_ref["slice"]["children"][0]["params"]["relative_index"] == 0.5
