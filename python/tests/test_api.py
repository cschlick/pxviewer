"""Tests and examples for the pxviewer Python API."""

import json

import pytest

from pxviewer import (
    Atom,
    create_example_view,
    create_fragment_view,
    create_view,
    create_volume_view,
    create_volume_view_from_data,
)


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


def test_create_fragment_view_round_trip(tmp_path):
    """Example of writing a small atom model and building an MVSJ for it."""
    atoms = [
        Atom(id=1, element="N", name="N", resname="ALA", resseq=1, chain="A", x=0.0, y=0.0, z=0.0),
        Atom(id=2, element="C", name="CA", resname="ALA", resseq=1, chain="A", x=1.5, y=0.0, z=0.0),
    ]
    bcif_path = tmp_path / "model.bcif"
    mvsj_path = tmp_path / "model.mvsj"

    mvsj = create_fragment_view(
        atoms,
        bcif_path=bcif_path,
        mvsj_path=mvsj_path,
        title="Fragment test",
    )

    assert bcif_path.exists()
    assert mvsj_path.exists()

    state = json.loads(mvsj)
    assert state["metadata"]["title"] == "Fragment test"
    assert "model.bcif" in state["root"]["children"][0]["params"]["url"]


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
