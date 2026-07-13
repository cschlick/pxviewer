"""Tests and examples for the pxviewer Python API."""

import json

import pytest

from pxviewer import Atom, create_example_view, create_fragment_view, create_view


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
