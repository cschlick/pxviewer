"""Scene building: the MVSJ documents the Python API emits.

These are the functions a user calls to describe a view without running a session --
``create_view`` and friends return an MVSJ string, so every exercise here is a check on
the node tree that string parses to. The MVSJ spec is a contract with Mol*, and the tree
shape (download -> parse -> structure/volume -> representation) is the part of it that
breaks silently: a wrongly nested node still serialises, it just renders nothing.
"""

from __future__ import absolute_import, division, print_function

import json
import os
import sys

from libtbx.test_utils import approx_equal

from pxviewer.regression.tst_utils import have, skip, tmp_dir

if not have("numpy"):
    skip("numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer import (                               # noqa: E402
    Volume,
    create_example_view,
    create_view,
    create_volume_view,
    create_volume_view_from_data,
    read_volume,
    set_volume_color,
    set_volume_opacity,
    set_volume_style,
)


def volume_repr(mvsj):
    """The single volume_representation node in a one-volume scene.

    root -> download -> parse -> volume -> volume_representation. Spelling the walk out
    once keeps the exercises about what the representation says rather than how deep it
    sits.
    """
    state = json.loads(mvsj)
    return state["root"]["children"][0]["children"][0]["children"][0]["children"][0]


def volumes_by_ref(mvsj):
    """Every volume node in a multi-volume scene, keyed by its ref."""
    state = json.loads(mvsj)
    found = {}
    for download in state["root"]["children"]:
        if download.get("kind") != "download":
            continue
        for parse in download.get("children", []):
            for node in parse.get("children", []):
                if node.get("kind") == "volume":
                    found[node["ref"]] = node
    return found


# -- structures ---------------------------------------------------------------


def exercise_create_view_builds_a_download_node():
    """The simplest scene: a structure fetched from a URL."""
    url = "https://www.ebi.ac.uk/pdbe/entry-files/1cbs.bcif"
    state = json.loads(create_view(url))

    assert state["kind"] == "single"
    assert state["root"]["kind"] == "root"

    download = state["root"]["children"][0]
    assert download["kind"] == "download"
    assert download["params"]["url"] == url


def exercise_the_example_view_has_polymer_and_ligand():
    """The built-in demo scene splits the structure into named components."""
    state = json.loads(create_example_view())

    download = state["root"]["children"][0]
    structure = download["children"][0]["children"][0]

    selectors = [c["params"]["selector"] for c in structure["children"]]
    assert "polymer" in selectors
    assert "ligand" in selectors


# -- volumes from a URL -------------------------------------------------------


def exercise_create_volume_view_builds_a_map_node():
    mvsj = create_volume_view(
        "density.mrc",
        isosurface_value=3.0,
        isosurface_kind="absolute",
        color="red",
        opacity=0.5,
    )
    state = json.loads(mvsj)
    assert state["kind"] == "single"

    download = state["root"]["children"][0]
    assert download["kind"] == "download"
    assert download["params"]["url"] == "density.mrc"

    parse = download["children"][0]
    assert parse["kind"] == "parse"
    assert parse["params"]["format"] == "map"

    volume = parse["children"][0]
    assert volume["kind"] == "volume"

    rep = volume["children"][0]
    assert rep["kind"] == "volume_representation"
    assert rep["params"]["type"] == "isosurface"
    assert approx_equal(rep["params"]["absolute_isovalue"], 3.0)


def exercise_no_isosurface_leaves_the_value_out():
    """With no isovalue given the MVSJ omits it entirely, so Mol* picks its own default.

    Emitting a zero or a placeholder here would look the same in the file but render a
    surface at the wrong level.
    """
    rep = volume_repr(create_volume_view("density.mrc"))
    assert rep["kind"] == "volume_representation"
    assert rep["params"]["type"] == "isosurface"
    assert "absolute_isovalue" not in rep["params"]
    assert "relative_isovalue" not in rep["params"]


def exercise_several_volumes_are_independently_addressable():
    mvsj = create_volume_view(volumes=[
        Volume(url="a.mrc", ref="vol1", color="red", opacity=0.5),
        Volume(url="b.mrc", ref="vol2", color="blue", opacity=0.8,
               isosurface_value=2.0, isosurface_kind="absolute"),
    ])
    state = json.loads(mvsj)
    assert len(state["root"]["children"]) == 2
    assert [d["params"]["url"] for d in state["root"]["children"]] == ["a.mrc", "b.mrc"]

    by_ref = volumes_by_ref(mvsj)
    assert sorted(by_ref) == ["vol1", "vol2"]
    assert approx_equal(
        by_ref["vol2"]["children"][0]["params"]["absolute_isovalue"], 2.0)


def exercise_a_volume_position_becomes_a_transform():
    mvsj = create_volume_view("density.mrc", position=(10.0, 0.0, -5.0))
    volume = json.loads(mvsj)["root"]["children"][0]["children"][0]["children"][0]

    transforms = [c for c in volume.get("children", []) if c["kind"] == "transform"]
    assert len(transforms) == 1
    assert transforms[0]["params"]["translation"] == [10.0, 0.0, -5.0]


def exercise_every_mvs_volume_feature_round_trips():
    """channel_id, rotation, instances, clip and grid_slice all reach the MVSJ.

    These are the corners of the volume spec pxviewer exposes but rarely exercises, so
    one scene uses all of them at once.
    """
    mvsj = create_volume_view(volumes=[
        Volume(
            url="em.mrc", ref="em", format="map",
            isosurface_value=0.5, isosurface_kind="absolute",
            rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], rotation_center=[0, 0, 0],
            position=[1, 2, 3], instances=[{"translation": [4, 0, 0]}],
            color="blue", opacity=0.6,
            clip={"type": "sphere", "center": [0, 0, 0], "radius": 5.0},
        ),
        Volume(
            url="bcif", ref="slice", format="bcif", channel_id="2FO-FC",
            representation="grid_slice", grid_slice_dimension="z",
            grid_slice_index=0.5, grid_slice_index_kind="relative",
            color="green", focus=False,
        ),
    ])
    by_ref = volumes_by_ref(mvsj)

    em = by_ref["em"]
    assert em["params"].get("channel_id") is None       # a map has no named channel
    assert em["children"][0]["kind"] == "transform"
    assert em["children"][0]["params"]["rotation"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert em["children"][1]["kind"] == "instance"
    rep = em["children"][2]
    assert rep["params"]["type"] == "isosurface"
    clip = [c for c in rep.get("children", []) if c["kind"] == "clip"][0]
    assert clip["params"]["type"] == "sphere"

    sliced = by_ref["slice"]
    assert sliced["params"]["channel_id"] == "2FO-FC"
    assert sliced["children"][0]["params"]["type"] == "grid_slice"
    assert sliced["children"][0]["params"]["dimension"] == "z"
    assert approx_equal(sliced["children"][0]["params"]["relative_index"], 0.5)


# -- isosurface style ---------------------------------------------------------


def exercise_mesh_is_edges_only():
    """A 'mesh' isosurface is chickenwire: wireframe on, faces off.

    There is deliberately no faces+edges combination -- the two styles are exclusive.
    """
    rep = volume_repr(create_volume_view("density.mrc", style="mesh"))
    assert rep["params"]["show_wireframe"] is True
    assert rep["params"]["show_faces"] is False


def exercise_wireframe_is_a_legacy_alias_for_mesh():
    """Scenes and sessions written before the rename still say 'wireframe'."""
    rep = volume_repr(create_volume_view("density.mrc", style="wireframe"))
    assert rep["params"]["show_wireframe"] is True
    assert rep["params"]["show_faces"] is False


def exercise_set_volume_style_touches_one_volume():
    mvsj = create_volume_view(volumes=[
        Volume(url="a.mrc", ref="vol1", style="mesh"),
        Volume(url="b.mrc", ref="vol2", style="mesh"),
    ])
    by_ref = volumes_by_ref(set_volume_style(mvsj, "vol1", "surface"))

    rep1 = by_ref["vol1"]["children"][0]
    assert rep1["params"]["show_wireframe"] is False    # switched to a filled surface
    assert rep1["params"]["show_faces"] is True

    rep2 = by_ref["vol2"]["children"][0]
    assert rep2["params"]["show_wireframe"] is True     # still a mesh
    assert rep2["params"]["show_faces"] is False


def exercise_set_volume_color_and_opacity_by_ref():
    mvsj = create_volume_view(volumes=[
        Volume(url="a.mrc", ref="vol1", color="red"),
        Volume(url="b.mrc", ref="vol2", color="blue"),
    ])
    mvsj = set_volume_color(mvsj, "vol1", "green")
    mvsj = set_volume_opacity(mvsj, "vol2", 0.25)
    by_ref = volumes_by_ref(mvsj)

    rep1 = by_ref["vol1"]["children"][0]
    colors = [c for c in rep1["children"] if c["kind"] == "color"]
    assert colors[0]["params"]["color"] == "green"

    rep2 = by_ref["vol2"]["children"][0]
    opacities = [c for c in rep2["children"] if c["kind"] == "opacity"]
    assert approx_equal(opacities[0]["params"]["opacity"], 0.25)


# -- volumes written from an array --------------------------------------------


def exercise_create_volume_view_from_data_writes_both_files():
    """One call writes the map and the scene that points at it."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[5, 5, 5] = 10.0
    with tmp_dir() as path:
        mrc_path = os.path.join(path, "model.mrc")
        mvsj_path = os.path.join(path, "model.mvsj")

        mvsj = create_volume_view_from_data(
            data,
            mrc_path=mrc_path,
            mvsj_path=mvsj_path,
            write_kwargs={"voxel_size": 2.0},
            view_kwargs={"isosurface_value": 2.0, "isosurface_kind": "relative"},
        )

        assert os.path.exists(mrc_path)
        assert os.path.exists(mvsj_path)
        state = json.loads(mvsj)
        assert "model.mrc" in state["root"]["children"][0]["params"]["url"]


def exercise_written_origin_and_voxel_size_survive_the_ccp4_round_trip():
    """The map keeps the geometry it was written with, and the scene keeps its position.

    Origin and position are different things that both move the density -- the first is
    in the file's header, the second is a transform Mol* applies -- so they are checked
    apart from each other.
    """
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[4, 4, 4] = 10.0
    with tmp_dir() as path:
        mrc_path = os.path.join(path, "model.mrc")
        mvsj_path = os.path.join(path, "model.mvsj")

        mvsj = create_volume_view_from_data(
            data,
            mrc_path=mrc_path,
            mvsj_path=mvsj_path,
            voxel_size=2.0,
            origin=(4.0, 0.0, 0.0),      # grid-aligned; cctbx snaps to whole voxels
            position=(1.0, 2.0, 3.0),
        )

        assert os.path.exists(mrc_path)
        assert os.path.exists(mvsj_path)

        read = read_volume(mrc_path)
        assert approx_equal(read["voxel_size"], (2.0, 2.0, 2.0))
        assert approx_equal(read["origin"], (4.0, 0.0, 0.0))

        volume = json.loads(mvsj)["root"]["children"][0]["children"][0]["children"][0]
        transforms = [c for c in volume.get("children", []) if c["kind"] == "transform"]
        assert len(transforms) == 1
        assert transforms[0]["params"]["translation"] == [1.0, 2.0, 3.0]


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
