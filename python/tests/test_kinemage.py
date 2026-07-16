"""Tests for the kinemage markup parser."""

from pxviewer.kinemage import parse_kinemage


def test_balllist_with_radius():
    # Real cbetadev output: a magenta ball with an r= radius (labels contain numbers).
    text = (
        "@balllist {CB dev Ball} color= magenta master= {Cbeta dev}\n"
        "{cb lys A  33   0.322 -103.46} r=0.322  38.432, 32.849, 8.581"
    )
    prims = parse_kinemage(text)
    assert prims == [{"kind": "balls", "name": "CB dev Ball", "color": [255, 0, 255],
                      "balls": [[[38.432, 32.849, 8.581], 0.322]]}]


def test_vectorlist_segments():
    # kin_vec style: {k} P start {k} L end -> one segment.
    text = (
        "@vectorlist {chain A} color= gold\n"
        "{a} P 1.0 2.0 3.0 {b} L 4.0 5.0 6.0"
    )
    prims = parse_kinemage(text)
    assert prims == [{"kind": "vectors", "name": "chain A", "color": [255, 215, 0],
                      "width": None,  # this list sets none, like MolProbity's rotamers
                      "segments": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]}]


def test_vector_width_is_kept():
    """Kinemage line width drives how thick we draw a vector: MolProbity uses width=4
    for its outlier markup and width=1 for CaBLAM's wheel outlines. A list that sets
    none reports None, and the renderer decides."""
    text = (
        "@vectorlist {bad Rama Ca} width= 4 color= green\n{a} P 0 0 0 {b} 1 1 1\n"
        "@vectorlist {cablam_wheels_lines} color=deadblack width= 1 alpha=0.75\n{c} P 0 0 0 {d} 2 2 2\n"
        "@vectorlist {chain A} color= gold\n{e} P 0 0 0 {f} 3 3 3"
    )
    assert [(p["name"], p["width"]) for p in parse_kinemage(text)] == [
        ("bad Rama Ca", 4), ("cablam_wheels_lines", 1), ("chain A", None),
    ]


def test_trianglelist_strip():
    # P starts the strip; 4 points -> 2 triangles.
    text = (
        "@trianglelist {t} color= yellow\n"
        "{a} P X 0 0 0 {b} 1 0 0 {c} 1 1 0 {d} 0 1 0"
    )
    prims = parse_kinemage(text)
    assert prims[0]["kind"] == "triangles"
    assert prims[0]["triangles"] == [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ]


def test_dotlist_points():
    text = "@dotlist {s} color= white\n{x} 1 2 3 {y} 4 5 6"
    prims = parse_kinemage(text)
    assert prims == [{"kind": "dots", "name": "s", "color": [255, 255, 255],
                      "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}]


def test_list_name_is_kept():
    """Callers pick lists out by name — CaBLAM drops its wheel outlines that way."""
    text = (
        "@trianglelist {cablam_wheels} alpha=0.75\n"
        "{} P X magenta 0 0 0\n{} magenta 1 0 0\n{} magenta 1 1 0\n"
        "@vectorlist {cablam_wheels_lines} color=deadblack width= 1\n{} P 0 0 0 {} 1 1 1"
    )
    assert [p["name"] for p in parse_kinemage(text)] == ["cablam_wheels", "cablam_wheels_lines"]


def test_per_point_colour_splits_primitives():
    """CaBLAM's wheels carry a colour per point ({} P X magenta x y z) instead of on
    the list header (which has none) — score-coded magenta/purple wedges. Each colour
    becomes its own primitive; ignoring them rendered the wheels white."""
    text = (
        "@trianglelist {cablam_wheels} alpha=0.75\n"
        "{} P X magenta 0 0 0\n{} magenta 1 0 0\n{} magenta 1 1 0\n"
        "{} P X purple 0 0 0\n{} purple 0 1 0\n{} purple 0 0 1"
    )
    prims = parse_kinemage(text)
    assert {tuple(p["color"]) for p in prims} == {(255, 0, 255), (160, 32, 240)}
    assert all(p["kind"] == "triangles" and len(p["triangles"]) == 1 for p in prims)


def test_per_point_colour_persists():
    """A per-point colour applies to following points until the next colour."""
    text = "@vectorlist {v} color= green\n{a} P red 0 0 0 {b} 1 1 1 {c} 2 2 2"
    prims = parse_kinemage(text)
    assert len(prims) == 1 and prims[0]["color"] == [255, 0, 0]  # red, not the list's green
    assert len(prims[0]["segments"]) == 2


def test_unknown_color_and_empty():
    assert parse_kinemage("") == []
    assert parse_kinemage("@vectorlist {v} color= chartreuse\n{a} P 0 0 0 {b} 1 1 1") == [
        {"kind": "vectors", "name": "v", "color": [255, 255, 255], "width": None,
         "segments": [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]}
    ]
