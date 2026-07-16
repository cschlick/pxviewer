"""Tests for the kinemage markup parser."""

from pxviewer.kinemage import parse_kinemage


def test_balllist_with_radius():
    # Real cbetadev output: a magenta ball with an r= radius (labels contain numbers).
    text = (
        "@balllist {CB dev Ball} color= magenta master= {Cbeta dev}\n"
        "{cb lys A  33   0.322 -103.46} r=0.322  38.432, 32.849, 8.581"
    )
    prims = parse_kinemage(text)
    assert prims == [{"kind": "balls", "color": [255, 0, 255],
                      "balls": [[[38.432, 32.849, 8.581], 0.322]]}]


def test_vectorlist_segments():
    # kin_vec style: {k} P start {k} L end -> one segment.
    text = (
        "@vectorlist {chain A} color= gold\n"
        "{a} P 1.0 2.0 3.0 {b} L 4.0 5.0 6.0"
    )
    prims = parse_kinemage(text)
    assert prims == [{"kind": "vectors", "color": [255, 215, 0],
                      "segments": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]}]


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
    assert prims == [{"kind": "dots", "color": [255, 255, 255],
                      "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}]


def test_unknown_color_and_empty():
    assert parse_kinemage("") == []
    assert parse_kinemage("@vectorlist {v} color= chartreuse\n{a} P 0 0 0 {b} 1 1 1") == [
        {"kind": "vectors", "color": [255, 255, 255], "segments": [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]}
    ]
