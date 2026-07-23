"""The random default-colour cycler (pxviewer.palettes)."""

from pxviewer.palettes import PaletteCycler, load_palettes


def _palettes_as_sets():
    return [set(group) for group in load_palettes()]


def test_a_block_of_four_comes_from_one_random_group():
    """Each run of four objects draws from a single palette group; the fifth rolls a new
    one. That is the whole algorithm: pick a group, hand out random colours from it, and
    change group every four objects."""
    groups = _palettes_as_sets()
    cyc = PaletteCycler(seed=1)
    colours = [cyc.next_colour() for _ in range(12)]

    for i in range(3):
        block = colours[i * 4:(i + 1) * 4]
        assert any(set(block) <= g for g in groups), f"block {i} spans palettes: {block}"


def test_no_two_objects_in_a_row_share_a_colour():
    """A random pick could repeat the previous colour and make two objects identical; the
    cycler avoids the immediately-previous one so neighbours always differ."""
    cyc = PaletteCycler(seed=7)
    colours = [cyc.next_colour() for _ in range(40)]
    assert all(a != b for a, b in zip(colours, colours[1:]))


def test_every_colour_is_a_real_palette_colour():
    valid = {c for group in load_palettes() for c in group}
    cyc = PaletteCycler(seed=3)
    assert all(cyc.next_colour() in valid for _ in range(30))


def test_the_group_changes_across_a_session():
    """Over enough objects, more than one group is used — colours are not stuck on the
    first palette."""
    groups = _palettes_as_sets()
    cyc = PaletteCycler(seed=2)
    used = set()
    for _ in range(24):
        c = cyc.next_colour()
        used |= {i for i, g in enumerate(groups) if c in g}
    assert len(used) > 4  # several distinct palettes touched, not just one group of four


def test_different_sessions_get_different_colours():
    """No seed -> entropy: two fresh cyclers almost never open on the same colour."""
    firsts = {PaletteCycler().next_colour() for _ in range(20)}
    assert len(firsts) > 1  # not a fixed deterministic sequence
