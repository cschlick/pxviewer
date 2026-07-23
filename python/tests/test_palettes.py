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


def test_suggested_colours_come_from_the_inventory_and_spread_by_hue():
    """The hand-pick swatches are drawn from the same bundled palettes the automatic
    defaults come from — hex, distinct, and spread around the hue circle so neighbouring
    swatches are not three versions of the same pink."""
    import colorsys

    from pxviewer.palettes import suggested_colours

    inventory = {c for group in load_palettes() for c in group}
    picks = suggested_colours(8)

    assert len(picks) == 8
    assert len(set(picks)) == 8                    # no duplicates
    assert all(c in inventory for c in picks)      # every one from the inventory
    assert all(c.startswith("#") for c in picks)   # hex, not CSS names

    def hue(c):
        r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        return colorsys.rgb_to_hsv(r, g, b)[0]

    hues = [hue(c) for c in picks]
    assert hues == sorted(hues)                    # walks the circle in order
    assert hues[-1] - hues[0] > 0.5                # and covers most of it


def test_asking_for_more_suggestions_than_exist_returns_them_all():
    from pxviewer.palettes import suggested_colours

    inventory = {c for group in load_palettes() for c in group}
    assert len(suggested_colours(10_000)) == len(inventory)


def test_different_sessions_get_different_colours():
    """No seed -> entropy: two fresh cyclers almost never open on the same colour."""
    firsts = {PaletteCycler().next_colour() for _ in range(20)}
    assert len(firsts) > 1  # not a fixed deterministic sequence
