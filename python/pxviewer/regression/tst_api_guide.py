"""The console's categorized API guide (``api``).

``build_groups`` introspects a class against the module's category map, so both halves
matter: the categorized methods, and the "Other" bucket that catches anything public but
unlisted -- that bucket is what stops a newly added API method from disappearing.
"""

from __future__ import absolute_import, division, print_function

import sys

from pxviewer.api_guide import ApiGuide, build_groups
from pxviewer.regression.tst_utils import have


class Stand_in(object):
    """A small class with one method in each of two real categories, plus an uncategorized
    one and a private one.

    ``LiveSession`` cannot show the interesting cases: it has no uncategorized public
    method to land in "Other". The pytest version replaced the module's ``_CATEGORIES``
    with a stub to get here; that is unnecessary, because the real map already lists
    ``select`` and ``color_by`` under two different headings, so a class defining exactly
    those two exercises the same paths against the shipping configuration.
    """

    def select(self, expr):
        """Show a set of atoms in the viewer."""

    def color_by(self, attribute):
        """Color atoms by a per-atom attribute."""

    def wobble(self):
        """An uncategorized extra method."""

    def _private(self):
        """Should never appear."""


def exercise_build_groups_categorizes_and_collects_extras():
    categories = dict((c, [r[0] for r in rows]) for c, rows in build_groups(Stand_in))

    assert categories["Selecting atoms"] == ["select"]
    assert categories["Representations & color"] == ["color_by"]
    # Public but unlisted lands in "Other"; privates never show at all.
    assert "wobble" in categories["Other"]
    assert all("_private" not in names for names in categories.values())


def exercise_a_category_with_no_matching_method_is_dropped():
    """An empty group is omitted rather than printed as a bare heading."""
    categories = [c for c, _ in build_groups(Stand_in)]
    assert "Measurements" not in categories       # the stand-in defines none of them
    assert "(nothing matched)" not in repr(ApiGuide(Stand_in))


def exercise_repr_lists_categories_signatures_and_docs():
    text = repr(ApiGuide(Stand_in))

    assert "Selecting atoms" in text
    assert "session.select(…)" in text            # takes an argument, so the ellipsis form
    assert "Show a set of atoms" in text


def exercise_find_filters_on_name_and_doc():
    guide = ApiGuide(Stand_in)
    text = repr(guide.find("color"))

    # color_by matches by name; select's doc says nothing about color, so it drops out.
    assert "color_by" in text
    assert "select(" not in text
    assert "matching" in text

    # `api("color")` is spelled differently but does the same thing.
    assert repr(guide("color")) == text


def exercise_find_with_no_matches_says_so():
    assert "(nothing matched)" in repr(ApiGuide(Stand_in).find("no-such-method"))


def exercise_html_rendering():
    """The console renders the guide as a table where the frontend supports it."""
    html = ApiGuide(Stand_in)._repr_html_()
    assert "<table" in html
    assert "session.color_by" in html


def exercise_the_real_livesession_is_mostly_categorized():
    """Against the shipping class the common methods are grouped, not dumped in Other.

    This is the exercise that notices when a method is renamed and the category map is
    not updated with it.
    """
    if not have("iotbx.data_manager"):
        # Returning rather than skip()ing: skip() ends the process, and everything above
        # is pure introspection that runs fine without cctbx.
        print("    (skipped: iotbx.data_manager not available)")
        return
    from pxviewer.live import LiveSession

    groups = dict((c, [r[0] for r in rows]) for c, rows in build_groups(LiveSession))

    assert "select" in groups["Selecting atoms"]
    assert "color_by" in groups["Representations & color"]
    assert "set_volume_color" in groups["Volumes"]


def exercise_the_full_guide_covers_both_objects_by_task_word():
    """``ApiGuide.full()`` is what the console binds as ``api`` — it has to cover
    ``app`` (AppCore) as well as ``session`` (LiveSession), with the right prefix
    on each, and a task-word search has to surface the capability an agent could
    not otherwise know exists (the set_volume_box carve, the clip spheres)."""
    if not have("iotbx.data_manager"):
        print("    (skipped: iotbx.data_manager not available)")
        return

    guide = ApiGuide.full()
    text = repr(guide)

    # Both prefixes, both surfaces.
    assert "session.select_by" in text
    assert "app.load_file" in text
    assert "Regions & clipping" in text

    # Task-word finds: the verb behind "clip a region" / "carve a huge map" /
    # "make it faster" does not have to be guessed.
    for term, expected in (
        ("clip", "app.set_volume_sphere"),
        ("clip", "app.set_model_clip"),
        ("box", "app.set_volume_box"),
        ("downsample", "app.set_volume_downsample"),
        ("mask", "app.set_volume_mask"),
        ("sigma", "app.set_volume_iso"),
    ):
        found = repr(guide.find(term))
        assert expected in found, f"find({term!r}) missed {expected}: {found}"

    # App rows must not print as session.* — the prefix is the call target.
    assert "session.set_volume_box" not in text


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
