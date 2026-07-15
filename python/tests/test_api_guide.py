"""Tests for the console's categorised API guide (`api`)."""

from pxviewer.api_guide import ApiGuide, build_groups


class _Api:
    """A tiny stand-in so the guide doesn't depend on the full LiveSession."""

    def select(self, expr):
        """Show a set of atoms in the viewer."""

    def color_by(self, attribute):
        """Colour atoms by a per-atom attribute."""

    def wobble(self):
        """An uncategorised extra method."""

    def _private(self):
        """Should never appear."""


# Point the category map at names our stand-in actually defines.
def _guide():
    return ApiGuide(_Api, groups=None)


def test_build_groups_categorises_and_collects_extras(monkeypatch):
    import pxviewer.api_guide as ag

    monkeypatch.setattr(ag, "_CATEGORIES", [("Selecting atoms", ["select", "color_by"])])
    groups = build_groups(_Api)

    cats = dict((c, [r[0] for r in rows]) for c, rows in groups)
    assert cats["Selecting atoms"] == ["select", "color_by"]
    # A public method not in any category lands in "Other"; privates never show.
    assert "wobble" in cats["Other"]
    assert all("_private" not in names for names in cats.values())


def test_apiguide_repr_and_find(monkeypatch):
    import pxviewer.api_guide as ag

    monkeypatch.setattr(ag, "_CATEGORIES", [("Selecting atoms", ["select", "color_by"])])
    guide = ApiGuide(_Api)

    text = repr(guide)
    assert "Selecting atoms" in text
    assert "session.select(…)" in text
    assert "Show a set of atoms" in text

    filtered = guide.find("colour")  # matches color_by's docstring
    ftext = repr(filtered)
    assert "color_by" in ftext and "select(" not in ftext
    assert "matching" in ftext

    # `api("colour")` is the same as `.find(...)`.
    assert repr(guide("colour")) == ftext


def test_apiguide_html(monkeypatch):
    import pxviewer.api_guide as ag

    monkeypatch.setattr(ag, "_CATEGORIES", [("Selecting atoms", ["select", "color_by"])])
    html = ApiGuide(_Api)._repr_html_()
    assert "<table" in html and "session.color_by" in html


def test_apiguide_covers_real_livesession():
    """Against the real class, common methods are categorised (not dumped in Other)."""
    from pxviewer.live import LiveSession

    groups = dict((c, [r[0] for r in rows]) for c, rows in build_groups(LiveSession))
    assert "select" in groups["Selecting atoms"]
    assert "color_by" in groups["Representations & colour"]
    assert "set_volume_color" in groups["Volumes"]
