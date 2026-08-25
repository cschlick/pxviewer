"""The hotspot field deposits every atom of an event at the event's severity.

``field.compute_field`` normalizes each event **per atom**: with
``s_a = sum_b exp(-d_ab^2 / 2 sigma^2)`` computed at each atom and ``w_a = severity / s_a``,
the reconstructed value at atom ``a`` is ``severity`` whether the footprint is a tight
cluster, a lone atom, or a mix of both. That is the property the whole map rests on --
it is what lets a lone severity-1.0 outlier peak at exactly 1.0, so the field can be read
on an absolute domain with 1.0 meaning "the community outlier cut".

The alternative -- one divisor for the whole event, taken from its densest part -- lets
the tight part of a footprint set the weight for all of it, so a spread event draws its
isolated atoms at a fraction of severity. Measured at 0.31 against 0.98 for a cluster of
eight with two atoms 12 A away; those atoms fall under the 0.5 display threshold and
vanish. :func:`exercise_a_spread_event_draws_all_of_itself` is that case, and fails
against the old rule.

Standalone by design: this directory is meant to be separable from pxviewer again, so
nothing here imports from it. The only setup is putting ``hotspots/`` on the path, which
is how the scripts there import each other.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                "hotspots"))

try:
    import numpy  # noqa: F401  (field.py needs it)
    from scipy.ndimage import gaussian_filter  # noqa: F401
except ImportError as exc:                                    # pragma: no cover
    print("  skipping: %s" % exc)
    print("OK")
    sys.exit(0)

from events import Event                                       # noqa: E402
from field import compute_field                                # noqa: E402

#: Sampling is trilinear off a 1 A grid, so a reconstructed value carries ~1e-4 of
#: interpolation error. Tight enough that a real normalization change cannot hide in it.
EXACT = 2e-3

#: The viewer hides field values below this, so an atom under it is invisible however
#: severe its event. The regression this file guards put spread atoms at 0.31.
DISPLAY_THRESHOLD = 0.5


def cube(origin=(0.0, 0.0, 0.0), step=1.5):
    """Eight atoms on a small cube -- a compact footprint, like a rotamer's core."""
    ox, oy, oz = origin
    return [(ox + dx * step, oy + dy * step, oz + dz * step)
            for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


def exercise_a_lone_point_event_peaks_at_its_severity():
    """1.0 in, 1.0 out: the anchor that makes the field readable on an absolute domain."""
    p = (0.0, 0.0, 0.0)
    for severity in (0.5, 1.0, 2.5):
        f = compute_field([Event("rama", severity, [p])])
        assert abs(f.sample(p) - severity) < EXACT, (
            "lone severity %.1f event read back %.4f" % (severity, f.sample(p)))
    assert compute_field([Event("rama", 1.0, [p])]).reference_level == 1.0


def exercise_a_uniform_cluster_reads_severity_at_every_atom():
    atoms = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (3.0, 3.0, 0.0)]
    f = compute_field([Event("rama", 1.0, atoms)])
    for a in atoms:
        assert abs(f.sample(a) - 1.0) < EXACT, "uniform cluster atom read %.4f" % f.sample(a)


def exercise_a_spread_event_draws_all_of_itself():
    """The regression: isolated atoms of a spread footprint must not dim.

    A cluster of eight with two atoms 12 A away -- the shape that cost the corpus its
    recall. Under a single whole-event divisor the far atoms read 0.31 and disappear
    under the display threshold; per-atom normalization puts them at ~0.96.
    """
    near = cube()
    far = [(12.0, 0.0, 0.0), (12.0, 1.5, 0.0)]
    f = compute_field([Event("rota", 1.0, near + far)])

    for a in far:
        v = f.sample(a)
        assert v > DISPLAY_THRESHOLD, (
            "isolated atom at %.4f is invisible (the pre-fix behaviour was 0.31)" % v)
        assert v > 0.9, "isolated atom dimmed to %.4f" % v
    for a in near:
        assert f.sample(a) > 0.8, "clustered atom dimmed to %.4f" % f.sample(a)

    # The point of the fix: neither part of the footprint is drawn at the other's expense.
    spread = min(f.sample(a) for a in far) / max(f.sample(a) for a in near)
    assert 0.8 < spread < 1.25, "the two halves disagree by %.2fx" % spread


def exercise_severity_does_not_grow_with_footprint_size():
    """Rule 6: a big event covers more volume at its severity, it does not read brighter."""
    small = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
    large = [(1.5 * i, 0.0, 0.0) for i in range(10)]
    for atoms in (small, large):
        f = compute_field([Event("rama", 1.0, atoms)])
        vals = [f.sample(a) for a in atoms]
        assert max(vals) < 1.15, "footprint of %d reads up to %.4f" % (len(atoms), max(vals))
        assert min(vals) > 0.8, "footprint of %d reads down to %.4f" % (len(atoms), min(vals))


def exercise_coincidence_is_the_signal():
    """Separate events add; one event, however spread, does not reach that."""
    p = (0.0, 0.0, 0.0)
    q = (1.0, 0.0, 0.0)

    together = compute_field([Event("rama", 1.0, [p]), Event("rota", 1.0, [p])])
    assert abs(together.sample(p) - 2.0) < EXACT, (
        "two coincident 1.0 events read %.4f" % together.sample(p))

    near = compute_field([Event("rama", 1.0, [p]), Event("rota", 1.0, [q])])
    assert near.sample(p) > 1.5, "events 1 A apart read %.4f" % near.sample(p)

    # A lone multi-atom event overshoots between its atoms, but nowhere near coincidence:
    # that gap is what lets "above 1.0" be read as coincidence rather than footprint shape.
    lone = compute_field([Event("rama", 1.0, cube())])
    assert float(lone.data.max()) < float(together.data.max()), (
        "a lone event peaks at %.4f, as high as real coincidence" % lone.data.max())


def exercise_an_event_that_deposits_nothing_is_skipped():
    """Zero severity and empty footprints drop out without disturbing the rest."""
    p = (0.0, 0.0, 0.0)
    q = (5.0, 0.0, 0.0)
    real = [Event("rama", 1.0, [p])]
    padded = [Event("rama", 1.0, [p]), Event("rota", 0.0, [q]), Event("clash", 1.0, [])]

    a = compute_field(real, padding=12.0)
    b = compute_field(padded, padding=12.0)
    # Compared by sampling rather than array equality: a skipped event still widens the
    # bounding box (_bounding_box takes every atom, severity or not), so the grids differ
    # in extent while the field itself must not.
    probes = [p, q, (2.5, 0.0, 0.0), (0.0, 2.0, 1.0)]
    for probe in probes:
        assert abs(a.sample(probe) - b.sample(probe)) < 1e-9, (
            "a skipped event changed the field at %r: %.6f vs %.6f"
            % (probe, a.sample(probe), b.sample(probe)))
    assert a.sample(q) < 0.1, "probe point is not far enough to be a real check"


def exercise_sampling_outside_the_box_is_zero():
    f = compute_field([Event("rama", 1.0, [(0.0, 0.0, 0.0)])])
    assert f.sample((500.0, 500.0, 500.0)) == 0.0


# --- real footprints -----------------------------------------------------------------
#
# Extracted from 1UBQ via events.extract_all, so these are the shapes the field actually
# sees rather than synthetic lattices. Values are at the production defaults
# (sigma=2.0, spacing=1.0, severity=1.0) and are what the current code produces --
# recorded, not derived, so a change in the deposit rule has to restate them.
#
# Two things these show that grid-aligned synthetic footprints hide:
#
#   * "every atom reconstructs to severity" is exact only for the idealized cases the
#     rule is derived from -- a lone atom, or a cluster whose atoms all have the same
#     self-overlap. A real footprint's atoms have differing overlaps, so an atom in a
#     sparser part of the chain reads a weighted average of its neighbours' weights:
#     0.72 to 0.89 here, converging to 0.77-0.93 as the grid is refined rather than to
#     1.0. That does not weaken the fix -- the point is clearing the display threshold,
#     and the old rule put the same atoms at 0.60 -- but the property is a limit, not
#     an identity.
#   * overshoot is grid-dependent, growing from ~7% at 1.0 A spacing to ~15% at 0.25 A
#     as the peak between atoms resolves. Any figure quoted for it is only meaningful
#     with a spacing attached.
#
REAL_FOOTPRINTS = {
    "rama": [(26.335, 27.77, 3.258), (26.85, 29.021, 3.898),
             (26.1, 29.253, 5.202), (24.865, 29.024, 5.33)],
    "cablam": [(26.849, 29.656, 6.217), (26.235, 30.058, 7.497),
               (27.906, 31.711, 7.264)],
    "rota2": [(28.523, 15.82, 8.182), (28.946, 16.445, 6.967)],
    "rota4": [(25.112, 24.88, 3.649), (25.353, 24.86, 5.134),
              (23.93, 23.959, 5.904), (24.447, 23.984, 7.62)],
    "rota8": [(22.085, 22.254, 16.581), (22.945, 21.951, 17.785),
              (22.437, 22.157, 19.065), (24.272, 21.544, 17.644),
              (23.204, 21.907, 20.192), (25.052, 21.285, 18.776),
              (24.517, 21.47, 20.03), (25.248, 21.302, 21.191)],
}

#: name -> (peak of the field, lowest value at any of the event's own atoms).
REAL_EXPECTED = {
    "rama": (1.0868, 0.8483),
    "cablam": (1.0696, 0.8524),
    "rota2": (0.9732, 0.8943),
    "rota4": (1.0943, 0.8308),
    "rota8": (1.1757, 0.7172),
}

#: What the superseded whole-event divisor produced for the same footprints. Pinned so
#: the numbers above cannot drift back toward it unnoticed. rota2 is absent: a symmetric
#: two-atom footprint has one distinct self-overlap, so both rules agree there.
PRE_FIX_MIN_ATOM = {
    "rama": 0.7778, "cablam": 0.7844, "rota4": 0.7243, "rota8": 0.5988,
}

#: Recorded values are deterministic for a given footprint; this leaves room for
#: floating-point drift while staying far inside the ~0.07-0.11 that the deposit rule
#: moves these by.
PINNED = 0.01


def exercise_real_footprints_hold_their_measured_values():
    for name, atoms in sorted(REAL_FOOTPRINTS.items()):
        f = compute_field([Event(name, 1.0, atoms)])
        peak_want, atom_want = REAL_EXPECTED[name]
        peak = float(f.data.max())
        atom = min(f.sample(a) for a in atoms)
        assert abs(peak - peak_want) < PINNED, (
            "%s peak %.4f, recorded %.4f" % (name, peak, peak_want))
        assert abs(atom - atom_want) < PINNED, (
            "%s lowest atom %.4f, recorded %.4f" % (name, atom, atom_want))


def exercise_real_footprints_overshoot_only_modestly():
    """The claim the absolute domain rests on: footprint shape cannot fake coincidence.

    A lone event peaks a few percent above its severity between its atoms. Two
    coincident severity-1.0 events reach 2.0. The margin is what lets a reader treat
    "well above 1.0" as coincidence rather than as a wide footprint.
    """
    p = (0.0, 0.0, 0.0)
    coincident = compute_field([Event("rama", 1.0, [p]), Event("rota", 1.0, [p])])
    for name, atoms in sorted(REAL_FOOTPRINTS.items()):
        peak = float(compute_field([Event(name, 1.0, atoms)]).data.max())
        assert peak < 1.25, "%s overshoots to %.4f" % (name, peak)
        assert peak < float(coincident.data.max()) - 0.5, (
            "%s peaks at %.4f, close enough to coincidence to be confused with it"
            % (name, peak))


def exercise_real_footprints_stay_above_the_display_threshold():
    """The recall fix, on the shapes it was made for."""
    for name, atoms in sorted(REAL_FOOTPRINTS.items()):
        f = compute_field([Event(name, 1.0, atoms)])
        worst = min(f.sample(a) for a in atoms)
        assert worst > DISPLAY_THRESHOLD, (
            "%s has an atom at %.4f, below the display threshold" % (name, worst))
        if name in PRE_FIX_MIN_ATOM:
            assert worst > PRE_FIX_MIN_ATOM[name] + 0.05, (
                "%s worst atom %.4f is back at the pre-fix %.4f"
                % (name, worst, PRE_FIX_MIN_ATOM[name]))


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
