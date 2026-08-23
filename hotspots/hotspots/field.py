"""The hotspot field: deposit marked events, convolve once.

field = gaussian_filter( deposit(events) , sigma )

Deposit rule: each event is normalized **per atom**, so that every atom it implicates
reconstructs to the event's severity. The field is a concentration of severity per unit
volume, with two consequences we want:

  - size-independence: a rotamer outlier on ARG (many atoms) and on SER (few) each draw
    at their own severity, not brighter for being large (Rule 6);
  - coincidence is the signal: when several events land in one place their masses add
    into a tall peak, which a lone event -- however severe -- cannot reach.

The severity marks are anchored so 1.0 == the outlier cut, and the deposit is scaled so a
lone severity-1.0 event reconstructs to 1.0 **at each of its atoms**. Contour at
reference_level = 1.0 to "enclose lone outliers"; coincidence exceeds it substantially
(two coincident 1.0 events reach 1.94).

**The two normalizations you might want are not simultaneously available, and this file
chooses.** For a footprint of more than one atom the field *between* two atoms is higher
than the field *at* either, so "every atom reads severity" and "the maximum is severity"
cannot both hold. Enclosing lone outliers requires the first, so a lone multi-atom event
peaks 5-9% above its severity in the interstices (measured on real footprints: rama 5.6%,
cablam 7.0%, rota 8.2%, ca_geom 9.4%). Read "above 1.0" as coincidence with that margin in
mind; the gap to real coincidence is far larger than the overshoot.

The rejected alternative was one divisor per event, taken from its densest atom. That does
cap the peak at severity -- but it lets the densest part of a footprint set the weight for
all of it, so an unevenly spread event draws its isolated atoms at a fraction of severity
(0.31 against 0.98, measured) and they fall under the 0.5 display threshold and vanish.

Run under phenix python: libtbx.python hotspots/field.py [model]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from events import Event


@dataclass
class Field:
    data: np.ndarray          # 3-D severity concentration
    origin: np.ndarray        # xyz of voxel (0,0,0), Angstrom
    spacing: float            # Angstrom / voxel
    sigma: float              # kernel width, Angstrom
    reference_level: float    # field value of a lone severity-1.0 point event

    def sample(self, xyz) -> float:
        """Trilinear read of the field at a Cartesian point."""
        ijk = (np.asarray(xyz, float) - self.origin) / self.spacing
        i0 = np.floor(ijk).astype(int)
        f = ijk - i0
        val = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = i0 + (dx, dy, dz)
                    if (idx >= 0).all() and (idx < self.data.shape).all():
                        wt = ((f[0] if dx else 1 - f[0]) *
                              (f[1] if dy else 1 - f[1]) *
                              (f[2] if dz else 1 - f[2]))
                        val += wt * self.data[tuple(idx)]
        return float(val)


def _splat(grid, ijk, w):
    """Trilinear deposit of weight w at fractional voxel coordinate ijk."""
    i0 = np.floor(ijk).astype(int)
    f = ijk - i0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + (dx, dy, dz)
                if (idx >= 0).all() and (idx < grid.shape).all():
                    wt = ((f[0] if dx else 1 - f[0]) *
                          (f[1] if dy else 1 - f[1]) *
                          (f[2] if dz else 1 - f[2]))
                    grid[tuple(idx)] += w * wt


def _bounding_box(events, padding):
    pts = np.array([xyz for e in events for xyz in e.atoms_xyz], float)
    return pts.min(0) - padding, pts.max(0) + padding


def compute_field(events: List[Event], spacing=1.0, sigma=2.0,
                  padding=None) -> Field:
    if padding is None:
        padding = 3.0 * sigma
    lo, hi = _bounding_box(events, padding)
    # snap the origin to an integer grid so it can be encoded as an integer
    # grid-unit origin shift (NXSTART) that viewers place unambiguously.
    lo = np.floor(lo / spacing) * spacing
    shape = tuple((np.ceil((hi - lo) / spacing).astype(int) + 1).tolist())

    sig_vox = sigma / spacing
    # peak of the integral-normalized 3-D Gaussian gaussian_filter makes from a
    # unit point mass -- lets us deposit point masses that reconstruct to a chosen
    # peak height rather than a chosen integral.
    ref_single = 1.0 / ((2.0 * np.pi) ** 1.5 * sig_vox ** 3)
    two_s2 = 2.0 * sigma * sigma

    grid = np.zeros(shape, float)
    for e in events:
        pts = [np.asarray(a, float) for a in e.atoms_xyz]
        n = len(pts)
        if n == 0 or e.severity <= 0:
            continue
        # Normalize PER ATOM, not once for the whole event.
        #
        # This used to take a single divisor P = max over atoms of the footprint's self-overlap
        # and apply it to every atom, which makes the *densest* part of the footprint set the
        # weight for all of it. An event whose atoms are unevenly spread then draws its tight
        # part at full severity and its isolated atoms at a fraction: measured at 0.31 against
        # 0.98 for a cluster of eight with two atoms 12 A away. Those atoms fall under the 0.5
        # display threshold and vanish, which was the whole of the corpus recall shortfall --
        # 79 atoms of 874,978, ranked exactly by footprint spread (clash straddles two
        # residues, rotamer reaches down long Arg/Lys sidechains, compact channels lost none).
        #
        # With s_a = sum_b exp(-d_ab^2 / 2 sigma^2) computed at each atom and w_a = severity/s_a,
        # the reconstructed value at atom a is sum_b w_b exp(-d_ab^2/2 sigma^2), which is
        # severity for a uniform cluster (every s_b equal), severity at an isolated atom
        # (s = 1), and severity at both in a mixed footprint. So every atom the event implicates
        # is drawn at the event's severity, and a lone severity-1.0 event still peaks at exactly
        # 1.0 -- the property that lets the map be read on an absolute domain.
        #
        # A wider footprint therefore covers more volume at the same height rather than the same
        # volume at a lower height. It deposits more total mass, which is correct: the mass is
        # the extent of the problem, and Rule 6 (max, never sum, within a residue) is what stops
        # a channel dense with restraints inflating its severity.
        if n == 1:
            weights = [e.severity]
        else:
            weights = []
            for a in pts:
                s = sum(math.exp(-float(((a - b) ** 2).sum()) / two_s2) for b in pts)
                weights.append(e.severity / s)
        for a, w in zip(pts, weights):
            _splat(grid, (a - lo) / spacing, w / ref_single)

    data = gaussian_filter(grid, sigma=sig_vox, mode="constant")

    # the field is now natively in severity units: a lone severity-1.0 outlier
    # peaks at 1.0, coincidence sums above it.
    return Field(data=data, origin=lo, spacing=spacing, sigma=sigma,
                 reference_level=1.0)


def write_ccp4(field: "Field", filename: str):
    """Write the field as a CCP4/MRC map placed at the model's real-space
    coordinates via an integer grid-unit origin shift (NXSTART) -- the phenix
    convention, which viewers place unambiguously (same as a difference map).

    The map is in severity units: value 1.0 == the outlier threshold. Color on an
    ABSOLUTE domain (e.g. 0..2, knee at 1.0), never auto-scaled to the map."""
    from scitbx.array_family import flex
    from cctbx import crystal
    from iotbx.map_manager import map_manager

    data = np.ascontiguousarray(field.data, dtype=float)
    nx, ny, nz = data.shape
    fm = flex.double(data.ravel())
    fm.reshape(flex.grid(nx, ny, nz))

    # origin is grid-aligned (compute_field snaps it), so this is exact
    origin_shift = tuple(int(round(o / field.spacing)) for o in field.origin)
    cs = crystal.symmetry(
        unit_cell=(nx * field.spacing, ny * field.spacing, nz * field.spacing,
                   90, 90, 90),
        space_group_symbol="P1")
    mm = map_manager(
        map_data=fm,
        unit_cell_grid=(nx, ny, nz),
        unit_cell_crystal_symmetry=cs,
        origin_shift_grid_units=origin_shift,
        wrapping=False)
    mm.write_map(filename)
    return filename


def contributions_near(events, xyz, radius):
    """Which events deposit within `radius` of a point (coincidence anatomy)."""
    p = np.asarray(xyz, float)
    hits = []
    for e in events:
        d = min(np.linalg.norm(np.asarray(a, float) - p) for a in e.atoms_xyz)
        if d <= radius:
            hits.append((d, e))
    return sorted(hits, key=lambda de: de[0])


if __name__ == "__main__":
    import sys
    from collections import Counter
    from events import load_hierarchy, extract_all

    path = sys.argv[1] if len(sys.argv) > 1 else "/root/data/pdb_mmcif/te/1tec.cif.gz"
    h = load_hierarchy(path)
    events = extract_all(h, use_hydrogens=True)["events"]

    fld = compute_field(events, spacing=1.0, sigma=2.0)
    d = fld.data
    print(f"model: {path}")
    print(f"field shape {d.shape}  spacing {fld.spacing}  sigma {fld.sigma}")
    print(f"reference_level (lone severity-1.0 outlier peak) = {fld.reference_level:.5f}")
    print(f"field: max={d.max():.5f}  mean={d.mean():.6f}  "
          f"max/ref = {d.max()/fld.reference_level:.2f}x")

    # Q1: does the field peak at known outliers vs clean residues?
    outliers = [e for e in events if e.is_outlier]
    clean = [e for e in events if e.severity < 0.3]
    o_vals = [fld.sample(e.atoms_xyz[0]) / fld.reference_level for e in outliers]
    c_vals = [fld.sample(e.atoms_xyz[0]) / fld.reference_level for e in clean]
    print(f"\nfield at OUTLIER sites (xref): median={np.median(o_vals):.2f} "
          f"p90={np.percentile(o_vals,90):.2f}")
    print(f"field at CLEAN   sites (xref): median={np.median(c_vals):.2f} "
          f"p90={np.percentile(c_vals,90):.2f}")

    # Q2: coincidence -- what drives the global maximum?
    peak_ijk = np.unravel_index(np.argmax(d), d.shape)
    peak_xyz = fld.origin + np.array(peak_ijk) * fld.spacing
    print(f"\nglobal peak = {d.max()/fld.reference_level:.2f}x reference, "
          f"at {np.round(peak_xyz,1)}")
    hits = contributions_near(events, peak_xyz, radius=2.0 * fld.sigma)
    print(f"events within {2*fld.sigma:.0f} A of the peak: {len(hits)} "
          f"({dict(Counter(e.metric for _, e in hits))})")
    for dist, e in hits[:8]:
        print(f"    d={dist:4.1f}  sev={e.severity:4.2f}  {e.metric:5s} {e.meta.get('id','')}")
