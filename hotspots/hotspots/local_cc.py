"""Native voxel and per-atom local map-model Pearson correlation fields."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve
from cctbx import maptbx
from scitbx.array_family import flex
from mmtbx.maps import correlation


def correlation_radius(resolution):
    """Smallest defensible local support: one resolution element, floor 2.5 A."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    return max(2.5, float(resolution))


def sphere_kernel(unit_cell, n_real, radius):
    """Discrete spherical kernel matching map grid points within radius."""
    n_real = np.asarray(n_real, dtype=int)
    # Conservative bounds from the grid-step vector lengths. Orthogonal boxed
    # maps are usual, but Cartesian distance below handles non-orthogonal cells.
    steps = []
    for axis in range(3):
        f = np.zeros(3)
        f[axis] = 1.0 / n_real[axis]
        steps.append(np.linalg.norm(np.asarray(unit_cell.orthogonalize(tuple(f)))))
    half = np.ceil(radius / np.asarray(steps)).astype(int)
    shape = tuple((2 * half + 1).tolist())
    kernel = np.zeros(shape, dtype=float)
    center = half
    for index in np.ndindex(shape):
        delta = np.asarray(index, dtype=float) - center
        frac = tuple(delta / n_real)
        cart = np.asarray(unit_cell.orthogonalize(frac), dtype=float)
        if np.dot(cart, cart) <= radius * radius + 1e-10:
            kernel[index] = 1.0
    if kernel.sum() < 2:
        raise ValueError("sphere contains fewer than two grid points")
    return kernel


def rolling_local_cc(exp_map, model_map, unit_cell, radius, valid_mask=None):
    """Whole-grid sphere Pearson CC from five local sum convolutions.

    Correlations use the same discrete spherical support as the pointwise cctbx
    primitive at grid-point centers. `valid_mask` controls output only; it does
    not alter a voxel's correlation support.
    """
    a = np.asarray(exp_map.as_numpy_array(), dtype=np.float64)
    b = np.asarray(model_map.as_numpy_array(), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("experimental and model maps must share a grid")
    kernel = sphere_kernel(unit_cell, a.shape, radius)
    n = float(kernel.sum())
    conv = lambda x: convolve(x, kernel, mode="constant", cval=0.0)
    sa, sb = conv(a), conv(b)
    saa, sbb, sab = conv(a * a), conv(b * b), conv(a * b)
    cov = sab - sa * sb / n
    va = np.maximum(saa - sa * sa / n, 0.0)
    vb = np.maximum(sbb - sb * sb / n, 0.0)
    denom = np.sqrt(va * vb)
    cc = np.full(a.shape, np.nan, dtype=float)
    good = denom > np.finfo(float).eps * n
    if valid_mask is not None:
        good &= np.asarray(valid_mask, dtype=bool)
    cc[good] = np.clip(cov[good] / denom[good], -1.0, 1.0)
    return cc, kernel


def model_envelope_mask(unit_cell, n_real, sites_cart, env_radius):
    """Boolean grid mask within env_radius of any model atom."""
    indices = maptbx.grid_indices_around_sites(
        unit_cell=unit_cell,
        fft_n_real=tuple(n_real),
        fft_m_real=tuple(n_real),
        sites_cart=sites_cart,
        site_radii=flex.double(sites_cart.size(), float(env_radius)))
    mask = np.zeros(tuple(n_real), dtype=bool)
    mask.flat[np.asarray(indices, dtype=np.int64)] = True
    return mask


def per_atom_local_cc(exp_map, model_map, unit_cell, sites_cart, radius):
    """Reference cctbx local CC evaluated independently at each atom."""
    values = []
    for site in sites_cart:
        cc = correlation.from_map_map_atom(
            map_1=exp_map, map_2=model_map, site_cart=site,
            unit_cell=unit_cell, radius=radius)
        values.append(float(cc) if cc is not None else float("nan"))
    return np.asarray(values, dtype=float)


def cc_to_concern(cc):
    """Parameter-free affine reversal of Pearson's native [-1,1] range."""
    cc = np.asarray(cc, dtype=float)
    concern = np.zeros(cc.shape, dtype=float)
    finite = np.isfinite(cc)
    concern[finite] = np.clip((1.0 - cc[finite]) / 2.0, 0.0, 1.0)
    return concern
