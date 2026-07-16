"""Geometry minimization with live intermediate states.

Restraints-only minimization ("regularisation"): pull a model back onto ideal bond
lengths, angles and the rest of its geometry restraints. No map is involved — that is
a separate target we can add to the same engine later.

The point of doing this here is that cctbx hands us the intermediate conformations.
Its LBFGS minimizers take a ``states_collector`` — any object with an
``add(sites_cart=...)`` method — and call it with each new conformation as they run
(see ``cctbx.geometry_restraints.lbfgs``). Forwarding those to a live session streams
the minimization into the viewer as it happens, instead of cutting from the start
straight to the answer.

This is deliberately *not* called real-space refine: that is a Phenix program (with
map targets, rotamer fitting and ADPs) and is not what this is.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


class _StateStream:
    """A cctbx ``states_collector`` that forwards each conformation to ``on_state``.

    cctbx calls ``add`` for every functional evaluation, including line-search probes
    that leave the coordinates untouched, so unchanged states are dropped. ``stride``
    thins what is left — a long minimization can emit hundreds of frames, and the
    viewer only needs enough to look continuous.
    """

    def __init__(self, on_state: Callable[[np.ndarray], Any], stride: int = 1):
        self._on_state = on_state
        self._stride = max(1, stride)
        self._last: Optional[np.ndarray] = None
        self.n_states = 0  # distinct conformations cctbx produced
        self.n_sent = 0    # how many we forwarded

    def add(self, sites_cart=None, hierarchy=None) -> None:
        if sites_cart is None:
            return
        coords = sites_cart.as_numpy_array()
        if self._last is not None and np.array_equal(coords, self._last):
            return  # a line-search probe that did not move the model
        self._last = coords
        self.n_states += 1
        if self.n_states % self._stride == 0:
            self.n_sent += 1
            self._on_state(coords)


def _deviations(grm, sites_cart) -> tuple:
    """(bond rmsd, angle rmsd) for ``sites_cart`` under the restraints."""
    energies = grm.energies_sites(sites_cart=sites_cart, compute_gradients=False)
    return energies.bond_deviations()[2], energies.angle_deviations()[2]


def minimize_geometry(
    model: Any,
    *,
    on_state: Optional[Callable[[np.ndarray], Any]] = None,
    max_iterations: int = 500,
    stride: int = 1,
) -> dict:
    """Minimize ``model`` against its geometry restraints, in place.

    ``on_state`` (optional) is called with an ``(N, 3)`` array for each intermediate
    conformation — pass a live session's ``push`` to stream the run into the viewer.
    Needs the monomer library, since it builds restraints. The model's sites are
    updated to the minimized ones; returns the bond/angle RMSDs either side of the run
    plus how many states it produced and forwarded.
    """
    import scitbx.lbfgs
    from cctbx import geometry_restraints
    from mmtbx.refinement import geometry_minimization

    model.process(make_restraints=True)
    grm = model.get_restraints_manager().geometry
    sites_cart = model.get_sites_cart()  # shifted in place by the minimizer
    bonds_before, angles_before = _deviations(grm, sites_cart)
    stream = _StateStream(on_state, stride) if on_state is not None else None

    geometry_minimization.lbfgs(
        sites_cart=sites_cart,
        geometry_restraints_manager=grm,
        geometry_restraints_flags=geometry_restraints.flags.flags(default=True),
        lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
            max_iterations=max_iterations),
        correct_special_position_tolerance=1.0,
        states_collector=stream,
    )
    model.set_sites_cart(sites_cart)
    if on_state is not None:
        # Always land on the real answer: a stride can drop the final state.
        on_state(sites_cart.as_numpy_array())
    bonds_after, angles_after = _deviations(grm, sites_cart)
    return {
        # The minimizer's own rmsd_bonds/rmsd_angles stay None unless it is given
        # termination cutoffs, so measure them from the restraints instead.
        "bonds_before": bonds_before, "bonds_after": bonds_after,
        "angles_before": angles_before, "angles_after": angles_after,
        "n_states": stream.n_states if stream else 0,
        "n_sent": stream.n_sent if stream else 0,
    }
