"""Minimization with live intermediate states.

Two targets, both in-place on a cctbx model:

* :func:`minimize_geometry` — restraints only ("regularisation"): pull a model back
  onto ideal bond lengths, angles and the rest of its geometry restraints.
* :func:`minimize_to_map` — restraints *and* density: also pull it into a map, with
  cctbx deriving the balance between the two targets.

The point of doing this here is that cctbx hands us the intermediate conformations.
Its LBFGS minimizers take a ``states_collector`` — any object with an
``add(sites_cart=...)`` method — and call it with each new conformation as they run.
Forwarding those to a live session streams the minimization into the viewer as it
happens, instead of cutting from the start straight to the answer.

This is deliberately *not* called real-space refine: that is a Phenix program (with
rotamer fitting, ADPs and its own macro-cycles) and is not what this is.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


class _Halt(Exception):
    """Internal: unwinds a minimizer that has no stop hook of its own.

    Only used for the map target — see :func:`minimize_to_map`.
    """


class _StateStream:
    """A cctbx ``states_collector``: forwards each conformation, and can halt the run.

    cctbx calls ``add`` for every functional evaluation, including any that leave the
    coordinates untouched, so unchanged states are dropped. ``stride`` thins what is
    left — a run emits hundreds of frames and the viewer only needs enough to look
    continuous. ``last`` always holds the newest conformation, thinned or not, which
    is what a halted run falls back to.
    """

    def __init__(
        self,
        on_state: Optional[Callable[[np.ndarray], Any]] = None,
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        stride: int = 1,
        halt_by_raising: bool = False,
    ):
        self._on_state = on_state
        self._should_stop = should_stop
        self._stride = max(1, stride)
        self._halt_by_raising = halt_by_raising
        self.last: Optional[np.ndarray] = None
        self.n_states = 0  # distinct conformations cctbx produced
        self.n_sent = 0    # how many we forwarded
        self.stopped = False

    def add(self, sites_cart=None, hierarchy=None) -> None:
        if sites_cart is None:
            return
        coords = sites_cart.as_numpy_array()
        if self.last is not None and np.array_equal(coords, self.last):
            return  # an evaluation that did not move the model
        self.last = coords
        self.n_states += 1
        if self._on_state is not None and self.n_states % self._stride == 0:
            self.n_sent += 1
            self._on_state(coords)
        if self._halt_by_raising and self._should_stop is not None and self._should_stop():
            self.stopped = True
            raise _Halt()


def _deviations(grm, sites_cart) -> tuple:
    """(bond rmsd, angle rmsd) for ``sites_cart`` under the restraints."""
    energies = grm.energies_sites(sites_cart=sites_cart, compute_gradients=False)
    return energies.bond_deviations()[2], energies.angle_deviations()[2]


def minimize_geometry(
    model: Any,
    *,
    on_state: Optional[Callable[[np.ndarray], Any]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    max_iterations: int = 500,
    stride: int = 1,
) -> dict:
    """Minimize ``model`` against its geometry restraints, in place.

    ``on_state`` (optional) is called with an ``(N, 3)`` array for each intermediate
    conformation — pass a live session's ``push`` to stream the run into the viewer.
    ``should_stop`` (optional) is polled each step; return True to halt early, leaving
    the model at the conformation reached so far. Needs the monomer library, since it
    builds restraints. Returns the bond/angle RMSDs either side of the run, the state
    counts, and whether it was stopped.
    """
    import scitbx.lbfgs
    from cctbx import geometry_restraints
    from mmtbx.refinement import geometry_minimization

    class _Haltable(geometry_minimization.lbfgs):
        """Stoppable minimizer. scitbx.lbfgs halts when callback_after_step returns
        True, which the base class already uses for its RMSD cutoffs — so defer to it
        first and only then check the caller. Defined here to keep the cctbx import
        lazy; note the base class runs the minimization inside __init__, so the stop
        hook has to be in place before that.
        """

        def __init__(self, *args, **kwargs):
            self.stopped = False
            super().__init__(*args, **kwargs)

        def callback_after_step(self, minimizer):
            if super().callback_after_step(minimizer) is True:
                return True
            if should_stop is not None and should_stop():
                self.stopped = True
                return True
            return None

    model.process(make_restraints=True)
    grm = model.get_restraints_manager().geometry
    sites_cart = model.get_sites_cart()  # shifted in place by the minimizer
    bonds_before, angles_before = _deviations(grm, sites_cart)
    stream = _StateStream(on_state, stride=stride) if on_state is not None else None

    minimizer = _Haltable(
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
        "stopped": minimizer.stopped,
        "weight": None,  # no map target, so nothing to balance the restraints against
    }


def minimize_to_map(
    model: Any,
    map_data: Any,
    *,
    on_state: Optional[Callable[[np.ndarray], Any]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    max_iterations: int = 150,
    stride: int = 1,
) -> dict:
    """Minimize ``model`` into ``map_data`` under its geometry restraints, in place.

    ``map_data`` is a cctbx flex map (``map_manager.map_data()``) that must already be
    in the model's frame — origin at zero and the same shift as the model. Loading a
    model and map together as a group gives that for free, since cctbx's
    ``map_model_manager`` aligns them.

    Weighting the map against the restraints is the part that is easy to get wrong, so
    cctbx derives it (``mmtbx.refinement.real_space.weight``); the value used comes
    back as ``weight``. Otherwise this behaves like :func:`minimize_geometry`.
    """
    from scitbx.array_family import flex
    from mmtbx.refinement.real_space import individual_sites

    model.process(make_restraints=True)
    restraints = model.get_restraints_manager()
    grm = restraints.geometry
    bonds_before, angles_before = _deviations(grm, model.get_sites_cart())

    # Unlike the geometry minimizer, this one exposes no callback_after_step to stop
    # on, so the states collector unwinds it instead — and the last conformation it
    # streamed is where a halted run lands.
    stream = _StateStream(
        on_state, should_stop=should_stop, stride=stride, halt_by_raising=True)
    weight = None
    try:
        result = individual_sites.easy(
            map_data=map_data,
            xray_structure=model.get_xray_structure(),
            pdb_hierarchy=model.get_hierarchy(),
            geometry_restraints_manager=restraints,
            max_iterations=max_iterations,
            states_accumulator=stream,
        )
        sites_cart = result.xray_structure.sites_cart()
        weight = result.w
    except _Halt:
        sites_cart = flex.vec3_double(stream.last)

    model.set_sites_cart(sites_cart)
    if on_state is not None:
        # Always land on the real answer: a stride can drop the final state.
        on_state(sites_cart.as_numpy_array())
    bonds_after, angles_after = _deviations(grm, sites_cart)
    return {
        "bonds_before": bonds_before, "bonds_after": bonds_after,
        "angles_before": angles_before, "angles_after": angles_after,
        "n_states": stream.n_states, "n_sent": stream.n_sent,
        "stopped": stream.stopped, "weight": weight,
    }


def minimize(model: Any, *, map_data: Any = None, **kwargs) -> dict:
    """Minimize ``model``, into ``map_data`` if one is given and onto its restraints
    alone otherwise. See :func:`minimize_to_map` and :func:`minimize_geometry`."""
    if map_data is None:
        return minimize_geometry(model, **kwargs)
    return minimize_to_map(model, map_data, **kwargs)
