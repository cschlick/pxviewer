"""Drag an atom and let the model give way, live.

Pulling an atom is a restraint, not a teleport: a reference coordinate restraint on the
dragged atom whose target is wherever the pointer is, then a few minimizer steps. The
atom does not arrive where you put it — it arrives where the geometry will let it, which
is the whole point. Everything nearby bends to accommodate it.

The cost has to be independent of the model. Minimizing 660 atoms takes ~32 ms a frame,
which is 31 fps for ubiquitin and about 3 for a real structure. So, like Coot, only a
zone around the dragged atom moves: a 12 A zone is ~130 atoms whether the structure has
660 or 66,000, and takes ~9 ms. The zone is built once when the drag starts; only the
target changes per frame.

A zone alone is not enough. ``grm.select`` keeps only the restraints wholly inside it, so
the zone has nothing tying it to the rest of the structure — pull on it and the whole
thing drifts off, edges first. Its boundary atoms are therefore pinned where they stand,
using the same restraint as the tug itself.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

__all__ = ["Tug", "ZONE_RADIUS", "TUG_SIGMA", "ANCHOR_SIGMA", "JIGGLE_AMPLITUDE"]

#: How much of the structure gives way, in Angstrom around the dragged atom. Big enough
#: that a residue can move without its neighbours fighting it; small enough to stay
#: interactive on any structure, since the cost is the zone's, not the model's.
ZONE_RADIUS = 8.0

#: The pull. Low sigma is a strong restraint: the atom follows the pointer closely and
#: the geometry has to argue with it. Too strong and the model tears; too weak and the
#: atom lags behind the pointer and feels dead.
TUG_SIGMA = 0.05

#: The pins. Much stronger than the pull, because the boundary is not meant to be a
#: negotiation — it is what stops the zone drifting away from the structure it belongs to.
ANCHOR_SIGMA = 0.01

#: Minimizer steps per drag frame. Enough that the model visibly chases the pointer,
#: few enough to leave the frame budget alone.
STEPS_PER_FRAME = 20

#: Shake amplitude for continuous "living" dragging, in Angstrom. Small on purpose:
#: 0.05 was harmless in testing (a held drag settled into density about as well as with
#: no shake, adding a ~0.007 A wander), while 0.1+ began to hold the fit back.
JIGGLE_AMPLITUDE = 0.05


def flex_vec3(array):
    """An (N, 3) numpy array as a cctbx flex.vec3_double."""
    from cctbx.array_family import flex

    return flex.vec3_double(np.ascontiguousarray(array, dtype=float))


class Tug:
    """One drag of one atom: the zone that gives way, and the pull on it.

    Built once when the drag begins (:meth:`begin`), stepped as the pointer moves
    (:meth:`move_to`), and closed out at the end (:meth:`finish`). The model is updated
    in place throughout, so whatever reads it afterwards — the tables, validation, a
    later minimization — sees the atoms where the drag left them.
    """

    def __init__(
        self,
        model: Any,
        atom: int,
        *,
        radius: float = ZONE_RADIUS,
        map_data: Any = None,
        map_weight: float = 10.0,
        steps: int = STEPS_PER_FRAME,
    ):
        from cctbx.array_family import flex

        self.model = model
        self.atom = int(atom)
        self.map_data = map_data
        self.map_weight = float(map_weight)
        self.steps = int(steps)

        # Only if they are not already built: processing costs seconds and rebuilds
        # what is already there, which at the start of every drag is a freeze.
        if not model.restraints_manager_available():
            model.process(make_restraints=True)
        self._full_sites = model.get_sites_cart()
        sites = self._full_sites.as_numpy_array()
        if not 0 <= self.atom < len(sites):
            raise ValueError(f"no atom {atom} in this model")

        zone = _zone_selection(model, sites, self.atom, radius)
        self._zone = flex.bool(zone.tolist())
        self._indices = np.flatnonzero(zone)
        # Where the dragged atom sits within the zone: the sub-manager renumbers.
        self._local = int(np.flatnonzero(self._indices == self.atom)[0])

        grm = model.get_restraints_manager().geometry
        self._grm = grm.select(self._zone)
        self._sites = self._full_sites.select(self._zone)

        anchors = _boundary_atoms(grm, self._full_sites, zone)
        self._pin(anchors)

    # -- the drag --------------------------------------------------------

    def set_target(self, target) -> None:
        """Aim the pull at ``target`` without stepping. Paired with :meth:`step` for a
        free-running drag, where the target moves under a minimizer that never stops."""
        self._tug(target)

    def step(self, jiggle: float = 0.0) -> np.ndarray:
        """One burst of minimizer steps toward the current target. Returns all sites.

        The atom will not reach the target, and should not: what comes back is where the
        geometry (and the map, if any) allows it to go.

        ``jiggle`` (Angstrom) shakes the zone before minimizing, so a held drag keeps
        moving instead of freezing at the first minimum it finds — a crude warmth that
        keeps the structure alive and nudges it out of shallow traps. Kept small: enough
        to shake is enough to break geometry, and the minimizer only pulls back what it
        can reach in a burst.
        """
        if jiggle > 0:
            noise = np.random.normal(0.0, jiggle, (self._sites.size(), 3))
            self._sites = self._sites + flex_vec3(noise)
        self._minimize()
        self._full_sites.set_selected(self._zone, self._sites)
        self.model.set_sites_cart(self._full_sites)
        return self._full_sites.as_numpy_array()

    def move_to(self, target) -> np.ndarray:
        """Aim at ``target`` and take one step. The whole of a discrete drag frame."""
        self.set_target(target)
        return self.step()

    def finish(self) -> np.ndarray:
        """End the drag, leaving the model where it stands."""
        self._grm.remove_reference_coordinate_restraints_in_place()
        return self._full_sites.as_numpy_array()

    @property
    def zone_size(self) -> int:
        """How many atoms are free to move — the per-frame cost, not the model's size."""
        return len(self._indices)

    # -- internals -------------------------------------------------------

    def _pin(self, anchors: np.ndarray) -> None:
        """Pin the zone's boundary where it stands, so the zone stays attached.

        ``grm.select`` drops every restraint that reaches outside the zone, which leaves
        the zone free-floating: without this it drifts bodily and its edges unravel.
        """
        from cctbx.array_family import flex
        from mmtbx.geometry_restraints import reference

        local = np.flatnonzero(np.isin(self._indices, anchors))
        if not len(local):
            return
        self._grm.adopt_reference_coordinate_restraints_in_place(
            reference.add_coordinate_restraints(
                sites_cart=self._sites.select(flex.size_t(local.tolist())),
                selection=flex.size_t(local.tolist()),
                sigma=ANCHOR_SIGMA))

    def _tug(self, target) -> None:
        """Re-aim the pull. The pins are re-made with it: cctbx removes reference
        restraints by selection, and re-stating both is cheaper than tracking them."""
        from cctbx.array_family import flex
        from mmtbx.geometry_restraints import reference

        self._grm.remove_reference_coordinate_restraints_in_place(
            selection=flex.size_t([self._local]))
        self._grm.append_reference_coordinate_restraints_in_place(
            reference.add_coordinate_restraints(
                sites_cart=flex.vec3_double([tuple(float(v) for v in target)]),
                selection=flex.size_t([self._local]),
                sigma=TUG_SIGMA))

    def _minimize(self) -> None:
        import scitbx.lbfgs
        from cctbx import geometry_restraints
        from mmtbx.refinement import geometry_minimization

        if self.map_data is not None:
            from cctbx.maptbx import real_space_refinement_simple

            # Unlike the geometry minimizer, this one does not shift the sites it is
            # given — it rebinds its own copy — so the answer has to be read back off
            # the minimizer. Handing it self._sites and hoping is a silent no-op.
            refined = real_space_refinement_simple.lbfgs(
                sites_cart=self._sites,
                density_map=self.map_data,
                # "fd" (finite differences) is what mmtbx's own real-space refinement
                # uses; the unit cell comes from the sub-manager's symmetry.
                gradients_method="fd",
                geometry_restraints_manager=self._grm,
                real_space_target_weight=self.map_weight,
                real_space_gradients_delta=0.25,
                lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
                    max_iterations=self.steps),
            )
            self._sites = refined.sites_cart
            return
        geometry_minimization.lbfgs(
            sites_cart=self._sites,
            geometry_restraints_manager=self._grm,
            geometry_restraints_flags=geometry_restraints.flags.flags(default=True),
            lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
                max_iterations=self.steps),
            correct_special_position_tolerance=1.0,
        )


def _zone_selection(model: Any, sites: np.ndarray, atom: int, radius: float) -> np.ndarray:
    """Whole residues within ``radius`` of the atom.

    Whole ones: half a residue in the zone would have its own bonds cut by ``grm.select``
    and would come apart.
    """
    near = np.linalg.norm(sites - sites[atom], axis=1) < radius
    zone = np.zeros(len(sites), dtype=bool)
    for residue in model.get_hierarchy().residue_groups():
        indices = residue.atoms().extract_i_seq()
        i_seqs = np.asarray(indices, dtype=int)
        if near[i_seqs].any():
            zone[i_seqs] = True
    return zone


def _boundary_atoms(grm: Any, sites: Any, zone: np.ndarray) -> np.ndarray:
    """Atoms inside the zone that are bonded to atoms outside it.

    Topological rather than geometric: what has to hold is the zone's *connection* to the
    structure, which is exactly the bonds that cross the line.
    """
    proxies = grm.pair_proxies(sites_cart=sites).bond_proxies
    boundary: List[int] = []
    for proxy in proxies.simple:
        i, j = proxy.i_seqs
        if zone[i] != zone[j]:
            boundary.append(i if zone[i] else j)
    return np.unique(np.asarray(boundary, dtype=int)) if boundary else np.empty(0, dtype=int)
