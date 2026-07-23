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

__all__ = [
    "Tug", "ZONE_RADIUS", "TUG_SIGMA", "ANCHOR_SIGMA", "JIGGLE_AMPLITUDE", "JIGGLE_STEPS",
]

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

#: Shake amplitude for continuous "living" dragging, in Angstrom, and the minimizer steps
#: that follow it. The two are a balance: shake then fully minimize and the shake is
#: undone. Kept faint on purpose. A big shake lightly damped (0.1 A, 5 steps ~0.11 A/frame)
#: reads as harsh vibration, because the shake is white noise per atom while the plain
#: minimizer moves atoms coherently. This is tuned down close to the plain minimizer's own
#: residual motion (~0.016 A/frame): just enough that a held drag does not look frozen,
#: not so much that it buzzes. The clean "it has come to rest" signal is the settle after
#: release, not the shake.
JIGGLE_AMPLITUDE = 0.03
JIGGLE_STEPS = 10


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
        mode: str = "sphere",
        flank: int = 0,
        selection: Optional[Any] = None,
        map_data: Any = None,
        map_weight: float = 10.0,
        steps: int = STEPS_PER_FRAME,
    ):
        """Build a drag around ``atom``. What gives way is set by ``mode``:

        - ``"sphere"`` (default): whole residues within ``radius`` A — Coot's sphere refine.
        - ``"residues"``: the dragged residue and ``flank`` residues each side of it along
          its chain. ``flank=0`` is a single-residue refine; ``flank=2`` a five-residue
          stretch. Sequence-based, so it does not balloon with a dense neighbourhood the way
          a sphere can, which is what makes it the right tool for nudging one sidechain or
          walking a loop.
        - ``"selection"``: exactly the residues the user picked (``selection`` is their atom
          indices), so an arbitrary set — a loop, an active site, two chains at an interface
          — can be the thing that moves while every atom around it stays put. The dragged
          atom's own residue is always included, so you can grab any handle.

        Everything downstream — the boundary pins, the restraint sub-manager, the map term —
        is the same whatever picks the zone; only the selection differs.
        """
        from cctbx.array_family import flex

        self.model = model
        self.atom = int(atom)
        self.map_data = map_data
        self.map_weight = float(map_weight)
        self.steps = int(steps)

        # Only if they are not already built: processing costs seconds and rebuilds
        # what is already there, which at the start of every drag is a freeze.
        if not model.restraints_manager_available():
            from . import edits
            edits.build_restraints(model)  # honours user restraint edits on the model
        self._full_sites = model.get_sites_cart()
        sites = self._full_sites.as_numpy_array()
        if not 0 <= self.atom < len(sites):
            raise ValueError(f"no atom {atom} in this model")

        if mode == "selection":
            zone = _selection_zone(model, self.atom, selection or [], len(sites))
        elif mode == "residues":
            zone = _residue_zone(model, self.atom, int(flank), len(sites))
        else:
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

    def step(self, jiggle: float = 0.0, steps: Optional[int] = None) -> np.ndarray:
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
        self._minimize(steps)
        self._full_sites.set_selected(self._zone, self._sites)
        self.model.set_sites_cart(self._full_sites)
        return self._full_sites.as_numpy_array()

    def move_to(self, target) -> np.ndarray:
        """Aim at ``target`` and take one step. The whole of a discrete drag frame."""
        self.set_target(target)
        return self.step()

    def settle(self, on_frame=None, max_iterations: int = 200) -> np.ndarray:
        """Relax the fragment to rest, holding the atom where it was let go.

        Not the same as stepping in a loop: each :meth:`step` restarts the minimizer, so
        it jitters around the minimum instead of reaching it. This is one continuous
        minimization to convergence, which decelerates cleanly to a stop — a released
        fling visibly settles rather than freezing mid-motion. ``on_frame`` receives each
        intermediate conformation (the whole model's sites), for streaming the wind-down.
        """
        import scitbx.lbfgs
        from cctbx import geometry_restraints
        from mmtbx.refinement import geometry_minimization

        # Hold the atom where it now is, so the fragment settles in place rather than
        # continuing toward wherever the pull was last aimed.
        self.set_target(tuple(self._sites[self._local]))

        owner = self

        class _Stream:
            def add(self, sites_cart=None, hierarchy=None):
                if sites_cart is None or on_frame is None:
                    return
                owner._full_sites.set_selected(owner._zone, sites_cart)
                on_frame(owner._full_sites.as_numpy_array())

        geometry_minimization.lbfgs(
            sites_cart=self._sites,
            geometry_restraints_manager=self._grm,
            geometry_restraints_flags=geometry_restraints.flags.flags(default=True),
            lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
                max_iterations=max_iterations),
            correct_special_position_tolerance=1.0,
            states_collector=_Stream(),
        )
        self._full_sites.set_selected(self._zone, self._sites)
        self.model.set_sites_cart(self._full_sites)
        return self._full_sites.as_numpy_array()

    def finish(self) -> np.ndarray:
        """End the drag, leaving the model where it stands."""
        self._grm.remove_reference_coordinate_restraints_in_place()
        return self._full_sites.as_numpy_array()

    @property
    def zone_size(self) -> int:
        """How many atoms are free to move — the per-frame cost, not the model's size."""
        return len(self._indices)

    @property
    def indices(self) -> np.ndarray:
        """Which atoms this drag can move, as positional indices into the full model.

        Every atom outside this set is bit-for-bit unchanged for the whole drag, which is
        what lets a frame be sent as a delta rather than a whole conformation (see
        ``LiveSession.push``). Fixed when the drag starts.
        """
        return self._indices

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

        # Only if there is anything to remove. remove-by-selection dereferences the
        # proxy list, which stays None until the first reference restraint is added —
        # and for a small, self-contained ligand there are no boundary atoms to pin, so
        # nothing seeds it before the first tug. (The no-selection remove in finish() is
        # already None-safe.)
        if self._grm.reference_coordinate_proxies is not None:
            self._grm.remove_reference_coordinate_restraints_in_place(
                selection=flex.size_t([self._local]))
        self._grm.append_reference_coordinate_restraints_in_place(
            reference.add_coordinate_restraints(
                sites_cart=flex.vec3_double([tuple(float(v) for v in target)]),
                selection=flex.size_t([self._local]),
                sigma=TUG_SIGMA))

    def _minimize(self, steps: Optional[int] = None) -> None:
        import scitbx.lbfgs
        from cctbx import geometry_restraints
        from mmtbx.refinement import geometry_minimization

        n = self.steps if steps is None else int(steps)

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
                    max_iterations=n),
            )
            self._sites = refined.sites_cart
            return
        geometry_minimization.lbfgs(
            sites_cart=self._sites,
            geometry_restraints_manager=self._grm,
            geometry_restraints_flags=geometry_restraints.flags.flags(default=True),
            lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
                max_iterations=n),
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


def _residue_zone(model: Any, atom: int, flank: int, n_atoms: int) -> np.ndarray:
    """The dragged atom's residue plus ``flank`` residues each side along its chain.

    Sequence order, not distance: neighbours are the residues before and after in the
    chain, so the zone is a stretch of the backbone rather than a ball of whatever happens
    to be nearby. Clamped at the ends of the chain block that holds the atom (a chain id
    reused for a later block — its waters, say — is a separate block and does not extend
    the stretch into it).
    """
    zone = np.zeros(n_atoms, dtype=bool)
    for m in model.get_hierarchy().models():
        for chain in m.chains():
            groups = list(chain.residue_groups())
            seqs = [np.asarray(rg.atoms().extract_i_seq(), dtype=int) for rg in groups]
            for i, iseqs in enumerate(seqs):
                if (iseqs == atom).any():
                    lo, hi = max(0, i - flank), min(len(groups) - 1, i + flank)
                    for j in range(lo, hi + 1):
                        zone[seqs[j]] = True
                    return zone
    return zone


def _selection_zone(model: Any, atom: int, selection: Any, n_atoms: int) -> np.ndarray:
    """Whole residues covering the user's ``selection``, plus the dragged atom's residue.

    Whole ones, like every other zone: half a residue in the zone has its own bonds cut by
    ``grm.select`` and comes apart. Adding the dragged atom's residue means a drag can start
    on any handle, even one just outside the picked set, without the pull having nothing to
    act on.
    """
    want = set(int(i) for i in selection)
    want.add(int(atom))
    zone = np.zeros(n_atoms, dtype=bool)
    for residue in model.get_hierarchy().residue_groups():
        iseqs = np.asarray(residue.atoms().extract_i_seq(), dtype=int)
        if want.intersection(iseqs.tolist()):
            zone[iseqs] = True
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
