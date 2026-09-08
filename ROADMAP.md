# Roadmap

Long-term features that came out of real use but are too big to do as tweaks.
Each entry records the motivating moment, so the feature can be judged later
against the need that produced it rather than remembered fondly in the abstract.

## Per-selection representations

**Today:** representation is one choice per model (the Appearance pane's
Representation combo — `set_model_representation`). The only per-selection
representation is automatic and transient: the Neighborhood layer, which draws
the residues within 5 Å of the current selection as ball-and-stick (with the
ribbon stepping aside there), follows the selection around, is governed by the
Selection pane's "Neighborhood in ball-and-stick" checkbox, and vanishes when
the selection clears.

**The feature:** user-authored, persistent per-selection layers — "make *this
selection* spacefill", kept until removed, independent of what is currently
selected. The plumbing largely exists: the neighborhood layer is exactly a
representation restricted to an atom set (`_apply_model_rep` restricts the main
rep away from a layer's atoms; layers carry stable ids in the id-keyed
representation store). What is missing is the product around it:

- a **layer list** per model in the Appearance pane (name, rep type, atom set,
  visibility, remove), rather than the single combo;
- an **"apply to selection"** action that captures the current selection (or
  typed expression) into a named layer;
- **overlap rules** — which layer wins when two claim an atom, and whether the
  main rep steps aside for every layer the way it does for the neighborhood;
- **persistence** across sessions (layers as expressions, so they survive
  coordinate changes) and behavior on model edits that add/remove atoms.

**Motivating moment (2026-09):** the validation-worklist work made row clicks
switch the whole model to ball-and-stick (one rep per model being all there
is), and switching back to Cartoon while the neighborhood layer stood read as
"ribbon added on top of sticks". A layer model makes such mixtures explicit
and user-controlled instead of emergent. Decision at the time: keep
one-rep-per-model plus the one automatic neighborhood layer (simple,
predictable), and build this only when mixed persistent views are missed in
real work.
