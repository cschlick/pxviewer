# Notes on `validation_events.py`

For the map-model / per-atom RSR project, the third consumer of this file. Written from the
pxviewer side after reviewing the geometry and map-fit additions and integrating them.
Everything below was measured, not inferred; the model is `1tec.pdb` (2737 atoms) unless
stated otherwise.

Canonical copy: `python/pxviewer/validation_events.py` in pxviewer. There is now exactly one
copy in this repo — a second, drifting copy is the failure this file exists to prevent.

## 1. The contract, unchanged

The file carries **native values and one localization**: which residue a result belongs to,
which atoms it implicates, and — for a validator — its own `outlier` boolean. It does not
carry scores. Turning a value into a score or a colour is the caller's business, and the
three consumers legitimately disagree there:

| consumer | maps events to |
| --- | --- |
| pxviewer | unbounded surprisal **severity**, `[0, 4]`, 1.0 = the community cut |
| `hotspots/` generator | bounded **concern**, `[0, 1]`, log-interpolated from a good/bad percentage |
| map-model / RSR | continuous map-fit fields, coloured directly |

What must never differ between us is the localization. That is the whole point: a
disagreement becomes impossible rather than merely unlikely.

## 2. What is verified exact

The covalent math is not approximately right, it is bit-for-bit right. Against mmtbx's own
`energies.bond_deviations()` / `angle_deviations()` on 1TEC:

| | this file | mmtbx |
| --- | --- | --- |
| `bond_rmsd` | 0.019348203633270267 | 0.019348203633270257 |
| `angle_rmsd` | 2.9798625995085537 | 2.979862599508552 |
| n bonds / angles | 2589 / 3547 | 2589 / 3547 |
| 4σ outliers | 24 / 174 | 24 / 174 (`get_bond_outliers` / `get_angle_outliers`) |

`sigma = 1/sqrt(weight)` matches `cctbx.geometry_restraints.weight_as_sigma`; the proxy and
restraint constructors, the `delta = ideal − model` sign convention, and the degree/angstrom
units all check out. `summarize`'s `rama_outlier_pct` (1.4793) and `rota_outlier_pct` (10.0)
match `ramalyze`/`rotalyze` exactly.

So the core is sound. Everything in §3 is integration around it.

## 3. What I changed — please don't revert these

### 3.1 Restraint building is a mutation, and it moves results

`extract_bonds` / `extract_angles` called `model.process(make_restraints=True)` on the
caller's model. `process()` sorts the hierarchy's atoms **in place** and resets their serials.
Measured consequence: asking for the covalent channel changes the *rotamer* answer — 26
outliers become 27, same file, same 2737 atoms. Any atom index recorded before the build then
points at a different atom.

There is a second edge specific to a host application. pxviewer folds a user's custom
bond/angle restraints in through its own `edits.build_restraints` (one build path, one lock).
A plain `process()` ignores those, and because an existing restraints manager is *reused*
rather than rebuilt, the edit-less manager is inherited by whatever runs next — I reproduced
a user's custom bond being silently dropped from minimize and drag.

**Changes:** `extract_bonds`, `extract_angles` and `extract_all` accept `geometry=`, the same
"pass what you already have" contract as `dots=` and `ramalyze_result=`. When falling back,
`extract_all` builds restraints *first*, before anything indexes the hierarchy, so one atom
numbering holds for the whole call. If your project owns a restraint build, inject it.

### 3.2 Covalent channels now filter by restraint origin — including `edits`

A restraints manager can carry secondary-structure **hydrogen-bond** distance and angle
restraints in the very same proxy arrays. Counting them as covalent inflates the RMSDs
against any phenix number.

cctbx's own `get_covalent_bond_proxies` is the obvious fix and is **too narrow**: it keeps
only origin `'covalent geometry'` (id 0). A restraint the user declared has origin `'edits'`
(id 4) — I verified this — and dropping it makes the channel disagree with the model actually
being refined. `covalent_origin_ids()` allows both and excludes the rest.

### 3.3 Hydrogen bonds were being counted as clashes

This one was mine, from the previous round, and it is the largest numerical correction here.
An H-bond sits well inside the vdW sum by construction, so it carries a negative gap. probe2
labels them `hb`; MolProbity subtracts H-bonded pairs before computing clashscore. We were not.

On 1TEC, of the 2661 probe2 rows with overlap ≥ 0.4 Å, **2376 were hydrogen bonds**. Effect
of excluding them:

| | before | after |
| --- | ---: | ---: |
| clash contacts | 618 | 536 |
| clash outliers | 47 | 5 |
| `clashscore` | 17.2 | 1.8 |

Also fixed: a row where only one side resolved was admitted as a one-element "pair", inventing
a contact. Both ends must now resolve.

Note the surviving 5 outlier pairs are consistent with probe2's own 40 `bo` (bad overlap) dot
rows collapsing per atom pair.

### 3.4 Q-score was silently misattributing every value

Three separate problems, in order of severity:

1. **Index misalignment.** `calc_qscore` returns one value per **non-hydrogen** atom. The code
   zipped the result against the full atom list, so as soon as a model had hydrogens every
   score landed on the wrong atom — no exception, just wrong numbers with correct-looking
   types. Now masked and scattered into a full-length array with `nan` for hydrogens, with a
   length check.
2. **Mutating the caller's manager.** `calc_qscore` strips hydrogens and calls `set_model`
   back onto the manager it is handed. Given the live one, it swaps out the model its owner is
   using — and inside `extract_fit` it then fed an H-stripped model to the extractors that ran
   after it, so their atom indices were in a different numbering. Now runs against
   `mmm.deep_copy()`.
3. **`probe_allocation_method` does not exist** in this cctbx. Verified signature:
   `calc_qscore(mmm, selection=None, shells=None, n_probes=8, rtol=0.9, nproc=1, log=..., debug=False)`.
   Removed.

Also, `QSCORE_SHELLS` was 21 shells from 0.0 with `n_probes=8`. The shells and probe count
*are* the metric's definition — a different set is a different number wearing the same name.
Now cctbx's own defaults (`np.linspace(0.1, 2.0, 20)`, 32 probes, rtol 0.9), which is what
`phenix.qscore` reports. A shell at radius 0.0 is particularly not free: every probe collapses
onto the atom centre and contributes duplicate copies of the peak-anchored density, biasing
upward.

### 3.5 Local resolution: FSC cutoff and a missing residue

Defaulted to `fsc_cutoff=0.5`. cctbx defaults to 0.143, and pxviewer's existing
`volume_io.local_resolution_from_half_maps` uses 0.143. Same metric name, same angstrom units,
systematically different values — any palette or threshold calibrated on one misreads the
other. Now `LOCAL_RESOLUTION_FSC_CUTOFF = 0.143`.

It also carried `residue=None` while every other channel carries a `ResidueKey`, so any
roll-up or join keyed on `.residue` dropped the channel entirely and silently. Fixed.

### 3.6 The roll-up is split in two

`per_atom`'s defaults — drop values ≤ 0, fill unmarked atoms with `0.0` — say that a value is
badness measured up from zero and that an untouched atom is clean. True of every geometry
channel, false of every map-fit one.

The corruption was silent. A cc gap is normally **negative** for the well-fit case, so the
severity defaults turn a good structure into a field of zeros, indistinguishable from atoms
never measured. For local resolution, `0.0 Å` is the *best possible* value, so an unmeasured
atom displays as perfectly resolved.

```
cc_gap events -0.12, -0.05, +0.03 on atoms 0-2, atom 3 unmeasured:
  per_atom(...)        -> [0.0, 0.0, 0.03, 0.0]     <- two worst atoms erased
  per_atom_field(...)  -> [-0.12, -0.05, 0.03, nan] <- correct
```

**Use `per_atom_field` for anything in `MAP_FIT_METRICS`.** `per_atom` now raises on map-fit
events given bare defaults, pointing at it. Saying what you mean still works: an explicit
`transform=`, or explicit `skip_nonpositive=`/`fill=`, both pass, and the guard only inspects
the metric actually requested, so a geometry roll-up over a mixed event list is unaffected.

## 4. Using it

```python
from validation_events import (
    extract_all, extract_fit, per_atom, per_atom_field,
    check_field_agreement, worse_than_percent, summarize, restrict)

# Geometry. Inject geometry= if your project owns a restraint build.
events = extract_all(model, metrics=("rama", "rota", "clash", "bond", "angle"),
                     geometry=my_restraints.geometry)
severity = per_atom(events, n_atoms, metric="clash",
                    transform=lambda e: min(e.value / 0.40, 4.0))

# Map-fit. cc_* needs a cctbx carrying mmtbx.maps.local_cc_star -- see §5.
fit = extract_fit(mmm, d_min, metrics=("qscore", "local_resolution", "rsr"))
field = per_atom_field(fit, n_atoms, metric="qscore")   # nan where unmeasured

# Aggregates over the whole model or a region.
summarize(restrict(events, region_atom_indices), n_atoms=len(region_atom_indices))
```

`check_field_agreement` maps a field back to atoms and confirms the hot places are the places
validation complained about. **Pass `concerning=worse_than_percent(2.0)`** for any continuous
field: judged against the outlier boolean alone it reports correct behaviour as failure,
because a concern curve rises well before the outlier cut. There is a worked example in
`HOTSPOTS_NOTES.md` §4.

## 5. Open items — yours

1. **`mmtbx.maps.local_cc_star` does not exist in stock cctbx.** Zero hits across all of
   site-packages here. So `cc_mapmodel` / `cc_half` / `cc_star` / `cc_gap` cannot run, and
   since `extract_fit`'s default `metrics` includes `cc_gap`, `extract_fit(mmm, d_min)` fails
   with defaults. It now raises a message saying so rather than a bare `ImportError`, but the
   dependency needs resolving — either vendor the module, or rebuild the channel on
   `mmtbx.maps.correlation` / `mmtbx.maps.map_model_cc`, which do exist.
2. **`extract_rsr` materialises the envelope through `list()`** —
   `np.array(list(env), dtype=np.int64)` on a `flex.size_t` that is 10⁶–10⁸ elements for a
   whole-protein envelope on a cryo-EM grid. Use `env.as_numpy_array()`. Two sites.
3. **`extract_rsr` drops residues with < 10 grid points** with no record. On a coarse grid a
   whole class of small residues vanishes from the field silently.
4. **`extract_rsr` uses `hierarchy.only_model()`**, which raises on a multi-model file, while
   `residue_atom_index` iterates all models — the two disagree on multi-model input.
5. **RSR calibration** is still owed against `phenix.real_space_correlation` / EDSTATS, as the
   docstring says. Also worth checking: `denom = sum|obs + calc|` can cancel toward zero on a
   mean-zero cryo-EM map, so it is not a stable normaliser there.
6. **`asu` bond proxies are excluded.** Symmetry-related covalent bonds (cross-symmetry
   disulfides, metal links) live in `.asu` with `i_seq`/`j_seq` rather than `i_seqs` and need
   the `gr.bond(sites_cart, asu_mappings, proxy)` overload. Absent from events and from
   `bond_rmsd`. Fine for a boxed/P1 model; matters for a crystal form.
7. **An `angle_proxy` carrying `sym_ops` fails silently.** `gr.bond` asserts loudly on a
   non-identity `rt_mx_ji`, but `gr.angle` ignores sym ops and returns a wrong angle. Rare,
   but silent.
8. **Polarity is the caller's.** Local resolution is lower-is-better while every other field is
   higher-is-better, and the max combination does not know that. It only bites where several
   events touch one atom — the per-atom metrics emit exactly one each — so it would matter if
   you ever fan a per-residue local-resolution variant out over its atoms.
9. **`index=` is accepted and unused** by `extract_qscore` and `extract_local_resolution`.
   Harmless, but it hides that those two never consult the residue index.
10. **`detail["sigma"]` can be `nan`** for a zero-weight restraint, which breaks strict JSON
    serialization.

## 6. Testing and environment

- Tests: `python/tests/test_validation_events.py` (13 tests, ~15 s — most use synthetic
  events, so they are cheap). They pin the localization agreement with pxviewer's scoring,
  the clash-per-contact collapse, the `geometry=` injection preserving host edits, the
  roll-up split, and the field-agreement checker including a **negative control** (a shuffled
  field must fail — a check that only ever passes is worthless).
- Do **not** run pxviewer's whole suite casually; a previous full run consumed ~10 GB.
- cctbx python here: `/Users/christopher/miniconda3/envs/pxviewer/bin/libtbx.python`.
- Self-test: `libtbx.python validation_events.py MODEL.pdb`. On 1TEC (events / outliers /
  atoms implicated): rama 338 / 5 / 20, rota 260 / **27** / 103, clash 536 / 5 / 10,
  bond 2589 / 24 / 46, angle 3547 / 174 / 447. `clashscore` 1.83, `bond_rmsd` 0.01935,
  `angle_rmsd` 2.980.

  Note the rotamer count: it is **27** here and **26** if you ask for `("rama", "rota")`
  alone. That is §3.1 in action — the self-test requests the covalent channels, restraints get
  built, `process()` reorders the hierarchy, and rotalyze answers differently. The self-test
  is the cheapest demonstration of why a host should inject `geometry=`.
- The map-fit extractors were exercised against cctbx's synthetic `map_model_manager`
  (`generate_map(d_min=3.0)`): `qscore` 86 events in 0.653–0.896, `rsr` 10 events, and
  `local_resolution` runs. That proves the API paths execute and the model is no longer
  mutated — it is **not** a check of the values, since that object has no genuine half maps.
  Numerical validation against real data is still open.
