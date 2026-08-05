# Proposal: cross-metric accumulation without becoming a validation metric

*Status: **tested, and the central hypothesis did not survive**, 2026-08-05. The calibration
half was right and is now implemented. The accumulation half was measured and adds almost
nothing — see [Measured outcome](#measured-outcome-the-accumulation-hypothesis-fails) at the
end before reading the argument below, which is preserved as written.*

---

## The result this is chasing

The interesting thing a continuous field can do, and a list of outliers cannot, is show a
place where **no individual problem crosses its threshold but several sub-threshold problems
sit on top of each other**, producing a region that deserves attention. That is the effect
worth showing, and it is the one thing in this project that a boolean validation report
structurally cannot reproduce.

The field already does this *within* a metric. `compute_field` deposits each event so it peaks
at its own severity and the grid sums — the code says so directly: *"a lone severity-1.0
outlier peaks at 1.0, coincidence sums above it."*

It does **not** do it across metrics. `build_concern_fields` clips each per-metric field to
`[0, 1]` and then takes `combined = np.maximum.reduce(...)`. A marginal rotamer and a marginal
clash in the same pocket take the max — they do not reinforce at all.

### What that costs, measured

Probe over 10 structures (`6cg7 9sf2 1kws 1tec 3dk2 3t62 4esi 4qm0 6c3j 9hgo`), 1.0 Å,
rama+rota+clash:

| | median | range |
|---|---:|---:|
| hot voxels no single event could have made hot | 44.8% | 26–59% |
| hot volume gained if metrics summed instead of maxed | **+30.4%** | +11–58% |

**Read the first row carefully — it is not yet the claim.** It counts any voxel no *single*
event could have raised past 0.5, which conflates the interesting case (several genuinely
marginal concerns co-locating) with a dull one (two strong outliers' Gaussian tails overlapping
between them). The honest criterion is stricter: hot voxels where *every* contributing event is
individually sub-threshold. That number will be smaller and is the only one that supports the
claim. **It has not been measured yet.**

The second row is the load-bearing one: roughly a third of the potentially-hot volume is
cross-metric coincidence that the `max` currently discards.

---

## The reframe: we are already weighting

The reason cross-metric aggregation looks like a trap is the weighting problem — how do you
weigh a rotamer against a clash, backbone strain against CaBLAM? Any answer looks like a
judgement that invites a demand for justification, and justification means becoming a
validation metric with the full rigor burden that implies.

But **the weights already exist.** A flagged rotamer reaches concern 1.0; a flagged clash
reaches 0.5, because clash is calibrated linearly to a saturation anchor at twice its
community cut. That is a weight: it says a flagged clash is worth half a flagged rotamer. Under
`max` it silently decides which channel wins every contested voxel — geometry outranks sterics
everywhere in the combined field, as an inherited accident rather than a decision.

So the question is not *should we introduce weights*. It is *should the weights we already
have be explicit and defensible*. Moving away from `max` does not create the problem; it
exposes it.

This also means the clash-calibration defect found in the corpus run and cross-metric
accumulation are **the same piece of work**, not two.

---

## The proposal

Four parts. The first two remove the need for weights; the third replaces N weights with one
stated convention; the fourth keeps it out of metric territory.

### 1. Calibrate every channel so its community cut lands at exactly 1.0

Every channel has a threshold somebody else defined: MolProbity's 0.05% / 0.3% percentiles,
the 0.4 Å clash overlap, |Z| ≥ 4σ for covalent geometry, CaBLAM's contours. Calibrate each so
**its own cut is concern 1.0**.

Then equal weighting across channels is not a numerical claim to defend. It is the statement
*"one flagged problem is one flagged problem, whichever validator flagged it"* — a display
convention, with the per-metric calibration **inherited** from the community rather than
invented here. That is the same argument HOTSPOTS.md already makes for severity, where the
consistency constraint (the `severity = 1.0` level set reproduces mmtbx's outlier sets exactly)
is the evidence the calibration was not made up.

Today only rama and rota satisfy this. Clash lands at 0.5; the covalent and CaBLAM channels
need checking.

**All metric-specific judgement lives here, where a community threshold justifies it.** The
combination step then has no free parameters at all.

### 2. Families: max within, accumulate across

A naive sum double-counts. Rama and CaBLAM are both backbone conformation and will fire
together constantly; summing them weights backbone strain several times over against sterics.

Group the channels by what they are evidence *about*, take `max` within a family (redundant
evidence about one thing) and accumulate across families (independent lines of evidence):

| family | channels |
|---|---|
| backbone conformation | rama, cablam, ca_geom, omega |
| sidechain conformation | rota |
| sterics | clash |
| covalent geometry | bond, angle, cbeta |
| map fit | qscore, cc_mapmodel / local CC |

The assignment is defensible from the physics — no data required and nothing fitted. And it is
exactly the money shot: a marginal rotamer *plus* a marginal clash *plus* strained geometry in
one pocket now reinforce, because they are three different families.

*Arguable placements, to settle rather than leave implicit:* `cbeta` measures the
backbone–sidechain junction and could sit in either geometry or sidechain; `local_resolution`
is a property of the **data**, not an error in the model, and probably should not enter the
aggregate at all.

### 3. One dial instead of N weights: the p-norm at p = 1

Rule 4 already specifies a p-norm, `(Σ sᵢᵖ)^(1/p)`, with no denominator so a clean channel
never dilutes. It spans the whole space: `p = ∞` is today's `max`, `p = 1` is full
accumulation. You defend **one number** rather than a weight vector.

`p = 1` has an interpretable meaning under cut-at-1.0 calibration:

> **Two independent half-cut problems in the same place equal one flagged problem.**

That is a sentence a referee can accept or reject on its face, with nothing fitted behind it.
`p = 2` is the conservative alternative — the same pair reaches 0.71 rather than 1.0 — if
accumulation at p = 1 proves too eager in practice.

Note this also **closes an existing divergence**: Rule 4 prescribes a p-norm and pxviewer's
severity generation follows it, while the concern generator uses `max`. The two generations
currently disagree about how metrics combine, which was probably never a decision.

### 4. Falsify the convention; do not fit it

**This is the part that keeps the rigor burden survivable.**

The moment weights are optimized against an outcome — PDB-REDO changes, rebuild decisions,
resolution improvement — the project owes train/test splits, cross-validation, baselines, and a
defence of the outcome variable itself. That spiral is not recoverable halfway, and avoiding it
is why the validation-metric aim was dropped in the first place.

But there is a real distinction between **fitting** and **checking**. A convention chosen on
principle and then falsification-tested carries a fraction of the burden of one fitted to data.

The machinery already exists: figure C's held-out-channel enrichment against a spatially
matched null. Choose the convention above on principle, then show that the regions it produces
— specifically the accumulation-only ones — are enriched for problems the field was *not* told
about, against the null. That is a sanity check on a stated choice, not a model.

If it fails, the convention is wrong and that is worth knowing. It is cheap to run now.

---

## What keeps this a visualization tool

The Goodhart hazard in HOTSPOTS.md is about a **global roll-up**, not about a spatial field. A
field you look at, with components retained, is hard to misuse. A single number per structure is
impossible to protect. So:

- **Never emit a scalar summary** of the field — no per-structure aggregate, ever.
- **Never rank structures** by it.
- **Always retain the component fields**, so any bright voxel can be attributed to what made
  it bright. Already a stated rule; accumulation makes it mandatory rather than good practice.
- **Describe results in navigation language.** "Regions the overlay highlights contain more
  than the one thing that highlighted them, so following it is worth the user's time" — not
  "our field detects problems the outlier lists miss," which is the same data phrased as a
  metric claim and will be read as one.

---

## Open problems, stated rather than glossed

**The display contract breaks.** Concern is bounded `[0, 1]` with 0.5 yellow, 0.75 orange,
1.0 red, and each channel is clipped to 1.0 before combination. Accumulate across five families
and values routinely exceed 1.0 and clip — flattening exactly the top-end contrast the change
is meant to create. Either the domain extends (severity already uses `[0, 4]` with 1.0 as the
cut, so there is precedent) or the ceiling has to be re-thought. **This is the biggest
unresolved issue in the proposal.**

**It may collapse the two generations.** An accumulating field with every cut at 1.0 is
approximately what *severity* already is. If so this is not a third quantity but a convergence
of concern and severity — which would be a simplification worth having, but it needs to be
faced deliberately rather than discovered later.

**Families reduce double-counting; they do not eliminate correlation.** Rotamer and clash are
correlated through sidechain packing, and they sit in different families by design. The
grouping is a mitigation, not a solution, and the proposal should say so.

**The strict accumulation-only measurement does not exist yet.** The 44.8% figure conflates two
phenomena (above). Measure the strict version before anyone writes a sentence about it.

---

## What it would take

1. Measure strict accumulation-only volume (all contributors sub-threshold) — reuses the corpus
   harness, hours not days.
2. Re-anchor clash, and audit cbeta / cablam / omega / bond / angle, so every cut is 1.0.
   Required regardless of this proposal.
3. Implement families + p-norm behind a flag, keeping `max` available for comparison.
4. Resolve the display domain.
5. Falsification-test the convention with the figure C null.
6. Re-run the corpus and compare A/B/C under both combination rules. Figure B is the check that
   matters: accumulation must not push hot volume away from concerning atoms.

---

## Measured outcome: the accumulation hypothesis fails

Measured 2026-08-05 on 50 structures (`corpus/accumulation.py`, seed 20260805, 46 ok), with
clash held out of the field entirely so it could serve as the signal the field was never told
about. Both combination rules were run over the identical structures and calibration, so the
difference between the columns *is* the accumulation contribution.

| | family (max within, p=1 across) | max (control) |
|---|---:|---:|
| hot volume from never-flagged observations, as share of all hot volume | 27.5% | **27.1%** |
| ... from observations that are individually *invisible* (concern < 0.5) | 0.5% | 0.1% |
| held-out clash enrichment inside that region | 1.23× | **1.23×** |
| spatially matched null | 0.97× | 0.96× |
| structures with p < 0.05 | 16.7% | 13.9% |

**Cross-metric accumulation changes nothing that matters.** Identical novel-volume share,
identical enrichment. It inflates the absolute hot volume (0.37% of the box against 0.24%) but
inflates the full field by the same proportion, so it surfaces no region that `max` did not.

### Why — and the survey already contained the answer

Co-located sub-threshold problems are rare because **problems of different kinds happen at
different residues**. `channel_survey.py` measured across-family residue-level Jaccard at
**0.000**: a residue with a marginal rotamer is essentially never the residue with a marginal
bond deviation. Accumulation needs coincidence within ~2 Å, and the coincidence does not exist.

There is an irony worth recording. The same measurement that *justifies* `max` within a family
(members mark the same residues, so they are redundant) is what *dooms* accumulation across
families: independent lines of evidence are independent in space too, so there is nothing to
accumulate. Those are one fact, and it points in opposite directions for the two halves of the
proposal.

### What survives, and it is a different claim

**27% of hot volume comes from observations MolProbity never flagged.** That is real, it holds
under either combination rule, and it is worth saying — but it is a property of having a
**continuous scale**, not of co-locality. Most of it is single observations sitting between
0.5 and 1.0 concern: individually visible, just never flagged.

The strict version of the original idea — *nothing here is individually visible, yet together
they light up* — accounts for **0.1–0.5%** of hot volume. It essentially does not happen.

And those regions are **weaker** evidence than the field overall: 1.23× enrichment against
1.92× for the whole field. Real (the null is correctly centred at 0.97) but modest, with
per-structure significance rare.

### Consequences

1. **Do not ship family/p-norm aggregation as the default.** It costs a display-contract
   argument and a taxonomy to defend, and buys no measurable improvement. It stays behind
   `--combine family` for anyone who wants to re-test it.
2. **The calibration half of this proposal was right and is independent of the above.** Every
   community cut now lands at 1.0, which fixed clash recall (0.355 → 0.983 on 1TEC) and is
   what makes the two combination rules comparable at all.
3. **The honest figure, if one is wanted, is the continuous-scale claim**, not the additive
   one: the overlay shows sub-outlier concerns that no outlier list contains, and they carry
   weak but non-zero independent signal. Keep it in navigation language.
4. **`max` remains the shipping combination rule** — now on a corrected calibration where it
   no longer means "geometry outranks sterics".
