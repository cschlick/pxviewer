# Design: two fields — a locator and a hotspot density

*Status: **design, approved in principle 2026-08-05** (bandwidth 6 Å agreed). Not implemented.
Supersedes the accumulation half of [AGGREGATION_PROPOSAL.md](AGGREGATION_PROPOSAL.md), whose
central hypothesis was measured and failed.*

---

## Summary

The current field cannot accumulate across residues, and that is a property of how it deposits
rather than a tuning problem. **A system that only accumulates within a residue does not need
to be a field at all — colouring residues would do the same job.** So the field's aggregation
is currently earning nothing; only its display value (seeing through the structure) is real.

The fix is to stop asking one field two questions:

| | **locator field** | **hotspot field** |
|---|---|---|
| question | *where exactly is this problem?* | *which neighbourhoods carry an unusual concentration of trouble?* |
| construction | peak-normalized splats, σ = 2 Å | severity-weighted intensity, R = 6 Å |
| unit | bounded concern, cut at 1.0 | **flagged-outlier-equivalents** |
| accumulates | within ~2 Å (sub-residue) | across the neighbourhood |
| status | **unchanged** — figure B validates it | new |

---

## Why the current design cannot do this

### The deposition model, plainly

Each event goes onto the grid in four steps:

1. **Severity** `s` — the calibrated concern, anchored so every channel's community cut is 1.0.
2. **Localization** — the event names a set of atoms.
3. **Point masses** at each of those atoms, weighted so that *this event alone*, after blurring,
   peaks at exactly `s`. (The `P` divisor in `field.compute_field` divides out the event's
   self-overlap, so a 12-atom rotamer and a 1-atom event both peak at `s` rather than 12×.)
4. **One Gaussian convolution** over the whole grid.

So: **each problem is a bump of height = its severity and width σ, and the field is the sum of
the bumps.** Per-metric fields are clipped to [0, 1]; metrics combine by maximum.

### Two consequences, both fatal to accumulation

**Peak-normalization caps the sum.** The field at a point is `Σ sᵢ·w(dᵢ)` with `w ≤ 1`, so N
coincident events reach at most `Σ sᵢ`. Two concerns of 0.2 can never make 0.5 — at any σ,
superimposed. This is not a kernel-width problem and no kernel change touches it.

**Gaussian tails are thin.** A neighbour contributes `exp(−d²/2σ²)` of its own peak: at σ = 2,
**16.4%** at 3.8 Å (adjacent Cα) and **0.1%** at 7.6 Å. Two events two residues apart would
each need severity 0.500 to reach threshold — i.e. each already individually hot.

### Measured, 7,822 cross-family sub-threshold pairs over 20 structures

| | |
|---|---:|
| nearest cross-family separation, median | 3.75 Å (p25 2.50, p75 5.40, p90 7.08) |
| that pair's concerns, median | **0.210 and 0.190** — sum 0.40, short of 0.5 |
| **too weak to reach 0.5 even coincident** | **61.5%** |
| strong enough but too far apart | 23.6% |
| within reach at σ = 2 | 14.9% |

Widening σ from 2 to 8 raises "within reach" only from 14.9% to 36.0% — and would take figure
B's half-maximum radius from 2.35 Å to 9.42 Å, destroying the result that carries the section.
The asymptote is ~38.5%, because the other 61.5% are ceiling-limited.

**The spatial ingredient is present** — neighbours at a median 3.75 Å, comfortably inside a
6 Å neighbourhood. It is the deposition model that cannot use it.

---

## The hotspot field

### Definition

$$\lambda(\mathbf{x}) \;=\; \sum_i s_i \, K\!\left(\frac{\lVert \mathbf{x}-\mathbf{x}_i\rVert}{R}\right)
\qquad R = 6\ \text{Å}$$

with `K(0) = 1` and compact support at `R`. Default kernel **Epanechnikov**, `K(u) = 1 − u²`
for `u ≤ 1`: smooth, standard in density estimation, and finite-support so "within 6 Å" is
literally true rather than approximately true.

`xᵢ` is the event's atom set as before, so localization is untouched — the same events, the
same atoms, a different way of accumulating them.

### The unit: flagged-outlier-equivalents

`K(0) = 1` and the cut-at-1.0 calibration together fix the scale with no new convention:

| situation | reads |
|---|---:|
| one flagged outlier, isolated | **1.0** |
| two mild concerns of 0.2, adjacent | 0.4 |
| ten mild concerns of 0.2 in one pocket | **~1.5–2.0** |
| a flagged outlier plus five mild neighbours | ~2 |

*"There is as much trouble in this pocket as two flagged outliers"* is a sentence a reader can
check. The anchor is inherited from whoever defined each community threshold, exactly as in the
concern calibration — nothing here is fitted.

Note what changed and what did not. The **ceiling on a single pair is still there** (two 0.2s
read 0.4, correctly less than one real outlier). What lifts is the ceiling on *many*: because
the kernel transmits most of an event's severity across the whole neighbourhood instead of
killing it at 2 Å, ten weak problems genuinely sum. That is the effect an outlier list cannot
reproduce, and it is inter-residue by construction.

### Display contract

This is **not concern** and must not be coloured as if it were. Separate quantity, separate
scale, separate name — `hotspot_density`, units *outlier-equivalents*:

```text
0.0   transparent
1.0   knee       — as much trouble here as one flagged outlier
2.5   saturated  — more than any single residue can account for
```

**Two anchors, both inherited, answering different questions.**

*Knee at 1.0.* Visibility starts where a neighbourhood holds one flagged outlier's worth of
trouble — so **a severe lone outlier still reaches the map, deliberately.** Multi-residue
accumulation is *one* argument for having a field, not its entrance requirement; a "where to
look" tool that hid the most obvious problems would be perverse. Measured over 67,292 residues
in 46 structures: 18.3% carry any concern, and **4.8% of all residues reach 1.0 unaided** —
the navigation budget this implies.

*Saturation at 2.5.* Set at the level a **single residue essentially cannot reach alone**, so
full intensity marks precisely the phenomenon only this field can show. Measured as per-residue
total (max within family, summed across families) over 12,287 concern-carrying residues:
median 0.505, p99 1.879, p99.9 2.253, and **8 of 12,287 (0.065%) reach 2.5 unaided**. It also
lands near p93 of the measured envelope density, so a criterion derived from single-residue
exclusion and one derived from "how much can a user visit" agree, having come from different
data.

The shelf in that distribution at exactly 1.0 — p90 1.000, p95 1.022 — is the cut-at-1.0
calibration working as designed: a quarter of concern-carrying residues have one flagged
outlier and nothing else, so they land on 1.0 precisely.

**Re-measure both anchors if the calibration or the family taxonomy changes.** They are
consequences of those, not independent of them, and both changed twice during development.

Absolute, fixed, identical in every structure — the same rule the concern contract already
states. Component fields are retained per family so any bright region can be attributed.

---

## Absolute, not null-calibrated

Both were considered. **Absolute wins on failure modes**, not on rigor.

Null-calibration asks *"is this neighbourhood worse than the rest of this structure?"*, and
that question answers backwards twice:

* **a pristine structure still shows hotspots** — the relatively-worst parts of an excellent
  model, where there is nothing to fix, so the tool wastes the user's time;
* **a uniformly bad structure shows nothing** — everything is equally bad so nothing stands
  out, precisely when the overlay is most needed.

Absolute gets both right: "nothing to see here" is a legitimate answer, and a bad structure
lights up everywhere because it *is* bad everywhere.

It also keeps a rule already on the books. The display contract in `AGENTS.md` and `README.md`
forbids min/max, percentile, sigma-scaled and viewport-relative colouring: *"the same concern
value must have the same color and opacity in every map."* Null-calibration is
structure-relative by construction, so adopting it would be reversing a stated decision, not
filling a gap.

**The null still has a job — as evidence, not as calibration.** Set the scale by the
outlier-equivalent anchor, then use the spatially matched null (figure C's machinery) to show
that regions above the threshold carry more than chance would put there. That is
falsify-don't-fit applied exactly as before: the null tests a stated convention instead of
defining it, and the number that ships is the number that gets defended.

---

## Known risks

**Packing bias is the main one.** A 6 Å ball in the buried core contains more atoms, therefore
more events, therefore higher intensity — from packing alone, not from being worse. This is the
same hazard as figure C's co-localization, and it must be measured before the field is
believed: compare intensity against local atom density, and check that hot regions are not
simply dense regions. If they are, the fallback is per-atom rather than per-volume
normalization, which stays absolute.

**Broad low-level warmth.** A large structure with many scattered mild problems may read warm
everywhere without anywhere being worth visiting. Measure the distribution of hot volume
against structure size before choosing where the knee sits.

**Bandwidth is genuinely ours.** Unlike every other constant in this project, 6 Å is not
inherited from a community threshold. It is chosen from the measured clustering (median nearest
cross-family neighbour 3.75 Å, p75 5.40, p90 7.08) so that the neighbourhood covers observed
co-location without reaching across the domain. It should be stated, fixed once, and **not
tuned per figure** — that is where this design would start becoming a fitted metric.

**Two fields is a UI cost.** Two overlays, two legends, one more thing to explain. Worth it only
if the hotspot field shows something the locator does not; if measurement says otherwise, drop
it rather than ship both.

---

## What does not change

* the locator field, its σ = 2, its calibration, and its display contract;
* figures A and B, which measure the locator and whose anchors reproduce exactly;
* the cut-at-1.0 calibration, which this design depends on for its unit;
* `max` as the locator's cross-metric rule, and the finding that family/p-norm accumulation
  adds nothing there.

---

## Implementation and validation

1. `density.py` beside `field.py`: Epanechnikov intensity at R = 6 Å, per family and combined,
   sharing the locator's grid so the two are voxel-comparable.
2. **Packing-bias check first**, before any figure: intensity vs local atom density across the
   50-structure set. If the correlation is strong, switch to per-atom normalization and re-check.
3. Corpus run over the existing 2,000-structure sample, reporting hot-volume distribution and
   its dependence on structure size, to place the knee.
4. Falsification with the held-out-channel null: do regions above 1.0 outlier-equivalents carry
   more held-out clash than the matched null? Compare against the locator field's 1.92×.
5. Only if 4 clears: a figure showing a region the hotspot field marks and the locator does
   not, with the per-family decomposition beside it.

Step 4 is the gate. If the hotspot field's regions are no more informative than the locator's,
this is two fields for one field's worth of information and should not ship.

---

## Measured: the packing-bias gate (step 2), 46 structures, 2026-08-05

`corpus/packing_bias.py`. The control is the identical kernel over heavy atoms, weight 1.

| | |
|---|---:|
| Spearman(intensity, packing) | **0.714** (p10 0.654, p90 0.781) |
| atom density, hot set vs envelope | **1.55×** (p90 1.99×) |
| after per-atom normalization, Spearman | **0.310** |
| hot-set overlap between the two definitions | 66.5% |

**Two findings, one of them a design error.**

**1. The knee was in the wrong place, and that is my error.** It was anchored at "one flagged
outlier's worth" without checking what a typical 6 Å neighbourhood already holds. Measured
distribution inside the envelope: **median 0.55, p95 2.89, p99 4.54, max 8.34**. A knee at 1.0
marks **33% of the envelope** — a third of the protein, which is not a hotspot map. The *unit*
is sound and interpretable; as a *threshold* 1.0 is meaningless and belongs near p95, ~3.0.

**2. Packing dependence is real and only half-removable.** Per-atom normalization cuts
Spearman from 0.714 to 0.310 — a large reduction, not an elimination. Hot regions are
systematically 1.55× denser than the envelope.

### What this does not yet distinguish

Some correlation with packing is *legitimate*: a buried core genuinely has more clashes and
more strain, so trouble really is denser there. The residual 0.310 could be real signal or
residual artifact, and the aggregate number cannot tell them apart.

**The diagnostic that would:** run the correlation channel by channel. Clash should track
packing (a buried atom has more neighbours to clash with — real), while Ramachandran and
rotamer should not (backbone and side-chain conformation are not obviously a function of local
density). If the residual is carried by clash it is signal; if rama and rota correlate with
packing just as strongly, it is artifact and the field is measuring the wrong thing.

### Status

**Not resolved, and step 4 is still the gate.** Two non-inherited constants now exist —
bandwidth 6 Å and a knee near 3.0 — plus an unsettled choice between per-volume and per-atom.
That is exactly the parameter accumulation this project exists to avoid, so no more should be
added before the falsification test says whether the field finds anything the locator does not.
If step 4 fails, none of these choices matter.

---

## Head-to-head: the density field loses (2026-08-05)

Both fields through **identical figure A/B/C code**, same 46 structures, same events, same
calibration, same 0.5 threshold — so only the accumulation differs.

| | concern (existing) | density (proposed) |
|---|---|---|
| **A** recall rama / rota / clash | 1.0000 / 1.0000 / 0.9999 | 1.0000 / 1.0000 / 1.0000 |
| **A** precision | 0.255 / 0.431 / 0.204 | 0.116 / 0.170 / 0.100 |
| **B** median distance | 1.38 / 1.43 / 1.93 Å | 2.33 / 2.43 / 2.58 Å |
| **B** p99 | 2.78 / 2.68 / 3.73 Å | 4.08 / 4.12 / 4.83 Å |
| **C** observed vs null | 2.06× vs 0.99× = **2.07×** | 1.65× vs 1.04× = **1.57×** |
| **C** p < 0.05 | **78.8%** of structures | 69.0% |

Recall ties. The density field is **worse on every other axis**: half the precision, ~1.7×
the spatial error, and — the one that decides it — **less informative about the held-out
channel**, 1.57× against the concern field's 2.07×.

The mechanism is dilution, and the packing diagnostic named it independently: **mean severity
per event correlates −0.047 with packing**, so a density-hot region is one holding *more*
events, not *worse* ones. Spreading severity over a 6 Å neighbourhood buys recall that was
already 1.0 and pays for it in everything else.

The only axis where density wins is that figure C is computable on more structures (42 of 46
against 33), purely because its regions are larger and more of them clear the ≥50-atom
inclusion rule. That is a property of region size, not of quality.

### Decision: one field, and it is the existing one

The density field is not better as a single field and there will not be a second one. The
accumulation it was built to provide is real but small (measured earlier: no gain over `max`),
and it costs precision and informativeness that are measured and larger.

### The caveat on this comparison

Both fields were run at threshold 0.5. The density field's natural display knee is 1.0, where
it would mark less volume and likely score better on B and C. It was not run there — because
that threshold has its own defect: a lone flagged outlier peaks at *exactly* 1.0, so recall
becomes a coin flip on grid read-back, the same failure the clash calibration had. The density
field is squeezed between the two: 0.5 dilutes it, 1.0 breaks its recall. Running the 1.0
variant would close this off properly, and until it is run the comparison is one operating
point rather than a curve.

### What survives

The packing question is answered and is **not** a problem: the 0.714 correlation is carried by
clash (rho 0.697) while conformation channels barely track packing (rama 0.139, rota 0.172) —
real physics, since a buried atom has more neighbours to collide with. So the existing
envelope-matched null in figure C is adequate, and no packing-matched null is needed.
