# Conclusions — interpretation log, 2026-08-05 → 2026-08-23

*What the corpus runs mean, as opposed to what they measured. This file is a **running log, not
paper text** — sections are dated and later ones supersede earlier ones. Read to the end before
acting on anything near the top.*

> ### Status of the earlier sections, as of 2026-08-23
>
> | § | topic | status |
> |---|---|---|
> | 1–2 | A and B are a faithfulness check | **Stands.** Corpus recall is now 99.99% of 874,978 flagged atoms over 1,828 structures — see §10 |
> | 3 | precision should probably not appear | **Settled: dropped.** And for a sharper reason than prevalence — see §10 |
> | 4 | clash calibration is a decision to make | **Settled: made.** Anchors moved to 0.30/0.40 Å; clash recall 0.532 → 1.000 |
> | 5 | figure C is the only non-tautological figure | **Retired.** The claim did not survive measurement — see §8b and §10 |
> | 6 | do not overclaim the corpus | **Stands**, and applies harder now |
> | 7 | the whole thing in three sentences | **Superseded by §10** |
> | 8, 8b | the visual justification | **Stands.** Capability real, on regions carrying nothing |
> | 9 | problems cluster by kind, not across | **Stands**, and is now measured through space — see §10 |
>
> The numbers cited below §10 come from `output/figures2000b/` (recall),
> `output/nbhd_matrix2/` (co-locality) and `output/subvis2000/` (sub-threshold).
> **`output/figures2000/` is the superseded run** — its clash numbers predate the calibration.

---

## 1. Figures A and B are a faithfulness check. Their value is negative-space.

Recall of 1.0000 (rama) and 0.9999 (rota) does not say the overlay is *good*. It says the
deposit-and-read-back round trip at 1.0 Å loses nothing. That is necessary, not impressive —
`FIGURES.md` is right that A and B largely measure the point-spread function of our own kernel.

Their real function is to **license the operating point**: σ = 2 Å, threshold 0.5, 1.0 Å
sampling. A shows that combination loses no flagged problem; B shows it does not point
anywhere wrong. Had recall come in at 0.9 the overlay would be broken and no other figure
would matter. Report them as the licence, not as the finding.

## 2. Lead with figure B. The numbers are tighter than they look.

Median 1.48 Å, p99 under 2.9 Å for both geometry channels. That deserves interpreting rather
than reporting: for a Gaussian read at threshold 0.5, the half-maximum radius is

    σ · sqrt(2 ln 2) ≈ 1.177 σ ≈ 2.35 Å

So the observed hot region is **tighter than a single isolated splat**, because most hot voxels
arise where neighbouring splats overlap, near the atoms themselves. The field's spatial extent
is the kernel and nothing else: no drift, no spurious signal, no misplacement anywhere in 22
million voxels.

**It is not a vacuous test.** Measured against flagged outliers alone the same figure grows a
22.85 Å tail. B *can* fail and does not, which is what makes it defensible under review.

*Honest caveat:* because the answer is the kernel, B is ultimately a statement that σ = 2 Å is
well matched to protein geometry (~3.8 Å between adjacent Cα). That is design justification,
not discovery. Say so rather than letting a reader infer more.

## 3. Precision should probably not appear in the paper.

rama 0.285 vs rota 0.506 reads as "rama is twice as bad". It is not. Rama outliers are 0.73% of
atoms against rota's 3.47%, and the concern curve legitimately marks everything below the 2%
favored boundary — so **the rarer the outlier, the more of the hot region is correctly hot but
unflagged**. Precision here measures prevalence, which is the same confound that ruled out the
ROC figure.

Figure B is the identical question asked in honest units. De-emphasise precision or drop it,
and let the distance histogram carry "is the blur acceptable".

## 4. The clash channel is a design decision the paper must make, not a result.

Clash recall of 0.61 is not the field failing. The 0.40 Å MolProbity cut maps to concern
**exactly 0.50** — the visibility threshold — so ~40% of flagged clashes sit under it by
construction.

The downstream consequence is larger than the recall number. The combined map is a voxel-wise
maximum, so **a flagged rotamer reaches 1.0 while a flagged clash reaches 0.5: geometry always
outranks sterics in the "where to look" field.** That is a substantive claim about what a user
sees, and it is currently an inherited accident rather than a choice. `concern.py` names it as
"worth revisiting; not worth changing silently"; it now has corpus evidence.

**Resolve before publication.** Either recalibrate clash so its community cut lands at 1.0 like
rama and rota, or state the operating point explicitly and own it.

*Unexplained, and the one number in B I cannot currently account for:* clash's per-structure
max reaches 7.80 Å, well beyond the geometry channels. The likely cause is that clash deposits
at the **contact point** while figure B measures distance to flagged *atoms*, giving a
systematic offset of roughly half a van der Waals overlap — but that is a hypothesis, not a
verified explanation. Check it before the number appears anywhere.

## 5. Figure C is the only figure that tells you something you did not put in.

1.92× enrichment with the null at **0.98×**. *The null is the result.* `FIGURES.md` warned that
co-localization — both signals living on atoms in a protein-shaped region — would manufacture
enrichment for free. A correctly built spatially-matched null must land on 1.0, and it does.
That closes the stated hazard, and it is the single strongest piece of evidence in the section.

**1.92× is modest, and correctly so.** It does not support "our overlay predicts clashes". It
supports exactly the licensed claim: a highlighted region carries roughly twice the base rate
of problems *beyond* the one that highlighted it, so following the overlay is worth the user's
time. Keep it in navigation language and fit nothing to it.

Two limitations to state plainly rather than bury:

* **Only 62.7% of structures reach p < 0.05 individually.** The claim lives in the corpus
  distribution, not in any single structure — so do not show one structure as if it were the
  evidence.
* **552 of 1,876 structures (29%) are excluded** by the inclusion rule, lacking clash outliers
  or a large enough held-out region. The claim is therefore conditional: *in structures that
  have both geometry problems and clashes.*

**Known gap.** Only clash-as-target was run. `FIGURES.md` asks to repeat with other held-out
channels; holding out rama and asking about rota (or the reverse) is a different and arguably
harder test, since those two are more plausibly correlated through backbone strain. Cheap now
that the harness exists, and worth doing before submission.

## 6. Do not overclaim the corpus.

It is **not a random sample of the PDB**: pre-filtered for tractability on another machine,
then restricted to entries declaring a polypeptide, then size-capped at 50,000 atoms — which
excludes exactly the large complexes where a "where to look" overlay is most useful. 1,877 of
2,000 succeeded.

None of this undermines A/B/C. But "we evaluated on 1,877 protein structures" overclaims;
"on a tractability-filtered sample of small-to-medium protein entries" is what we have. The
size cap is the one worth actually fixing — it is cheap to recover with a low-concurrency pass
and it removes the most awkward sentence in the methods.

## 7. The whole thing in three sentences

> The overlay is spatially faithful: every hot voxel in 22 million sits within about a residue
> of something genuinely concerning, and it loses no flagged geometry problem at its shipping
> operating point. Regions it highlights carry roughly twice the base rate of *other* problems,
> against a spatially-matched null that correctly reads 1.0 — so following it is worth a user's
> time, which is a navigation claim and not a metric. The clash channel's calibration currently
> places its outlier cut at the visibility threshold, which both suppresses ~40% of flagged
> clashes and lets geometry outrank sterics in the combined field; that needs a decision before
> publication.

## Actions this implies

| | action | why |
|---|---|---|
| 1 | Decide the clash calibration | Blocks publication; changes what the combined field shows |
| 2 | Restate `FIGURES.md`'s figure C anchor | Its 6.1× is from the uncalibrated clash path |
| 3 | Verify the clash 7.80 Å figure-B max | Only unexplained number in B |
| 4 | Run figure C with other held-out channels | `FIGURES.md` asks for it; harness makes it cheap |
| 5 | Recover the 35 size-capped structures | Removes the weakest sentence in the methods |
| 6 | [../AGGREGATION_PROPOSAL.md](../AGGREGATION_PROPOSAL.md) — **tested, hypothesis failed** | Cross-metric accumulation adds nothing; `max` stays. Read the "Measured outcome" section |
| 7 | [../HOTSPOT_DENSITY_DESIGN.md](../HOTSPOT_DENSITY_DESIGN.md) | The successor design: a second, density-based field that can accumulate across residues. Gate is step 4 |

---

## 8. The field's justification is visual, and it is now measured (2026-08-05)

Every accumulation test in this project asked *does a voxel cross the display threshold*. **A
volume render never asks that question.** It integrates opacity along a view ray,
`alpha = 1 - exp(-k * integral of concern)`, so faint concern that never crosses 0.5 can still
composite into something you can see. Measuring accumulation in field space was the wrong test
for a visualization claim; `corpus/alpha_accumulation.py` measures it in image space.

The marker comparison is definitional rather than modelled: along a ray that never crosses the
threshold, a MolProbity marker representation shows **nothing** — markers exist only where a
validator flagged something. So the only question is whether the field shows something there.

Measured over 46 structures, 136 views, with `k` fixed *before* seeing results so a lone
flagged outlier reads alpha = 0.6:

| | |
|---|---:|
| envelope rays that never cross the threshold | **18.2%** |
| line integral, clean envelope ray | 0.19 |
| line integral, sub-threshold ray | **2.91** |
| line integral, ray crossing the threshold | 12.95 |
| **contrast above background, median** | **0.383 alpha** |
| p90 | 0.645 |
| rays clearing the 0.05 visibility floor | **92.7%** |

Robust across a 4× sweep of `k` (median contrast 0.215 → 0.597), so the conclusion does not
rest on the calibration constant.

**So the claim holds.** On 18.2% of the structure — nearly a fifth — markers show nothing and
the field composites faint concern into a contrast roughly 7.7× the conservative visibility
floor. That is a real capability specific to a translucent volume, and it is the one thing here
that discrete markers structurally cannot do.

**It is also the claim this project came closest to discarding on bad evidence.** It was
reported dead twice on the strength of threshold-crossing tests that could not see it.

### What this does not establish

**Visible is not the same as worth seeing.** This measures the physics of compositing, not
whether a user benefits. Two gaps remain, and the second is the important one:

* *Perception.* 0.383 alpha is far above any plausible just-noticeable difference, so
  "noticeable" is safe; "useful" is not the same claim and would need a user study.
* *Informativeness.* Are the sub-threshold-only regions worth visiting, or is the field
  visibly rendering noise? Partial evidence exists and it is weak: regions built only from
  never-flagged observations carried 1.23× held-out enrichment against a 0.97× null
  (`corpus/accumulation.py`). Real, but well below the 2.07× the threshold-crossing regions
  manage. **The honest statement is that the field visibly shows something markers cannot, and
  that something carries weak but non-zero independent signal.**

### 8b. …but what it uniquely shows is not worth seeing

The other half, measured (`corpus/subthreshold_value.py`, 46 structures, clash held out of the
field). Sub-threshold regions are defined by *distance* — faint voxels more than 6 Å from any
marked voxel, so the set excludes the halo of problems the markers already show.

| region | share of volume | obs/null | p<0.05 |
|---|---:|---:|---:|
| **sub-threshold only** | 39.4% | **0.86×** | 5% |
| marked | 60.6% | **1.48×** | 42% |

**The faint regions carry no signal — they are marginally *depleted*, and significant at chance
rate.** In hindsight this is what should have been expected: clashes concentrate where geometry
is genuinely bad, and a sub-threshold region is by definition one where geometry is mildly off
but *not* bad. It is the "fine, actually" part of the structure.

### So the defence fails on its second leg

Two turns of argument reduce to one sentence: **the field can show what markers cannot, and
what it uniquely shows is not worth looking at.** A capability with no demonstrated benefit.

Caveats, because the result should not be overstated either:

* This tests one held-out channel. Faint geometry strain might predict something else — poor
  map fit, say — which no map-free corpus can check.
* "Worth seeing" is operationalised as *contains a held-out clash outlier*. A user might value
  seeing mild strain for its own sake. But that is a much weaker claim than "shows places worth
  visiting", and a table of mildly-strained residues would serve it.

### What survives, and what the paper should claim

Three expansions of the field have now been measured and all three failed: cross-metric
accumulation (adds nothing over `max`), the density construction (dominated at every operating
point, its unique volume near-null), and sub-threshold value (0.86×). One thing measured
positive: faint concern really does composite into a visible contrast (0.383 alpha) — but on
regions that carry nothing.

The pattern is consistent and worth stating plainly: **the field is a good renderer of what
validation already found, and every attempt to make it find more has failed.** That is exactly
what `FIGURES.md` claimed from the start — a visualization layer, not a new validation score —
and the honest paper claims precisely that and no more:

* it loses no flagged problem at its operating point (figure A);
* it puts hot voxels within about a residue of what they represent (figure B);
* the regions it marks carry ~2× the base rate of *other* problems (figure C);
* it renders through occluding geometry, which per-atom colouring cannot.

Nothing about accumulation. Nothing about finding what validators miss.

---

## 9. Measured: problems cluster with their own kind, not across kinds (2026-08-06)

`corpus/clustering.py`, 46 structures. Clark-Evans ratio `R = observed mean nearest-neighbour
distance / null mean`, where the null re-places every event on a **randomly chosen heavy atom
of the same structure** — the control that matters, since events can only occur where atoms
are, and a uniform-box null would report the shape of the protein as clustering.

| severity | neighbour | observed | null | R | clustered in |
|---|---|---:|---:|---:|---:|
| flagged | any kind | 3.92 Å | 4.45 Å | **0.825** | 98% |
| flagged | cross-family | 7.40 Å | 7.26 Å | 0.945 | 67% |
| sub-threshold | any kind | 2.99 Å | 3.21 Å | **0.925** | 93% |
| sub-threshold | cross-family | 4.18 Å | 4.16 Å | **0.978** | 67% |

**This corrects an earlier claim in this document and in discussion.** Sub-threshold problems
are *not* spatially random — they cluster, in 93% of structures. A back-of-envelope Poisson
estimate suggested otherwise and was wrong; the difference is the null, since a Poisson process
in a box does not know that atoms are unevenly distributed.

### The statement that survives, and it explains everything else here

> **Validation problems cluster strongly with their own kind and barely at all across kinds.**

Every result in this document follows from it, quantitatively:

* **Within-metric accumulation works** and the field already does it — same-kind events cluster
  (R = 0.825 / 0.925), which is why a floppy loop renders as one region rather than three dots.
* **Cross-metric accumulation fails** because at R = 0.978 there is nothing to accumulate. Not
  a kernel-width problem, not a threshold problem: the arrangement a cross-metric field needs
  does not exist in the data, so no combination rule could have produced it.
* **Figure C's 2.07× and the sub-threshold 0.86×** fall straight out: cross-family clustering
  is weak-but-real for flagged events (0.945) and absent for mild ones (0.978), so marked
  regions predict other problems modestly and faint regions predict nothing.

There is a design irony worth keeping. The family taxonomy — max *within*, accumulate
*across* — is structurally right and exactly inverted from where the signal sits. The
redundancy is within a family, where the max is correct; the accumulation is across families,
where there is nothing to add.

This also bounds any future hotspot field: **no combination rule can extract cross-metric
coincidence that is not in the data.**

---

## 10. Through space, and below the cut (2026-08-23)

Section 9 measured clustering with Clark-Evans on 46 structures. This is the same question
asked properly — on 2,000 structures, on one axis, with a positive control — plus the two
things that turned out to matter more.

### The instrument was certified before the negatives were believed

Every through-space result here is a negative or near-negative, and a negative is worth
nothing unless the instrument would have reported a positive. So a halo was planted in real
coordinates: random centres in 2.5% of residues, **+0.15 concern within 5 Å**, sequence-far
only, on a background of 0.30 — a signal of exactly **1.50×**. It recovered **1.52×**, with the
4–6 Å bin returning 1.13 rather than 1.50 because the halo stops at 5 Å and that bin is half
outside it. Correct dilution, right amplitude. `corpus/synthetic_control.py`.

Planting at 1.50× was deliberate: it is the size of the real effect. A control planted at 5×
would have proved only that the instrument sees loud things.

### Outliers are not islands — but the extent is in sequence, not space

Elevation around a flagged residue is 4.7–8.3× at ±1, **1.35–2.51× at ±3**, 1.26–1.85× at ±5.
The ±1 column proves little (φ/ψ of residue *i* uses atoms of *i±1*); ±3 is the real signal.

Through space, once chain neighbours are excluded, the near bin does not fall to 1.0 — it falls
**below** it, to 0.26–0.72. Residues packed tight against an outlier but far from it in sequence
are slightly *better* modelled than average. What survives is 1.51–1.64× in the 4–6 Å bin, and
only for the three backbone-conformation channels. Rotamer, Cβ and omega are flat at 1.08–1.12×.

**A problem's extent is real and it is mostly an extent along the chain.** That is the shape a
field should be depicting.

### The cross-kind matrix, and a confound worth recording

The metric-by-metric matrix must be measured **sequence-far**. At 2–4 Å over all neighbours it
looked spectacular — rama→cablam 5.93, omega→ca_geom 7.59, the whole table warm — and it was
measuring validator coupling: adjacent residues share atoms *across* channels as well as within
them, since rama and omega read the same peptide and cablam and ca_geom are both built from Cα
geometry. Every cell falls 3–5× once chain neighbours are excluded.

Corrected, the diagonal holds (rama 1.64, ca_geom 1.56, cablam 1.51) and everything outside the
backbone block sits at 0.8–1.2. This agrees with §9's Clark-Evans (cross-kind R = 0.978) instead
of contradicting it, which is the check that decided it. **The backbone block is one kind
measured three ways, not three kinds co-locating.**

The rota row is the cleanest negative in the project: 0.82–1.27 across the board, own diagonal
1.08, on the largest sample (35,620 outliers, 1,340 structures). Rotamer problems are islands.

### What the field adds, and the number that nearly went in wrong

The mean concern near an outlier is elevated 1.35–2.51× — and the absolute means are
**0.03–0.11**, which would render as nothing. A mean cannot tell *every neighbour faintly warm*
(a haze worth nothing) from *97% at zero and a few at 0.6* (a handful of clearly-drawn
residues). Counting by band separates them, and it is the second: near a rama outlier **4.09%**
of non-outlier neighbours carry half an outlier or more, against **1.42%** at random — 2.87×.
ca_geom and omega reach 3.38×.

Across the corpus the field draws **157,760 residues where a markup draws 81,686** — 76,074
sub-threshold additions, a **1.93×** population. About one in five of the additions sits beside
a flagged outlier at ~3× the background rate.

**The honest counterweight belongs in the paper, not a footnote:** ~80% of what the field adds
is nowhere near a flagged outlier. The co-locality argument justifies a fifth of the addition.
The rest is measured but unexplained.

### Why precision is dropped — a sharper reason than §3 gave

§3 argued precision measures prevalence. True, but the decisive objection is different:
Ramachandran precision of 0.162 implies the field adds **6.2×** what the markup shows, against
**2.12×** measured by counting residues. The gap is the kernel's own width — σ ≈ 2 Å is wider
than the 3.8 Å Cα spacing, so an isolated outlier marks its neighbours' *atoms* above 0.5 even
when those residues carry no concern at all. Precision counts blur as an addition and would
have overstated the field's contribution threefold. The residue count is blur-free by
construction and carries the claim alone.

### The recall figure earns its place by failing

99.99% — 874,899 of 874,978 flagged outlier atoms, 1,828 structures, four of six channels
exact. That number is not evidence for anything; the field is built from these events, so 1.000
is the expected result. **Its value is that it broke three times.** Clash at 0.532 against
rama's 1.000 exposed the calibration; a hydrogen check that accepted "any H present" let
partially-hydrogenated deposits skip reduce2; and the covalent roll-up drew 8 atoms where 30
were flagged. None would have been found by reading the code.

The 79 remaining misses share one cause — the single-divisor normalisation in `field.py`, still
open as item 1 in [OUTSTANDING.md](OUTSTANDING.md).

### The statement now

> **Problems cluster with their own kind, mostly along the chain. Different kinds do not
> co-locate. What a field adds over a markup is not accumulation — it is the continuous
> sub-threshold value of a single channel, which is real, sparse, and about twice the
> population a threshold shows.**

This closes the last route back to a predictive claim. §5 hoped figure C would show the field
knows something it was not told; §8b showed its unique regions carry nothing (0.86×, below
null); and the corrected matrix explains why — at 0.8–1.2 cross-kind there is nothing for a
held-out channel to be enriched by. **No combination rule can extract cross-metric coincidence
that is not in the data**, and none of the field's value depends on it.
