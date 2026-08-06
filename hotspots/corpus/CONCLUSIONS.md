# Conclusions — interpretation checkpoint, 2026-08-05

*What the 2,000-structure corpus run means, as opposed to what it measured. The numbers live
in [README.md](README.md) and `output/figures2000/figures.json`; this file is the reading of
them. It is a **checkpoint, not paper text** — written while the results are fresh so the
reasoning survives, and expected to be revised.*

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
