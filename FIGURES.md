# Figures: what to compute, and what not to claim

A handoff for whoever runs this at corpus scale. pxviewer is a **visualization tool**; the
hotspot overlay is a light, deliberately simple view showing where several validators agree.

**The framing is not decorative — it decides which figures are legitimate.** An earlier aim,
to make the overlay a defensible validation *metric*, was dropped (see `HOTSPOTS.md`,
"Direction not taken"). So the figures that would have supported it are out of scope: no
calibration-holds-across-the-PDB panel, no co-occurrence-against-a-null, no PDB-REDO region
recovery, no ROC. Those defend a number. We are not proposing a number.

What is in scope is **rendering fidelity**: when the overlay sends a user somewhere, is there
something there, and how far off is it? That is a visualization claim, answerable in
angstroms, and it cannot be misread as a metric.

## The figure set

Screenshots, produced through the headless path — no corpus needed:

1. The application, whole window.
2. Architecture, with the wire made explicit: topology once, coordinates continuously.
3. **Live difference map during a drag** — before / mid-drag / after. The capability that
   most visibly cannot work without the streaming architecture.
4. One scene, several object types: model + 2Fo−Fc + Fo−Fc + a reflections-derived map.
5. Coloring channels on fixed domains: element, B-factor, occupancy, Q-score.
6. **Why a 3-D field at all** — per-atom colouring on a cartoon (problem invisible: no side
   chains drawn, buried atoms behind the surface) beside the translucent field (you can see
   through to it). The argument is entirely about seeing.
7. The aggregation is trivial and decomposable: combined overlay beside the per-channel
   overlays. Showing that the aggregation is dumb is the honest move.

Measured, needs a size ladder but no validation:

8. Responsiveness vs structure size: frame-update time against atom count, full frames vs
   deltas, plus tug round-trip latency. Cheap — no reduce2, no probe2.

Measured, needs the corpus — **specified in full below**:

- **A.** Does the overlay lose anything? (operating point)
- **B.** When it points somewhere, how far off is it? (spatial error)
- **C.** Is a flagged region worth visiting for reasons beyond the one that flagged it?

## What already exists here

**Copy `python/pxviewer/validation_events.py`** — one file, no pxviewer imports. It owns which
residue a result belongs to and which atoms it implicates. Reimplementing that localization is
how two projects silently disagree about *where* a problem is. `VALIDATION_NOTES.md` covers it.
`check_field_agreement` in it already computes most of A and B, including the
`worse_than_percent` predicate figure B depends on.

**`hotspots/hotspots/run_corpus.py` is a starting point, not a mandate.** It already does
sharded parallelism (`--shard K/N`), resume, per-model failure isolation, one JSON line per
model appended as it goes, and a `--report` merge — and it QCs each model with the agreement
check rather than trusting that "no exception" means "correct". It has no per-observation
dump, which every figure below wants. Keep it, extend it, or throw it away; it is here so the
wheel does not get reinvented, not because it has to survive.

`make_concern_maps.generate()` is the same code path the CLI runs, exposed as a function with
a `fields_out=` argument that hands back the in-memory fields — so a driver can sample them
without re-reading CCP4s.

---

## Figure A — the operating point

**Claim:** the overlay does not lose flagged problems at the threshold it actually ships at.

**Compute.** Per structure, per channel: sample the concern field back at every atom, then at
the display threshold (0.5) record recall (fraction of flagged outlier atoms the field marks)
and precision (fraction of marked atoms that are flagged). Report the corpus distribution of
recall — a violin or ECDF per channel — and name the structures where recall < 1.

**Recall is the number that matters.** A field that loses a real outlier is wrong.

> **Corpus result, 1,840 structures, all nine channels: 99.9953% — 1,157,192 of 1,157,246
> flagged outlier atoms. Eight of the nine are exact in every structure.**
>
> | channel | structures | flagged atoms | recalled | exact in |
> |---|---:|---:|---:|---:|
> | rama | 793 | 23,555 | 100.00% | 100.0% |
> | cablam | 1,623 | 80,559 | 100.00% | 100.0% |
> | ca_geom | 1,265 | 24,241 | 100.00% | 100.0% |
> | rota | 1,584 | 145,861 | 100.00% | 100.0% |
> | cbeta | 570 | 3,337 | 100.00% | 100.0% |
> | omega | 348 | 2,426 | 100.00% | 100.0% |
> | bond | 1,263 | 92,633 | 100.00% | 100.0% |
> | angle | 1,525 | 142,266 | 100.00% | 100.0% |
> | clash | 1,830 | 642,368 | 99.99% | 97.7% |
>
> The 54 lost atoms are all clash, and the cause is a design choice rather than a defect:
> `_clash_event` deposits at the **contact point** between the two atoms, because the interface
> is where the problem physically is, while recall measures distance to the flagged *atoms* —
> about half a van der Waals overlap away. Moving the deposit onto the atoms would improve this
> number and make the picture worse. It is the same mechanism behind clash's 7.80 Å figure-B
> maximum, which this file previously listed as unexplained.
>
> Two caveats belong beside the table. **omega** recalls 1.000 only once ordinary cis-prolines
> are set aside (613 across the corpus): omegalyze flags every non-trans peptide and the
> calibration scores cis-Pro 0.0 on purpose, so that exclusion is a judgement about what counts
> as a problem, not a measurement. **bond and angle** require restraint interpretation, which
> failed on 106 structures — without those two channels total failures would be ~18 rather than
> ~124, so their rows rest on a slightly different and non-randomly selected sample.

**Precision is not reported.** It was, and it should not be: on the corpus, Ramachandran
precision is 0.162, which implies the field adds 6.2× what the markup shows — against 2.12×
measured by counting residues directly. The gap is the kernel's own width. σ ≈ 2 Å is wider
than the ~3.8 Å between adjacent Cα atoms, so an isolated outlier marks its neighbours' atoms
above 0.5 even when those residues carry no concern at all, and precision counts that blur as
an addition. Use the residue-level count instead: it is blur-free by construction.

**Not a curve.** No ROC: positives are ~0.7% of atoms for Ramachandran, and ROC is insensitive
to prevalence, so it reads ~0.999 for a field that is only mediocre. Measured on 1TEC:

| channel | positive atoms | ROC-AUC | PR-AUC | precision @0.5 | recall @0.5 |
|---|---:|---:|---:|---:|---:|
| rama | 0.73% | 0.9988 | 0.730 | 0.588 | 1.000 |
| rota | 3.47% | 0.9923 | 0.662 | 0.657 | 0.989 |

The ROC column is why there is no ROC figure. If a curve is wanted anyway, use PR.

---

## Figure B — spatial error, and the one that carries the section

**Claim:** when the overlay marks a place, the thing it is marking is within about a residue.

**Compute.** Per structure, per channel: take every voxel at or above the display threshold,
find the distance to the nearest atom the field was built from, and pool the distances across
the corpus into a histogram in angstroms. Report the median, p90, and max.

**Get the target set right — this is the whole figure.** Measure distance to atoms that are
*concerning*, not to atoms that are *flagged outliers*. The concern curve starts rising at the
2.0% favored boundary, well before the 0.05% outlier cut, so the field legitimately marks
residues MolProbity never flags. Pass `worse_than_percent(2.0)` as the predicate. Measured on
1TEC, the choice changes the answer completely:

| channel | target set | median | p90 | max |
|---|---|---:|---:|---:|
| rama | flagged outliers only | 1.57 Å | 10.85 Å | 22.85 Å |
| rama | **concerning (< 2%)** | **1.20 Å** | **1.86 Å** | **2.74 Å** |
| rota | flagged outliers only | 1.71 Å | 5.61 Å | 10.83 Å |
| rota | **concerning (< 2%)** | **1.41 Å** | **2.12 Å** | **2.92 Å** |

Against outliers alone the figure grows a 23 Å tail and looks broken. Against the set the
field was actually built from, **every hot voxel is within 2.7 Å of it** — which is the
result, and it is a good one.

**Why this figure and not precision.** Precision at 0.5 is 0.59 for Ramachandran on 1TEC (0.162
across the corpus), which sounds poor and is not: the remainder are overwhelmingly neighbours of
concerning residues, because the σ ≈ 2 Å splat is wider than the ~3.8 Å between adjacent Cα
atoms. A PR curve counts the blur as error and makes a correct field look mediocre. A distance
histogram shows the blur sitting where it should, in physical units. That is the difference
between defending a number and describing a rendering.

---

## Figure C — is a flagged region worth visiting?

**Claim:** a region the overlay highlights often contains problems beyond the one that
highlighted it — so it is useful as a navigation aid, not merely a redraw of one list.

**Compute.** Hold a channel out. Build the field from Ramachandran + rotamer only, sample at
atoms, and ask what fraction of atoms above the threshold carry a **clash** outlier, against
the base rate. Report the enrichment against a spatially matched null. Repeat with other
held-out channels; report the corpus distribution.

> **Measured, and it does not support the claim. Do not quote 6.1×.**
>
> This section previously read "clash outliers are 0.365% of atoms overall and 2.222% in the
> held-out hot region — a **6.1× enrichment** (n = 180 hot atoms)". That base rate reproduces
> to four significant figures only on the **uncalibrated** `--heavy-atom-clashes` preview path,
> which finds 5 clash events on 1TEC. On the calibrated hydrogen path — the one every result
> in this project uses — 1TEC has 96 clash events and a base rate of 6.2843%. The 6.1× was an
> artefact of the preview path and has been removed rather than restated.
>
> The corpus number, over 1,324 structures with the spatially matched null this section
> already required, is **1.92× against a 0.98× null**. That is a real but small effect, and it
> is not the figure this section was written to be.
>
> Two later measurements closed it. Regions the field marks that contain *no* flagged outlier
> carry an enrichment of **0.86×** — below the null, so the field's unique regions carry no
> signal. And different kinds of problem do not co-locate: the metric-by-metric neighbourhood
> matrix is 0.8–1.2 outside the backbone block, Clark-Evans cross-kind R = 0.978, residue-level
> Jaccard 0.000. There is little for a held-out channel to be enriched *by*.
>
> `corpus/figure_data.py` therefore skips this figure by default (`--no-figure-c`); pass the
> flag's inverse to regenerate the number. **The project no longer makes a predictive claim.**
> What replaced it is the sub-threshold argument: the field draws 76,074 residues at half an
> outlier or worse that no markup shows, roughly doubling the population worth a look.

**A and B are fidelity checks, not discoveries.** Both measure how much the Gaussian destroyed
— the point-spread function of our own kernel — which is information we put in ourselves. That
is worth reporting as verification and worth nothing as evidence. Figure A earns its place by
*failing*: clash recall sat at 0.532 against Ramachandran's 1.000 and exposed a mis-calibration,
and two further defects surfaced the same way. Corpus recall now stands at **99.99%** of
874,978 flagged outlier atoms over 1,828 structures.

**Two hazards, both serious.**

*The null.* Both signals live on atoms, in a protein-shaped region, so some enrichment comes
free from co-localization rather than from any real relationship. A naive label shuffle is not
a valid null. Use a spatially matched one — resample the held-out hot volume at random
positions within the same molecular envelope, preserving volume and shape — and report
enrichment against *that*. This is why the preview number was uninterpretable as well as wrong:
measured against a spatially matched null the corpus figure is 1.92× against a 0.98× null.

*The framing.* This is one sentence away from being a metric claim. "Our score predicts
clashes" is exactly the reading that was removed from this project. The legitimate statement
is about the view: *regions the overlay highlights frequently contain more than the one thing
that highlighted them, so following it is worth the user's time.* Keep it in the language of
navigation, and do not fit anything to it.

---

## Corpus notes

- Cheap: A, B, and the rama/rota half of C need no hydrogens and no probe2 — a few seconds per
  structure. Only clash-as-target in C needs reduce2 + probe2 (~35–45 s per medium structure).
- Compute the per-observation dump once — structure, chain/resseq, metric, native value,
  concern, outlier flag, and the sampled field value — and make all three figures from it
  afterwards. Do not couple the run to a figure you may restate.
- Output pixel size changes the answer. At 2.5 Å the field starts losing flagged outliers and
  at 3.0 Å loses more than half; see `hotspots/README.md`. Generate at 1.0 Å for anything the
  figures depend on, not the 2.0 Å fast-viewing default.
- Anchor values above are 1TEC at 2.0 Å, core channels. Reproduce them before trusting a
  corpus pipeline.
