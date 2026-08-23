# Outstanding

Open items as of 2026-08-23, after the 2,000-structure co-locality, sub-threshold and recall
runs completed. Ordered by what would block or embarrass a paper submission, not by effort.

> **Read [CONCLUSIONS.md](CONCLUSIONS.md) first** for what the results *mean*. This file is the
> task list; that one is the reasoning.

## Blocks the paper

**1. `field.py` normalises each event by a single divisor set by its densest atom.** An event
whose atoms are unevenly spread draws its tight part at full strength and its isolated atoms at
a fraction of it — measured directly at **0.31 against 0.98** for a cluster of eight with two
atoms placed 12 Å away. `compute_field` computes one `P = max over atoms of sum_b exp(-d²/2σ²)`
and applies `w = severity/P` to every atom equally, so the divisor is set by the densest region
and under-weights everything outside it.

This is the sole cause of every recall miss in the corpus: 79 atoms of 874,978, ranked exactly
by footprint spread — clash straddles two residues (misses in 2.4% of structures), rotamer
reaches down long Arg/Lys sidechains (0.6%), and the compact-footprint channels miss nothing.
It is also why `bond` (9.6%) and `angle` (3.6%) had to be dropped from the recall figure.

The fix is **per-atom normalisation**: `w_a = severity / S_a` with `S_a` that atom's own local
neighbour sum. Every implicated atom then reads ≈ severity, a uniform cluster still reads
severity rather than summing above it, and a lone concern-1.0 event still peaks at exactly 1.0.
It touches all nine channels and every figure, so it needs its own verification run (~7 h wall
on 6 shards) rather than being folded into anything else.

**Decided to ship recall as-is first** — the published figure is 99.99% and states this
limitation — so this is a field improvement, not a correction to a published number.

## Fixed 2026-08-06

**2. `FIGURES.md`'s figure C anchor. FIXED — the claim is retired, not restated.** The 6.1×
enrichment reproduced only on the uncalibrated `--heavy-atom-clashes` preview path (5 clash
events on 1TEC against 96 on the calibrated path). The corpus number against the spatially
matched null is 1.92× vs a 0.98× null over 1,324 structures.

Two later measurements closed the claim rather than rescuing it: regions the field marks that
contain no flagged outlier carry **0.86×** enrichment — below the null — and different kinds of
problem do not co-locate (matrix 0.8–1.2 outside the backbone block, Clark-Evans cross-kind
R = 0.978, Jaccard 0.000), so there is little for a held-out channel to be enriched by.
`figure_data.py` skips figure C by default (`--no-figure-c`). **The project no longer makes a
predictive claim**; the sub-threshold argument replaced it.

**3. The clash calibration collided with the display threshold. FIXED.** Concern was linear to
`CLASH_SATURATION_OVERLAP_A = 0.80`, so MolProbity's 0.40 Å cut landed at concern exactly 0.50
— the visibility threshold — and ~40% of flagged clashes fell under it. The anchors are now
0.30 Å → 0, 0.40 Å → 1, matching every other channel. **Clash recall went 0.532 → 1.000**
(median, 99.99% of atoms) with rama and rota bit-identical. `calibration_cuts()` asserts all
eleven cuts land at 1.0.

**4. `model_has_hydrogens` accepted "any H present". FIXED.** probe2 requires *both* polar and
non-polar hydrogens, so a deposit carrying only some — polar-only, a few on waters, a ligand
modelled with H in an otherwise heavy-atom protein — passed the check, skipped reduce2, and
died inside probe2 with a message indistinguishable from a real extraction failure. The test now
requires a C-bound and an N/O/S-bound hydrogen. 1fca went from failed to ok.

**5. The covalent roll-up discarded the atoms of every non-worst flagged restraint. FIXED.**
`_worst_per_residue` kept each residue's worst bond/angle event *and only that restraint's two
or three atoms*. Bond recall was below 1.0 in 39% of structures (p05 0.500, min 0.200), with hit
counts on exact integer fractions of the flagged count. The severity roll-up is untouched — still
max, never sum, which is what Rule 6 is for — and only the footprint is unioned over the
residue's flagged restraints. Verified 0.200/0.231/0.333/0.333 → 1.000 on the four worst, with
`angle` (the channel the roll-up protects) unchanged at 1.00–1.04×.

**6. omega recall was scored against cis-proline. FIXED.** omegalyze flags every non-trans
peptide, so ordinary cis-prolines arrived flagged while `_omega_concern` scores them 0.0 on
purpose. Recall read 0.000 on every structure whose only flagged peptides were cis-Pro. The
count set aside is recorded per structure as `n_excluded` (2,324 across the corpus). **omega is
excluded from the published recall figure anyway** — its 1.000 depends on that exclusion, which
is a judgement about what counts as a problem rather than a measurement.

## Fixed 2026-08-04

**7. Rotamer events implicated backbone hydrogens. FIXED.** A hydrogen now inherits its parent
heavy atom's classification via `hydrogen_parents()`. Applied to both copies of
`validation_events.py`. Verified on 6cg7 / 9sf2 / 1kws: backbone-H implication went from 15/15
and 1/1 to 0. Corpus figures unaffected — they are evaluated on heavy atoms only.

**8. Upstream probe2 crash on a parentless atom. PATCHED LOCALLY.** Guarded in
`/root/phenix/modules/cctbx_project/mmtbx/programs/probe2.py`. See
[UPSTREAM_BUGS.md](UPSTREAM_BUGS.md) — why H placement emits an unlinked atom is still open.

**9. Stale `extract_rsr` docstring in the pipeline copy. FIXED.**

## Corpus coverage

**10. 35 structures excluded by the 50,000-atom cap**, plus 124 failures on the nine-channel
run. **108 of those failures are restraint interpretation**, which runs only because `bond` and
`angle` were requested — excluding those two channels puts failures at ~16. The size cap biases
coverage against large complexes, which is where a "where to look" overlay arguably matters
most. Recoverable with a low-concurrency pass.

**11. The corpus is not a random sample of the PDB.** `pdb_population.txt` was pre-filtered for
tractability on another machine, and this project then required a polypeptide entity. Both
filters are documented in [README.md](README.md); neither should be described as random.

**12. Clash cannot be measured by the neighbourhood design.** Rolled up per residue at the
0.40 Å contact cut it flags a median 29.4% of residues (p90 62.8%, max 77.8%) against 1.5–3.6%
for the other channels, so the design's precondition — a flagged set sparse enough to leave a
far field — fails outright, and the bin counts show it (residue counts peak at 6 Å and fall
instead of rising). Matching the other channels' selectivity needs a 0.80 Å threshold, which is
inventing a cut rather than inheriting one. **Excluded from the co-locality findings** and
reported as a limitation. It remains in the recall figure, which the field ships it for.

## Housekeeping

**13. `test_hotspots_standalone.py` was deleted upstream** when `python/tests` was removed for
the cctbx test convention. It covered *this* subproject (`make_concern_maps`,
`calibration_cuts`, `events._load_shared`, `run_corpus`); the converted
`python/pxviewer/regression/tst_hotspots.py` covers the viewer's own `hotspots.py` instead. Port
it to `regression/tst_hotspots_standalone.py` under the `tst_` convention (no pytest) so the
coverage is not silently lost. A copy of the 127 lines is preserved in the commit that deleted
it.

**14. `hotspots/run_corpus.py` is superseded** by `corpus/figure_data.py`. Decide: delete, or
keep as the lightweight QC runner it was. Its only test went with item 13.

**15. `salvage_observations.py` exists only for data written before the gzip-append bug was
fixed.** New runs write one file per process. Delete once no legacy run output matters.

**16. Cosmetic:** the voxel-cap skip reason renders numpy integers verbatim
(`grid [np.int64(...), ...]`). One `int()` in the f-string.

**17. Tests cannot run on this machine.** Phenix python has no pytest; the miniforge python has
pytest and cctbx but segfaults during field generation (numpy 1.26.4 / scipy 1.18.0 ABI
mismatch). Decided: leave unrun here rather than mutate the Phenix install. Note the repo has
since moved to the cctbx `tst_` convention, which does not need pytest — so a ported test would
be runnable here.
