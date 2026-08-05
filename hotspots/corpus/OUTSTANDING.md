# Outstanding

Open items as of 2026-08-04, after the 2,000-structure figure run completed. Ordered by what
would block or embarrass a paper submission, not by effort.

> **Read [CONCLUSIONS.md](CONCLUSIONS.md) first** for what the results *mean* — which figure to
> lead with, why precision should probably be dropped, and why the clash calibration is a
> decision rather than a finding. This file is the task list; that one is the reasoning.

Two items were added there and are not repeated below: verify clash's unexplained 7.80 Å
figure-B maximum, and run figure C with held-out channels other than clash.

## Blocks the paper

**1. `FIGURES.md`'s figure C anchor is wrong and must be restated.** It reports "clash outliers
are 0.365% of atoms overall … a 6.1× enrichment". That base rate reproduces to four
significant figures on the **uncalibrated** `--heavy-atom-clashes` path (5 clash events on
1TEC) and not at all on the calibrated hydrogen path (96 events, 6.2843%). The corpus figure —
1.92× against a 0.98× null over 1,324 structures — is the calibrated number. Anyone quoting
6.1× is quoting the preview path. See [README.md](README.md).

**2. The clash channel's calibration collides with the display threshold.** Concern is linear
to `CLASH_SATURATION_OVERLAP_A = 0.80`, so MolProbity's 0.40 Å cut lands at concern exactly
**0.50** — the visibility threshold itself — and ~40% of flagged clashes fall under it once the
Gaussian read-back is applied. That is why figure A's clash recall is 0.61 while rama is 1.000.
`concern.py` already calls the asymmetry "inherited rather than chosen … worth revisiting; not
worth changing silently". It now has corpus evidence. Either the calibration moves or the paper
states the clash channel's operating point explicitly; it should not be reported as a field
failure, because it is not one.

## Fixed 2026-08-04

**3. Rotamer events implicated backbone hydrogens. FIXED.** `MAINCHAIN` excludes the backbone
*by name* and lists only heavy atoms, so the amide `H` and alpha `HA` passed the filter and
landed in the "sidechain" — contradicting Rule 2. A hydrogen now inherits its parent heavy
atom's classification via the file's existing `hydrogen_parents()`.

Applied to **both** copies (`pxviewer/python/pxviewer/validation_events.py` and
`map_model_validation/pipeline/lib/validation_events.py`) so they cannot disagree about *where*
a rotamer problem is, which is the whole reason that file is shared. Verified on 6cg7 / 9sf2 /
1kws: backbone-H implication went from 15/15 and 1/1 to **0**, and 6cg7 A LYS23 now carries
`CB CG CD CE NZ` plus only its own sidechain hydrogens.

**The corpus figures are unaffected and need no recompute** — they were evaluated on heavy
atoms only, and heavy-atom sets are provably unchanged (`hydrogen_parents` maps only hydrogens,
so for a heavy atom the new predicate reduces to the original test).

**4. Upstream probe2 crash on a parentless atom. PATCHED LOCALLY.** Guarded in
`/root/phenix/modules/cctbx_project/mmtbx/programs/probe2.py` (branch `cschlick-dev`,
**uncommitted** — review and upstream). The real diagnostic now surfaces, and it answered part
of the open question: the atom is detached at *every* level, not just from its atom group. See
[UPSTREAM_BUGS.md](UPSTREAM_BUGS.md) — the remaining question of why H placement emits an
unlinked atom is still open.

**5. Stale `extract_rsr` docstring in the pipeline copy. FIXED.** It stated
`R = sum|obs - calc| / sum|obs + calc|` while the code computes `sum|obs| + sum|calc|` — it
documented the bug the code had already fixed. Both copies now state the formula they compute.

> Note: 3 and 5 touch `map_model_validation/pipeline/lib/validation_events.py`, which was
> previously to be left alone as in-use. Item 3 is a behaviour change there (rotamer events no
> longer carry backbone hydrogens); item 5 is comment-only. Both are additive corrections, but
> revert them if that project wants its own timing.

## Corpus coverage

**6. 35 structures excluded by the 50,000-atom cap**, plus 88 failures (72 reduce2/probe2
hydrogen refusals on ligands and modified nucleotides, 13 improper rotation matrices, 3 other).
1,877 of 2,000 succeeded. The size cap biases coverage against large complexes, which is where a
"where to look" overlay arguably matters most. Recoverable with a low-concurrency pass (1–2
workers, no memory contention); the cap exists for an 8-shard machine, not for physics.

**7. The corpus is not a random sample of the PDB.** `pdb_population.txt` was pre-filtered for
tractability on another machine, and this project then required a polypeptide entity. Both
filters are documented in [README.md](README.md); neither should be described as random.

## Housekeeping

**8. `hotspots/run_corpus.py` is superseded** by `corpus/figure_data.py`, which took its
failure isolation, resume, sharding and agreement check. It still has a test
(`test_hotspots_standalone.py::test_the_corpus_runner_reports_recall_and_survives_a_bad_model`).
Decide: delete both, or keep it as the lightweight QC runner it was.

**9. `salvage_observations.py` exists only for data written before the gzip-append bug was
fixed.** New runs write one file per process and do not need it. Delete once no legacy run
output matters.

**10. Cosmetic:** the voxel-cap skip reason renders numpy integers verbatim
(`grid [np.int64(...), ...]`). One `int()` in the f-string.

**11. Tests cannot run on this machine.** Phenix python has no pytest; the miniforge python has
pytest and cctbx but segfaults during field generation (numpy 1.26.4 / scipy 1.18.0 ABI
mismatch). Decided: leave unrun here rather than mutate the Phenix install.
