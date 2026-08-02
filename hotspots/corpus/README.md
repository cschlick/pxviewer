# Corpus

Everything that drives hotspot generation over many structures, and the figure data for the
hotspots section of the pxviewer paper. The figure specification is `../../FIGURES.md`.

This directory is deliberately outside `../hotspots/`. That package is the upstreamable
library — five reusable modules and two CLI entry points, importing nothing but cctbx, numpy
and scipy. Corpus driving is not part of that contract, and keeping it here means the library
stays clean when the subproject is split back out.

## `pdb_population.txt` — the frozen population

218,264 PDB IDs, one per line, lowercase, sorted, unique.

```
sha256  623529c31ace9888d5163b6c4a004bb2e46d8a4c2973b6e058322f0b6adcee9f
```

**This file is the reproducibility anchor and must not be regenerated in place.** Every
sample the figures use is drawn from it, so changing it silently changes what "seed 12345"
means. If the population ever needs to grow, add a *new* file beside this one and record
which figures used which.

**Provenance.** Supplied 2026-08-02 as PDB entries known not to cause memory or parsing
failures during a prior run on another machine. It is therefore already filtered for
tractability, and that filter is not characterized here — it is not a random sample of the
PDB, and no claim should describe it as one.

**Coverage, verified 2026-08-02** against the local mirror at `/root/data/pdb_mmcif`
(256,448 entries, 85 GB, ~350 KB/entry):

| | count |
|---|---:|
| listed and present on the mirror | 218,264 |
| listed but **missing** from the mirror | 0 |
| on the mirror but excluded from the list | 38,184 |

## Why there is no seeded-sampling machinery

A seeded draw is only reproducible if the population is. The live mirror is an rsync target
that grows — 1,137 entries changed in the month before this was written — so sampling
directly from a directory listing would silently return a different set on every re-run.

Freezing the population into the file above removes that problem entirely, and with it the
need for anything cleverer: plain `random.seed(N); random.sample(...)` over a fixed, committed
list is exactly reproducible. Hash-based membership was considered and dropped as unnecessary
complexity once the population stopped moving.

Two hazards the frozen list does **not** cover, because a PDB ID is not a content identifier:

* **Revision in place** — an entry can be re-refined and re-released under the same 4-char
  code, with different atoms and therefore different outliers. Same ID, different answer, and
  nothing looks wrong.
* **Obsoletion** — an entry can be withdrawn and disappear from the mirror.

Both are handled by snapshotting and checksumming the *drawn sample* rather than the whole
population, which is cheap: at ~350 KB/entry a 5,000-structure sample is ~1.7 GB.

## Cost

Measured on this machine (12 cores, 23 GB RAM), 1TEC at 1.0 Å output pixel size:

| channels | per structure |
|---|---:|
| rama + rota (no hydrogens, no probe2) | 11.4 s |
| + clash (reduce2 + probe2, the calibrated path) | 37.3 s |

1TEC is 279 residues, so both figures are optimistic floors rather than corpus medians.
Running the full 218,264-entry population is **not** free at this scale — roughly 3 days on
10 shards for the cheap channels alone, ~9 days including clash — so the figures are computed
on a sample, and the sample size is chosen from a size-stratified timing measurement rather
than from 1TEC.

Generate figure data at **1.0 Å**, not the 2.0 Å fast-viewing default: at 2.5 Å the field
already loses one flagged outlier in five. See the pixel-size table in `../README.md`.

## The harness

```
screen_population.py   population -> entries that declare a polypeptide entity
draw_sample.py         seeded draw from the screened population, frozen to a file
figure_data.py         the corpus run: fields in memory, figures A/B/C, per-observation dump
make_figures.py        reduce a run into figures.json (seconds, so a figure can be restated)
```

`figure_data.py` never writes a CCP4. The figures need sampled numbers, and a corpus of maps
nobody opens costs ~9 MB per small structure at 1.0 Å.

### The evaluation universe is heavy atoms only

Hydrogens sit ~1 Å from their parent heavy atom, so at σ = 2 Å they carry no independent
spatial signal — asking whether the field marks an atom 1 Å from another marked atom is not
an independent test. More importantly, including them makes recall mean *different things*
for depositions that ship hydrogens and those that do not, which is a corpus-level confound:
pooling those into one distribution compares unlike things.

Clash survives the restriction because `extract_clashes(inherit_to_heavy=True)` already hands
each hydrogen's clash to its parent heavy atom. Verified on 1TEC: **0 of 96 clash outlier
events lose their heavy atom**, 172 of 281 implicated atoms are heavy.

The restriction applies to the whole universe — flagged set, hot set and sampled values
alike — never to one side of a ratio.

### Anchor reproduction

`FIGURES.md` asks that its 1TEC anchors be reproduced before a corpus pipeline is trusted.
At 2.0 Å, heavy-atom universe, figures A and B match exactly:

| | recall | precision | prevalence | | median | p90 | max |
|---|---:|---:|---:|---|---:|---:|---:|
| rama | 1.000 | 0.588 | 0.73% | rama | 1.20 Å | 1.86 Å | 2.74 Å |
| rota | 0.989 | 0.657 | 3.47% | rota | 1.41 Å | 2.12 Å | 2.92 Å |

## Two findings the anchors did not carry

**1. The clash channel's calibration sits exactly on the display threshold.** Clash concern is
linear with `CLASH_SATURATION_OVERLAP_A = 0.80`, so the 0.40 Å MolProbity cut lands at concern
**0.50** — the display threshold itself. A flagged clash therefore only just reaches
visibility, and the Gaussian read-back (a splat peak falling between voxels recovers ~0.91 of
its amplitude) pulls many of them under it. Measured on 6cg7: of 94 flagged clash heavy atoms,
sampled median 0.590, and only **61.7%** reach 0.5. Rama and rota do not have this problem
because their cut lands at concern 1.0. `concern.py` names the asymmetry and calls it
"inherited rather than chosen … worth revisiting; not worth changing silently" — figure A is
where it becomes visible, and it is a property of the calibration, not of the field.

**2. `FIGURES.md`'s figure C anchor was computed on the uncalibrated clash path.** Its stated
base rate — "clash outliers are 0.365% of atoms overall" — reproduces exactly on the
*heavy-atom* clash pass and not on the calibrated hydrogen-added one:

| clash path | outlier events on 1TEC | flagged heavy atoms | base rate |
|---|---:|---:|---:|
| `--heavy-atom-clashes` (preview) | 5 | 10 / 2737 | **0.3654%** |
| hydrogens + probe2 (calibrated) | 96 | 172 / 2737 | 6.2843% |

So the 6.1× headline rests on **5 clash events in one structure**, from the path `../README.md`
calls "a labeled preview, not the calibrated default". The corpus run uses the calibrated path
and reports a smaller enrichment against a much larger base rate. The anchor needs restating.
