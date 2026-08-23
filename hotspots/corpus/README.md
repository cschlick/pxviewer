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

## Nine-channel survey — 50 structures, 2026-08-05

`channel_survey.py` over a seeded 50-structure draw (seed 20260805), all nine channels,
46 ok. Median 46.5 s/structure with restraints built.

| channel | family | deposit% | outlier% | hot% of box |
|---|---|---:|---:|---:|
| rama | backbone | 4.7 | 0.27 | 0.04 |
| rota | sidechain | 14.2 | 3.31 | 0.35 |
| clash | sterics | 19.5 | 8.42 | **4.12** |
| cablam | backbone | 6.4 | 1.57 | 0.19 |
| ca_geom | backbone | 4.8 | 0.41 | 0.07 |
| cbeta | cbeta | 2.0 | 0.19 | 0.01 |
| omega | omega | 5.4 | 0.08 | 0.01 |
| bond | covalent | 3.0 | 0.97 | 0.05 |
| angle | covalent | 9.1 | 3.79 | 0.25 |

**Clash dominates**: 4.12% of the box against 0.35% for the next channel. Any aggregate is
mostly clash unless that is deliberately handled.

### The family grouping is supported, with two corrections the data forced

Residue-level Jaccard, within-family vs across-family: **median 0.088 within, 0.000 across**
(across-family maximum 0.055). Channels in a family really do mark the same residues, which is
what justifies combining them by `max` rather than accumulating them.

Two members of the proposed taxonomy did not survive contact with the measurement:

* **cbeta** has Jaccard **0.000** with both bond and angle — no shared residues at all —
  against 0.047 with rota. It is the backbone–sidechain junction, not a restraint deviation.
  Given its own family.
* **omega**'s within-backbone Jaccards (0.037 rama, 0.053 cablam, 0.045 ca_geom) sit at or
  *below* the across-family maximum, so it is not redundant with backbone conformation either.
  Also given its own family. Peptide-bond planarity is a distinct property from phi/psi, so
  the physics and the data agree here.

Final grouping: `backbone` (rama, cablam, ca_geom), `sidechain` (rota), `sterics` (clash),
`covalent` (bond, angle), `cbeta`, `omega`.

> **Do not read the Spearman half of that matrix.** It is computed over the union of residues
> either channel marks, filling zero where one is silent; across largely disjoint supports
> that construction returns negative values for everything and means nothing. Jaccard is the
> measure that answers the question. Left in the output rather than deleted so nobody
> recomputes it and reaches the same wrong conclusion.

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

## Results — 2,000-structure run, 2026-08-03

> **Superseded for clash.** This run predates the clash recalibration, when MolProbity's 0.40 Å
> cut landed at concern exactly 0.50 — the display threshold — so ~40% of flagged clashes fell
> under it and clash recall reads 0.532. It is **1.000** on the current calibration. Three
> further defects were fixed afterwards (see [OUTSTANDING.md](OUTSTANDING.md) items 4–6). The
> current run is `output/figures2000b/`: nine channels, 1,828 structures, **99.99% recall** of
> 874,978 flagged atoms. This data now lives at
> `output/figures2000_SUPERSEDED_pre_clash_calibration/`; the rama and rota numbers below still
> hold.

`sample_2000_seed20260802.txt`, 1.0 Å output pixel, calibrated (hydrogen) clash path.
69.0 h of compute.

**Corpus.** 2,000 attempted → **1,877 ok**, 88 failed, 35 skipped for size. Every structure
has a terminal outcome; nothing was left deferred.

| failure | n | what it is |
|---|---:|---|
| reduce2/probe2 hydrogen | 72 | ligands and modified nucleotides reduce2 will not protonate |
| improper rotation matrix | 13 | deposited symmetry cctbx rejects |
| other cctbx `Sorry` | 2 | invalid atom radius; missing bonding information |
| `AttributeError` | 1 | upstream bug — see [UPSTREAM_BUGS.md](UPSTREAM_BUGS.md) |

All are data properties or upstream defects, not defects here. The 35 skips are the size cap.

### Figure A — the operating point

| channel | structures | pooled recall | pooled precision | recall = 1.0 |
|---|---:|---:|---:|---:|
| rama | 806 | **1.0000** | 0.285 | 100.0% |
| rota | 1,613 | **0.9999** | 0.506 | 99.6% |
| clash | 1,866 | 0.6144 | 0.432 | 0.1% |

Ramachandran loses **nothing** across the corpus; rotamer loses one atom in ~10,000.

**Clash is a calibration artefact, not a field failure** — its 0.40 Å community cut maps to
concern exactly 0.50, the display threshold itself, so roughly 40% of flagged clashes sit
just under visibility by construction. See the finding above.

### Figure B — spatial error (the figure that carries the section)

Distance from every hot voxel to the nearest *concerning* atom, pooled over **22.0 M** voxels:

| channel | voxels | median | p90 | p99 | worst structure max |
|---|---:|---:|---:|---:|---:|
| rama | 1,072,178 | 1.48 Å | 2.28 Å | 2.88 Å | 3.94 Å |
| rota | 4,442,038 | 1.48 Å | 2.17 Å | 2.78 Å | 4.87 Å |
| clash | 16,532,612 | 1.88 Å | 2.88 Å | 3.73 Å | 7.80 Å |

Every channel's p99 is under 3.8 Å — inside a residue. The claim "when the overlay marks a
place, the thing it is marking is within about a residue" holds at corpus scale, and the 1TEC
anchor was not optimistic.

### Figure C — held-out enrichment against a spatially matched null

1,324 structures usable, 552 excluded by the stated rule (≥50 atoms in the held-out region,
≥1 clash outlier, ≥10 null placements).

| | |
|---|---:|
| observed enrichment, median | **1.92×** |
| **null enrichment, median** | **0.98×** |
| observed / null | 1.97× |
| enriched above 1.0 | 91.3% of structures |
| p < 0.05 | 62.6% of structures |

**The null landing on 0.98× is the result to trust.** A spatially matched null must come out
at 1.0 if it is built correctly; that it does is the evidence the 1.92× is not the free
co-localization enrichment `FIGURES.md` warns about. Keep the claim in the language of
navigation — regions the overlay highlights contain more than the one thing that highlighted
them — and do not fit anything to it.
