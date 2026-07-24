# Validation hotspots — design notes

Status: **design only, nothing built.** This records the reasoning behind a proposed
per-residue "hotspot" score that aggregates several validation metrics into one field you
can color by, so the eye goes straight to the places worth rebuilding.

## The idea

Aggregate the validation metrics we already compute — restraint geometry deviation,
Ramachandran, rotamers, clashes, map fit — into a single per-residue number, and color the
model by it. Red means "look here."

The need is real. A crystallographer at the screen does not want six lists to cross-
reference; they want to know where to go next. No viewer really ships this.

The naive version — average the metrics — does not survive contact with the problem. What
follows is the case against it and the design that comes out the other side.

## Three objections to the naive version

### 1. Averaging is the wrong operator for outlier hunting

The metrics are unequal in severity. One 0.5 Å clash is a real error; averaged against five
clean metrics it becomes a mild orange smudge.

A mean is a low-pass filter, and this is an outlier detector. The two are in direct
opposition. Whatever combines the metrics must **preserve severity**, which means something
max-flavored, not mean-flavored.

### 2. Restraint deviations are partly a readout of your refinement weights

This is the objection that most shapes the design.

Refinement actively minimizes geometry deviation while trading it against map fit, and the
X-ray/geometry weight decides how error distributes between the two. Refine tightly and the
geometry outliers vanish *by construction* while the error relocates into the residuals.

So a score mixing "deviation from ideal geometry" with "fit to density" is partly measuring
how the weight was set, not how good the model is. Ramachandran, rotamers and clashes are
cleaner precisely because they are conventionally left unrestrained — which is why
MolProbity leans on them.

**Consequence:** bond/angle deviations stay out of the default score until we can show they
add signal rather than echo the refinement weight.

### 3. "Worst" is not the useful question

Color raw badness and the surface loops and high-B regions light up. Every time, in every
structure. That is true and useless — you already knew those were floppy.

The information is in **badness beyond what is expected there**, given local B-factor, local
map power, and resolution. The hotspot worth having is not "this is bad," it is "this is
worse than it has any right to be."

## The scaling and weighting problem

Two questions, which turn out to be one:

- how do you put a 0.03 Å bond deviation on the same scale as a 0.4 Å clash?
- do you use the boolean outlier flag, or the underlying continuous value?

### Every metric is natively continuous; the boolean is a downstream cut

- `ramalyze` returns `res.score` as a **percentage** — the probability density at that φ/ψ
  against the reference ensemble. The 0.05% outlier flag is applied to it afterward.
- `rotalyze` likewise (0.3% cut).
- Probe returns clash **overlap in Å**; the 0.4 Å cut is downstream.
- Q-score is continuous with no natural threshold at all.

So this is not a choice between two representations of equal standing. The boolean is lossy,
derived, and re-derivable from the field; the field can never be recovered from the boolean.

**But** the threshold is where the calibration lives. The 0.05% cut is not arbitrary — it
encodes "here is where structures start being wrong," fit against a large curated reference
set. Raw density values are not commensurable across metrics. The cuts are.

### Use the threshold as the unit, not the output

Define each metric's severity so it equals **1.0 exactly at its community outlier cut**, 0 at
unremarkable, >1 beyond. Everything becomes dimensionless, and **no weight is invented** — the
calibration is inherited from MolProbity and wwPDB, which is worth far more than anything
hand-tuned here.

The principled form is a **surprisal**:

```
sᵢ = −log₁₀ P(this bad or worse | the structure is correct)
```

Everything becomes decades of surprise under a common null hypothesis. A Ramachandran
outlier lands at −log₁₀(0.0005) ≈ 3.3; a rotamer outlier at ≈ 2.5. That asymmetry is correct
and comes for free — a Ramachandran outlier genuinely is rarer and more damning, and the
reference data says so rather than us deciding it.

For rama and rota this is arithmetic, not research: the percentage is already returned.

### Combining: one dial between the two poles

```
S = ( Σᵢ sᵢᵖ )^(1/p)
```

- `p → ∞` — max: pure severity-preserving, the boolean-flavored end
- `p = 1` — sum
- `p ≈ 3–4` — behaves like max, with a modest bonus when several metrics fire together

A single parameter spans the whole space between "worst offense" and "average," and it can be
tuned against real structures instead of argued about.

**The trap it avoids:** a genuinely misbuilt residue fires Ramachandran *and* rotamer *and* a
clash — one physical error, three correlated readouts. Summing triple-counts it, so a residue
with one severe problem ranks below a residue with three mild ones. `p ≈ 4` rewards
corroboration without triple-counting.

### Modulate by data support

Severity alone still ranks flexible surface loops top, forever. What you actually want is
**wrong *and* well-supported by data** — where something incorrect has been confidently built
into real density. A rotamer outlier at B=90 in mush is a shrug; the same outlier in strong,
well-resolved density should glow.

```
hotspot = S × confidence
```

with confidence from Q-score / local map power / B. This is the difference between "the model
is worst here" (already obvious) and "the model is most confidently wrong here" (not
eyeballable).

## Design rules

**The aggregate navigates; the components diagnose.** Color draws the eye; hover/click shows
which metric fired and by how much. The score ranks, it never concludes. This is also what
keeps it honest.

**Booleans live in the badges, not the score.** Continuous field for color and ranking;
booleans for the countable annotations we report and cross-check against the PDB validation
report. They stay consistent because the boolean is simply the level set at `severity = 1.0`.

**Per-residue is the display unit.** A residue is what you rebuild. Compute per-atom where the
metric is per-atom (clash, Q-score), display per-residue as the max over its atoms.

## Prior art

- **MolProbity score** — already a log-weighted aggregate of clashscore + rotamer +
  Ramachandran, calibrated so it reads as "the resolution at which this quality would be
  typical." It is *global*. This proposal is essentially **localizing MolProbity score to a
  per-residue field**, which nobody has really shipped in a viewer. (Check the exact
  coefficients against MolProbity source before relying on them — not trusted from memory.)
- **MolProbity multi-criterion kinemage / chart** — closest existing tool. Note what it does:
  *superimposes* markers, does not average them.
- **wwPDB validation report per-residue plots** — per-chain strips marking residues by their
  issues rather than merging the underlying values.

## How this differs from wwPDB validation and MolProbity

The point of this section is to be clear about what is genuinely new here and what is just
re-plumbing something that already exists. Most of it is re-plumbing. Two things are not.

|                        | MolProbity score            | wwPDB validation report                            | this proposal                      |
| ---------------------- | --------------------------- | -------------------------------------------------- | ---------------------------------- |
| granularity            | global (one number)         | global sliders + per-residue strips                 | per-residue continuous field       |
| inputs                 | clash, rotamer, Ramachandran| R/Rfree, clashscore, rama, rotamer, RSRZ            | rama, rotamer, clash, map fit      |
| value type             | rates of boolean events     | percentiles; per-residue outlier flags              | continuous surprisal               |
| combination            | weighted log-sum (additive) | count of outlier categories per residue             | p-norm, `p ≈ 4`                    |
| weights                | fitted to track resolution  | not combined at all                                 | none — inherited from thresholds   |
| resolution-normalized  | yes, by construction        | yes for percentiles and RSRZ; **no** for rama/rota %| yes throughout                     |
| data support           | absent                      | RSRZ, reported separately                           | multiplies geometry severity       |
| purpose                | judge / compare structures  | judge / report                                      | navigate within one structure      |

### The closest existing thing is RSRZ

wwPDB's real-space R Z-score is per-residue, continuous, and normalized against what is
expected at that resolution. That is exactly the philosophy argued for above — "surprise, not
badness" — and it is already standard practice.

The gap is that **RSRZ does it only for map fit.** Geometry is still reported as boolean
outlier flags. A fair one-line summary of this proposal is: *apply RSRZ's treatment to the
geometry metrics too, and put the result on one scale.*

### wwPDB's per-residue aggregate is a boolean count

The residue-property strips color a residue by how many outlier *categories* it trips. That is
an aggregate, and it is precisely the design argued against above: counting booleans discards
severity entirely, so a 0.45 Å clash and a 1.2 Å clash are the same pixel, and a residue with
three marginal flags outranks a residue with one catastrophic one.

Useful for a report you have to defend and audit. Not useful for deciding where to click.

*(Verify the exact color bands against a current wwPDB report before matching them — the
0/1/2/3+ scheme here is from memory.)*

### MolProbity already excludes bond/angle RMSD — precedent for objection 2

The MolProbity score is built from clashscore, rotamer and Ramachandran, and pointedly leaves
out restraint-geometry RMSD, which the wwPDB report does show (as bond/angle RMSZ). The usual
justification is the one given above: those terms are restrained during refinement, so they
report on the restraints and their weight rather than on the model.

So objection 2 is not a novel worry — it is the existing consensus, and we are following it
rather than departing from it.

### What is actually new

1. **Per-residue, continuous, across *all* the metrics at once.** MolProbity score is global.
   RSRZ is per-residue but map-fit only. wwPDB per-residue is boolean. Nothing currently
   combines calibrated continuous severity across geometry *and* fit at residue granularity.
2. **Multiplying severity by data support.** No existing metric asks "is this residue wrong
   *and* well-supported by density." wwPDB carries RSRZ alongside geometry flags but never
   crosses them. This is the part most likely to surface things a crystallographer would not
   otherwise find, and also the part with no prior calibration to lean on.

Also new, though more of a design choice than a contribution: **no fitted weights.** MolProbity
fits coefficients so the score tracks resolution; here the relative importance falls out of
each metric's own reference distribution. Different philosophy — theirs is calibrated to a
target, ours is calibrated to nothing but the null hypothesis.

### Consistency constraint (testable)

Because our booleans are the level set at `severity = 1.0`, the outlier counts we display
**must reproduce MolProbity's** — same residues flagged, same totals. If they diverge, our
severity mapping is miscalibrated, not MolProbity.

This is worth an actual test: run both over a structure and assert the flagged sets match.

### X-ray vs cryo-EM: the fit term forks

Q-score is a cryo-EM metric (Pintilie; now in EMDB validation). For X-ray the established
per-residue fit measures are RSRZ / RSCC. So `confidence` — and any map-fit severity component
— needs a per-experiment implementation rather than one shared path.

### One thing we lose

MolProbity score and wwPDB percentiles are standard, so people compare them across structures
and across papers. A bespoke score is not comparable to anything.

The surprisal scale is *absolute* rather than relative to the current structure, so in
principle it stays comparable across structures — but only if the calibration is right. Worth
protecting deliberately: resist any temptation to normalize the color ramp to the current
structure's own min/max, which would destroy exactly this property (the same reason Q-score is
colored on a fixed 0–1 domain rather than a stretched one).

## The hazard

Color a structure red and people will refine until it is green. Over-restrain and the score
improves while the model gets worse — a textbook Goodhart target, and the metric would be
screenshotted and put in papers.

If this gets built, the components breakdown is **not optional**; it is the thing that stops
the number from being gamed.

## What exists here already

Most ingredients are in place:

- `python/pxviewer/validation/` — `ramachandran.py`, `rotamers.py`, `cablam.py`,
  `cbetadev.py`, `omegalyze.py`, `rama_z.py`
- `python/pxviewer/probe.py` — clashes/contacts
- `python/pxviewer/qscore.py` — per-atom map fit
- `python/pxviewer/geometry.py`, `edits.py` — restraints

Delivery is solved: the Q-score work built the per-atom attribute path — `session.set_attribute(name, values)` + `session.color_by(name, palette=…, domain=…)` pushes a float array to the frontend's `pxviewer-attribute` theme. A hotspot score is one more array through the same pipe.

## Proposed first pass

Continuous surprisal + p-norm over the four metrics that are cleanest and already available:

1. Ramachandran (percentage → surprisal)
2. Rotamer (percentage → surprisal)
3. Clash (overlap Å, 0.4 Å ≡ severity 1.0)
4. Map fit — Q-score against the expected-Q-vs-resolution curve for cryo-EM; RSRZ/RSCC for
   X-ray (see the fork noted above)

Validate it against the existing tools before trusting it: the `severity = 1.0` level set must
reproduce MolProbity's flagged residues exactly, and the hotspots it ranks highest should be a
superset of what the wwPDB per-residue strips already mark — if it misses something the
boolean count catches, the severity mapping is wrong.

Leave as badges rather than score components:

- **cablam** — Ramachandran is not per-residue independent, and cablam exists precisely
  because rama alone misdiagnoses backbone problems at low resolution
- **cbetadev**, **omegalyze** — near-binary in practice, so they want floor treatment rather
  than a smooth ramp
- **bond/angle deviations** — objection 2 above; add only if shown to carry signal
  independent of the refinement weight

## Open questions

- Where do the reference tail probabilities come from for clash overlap? rama/rota ship
  theirs; clash would need calibrating against a set of high-resolution structures, or
  anchoring at the 0.4 Å cut with an assumed shape.
- Does `confidence` multiply, or is it better as a second visual channel (e.g. severity to
  hue, confidence to opacity) so the two are not conflated into one number?
- Resolution normalization: normalize each component, or normalize `S` once at the end?
- Should `p` be exposed to the user, or fixed after tuning?
- Rama/rota percentages are **not** resolution-normalized (unlike RSRZ and unlike MolProbity
  score's fitted calibration). Do we normalize them ourselves, and against what reference?
- Is there any value in also reporting a global roll-up of the hotspot field, or does that
  just reinvent MolProbity score worse? Leaning: don't — the global question is answered, and
  a second incomparable global number is exactly the Goodhart risk above.
