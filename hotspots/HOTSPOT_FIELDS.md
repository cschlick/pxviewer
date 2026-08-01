# Hotspot fields: fixed design and calibration

## Scope

Hotspot visualization is not a new validation metric. It is a way to turn
discrete validation annotations into continuous spatial fields. A clash hotspot,
Ramachandran hotspot, rotamer hotspot, and Q-score hotspot remain separate
scientific quantities. The optional combined map answers only **where should the
user look?** It must not be reported as a model-quality score.

The term *hotspot* is descriptive: a localized concentration of concern. It does
not imply a statistical hotspot test, a p-value for spatial clustering, or a
marked-point-process model.

## Fixed field contract

Every metric adapter produces observations with:

1. a native value;
2. one or more Cartesian locations forming the observation's footprint;
3. a bounded concern value `c` in `[0, 1]`;
4. metadata sufficient to recover the original validation result.

`c = 0` means that the observation is at or beyond the metric's reassuring
anchor. `c = 1` means that it has reached the visualization's saturation anchor.
It does **not** mean that different metrics have become statistically equivalent.

Each observation is deposited on a 1 Å Cartesian grid and convolved with a
peak-normalized Gaussian of sigma 2 Å. For a multi-atom footprint, deposition is
normalized so that the isolated observation peaks at its concern value rather
than growing with residue size. Observations from the same metric add, after
which the complete metric field is clipped to `[0, 1]`:

```text
C_m(x) = clip(sum_i c_mi K_mi(x), 0, 1)
```

The optional combined field is the voxel-wise maximum:

```text
H(x) = max_m C_m(x)
```

Raw metrics are never summed with one another. Multiple observations within one
metric may reinforce a location up to 1; adding another metric cannot make the
combined field exceed 1. The viewer should retain the separate maps so it can
show *why* a combined-map voxel is highlighted.

Do not replace or visually normalize concern values with per-model min-max,
percentile, rank, standard-deviation, histogram, or viewport-relative scaling.
The primary display uses absolute concern directly for both hue and opacity.
Percentile is permitted only as an optional secondary relative-contrast mode.

## MolProbity calibration

MolProbity values are calculated by the existing cctbx analyzers. The hotspot
code does not reimplement validation contours; it converts analyzer results to
concern and projects them into space.

### Ramachandran

Input is the probability-like `ramalyze` score expressed as a percentage.
Concern is linear in log probability between the established favored boundary
and the residue-class-specific outlier boundary:

```text
good = 2.00%
bad  = 0.05%  for general residues
       0.20%  for cis-Pro
       0.10%  for Gly, trans-Pro, pre-Pro, and Ile/Val

c_rama(p) = clip((log(p) - log(good)) / (log(bad) - log(good)), 0, 1)
```

These are the boundaries used by `mmtbx.validation.ramalyze.evalScore`. Thus all
favored observations contribute zero, allowed observations span `(0,1)`, and
outliers saturate at one. The residue-class distinction is mandatory; using a
single 0.05% cutoff misclassifies cis-Pro and the other special classes.

The footprint is the validated residue's `N`, `CA`, `C`, and `O` atoms. This is
a backbone-conformation visualization, not an atom blame assignment.

### Rotamer

Input is the `rotalyze` score expressed as a percentage. The cctbx favored and
outlier boundaries are 2.00% and 0.30%, respectively:

```text
c_rota(p) = clip((log(p) - log(2.00)) / (log(0.30) - log(2.00)), 0, 1)
```

Favored rotamers contribute zero, allowed rotamers span `(0,1)`, and rotamer
outliers saturate at one. The footprint is all non-hydrogen side-chain atoms,
excluding `N`, `CA`, `C`, `O`, and `OXT`.

### Clashes

Input is the magnitude of Probe's negative overlap in Å. Concern is linear and
saturates at 0.80 Å overlap:

```text
c_clash(overlap) = clip(abs(min(overlap, 0)) / 0.80, 0, 1)
```

The 0.80 Å saturation anchor is an explicit visualization choice: twice the
0.40 Å MolProbity clash-reporting boundary. Consequently, a just-reported clash
starts at concern 0.5, a 0.60 Å overlap has concern 0.75, and overlaps of 0.80 Å
or worse have concern 1.

**The sub-threshold gap is closed.** This channel used to have no values between
0 and 0.5, because `mmtbx.validation.clashscore` reports only clashes at or past
the 0.40 Å boundary. Extraction now goes through `probe2` (see
[`hotspots/events.py`](hotspots/events.py)), which reports every contact with a
negative gap, so the tail below the reporting boundary is populated. On 1TEC with
hydrogens, 1380 of 1476 contacts are sub-threshold. Remove the old limitation
wherever it is still repeated.

Hydrogen-added MolProbity clashes are the default: `extract_all` places hydrogens
with reduce2 before probing, on a separate model so the input's residue numbering
is untouched. A heavy-atom-only map is a separately labeled preview
(`--heavy-atom-clashes`) and must not be presented as the calibrated default; the
manifest records which was used in `molprobity.hydrogens`, and `clash_n_atoms` is
the atom count the clashscore was normalized on.

A clash is deposited at Probe's contact point. The shared extractor implicates
both atoms of the pair (and each one's parent heavy atom) and carries the contact
point alongside, so this deposition choice stays here rather than being made by
omission.

## Q-score calibration and integration boundary

We will **not modify cctbx Q-score to generate a spatial field**. The installed
`cctbx.maptbx.qscore.calc_qscore()` already returns `qscore_per_atom` and
`qscore_records`, including atom coordinates. Its command-line program can emit
the same records as JSON and can write `_atom_site.qscore` in a working mmCIF.
That is the correct validator API. Q-score owns measurement; hotspot visualization
owns spatial projection.

The adapter consumes the per-atom Q-score and deposits each value at its atom's
Cartesian position. It does not average to residues first, because doing so would
discard Q-score's native spatial resolution and make large and small residues
behave differently.

Q-score is strongly resolution-dependent, so a universal raw-Q threshold is not
defensible. For proteins, the initial expected value is the regression reported
with the Q-score method:

```text
Q_expected(resolution_A) = 1.1192 - 0.1775 * resolution_A
```

Concern measures a deficit below that expectation and saturates 0.20 Q units
below it:

```text
c_q(Q) = clip((Q_expected - Q) / 0.20, 0, 1)
```

Thus `Q >= Q_expected` contributes zero, a deficit of 0.10 contributes 0.5, and
a deficit of at least 0.20 contributes one. The 0.20 width is a provisional,
declared visualization bandwidth, not a published Q-score outlier threshold. It
must remain configurable and should be checked on the viewer-paper examples.

For nucleic-acid-only or mixed-composition maps, callers should supply an
explicit expected Q appropriate to that use case rather than silently applying
the protein regression. Missing or non-finite Q-scores are omitted. The manifest
must record the resolution or explicit expected Q and the 0.20 saturation deficit.

The Q-score method and its resolution dependence are described by Pintilie et
al., *Nature Methods* 17, 328–334 (2020),
<https://doi.org/10.1038/s41592-020-0731-1>.

## Local map-model correlation: a native voxel field

Local map-model correlation is different from the discrete validators above: it
can be evaluated natively at every map-grid point. We therefore preserve a raw
local-CC field and do **not** deposit atom scores or apply the 2 Å hotspot
Gaussian to it.

The experimental map and a model-calculated map must be on the same boxed grid.
The calculated map uses the supplied resolution and electron scattering unless
the caller explicitly chooses another scattering table. At every evaluated grid
point, the raw field is the Pearson correlation inside a Cartesian sphere:

```text
r = max(2.5 Å, reported resolution)

CC(x) = cov(A, B) / sqrt(var(A) var(B))
```

`A` is the experimental map and `B` is the model-calculated map. The 2.5 Å floor
prevents a nominal support smaller than a credible resolution element. An
explicit radius override is allowed for experiments but must be recorded.

The field is computed from five spherical convolutions—local sums of `A`, `B`,
`A*B`, `A*A`, and `B*B`. This is algebraically the same calculation as
`mmtbx.maps.correlation.from_map_map_atom`; only the evaluation sites differ.
On 29KM, 100 grid points agreed with the built-in pointwise function to maximum
absolute error `1.1e-14`.

Only voxels within `env_r` of a model atom are retained. The default is
`env_r = r`. This envelope is a computational and display mask only: changing it
does not change any retained voxel's correlation. Raw CC is NaN outside the
envelope, and a separate binary envelope map is written for viewers that do not
handle NaNs. Flat spheres with zero variance are also NaN.

Per-atom CC is calculated separately with the built-in pointwise function at the
exact atom coordinates. Residue values, when wanted for tables, are arithmetic
means over their finite non-hydrogen atom values. The voxel field is not obtained
by interpolating those atom values.

The raw Pearson field is the primary scientific output. For inclusion in the
optional combined “where to look” map, use the parameter-free reversal of the
native Pearson range:

```text
C_local_cc(x) = (1 - CC(x)) / 2
```

This maps perfect correlation `+1` to concern 0, zero correlation to 0.5, and
perfect anticorrelation `-1` to 1. It introduces no fitted cutoff and does not
claim that a particular CC is an outlier. Missing/outside-envelope voxels have
concern zero and do not enter the combined maximum.

## Generic metric adapter

A future metric must declare all of the following before it can participate:

- whether high or low native values are concerning;
- fixed good and saturation anchors, with their source or rationale;
- any nonlinear transform, such as logarithmic probability;
- the native observation footprint;
- units and provenance stored in the manifest;
- behavior for missing, infinite, and out-of-range values.

The standard mappings are:

```text
high-is-bad: clip((x - good) / (bad - good), 0, 1)
low-is-bad:  clip((good - x) / (good - bad), 0, 1)
```

An adapter without defensible fixed anchors may generate an individually labeled
experimental layer, but it must not enter the default combined map.

## Output and display

The bounded concern map is the authoritative primary display field for every
metric and for `combined`. It drives both hue and opacity in a direct-volume
representation using one fixed `[0,1]` domain across all metrics and structures:

```text
concern 0.00  fully transparent
low concern  transparent/faint
concern 0.50  yellow
concern 0.75  orange
concern 1.00  red
```

No per-field percentile, min/max normalization, sigma scaling, or
viewport-relative statistics may alter this primary mapping. The same concern
value must look the same in every structure. Maps use Cartesian placement via
CCP4 `NXSTART`.

Output sampling is a performance control. Deposited metric fields default to
1 Å pixels and accept a coarser `--output-pixel-size`; sigma remains expressed
in Å and concern calibration is unchanged. Native local CC is always calculated
on the experimental map grid. If a coarser output pixel size is requested, only
the local-CC concern, optional percentile, and envelope mask are resampled for
display. The raw local-CC field remains on its native grid. Manifests record the
requested and actual output pixel sizes and grid dimensions.

The manifest carries this machine-readable contract:

```json
{
  "primary_display": {
    "field": "concern",
    "domain": [0.0, 1.0],
    "normalization": "none",
    "representation": "direct-volume",
    "opacity": "absolute_concern",
    "hue": "absolute_concern",
    "color_anchors": {
      "0.0": "transparent",
      "0.5": "yellow",
      "0.75": "orange",
      "1.0": "red"
    }
  },
  "percentile_display": {
    "role": "optional_relative_contrast",
    "determines_visibility": false
  }
}
```

### Optional deterministic percentile mode

Percentile maps remain secondary outputs for analysis or an explicitly selected
relative-contrast mode. They are not required for normal coloring and must not
determine whether a voxel is visible.

Quantiles are calculated over a fixed eligible population:

```text
finite AND inside the metric support mask AND concern >= 0.05
```

For native local CC, the support mask is the model envelope. For deposited
metrics, concern of at least 0.05 defines effective support and excludes numerical
Gaussian tails and box padding. For the combined map it is the union implicit in
the combined concern field. Quantiles are calculated once for the complete field,
never from the current camera view, clipping plane, contour, or enabled subset.

Each eligible voxel is mapped to its midrank empirical percentile. If `L` values
are strictly smaller and `R` values are smaller or equal among `N` eligible
values, its displayed percentile is:

```text
u = (L + R) / (2 N)
```

Midrank makes ties deterministic: every voxel in a saturated plateau receives
the same percentile. Ineligible voxels receive percentile zero. The manifest
records the eligibility rule, eligible count, concern gate, and concern values at
the 50th, 80th, 95th, and 99th percentiles.

An optional relative-contrast palette may follow percentile rank:

```text
below 50th percentile  transparent
50th-80th percentile   blue/cyan
80th-95th percentile   yellow
95th-99th percentile   orange
99th-100th percentile  red
```

Even in this optional mode, visibility and opacity follow absolute concern, not
percentile. The legend and picking UI should show percentile, bounded concern,
and the metric's native value.

Example legend text:

```text
Top 1% of this field: concern >= 0.41; for local CC, CC <= 0.18
```

The viewer should expose metric toggles. For a combined hotspot, it should sample
the individual fields and name the metric or metrics responsible for the local
maximum. The combined map must be labeled “where to look,” never “model quality.”

## Current command

MolProbity-only fields:

```bash
source /root/phenix/build/setpaths.sh
cd /root/hotspots/hotspots
libtbx.python make_concern_maps.py MODEL.cif OUTPUT_DIRECTORY
```

Add previously calculated cctbx Q-score records:

```bash
libtbx.python make_concern_maps.py MODEL.cif OUTPUT_DIRECTORY \
  --qscore-json MODEL_qscore_000.json --resolution 3.1
```

For an explicitly calibrated non-protein case, replace `--resolution` with
`--expected-q VALUE`.

Generate native local map-model CC, its bounded concern view, an envelope mask,
and exact per-atom values:

```bash
libtbx.python make_local_cc_map.py MODEL.cif EXPERIMENTAL_MAP.mrc \
  RESOLUTION_A OUTPUT_DIRECTORY
```

## Known limitations requiring follow-up

1. Sigma 2 Å and spacing 1 Å are visualization defaults, not optimized values.
2. Probe omits sub-threshold contacts, causing a discontinuity in clash concern.
3. Repeated correlated clashes can saturate a larger clash region, although they
   can no longer overpower another metric numerically after the `[0,1]` cap.
4. The Q-score deficit width of 0.20 needs an example-based sensitivity check.
5. The combined CCP4 map does not encode the winning metric. The viewer must
   retain and sample the component maps, or a later format must carry provenance.
6. Local map-model CC depends on calculated-map choices such as resolution,
   scattering table, model B factors, and experimental-map preprocessing. These
   parameters belong in the manifest and maps with different preparation should
   not be compared as if only model fit differed.
