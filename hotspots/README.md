# Hotspots

Hotspots converts molecular validation results into spatial maps for display in
a molecular viewer. It is a visualization layer, not a new validation score.

> **This directory lives inside the pxviewer repository, and is meant to be separable
> again.** It was developed as its own repo and will likely be upstreamed as one, so
> nothing here imports from pxviewer. The single dependency in the other direction is the
> shared validation extractor, `validation_events.py`, which is deliberately kept as **one
> file** rather than copied — `hotspots/events.py` resolves it by explicit path, preferring
> a copy sitting beside itself if there is one. To split this directory back out: copy
> `python/pxviewer/validation_events.py` into `hotspots/`, and nothing else changes.
>
> Run it with the cctbx python, from this directory:
> ```bash
> libtbx.python hotspots/make_concern_maps.py MODEL OUTPUT_DIR --output-pixel-size 2.0
> ```

The supported workflows produce:

- separate clash, Ramachandran, and rotamer hotspot maps;
- an optional Q-score hotspot map from cctbx per-atom Q-score output;
- a combined “where to look” map formed by the voxel-wise maximum of the
  available concern fields;
- a native voxel-level local map–model correlation field;
- optional percentile maps for relative-contrast analysis;
- JSON manifests carrying the authoritative display contract.

The scientific calibration and field definitions are documented in
[HOTSPOT_FIELDS.md](HOTSPOT_FIELDS.md).

## Requirements

Run the programs with a configured Phenix/cctbx Python environment. The tested
environment is:

```bash
source /root/phenix/build/setpaths.sh
```

Use `libtbx.python`, not the system or Miniforge Python. The latter may import
parts of cctbx but lack the MolProbity reference data required by `rotalyze`.

All commands below are run from the repository root:

```bash
cd /root/hotspots
```

Input models may be PDB or mmCIF, optionally gzip-compressed. Experimental maps
may be MRC, CCP4, or gzip-compressed map files accepted by cctbx.

## Quick start: MolProbity hotspot maps

Generate clash, Ramachandran, rotamer, combined, and optional percentile maps:

```bash
libtbx.python hotspots/make_concern_maps.py \
  /path/to/model.cif \
  /path/to/output_directory
```

Hydrogen-added MolProbity clashes are used by default. This is the calibrated
mode. For a labeled heavy-atom preview:

```bash
libtbx.python hotspots/make_concern_maps.py \
  /path/to/model.cif \
  /path/to/output_directory \
  --heavy-atom-clashes
```

Optional spatial parameters are:

```text
--sigma 2.0               Gaussian width in Å
--output-pixel-size 1.0    output voxel/pixel size in Å
```

For faster viewer rendering and smaller files, choose a coarser output pixel
size, for example:

```bash
libtbx.python hotspots/make_concern_maps.py \
  /path/to/model.cif /path/to/output_directory \
  --output-pixel-size 2.0
```

This setting changes sampling density, not the Gaussian width or `[0,1]`
calibration. `--spacing` remains accepted as a compatibility alias. Do not change
`--sigma` merely to alter visual contrast.

For a model named `model.cif`, the principal outputs are:

```text
model_clash_hotspot.ccp4
model_rama_hotspot.ccp4
model_rota_hotspot.ccp4
model_combined_hotspot.ccp4

model_clash_hotspot_percentile.ccp4
model_rama_hotspot_percentile.ccp4
model_rota_hotspot_percentile.ccp4
model_combined_hotspot_percentile.ccp4

model_hotspots.json
```

The non-`percentile` maps contain authoritative bounded concern in `[0,1]` and
drive both hue and opacity. Percentile maps contain deterministic empirical rank
for an optional relative-contrast mode. The JSON manifest records calibration,
event counts, map paths, display contract, and optional display quantiles.

## Add Q-score

Q-score is calculated by cctbx. Hotspots consumes its per-atom JSON records and
projects them into space; it does not reimplement Q-score.

First calculate Q-score from the same model and experimental map:

```bash
mkdir -p /path/to/qscore_output

mmtbx.development.qscore \
  /path/to/model.cif \
  /path/to/experimental_map.mrc \
  qscore.nproc=4 \
  --json-filename /path/to/qscore_output/model_qscore.json
```

Then supply those records and the reported map resolution:

```bash
libtbx.python hotspots/make_concern_maps.py \
  /path/to/model.cif \
  /path/to/output_directory \
  --qscore-json /path/to/qscore_output/model_qscore.json \
  --resolution 3.1
```

This adds:

```text
model_qscore_hotspot.ccp4
model_qscore_hotspot_percentile.ccp4
```

and includes Q-score in the maximum-combined map.

The Q-score JSON and hotspot model must describe the same atoms in the same
Cartesian frame. For the default protein calibration, expected Q is calculated
from resolution. For a separately calibrated non-protein or mixed case, provide
an explicit expectation instead:

```bash
--expected-q 0.55
```

Do not provide both merely as alternatives; `--expected-q` takes precedence.

## Native local map–model correlation

Local map–model correlation is generated directly at map voxels. It is not made
by smoothing atom values.

```bash
libtbx.python hotspots/make_local_cc_map.py \
  /path/to/model.cif \
  /path/to/experimental_map.mrc \
  3.1 \
  /path/to/output_directory
```

The positional `3.1` is the reported resolution in Å. The default correlation
radius is:

```text
max(2.5 Å, reported resolution)
```

The command boxes the map and model together, calculates an electron-scattering
model map on the same grid, and evaluates sphere-local Pearson correlation only
within the model envelope.

Outputs are:

```text
model_local_cc.ccp4               raw Pearson CC; NaN outside envelope
model_local_cc_concern.ccp4       bounded (1 - CC) / 2
model_local_cc_percentile.ccp4    optional relative-contrast percentile
model_local_cc_mask.ccp4          binary evaluated-voxel mask
model_local_cc_atoms.json         exact built-in CC at each atom
model_local_cc_residues.json      mean finite atom CC by residue
model_local_cc.json               parameters, statistics, and paths
```

Optional parameters:

```text
--radius R              override the correlation radius in Å
--envelope-radius R     change only the evaluated/displayed model band
--box-cushion R         override boxing cushion in Å
--scattering-table NAME default: electron
--output-pixel-size R   coarsen viewer maps to approximately R Å pixels
```

`--envelope-radius` does not change an individual voxel's CC. It only changes
which voxels are retained. Avoid setting `--radius` below one resolution element;
the resulting field has unjustifiably high variance.

`--output-pixel-size` is an export/performance control. Local CC is still
calculated on the experimental map's native grid. The concern, percentile, and
mask maps are resampled onto the requested coarser grid; the raw local-CC map
remains native and uninterpolated. For example:

```bash
libtbx.python hotspots/make_local_cc_map.py \
  MODEL.cif MAP.mrc 3.1 OUTPUT_DIR --output-pixel-size 2.0
```

The requested and actual pixel sizes and both grid dimensions are recorded in
the manifest. Finer-than-native output is rejected because it increases file
size without adding information.

The convolution implementation was checked against
`mmtbx.maps.correlation.from_map_map_atom` at 100 grid centers on 29KM. Maximum
absolute disagreement was `1.1e-14`.

## Authoritative viewer display

The primary visualization uses the bounded concern map directly for both hue
and opacity. Use a direct-volume representation with one fixed `[0,1]` domain
for every metric and every structure:

```text
concern 0.00  fully transparent
low concern  transparent/faint
concern 0.50  yellow
concern 0.75  orange
concern 1.00  red
```

Apply no min/max normalization, percentile normalization, sigma scaling, or
viewport-relative statistics. The same concern value must have the same color
and opacity in every map.

Percentile maps are optional secondary outputs for relative-contrast analysis.
They must not be required for normal coloring and must never determine whether a
voxel is visible. If an optional percentile mode is exposed, quantiles are fixed
at map-generation time and never recalculated from the viewport or clipping
plane. A viewer tooltip should still report native metric value, bounded concern,
and optionally within-field percentile.

## Coordinate placement

Generated CCP4 maps carry Cartesian placement using `NXSTART` /
`origin_shift_grid_units`. The CCP4 `ORIGIN` header is not used as the sole
placement mechanism. Load the output map and its source model without applying
an additional manual translation.

## Programmatic components

The reusable modules are:

```text
hotspots/events.py       MolProbity result extraction
hotspots/concern.py      metric calibration and maximum combination
hotspots/field.py        deposition, Gaussian convolution, CCP4 writing
hotspots/local_cc.py     native voxel and per-atom local CC
hotspots/color_scale.py  absolute display contract and optional percentiles
```

The supported command-line entry points are:

```text
hotspots/make_concern_maps.py
hotspots/make_local_cc_map.py
```

`hotspots/make_map.py` is the superseded additive-severity prototype. Do not use
it to generate current viewer maps. The analysis scripts remain for historical
ablation and consistency work; they are not required for normal map generation.

## Smoke tests

Check imports and syntax:

```bash
source /root/phenix/build/setpaths.sh
libtbx.python -m py_compile hotspots/*.py
```

Check CLI availability:

```bash
libtbx.python hotspots/make_concern_maps.py --help
libtbx.python hotspots/make_local_cc_map.py --help
```

A successful generation must satisfy all of the following:

- every path listed in the manifest exists;
- bounded concern and optional percentile maps contain only `[0,1]` values;
- local-CC values inside the mask are in `[-1,1]`;
- model and maps overlay without a manual origin correction;
- manifests record the actual resolution, radii, spacing, and calibration.
- manifests record native and output pixel sizes when resampling is requested.

## Important limitations

- ~~Probe exposes only reportable clashes, so the clash layer has no
  sub-threshold tail.~~ Fixed: extraction moved to `probe2`, which reports every
  contact with a negative gap. On 1TEC with hydrogens, 1380 of 1476 contacts fall
  below the 0.40 Å reporting boundary and now carry concern.
- Q-score concern is resolution-dependent; the `0.20` saturation deficit is a
  declared visualization choice rather than a published outlier threshold.
- Local CC depends on map preprocessing, model-map resolution, scattering table,
  B factors, and boxing choices.
- The combined map records maximum concern but does not encode which metric won.
  Keep the component maps available for attribution.
- Percentile output is structure-relative and optional. It must not control the
  primary display, visibility, or cross-structure comparison.
