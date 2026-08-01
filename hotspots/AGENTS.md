# Operational instructions for agents

This directory generates molecular-validation hotspot maps. Use it as a
visualization tool; do not describe the combined field as a quality metric.

It now lives inside the pxviewer repository but is kept separable — see the note at the
top of [README.md](README.md) before adding any dependency on pxviewer.

## Environment

Always run cctbx-dependent commands under the cctbx python, from this directory:

```bash
cd hotspots
/Users/christopher/miniconda3/envs/pxviewer/bin/libtbx.python hotspots/make_concern_maps.py ...
```

Do not use the Miniforge/base Python for MolProbity extraction.

## Validation extraction is shared, and not ours to fork

Which residue a validation result belongs to and which atoms it implicates is decided by
`validation_events.py`, one file shared verbatim with pxviewer. This directory owns the
*calibration* on top of it (bounded `[0, 1]` concern, in `concern.py`) and nothing else.
pxviewer maps the same events to a different scale; neither is a rescaling of the other,
and they must never disagree about *where* a problem is.

Do not add a localization rule to `events.py`. Change the shared file, and check
`python/tests/test_validation_events.py` still passes.

Clash extraction uses **probe2**, which ships with cctbx, not
`mmtbx.validation.clashscore`, which shells out to the classic Duke `probe` binary that is
not installed here. Do not reintroduce the latter.

## Supported commands

MolProbity concern maps and optional percentile maps:

```bash
libtbx.python hotspots/make_concern_maps.py MODEL OUTPUT_DIR
```

Use `--output-pixel-size 2.0` (or another coarser value in Å) to reduce map size
and viewer cost without changing metric calibration.

With previously calculated cctbx Q-score JSON:

```bash
libtbx.python hotspots/make_concern_maps.py MODEL OUTPUT_DIR \
  --qscore-json QSCORE.json --resolution RESOLUTION_A
```

Native voxel local map–model CC:

```bash
libtbx.python hotspots/make_local_cc_map.py \
  MODEL EXPERIMENTAL_MAP RESOLUTION_A OUTPUT_DIR
```

For local CC, `--output-pixel-size` resamples only viewer concern/percentile/mask
maps. Raw local CC remains on the native calculation grid.

Do not use `hotspots/make_map.py`; it is the superseded additive-severity
prototype.

## Output interpretation

- `*_hotspot.ccp4` and `*_concern.ccp4`: authoritative bounded concern,
  used directly for both hue and opacity on a fixed `[0,1]` domain.
- `*_percentile.ccp4`: optional relative-contrast analysis mode only.
- `*_local_cc.ccp4`: raw scientific local Pearson correlation.
- `*_mask.ccp4`: valid/evaluated voxels.
- `*.json`: required provenance, parameters, and display quantiles.

The primary representation is direct-volume with no normalization: concern 0 is
transparent, 0.5 yellow, 0.75 orange, and 1.0 red. Percentiles must not determine
visibility or normal coloring. Never recompute quantiles from the viewport.
Retain component fields whenever producing a combined field.

Read [README.md](README.md) for complete commands and
[HOTSPOT_FIELDS.md](HOTSPOT_FIELDS.md) for calibration details.
