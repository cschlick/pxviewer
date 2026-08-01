# Viewer integration notes

## Current bug: mixed severity and concern semantics

The Rama cloud in the supplied 1TEC screenshot is not missing data. The viewer
is mixing two incompatible value systems:

- the table and validation markers use the superseded, unbounded surprisal
  **severity** values;
- the volume cloud uses the current bounded `[0,1]` **concern** field.

As a result, the table can show a nonzero Rama number while the authoritative
Rama concern map is correctly zero at that residue.

## Concrete 1TEC evidence

| Residue | Value shown in viewer | Ramalyze score | Bounded Rama concern |
|---|---:|---:|---:|
| E270 ARG | 0.19 | 24.48% | 0.000 |
| E66 GLN | 0.27 | 13.15% | 0.000 |
| E85 ASN | 0.43 | 3.84% | 0.000 |
| E202 VAL | 0.41 | 4.45% | 0.000 |
| E38 ASP | 0.85 | 0.152% | 0.699 |
| E185 ASP | 0.65 | 0.731% | 0.273 |

The first four residues are in the Ramachandran favored region because their
scores exceed the 2% favored boundary. Their bounded Rama concern is therefore
exactly zero. They should not produce a Rama cloud even though the legacy
surprisal column contains values such as 0.19 or 0.43.

Many magenta clusters in the screenshot are rotamer or clash annotations. The
viewer continues to show annotations from every metric while `Field: Rama` is
selected, so unrelated markers appear to be Rama clusters lacking field color.

The generated 1TEC Rama CCP4 was checked independently:

- all five Rama outliers have local map peaks of approximately `0.88–1.0`;
- allowed Rama observations with concern above `0.1` have corresponding density;
- the field is bounded to `[0,1]`;
- CCP4 `NXSTART` placement is correct.

This is therefore a viewer semantics and filtering problem, not a Rama field
generation problem.

## Required viewer changes

### 1. Use concern consistently

The active metric's bounded concern must drive:

- volume hue;
- volume opacity;
- the threshold slider;
- the primary value shown in the results table;
- marker visibility when markers are threshold-filtered.

Do not compare a slider expressed in concern with the legacy surprisal severity.
Do not label a native score or legacy severity as concern.

If native validation values are useful, show them in separately labeled columns,
for example:

```text
Rama concern
Rama probability (%)
Rotamer concern
Rotamer probability (%)
Clash concern
Clash overlap (Å)
```

### 2. Filter or distinguish markers by selected field

When the selected field is Rama, the default marker display should show Rama
annotations only. The same rule applies to rotamer, clash, Q-score, and local CC.

If the product must show annotations from every metric simultaneously, encode
their source unmistakably and state that marker visibility is independent of the
selected cloud. Unrelated rotamer/clash markers must not look like missing Rama
clouds.

### 3. Restore the absolute display contract

The bounded concern map is the authoritative primary display field. Use it
directly for both hue and opacity in a direct-volume representation:

```text
concern 0.00  fully transparent
low concern  transparent/faint
concern 0.50  yellow
concern 0.75  orange
concern 1.00  red
```

Use one fixed `[0,1]` domain for every metric and structure. Apply no per-field
percentile normalization, min/max normalization, sigma scaling, or
viewport-relative statistics.

Percentile maps are optional relative-contrast analysis products. They must not
be required for normal coloring and must not determine whether a voxel is
visible.

### 4. Update stale interface text

The screenshot says that concern controls visibility while percentile controls
hue. That description is obsolete. Replace it with language equivalent to:

> Bounded concern controls both hue and opacity on a fixed 0–1 scale. The
> combined field indicates where to look, not model quality. Component fields
> identify which validation source raised concern.

The UI label `Severity threshold` should become `Concern threshold` when it is
controlling a bounded concern field.

## Manifest contract

Generated manifests now contain:

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

The viewer should follow this metadata rather than infer a display scale from
map statistics.

## Acceptance checks

The integration is corrected when all of the following hold:

1. Selecting Rama and setting concern threshold `0.1` does not show Rama markers
   for E270, E66, E85, or E202; their Rama concern is zero.
2. E38 and E185 show Rama field density and Rama markers at thresholds below
   their concerns (`0.699` and `0.273`, respectively).
3. All five 1TEC Rama outliers show strong Rama density near concern 1.
4. Selecting Rama does not leave visually identical rotamer/clash markers that
   can be mistaken for Rama annotations.
5. The table never shows values above 1 in a column labeled `concern`.
6. The threshold label reads `Concern threshold` and uses `[0,1]`.
7. Concern 0.5, 0.75, and 1.0 have the same yellow, orange, and red appearance
   across every structure and metric.
8. Changing the viewport or visible subset does not change volume colors.
