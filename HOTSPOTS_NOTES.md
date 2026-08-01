# Notes for the hotspots generator

Written from the pxviewer side, as the counterpart to `hotspots/VIEWER_NOTES.md`. Everything
below was measured on `1tec.pdb` with the maps this repo currently ships under
`debug/hotspots-sample/` (2.0 Å) and `debug/hotspots-sample-1A/` (1.0 Å).

Design background for the viewer's own scoring is in [HOTSPOTS.md](HOTSPOTS.md); the section
"The second generation: imported concern fields" is the part that concerns you.

## 1. The viewer bugs you reported are fixed

`VIEWER_NOTES.md` diagnosed the viewer as mixing two value systems. That was right, and the
mechanism was one line of asymmetry: computing a severity score cleared any imported concern
field, but importing a concern field did **not** clear a computed score. So a severity table
and a severity-coloured model stayed on screen beside a concern map that correctly read zero
in the same place.

What changed:

- **Concern is now its own quantity.** `python/pxviewer/concern.py` owns the imported product;
  `python/pxviewer/hotspots.py` owns computed severity. Importing and computing clear each
  other, so a model is in exactly one mode and the two scales can no longer be shown together.
- **The display contract is read from the manifest, not hardcoded.** `primary_display.color_anchors`
  drives the direct-volume ramp. Absolute concern drives both hue and opacity on a fixed
  `[0, 1]` domain — no percentile, min/max, sigma, or viewport-relative normalization anywhere.
- **Percentile no longer decides visibility.** The old contour path masked concern to
  `percentile >= 0.5`, contradicting your `determines_visibility: false`. That path is gone.
  Percentile maps are imported when present and are now **optional** — a manifest without them
  imports and draws normally.
- **The residue table is sampled from the concern maps themselves**, so its values are bounded
  `[0, 1]`, labelled `concern`, and cannot disagree with what is drawn.
- **Interface text** was rewritten; the threshold reads `Concern threshold` on a `[0, 1]` scale.

Your acceptance checks 2–8 pass. Check 1 does not, and we are deliberately not making it pass
— see §4.

## 2. What the viewer reads from your manifest

Consumed today:

| key | use |
| --- | --- |
| `outputs.<metric>.concern` | the authoritative field; required |
| `outputs.<metric>.color_percentile` | optional; imported, never used for visibility |
| `outputs.concern` / `outputs.color_percentile` | the flat local-CC shape, read as one `local_cc` field |
| `primary_display.color_anchors` | positions of yellow / orange / red |
| `molprobity.omitted_metrics`, `molprobity.reason` | surfaced in the status line |

`color_scaling.<metric>` is parsed and stored but **not** consumed — percentile is not used
for display, so its quantiles have nowhere to go. Not a request to remove it; just don't
assume it reaches the screen.

Rules the viewer enforces on import, each of which will reject a map with a clear message:

- concern must be bounded to `[0, 1]` (a map spanning `[0, 2.5]` is refused, not rescaled);
- values must be finite;
- if a percentile map is present it must be on the same grid, same origin, same `pixel_sizes`
  (1e-6) and same `shift_cart` (1e-5) as its concern map;
- `color_anchors` must ascend `0 <= yellow < orange < red <= 1`, or the documented defaults
  (0.5 / 0.75 / 1.0) are used instead.

Placement is honoured through **CCP4 NXSTART** (`map_data().origin()`) *and* `shift_cart`:
`origin = steps @ map_data().origin() - shift_cart()`. Considering only `shift_cart` put your
standalone maps at Cartesian zero, which is fixed. The 1TEC sample lands at `(-48, -36, -40)`.

The metric list is data-driven, so **if you start emitting a clash field the viewer will pick
it up with no change** — it appears in the field selector and gets its own table column.

## 3. Please adopt `python/pxviewer/validation_events.py`

Copy the file whole into `hotspots/`. It has no pxviewer imports and no relative imports, so
`from validation_events import ...` works as-is under Phenix python. It self-tests:

```bash
libtbx.python validation_events.py MODEL.pdb
```

We were each extracting ramalyze/rotalyze/clash results and deciding which atoms they
implicated, separately. The rules agreed by luck rather than by construction, and they had
already drifted in two ways: `hotspots/events.py` drops side-chain hydrogens from a rotamer
result where pxviewer keeps them, and it keys residues by `(chain, resseq, icode)` where
pxviewer used a formatted `resid` string — a difference that fails silently on join rather
than raising.

The module carries **native values, never calibrated ones**: a Ramachandran result travels as
its probability percentage, a clash as its overlap in Å, plus the validator's own `outlier`
boolean. Calibration stays on each side, which is where we legitimately differ — you map to
bounded concern, we map to unbounded surprisal severity. What must not differ is *which
residue a result belongs to, which atoms it implicates, and whether it was flagged*.

pxviewer's three severity functions are now thin calibrations over this module, so the
sharing is enforced rather than intended.

Two behaviours you would inherit, both of which matter to you specifically:

**Clash uses `probe2`, not `mmtbx.validation.clashscore`.** This is the fix for your Probe
blocker. `clashscore` shells out to the classic Duke `probe` binary, which is not installed
in the pxviewer conda environment — so `make_concern_maps.py` currently fails with
`RuntimeError: Probe could not be detected` whether or not `--heavy-atom-clashes` is passed,
and every sample we have omits clash. `mmtbx.probe2` ships with cctbx, is the same MolProbity
contact analysis, and is present. Switching `extract_clash` to it should give you a working
clash channel here.

**Clash events are collapsed to one per atom pair, keeping the worst overlap.** probe2 emits a
row per surface *dot*, so one contact arrives dozens of times — on 1TEC that is 28192 rows
versus 618 actual contacts. This matters more for you than for us: we take a max per atom, so
the duplication is harmless, but you *sum* within a metric before clipping
(`C_m = clip(Σ c·K, 0, 1)`). One kernel per dot would weight a contact by how thoroughly it
happened to be dotted and saturate the field. The atoms implicated are identical either way
(52 on 1TEC, heavy-atom pass), so this changes granularity, not localization.

Reference counts from the shared extractor on 1TEC:

| metric | events | outliers | atoms implicated |
| --- | ---: | ---: | ---: |
| rama | 338 | 5 | 20 |
| rota | 260 | 26 | 95 |
| clash | 618 | 47 | 52 |

## 4. The sanity check you asked for

`check_field_agreement` in the same module does what you described: sample the field back at
every atom and confirm the hot places are the places validation complained about.

```python
from validation_events import extract_all, check_field_agreement, worse_than_percent

events = extract_all(model)                      # or reuse results you already have
sampled = <your field, read at each atom, hierarchy order>
report = check_field_agreement(
    events, sampled, sites_cart,
    metric="rama", hot_threshold=0.5, tolerance_a=4.0,
    concerning=worse_than_percent(2.0))          # <- see the warning below
print(report.summary())
assert report.recall == 1.0
```

It asks two deliberately asymmetric questions.

**Recall** — does every atom the validator called an *outlier* reach the threshold? This
always keys off the outlier boolean, because never losing a real outlier is the one guarantee
such a field owes. A miss here is a genuine defect.

**Explained** — is every hot atom near something that could have put signal there? This takes
a distance `tolerance_a` rather than demanding identity.

**The one thing to get right: pass `concerning=worse_than_percent(2.0)`.** Judged against
outliers alone, the check reported atoms 12–21 Å from any Ramachandran outlier as
"unexplained" on your own 1TEC sample — far beyond a 2 Å splat, and it looks like a serious
bug until you see why. Your concern curve starts rising at the 2.0% good boundary, well before
the 0.05% outlier cut, so the field legitimately marks residues MolProbity never flags: on
1TEC only 5 of 338 Ramachandran results are outliers, but many more deposit real concern
(E84 at 0.2429% deposits ~0.57, E38 at 0.1519% deposits 0.699). Widening the predicate to your
own good boundary resolved every unexplained atom. Judging a continuous field against a
boolean cut reports correct behaviour as failure.

Results on the shipped samples, threshold 0.5, tolerance 4 Å:

| field | recall | explained |
| --- | --- | --- |
| rama 1.0 Å | 1.000 (20/20) | 1.000 (50/50) |
| rota 1.0 Å | 1.000 (95/95) | 1.000 (165/165) |
| rama 2.0 Å | 1.000 (20/20) | 1.000 (34/34) |
| rota 2.0 Å | 0.989 (94/95) | 1.000 (143/143) |

The single 2.0 Å rota miss is one atom sampling 0.492 against a 0.5 threshold — grid
smoothing, and it resolves at 1.0 Å.

**Keep a negative control.** A shuffled field scores recall 0.000 with 20 missed. A check that
only ever passes is not a check, and this one is cheap to fool yourself with.

## 5. Why acceptance check 1 cannot be met, and why that is fine

Check 1 asked that E85 not appear as a Rama hotspot at threshold 0.1, since its bounded Rama
concern is exactly 0.000.

It does appear, and the field is right. E85 itself is favored at 3.8369% and deposits exactly
0.000, as you said. But its neighbour **E84 scores 0.2429%** — not an outlier (the cut is
0.05%) yet well inside your concern curve, depositing about **0.572**. The σ ≈ 2 Å splat is
wider than the ~3.8 Å between adjacent Cα atoms, so E85's atoms genuinely sit in E84's
density. Sampled Rama concern at E85 is 0.348 at 2.0 Å and 0.432 at 1.0 Å — finer sampling
*raises* it, because it resolves more of the neighbour's peak.

Note that E85's signal comes from a residue MolProbity never flagged. That is the same reason
§4 insists on `worse_than_percent(2.0)`: a check keyed to the outlier boolean would call E85
unexplained and be looking for a bug that is not there.

The same pattern around the E53 outlier, sampled at 2.0 Å against what each residue deposits:

| residue | score % | outlier | deposits | sampled |
| --- | ---: | --- | ---: | ---: |
| E52 | 0.3234 | no | 0.494 | 0.872 |
| E53 | 0.0231 | **yes** | 1.000 | 0.908 |
| E54 | 1.0614 | no | 0.172 | 0.759 |

E54 reads four times what it deposits. This is the kernel doing exactly what it is for.

So the viewer's table ranks **neighbourhoods, not residues**, and says so rather than implying
attribution. We deliberately did not add a de-blurring heuristic to make the number look
cleaner; recovering per-residue concern from the grid after the splat is an ill-posed
deconvolution. The user reviewed this and accepted it ("visible is better than not visible
anyway").

**This is the one change that would genuinely improve things: emit per-residue concern in the
manifest.** You have those values before you splat them. With them the viewer could show true
per-residue attribution beside the field reading, and check 1 becomes meaningful instead of
impossible. Suggested shape, but any stable form works:

```json
"per_residue": {
  "rama": [{"chain": "E", "resseq": 38, "icode": "", "concern": 0.699, "native_percent": 0.152}]
}
```

## 6. Output pixel size costs more fidelity than it looks

`--output-pixel-size 2.0` is documented as the fast-viewing setting, and it is — but the
splat peak falls between voxels, so sampled values sit below the deposited concern:

| residue | deposited | sampled 2.0 Å | sampled 1.0 Å |
| --- | ---: | ---: | ---: |
| E38 | 0.699 | 0.483 | 0.663 |
| E185 | 0.273 | 0.201 | 0.260 |

Same calibration either way — only sampling density changes, which is exactly why `--sigma`
must not be adjusted to compensate. Cost is 6.6 MB versus 968 KB for 1TEC. Worth documenting
the tradeoff in `README.md` so 2.0 Å is not chosen for work where the numbers are read.

## 7. Asks, in priority order

1. **Per-residue concern in the manifest** (§5). Unblocks real attribution; nothing else here
   comes close in value.
2. **Adopt `validation_events.py`** (§3) and run `check_field_agreement` with
   `worse_than_percent(2.0)` plus a negative control as part of generation (§4).
3. **Switch clash extraction to probe2** (§3) so the channel works in cctbx-only environments.
   The viewer will display a clash field automatically once one exists.
4. **Document the output-pixel-size fidelity tradeoff** (§6).

## 8. Environment notes

- Run under this machine's cctbx python:
  `/Users/christopher/miniconda3/envs/pxviewer/bin/libtbx.python`. `AGENTS.md` still points at
  `/root/phenix/build/setpaths.sh` and `/root/hotspots`, which are from a different machine.
- Do not install Probe or other packages without asking; probe2 removes the need.
- The samples in `debug/` are gitignored, so regenerate rather than expecting them in a clone.
