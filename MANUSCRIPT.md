# pxviewer: A cctbx-native, streaming molecular graphics application for interactive model building and validation, with an aggregated per-atom validation "hotspot" metric

**Authors:** C. Schlick¹ *(and colleagues)*
**Affiliation:** ¹ *[affiliation to be added]*
**Correspondence:** *[email to be added]*

---

## Abstract

Macromolecular model building and validation are dominated by a small number of desktop
applications whose data models are their own: a structure is parsed into the program's internal
representation, and the crystallographic or cryo-EM computations that inform rebuilding —
restraint geometry, real-space and reciprocal-space refinement, MolProbity-style validation —
are performed either in a separate process or against a second copy of the model. We present
**pxviewer**, a molecular graphics application built on a different premise: the *Computational
Crystallography Toolbox* (cctbx) is the single source of truth for every atom, and the graphical
front end never parses a structure. Models are read in Python with cctbx into an
`mmtbx.model.manager`, and everything the viewer displays is derived from that object's hierarchy
and **streamed** to a Mol\*-based WebGL front end over a lightweight binary bridge. A fixed
topology is sent once; thereafter only coordinates flow, so interactive operations — tugging an
atom, running a geometry minimization, following a live difference map — update the view in place
rather than reparsing. The application is delivered as a single desktop program (a Qt shell
hosting the web front end) but the same Python session drives a browser directly, a Jupyter
console, or a headless renderer.

Beyond re-architecting a familiar workflow, we introduce a novel validation aggregate we call the
**validation hotspot score**: a single, calibrated, per-atom severity field that fuses
Ramachandran, rotamer, all-atom clash, and map-fit quality into one number whose value is
*absolute* (1.0 is exactly the community outlier threshold for each contributing metric, whatever
its native units) and whose provenance is always visible. The score is rendered both as a direct
per-atom coloring and as a value-colored 3-D volumetric "cloud", and it is computed on a shared
analysis cache so that requesting it also populates the conventional validation tables, and vice
versa. We describe the score's calibration, its topological atom-assignment rules, its
combination operator, and the rendering and performance engineering required to make a
raymarched severity cloud usable on a laptop.

---

## 1. Introduction

Interactive model building — the iterative correction of an atomic model against experimental
density — is a mature activity supported by well-known tools such as Coot (Emsley et al., 2010)
and, increasingly, ChimeraX (Pettersen et al., 2021) with ISOLDE (Croll, 2018). Validation, the
assessment of a model's stereochemical and experimental plausibility, is likewise supported by
established software: MolProbity (Williams et al., 2018) for all-atom contacts, Ramachandran and
rotamer analysis, and the Phenix (Liebschner et al., 2019) and cctbx (Grosse-Kunstleve et al.,
2002) ecosystems for the underlying computation. General-purpose visualization is served by
PyMOL, ChimeraX, and the web-native Mol\* toolkit (Sehnal et al., 2021).

These tools share an architectural assumption that is rarely stated: the visualizer owns the
model. A structure is parsed from a file into the program's internal atom representation, and
computations that require crystallographic machinery (restraints, refinement, validation
statistics) are either delegated to an external program that re-reads the same file, or performed
against a second, separately maintained copy of the coordinates. This is workable, but it has
costs. The two representations can drift; the validation numbers a user sees are computed by
different code than the geometry that drives rebuilding; and interactivity across the boundary
(refine a little, look, refine again) requires marshalling coordinates back and forth.

pxviewer takes the opposite position. Its design thesis is that **the computational toolkit
should be the single source of truth, and the visualizer should be a live view onto it.** Concretely,
every atom the user sees lives in one `mmtbx.model.manager`; restraints, minimization, phasing,
and validation all operate on that one object; and the graphical front end is a thin renderer fed
a stream of coordinates. Nothing is parsed in the browser.

This paper describes the application, the rationale for its architecture, and a novel validation
aggregate — the *hotspot score* — that the architecture made natural to build.

> **Figure 1 (screenshot).** *The pxviewer desktop application, whole-window. A cartoon model
> and an electron-density surface are shown in the central Mol\* viewport; the right-hand control
> panel shows the tabbed interface (Scene, Tools, Validation, Hotspots, Geometry, Console,
> Settings) with the Scene tab active, listing the loaded objects. Capture at the default window
> size with one map+model pair loaded (the bundled 1UBQ map+model demo) so the object grouping is
> visible.*

---

## 2. Background and related work

**Visualization toolkits.** Mol\* (Sehnal et al., 2021) is a WebGL molecular graphics library
designed for large structures in the browser; it provides a state-tree data model, a
representation system, and GPU renderers including isosurface and direct-volume (raymarched)
volume rendering. pxviewer embeds Mol\* as its rendering engine but does not use its parsers:
where Mol\* would ordinarily load a mmCIF file, pxviewer feeds it a topology it generated from a
cctbx hierarchy and streams coordinates thereafter.

**Model building and refinement.** Coot pioneered the tight loop of manual rebuilding against
density with live validation feedback. ISOLDE embedded interactive molecular dynamics into
ChimeraX. Both keep their own model and call out to, or reimplement, the crystallographic
computations. pxviewer's interactive refinement (Section 5.4) instead drives cctbx restraints and
minimization directly on the displayed model.

**Validation.** MolProbity established the reference all-atom contact analysis (via *reduce* for
hydrogen placement and *probe* for contact dots; Word et al., 1999) together with Ramachandran
(Lovell et al., 2003) and rotamer analysis, C-beta deviation, CaBLAM, and the composite
*MolProbity score*, a resolution-calibrated log-weighted aggregate of clashscore, rotamer, and
Ramachandran outlier rates. The wwPDB validation reports present per-residue outlier bands and a
per-residue real-space fit Z-score (RSRZ). Q-score (Pintilie et al., 2020) provides a per-atom
map-model fit measure for cryo-EM. pxviewer computes all of these through cctbx/mmtbx, and its
hotspot score (Section 6) can be read as a *per-residue, per-atom localization* of the composite
idea that MolProbity score expresses globally.

---

## 3. Architecture

pxviewer is organized into three cooperating layers (Figure 2): a Python computational backend
built on cctbx, a live binary bridge, and a Mol\*-based WebGL front end, all wrapped by a desktop
shell.

> **Figure 2 (diagram, not a screenshot).** *A three-layer architecture diagram. Left: the Python
> process — cctbx `DataManager` → `mmtbx.model.manager` → a `LiveSession` that owns the
> topology and per-atom data. Middle: the binary WebSocket bridge, labelled with the seven message
> tags (TOPOLOGY, FRAME, ATTRIBUTE, DOTS, MAP, FRAME_DELTA, HOTSPOT_VOLUME). Right: the browser —
> a Mol\* plugin whose state tree holds one streamed trajectory plus streamed volumes. A dashed box
> around all three labelled "QtWebEngine desktop shell (PySide6)". Duplex arrows on the bridge
> indicate that picks and query results flow back to Python.*

### 3.1 The cctbx backend

Model I/O goes exclusively through cctbx's `DataManager`, producing an `mmtbx.model.manager`
whose `pdb_hierarchy` is the authoritative atom list. A vectorized adapter
(`cctbx_io.model_to_arrays`) exposes the hierarchy's columns — coordinates, element, atom and
residue names, chain identifiers, residue numbers, B-factors, occupancies, altlocs — in the atom
order that the rest of the system uses. All heavier computation is likewise cctbx/mmtbx: geometry
restraints and their user edits (`edits.py`, `geometry.py`), geometry and real-space minimization
with live intermediate states (`minimize.py`), reflection handling and map calculation
(`reflections.py`, `volume_io.py`), ligand construction from the monomer library or from SMILES
via RDKit (`ligands.py`), hydrogen placement via *reduce2* (`hydrogens.py`), all-atom contacts via
*probe2* (`probe.py`), and the MolProbity validators (`validation/`).

### 3.2 The live coordinate bridge

The bridge (`live.py`) is a `LiveSession`: a small WebSocket server that speaks a compact binary
protocol. Its central design decision is that **topology is sent once and coordinates stream**.
The message tags are:

| Tag | Name | Payload |
| --- | --- | --- |
| 0 | TOPOLOGY | a BinaryCIF `_atom_site` block plus bonds, sent on connect |
| 1 | FRAME | a full coordinate array (all atoms) |
| 2 | ATTRIBUTE | a named per-atom scalar array, for color-by-value |
| 3 | DOTS | a probe2 contact-dot surface (positions, spikes, colors) |
| 4 | MAP | a small live density box (affine + f32 grid) |
| 5 | FRAME_DELTA | only the atoms that moved: indices + coordinates |
| 6 | HOTSPOT_VOLUME | a validation-severity grid, drawn as a value-colored cloud |

Because atom identity is positional (the *i*-th coordinate is always the *i*-th atom of the
topology), a frame is a bare float array and requires no matching logic in the browser. When a
drag or minimization moves only a handful of atoms, a FRAME_DELTA carries just those indices,
falling back to a full FRAME when the moved set is large enough that the delta would not save
bandwidth. Attribute and volume messages are similarly self-contained: a per-atom coloring is a
named float array the front end maps through a color scale, and a live map is an affine plus a
raw grid dropped onto a Mol\* volume with no crystallography performed in the browser. Late-joining
viewers are brought up to date by replaying the current topology, attributes, representations,
overlays, and any live map or severity cloud.

### 3.3 The Mol\* front end

The front end (`frontend/src/live.ts`, `index.tsx`; TypeScript, Mol\* 5.10.1, bundled with esbuild)
builds the Mol\* state tree once from the streamed topology and thereafter swaps coordinates in
place, so a per-frame update is a coordinate refresh of existing representations rather than a
reparse and rebuild (the README refers to this as "Level 1" in-place updating). Custom Mol\* state
transforms wrap the streamed data: a live trajectory whose conformation is replaced per frame, a
live difference-map volume, and a hotspot-severity volume. A custom per-atom attribute color theme
maps streamed scalar arrays onto atoms through a color scale with an explicit, fixed domain. The
front end also renders MolProbity-style markup (contact dots, outlier markers) received over the
DOTS channel, and forwards user picks back to Python.

### 3.4 The desktop shell

The everyday delivery is a single desktop application: a PySide6 (Qt) window whose central widget
is a `QWebEngineView` hosting the Mol\* page, with a docked control panel of tabbed tools. Because
the front end is an ordinary web page served locally, the *same* Python `LiveSession` can instead
be pointed at a browser tab, embedded in a Jupyter notebook, or driven headlessly to render images
— the desktop shell is one client among several. The controls are organized into tabs (Scene,
Tools, Validation, Hotspots, Geometry, Console, Settings); an embedded IPython console
(`console.py`) shares the live objects, so the GUI and scripting operate on the same session.

---

## 4. Design rationale

**One source of truth.** Keeping every atom in a single cctbx model eliminates the drift between
"the model you see" and "the model the science runs on." Validation numbers, restraint geometry,
and refinement all read and write the same coordinates the renderer displays.

**Stream, do not parse.** Sending a fixed topology once and coordinates thereafter makes
interactivity cheap: the expensive step (building the topology, bonds, and representations) happens
at load, and each subsequent frame is a coordinate update. This is what allows drag-to-refine and
live minimization to feel continuous rather than stuttering through reparses.

**Positional identity.** Fixing that the *i*-th streamed coordinate is the *i*-th topology atom
removes an entire class of matching logic from the hot path and keeps frames to a bare float
buffer. It also underpins the attribute-coloring and severity channels, which are likewise
index-aligned arrays.

**Compute once, view many ways.** Because the front end is a client of a session rather than an
application in its own right, the same computation serves the desktop, a browser, a notebook, and a
headless renderer. The figures in this paper, for example, are produced by the headless path.

**Share expensive analysis.** Validation and the hotspot score both depend on the same costly
mmtbx analyzers. Rather than each feature paying separately, pxviewer memoizes those analyzers in a
per-model analysis cache that is dropped whenever the coordinates change (Section 6.6), so
requesting either result makes the other nearly free — a design that is only coherent because
there is one model to key the cache on.

---

## 5. Core capabilities

### 5.1 Model input and representation

Models are read by cctbx and shown as any of the standard Mol\* representations (cartoon,
ball-and-stick, spacefill, and so on). Because the topology carries residue and chain labels from
the hierarchy, secondary-structure cartoon and per-chain coloring work without a browser-side
parse. Multiple objects are grouped in the object panel; a model and the reflections or map that
phased it are shown together as a group.

> **Figure 3 (screenshot).** *The Scene tab object panel with several loaded objects: a map+model
> group and one or more standalone models, showing the grouping (indented group members under a
> bold group header) and the color swatches. Capture after loading the map+model demo plus one
> additional standalone model, so the root-indented standalone is visible alongside the group.*

### 5.2 Coloring, including the per-atom attribute channel

In addition to Mol\*'s built-in themes (element, chain, secondary structure, residue type,
hydrophobicity, B-factor, occupancy), pxviewer adds *computed* per-atom colorings that are not
themes at all but streamed scalar arrays: **Q-score** (per-atom map-model fit; Pintilie et al.,
2020) and the **hotspot severity** field (Section 6). These travel over the ATTRIBUTE channel and
are mapped through a color scale on a *fixed* domain, so that a color denotes the same value in
every structure rather than being stretched to the current one.

### 5.3 Volumes

Electron-density and cryo-EM maps are shown as isosurfaces (surface or mesh) with a sigma-scaled
contour level, or as a value-colored slice. Reflections are read by cctbx and maps computed on the
Python side; difference maps are drawn with a negative contour in a second color. Maps and models
compose in one scene.

### 5.4 Interactive refinement

pxviewer supports a Coot-like interactive loop driven entirely on the cctbx model. A user may
**tug** an atom (or a defined scope — a single residue, a stretch of residues, or a selection) and
the model relaxes toward the pointer under geometry restraints, streaming intermediate
conformations as FRAME_DELTA messages. A geometry or real-space **minimization** streams its
intermediate states the same way. When a model has been phased, an X-ray **difference map** is
recomputed around the region under manipulation and streamed as a live MAP box, so the user sees
density respond to rebuilding in near real time.

> **Figure 4 (screenshot, ideally a two- or three-panel sequence).** *Interactive tug refinement
> with a live difference map. Panel (a): a sidechain placed slightly out of density, with green/red
> difference density indicating the misfit. Panel (b): mid-drag, the sidechain being pulled toward
> the pointer. Panel (c): after release, the difference density reduced. Use the bundled X-ray
> minimization tutorial structure so the difference map is meaningful.*

### 5.5 Restraint edits and ligands

Custom geometry-restraints edits (bond, angle, dihedral) can be read from and written to Phenix
`geometry_restraints.edits` files and applied live, redrawing the affected restraint notation in
the viewport. Ligands can be built from the monomer library or from a SMILES string (RDKit embeds a
conformer that supplies both coordinates and on-the-fly restraints), placed at a marker, and
refined into density.

### 5.6 Validation

Every registered MolProbity per-residue validator — Ramachandran, rotamers, CaBLAM, C-beta
deviation, omega (peptide planarity), and Ramachandran-Z — runs through mmtbx and produces a table
of residues plus 3-D markup drawn in the viewport. All-atom contacts (clashes and contacts) are a
separate, heavier analysis: hydrogens are added with *reduce2* and *probe2* is run on the
hydrogenated model, producing the MolProbity contact-dot surface as two independently toggleable
overlays. A single control shows or hides all validation markup at once.

> **Figure 5 (screenshot).** *The Validation tab after a run: the sub-tab strip (Clashes &
> contacts, Ramachandran, Rotamers, CaBLAM, C-beta, Omega, Rama-Z), one validator's residue table
> visible with a selected outlier row, and the corresponding MolProbity markup (e.g. green
> Ramachandran C-alpha vectors or gold rotamer sidechain markup) drawn on the model in the
> viewport. Show the "Hide all markers" button.*

---

## 6. The validation hotspot score (novel contribution)

The central novel contribution of pxviewer is an aggregated validation metric designed for
*navigation*: a single, calibrated, per-atom severity field that tells a modeller where to look
next, computed and rendered so that the aggregate is only ever allowed to rank while its
constituents remain individually visible.

### 6.1 Motivation and the failure of the naive aggregate

A modeller correcting a structure consults several validation reports — geometry, Ramachandran,
rotamers, clashes, map fit — and must mentally fuse them to decide where to work. The obvious
automation, averaging the metrics into one score, fails for three reasons that shaped the design.

*First, averaging is the wrong operator for outlier detection.* The metrics are unequal in
severity; one serious clash averaged against five clean metrics becomes a mild smudge. A mean is a
low-pass filter and this is an outlier detector. The combination must **preserve** severity.

*Second, restraint deviations partly report the refinement weights, not the model.* Refinement
actively minimizes geometry deviation while trading it against map fit, so a score mixing "deviation
from ideal geometry" with "fit to density" is partly measuring how the weight was set. Ramachandran,
rotamers, and clashes are cleaner precisely because they are conventionally left unrestrained —
which is why MolProbity leans on them, and why bond/angle deviations are excluded from our default
score.

*Third, "worst" is not the useful question.* Coloring raw badness lights up flexible surface loops
and high-B regions in every structure — true but useless. The information is in badness *beyond
what is expected there*.

### 6.2 Calibration: the community threshold as the unit

Every contributing metric is natively continuous, and its familiar boolean "outlier" flag is a
threshold applied downstream: `ramalyze` reports a Ramachandran probability percentage whose 0.05%
contour is the outlier cut; `rotalyze` similarly at 0.3%; probe reports a signed overlap in
angstroms whose 0.4 Å magnitude is a "serious clash." pxviewer defines each metric's **severity**
so that it equals exactly **1.0 at that community threshold**, 0 at unremarkable, and >1 beyond.
No weights are invented: the relative importance of the metrics is inherited from the reference
distributions that set those thresholds, and a severity of 1.0 means the same thing — "at the
outlier cut" — for every metric.

For the geometry metrics this is a **surprisal**: a conformation's reference-density percentage
*p* becomes −log₁₀(*p*), divided by the surprisal at the cut, so severities are commensurable
"decades of surprise under the null hypothesis that the structure is correct." Because the
reference distributions differ, a Ramachandran outlier (≈3.3 decades) is intrinsically more severe
than a rotamer outlier (≈2.5) — an asymmetry that is correct and free. The map-fit term is anchored
by convention rather than a calibrated tail (there is no community threshold for a per-atom Q-score);
this is the weakest link in the calibration and is documented as such.

Severity is displayed on a **fixed** domain (0 to 2, where 1.0 is the cut) rather than stretched to
the structure's own range, so the same color denotes the same quality in every structure — the same
absolute-scale principle applied to Q-score coloring.

### 6.3 Topological atom assignment

Because the metrics have different native loci, and several are sub-residue, the score is defined
per *atom* with an explicit, topological assignment of each metric to the atoms whose coordinates
produced it:

| Metric | Atoms it implicates |
| --- | --- |
| Q-score / map fit | the atom itself |
| Clash | both atoms of the clashing pair, full severity each |
| Rotamer | sidechain atoms only (not the backbone) |
| Ramachandran | backbone N, CA, C, O of the scored residue |
| C-beta deviation | CB alone |
| Omega | the peptide-bond atoms of the two residues |
| Bond / angle | the restraint's atoms |

Computing per atom, rather than per residue, is what lets the picture say *what to fix*: a rotamer
outlier glows on the sidechain while its own backbone stays clean, because the outlier is a
statement about χ angles and says nothing about the backbone. Ramachandran is assigned narrowly to
its own residue's backbone rather than smeared onto the neighbors whose atoms also enter φ/ψ.

### 6.4 Combination: a p-norm

Per-atom severities are combined with a p-norm,

  *S(a)* = ( Σₘ *sₘ(a)*ᵖ )^(1/p),  with *p* ≈ 4,

taking the worst instance within each metric first, then across metrics. As *p* → ∞ this is a max
(pure severity, no credit for corroboration); at *p* = 1 it is a sum (which would triple-count one
physical error seen by three metrics). At *p* ≈ 4 it behaves like a max but gives a modest bonus
when several metrics fire on the same atom, so a genuinely misbuilt residue that trips Ramachandran,
rotamer, *and* clash ranks above one with a single marginal problem — without the sum's
triple-counting. Crucially, the p-norm has **no denominator** and 0 is its identity, so a metric that
does not apply to an atom (a carbonyl oxygen has no rotamer term) contributes nothing and atoms with
different numbers of applicable metrics remain directly comparable.

That last property has a valuable consequence: the score **works without a map**. The map-fit term
is simply dropped, and because 0 is the identity, an atom whose fit was clean scores identically with
or without one. A geometry-only score sits on the *same absolute scale* as a map-inclusive one — it
is missing a term, not rescaled.

### 6.5 Rendering: per-atom coloring, a contour, and a value-colored cloud

The score reaches the viewport through the same per-atom attribute channel used by Q-score. Because
a cartoon draws no sidechains, the per-atom field is broadcast to the residue's worst value for
representations that do not draw atoms, so the rotamer component (which lives on sidechains) remains
visible on a ribbon.

The model coloring runs the **background color at its clean end**, so unremarkable protein fades into
the viewport and only the hotspots read; the background is queried from the renderer rather than
assumed. Two 3-D renderings are offered: a translucent **contour** shell at the calibrated cut
(cheap, unambiguous as a surface), and a value-colored **cloud** — a Mol\* direct-volume raymarch in
which each voxel is colored by severity and made transparent where clean, the look of a
local-resolution map. The severity field is built by a distance-weighted p-norm of the per-atom
values (not a sum, which would light up the dense core merely for containing more atoms), on a grid
whose kernel width is the "action scale" of a residue plus its environment.

> **Figure 6 (screenshot, multi-panel).** *The hotspot score on one structure. Panel (a): per-atom
> coloring on a cartoon — most of the model near-white (fading into the background), with sidechains
> and loops picked out in yellow → orange → red. Panel (b): the 3-D severity cloud (direct-volume)
> over the same model, a translucent value-colored haze with red cores at the worst residues, the
> model visible through it. Panel (c): the contour rendering. Use a structure with genuine outliers
> (e.g. the bundled 1TEC) so the pattern is rich. Capture panels (a) and (b) from the same
> viewpoint.*

> **Figure 7 (screenshot).** *The Hotspots tab control panel: the map-fit-term selector, the "Use
> hydrogens for clashes (accurate, much slower)" checkbox, the Find hotspots button, the 3-D
> show/style (Cloud/Contour) and quality (Low/Medium/High) selectors, the cloud opacity-knee slider,
> the summary line, and the ranked residue table whose columns break the aggregate back into its
> Ramachandran / Rotamer / Clash / Map-fit components. This figure documents that the aggregate is
> always shown alongside its parts.*

The severity cloud required specific rendering engineering. Because a raymarch composites opacity
front-to-back, a coarse step across a steep opacity ramp paints visible concentric "shell" artifacts
rather than a diffuse haze; the smoothness therefore comes from a small *step size* and low per-step
opacity, while performance comes from a coarser grid (less texture) and empty-space skipping — not
from taking larger steps. Because the sweet spot is hardware-dependent, cloud quality is exposed as a
Low/Medium/High control that trades grid resolution and step count against frame rate, defaulting to
the interactive setting.

### 6.6 Shared analysis, hydrogens, and populate-both

The hotspot score and the validation tables depend on overlapping, expensive computation. pxviewer
memoizes the shared mmtbx analyzers — Ramachandran, rotamers, and probe (with and without added
hydrogens) — in a per-model **analysis cache** that is dropped whenever the atoms move, so a stale
geometry can never be shown. Requesting either result populates both tabs from the one shared
analysis, so whichever the user asks for, the other is already available.

Clash detection deserves special note. MolProbity clashes are overwhelmingly hydrogen-mediated:
scoring the bare model finds a small fraction of the clashes found after adding hydrogens with
*reduce2* (on a benchmark structure, tens versus hundreds of flagged atoms). Adding hydrogens and
probing the larger model is by far the most expensive step in the stack, however, so the hotspot
score offers hydrogen-based clashes as an **opt-in** (the accurate MolProbity path) while defaulting
to a fast heavy-atom-only pass; the per-residue validators, which do not use hydrogens, are
unaffected, and the Clashes tab, which requires them, always uses them and shares that run. When
hydrogens are used, clash severity is accumulated on the hydrogenated model and mapped back onto the
scored model's atom order, with each hydrogen's severity handed up to its parent heavy atom so the
signal survives in representations that do not draw hydrogens.

### 6.7 Relationship to existing metrics

The hotspot score is best understood as a **per-residue, per-atom localization of the composite idea
that the global MolProbity score expresses**. The wwPDB per-residue RSRZ already applies exactly this
"surprise, not badness, normalized to expectation" philosophy — but only to map fit; the geometry
metrics in that report remain boolean outlier flags. The hotspot score extends the RSRZ treatment to
the geometry metrics and puts everything on one absolute scale. Two aspects are genuinely new: a
single calibrated continuous severity spanning geometry *and* fit at residue/atom granularity, and
the option to modulate severity by local data support (asking not just "is this wrong" but "is this
wrong *and* well-supported by density"). A deliberate design rule keeps the aggregate honest: it is
allowed only to rank, and the residue table always decomposes it into its named components (Figure
7), so the number cannot be refined against as a target without its provenance being visible — the
same reason the wwPDB report stacks bands rather than merging them.

---

## 7. Implementation notes and performance

Several engineering decisions keep the interactive paths responsive on modest hardware (development
targeted a laptop as the floor). Coordinate updates use delta frames when few atoms move. The
severity cloud's cost is managed by the quality dial and empty-space skipping described above.
Long-running operations (hydrogen placement, probe, a full score, minimization, phasing) run on
background threads and are surfaced through a single, unified busy indicator so that a
multi-second computation never appears as a frozen interface; the button that starts each operation
is disabled for its duration to prevent duplicate work. The application is approximately 21,000
lines across the Python package and the TypeScript front end, and ships as a self-contained,
noarch conda package with the built Mol\* bundle included.

---

## 8. Availability and requirements

pxviewer is distributed as a conda package (the model-I/O dependencies — `cctbx-base` and the
`chem_data` monomer/validation library — are available only on conda channels). It runs on Linux
and macOS; the desktop shell requires PySide6/Qt with QtWebEngine, and interactive validation and
refinement require the cctbx monomer library. The built front end is bundled, so an install is
self-contained. *[Repository URL, license, and version to be added.]*

---

## 9. Discussion, limitations, and future work

pxviewer demonstrates that a molecular graphics application can treat a computational toolkit as its
single source of truth and remain fully interactive by streaming coordinates rather than reparsing
files. The architecture makes cross-feature sharing natural — most visibly in the hotspot score,
which reuses the same analyzers as conventional validation.

Several limitations are acknowledged. The map-fit severity anchors are conventions rather than
calibrated community thresholds, the one uncalibrated link in an otherwise threshold-anchored score;
placing them on a reference distribution (as RSRZ does for real-space fit) is the highest-value
future work. Accurate, hydrogen-based clash detection is expensive, and although it is now optional
and shared, the underlying *probe2* computation dominates the first score of a structure; a
clashes-only mode that skips the full contact-dot surface would help. The severity cloud, being a
raymarch, remains the most demanding renderer and is quality-gated accordingly. Finally, the score's
"wrong *and* well-supported" modulation by data support, while implemented in spirit through the
map-fit term, invites a more principled treatment.

Future directions include calibrating the map-fit anchors, an X-ray-native per-residue fit term
(RSRZ/RSCC) to complement the cryo-EM Q-score, and broadening the shared-analysis model so that
refinement and validation feedback close the loop even more tightly than the live difference map
already allows.

---

## Acknowledgements

*[To be added.]* This work builds directly on the Computational Crystallography Toolbox (cctbx),
the MolProbity validation suite, and the Mol\* visualization toolkit.

## References

*(Author–year; to be formatted to the target journal's style.)*

- Croll, T.I. (2018). ISOLDE: a physically realistic environment for model building into
  low-resolution electron-density maps. *Acta Cryst. D* **74**, 519–530.
- Emsley, P., Lohkamp, B., Scott, W.G., Cowtan, K. (2010). Features and development of Coot.
  *Acta Cryst. D* **66**, 486–501.
- Grosse-Kunstleve, R.W., Sauter, N.K., Moriarty, N.W., Adams, P.D. (2002). The Computational
  Crystallography Toolbox. *J. Appl. Cryst.* **35**, 126–136.
- Liebschner, D., et al. (2019). Macromolecular structure determination using X-rays, neutrons and
  electrons: recent developments in Phenix. *Acta Cryst. D* **75**, 861–877.
- Lovell, S.C., et al. (2003). Structure validation by Cα geometry: φ, ψ and Cβ deviation.
  *Proteins* **50**, 437–450.
- Pettersen, E.F., et al. (2021). UCSF ChimeraX: Structure visualization for researchers, educators,
  and developers. *Protein Sci.* **30**, 70–82.
- Pintilie, G., et al. (2020). Measurement of atom resolvability in cryo-EM maps with Q-scores.
  *Nat. Methods* **17**, 328–334.
- Sehnal, D., et al. (2021). Mol\* Viewer: modern web app for 3D visualization and analysis of large
  biomolecular structures. *Nucleic Acids Res.* **49**, W431–W437.
- Williams, C.J., et al. (2018). MolProbity: More and better reference data for improved all-atom
  structure validation. *Protein Sci.* **27**, 293–315.
- Word, J.M., et al. (1999). Visualizing and quantifying molecular goodness-of-fit: small-probe
  contact dots with explicit hydrogen atoms. *J. Mol. Biol.* **285**, 1711–1733.
