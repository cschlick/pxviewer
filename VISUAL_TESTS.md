# Visual test passes

End-to-end walks that cover what the regression suite cannot: chained state the
unit tests never leave standing, and results that only a rendered frame can show.
Two things make a check belong here rather than in `regression/`: **(A)** it needs
chained state — a clip standing while representations rebuild, a hidden model
receiving a selection — and **(B)** its verdict lives in the pixels or in the
behavior across steps, not in a single state read.

Most of the time these run automated. `python/pxviewer/regression/visual/` holds
a driver (`harness.py`) and one `tst_visual_*.py` per scripted pass — the real
app, real QtWebEngine, real GPU, asserting on state and pixel-diffs and writing
screenshots to `$PXVIEWER_VISUAL_DIR` (default `$TMPDIR/pxviewer-visual`) for
review:

```bash
PXVIEWER_VISUAL=1 libtbx.python -m pxviewer.run_tests                     # with the suite
libtbx.python python/pxviewer/regression/visual/tst_visual_chains.py     # one pass
```

The human half of each pass is what a script cannot judge — "this feels wrong",
resizes, the half-second after an action. Steps that need a real gesture or a
subjective call are marked *(manual)* in their pass.

How to use them:

- Each pass is 5–15 minutes scripted or by hand. A full sweep is ~100 minutes;
  before a release, run everything. After a focused change, run the pass that
  owns the area plus **Pass 0** and **Pass 10**.
- When running by hand: the **Watch for** lines are the point. Do the step
  slowly, then actually look — most visual bugs live in the half-second after an
  action, in resizes, and in the second time you do something.
- Run on the hardware and screen size you actually use, and at least once on a
  small window (13"-laptop-sized) — several past bugs only existed at panel
  widths.
- Keep the terminal you launched from visible: stray tracebacks, Qt warnings,
  and asyncio "task destroyed" noise are all bugs even when the GUI looks fine.
  (Automated runs keep stderr too — check the script's output after the `[PASS]`
  lines.)

Automated regression coverage is documented in `TESTING.md`; nothing here
replaces it.

---

## Pass 0 — Launch and first look (2 min)

1. Launch `pxviewer` from a terminal.
   - **Watch for:** the splash appears promptly (no long blank gap), then the main
     window replaces it cleanly — no flash of a half-built window.
2. With macOS set to **dark** appearance, launch again.
   - **Watch for:** the app is fully light regardless — no dark panels, no
     half-light/half-dark mix, no unreadable text anywhere. Dark mode is pinned off.
3. Look over the right panel: six icon tabs — Scene, Geometry, Validation, Tools,
   Console, Settings (Hotspots is a sub-tab of Validation, not a tab of its own).
   - **Watch for:** tabs share the full bar width with no dead gray strip on the right;
     icons are crisp (not blurry — a HiDPI regression); the selected tab's underline is
     visible; hovering shows a label.
4. Hover every toolbar icon button (Open, Tutorials, Write, Pair, trash, reset view,
   picture, dock, mouse, help).
   - **Watch for:** each has a meaningful tooltip; buttons are uniformly sized and
     framed; the Open and Tutorials buttons show **no** menu-indicator dot.

## Pass 1 — Opening and fetching (10 min)

1. Open ▸ **Open file(s)…** — pick a local PDB.
   - **Watch for:** the model appears framed sensibly; its row appears in Objects with a
     bold name (it is active); the Appearance pane header says `Appearance · <name>`.
2. Open ▸ **Fetch from PDB / EMDB…** — fetch `1ubq`, model only.
   - **Watch for:** progress starts at 0 and moves *during* the download (a frozen bar
     that jumps to done means streaming broke); stage wording is readable
     ("downloading", "decompressing"); the dialog can't be left in a stuck state.
3. Fetch `9r04` with EMDB `53478` including the map (large — a real progress test).
   - **Watch for:** byte counts read like a person wrote them ("49.6 MB"); the UI stays
     responsive while downloading; cancel/close mid-fetch leaves no broken half-state.
4. Fetch a nonsense ID (e.g. `9zzz`).
   - **Watch for:** the failure says *which* entry and what to do, in one readable
     message — not a raw traceback or a silent nothing.
5. Re-run a tutorial fetch you have done before.
   - **Watch for:** cached files are reused ("cached" flashes past, near-instant), not
     re-downloaded.

## Pass 2 — Objects panel semantics (10 min)

Load two models and one map first.

1. Read a row cold: eye icon, then name. Click each **eye**.
   - **Watch for:** the object disappears/reappears in the viewport immediately; the
     eye swaps to the dimmed eye-off; the *camera does not move*; nothing else flickers;
     the eye is never clipped for a nested (pinned) row.
2. Click each row in turn.
   - **Watch for:** the highlight moves; a clicked *model* becomes bold (active) and the
     previous bold clears; the Appearance pane retitles and rebuilds to the clicked
     object with no flicker or stray floating widgets; a map row highlights without
     stealing the bold from the active model.
3. Click atoms of the *non-active* model in the viewport.
   - **Watch for:** that model's row highlights and goes bold by itself — the panel
     follows your attention. The Selection pane description and atoms table follow too.
4. Load the X-ray demo (model + reflections group).
   - **Watch for:** the group header reads as a header (bold, spanning); members are
     indented under it; the reflections row has no eye (nothing drawable); names align
     whether grouped or loose.
5. Remove objects with the trash button, ending with the active model.
   - **Watch for:** removing the active model promotes another cleanly (bold moves);
     removing the last object leaves the pane in its empty state — title back to plain
     "Appearance", hint text "Select an object above…", no stale name.
6. Load a file with a long name.
   - **Watch for:** the name elides with "…" instead of widening the panel; hovering
     shows the full name.

## Pass 3 — Model appearance (10 min)

1. Cycle every representation (cartoon, ball-and-stick, spacefill, …) on a protein.
   - **Watch for:** each switch is prompt; no leftover geometry from the previous style;
     the dropdown reflects reality after tutorials or measurement tools switch styles
     behind your back.
2. Cycle every Color option, including **By B-factor** and **By occupancy**.
   - **Watch for:** the value-scale group appears only for the value colorings, with
     sane numbers and units (Å² for B); **Fit** snaps the range to the model; **Reset**
     restores defaults; typing your own range recolors and then *stays put* — nothing
     silently recomputes your scale afterwards.
3. Open the structure-type dropdown; toggle Water, Protein, Mol* interactions.
   - **Watch for:** the click that opens the popup does not itself toggle an item;
     each toggle hides/shows just that class; everything-off leaves an empty but stable
     viewport.
4. Set a custom color where offered.
   - **Watch for:** swatches render as color chips, not hex strings; a picked custom
     color joins the list and stays selected after the pane rebuilds.
5. The pane's clip controls are two independent rows — **Clip slab** (near/far along
   the view axis) and **Clip sphere** (Off / On selection / Around view center /
   Fixed point, with a radius). Select a residue, then open the model's Appearance
   pane.
   - **Watch for:** the Clip sphere row reads **On selection** — the default mode,
     which is what makes selections clip at all. A selection clip is not invisible
     state, it lands here and is editable. Switching the row to Off lifts the
     standing sphere *and* disarms the policy (later selections won't re-clip);
     **Around view center** follows the camera target; **Fixed point** stays put
     while the camera moves. The slab slider keeps its positions through all of
     it — slab and sphere compose, they never overwrite each other.

## Pass 4 — Map appearance (10 min)

Open a map (or the cryo-EM tutorial data).

1. Drag the **Level** slider slowly end to end.
   - **Watch for:** contour follows the drag without stutter; the far right end takes
     the surface all the way to *nothing visible*; slider and spin stay in sync; letting
     go does not trigger a second "correction" render.
2. Scroll the mouse wheel over the viewport with the map's row current.
   - **Watch for:** the wheel contours *this* map (not another object); direction feels
     natural; the Level controls track the new value.
3. Switch Style surface ↔ mesh; change colors.
   - **Watch for:** clean swap, no double-draw; color applies to the whole surface.
4. Change **Downsample** through Full / 2× / 4× / 8×.
   - **Watch for:** detail changes only when *you* change this dropdown — never
     mid-drag on the Level slider (the old "recalculates and looks bad on release" bug);
     Full is slower but works; the choice sticks.
5. Pair the model with its map (Pair button) and use masking in Tools.
   - **Watch for:** density away from the molecule disappears; the refined-against map
     is unaffected (minimize still behaves).
5a. With several maps open — one from a file, one from reflections, a difference map —
   tick **Settings ▸ Viewer ▸ Draw map density within**, change the radius, untick it.
   - **Watch for:** it starts unticked and every map is drawn in full. Every map then
     follows the control together, including the ones already open and the ones that
     came from different places. Bounding is the default for all of them; a map drawn at
     some radius of its own, or one that ignores the control because it was made before
     the setting changed, is the unevenness this replaced. Each map's own Sphere row
     (Appearance pane, **Around view center**) is the same primitive per map — the
     setting changes it on every map, the row overrides it on one.

6. Make a difference map (Tools ▸ Map tools ▸ **Difference**, or phase one with Make
   maps).
   - **Watch for:** TWO rows appear — the map, and **negative contour** nested under it,
     each with its own eye. Every contour on screen is owned by a row: anything drawn
     that no row can hide and no slider can reach is the bug this checks.
7. Untick the **negative contour** row's eye.
   - **Watch for:** the red goes, the green stays. Now toggle the *map's* eye off and on
     — the red must **stay** off, because "just the green one" is a reading of the map,
     not a half-hidden object.
8. Select the **negative contour** row.
   - **Watch for:** its own pane — Style, Color, Opacity, and a Level under
     **Link level with positive map**. Change its Style and Opacity and confirm the
     green contour does not move: with a row each, a control on one that silently
     changed the other would be a lie.
9. Untick **Link level with positive map**, then drag that pane's **Level**.
   - **Watch for:** only the red contour changes. Select the map's row and drag its
     Level — only the green changes. Re-tick the link and the red jumps to the map's
     level *as it stands now*, not as it was when you unlinked.
10. With the negative contour hidden, load any file (a scene rebuild).
   - **Watch for:** the red contour stays off. Its replay reaches the viewer before the
     new scene has been parsed, so a lookup that does not wait for the contour silently
     drops the setting and the red comes back.
11. Untick the map's eye, drag the **Level** slider somewhere clearly different, then
   tick the eye again.
   - **Watch for:** the map returns contoured at the level you left the slider on, not
     the one it had when you hid it. If it returns at the old level, the panel and the
     viewer now disagree and the Level control is **dead** at that number until nudged
     to a different one — which is what "the controls stopped responding" looks like.
     ``window.__dumpVol('<ref>')`` in the viewport console reports each contour's live
     level and whether it is hidden.
12. Drag the **Level** slider right across its range in one sweep, on the biggest map
   you have.
   - **Watch for:** the map follows the handle and stops when you let go. It must not
     keep working through the levels you dragged past — on a real box that is minutes
     of it "clicking its way" towards your setpoint. To check the mechanism rather than
     the feel, read ``window.__isoApplied()`` in the viewport console before and after
     one sweep: one sweep emits ~56 level changes, and the viewer should draw a handful,
     not all of them. (Measured on the same sweep: 55 drawn before coalescing, 4 after.)

## Pass 5 — Selection and oriented focus (10 min)

Load a protein (1ubq works).

1. In the Selection pane, confirm **Focus on selection** is checked by default, and
   in the model's Appearance pane confirm **Clip sphere** reads **On selection**.
   Type `resseq 29` and apply.
   - **Watch for:** the residue fills most of the frame; backbone N on the left, C on
     the right, side chain pointing up; nothing important lands offscreen; near/far
     clipping isolates the residue from the rest of the molecule.
2. With a clip applied, set the model's **Clip sphere** to **Off** without applying
   anything new.
   - **Watch for:** the surrounding structure reappears immediately — the mode
     governs the selection's clip both ways: Off lifts it, On selection puts it
     right back without a fresh selection. Apply a selection while Off: same
     framing, but the whole structure stays visible around it.
3. Uncheck **Focus on selection**, apply a different residue.
   - **Watch for:** the selection highlights but the camera stays put.
4. Select a helix or a whole chain (e.g. `resseq 20:35`).
   - **Watch for:** the view orients along the selection's long axis, centered, filling
     the frame — not a random skewed angle.
5. Press space repeatedly (residue navigation), then shift/space back.
   - **Watch for:** each step lands oriented the same way as typed selections; stepping
     honors the object's Clip sphere mode; no drift or roll accumulating over many
     steps.
5a. Click a row in any table — a validation sub-tab, the bonds or angles table, the
   atoms table — then press space repeatedly, and shift/space back.
   - **Watch for:** the *selection* walks the table a row at a time and the viewport
     follows each row, rather than the model stepping along the chain underneath a list
     that never moves. It stops at the first and last row: running off the end of a
     worklist should be visible, and the model jumping at that moment would read as a
     bug.
   - **Watch for:** the table you last engaged *keeps* the key — switch to the Scene
     tab or click the viewport and space still walks that table, not the residue
     chain. The residue walk is only the default before any table has been touched;
     switching to another table (its sub-tab, a row) is what moves the key to it.
   - **Watch for:** each row is a real selection — Components/Atoms/restraint rows
     alike frame, isolate and dress the neighborhood as the Selection pane's Focus
     checkbox, the object's Clip sphere mode and the Settings tab's Neighborhood
     checkbox say, so stepping the list clips only while the mode is **On
     selection**, and a restraint row's sphere centers on the restraint's atoms
     rather than inheriting whatever a previous selection left.
6. Pick atoms in the viewport; watch the description label and atoms table. Select rows
   in the atoms table instead.
   - **Watch for:** both directions agree; the label counts what you actually picked;
     Hide-selected / Show-selected enable only when something is selected, and do what
     they say.
7. Watch the **Click selects** control in the Selection pane while clicking the
   viewport, then engage the Components table (its sub-tab or a row) and click
   again, then a restraint sub-tab and click once more.
   - **Watch for:** the control says Atom by default and every click takes
     exactly the atom under the cursor — the selection box names it
     (`… and name CA`) and the label and atoms table agree. Engaging
     Components or a validation table flips it to Residue (clicks take the
     whole residue); the Atoms table and restraint sub-tabs — whose rows are
     atom collections — flip it back. Setting it by hand holds until the next
     table engagement, and Shift-click grows and shrinks by the same unit —
     atoms in Atom, residues in Residue. Representation bounds the aim too:
     on a Cartoon ribbon a click can only aim at a residue, so even Atom
     takes the residue there; on Ball & stick (and the ball-and-stick
     neighborhood layer under a standing selection) it is genuinely per-atom.

## Pass 6 — Local resolution (15 min)

Run **Tutorials ▸ Look at local resolution** twice.

1. First run (cold, with the cache cleared or on a fresh machine).
   - **Watch for:** fetch progress for each entity; the local-resolution computation
     announces itself (not a silent hang); the busy indication holds until the colored
     surface is *actually* draggable — not until some internal step.
2. The Objects panel afterwards.
   - **Watch for:** ONE row for the map — no phantom second "resolution" object, no
     giant blue spheroid, no tiny broken checkbox.
3. Appearance: Color ▸ **Local resolution**.
   - **Watch for:** the sub-panel appears only when this coloring is on and matches the
     pane's usual look (not undersized); the color range shows real Å numbers; **Fit**
     and **Reset** behave; your chosen range survives level changes.
4. Drag Level at the default 4× downsample, then at Full.
   - **Watch for:** 4× is fluid; Full is honest about being slower but never swaps
     detail behind your back; zoomed *in*, dragging Level still never degrades-then-
     restores.
5. Second run of the tutorial.
   - **Watch for:** the saved resolution map loads from disk — seconds, not the
     original computation.
6. Map-model CC: run **Tutorials ▸ Real-space refine into cryo-EM density** (it loads
   7BV2 + its map with one helix displaced), then set the model's **Color** to
   **By map-model CC**.
   - **Watch for:** it computes in a few seconds and paints atoms teal→pink on the
     0–1 correlation scale; the displaced helix reads pink against a teal molecule;
     rotation stays fluid; selecting a residue keeps the CC colors on the
     neighborhood sticks (not element colors); picking another color and
     re-picking recomputes against the model as it now stands.

## Pass 7 — Tutorials sweep (15 min)

The scripted half lives in `regression/visual/tst_visual_tutorials.py`: it
starts each bundled-data tutorial, walks every step to the end, and checks the
coach opens, reserves its height, and is put away cleanly (the three
fetch-dependent tutorials run under `PXVIEWER_VISUAL_NET=1`). What it does not
do is *perform* each step's action — per-step predicates are exercised by
`tst_desktop_tutorials.py`; the hands-on checks below are the rest.

1. Open the Tutorials menu (graduation cap).
   - **Watch for:** ten entries, stable order, titles match what they teach.
2. Run **Open a model** and **Alternate conformations** to completion, doing exactly
   what each step says.
   - **Watch for:** wording matches the real UI ("Objects list", "Appearance pane" —
     never a label that doesn't exist); each step auto-advances the moment you do the
     thing (a step that never advances = a broken predicate); the coach pane keeps ONE
     height for the whole tutorial — no jiggling between steps; highlighted target
     widgets are the right ones; exiting mid-tutorial restores the normal layout.
3. Start each remaining tutorial and complete its first step or two, including both
   demo-data ones (cryo-EM, X-ray) and the restraint-edits one — it should load the
   sample PHIL, then clear it and author the same bond by hand.
   - **Watch for:** each loads its own example without touching your other loaded
     objects unexpectedly; fetch-dependent tutorials fail readably when offline.

## Pass 8 — Geometry and editing (15 min)

Load the X-ray demo or a model with restraints available.

1. Minimize: press play; watch; press pause; press play again.
   - **Watch for:** only the live button is enabled and accent-filled; the structure
     visibly relaxes; pausing stops promptly with a status message; a converged run
     doesn't pretend to still be working.
2. Enable **refine drag** and tug an atom; then switch to **pick**.
   - **Watch for:** the two modes are mutually exclusive (buttons show it); dragging
     moves atoms — it never *also* selects; with the X-ray demo, the difference map
     updates live in a box around the drag; arming a drag pauses a running minimize.
3. Geometry tab: Components, Atoms, and each restraint subtab; click rows, then
   step them with Space.
   - **Watch for:** Components lists one row per residue and stepping walks the
     residues (the row is the same selection a residue click makes); Atoms steps
     one atom at a time; tables fill, sort, and follow the active model; clicking
     a restraint row selects its atoms — marked, framed and clipped like any
     other selection — and draws its measurement notation on top.
4. Validation tab: press play with the default ticks; open every subtab; click rows.
   Then tick **Clashes & contacts** and run again.
   - **Watch for:** results appear per-section; only ticked checks produce subtabs;
     clicking a finding focuses the culprit — the first click switches the model to
     ball-and-stick and later clicks feel instant, with no node errors in the
     terminal; switching back to Cartoon in Appearance sticks (row clicks don't
     re-flip it); the staleness warning appears after you
     edit the model and clears on re-run; the clashes run adds a "+ H" object, hides
     the original, and lights up the Contacts/Clashes toggles.
5. Place a marker; build a ligand from SMILES at it (e.g. `CCO`). Try a ligand the
   monomer library doesn't know.
   - **Watch for:** the built ligand appears at the marker as its own object; the
     unknown ligand *offers* restraint inference rather than failing mutely — from
     Minimize and drag too, not only the Geometry tab.
6. Validation ▸ **Hotspots** sub-tab: run it on the loaded structure.
   - **Watch for:** results render; overshoot behavior matches expectations from the
     pinned footprints; no layout breakage in its panel.

## Pass 9 — Persistence, settings, window management (5 min)

1. Change several stateful things: Focus/Clip checkboxes, Downsample, a color. Quit and
   relaunch.
   - **Watch for:** your choices held; nothing else leaked between sessions.
2. Settings tab: change what it offers; confirm effects.
3. Shrink the window toward 13"-laptop size.
   - **Watch for:** no horizontal scrollbars in panes; the Objects list gives up its
     spare height instead of pushing everything into a scrollbar; the coach pane, tab
     bar, and Appearance pane all stay usable; nothing overlaps.
4. Dock/undock the panel; use the picture (screenshot) button and reset view; open the
   mouse-bindings and help dialogs.
   - **Watch for:** dialogs open centered and close cleanly; the saved screenshot matches
     the viewport; reset view actually reframes.

## Pass 10 — Stress and rough handling (10 min)

1. Load five-plus objects (models + maps), then rapidly: switch tabs, click tree rows,
   toggle eyes, drag Level.
   - **Watch for:** no crash (the historic tree SIGSEGV lived here), no widget flicker
     storm, no stray floating combo-box windows, no growing lag.
2. Remove a model *while* it is minimizing; remove a map mid-Level-drag.
   - **Watch for:** clean stop, no orphaned busy state, no traceback in the terminal.
3. Kill the network (Wi-Fi off), then try a fetch.
   - **Watch for:** a readable failure, and the app fully usable afterwards; no `.part`
     litter adopted as a real file by a later cached run.
4. Hide every object; select-all in an empty scene; apply a selection matching nothing.
   - **Watch for:** empty states everywhere are calm and labeled — no error dialogs
     for ordinary emptiness.
5. Quit the app from a busy moment (mid-render, tutorial open).
   - **Watch for:** the process exits; the terminal shows no "task was destroyed" or Qt
     object-deleted warnings.

---

## Pass 11 — Chained sequences (mostly automated) (15 min)

Multi-step chains where each step leaves standing state the next step leans on —
this is where interaction bugs (stale caches, orphaned scene state, echo wipes,
clips surviving rebuilds) actually hide. **Steps 1–5, 7–12 and 14 are scripted**
in `regression/visual/tst_visual_chains.py`; steps 6 and 13 need a real gesture
or a map and stay *(manual)*. When running by hand, `runJavaScript` on the
viewport page reads Mol* state and the `app._models`/`_scene_selection`/
`_auto_clip` keys the Python side keeps are the assertions' ground truth. Two
models are needed; 1ubq + one more protein works.

1. Load A → eye it off → load B → eye it off → eye A back on.
   - **Watch for:** A reappears exactly where it was (hidden objects keep view
     state); the row styles track reality at every step; showing/hiding never moves
     the camera.
2. With A shown and B hidden, click B's row so it is the *active* model, then apply
   `resseq 5` in the Selection pane.
   - **Watch for:** the selection applies to the hidden B — the label and atoms
     table agree with the hidden model; unhiding B afterwards shows the highlight
     already standing.
3. On A, apply `resseq 29` with Focus and Clip on. Then cycle its representation
   cartoon → ball-and-stick → cartoon.
   - **Watch for:** the clip sphere survives the rebuild — clip objects are baked
     into each new representation, not just the one they were applied to; the view
     after the round trip is still the clipped residue, not the lifted scene.
4. Set the model's **Clip sphere** to **Off**, wait a beat, back to **On
   selection**. Compare against the clipped frame (screenshot diff or eye).
   - **Watch for:** the same clipped view returns — depth slab included, not just
     the sphere. A reapplied frame that matches the *lifted* frame is the bug.
5. With the clip standing, engage the Components table and Space-step five rows.
   - **Watch for:** the sphere re-centers on each stepped residue and the
     neighborhood context follows; no stepped residue is ever rendered outside
     its own sphere.
6. *(manual)* Shift-click a second residue in the viewport to grow the selection.
   - **Watch for:** the sphere re-fits to the grown selection; nothing added is
     clipped out of view; the camera does not move.
7. Clear the selection entirely, then switch **Clip sphere** Off and back to
   On selection with nothing selected.
   - **Watch for:** re-arming an empty selection is a calm no-op — no sphere
     stranded mid-scene, no camera lurch, no state marker claiming a clip exists.
8. Restore a selection clip (`resseq 29` again), then Minimize for a few seconds
   and pause.
   - **Watch for:** the sphere stands where it was fit — it does not chase moving
     atoms. Then open a Geometry restraint sub-tab: the model/delta/residual
     columns must describe the *moved* sites, not the pre-minimize snapshot (the
     stale-geometry-cache bug lived here).
9. With the clip on A standing, select a residue on B. Then eye A off and on.
   - **Watch for:** clip state is per-model — B's selection never writes into A's
     viewer; A returns still clipped around its own selection; B's view is
     unaffected.
10. Rapid storm: flip A's **Clip sphere** Off/On selection ten times fast, cycle A's
    representation five times, click three restraint rows quickly in succession.
    - **Watch for:** the final state wins everywhere (no mid-storm commit lands
      last); the last restraint row clicked stays selected after its WS echo; the
      terminal shows no Mol* exceptions. (A `uniform*: no array` warning when a
      clip is *cleared* is a known benign Mol* shader-rebuild message.)
11. Remove A while its clip stands. Then remove B.
    - **Watch for:** the clip dies with the model — no orphaned sphere drawn over
      the remaining scene; removing the last object leaves the empty state clean.
12. Reload A and apply `resseq 29` again.
    - **Watch for:** a fresh clip behaves like a fresh clip — sphere centered on
      the new selection, depth slab tight; no bookkeeping from the removed model's
      clip leaks into the new one.
13. *(manual)* Load a map alongside a clipped model (difference map or cryo-EM demo),
    drag its Level while the selection clip stands, then lift and re-apply the
    model clip.
    - **Watch for:** map contours and model clips are independent channels — the
      map neither inherits nor eats the model's sphere; the Level still tracks
      mid-clip; lifting and re-applying the model clip leaves the map untouched.
14. Drive **Hide selected** then **Show selected** on the Selection pane with a
    clip standing.
    - **Watch for:** hidden atoms leave the highlight consistently; the sphere and
      context still describe the selection you made, and showing them back lands
      exactly where they were.
15. *(manual)* Long loop: steps 3–5, 8, 10 chained three times back to back.
    - **Watch for:** no growth in console warnings, memory, or stray state — the
      same sequences must behave identically on the third pass as the first.
