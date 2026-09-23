# Manual test passes

Interactive walks that cover the program's functionality end to end, written to surface
what the regression suite cannot: visual glitches, layout breakage, sluggishness, and
"this feels wrong" usability problems. The suite proves the machinery fires; these passes
are where a human notices that the result looks bad.

How to use them:

- Each pass is 5–15 minutes and independent. A full sweep is ~90 minutes; before a
  release, run everything. After a focused change, run the pass that owns the area plus
  **Pass 0** and **Pass 10**.
- The **Watch for** lines are the point. Do the step slowly, then actually look —
  most visual bugs live in the half-second after an action, in resizes, and in the
  second time you do something.
- Run on the hardware and screen size you actually use, and at least once on a small
  window (13"-laptop-sized) — several past bugs only existed at panel widths.
- Keep the terminal you launched from visible: stray tracebacks, Qt warnings, and
  asyncio "task destroyed" noise are all bugs even when the GUI looks fine.

Automated coverage is documented in `TESTING.md`; nothing here replaces it.

---

## Pass 0 — Launch and first look (2 min)

1. Launch `pxviewer` from a terminal.
   - **Watch for:** the splash appears promptly (no long blank gap), then the main
     window replaces it cleanly — no flash of a half-built window.
2. With macOS set to **dark** appearance, launch again.
   - **Watch for:** the app is fully light regardless — no dark panels, no
     half-light/half-dark mix, no unreadable text anywhere. Dark mode is pinned off.
3. Look over the right panel: six icon tabs — Scene, Tools, Validation, Geometry,
   Console, Settings (Hotspots is a sub-tab of Validation, not a tab of its own).
   - **Watch for:** tabs share the full bar width with no dead grey strip on the right;
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
   change **Settings ▸ Viewer ▸ Draw map density within**, then untick it.
   - **Watch for:** every map follows, including the ones already open and the ones that
     came from different places. Bounding is one setting for all of them; a map drawn at
     some radius of its own, or one that ignores the control because it was made before
     the setting changed, is the unevenness this replaced. There is no per-map Radius row
     any more.

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

1. In the Selection pane, confirm **Focus on selection** and **Clip to selection** are
   both checked by default. Type `resseq 29` and apply.
   - **Watch for:** the residue fills most of the frame; backbone N on the left, C on
     the right, side chain pointing up; nothing important lands offscreen; near/far
     clipping isolates the residue from the rest of the molecule.
2. Uncheck **Clip to selection**, apply again.
   - **Watch for:** same framing, but the whole structure stays visible around it.
3. Uncheck **Focus on selection**, apply a different residue.
   - **Watch for:** the selection highlights but the camera stays put.
4. Select a helix or a whole chain (e.g. `resseq 20:35`).
   - **Watch for:** the view orients along the selection's long axis, centred, filling
     the frame — not a random skewed angle.
5. Press space repeatedly (residue navigation), then shift/space back.
   - **Watch for:** each step lands oriented the same way as typed selections; stepping
     honours the clip checkbox; no drift or roll accumulating over many steps.
6. Pick atoms in the viewport; watch the description label and atoms table. Select rows
   in the atoms table instead.
   - **Watch for:** both directions agree; the label counts what you actually picked;
     Hide-selected / Show-selected enable only when something is selected, and do what
     they say.

## Pass 6 — Local resolution (15 min)

Run **Tutorials ▸ Look at local resolution** twice.

1. First run (cold, with the cache cleared or on a fresh machine).
   - **Watch for:** fetch progress for each entity; the local-resolution computation
     announces itself (not a silent hang); the busy indication holds until the coloured
     surface is *actually* draggable — not until some internal step.
2. The Objects panel afterwards.
   - **Watch for:** ONE row for the map — no phantom second "resolution" object, no
     giant blue spheroid, no tiny broken checkbox.
3. Appearance: Color ▸ **Local resolution**.
   - **Watch for:** the sub-panel appears only when this coloring is on and matches the
     pane's usual look (not undersized); the colour range shows real Å numbers; **Fit**
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
     rotation stays fluid; selecting a residue keeps the CC colours on the
     neighbourhood sticks (not element colours); picking another colour and
     re-picking recomputes against the model as it now stands.

## Pass 7 — Tutorials sweep (15 min)

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
   demo-data ones (cryo-EM, X-ray) and both restraint-edit ones.
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
3. Geometry tab: Atoms subtab and each restraint subtab; click rows.
   - **Watch for:** tables fill, sort, and follow the active model; clicking a restraint
     row marks it in the viewport in ball-and-stick.
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
   - **Watch for:** results render; overshoot behaviour matches expectations from the
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
   - **Watch for:** dialogs open centred and close cleanly; the saved screenshot matches
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
   - **Watch for:** empty states everywhere are calm and labelled — no error dialogs
     for ordinary emptiness.
5. Quit the app from a busy moment (mid-render, tutorial open).
   - **Watch for:** the process exits; the terminal shows no "task was destroyed" or Qt
     object-deleted warnings.
