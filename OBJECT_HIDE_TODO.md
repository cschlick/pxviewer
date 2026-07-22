# Object hide: the segfault is fixed; making it smooth is what's left

This is a self-contained brief for whoever picks up the smooth-hide work. It assumes no
knowledge of prior conversations.

## Context

pxviewer is a desktop molecular-structure viewer: a PySide6/Qt app whose viewport is a
**QtWebEngine** (Chromium) view running a **Mol\*** (molstar) TypeScript frontend. The Python
backend streams data to the frontend over WebSockets.

## SOLVED: the segfault was never the GPU

Earlier versions of this brief said "this app's QtWebEngine GL context does not survive any
in-place GPU state change on this Mac" and sent people to chase ANGLE backends, driver
versions and streaming races. **That was a misdiagnosis.** The crash was a plain Qt
use-after-free in the object list, with no renderer involved at all.

### What actually happened

macOS crash reports (`~/Library/Logs/DiagnosticReports/python3.12-*.ips`) all showed the same
faulting stack, in the **main app process** — not the WebEngine GPU/render subprocess:

```
0  QTreeWidgetItem::setData(int, int, QVariant const&)      <- crash (EXC_BAD_ACCESS)
1  QTreeWidgetItemWrapper::setData(...)                      <- PySide6 virtual dispatch
2  QTreeModel::setData(...)
3  QStyledItemDelegate::editorEvent(...)                     <- the checkbox being toggled
5  QAbstractItemView::edit(...)
7  QAbstractItemView::mouseReleaseEvent(...)                 <- still inside the mouse release
```

The sequence:

1. You click a visibility checkbox. Qt runs `QTreeWidgetItem::setData(CheckStateRole)` from
   the tree's own mouse-release/edit stack.
2. `setData` emits `itemChanged` **synchronously**, mid-call.
3. `_on_tree_item_changed` ran `set_model_visible` / `set_volume_visible` right there, which
   reached `_emit_loaded_changed` -> `_on_loaded_changed` -> `QTreeWidget.clear()`.
4. `clear()` destroyed every item — including the one whose `setData` was still on the stack.
5. `setData` and its callers kept using that freed item as the stack unwound. SIGSEGV, with
   `KERN_INVALID_ADDRESS ... (possible pointer authentication failure)`.

### Why it looked like a GPU bug

- It fired on a *hide*, so it correlated perfectly with whatever hide mechanism was in play.
- **Three unrelated hide mechanisms all crashed identically** — the clip slab, Mol\*'s
  `setSubtreeVisibility`, and the plain scene reload. Three different renderer paths sharing
  one byte-identical Qt backtrace is the tell that none of them was the cause.
- It reproduced on **SwiftShader (pure CPU) as well as the real GPU**. A GPU-context bug that
  reproduces on a software rasterizer is not a GPU-context bug.
- The "stable reload baseline" crashed too — a crash report at 01:08 lands after the 00:11
  revert to reload-based hiding, with the same stack as the in-place ones. The reload path was
  never actually safe; it just crashed less often.
- "Crashed on rapid toggling" fits exactly: fast clicking maximizes the chance of the rebuild
  landing inside a live mouse-release stack.

### The fix

Read the plain values off the item, then apply the change on the next event-loop turn
(`QTimer.singleShot(0, ...)`), once Qt has finished with the item. Nothing that runs inside
`itemChanged` may touch the tree. See `ControlsWindow._on_tree_item_changed` /
`_apply_visibility` in `python/pxviewer/desktop.py`.

Two sibling paths had the identical defect and were hardened the same way:

- `_on_tree_current_changed` -> `set_active_model` -> rebuild, from inside the tree's
  selection handling.
- `_on_active_radio` -> `set_active_model` -> rebuild. Worse: the radio lives *in* the tree
  via `setItemWidget`, so the rebuild deletes the very button still delivering its own click.

Regression test: `test_toggling_a_visibility_box_does_not_rebuild_the_tree_inside_the_signal`
in `python/tests/test_desktop.py` (verified to fail without the fix).

**The general rule for this codebase: never rebuild the object tree from inside a signal
emitted by one of its own items or item widgets. Defer it.**

## What's left: kill the flicker

Hiding is still **reload-based**, so it works and no longer crashes, but hiding one object
reloads the page and every other object briefly blanks. Making it smooth means in-place
hiding — and the thing that blocked it is gone.

Commit `64da4c5` ("Hide objects in place (no flicker)") already implemented this end to end
and was reverted in `1b32eff` **only because of the crash diagnosed above**. Reverting that
revert is the obvious starting point:

```
git revert 1b32eff     # restores in-place hiding via setSubtreeVisibility
```

That commit's design: `LiveSession.set_structure_visible` toggles a model's own structure
(replayed to late clients, so a reload keeps a hidden model hidden); `set_volume_visible`
toggles a map's isosurface by ref; `_reassert_hidden_volumes` re-hides maps after any reload.

**One unresolved symptom to expect.** The revert commit reported two problems: the crash
(solved) and that `setSubtreeVisibility` *"didn't hide"* visually on macOS. The second is a
separate, real issue and is now the only thing standing between here and smooth hiding. Leads:

1. Confirm it sets state at all: after a toggle, check `cell.state.isHidden` on the target
   ref. Headless it did set correctly, so the plumbing works — the question is the render.
2. The render probably just didn't sync. Force it: `plugin.canvas3d?.requestDraw(true)` after
   the toggle.
3. Try Mol\*'s higher-level command instead of the raw helper:
   `PluginCommands.State.ToggleVisibility.dispatch(plugin, { state: plugin.state.data, ref })`.
4. Check the ref actually targets the drawn node. For a volume, `findVolumeReprCell(plugin,
   ref)` must resolve to the isosurface repr cell; for a model, hiding the *structure* ref may
   need to be the representation ref instead.

Do this work on the Mac with hardware WebGL, and re-verify by toggling rapidly — that is what
used to crash, and it should now be boring.

## Also worth revisiting

`_can_hide` still refuses hiding on software rendering (checkboxes non-checkable, a click
flashes why). That restriction exists only because hiding "segfaulted on software" — which
was this same tree bug, not the renderer. It is probably safe to lift entirely, but software
rendering has not been re-tested since the fix, so it was left in place. Verify, then remove
the policy and its tests together.

## Key code

- `python/pxviewer/desktop.py` — `DesktopApp`. Hide entry points: `set_model_visible`,
  `set_volume_visible`. Scene composition: `_reload_viewport`, `_model_ws`,
  `_write_volume_scene`, `_control_session`. Object lists: `self._models`, `self._volumes`
  (dicts with `"visible"`, `"session"`, `"ref"`). Tree: `ControlsWindow._on_loaded_changed`
  (rebuilds it), `_on_tree_item_changed` (the deferral).
- `python/pxviewer/live.py` — `LiveSession` (one per model, its own WebSocket). Sends JSON
  text control messages and tagged binary frames; replays state to late clients in `_handler`.
- `frontend/src/live.ts` — the Mol\* integration. `connectLive(plugin, url)` runs one per model
  in ONE shared plugin. `class LiveViewer` holds `this.structure`. `handleControlMessage`
  dispatches text messages. Volumes load from an MVSJ scene (`?mvsj=...`);
  `findVolumeReprCell(plugin, ref)` locates a volume's repr cell by ref.

## Build / run / test

- Env: `conda activate pxviewer` (conda-forge channels only; never accept Anaconda's ToS).
- Python is an editable install, so Python edits are live on restart.
- **Frontend edits require a rebuild**: `cd frontend && npm run build` (writes `build/index.js`,
  which is gitignored), then restart the app.
- Run: `python -m pxviewer desktop` (or `pxviewer desktop`).
- Tests: `QT_QPA_PLATFORM=offscreen python -m pytest python/tests/test_desktop.py
  python/tests/test_live.py python/tests/test_gui_fuzz.py -q`. Hide behavior is covered by
  `test_hiding_a_model_reloads_the_scene_without_it`,
  `test_hiding_a_map_reloads_the_scene_without_it`,
  `test_toggling_a_visibility_box_does_not_rebuild_the_tree_inside_the_signal`,
  `test_software_pins_a_model_and_says_why_on_click`, and the `_control_session_is_reachable`
  invariant in `python/tests/gui_invariants.py`. (The full `tests/` run in one process hangs
  on a pre-existing leaked-thread issue — run per-file.)
- Five tests in `test_desktop.py` fail on a clean checkout for unrelated pre-existing reasons
  (`test_minimize_buttons_show_which_state_is_live`, `test_validation_subtabs_and_row_focus`,
  `test_restraint_row_marks_all_atoms_and_draws_its_notation`,
  `test_atom_precision_actions_switch_a_ribbon_to_ball_and_stick`,
  `test_residue_orientation_and_space_navigation`). Don't be alarmed by them; don't blame them
  on hide work.

## Constraints

- Commits authored as `cschlick <cschlick@users.noreply.github.com>`, no AI attribution.
- Push only when explicitly asked.
- Above all: **do not leave the app in a state that segfaults.**
