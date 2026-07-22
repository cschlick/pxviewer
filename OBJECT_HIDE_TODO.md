# Object hide: resolved — a post-mortem worth keeping

Hiding an object used to segfault the app, and the workaround for that (hide by reloading the
page) made every other object flicker and moved the camera. Both are fixed. This file records
*why*, because the original diagnosis was wrong in a way that cost a lot of time and would be
easy to repeat.

## The segfault was never the GPU

It was blamed on QtWebEngine: "this app's GL context does not survive any in-place state
change on this Mac; only a full page reload is stable." That sent people chasing ANGLE
backends, driver versions, Mol\* internals and streaming races. All of it was wasted.

The macOS crash reports (`~/Library/Logs/DiagnosticReports/python3.12-*.ips`) put the fault in
the **main app process**, not the WebEngine GPU subprocess, with no graphics code in the stack:

```
0  QTreeWidgetItem::setData          <- EXC_BAD_ACCESS
2  QTreeModel::setData
3  QStyledItemDelegate::editorEvent  <- the checkbox being toggled
7  QAbstractItemView::mouseReleaseEvent
```

Qt emits `itemChanged` **synchronously from inside** `QTreeWidgetItem::setData`, while still
on the tree's mouse-release/edit stack. `_on_tree_item_changed` applied the visibility change
right there, which reached `_on_loaded_changed` -> `QTreeWidget.clear()` and destroyed the very
item whose `setData` was executing. Qt kept using the freed item as the stack unwound.

### The tells that it was not the renderer

- **Three unrelated hide mechanisms** (clip slab, `setSubtreeVisibility`, scene reload) crashed
  with one *byte-identical* backtrace. Three different render paths cannot share one bug.
- It reproduced on **SwiftShader**, a pure-CPU rasterizer, as readily as on the GPU.
- The "stable" reload baseline **crashed too** — a crash report at 01:08 postdates the 00:11
  revert to it. It was never safe; it just crashed less often.
- "Crashes on rapid toggling" is a use-after-free signature: fast clicks maximize the chance
  the rebuild lands inside a live mouse-release stack.

### The fix

Read the plain values off the item, then apply the change on the next event-loop turn
(`QTimer.singleShot(0, ...)`). See `ControlsWindow._on_tree_item_changed` / `_apply_visibility`.
Two sibling paths had the identical defect and got the same treatment: `_on_tree_current_changed`
and `_on_active_radio` (the radio lives *in* the tree via `setItemWidget`, so the rebuild
deleted the button still delivering its own click).

**The rule for this codebase: never rebuild the object tree from inside a signal emitted by
one of its own items or item widgets. Defer it.**

Because a use-after-free only faults when the freed memory is reused, it does not reproduce on
demand — which is exactly why it looked hardware-specific. `MallocScribble=1` makes it
deterministic: pre-fix it SIGSEGVs on the first checkbox click, post-fix 21 clicks pass.

Guard: `test_toggling_a_visibility_box_does_not_rebuild_the_tree_inside_the_signal`.

## The flicker and the camera jump

Both were consequences of the reload workaround, and both went away with in-place hiding
(`git revert` of the revert, restoring `setSubtreeVisibility`):

- **Flicker**: hiding one object reloaded the whole page, so every other object blanked and
  redrew. In-place hiding touches only the toggled object's render state.
- **Camera jump**: a fresh page re-ran the scene's focus, reframing the view on whatever
  object it decided to centre. With no reload there is nothing to re-focus, so the camera
  stays exactly where the user left it.

The revert commit had also claimed `setSubtreeVisibility` "didn't hide" on macOS. That does not
reproduce — it was almost certainly confounded by the crash. Verified on this Mac's GPU with
the app maximized (a real canvas; note that an unsized viewport reports a 0x0 canvas and draws
nothing, which will mislead any measurement): hiding a model drops the visible render objects
from 5 to 3, hiding a map 5 to 4, both restore on show, the camera does not move, no page
reload occurs, the object stays hidden while coordinates stream, and rapid toggling is stable.

## Still open

`_can_hide` refuses hiding on software rendering: the checkboxes are non-checkable and a click
flashes why. That exists only because hiding "segfaulted on software" — which was this same
tree bug, not the renderer, so it is very likely safe to lift. It was left in place because
software rendering has not been re-tested since the fix. Verify, then remove the policy and
its tests (`test_software_pins_a_model_and_says_why_on_click`,
`test_software_pins_a_map_and_says_why_on_click`) together.

## Key code

- `python/pxviewer/desktop.py` — `DesktopApp.set_model_visible` / `set_volume_visible` (in
  place, no reload), `_reassert_hidden_volumes` (re-hides maps after a reload from add/remove),
  `_reload_viewport`, `_write_volume_scene`, `_control_session`. Tree:
  `ControlsWindow._on_loaded_changed` (rebuilds it), `_on_tree_item_changed` (the deferral).
- `python/pxviewer/live.py` — `LiveSession.set_structure_visible` (a model's own visibility,
  replayed to late clients so a reload keeps it hidden), `set_volume_visible` (a map by ref).
- `frontend/src/live.ts` — `LiveViewer.setStructureVisible`, `setVolumeVisible`, both via
  Mol\*'s `setSubtreeVisibility`; `findVolumeReprCell` locates a volume's repr cell by ref.

## Build / run / test

- Env: `conda activate pxviewer` (conda-forge channels only; never accept Anaconda's ToS).
- Python is an editable install, so Python edits are live on restart.
- **Frontend edits require a rebuild**: `cd frontend && npm run build`, then restart the app.
- Run: `python -m pxviewer desktop`.
- Tests: `QT_QPA_PLATFORM=offscreen python -m pytest python/tests/test_desktop.py
  python/tests/test_live.py python/tests/test_gui_fuzz.py -q` — run per-file, the full
  `tests/` run in one process hangs on a pre-existing leaked-thread issue.
- Five tests in `test_desktop.py` fail on a clean checkout for unrelated pre-existing reasons
  (`test_minimize_buttons_show_which_state_is_live`, `test_validation_subtabs_and_row_focus`,
  `test_restraint_row_marks_all_atoms_and_draws_its_notation`,
  `test_atom_precision_actions_switch_a_ribbon_to_ball_and_stick`,
  `test_residue_orientation_and_space_navigation`). Don't blame them on hide work.

## Constraints

- Commits authored as `cschlick <cschlick@users.noreply.github.com>`, no AI attribution.
- Push only when explicitly asked.
