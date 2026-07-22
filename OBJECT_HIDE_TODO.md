# Object hide: make it smooth, or fix the macOS QtWebEngine segfault

This is a self-contained brief for whoever picks up the smooth-hide work. It assumes no
knowledge of prior conversations.

## Context

pxviewer is a desktop molecular-structure viewer: a PySide6/Qt app whose viewport is a
**QtWebEngine** (Chromium) view running a **Mol\*** (molstar) TypeScript frontend. The Python
backend streams data to the frontend over WebSockets. The crash below reproduces only on macOS
with a real GPU (hardware WebGL); on software rendering (SwiftShader) hiding is refused entirely,
so you must debug on the Mac.

## The bug to fix

Hiding a loaded object (a model or a map) from the object-list checkboxes must be smooth. Right
now it works but **flickers**: hiding one object reloads the entire viewport page, so every
other object briefly blanks and redraws. The goal is in-place hiding (toggle one object's
visibility without disturbing the others) — but every in-place attempt so far **segfaults the
whole app on this Mac's GPU**.

## What's been tried and FAILED (both segfault on the real macOS GPU)

1. **Clip slab**: hiding a model by sending it a full-scene clip plane (`set_clip(1,1)`, a
   closed slab) so nothing draws. → segfault on hide.
2. **`setSubtreeVisibility`** (Mol\*'s own standard hide, from
   `molstar/lib/mol-plugin/behavior/static/state`): called on the model's structure state ref,
   or on a volume's isosurface repr cell. → on macOS it (a) did **not** visually hide, and
   (b) **segfaulted** on rapid on/off toggling. Headless on Linux/SwiftShader it DID set
   `cell.state.isHidden = true` with no reload and no error — so the Python→frontend plumbing is
   correct; the failure is specific to the macOS QtWebEngine GL context.

Two independent in-place mechanisms both crashing strongly suggests **this app's QtWebEngine GL
context does not survive any in-place GPU state change on this Mac** — only a full page reload
(a clean context teardown) is stable. That may be abnormal (a healthy WebGL context toggles
visibility fine), so it's likely a **version / driver issue** with the conda `qt6-webengine` +
Mol\* + macOS-GPU combination, OR a race between the app's continuous coordinate streaming and
the visibility change.

## Current stable baseline (do not regress this)

Hiding is currently **reload-based** and stable: `DesktopApp.set_model_visible` /
`set_volume_visible` set the entry's `visible` flag and call `_reload_viewport()`, which
recomposes the scene URL from only the visible objects and reloads the QtWebEngine page. It
flickers but never crashes. **Keep the app crash-free at all times**; if in-place proves
impossible, leaving the reload behavior in place is an acceptable outcome.

## Goal

Diagnose *why* an in-place visibility change segfaults on this Mac, and if fixable, implement
smooth in-place hiding (hide/show one object with no page reload, no flicker).

## Key code

- `python/pxviewer/desktop.py` — `DesktopApp`. Hide entry points: `set_model_visible`,
  `set_volume_visible`. Scene composition: `_reload_viewport`, `_model_ws`,
  `_write_volume_scene`, `_control_session`. Object lists: `self._models`, `self._volumes`
  (dicts with `"visible"`, `"session"`, `"ref"`). `_can_hide` gates hiding (True on hardware;
  hiding is refused on software rendering — the checkboxes are non-checkable there).
- `python/pxviewer/live.py` — `LiveSession` (one per model, its own WebSocket). Sends JSON text
  control messages (`{"type": ...}`) and tagged binary frames; replays state to late clients in
  `_handler`. `set_clip` is a real feature (front/rear slab) — the failed hide overloaded it.
- `frontend/src/live.ts` — the Mol\* integration. `connectLive(plugin, url)` runs one per model
  in ONE shared plugin. `class LiveViewer` holds `this.structure` (a
  `StateObjectSelector<Structure>`). `handleControlMessage` dispatches text messages. Volumes
  load from an MVSJ scene (`?mvsj=...`); `findVolumeReprCell(plugin, ref)` locates a volume's
  repr cell by ref.

## Debugging leads (rough priority)

1. **Get the actual crash cause.** Run with Chromium logging and reproduce a hide-toggle crash;
   the last GL/Chromium lines before the segfault usually name the failing call:
   `QTWEBENGINE_CHROMIUM_FLAGS="--enable-logging=stderr --v=1" python -m pxviewer desktop`
   Also read the macOS crash report in `~/Library/Logs/DiagnosticReports/` — is it the app
   process or the QtWebEngine **GPU/render subprocess** that dies?
2. **Version check** — a mismatched/old WebEngine is the prime suspect:
   `conda list | grep -iE "qt6|webengine|pyside|chromium"` and the Mol\* version in
   `frontend/package.json` / `frontend/node_modules/molstar/package.json`.
3. **Isolate pxviewer vs. the stack.** Build a minimal standalone Mol\* HTML page that loads a
   structure and toggles `setSubtreeVisibility` on a timer, open it in a bare QtWebEngine view
   (or Safari/Chrome), and see if the visibility toggle alone crashes. If it crashes in a bare
   QtWebEngine but not in Chrome, it's the QtWebEngine build.
4. **Streaming race.** pxviewer pushes coordinate frames continuously (the trajectory
   version-bumps and re-renders). A visibility change concurrent with a frame re-render could
   corrupt state. Try hiding a **static** model (no active minimization/drag) vs. a streaming
   one; try pausing the stream around the toggle; try debouncing / `requestAnimationFrame`
   around the visibility change.
5. **Different Mol\* API.** Instead of raw `setSubtreeVisibility`, try
   `PluginCommands.State.ToggleVisibility.dispatch(plugin, { state: plugin.state.data, ref })`,
   and ensure a redraw (`plugin.canvas3d?.requestDraw(true)`) — the "didn't visually hide"
   symptom suggests the render didn't sync.
6. **GPU-process resilience.** Test whether specific `QTWEBENGINE_CHROMIUM_FLAGS`
   (e.g. `--disable-gpu-sandbox`, or a different ANGLE backend `--use-angle=metal` /
   `--use-angle=gl`) change the crash — that would point squarely at the WebEngine GPU layer.

## Build / run / test

- Env: `conda activate pxviewer` (conda-forge channels only; never accept Anaconda's ToS).
- Python is an editable install, so Python edits are live on restart.
- **Frontend edits require a rebuild**: `cd frontend && npm run build` (writes `build/index.js`,
  which is gitignored), then restart the app.
- Run: `python -m pxviewer desktop` (or `pxviewer desktop`).
- Tests (Linux/headless): `QT_QPA_PLATFORM=offscreen python -m pytest
  python/tests/test_desktop.py python/tests/test_live.py python/tests/test_gui_fuzz.py -q`.
  Hide behavior is covered by `test_hiding_a_model_reloads_the_scene_without_it`,
  `test_hiding_a_map_reloads_the_scene_without_it`,
  `test_software_pins_a_model_and_says_why_on_click`, and the `_control_session_is_reachable`
  invariant in `python/tests/gui_invariants.py`. If you change the hide mechanism, update these;
  keep the software "no hiding" policy intact. (The full `tests/` run in one process hangs on a
  pre-existing leaked-thread issue — run per-file.)

## Constraints

- Commits authored as `cschlick <cschlick@users.noreply.github.com>`, no AI attribution.
- Push only when explicitly asked.
- Above all: **do not leave the app in a state that segfaults.** If in-place can't be made safe,
  keep the stable reload behavior.
