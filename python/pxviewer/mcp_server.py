"""An MCP server that hands agents the app's own Python kernel.

The semantic surface is one tool — ``python(code)`` — which executes in the
app's embedded IPython kernel: the very namespace the Console tab shows
(``session``, ``app``, ``api``, ``help``). The objects, not the
schema, are the API; the kernel is shared state, so names an agent assigns
in one call are there for the next. The remaining tools are operational:
``status`` for scene + kernel state, ``ping``/``reload_viewport`` for
viewport health, ``screenshot``/``save_screenshot`` for seeing the frame.

The server owns a real app — the ``DesktopApp`` with its windows when a
display exists, or the GUI-agnostic ``AppCore`` (headless Chromium hosting the
Mol* page) when it does not — so an agent's figure work happens in the same
viewer code the user sees.

Threading: Qt must live on the main thread (macOS requires it), so the MCP
loop runs on a daemon thread. Tool calls that touch widgets are marshalled
onto the app thread through :meth:`AppCore.invoke_on_gui`; ``LiveSession``
calls (screenshot, camera, background) are already thread-safe and run
directly on the MCP thread; ``python`` code runs on the kernel's thread
with ``app`` proxied back onto Qt.

Run it as ``python -m pxviewer mcp`` and point a client at the stdio
transport, e.g. in an MCP client config::

    {"command": "/path/to/env/bin/python", "args": ["-m", "pxviewer", "mcp"]}
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

_app = None          # AppCore (desktop or headless), main thread

MCP_PORT = 8753      # fixed so MCP client configs can name a stable URL

# The selection grammar below is verified against this build's iotbx selection
# machinery (model.selection() after lazy pdb_interpretation). Constructs that
# fail are listed explicitly — several of them fail *silently* with zero atoms,
# so the list must not overpromise.
SELECTION_LANGUAGE = """\
Selection strings are cctbx/Phenix selections (iotbx/mmtbx selection), used
via app.select_by_expression(expr) or session.model.selection(expr).

Fields: chain <id>, resseq <n> or resseq <n>:<m> (colon ranges), resid <n>[<icode>],
resname <3-letter code>, name <atom name>, element <symbol>, altloc <id>,
occupancy <op> <f> and bfactor <op> <f> where <op> is < <= > >= = !=
Predicates: all, protein (alias peptide), water, hetero, backbone, sidechain,
rna, dna, nucleotide, phosphate, ribose
Spatial: within(<angstroms>, <selection>) selects atoms within the radius;
residues_within(<angstroms>, <selection>) selects whole residues within the radius
Logic: and or not, parentheses. Multiple names/values need 'or'
('name CA or name CB'), never space lists.

Examples:
  chain A and resseq 5:14               first ten residues of chain A
  resname LIG                           the ligand (check codes via the hierarchy)
  residues_within(6, resname HEM)       residues within 6 A of heme
  within(4, chain B)                    atoms within 4 A of chain B
  element ZN                            zinc ions
  chain A and resseq 29 and name CA     C-alpha of a residue
  backbone and resseq 23:34             just the helix backbone
  protein and not water                 polymer atoms only

NOT supported (syntax error or silent no-match):
  dash ranges (resseq 1-10 matches NOTHING — use resseq 1:10),
  comparisons on resseq (resseq > 70 — use a range),
  value lists (resseq 5 7 9, name CA CB — use 'or'),
  wildcards (name 'C*'), xor, aniso, nucleic (use nucleotide).

Translating prose: inspect the hierarchy first — session.model.get_hierarchy(),
api.find('select') — to map requests like "the first helix tip" or "the
ligand" onto real chains/resseqs, then select."""


# Verified against this build's cctbx the same way the selection grammar was:
# every expression below was run. The recurring mistakes this exists to head
# off: selection prose reaching the viewing API instead of the cctbx resolvers,
# flex arrays mistaken for numpy, and set_sites_cart assumed to re-render on
# its own.
CCTBX_IDIOMS = """\
The objects behind session/app are cctbx's own — knowing the data structures
saves a discovery round-trip.

session.model is an mmtbx.model.manager:
  model.get_hierarchy()       iotbx hierarchy: models -> chains ->
                              residue_groups -> atom_groups -> atoms
                              (atom.name/.element/.occ/.b/.xyz/.i_seq)
  model.get_sites_cart()      flex.vec3_double — (x,y,z) Angstrom per atom
  model.crystal_symmetry()    unit cell + space group
  model.get_xray_structure()  xray_structure with .scatterers()
Atom identity is positional: hierarchy i_seq, the wire index session calls
take, and the row index of every flex array are the SAME number.

ONE selection language — the cctbx/Phenix grammar above, always resolved
Python-side; the wire only ever carries integer index sets:
  model.selection(expr)              -> flex.bool of length n_atoms
  app.select_by_expression(expr)     resolve + make it the scene selection
  session.select_by(selection=expr)  -> Selection object (.indices, .mask, ...)
  session.select(sel_or_indices)     viewing ops take a Selection, i_seq index
                                     lists, or a length-N bool mask — never a
                                     string; prose enters via select_by only
  sel.iselection()                   flex.size_t of matching i_seqs
Never invent an atom index — derive it from the hierarchy or a selection.

flex arrays are cctbx's arrays, not numpy:
  v.as_numpy_array()    copy out (vec3_double -> (N,3) float64)
  flex.vec3_double(arr) build from an (N,3) numpy array
  flex.size_t(list)     build an index array from python ints
  import: from scitbx.array_family import flex

Moving atoms does NOT move the view by itself:
  arr = model.get_sites_cart().as_numpy_array()
  arr[sel.iselection().as_numpy_array()] += (1.0, 0.0, 0.0)
  model.set_sites_cart(flex.vec3_double(arr))
  session.push(model.get_sites_cart())   # stream the new conformation
  (pass changed=iselection to stream only the moved atoms)

One hop further:
  app.model_mmm()              mmtbx.map_model_manager (model+map+ncs),
                               or None until a map is paired
  mmm.map_manager().map_data() flex.double 3D grid of density values
  app.map_for_model()          the viewer's map data, or None
  app.reflections_for_model()  the reflections the model was phased
                               against — entry['data'].arrays is the list
                               of cctbx miller_array objects — or None

Long cctbx calls (minimize, real-space refine, map recompute) block the
kernel thread — prefer app.run_background(fn, name=..., label=...) and
watch app.job_state() / status rather than exec'ing them directly."""


async def _qt(fn):
    """Run ``fn`` on the Qt thread and await its result."""
    return await asyncio.wrap_future(_app.invoke_on_gui(fn))


def _session():
    session = _app._control_session()
    if session is None:
        raise RuntimeError("viewport not connected yet — try again in a moment")
    return session


def _wait_ready(timeout: float = 60.0) -> None:
    """Block until the Mol* page answers a screenshot — the signal it is live.

    Wait for the session, then ask once with the whole remaining budget: the
    first frame under software WebGL compiles shaders and can take tens of
    seconds. Retrying would only stack more screenshot renders on the same
    serial apply chain — each abandoned call still runs to completion in the
    page, holding every later request. On timeout a chain-bypassing ping says
    whether the page is dead or just stuck behind one slow state-tree update.
    """
    deadline = time.time() + timeout
    session = None
    while time.time() < deadline:
        session = _app._control_session()
        # A session with no client connected yet is a page still navigating —
        # keep polling; screenshotting now would burn the budget pre-connect.
        if session is not None and session.client_count >= 1:
            break
        time.sleep(0.5)
    if session is not None and session.client_count >= 1:
        try:
            if session.screenshot(timeout=max(1.0, deadline - time.time())):
                return
        except Exception:
            pass
    raise RuntimeError(_unready_reason())


def _blank_frame(png: bytes) -> bool:
    """A uniform-color frame is the page's "nothing drawn yet" — retryable.

    The ws chain serializes message application, but the Mol* representation
    build for a fresh structure is async in-page and not part of it: a shot can
    answer after every queued message and still catch the pre-draw canvas. An
    all-one-color picture is that blank — a rendered scene is never uniform.
    """
    try:
        import io
        from PIL import Image, ImageStat
        small = Image.open(io.BytesIO(png)).convert("L").resize((64, 48))
        return ImageStat.Stat(small).stddev[0] < 0.5
    except Exception:
        return False  # cannot judge — do not claim blankness


def _shot_frame(session, *, timeout: float, retries: int = 6,
                settle: float = 1.5) -> "bytes | None":
    """One screenshot, retried while it comes back a blank canvas."""
    png = None
    for _ in range(max(1, retries)):
        png = session.screenshot(timeout=timeout)
        if not png or not _blank_frame(png):
            return png
        time.sleep(settle)
    return png


def _unready_reason() -> str:
    """Why the viewport is not answering — a stalled apply chain reports its
    blocker; silence means the page itself is gone."""
    session = _app._control_session()
    if session is not None:
        try:
            pong = session.ping(timeout=2)
        except Exception:
            pong = None
        if pong is not None:
            busy = pong.get("busy")
            if busy:
                return (
                    f"viewport alive but its apply chain is held by {busy} "
                    f"({pong.get('busyMs')}ms, {pong.get('pending')} queued)"
                )
            return "viewport alive but screenshots time out — render loop stalled?"
    return "viewport never came up"


def attach(app, port: int = MCP_PORT) -> "str | None":
    """Serve the MCP tool surface over HTTP for an already-running app.

    Called on the app thread during ``AppCore.start`` so the invoker's
    signals land on the right loop. Returns the URL clients point at, or None
    if the ``mcp`` package is absent. If the port is already held (a second
    pxviewer instance), the serve thread retries for a few minutes so the
    endpoint comes up when the holder exits.
    """
    global _app
    try:
        import mcp  # noqa: F401
    except ImportError:
        return None

    _app = app
    server = _make_server()
    server.settings.host = "127.0.0.1"
    server.settings.port = port
    # python(code) is arbitrary execution, so the endpoint must not be
    # reachable from a drive-by webpage either: enforce DNS-rebinding checks
    # (Host pinned to loopback) and refuse any request carrying an Origin
    # header — real MCP clients don't send one, browsers always do.
    server.settings.transport_security.enable_dns_rebinding_protection = True
    server.settings.transport_security.allowed_hosts = [
        "127.0.0.1:%d" % port, "localhost:%d" % port]
    server.settings.transport_security.allowed_origins = []

    def serve():
        # A second instance loses the port race — keep retrying briefly so it
        # picks the endpoint up once the holder exits, rather than never
        # having MCP at all. Bounded so a permanently busy port doesn't spam.
        import errno  # noqa: F401
        deadline = time.time() + 600
        noted_busy = False
        while True:
            try:
                server.run(transport="streamable-http")
                return  # clean stop
            except OSError:
                if time.time() > deadline:
                    print("pxviewer: MCP port still busy — no endpoint",
                          file=sys.stderr)
                    return
                if not noted_busy:
                    print("pxviewer: MCP port busy — retrying while the holder "
                          "is up", file=sys.stderr)
                    noted_busy = True
                time.sleep(10)
            except Exception as exc:  # noqa: BLE001 — nothing else reports it
                print("pxviewer: MCP server stopped: %r" % exc, file=sys.stderr)
                return

    threading.Thread(target=serve, name="pxviewer-mcp", daemon=True).start()
    url = f"http://127.0.0.1:{port}/mcp"
    print(f"pxviewer: MCP tools at {url}  (PXVIEWER_MCP=0 to disable)", flush=True)
    return url


def _start_app():
    """Build the app and return it.

    With a display this is the real ``DesktopApp`` — an
    agent's figure work happens in the window the user is watching. With no
    display — or ``PXVIEWER_HEADLESS=1`` — it is the GUI-agnostic ``AppCore``:
    every semantic operation (load, fetch, select, validate, kernel, MCP)
    runs, and the Mol* page is hosted by a headless Chromium when playwright
    is installed (screenshots work; without it, everything but pixels does).
    """
    global _app
    import os

    from pxviewer import gpu as gpu_backend

    # A *real* display gets the desktop (the agent's work is watchable); no
    # display gets the GUI-agnostic core — nobody watches a spawned Xvfb, so
    # don't spin one up: skip QtWidgets entirely and host the Mol* page in
    # headless Chromium. PXVIEWER_HEADLESS forces the core path.
    headless_env = os.environ.get("PXVIEWER_HEADLESS", "").lower() in (
        "1", "true", "yes")
    # DISPLAY/WAYLAND_DISPLAY are X11/Wayland notions — macOS and Windows
    # always have a windowing system.
    have_display = sys.platform in ("darwin", "win32") or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if have_display and not headless_env:
        gpu_mode = gpu_backend.configure(None)
        from PySide6.QtWidgets import QApplication

        from pxviewer.desktop import _set_macos_process_name

        _set_macos_process_name()
        qt_app = QApplication(["pxviewer"])
        qt_app.setApplicationName("pxviewer")

        from pxviewer.desktop import DesktopApp

        app = DesktopApp(port=0, can_hide=(gpu_mode == "hardware"))
        app._webapp.start()
        app._main.show()
        app._controls.widget().show()
        app._main.resize(1400, 900)
        app._reload_viewport()  # load the Mol* page into the QtWebEngine view
    else:
        from pxviewer.appcore import AppCore
        from pxviewer.browser import try_browser_viewport

        viewport = try_browser_viewport()
        if viewport is not None:
            print("pxviewer: headless viewport via Chromium", flush=True)
        app = AppCore(port=0, viewport=viewport)
        app._webapp.start()
        app._reload_viewport()  # navigates the browser host, if one attached
    _app = app
    return app._app, app


def _make_server():
    # FastMCP resolves annotations with eval_str against the module globals, so
    # Image must be visible there even though mcp is imported lazily.
    global Image
    from mcp.server.fastmcp import FastMCP, Image

    mcp = FastMCP("pxviewer", instructions=(
        "pxviewer is a macromolecular viewer; you are driving a real, visible "
        "instance the user is watching — everything on screen is what the "
        "user sees, and selections you make are the app's selections.\n\n"
        "The surface is a live kernel, not a schema: python(code) executes "
        "in the app's own IPython kernel — the same namespace the Console "
        "tab shows — so the objects are the API:\n\n"
        "  session  the active model's LiveSession: selections, "
        "representations, camera, measurements, primitives. "
        "session.model is the mmtbx model — get_sites_cart() returns a "
        "flex.vec3_double, get_hierarchy() the full hierarchy.\n"
        "  app      the AppCore (desktop or headless), proxied so calls run "
        "on the app thread: app.load_file(path), app.load_files(paths), "
        "app.fetch_and_load(pdb_id='1ubq', entities=['model','map']), "
        "app.select_by_expression(expr), app.job_state().\n"
        "  api      a categorized map of the API — api.find('color') "
        "locates verbs; api itself lists them.\n"
        "  help     bare help() prints the console guide; "
        "help(obj) is Python's own help.\n\n"
        "Capabilities — these exist, use them rather than working "
        "around their absence:\n"
        "  Regions & clipping: every model and map has a live clip "
        "sphere (app.set_model_sphere, app.set_volume_sphere) and slab "
        "clip (app.set_model_clip, app.set_volume_clip) — shader-side, "
        "instant, reversible. For a huge map, app.set_volume_box carves "
        "the served grid before isosurface extraction and "
        "app.set_volume_downsample thins it — never make clipped or "
        "cropped copies of map files to show a region.\n"
        "  Selections are literal: app.select_by_expression / "
        "session.select_by make real selections, then focus, clip, "
        "highlight, measure, or give them their own representation "
        "layer (session.add_representation(on=sel)).\n"
        "  Representations compose: set_representation replaces the "
        "stack, add_representation layers, color_by resets to one rep.\n"
        "  Maps: level in sigma, style surface|mesh, color, opacity, "
        "visibility; difference maps carry a negative contour with its "
        "own controls; app.set_volume_mask cuts a map to a model's "
        "footprint.\n"
        "  Coordinates: edit session.model then session.push(); "
        "app.minimize_model refines; the tug tools pull atoms into "
        "density.\n"
        "  Analysis: measurements, clashes, probe dots, "
        "app.run_validation, real-space difference and resolution maps, "
        "hotspot fields.\n"
        "  Output & jobs: screenshot/save_screenshot for eyes, "
        "app.write_object saves an object to disk, app.run_background + "
        "app.job_state() for long work.\n\n"
        "Names you assign persist between calls — stash intermediate "
        "objects in the namespace rather than re-deriving them. Results "
        "come back as {stdout, stderr, result, data, error}: result is the "
        "repr of the last expression, data is its full mimebundle (an "
        "IPython Image lands as image/png), error carries the traceback.\n\n"
        "Discovery: status() first for loaded objects and running jobs; "
        "then api.find(...), dir(session), help(session.method) to map the "
        "verbs. cctbx objects (session.model, app.model_mmm(), "
        "app.map_for_model(), app.reflections_for_model()) are the real "
        "library objects — the whole cctbx API is available.\n\n"
        "Long work: exec blocks the kernel — give minimize/validation "
        "calls room with the timeout arg, or kick them off via "
        "app.run_background and watch job_state().\n\n"
        "Diagnostics and recovery: ping() answers whether the Mol* page is "
        "alive and, if merely busy, what is holding its apply chain; "
        "screenshot() returns the frame so you can see the scene; "
        "save_screenshot(path) writes a PNG; reload_viewport() rebuilds a "
        "dead page (Python-side state survives).\n\n"
        + SELECTION_LANGUAGE + "\n\n" + CCTBX_IDIOMS))

    @mcp.tool()
    async def python(code: str, timeout: float = 120.0) -> dict:
        """Execute ``code`` in the app's own IPython kernel — the same
        namespace the Console tab shows. The objects are the API:

        - ``session``: the active model's LiveSession (selection,
          representations, camera, measurements); re-binds on model switch.
        - ``app``: the AppCore through a proxy whose calls run on the app
          thread — load_file/load_files/fetch_and_load/select_by_expression/
          run_background/job_state, everything the panels can do.
        - ``api``: categorized map of both objects (``api.find("clip")``,
          ``api.find("map")``) — and ``dir(app)``/``dir(session)`` list
          every verb directly.
        - ``help``: the console guide (bare) or help(obj).

        Scenes compose — representations are a stack, not a mode:
        ``session.set_representation("cartoon")`` replaces the whole stack;
        ``session.add_representation("ball_and_stick", on=sel)`` layers a
        subset on top, where ``sel`` is anything ``session.select_by(...)``
        returns (e.g. ``session.select_by(selection="resseq 70:76")``).
        ``session.color_by(...)`` resets to one rep — re-add layers after it.

        Variables persist between calls — the kernel is shared state.
        Returns {status, stdout, stderr, result, data, error}: ``result``
        is the repr of the last expression; ``data`` is its mimebundle (an
        IPython Image shows up as image/png); ``error`` carries ename/
        evalue/traceback, or a Timeout entry if ``timeout`` expired (the
        kernel keeps running the code — call again with a longer timeout).
        """
        kernel = await _qt(_app.ensure_kernel)
        if kernel is None:
            raise RuntimeError(
                "the 'console' extra (ipykernel) is not installed — "
                "pip install 'pxviewer[console]'")
        return await asyncio.to_thread(kernel.execute, code, float(timeout))

    @mcp.tool()
    async def status() -> dict:
        """Scene state: loaded objects, selections, camera, markers, the
        kernel's connection file, plus `busy`/`job_errors` — running and
        recently-failed background jobs (fetch, validation, minimize). Check
        `job_errors` when an exec's effect never appears. `camera`/
        `background` are null until a viewer session exists; the scene lists
        answer regardless."""
        def work():
            return {
                "models": [{"id": e["id"], "name": e.get("name"),
                            "rep": e.get("rep"), "color": e.get("color"),
                            "active": e["id"] == _app._active_model_id}
                           for e in _app._models],
                "volumes": [{"id": e["id"], "name": e.get("name"),
                             "iso": e.get("iso"), "opacity": e.get("opacity")}
                            for e in _app._volumes],
                "selection": {mid: len(ix) for mid, ix in
                              _app._scene_selection.items()},
                "markers": [{"id": m["id"], "position": m["position"],
                             "visible": m["visible"]} for m in _app._markers],
                "reflections": [{"id": r["id"], "name": r.get("name")}
                                for r in _app._reflections],
            }
        state = await _qt(work)
        jobs = _app.job_state()
        state["busy"] = jobs["busy"]
        state["job_errors"] = jobs["errors"]
        kernel = _app._kernel
        state["kernel"] = ({"connection_file": kernel.connection_file}
                           if kernel is not None else None)
        session = _app._control_session()
        state["camera"] = session.camera_state() if session else None
        state["background"] = session.background_color() if session else None
        return state

    @mcp.tool()
    async def ping() -> dict:
        """Liveness and health of the Mol* page, bypassing the apply chain.

        Reports whether a viewer client is connected at all; when it is, the
        page answers immediately with `pending` (messages queued behind
        scene work) and `busy`/`busyMs` (the state-tree update currently
        holding the chain, and for how long). A stalled chain makes every
        other call look dead — this tells "dead" from "busy", and names the
        culprit. `lastStall` is the most recent stall report the page sent."""
        session = _app._control_session()
        if session is None:
            return {"alive": False, "clients": 0, "detail": "no viewer session"}
        if session.client_count < 1:
            return {"alive": False, "clients": 0,
                    "detail": "session up but no page connected"}
        pong = session.ping(timeout=3)
        if pong is None:
            return {"alive": False, "clients": session.client_count,
                    "detail": "page connected but not answering — render process dead?"}
        return {"alive": True, "clients": session.client_count, **pong,
                "lastStall": session._last_chain_stall}

    @mcp.tool()
    async def reload_viewport() -> dict:
        """Reload the Mol* page — the recovery for a wedged or crashed viewer.

        Python-side state (models, selections, representations, clips,
        primitives) survives; the page rebuilds from it on reconnect. Use
        when ping() reports the page is dead, or when calls fail with
        'viewport never came up' and ping does not answer."""
        await _qt(lambda: _app._reload_viewport())
        _wait_ready(timeout=60)
        return {"reloaded": True, "camera": _session().camera_state()}

    @mcp.tool()
    async def save_screenshot(path: str) -> dict:
        """Write the current viewport frame to a PNG file. Returns only after
        the bytes are on disk: the page's screenshot answer is queued behind
        its scene work, so the image shows the state the earlier calls
        requested — never a pre-update frame, and 'saved' is never claimed
        for a write that did not happen."""
        session = _session()
        png = _shot_frame(session, timeout=60)
        if not png:
            raise RuntimeError(
                "the viewport did not answer — check ping() for a stuck "
                "apply chain, or reload_viewport()")
        out = Path(path).expanduser()
        out.write_bytes(png)
        return {"saved": str(out), "bytes": out.stat().st_size}

    @mcp.tool()
    async def screenshot() -> Image:
        """Capture the current Mol* viewport frame and return it as an image.
        Call this after making changes to *see* the scene — it is the way to
        check that a figure looks right before saving the view."""
        _wait_ready(timeout=45)  # first SwiftShader frame can take tens of seconds
        png = _shot_frame(_session(), timeout=60)
        if not png:
            raise RuntimeError("screenshot came back empty")
        return Image(data=png, format="png")

    return mcp


def run() -> int:
    """Start the app on the main thread, the MCP stdio loop on a daemon
    thread, and run the Qt event loop until the app exits."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("pxviewer mcp needs the 'mcp' package: pip install 'mcp>=1.10,<2'",
              file=sys.stderr)
        return 1

    # stdout is the JSON-RPC channel in stdio mode — nothing else may write
    # there. Park the real channel on a saved fd for the protocol writer, then
    # point fd 1 at stderr so prints (kernel banner, Qt warnings) stay visible
    # but can never corrupt the stream. Done before app start so startup
    # chatter is covered too.
    import anyio
    from mcp.server.stdio import stdio_server
    out_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    qt_app, app = _start_app()
    server = _make_server()

    async def serve_stdio():
        async with stdio_server(stdout=anyio.wrap_file(os.fdopen(out_fd, "w"))) \
                as (read_stream, write_stream):
            await server._mcp_server.run(
                read_stream, write_stream,
                server._mcp_server.create_initialization_options())

    def serve():
        try:
            anyio.run(serve_stdio)
        except Exception as exc:  # noqa: BLE001 — nothing else reports it
            print("mcp server stopped: %r" % exc, file=sys.stderr)
        finally:
            # stdin EOF (the client left) ends the protocol loop — without
            # this the Qt loop would run forever, orphaned.
            app.invoke_on_gui(app._app.quit)

    threading.Thread(target=serve, name="pxviewer-mcp", daemon=True).start()

    # A client may take a moment to send its first tool call; the viewport
    # page connects as soon as the window is up, which also warms the session
    # so the first screenshot does not race the page load.
    return app._app.exec()


if __name__ == "__main__":
    sys.exit(run())
