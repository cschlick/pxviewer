"""An MCP server that hands agents the app's own Python kernel.

The semantic surface is one tool — ``python(code)`` — which executes in the
app's embedded IPython kernel: the very namespace the Console tab shows
(``session``, ``app``, ``api``, ``np``, ``help``). The objects, not the
schema, are the API; the kernel is shared state, so names an agent assigns
in one call are there for the next. The remaining tools are operational:
``status`` for scene + kernel state, ``ping``/``reload_viewport`` for
viewport health, ``screenshot``/``save_screenshot`` for seeing the frame.

The server owns a real ``DesktopApp`` — the same windows a user opens — so an
agent's figure work happens in the viewer the user is watching.

Threading: Qt must live on the main thread (macOS requires it), so the MCP
loop runs on a daemon thread. Tool calls that touch widgets are marshalled
onto the Qt thread through :meth:`DesktopApp.invoke_on_gui`; ``LiveSession``
calls (screenshot, camera, background) are already thread-safe and run
directly on the MCP thread; ``python`` code runs on the kernel's thread
with ``app`` proxied back onto Qt.

Run it as ``python -m pxviewer mcp`` and point a client at the stdio
transport, e.g. in an MCP client config::

    {"command": "/path/to/env/bin/python", "args": ["-m", "pxviewer", "mcp"]}
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

_app = None          # DesktopApp, main thread

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

    Requests sent before the viewer's handler exists are dropped, so this is
    short repeated tries rather than one long wait (the wait_viewer pattern).
    On timeout a chain-bypassing ping says whether the page is dead or just
    stuck behind one slow state-tree update.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        session = _app._control_session()
        if session is not None:
            try:
                if session.screenshot(timeout=5):
                    return
            except Exception:
                pass
        time.sleep(0.5)
    raise RuntimeError(_unready_reason())


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
    """Serve the MCP tool surface over HTTP for an already-running DesktopApp.

    Called on the Qt thread during ``DesktopApp.start`` so the invoker's
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
    """Build the desktop app the way run_desktop starts it; return the app."""
    global _app
    from PySide6.QtWidgets import QApplication

    from pxviewer import gpu as gpu_backend

    gpu_mode = gpu_backend.configure(None)
    from pxviewer.desktop import _set_macos_process_name
    _set_macos_process_name()
    qt_app = QApplication(["pxviewer"])
    qt_app.setApplicationName("pxviewer")

    from pxviewer.desktop import DesktopApp

    can_hide = gpu_mode == "hardware"
    app = DesktopApp(port=0, can_hide=can_hide)
    app._webapp.start()
    app._main.show()
    app._controls.widget().show()
    app._main.resize(1400, 900)
    _app = app
    return qt_app, app


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
        "  app      the DesktopApp, proxied so calls run on the Qt thread: "
        "app.load_file(path), app.load_files(paths), "
        "app.fetch_and_load(pdb_id='1ubq', entities=['model','map']), "
        "app.select_by_expression(expr), app.job_state().\n"
        "  api      a categorized map of the API — api.find('color') "
        "locates verbs; api itself lists them.\n"
        "  np       numpy.  help — bare help() prints the console guide; "
        "help(obj) is Python's own help.\n\n"
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
        - ``app``: the DesktopApp through a proxy whose calls run on the Qt
          thread — load_file/load_files/fetch_and_load/select_by_expression/
          run_background/job_state, everything the panels can do.
        - ``api``: categorized map of the API (api.find("color")).
        - ``np``: numpy.  ``help``: the console guide (bare) or help(obj).

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
        png = session.screenshot(timeout=60)
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
        _wait_ready(timeout=15)
        png = _session().screenshot(timeout=30)
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

    qt_app, app = _start_app()
    server = _make_server()

    def serve():
        try:
            server.run()
        except Exception as exc:  # noqa: BLE001 — nothing else reports it
            print("mcp server stopped: %r" % exc, file=sys.stderr)

    threading.Thread(target=serve, name="pxviewer-mcp", daemon=True).start()

    # A client may take a moment to send its first tool call; the viewport
    # page connects as soon as the window is up, which also warms the session
    # so the first screenshot does not race the page load.
    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(run())
