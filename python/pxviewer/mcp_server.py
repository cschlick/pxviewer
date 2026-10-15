"""An MCP server that drives a live pxviewer session.

The server owns a real ``DesktopApp`` — the same windows a user opens — so an
agent's figure work happens in the viewer the user is watching, and everything
the desktop can do (models, maps, selections, camera, saved views) is
available to tools verbatim.

Threading: Qt must live on the main thread (macOS requires it), so the MCP
stdio loop runs on a daemon thread. Tool calls that touch widgets are
marshalled onto the Qt thread through a queued signal; ``LiveSession`` calls
(screenshot, camera, background) are already thread-safe — they block on the
browser round trip while the Qt loop delivers the answer — and run directly
on the MCP thread.

Run it as ``python -m pxviewer mcp`` and point a client at the stdio
transport, e.g. in an MCP client config::

    {"command": "/path/to/env/bin/python", "args": ["-m", "pxviewer", "mcp"]}
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

_app = None          # DesktopApp, main thread
_invoker = None      # _Invoker, main thread

MCP_PORT = 8753      # fixed so MCP client configs can name a stable URL

# The selection grammar below is verified against this build's iotbx selection
# machinery (model.selection() after lazy pdb_interpretation). Constructs that
# fail are listed explicitly — several of them fail *silently* with zero atoms,
# so the list must not overpromise.
SELECTION_LANGUAGE = """\
Selection strings are cctbx/Phenix selections (iotbx/mmtbx selection).

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
  resname LIG                           the ligand (check codes via structure_info)
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

Translating prose: call structure_info() or secondary_structure() FIRST to map
requests like "the first helix tip" or "the ligand" onto real chains/resseqs,
then select(). select() focuses the camera on the match by default."""


def _model_session(name):
    """Resolve a model name (or active/first) to (id, LiveSession)."""
    mid = _model_id(name)
    entry = _app._model_entry(mid) if mid else None
    session = entry.get("session") if entry else None
    if session is None:
        raise ValueError("no such model: %r" % name)
    return mid, session


def _make_invoker():
    from PySide6.QtCore import QObject, Signal

    class _Invoker(QObject):
        run_job = Signal(object)

        def __init__(self):
            super().__init__()
            self.run_job.connect(self._run)

        def _run(self, job):
            fn, fut = job
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as exc:
                    fut.set_exception(exc)

    return _Invoker()


def _invoke(fn):
    """Run ``fn`` on the Qt thread, return a Future for its result."""
    fut = Future()
    _invoker.run_job.emit((fn, fut))
    return fut


async def _qt(fn):
    """awaitable _invoke for use inside async tool handlers."""
    return await asyncio.wrap_future(_invoke(fn))


def _session():
    session = _app._control_session()
    if session is None:
        raise RuntimeError("viewport not connected yet — try again in a moment")
    return session


def _wait_ready(timeout: float = 60.0) -> None:
    """Block until the Mol* page answers a screenshot — the signal it is live.

    Requests sent before the viewer's handler exists are dropped, so this is
    short repeated tries rather than one long wait (the wait_viewer pattern).
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
    raise RuntimeError("viewport never came up")


def _model_id(name):
    """Resolve a model by name (or the active/first model when name is None)."""
    if name is None:
        return _app._active_model_id or (_app._models[0]["id"] if _app._models else None)
    for entry in _app._models:
        if entry["id"] == name or entry.get("name") == name:
            return entry["id"]
    return None


def _volume_id(name):
    if name is None:
        return _app._volumes[0]["id"] if _app._volumes else None
    for entry in _app._volumes:
        if entry["id"] == name or entry.get("name") == name:
            return entry["id"]
    return None


def attach(app, port: int = MCP_PORT) -> "str | None":
    """Serve the MCP tool surface over HTTP for an already-running DesktopApp.

    Called on the Qt thread during ``DesktopApp.start`` so the invoker's
    signals land on the right loop. Returns the URL clients point at, or None
    if the ``mcp`` package is absent. If the port is already held (a second
    pxviewer instance), the serve thread retries for a few minutes so the
    endpoint comes up when the holder exits.
    """
    global _app, _invoker
    try:
        import mcp  # noqa: F401
    except ImportError:
        return None

    _app = app
    _invoker = _make_invoker()
    server = _make_server()
    server.settings.host = "127.0.0.1"
    server.settings.port = port

    def serve():
        # A second instance loses the port race — keep retrying briefly so it
        # picks the endpoint up once the holder exits, rather than never
        # having MCP at all. Bounded so a permanently busy port doesn't spam.
        import errno
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
    global _app, _invoker
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
    _invoker = _make_invoker()
    return qt_app, app


def _make_server():
    # FastMCP resolves annotations with eval_str against the module globals, so
    # Image must be visible there even though mcp is imported lazily.
    global Image
    from mcp.server.fastmcp import FastMCP, Image

    mcp = FastMCP("pxviewer", instructions=(
        "pxviewer is a macromolecular viewer; these tools drive a real, "
        "visible instance the user is watching. Typical figure workflow: "
        "load() -> select()/set_representation()/set_volume() -> "
        "reset_view()/set_camera() -> screenshot() to inspect the frame -> "
        "save_view() for a reproducible .pxview.json.\n\n"
        + SELECTION_LANGUAGE))

    @mcp.tool()
    async def status() -> dict:
        """Scene state: loaded objects, selections, camera."""
        def work():
            return {
                "models": [{"id": e["id"], "name": e.get("name"),
                            "rep": e.get("rep"), "color": e.get("color")}
                           for e in _app._models],
                "volumes": [{"id": e["id"], "name": e.get("name"),
                             "iso": e.get("iso"), "opacity": e.get("opacity")}
                            for e in _app._volumes],
                "selection": {mid: len(ix) for mid, ix in
                              _app._scene_selection.items()},
            }
        state = await _qt(work)
        state["camera"] = _session().camera_state()
        state["background"] = _session().background_color()
        return state

    @mcp.tool()
    async def load(path: str) -> dict:
        """Load a structure (PDB/mmCIF) or map (MRC/CCP4) file into the scene."""
        def work():
            return _app.load_file(str(Path(path).expanduser()))
        result = await _qt(work)
        _wait_ready()
        return {"loaded": result}

    @mcp.tool()
    async def load_view(path: str) -> dict:
        """Restore a saved .pxview.json — objects, representations, camera."""
        from pxviewer import viewstate
        await _qt(lambda: viewstate.load_view(
            _app, str(Path(path).expanduser())))
        _wait_ready()
        return {"loaded_view": path}

    @mcp.tool()
    async def save_view(path: str, bundle_data: bool = True) -> dict:
        """Save the current scene as a .pxview.json (with bundled data by
        default, so the view reproduces anywhere)."""
        from pxviewer import viewstate
        doc = await _qt(lambda: viewstate.save_view(
            _app, str(Path(path).expanduser()), bundle_data=bundle_data))
        return {"saved": path, "objects": len(doc.get("objects", []))}

    @mcp.tool()
    async def structure_info(model: str = None) -> dict:
        """Structural inventory of a model: chains with their residue ranges,
        polymer residue counts, water count, and non-polymer (ligand/ion)
        residues. Call this to ground prose requests like 'the ligand' or
        'the C-terminus of chain B' into concrete chains and resseqs."""
        def work():
            mid, session = _model_session(model)
            model_obj = getattr(session, "model", None)
            if model_obj is None:
                return {"model": mid, "atoms": session._data.n_atoms,
                        "detail": "no coordinate model — chains unavailable"}
            from iotbx.pdb import common_residue_names_get_class as get_class
            polymer = {"common_amino_acid", "modified_amino_acid",
                       "common_rna_dna", "modified_rna_dna"}
            chains = {}
            waters = 0
            hetero = []
            for chain in model_obj.get_hierarchy().chains():
                cid = chain.id.strip()
                info = chains.setdefault(cid, {"residues": 0, "seqs": []})
                for rg in chain.residue_groups():
                    resname = rg.atom_groups()[0].resname.strip()
                    seq = rg.resseq_as_int()
                    kind = get_class(resname)
                    if kind in polymer:
                        info["residues"] += 1
                        info["seqs"].append(seq)
                    elif kind == "common_water":
                        waters += 1
                    else:
                        hetero.append({"chain": cid, "resname": resname,
                                       "resseq": seq})
            chain_rows = [
                {"chain": cid, "residues": info["residues"],
                 "first": min(info["seqs"]) if info["seqs"] else None,
                 "last": max(info["seqs"]) if info["seqs"] else None}
                for cid, info in chains.items()]
            return {"model": mid, "atoms": session._data.n_atoms,
                    "chains": chain_rows, "waters": waters, "hetero": hetero}
        return {"info": await _qt(work)}

    @mcp.tool()
    async def secondary_structure(model: str = None) -> dict:
        """Helix and strand segments per chain, in file order, as
        {chain, start, end} resseq ranges — the grounding for 'the first
        alpha helix' (helices[0]; its N-terminal tip is 'start') or 'the
        sheet before the active site'. Empty when the model carries no
        secondary-structure annotation."""
        def work():
            mid, session = _model_session(model)
            helices = []
            strands = []
            for chain, beg, end, kind in session._data.secondary_structure:
                row = {"chain": chain, "start": beg, "end": end}
                (helices if kind == "helix" else strands).append(row)
            return {"model": mid, "helices": helices, "strands": strands}
        return await _qt(work)

    @mcp.tool()
    async def select(expression: str, focus: bool = True) -> dict:
        """Apply a cctbx/Phenix selection string (see the server instructions
        for the grammar) and, by default, aim the camera at the match.
        Returns the atom count plus the matched residues so you can verify
        the selection caught what the prose intended."""
        def work():
            controls = _app._controls
            controls._select_expr.setText(expression)
            try:
                n = _app.select_by_expression(expression, focus=focus)
            except Exception as exc:
                from pxviewer.desktop import _accent
                controls._selection_label.setText(
                    f"<span style='color:{_accent(_app._main, 'error')}'>"
                    f"{exc}</span>")
                raise
            controls._select_expr_model = (
                _app._active_model_id if expression.strip() else None)
            controls._selection_label.setText(
                "selection cleared" if not expression.strip()
                else f"{n} atom(s) selected")
            residues = []
            seen = set()
            mid = _app._active_model_id
            entry = _app._model_entry(mid) if mid else None
            session = entry.get("session") if entry else None
            if session is not None and mid in _app._scene_selection:
                arrays = session._data.arrays
                for i in _app._scene_selection[mid]:
                    key = (arrays.chain[i], int(arrays.resseq[i]),
                           arrays.resname[i])
                    if key not in seen:
                        seen.add(key)
                        residues.append(
                            f"{key[0]}/{key[2]}{key[1]}" if key[0]
                            else f"{key[2]}{key[1]}")
            return n, residues
        n, residues = await _qt(work)
        return {"atoms": n, "residues": residues[:40],
                "truncated": len(residues) > 40, "focus": focus}

    @mcp.tool()
    async def clear_selection() -> dict:
        """Clear the standing selection."""
        await _qt(lambda: _app.clear_selection())
        return {"selected": {}}

    @mcp.tool()
    async def set_representation(rep: str, model: str = None,
                                 color: str = None) -> dict:
        """Set a model's representation ('cartoon', 'ball-and-stick',
        'spacefill', …) and optionally its color theme."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.set_model_representation(mid, rep)
            if color is not None:
                _app.set_model_color(mid, color)
            return mid
        return {"model": await _qt(work), "rep": rep, "color": color}

    @mcp.tool()
    async def set_volume(name: str = None, iso: float = None,
                         opacity: float = None, color: str = None,
                         visible: bool = None) -> dict:
        """Adjust a map's isosurface: iso level, opacity, color, visibility."""
        def work():
            vid = _volume_id(name)
            if vid is None:
                raise ValueError("no such volume: %r" % name)
            if iso is not None:
                _app.set_volume_iso(vid, iso)
            if opacity is not None:
                _app.set_volume_opacity(vid, opacity)
            if color is not None:
                _app.set_volume_color(vid, color)
            if visible is not None:
                _app.set_volume_visible(vid, visible)
            return vid
        return {"volume": await _qt(work)}

    @mcp.tool()
    async def set_background(color: str) -> dict:
        """Set the viewport background (hex color, e.g. '#101018')."""
        _session().set_background(color)
        return {"background": color}

    @mcp.tool()
    async def reset_view() -> dict:
        """Reframe the camera on the whole scene."""
        _session().reset_view()
        return {"ok": True}

    @mcp.tool()
    async def set_camera(state: dict) -> dict:
        """Set the camera from a state dict as returned by camera_state."""
        _session().set_camera(state)
        return {"ok": True}

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
