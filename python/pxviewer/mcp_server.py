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

    # -- python-side introspection -----------------------------------------
    # The viewport wire shows a subset of the app's state; these read the
    # model entries and cctbx data directly.

    @mcp.tool()
    async def selection_state() -> dict:
        """The current scene selection — per model, the selected atom count
        and residue labels. This is the same state the GUI tables and
        clip-sphere see: a selection is a selection."""
        def work():
            out = {}
            for mid, indices in _app._scene_selection.items():
                entry = _app._model_entry(mid)
                session = entry.get("session") if entry else None
                residues, seen = [], set()
                if session is not None:
                    arrays = session._data.arrays
                    for i in indices:
                        key = (arrays.chain[i], int(arrays.resseq[i]),
                               arrays.resname[i])
                        if key not in seen:
                            seen.add(key)
                            residues.append(
                                f"{key[0]}/{key[2]}{key[1]}" if key[0]
                                else f"{key[2]}{key[1]}")
                out[mid] = {"atoms": len(indices),
                            "residues": residues[:40],
                            "truncated": len(residues) > 40,
                            "name": entry.get("name") if entry else None}
            return out
        return {"selection": await _qt(work)}

    @mcp.tool()
    async def atoms(expression: str, model: str = None) -> dict:
        """Per-atom records for a selection without applying it — for
        questions the grammar can't ask alone ('which tyrosine is closest to
        residue 40': select resname TYR here and compare xyz against the
        reference's). Fields: i (positional index), name, chain, resseq,
        resname, element, xyz, b, occ. Capped at 2000 atoms; 'centroid' is
        always over the full match."""
        def work():
            mid, session = _model_session(model)
            data = session._data
            indices = data.selection_indices(expression)
            a = data.arrays
            cap = 2000
            atoms = []
            for i in indices[:cap]:
                i = int(i)
                atoms.append({
                    "i": i, "name": a.name[i], "chain": a.chain[i],
                    "resseq": int(a.resseq[i]), "resname": a.resname[i],
                    "element": a.element[i],
                    "xyz": [round(float(a.x[i]), 3), round(float(a.y[i]), 3),
                            round(float(a.z[i]), 3)],
                    "b": None if a.b is None else round(float(a.b[i]), 2),
                    "occ": None if a.occ is None else round(float(a.occ[i]), 3)})
            centroid = None
            if indices:
                import numpy as np
                centroid = np.stack([a.x, a.y, a.z],
                                    axis=1)[list(indices)].mean(axis=0)
                centroid = [round(float(v), 3) for v in centroid]
            return {"model": mid, "atoms": atoms, "matched": len(indices),
                    "truncated": len(indices) > cap, "centroid": centroid}
        return await _qt(work)

    @mcp.tool()
    async def model_state(model: str) -> dict:
        """Full Python-side state of one model: appearance (clip, sphere,
        rep, color), structure-type layers, hidden-atom count, conformer,
        attribute-coloring domain, restraint-edit count, and whether the
        geometry restraints have been processed (the objects-tree checkbox)."""
        def work():
            mid = _model_id(model)
            entry = _app._model_entry(mid) if mid else None
            if entry is None:
                raise ValueError("no such model: %r" % model)
            from pxviewer.desktop import _restraints_built
            session = entry["session"]
            attribute = entry.get("attribute")
            return {
                "id": mid, "name": entry.get("name"),
                "atoms": session._data.n_atoms if session else None,
                "rep": entry.get("rep"), "color": entry.get("color"),
                "visible": entry.get("visible", True),
                "types": _app.model_structure_types(mid),
                "hidden_atoms": len(entry.get("hidden_atoms") or ()),
                "conformer": entry.get("conformer"),
                "conformers": entry.get("conformers") or [],
                "value_domain": attribute.get("domain") if attribute else None,
                "edits": len(_app.model_edits(mid)),
                "restraints_built": _restraints_built(entry),
                "selected": len(_app._scene_selection.get(mid, ())),
                **_app.model_appearance(mid),
            }
        return {"model": await _qt(work)}

    @mcp.tool()
    async def volume_state(volume: str) -> dict:
        """Full Python-side state of one volume: style/color/iso/opacity,
        clip, sphere, mask radius, negative contour, source and pairing."""
        def work():
            vid = _volume_id(volume)
            entry = _app._volume_entry(vid) if vid else None
            if entry is None:
                raise ValueError("no such volume: %r" % volume)
            state = _app.volume_appearance(vid)
            state.update({"id": vid, "name": entry.get("name"),
                          "source": entry.get("source")})
            return state
        return {"volume": await _qt(work)}

    @mcp.tool()
    async def reflections() -> dict:
        """The loaded reflection files — ids and names for make_maps."""
        def work():
            return [{"id": r["id"], "name": r.get("name"),
                     "group": r.get("group")} for r in _app._reflections]
        return {"reflections": await _qt(work)}

    @mcp.tool()
    async def primitives() -> dict:
        """The measurement/label primitives drawn on the control session —
        ids for remove_primitive."""
        session = _session()
        out = []
        for pid, message in session._primitives.items():
            out.append({"id": pid, "kind": message.get("kind")})
        return {"primitives": out}

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
                         visible: bool = None,
                         style: str = None, mask_radius: float = None,
                         position: list = None,
                         negative_iso: float = None,
                         negative_color: str = None,
                         negative_style: str = None,
                         negative_opacity: float = None,
                         negative_visible: bool = None) -> dict:
        """Adjust a map: iso level (sigma), opacity, color (hex), visibility.
        style is 'isosurface'|'mesh'|'dots'; mask_radius (A) restricts display
        to within that of the paired model's atoms; position translates the map
        by [dx, dy, dz] A. negative_iso enables a second (negative) contour with
        its own color/style/opacity/visible."""
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
            if style is not None:
                _app.set_volume_style(vid, style)
            if mask_radius is not None:
                _app.set_volume_mask(vid, mask_radius)
            if position is not None:
                _app.set_volume_position(vid, list(map(float, position)))
            if negative_iso is not None:
                _app.set_volume_negative_iso(vid, negative_iso)
            if negative_color is not None:
                _app.set_volume_negative_color(vid, negative_color)
            if negative_style is not None:
                _app.set_volume_negative_style(vid, negative_style)
            if negative_opacity is not None:
                _app.set_volume_negative_opacity(vid, negative_opacity)
            if negative_visible is not None:
                _app.set_volume_negative_visible(vid, negative_visible)
            return vid
        return {"volume": await _qt(work)}

    @mcp.tool()
    async def set_clip_slab(front: float, back: float, model: str = None,
                            volume: str = None) -> dict:
        """Clip an object to a front/rear slab — offsets in angstroms along the
        view direction (front near the camera). Give model or volume."""
        def work():
            if volume is not None:
                vid = _volume_id(volume)
                if vid is None:
                    raise ValueError("no such volume: %r" % volume)
                _app.set_volume_clip(vid, front, back)
            else:
                mid = _model_id(model)
                if mid is None:
                    raise ValueError("no such model: %r" % model)
                _app.set_model_clip(mid, front, back)
        await _qt(work)
        return {"clip": [front, back]}

    @mcp.tool()
    async def set_clip_sphere(mode: str, radius: float = None,
                              model: str = None, volume: str = None) -> dict:
        """Clip-sphere policy for an object: 'selection' (sphere re-fits to each
        selection — armed even while dormant; models only), 'view' (sphere
        follows the camera target, Coot-style), 'point' (frozen at the current
        center), 'off'. radius is in angstroms; a selection-mode radius is
        fitted, so pass one only for view/point. Give model or volume."""
        def work():
            if volume is not None:
                vid = _volume_id(volume)
                if vid is None:
                    raise ValueError("no such volume: %r" % volume)
                _app.set_volume_sphere(vid, mode, radius)
                return _app.volume_sphere_state(vid)
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.set_model_sphere(mid, mode, radius)
            return _app.model_sphere_state(mid)
        return {"sphere": await _qt(work)}

    # -- objects ---------------------------------------------------------

    @mcp.tool()
    async def fetch(pdb_id: str = None, emdb_number: int = None,
                    entities: list = None, work_dir: str = None) -> dict:
        """Fetch an entry from the PDB/EMDB archives and load it. entities is a
        subset of ['model', 'reflections', 'map'] (default: model + map for an
        emdb_number, model for a pdb_id). Waits for the objects to arrive."""
        def work():
            ents = entities or (["model", "map"] if emdb_number is not None
                                and pdb_id is None else ["model"])
            before = (len(_app._models), len(_app._volumes),
                      len(_app._reflections))
            _app.fetch_and_load(pdb_id=pdb_id, emdb_number=emdb_number,
                                entities=ents,
                                work_dir=str(Path(work_dir).expanduser())
                                if work_dir else None)
            return before, ents
        before, ents = await _qt(work)
        deadline = time.time() + 120
        while time.time() < deadline:
            counts = await _qt(lambda: (len(_app._models), len(_app._volumes),
                                        len(_app._reflections)))
            if counts != before:
                break
            await asyncio.sleep(0.5)
        _wait_ready(timeout=60)
        return {"fetched": ents, "pdb_id": pdb_id, "emdb_number": emdb_number}

    @mcp.tool()
    async def remove(kind: str, ident: str) -> dict:
        """Remove an object: kind is 'model', 'volume', 'reflections' or
        'marker'; ident is its id or name."""
        def work():
            if kind == "model":
                mid = _model_id(ident)
                if mid is None:
                    raise ValueError("no such model: %r" % ident)
                _app.remove_model(mid)
            elif kind == "volume":
                vid = _volume_id(ident)
                if vid is None:
                    raise ValueError("no such volume: %r" % ident)
                _app.remove_volume(vid)
            elif kind == "reflections":
                rids = {r["id"] for r in _app._reflections} | {
                    r.get("name") for r in _app._reflections}
                if ident not in rids:
                    raise ValueError("no such reflections: %r" % ident)
                rid = next(r["id"] for r in _app._reflections
                           if ident in (r["id"], r.get("name")))
                _app.remove_reflections(rid)
            elif kind == "marker":
                _app.remove_marker(ident)
            else:
                raise ValueError("kind must be model|volume|reflections|marker")
            return ident
        return {"removed": await _qt(work), "kind": kind}

    @mcp.tool()
    async def set_active_model(model: str) -> dict:
        """Make a model the active one — the object the selection box, atoms
        table and geometry act on."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.set_active_model(mid)
            return mid
        return {"active_model": await _qt(work)}

    @mcp.tool()
    async def set_object_visible(kind: str, ident: str, visible: bool) -> dict:
        """Show or hide an object — the eye icon. kind is 'model' or 'volume'."""
        def work():
            if kind == "model":
                mid = _model_id(ident)
                if mid is None:
                    raise ValueError("no such model: %r" % ident)
                _app.set_model_visible(mid, visible)
            elif kind == "volume":
                vid = _volume_id(ident)
                if vid is None:
                    raise ValueError("no such volume: %r" % ident)
                _app.set_volume_visible(vid, visible)
            else:
                raise ValueError("kind must be model or volume")
        await _qt(work)
        return {"visible": visible, "kind": kind, "id": ident}

    @mcp.tool()
    async def write_object(kind: str, ident: str, path: str) -> dict:
        """Write an object's data — the Save menu's 'Write object…': a model's
        coordinates (PDB/mmCIF), a volume's map, a reflection file."""
        def work():
            _app.write_object(kind, ident, str(Path(path).expanduser()))
        await _qt(work)
        return {"written": path, "kind": kind, "id": ident}

    @mcp.tool()
    async def make_maps(reflections: str, model: str) -> dict:
        """Compute density maps from a reflection file and a model (model
        phases), paired so masking and map-fitting work. Waits for the maps."""
        def work():
            mid = _model_id(model)
            rid = next((r["id"] for r in _app._reflections
                        if reflections in (r["id"], r.get("name"))), None)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            if rid is None:
                raise ValueError("no such reflections: %r" % reflections)
            before = len(_app._volumes)
            _app.make_maps(rid, mid)
            return before
        before = await _qt(work)
        deadline = time.time() + 180
        while time.time() < deadline:
            if await _qt(lambda: len(_app._volumes)) != before:
                break
            await asyncio.sleep(0.5)
        _wait_ready(timeout=60)
        return {"ok": True}

    @mcp.tool()
    async def pair_model_with_map(model: str, volume: str) -> dict:
        """Pair a model with a map into one cctbx manager — required before
        minimizing into density or masking by model."""
        def work():
            mid = _model_id(model)
            vid = _volume_id(volume)
            if mid is None or vid is None:
                raise ValueError("bad model or volume id")
            return _app.pair_model_with_map(mid, vid)
        return {"paired": await _qt(work)}

    @mcp.tool()
    async def generate_ligand_restraints(model: str = None) -> dict:
        """Build monomer dictionaries (rdkit) for the model's unknown ligands —
        the answer when restraints fail on an unrecognized residue. Returns
        {code: smiles} so the perceived chemistry can be checked."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            return _app.generate_ligand_restraints(mid)
        return {"generated": await _qt(work)}

    # -- appearance --------------------------------------------------------

    @mcp.tool()
    async def set_type_hidden(model: str, label: str, hidden: bool) -> dict:
        """Show/hide one structure-type layer of a model (e.g. 'cartoon',
        'ball-and-stick') — the per-type eye in the model's Show menu."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.set_model_type_hidden(mid, label, hidden)
        await _qt(work)
        return {"model": model, "type": label, "hidden": hidden}

    @mcp.tool()
    async def set_conformer(model: str, conformer: str = None) -> dict:
        """Switch a multi-conformer (ensemble/NMR) model's shown conformer;
        None returns to 'all'."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.set_model_conformer(mid, conformer)
        await _qt(work)
        return {"model": model, "conformer": conformer}

    @mcp.tool()
    async def color_by(model: str = None, kind: str = None,
                       domain: list = None) -> dict:
        """Color a model by a quantity: 'bfactor', 'occupancy' (per-atom value
        ramps with a settable domain), 'qscore', 'mapcc' (needs a paired map),
        'hotspots' (validation severity cloud coloring). kind=None resets."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            if kind in ("bfactor", "occupancy", None):
                _app.color_model_by_property(mid, kind)
                if domain is not None:
                    _app.set_model_value_domain(mid, domain[0], domain[1])
            elif kind == "qscore":
                _app.color_model_by_qscore(mid)
            elif kind == "mapcc":
                _app.color_model_by_mapcc(mid)
            elif kind == "hotspots":
                _app.color_model_by_hotspots(mid)
            else:
                raise ValueError("kind must be bfactor|occupancy|qscore|mapcc|"
                                 "hotspots (or omit for none)")
            return mid
        return {"model": await _qt(work), "colored_by": kind}

    @mcp.tool()
    async def set_selected_hidden(hide: bool) -> dict:
        """Hide (or un-hide) the currently selected atoms — partial
        representation, same as the Selection pane buttons."""
        await _qt(lambda: _app.hide_selected() if hide else _app.show_selected())
        return {"hidden": hide}

    @mcp.tool()
    async def set_computed_interactions(visible: bool) -> dict:
        """Show/hide Mol*'s geometry-inferred non-covalent interaction overlay."""
        _session().set_computed_interactions(visible)
        return {"computed_interactions": visible}

    @mcp.tool()
    async def set_model_interactions(model: str, visible: bool) -> dict:
        """Toggle a single model's computed-interaction overlay (its row's
        'interactions' switch)."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.set_model_interactions(mid, visible)
        await _qt(work)
        return {"model": model, "interactions": visible}

    @mcp.tool()
    async def set_interactions(pairs: list) -> dict:
        """Draw explicit typed atom pairs as interaction notation —
        [{'a': i, 'b': j, 'kind': 'hydrogen-bond'}, …]. Kinds: hydrogen-bond,
        weak-hydrogen-bond, ionic, hydrophobic, pi-stacking, cation-pi,
        halogen-bond, metal-coordination, water-bridge, covalent; an empty
        list clears."""
        drawn = _session().set_interactions(pairs)
        return {"interactions": len(drawn)}

    @mcp.tool()
    async def set_clashes(pairs: list) -> dict:
        """Draw red clash markers between atom-index pairs [[i, j], …];
        an empty list clears."""
        drawn = _session().set_clashes(pairs)
        return {"clashes": len(drawn)}

    @mcp.tool()
    async def set_focus_surroundings(enabled: bool) -> dict:
        """Show/hide Mol*'s click-focus ball-and-stick neighborhood layer."""
        _session().set_focus_surroundings(enabled)
        return {"focus_surroundings": enabled}

    @mcp.tool()
    async def set_localres(volume: str, on: bool = None,
                           downsample: int = None, domain: list = None) -> dict:
        """Color a full map by its pinned local-resolution map: 'on' toggles
        the coloring, 'downsample' coarsens the value grid, 'domain' [lo, hi]
        sets the color scale's range in A."""
        def work():
            vid = _volume_id(volume)
            if vid is None:
                raise ValueError("no such volume: %r" % volume)
            if on is not None:
                _app.set_color_by_resolution(vid, on)
            if downsample is not None:
                _app.set_localres_downsample(vid, downsample)
            if domain is not None:
                _app.set_localres_domain(vid, domain[0], domain[1])
        await _qt(work)
        return {"volume": volume, "localres": True}

    # -- camera / navigation ----------------------------------------------

    @mcp.tool()
    async def zoom_view(direction: int = -1) -> dict:
        """Step the camera in (direction < 0) or out (> 0), target and
        orientation kept."""
        await _qt(lambda: _app.zoom_view(direction))
        return {"zoomed": direction}

    @mcp.tool()
    async def focus_residue(chain: str, resid: str) -> dict:
        """Select + focus one residue by chain id and resid (MolProbity's
        resseq+icode string, e.g. '42' or '42A') on the active model — the same
        pipeline as a table row."""
        def work():
            return _app.focus_residue(chain, resid)
        expr = await _qt(work)
        return {"expression": expr}

    @mcp.tool()
    async def step(step: int = 1) -> dict:
        """Move the focused residue one position along its chain — the
        space-bar walk; each step is a selection."""
        await _qt(lambda: _app.advance_residue(step))
        return {"stepped": step}

    @mcp.tool()
    async def mouse_selection() -> dict:
        """The atoms the user has picked in the viewport (positional indices)."""
        sel = _session().mouse_selection
        return {"indices": list(sel.indices)}

    @mcp.tool()
    async def wait_for_selection(timeout: float = 60.0) -> dict:
        """Block until the user next changes the viewport selection (pick
        mode must be on); returns the picked atom indices."""
        session = _session()
        session.enable_mouse_selection()
        sel = session.wait_for_selection(timeout=timeout)
        return {"indices": list(sel.indices) if sel is not None else None}

    # -- markup ------------------------------------------------------------

    @mcp.tool()
    async def measure(kind: str, atoms: list = None) -> dict:
        """Draw a measurement: kind is 'distance', 'angle' or 'dihedral'.
        With atoms=None it measures the current selection (which must hold
        exactly 2/3/4 atoms); with an explicit atom-index list it adds the
        notation directly."""
        if atoms is not None:
            session = _session()
            if kind == "distance":
                p = session.add_distance(*atoms[:2])
            elif kind == "angle":
                p = session.add_angle(*atoms[:3])
            elif kind == "dihedral":
                p = session.add_dihedral(*atoms[:4])
            else:
                raise ValueError("kind must be distance|angle|dihedral")
            return {"primitive": p.id}
        result = await _qt(lambda: _app.measure_selection(kind))
        return {"result": result}

    @mcp.tool()
    async def add_label(atom: int, text: str) -> dict:
        """Annotate an atom with a text label that tracks it."""
        p = _session().add_label(atom, text)
        return {"primitive": p.id}

    @mcp.tool()
    async def remove_primitive(primitive_id: str) -> dict:
        """Remove one measurement/label primitive by id."""
        _session().remove_primitive(primitive_id)
        return {"removed": primitive_id}

    @mcp.tool()
    async def clear_primitives() -> dict:
        """Remove every measurement/label primitive."""
        _session().clear_primitives()
        return {"cleared": True}

    # -- analysis / editing --------------------------------------------------

    @mcp.tool()
    async def run_validation(clashes: bool = False,
                             hotspots: bool = False,
                             keys: list = None) -> dict:
        """Run the MolProbity validations on the active model as a background
        job — results land in the Validation tab as each check finishes.
        clashes=True adds the reduce2+probe2 all-atom contact analysis (the
        heavy one); hotspots=True adds the severity aggregate."""
        await _qt(lambda: _app.run_validation(keys=keys, clashes=clashes,
                                              hotspots=hotspots))
        return {"started": True,
                "note": "running in the background; results land in the "
                        "Validation tab"}

    @mcp.tool()
    async def minimize(use_map: bool = False, wait: float = 120.0) -> dict:
        """Minimize the active model onto its geometry restraints (or into a
        paired map with use_map=True). The run is continuous — it holds at the
        minimum — so this waits for the hold then stops it, leaving the model
        minimized."""
        def work():
            _app.minimize_model(use_map=use_map)
        await _qt(work)
        got = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _app._minimize_idle.wait(wait))
        await _qt(lambda: _app.stop_minimization())
        return {"minimized": True, "reached_minimum": bool(got)}

    @mcp.tool()
    async def set_attribute(name: str, values: list) -> dict:
        """Register a named per-atom scalar attribute on the control session —
        length must equal the atom count; color with it via color paths."""
        _session().set_attribute(name, values)
        return {"attribute": name, "n": len(values)}

    @mcp.tool()
    async def list_edits(model: str = None) -> dict:
        """The custom restraint edits (bonds/angles/…) carried on a model."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            return _app.model_edits(mid)
        return {"edits": await _qt(work)}

    @mcp.tool()
    async def add_edit(kind: str, sigma: float = None) -> dict:
        """Author a bond/angle/dihedral restraint from the active model's
        selected atoms (needs exactly 2/3/4 selected), taking the current
        geometry as the target."""
        def work():
            return _app.add_edit_from_selection(
                _app._active_model_id, kind, sigma=sigma)
        return {"edit": await _qt(work)}

    @mcp.tool()
    async def remove_edit(model: str, index: int) -> dict:
        """Drop the edit at index (see list_edits) and rebuild restraints."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.remove_edit(mid, index)
        await _qt(work)
        return {"removed": index}

    @mcp.tool()
    async def clear_edits(model: str) -> dict:
        """Drop every custom edit on a model and rebuild its restraints."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.clear_edits(mid)
        await _qt(work)
        return {"cleared": True}

    @mcp.tool()
    async def load_edits(model: str, path: str) -> dict:
        """Load custom restraint edits from a PHIL file onto a model."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            return _app.load_edits(mid, str(Path(path).expanduser()))
        return {"loaded": await _qt(work)}

    @mcp.tool()
    async def save_edits(model: str, path: str) -> dict:
        """Save a model's custom restraint edits to a PHIL file."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.save_edits(mid, str(Path(path).expanduser()))
        await _qt(work)
        return {"saved": path}

    @mcp.tool()
    async def save_screenshot(path: str) -> dict:
        """Write the current viewport frame to a PNG file."""
        def work():
            _app.save_screenshot(str(Path(path).expanduser()))
        await _qt(work)
        return {"saved": path}

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
