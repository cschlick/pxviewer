"""Save and restore a view: the scene as JSON, optionally with its data bundled.

A saved view is a *figure file*: one JSON document naming everything a view is
made of — the objects, their representations, colors, clip slab and sphere,
selections and context, visibility, and the camera — plus a sibling ``*.data``
directory holding the coordinates and maps the JSON cannot describe.

``bundle_data`` decides what goes in that directory:

- ``True`` (the default): every object's data travels with the view — models
  materialize as mmCIF (the *current* coordinates, so dragged or minimized
  geometry reproduces exactly), maps write their working-frame copy. The file
  is self-contained and archivable; this is what figure reproduction wants.
- ``False``: objects that came from a file keep a reference (path + size +
  mtime) instead of a copy — twenty figures off one dataset stay twenty small
  JSONs. Anything without an external source still materializes: generated
  demos, computed maps and edited coordinates cannot be re-derived from a
  path, and a reference that cannot resolve is not a view.

Restoring is replay through the same setters the panes drive, in an order
chosen so nothing clobbers what a later step writes: objects load, then their
appearance, then selections, then clip state, then the camera (anything that
focuses would move it — it goes last).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

VIEW_FORMAT = "pxviewer-view"
VIEW_VERSION = 1
VIEW_SUFFIX = ".pxview.json"


def _data_dir(path: Path) -> Path:
    """The sibling directory a bundled view's data files live in."""
    return path.parent / f"{path.name[:-len('.json')]}.data" \
        if path.name.endswith(".json") else path.parent / f"{path.name}.data"


def _stat(path: Path) -> dict:
    try:
        st = path.stat()
    except OSError:
        return {}
    return {"size": st.st_size, "mtime": st.st_mtime}


# -- save ----------------------------------------------------------------------

def save_view(app, path, *, bundle_data: bool = True) -> dict:
    """Serialize the scene to ``path``; returns the document written.

    ``app`` is the DesktopApp; entries are read off ``_models``/``_volumes``
    directly so a view captures *everything* the scene knows, not just what a
    pane last touched.
    """
    path = Path(path)
    if not str(path).endswith(".json"):
        path = path.with_suffix(VIEW_SUFFIX)
    data_dir = _data_dir(path)
    objects = []
    for entry in app._models:
        objects.append(_model_spec(app, entry, path, data_dir, bundle_data))
    for entry in app._volumes:
        objects.append(_volume_spec(app, entry, path, data_dir, bundle_data))

    camera = None
    background = None
    control = app._control_session()
    if control is not None:
        try:
            camera = control.camera_state()
        except Exception:
            pass
        try:
            background = control.background_color()
        except Exception:
            pass

    doc = {
        "format": VIEW_FORMAT,
        "version": VIEW_VERSION,
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bundle_data": bool(bundle_data),
        "camera": camera,
        "background": background,
        "active": app._active_model_id,
        "objects": objects,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return doc


def _model_spec(app, entry, path, data_dir, bundle_data) -> dict:
    mid = entry["id"]
    with app._scene_lock:
        selection = sorted(app._scene_selection.get(mid, ()))
    attribute = entry.get("attribute") or {}
    spec = {
        "kind": "model",
        "id": mid,
        "name": entry["name"],
        "source": _source_spec(entry.get("source")),
        "data": None,
        "visible": bool(entry["visible"]),
        "group": _group_name(app, entry.get("group")),
        "rep": entry.get("rep"),
        "color": entry.get("color"),
        # Serialized, not re-rolled: the palette pick is random per session, so
        # without it a reloaded view could come back a different color.
        "color_default": entry.get("color_default"),
        "attribute": (
            {"name": attribute.get("name"), "domain": attribute.get("domain"),
             "palette": attribute.get("palette")}
            if attribute.get("name") else None),
        "hidden_types": sorted(entry.get("hidden_types") or ()),
        # Indices, not the expression: exact is the point, and
        # _selection_expression declines partial residues anyway.
        "hidden_atoms": sorted(entry.get("hidden_atoms") or ()),
        "clip": list(entry.get("clip") or (0.0, 1.0)),
        "sphere": dict(app._sphere_state(entry)),
        "clip_depth": entry.get("_auto_clip_depth"),
        "interactions": bool(entry.get("interactions")),
        # Summaries only: the cctbx restraint objects are not JSON-able, and the
        # coordinates they produced are already baked into the bundled file.
        "edits": app.model_edits(mid),
        "selection": {"indices": selection,
                      "expression": (app._selection_expression(entry, selection)
                                     if selection else "")},
        "context": bool(entry.get("context_on")),
    }
    spec["data"] = _model_data(app, entry, path, data_dir, bundle_data)
    return spec


def _volume_spec(app, entry, path, data_dir, bundle_data) -> dict:
    spec = {
        "kind": "volume",
        "id": entry["id"],
        "name": entry["name"],
        "source": _source_spec(entry.get("source")),
        "data": None,
        "visible": bool(entry["visible"]),
        "group": _group_name(app, entry.get("group")),
        "iso": entry.get("iso"),
        "color": entry.get("color"),
        "opacity": entry.get("opacity"),
        "style": entry.get("style"),
        "clip": list(entry.get("clip") or (0.0, 1.0)),
        "sphere": dict(app._sphere_state(entry)),
        "negative_color": entry.get("negative_color"),
        "iso_kind": entry.get("iso_kind"),
        "negative_iso": entry.get("negative_iso"),
        "negative_visible": entry.get("negative_visible"),
        "negative_opacity": entry.get("negative_opacity"),
        "negative_style": entry.get("negative_style"),
        "mask_radius": entry.get("mask_radius"),
        "is_resolution": bool(entry.get("is_resolution")),
        "pinned_to": entry.get("pinned_to"),
        "color_by_resolution": bool(entry.get("color_by_resolution")),
    }
    spec["data"] = _volume_data(app, entry, path, data_dir, bundle_data)
    return spec


def _source_spec(source) -> Optional[dict]:
    """The provenance block, freshened with the file's current stat."""
    if not isinstance(source, dict):
        return None
    out = dict(source)
    p = out.get("path")
    if p:
        out.update(_stat(Path(p)))
    return out


def _group_name(app, gid) -> Optional[str]:
    group = app._groups.get(gid) if gid else None
    return group.get("name") if isinstance(group, dict) else None


def _model_data(app, entry, path, data_dir, bundle_data) -> Optional[str]:
    """Where the model's coordinates live, relative to the view file — or None
    when the source reference stands for them."""
    source = entry.get("source") or {}
    dirty = entry.get("_coords_dirty") or entry.get("edits")
    # Only a single-file source can stand in: a group member's own file is not
    # distinguishable from its siblings' (they share the "files" source).
    if (not bundle_data and not dirty
            and source.get("kind") == "file"
            and Path(source.get("path") or "").is_file()):
        return None
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{entry['id']}.cif"
    app.write_object("model", entry["id"], str(out))
    return f"{data_dir.name}/{out.name}"


def _volume_data(app, entry, path, data_dir, bundle_data) -> Optional[str]:
    source = entry.get("source") or {}
    if (not bundle_data
            and source.get("kind") == "file"
            and Path(source.get("path") or "").is_file()):
        return None
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{entry['id']}.map"
    # The real map in the viewer's working frame — the same write the display
    # copy takes. A mask stays a mask-radius spec, applied again on restore.
    entry["data"].write_map(str(out), working_frame=True)
    return f"{data_dir.name}/{out.name}"


# -- load ----------------------------------------------------------------------

def load_view(app, path) -> dict:
    """Restore a saved view into ``app``. Returns ``{spec_id: new_id}``.

    Unknown keys are ignored, so a newer file reads as much as this build can —
    and missing data raises on the object that names it, leaving the rest of
    the view to restore around it.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get("format") != VIEW_FORMAT:
        raise ValueError(f"{path.name} is not a pxviewer view file")
    if int(doc.get("version") or 0) > VIEW_VERSION:
        raise ValueError(
            f"{path.name} is a newer view format (version {doc['version']})")

    app.stop_demo()
    # A saved view is a whole scene, so restore means replace: objects already
    # loaded leave first, or the result is a hybrid neither file describes. The
    # menu path confirms before it gets here — the scene may hold unsaved work.
    for rid in [r["id"] for r in list(app._reflections)]:
        try:
            app.remove_reflections(rid)
        except Exception:
            pass
    for vid in [v["id"] for v in list(app._volumes)]:
        try:
            app.remove_volume(vid)
        except Exception:
            pass
    for mid in [m["id"] for m in list(app._models)]:
        try:
            app.remove_model(mid)
        except Exception:
            pass

    id_map = {}
    pending = []          # (spec, kind, new_id) — state applied after all loads
    groups: dict = {}     # saved group name -> {"models": [], "volumes": []}
    data_dir = _data_dir(path)

    def note(spec, kind, nid):
        if nid:
            id_map[spec["id"]] = nid
            pending.append((spec, kind, nid))
            g = spec.get("group")
            if g:
                groups.setdefault(g, {"models": [], "volumes": []})[
                    "models" if kind == "model" else "volumes"].append(nid)

    for spec in doc.get("objects") or []:
        kind = spec.get("kind")
        try:
            nid = _load_object(app, spec, path, data_dir)
        except Exception:
            nid = None    # a missing source strands its object, not the view
        note(spec, kind, nid)

    # Pairing: objects that shared a group are re-paired so mask/alignment
    # semantics hold. Both blobs are already in one frame, so the relocation
    # inside pair_model_with_map is a no-op where it matters.
    for members in groups.values():
        for mid in members["models"]:
            for vid in members["volumes"]:
                try:
                    app.pair_model_with_map(mid, vid)
                except Exception:
                    pass

    for spec, kind, nid in pending:
        try:
            if kind == "model":
                _apply_model_state(app, spec, nid)
            else:
                _apply_volume_state(app, spec, nid)
        except Exception:
            pass

    # Resolution pins reference other volumes by the *saved* id — remap to the
    # new one, then enable color-by-resolution (it needs its pin first).
    for spec, kind, nid in pending:
        if kind != "volume":
            continue
        entry = app._volume_entry(nid)
        if entry is None:
            continue
        pinned = spec.get("pinned_to")
        if pinned and pinned in id_map:
            entry["pinned_to"] = id_map[pinned]
            parent = app._volume_entry(id_map[pinned])
            if parent is not None:
                parent["resolution_map"] = nid
    for spec, kind, nid in pending:
        if kind == "volume" and spec.get("color_by_resolution"):
            try:
                app.set_color_by_resolution(nid, True)
            except Exception:
                pass

    active = id_map.get(doc.get("active") or "")
    if active:
        try:
            app.set_active_model(active)
        except Exception:
            pass

    background = doc.get("background")
    camera = doc.get("camera")
    control = app._control_session()
    if background and control is not None:
        try:
            control.set_background(background)
        except Exception:
            pass
    if camera and control is not None:
        # The camera is the last word: a camera-set into a building viewer loses
        # to the build's auto-frame, and the page may not even be connected here
        # (blocking to wait for it starves the navigation itself). set_camera
        # arms a one-shot that the session delivers on the client's "ready" —
        # the post-build ack that puts it safely behind the frame.
        try:
            control.set_camera(camera)
        except Exception:
            pass
    return id_map


def _load_object(app, spec, path, data_dir) -> Optional[str]:
    """Get the object's data on screen; return its new id."""
    data_rel = spec.get("data")
    source = spec.get("source") or {}
    file_path = None
    if data_rel:
        file_path = path.parent / data_rel
        if not file_path.is_file():
            raise FileNotFoundError(f"view data missing: {data_rel}")
    elif source.get("kind") == "file":
        file_path = Path(source["path"])
        if not file_path.is_file():
            # Stale reference: the file moved since the view was saved. The one
            # retry worth making is a re-fetch, for objects a fetch produced.
            if source.get("fetched"):
                file_path = _fetch_to_file(source["fetched"], spec)
            if file_path is None or not file_path.is_file():
                raise FileNotFoundError(f"referenced file missing: {source['path']}")
    if file_path is None:
        return None

    if spec.get("kind") == "model":
        from .live import LiveSession
        session = LiveSession.from_model_file(str(file_path))
        return app._add_model(session, spec.get("name") or file_path.name)
    if spec.get("kind") == "volume":
        from .volume_io import VolumeData
        vid = app._add_volume(
            VolumeData.from_map_file(str(file_path)),
            spec.get("name") or file_path.name,
            color=spec.get("color"), iso=spec.get("iso"),
            negative_color=spec.get("negative_color"),
            iso_kind=spec.get("iso_kind") or "relative",
            style=spec.get("style") or "surface")
        if spec.get("is_resolution"):
            # Set before anything draws it: the flag keeps it out of the volume
            # scene on the next reload, where a draw-first-set-later order leaks
            # a giant featureless blob over the data.
            app._volume_entry(vid)["is_resolution"] = True
            app._reload_viewport()
        return vid
    return None


def _fetch_to_file(fetched: dict, spec: dict) -> Optional[Path]:
    """Re-download a fetched object's data when the view references it."""
    entity = "model" if spec.get("kind") == "model" else "map"
    pdb_id = fetched.get("pdb_id")
    emdb_number = fetched.get("emdb_number")
    if not (pdb_id or emdb_number):
        return None
    import tempfile

    from . import fetch as fetchmod
    out = Path(tempfile.mkdtemp(prefix="pxview-fetch-"))
    paths = fetchmod.fetch_entry(
        entities=[entity], work_dir=out,
        pdb_id=pdb_id, emdb_number=emdb_number)
    return paths.get(entity)


def _apply_model_state(app, spec, mid) -> None:
    """Replay one model's saved appearance onto its restored entry.

    Order matters: representation and colors first (the context layer builds
    over the final rep), then the standing selection, then clip state last —
    the selection pipeline owns the sphere, so the saved sphere goes down after
    it or the pipeline would overwrite (or lift) the restored one.
    """
    entry = app._model_entry(mid)
    if entry is None:
        return
    if spec.get("color_default") is not None:
        # Before the rep apply below — the palette pick is read there.
        entry["color_default"] = spec["color_default"]
    if spec.get("rep"):
        app.set_model_representation(mid, spec["rep"])
    if spec.get("color") is not None:
        app.set_model_color(mid, spec["color"])
    attribute = spec.get("attribute") or {}
    domain = attribute.get("domain")
    if domain and len(domain) == 2:
        app.set_model_value_domain(mid, domain[0], domain[1])
    hidden_types = set(spec.get("hidden_types") or ())
    hidden_atoms = set(spec.get("hidden_atoms") or ())
    if hidden_types or hidden_atoms:
        entry["hidden_types"] = hidden_types
        entry["hidden_atoms"] = hidden_atoms
        app._apply_model_rep(entry)

    indices = (spec.get("selection") or {}).get("indices") or []
    session = entry["session"]
    if indices and getattr(session, "model", None) is not None:
        # focus=False — the saved camera restores later, once. clip=False so a
        # selection while "selection"-armed does not refit the sphere: the saved
        # state below is verbatim and wins.
        app._select_fragment(mid, session, entry, list(indices),
                             focus=False, clip=False,
                             context=bool(spec.get("context")))

    clip = spec.get("clip")
    if clip and len(clip) == 2:
        app.set_model_clip(mid, clip[0], clip[1])
    sphere = spec.get("sphere") or {}
    mode = sphere.get("mode")
    if mode:
        app._apply_model_sphere(
            entry, mode, radius=sphere.get("radius"),
            center=sphere.get("center"), depth=spec.get("clip_depth"))
    if "clip_depth" in spec:
        # _auto_clip_depth is caller bookkeeping _apply_model_sphere declines to
        # own — the restore is the caller here, and the parked slab has to come
        # back onto the entry too or a re-save (or a later re-apply) loses it.
        entry["_auto_clip_depth"] = spec["clip_depth"]
    if spec.get("visible") is not None:
        app.set_model_visible(mid, bool(spec["visible"]))
    entry["interactions"] = bool(spec.get("interactions"))
    try:
        entry["session"].set_computed_interactions(entry["interactions"])
    except Exception:
        pass


def _apply_volume_state(app, spec, vid) -> None:
    entry = app._volume_entry(vid)
    if entry is None:
        return
    if spec.get("iso") is not None:
        app.set_volume_iso(vid, spec["iso"])
    if spec.get("style"):
        app.set_volume_style(vid, spec["style"])
    if spec.get("opacity") is not None:
        app.set_volume_opacity(vid, spec["opacity"])
    if spec.get("color") is not None:
        app.set_volume_color(vid, spec["color"])
    # Each leg guards on its own key: gating the block on negative_color would
    # drop a negative iso/style saved with the default color.
    for key, setter in (
            ("negative_iso", app.set_volume_negative_iso),
            ("negative_opacity", app.set_volume_negative_opacity),
            ("negative_style", app.set_volume_negative_style),
            ("negative_color", app.set_volume_negative_color)):
        if spec.get(key) is not None:
            setter(vid, spec[key])
    if spec.get("negative_visible") is not None:
        app.set_volume_negative_visible(vid, bool(spec["negative_visible"]))
    clip = spec.get("clip")
    if clip and len(clip) == 2:
        app.set_volume_clip(vid, clip[0], clip[1])
    sphere = spec.get("sphere") or {}
    if sphere.get("mode"):
        # Verbatim: the saved mode/radius/center, composed with the slab by the
        # send. set_volume_sphere's own center defaults would only second-guess
        # what was saved.
        entry["sphere"] = {"mode": sphere["mode"], "radius": sphere.get("radius"),
                           "center": sphere.get("center")}
        app._send_volume_clip(entry)
    if spec.get("mask_radius") is not None:
        try:
            app.set_volume_mask(vid, spec["mask_radius"])
        except Exception:
            pass     # needs the pair; restored above when both came back
    if spec.get("visible") is not None:
        app.set_volume_visible(vid, bool(spec["visible"]))
