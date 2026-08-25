"""Fetch models, maps, half-maps and reflections from the PDB/EMDB.

A thin wrapper over cctbx's own :mod:`iotbx.pdb.fetch`: it downloads the requested
pieces of an entry into a working directory and hands back the file paths, so the rest
of the app can load them like any other files. Cryo-EM maps are served gzipped; those are
decompressed on the way in (models and structure factors arrive already decompressed).

The one thing cctbx's fetch does not do is turn a PDB id into the EMDB number its maps
live under — that mapping is looked up from the RCSB REST API (standard library only).

Downloads are streamed rather than read whole, for two reasons. A cryo-EM half-map is
~50 MB gzipped and a pair of them is over 100 MB, which is minutes on a slow link: read
in one call, that is an unbroken silence with no way to tell a slow download from a hung
one, so :func:`fetch_entry` takes a ``progress`` callback and reports bytes as they
arrive. And each file lands on a ``.part`` beside its destination and is renamed only
once complete, so an interrupted download can never be mistaken for a finished one --
which matters because ``reuse_existing`` skips files that are already there.

Failures are raised as :class:`FetchError` with the URL and something to do about it.
The underlying errors (``HTTPError``, ``URLError``, ``socket.timeout``, cctbx's ``Sorry``)
are various and none of them say which entry or file was being fetched.
"""

from __future__ import annotations

import gzip
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

#: Read size for streamed downloads. Big enough that the syscall overhead is nothing,
#: small enough that progress moves visibly on a slow link.
_CHUNK = 1 << 16

#: A syntactically valid stand-in PDB id, for fetching EMDB entities when the caller gave
#: only an EMDB number. iotbx.pdb.fetch validates its ``id`` argument before looking at the
#: entity, and rejects anything whose first character is not 1-9 -- so the obvious "0000"
#: raises "Invalid pdb id" and makes an EMDB-only fetch impossible. The map URLs are built
#: from ``emdb_number`` alone, so the value is never used for anything else.
_EMDB_ONLY_PDB_ID = "1xxx"

#: entity -> what to call it when telling the user what is being fetched.
_LABELS = {
    "model": "model",
    "reflections": "structure factors",
    "map": "map",
    "half_map_1": "half-map 1",
    "half_map_2": "half-map 2",
}


class FetchError(RuntimeError):
    """A download failed, with an explanation the user can act on."""


#: Called as ``progress(entity, stage, done_bytes, total_bytes_or_None)`` during a fetch.
#: ``stage`` is "downloading", "decompressing", "cached" or "done". ``total`` is None when
#: the server does not send a Content-Length.
ProgressFn = Callable[[str, str, int, Optional[int]], None]


def describe(entity: str) -> str:
    """A human name for an entity key, for progress and error messages."""
    return _LABELS.get(entity, entity)


def format_bytes(n: Optional[int]) -> str:
    """``52035724`` -> ``"49.6 MB"``; None -> ``"?"``."""
    if n is None:
        return "?"
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < step or unit == "GB":
            return "%.0f %s" % (value, unit) if unit == "B" else "%.1f %s" % (value, unit)
        value /= step
    return "%.1f GB" % value


def _content_length(response) -> Optional[int]:
    """The declared size of a response, if it declares one.

    ``iotbx.pdb.fetch.fetch`` hands back the raw urlopen response for maps and a
    ``GzipFile`` wrapper for models; only the former carries headers, and for the latter
    the header would describe the compressed size anyway, not what we are about to write.
    """
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("Content-Length")
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def _stream_to_file(response, path: Path, *, entity: str, progress: Optional[ProgressFn]):
    """Copy ``response`` to ``path`` in chunks, reporting progress. Returns bytes written."""
    total = _content_length(response)
    done = 0
    if progress:
        progress(entity, "downloading", 0, total)
    with open(path, "wb") as handle:
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if progress:
                progress(entity, "downloading", done, total)
    return done

# Our entity keys -> (iotbx.pdb.fetch entity name, filename template). The templates use
# {pdb} and {emdb}; a model is fetched as mmCIF (more complete than the legacy PDB format).
_ENTITIES = {
    "model": ("model_cif", "{pdb}.cif"),
    "reflections": ("sf", "{pdb}-sf.cif"),
    "map": ("em_map", "emd_{emdb}.map"),
    "half_map_1": ("em_half_map_1", "emd_{emdb}_half_map_1.map"),
    "half_map_2": ("em_half_map_2", "emd_{emdb}_half_map_2.map"),
}
# Which entities are EMDB (need an emdb number) vs PDB (need a pdb id).
_EM_ENTITIES = {"map", "half_map_1", "half_map_2"}


def default_work_dir() -> Path:
    """Where downloads land unless the user picks somewhere else: ``~/pxviewer-data``
    (a data directory of its own, kept clear of any ``~/pxviewer`` source checkout)."""
    return Path.home() / "pxviewer-data"


def emdb_for_pdb(pdb_id: str, *, timeout: float = 30.0) -> Optional[str]:
    """The EMDB number a PDB entry's maps live under (e.g. ``"1234"``), or ``None``.

    Cryo-EM entries carry one or more ``EMD-####`` ids in their RCSB metadata; X-ray
    entries carry none. Returns the first, stripped to its digits. Any network or schema
    problem is swallowed into ``None`` — the caller can then ask the user for the number.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.strip().lower()}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            meta = json.load(response)
    except Exception:
        return None
    ids = (meta.get("rcsb_entry_container_identifiers") or {}).get("emdb_ids") or []
    if not ids:
        return None
    return str(ids[0]).split("-")[-1]  # "EMD-1234" -> "1234"


def reported_resolution(pdb_id: str, *, timeout: float = 30.0) -> Optional[float]:
    """The resolution an entry is deposited at, in Angstrom, or ``None``.

    Worth asking for because cctbx's own estimate can be well off, and a local-resolution
    calculation is clamped by it: ``local_resolution_map`` derives its ``d_min`` from
    ``map_model_manager.resolution()``, and no voxel can come out finer than that. On
    EMD-53478 the estimate is 7.7 A against a deposited 4.2 A, which floors the entire
    local-resolution map at 6.4 A and hides exactly the variation it is meant to show.

    Any network or schema problem is swallowed into ``None`` -- the caller falls back to
    cctbx's estimate, which is the old behaviour rather than a failure.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.strip().lower()}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            meta = json.load(response)
        values = (meta.get("rcsb_entry_info") or {}).get("resolution_combined") or []
        return float(values[0]) if values else None
    except Exception:
        return None


def recommended_contour(emdb_number: str, *, timeout: float = 30.0) -> Optional[float]:
    """The contour level an EMDB entry's authors recommend, or ``None``.

    A cryo-EM map's useful contour is a property of that map, not a constant: pxviewer's
    default of 1.5 sigma puts 2.5% of EMD-53478's voxels inside the surface, which draws
    the particle envelope wrapped in a haze of solvent noise rather than anything you can
    read. The deposited level is 3.7 sigma for the same map -- 0.57% of voxels -- and is
    what the authors intend the map to be looked at through.

    Returns the primary contour where the entry marks one, else the first. Any network or
    schema problem is swallowed into ``None``, leaving the caller with its own default.
    """
    number = str(emdb_number).strip().lstrip("EMDemd-") or str(emdb_number).strip()
    url = f"https://www.ebi.ac.uk/emdb/api/entry/EMD-{number}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            meta = json.load(response)
        contours = ((meta.get("map") or {}).get("contour_list") or {}).get("contour") or []
        if not contours:
            return None
        primary = next((c for c in contours if c.get("primary")), contours[0])
        level = primary.get("level")
        return float(level) if level is not None else None
    except Exception:
        return None


def fetch_entry(
    *,
    entities: Iterable[str],
    work_dir,
    pdb_id: Optional[str] = None,
    emdb_number: Optional[str] = None,
    mirror: str = "rcsb",
    log=None,
    progress: Optional[ProgressFn] = None,
    reuse_existing: bool = False,
) -> Dict[str, Path]:
    """Download ``entities`` for an entry into ``work_dir``; return ``{entity: Path}``.

    ``entities`` is any subset of ``model``, ``reflections``, ``map``, ``half_map_1``,
    ``half_map_2``. Model/reflections need ``pdb_id``; the maps need ``emdb_number`` — if
    that is not given but a ``pdb_id`` is, it is looked up (see :func:`emdb_for_pdb`).

    Missing prerequisites raise ``ValueError`` before anything is downloaded. Each map is
    gunzipped to a plain ``.map`` so cctbx can read it directly.

    ``progress`` is called as ``progress(entity, stage, done, total)`` while each file is
    fetched; see :data:`ProgressFn`. ``reuse_existing`` keeps a file that is already in
    ``work_dir`` instead of downloading it again -- worth setting for the 100 MB of
    half-maps a local-resolution run needs, and safe because a partial download lives on
    a ``.part`` and is only renamed into place once complete. The default stays
    "overwrite", so a plain re-fetch is still a deliberate refresh.

    Any download failure is raised as :class:`FetchError` naming the entry, the file and
    the URL.
    """
    import iotbx.pdb.fetch as F

    entities = [e for e in entities if e in _ENTITIES]
    if not entities:
        raise ValueError("nothing selected to fetch")

    wants_pdb = any(e not in _EM_ENTITIES for e in entities)
    wants_em = any(e in _EM_ENTITIES for e in entities)
    if wants_pdb and not pdb_id:
        raise ValueError("a PDB id is needed for the model or reflections")
    if wants_em:
        if not emdb_number and pdb_id:
            emdb_number = emdb_for_pdb(pdb_id)
        if not emdb_number:
            raise ValueError(
                "an EMDB number is needed for the map or half-maps "
                "(none is associated with this PDB id — enter it directly)")

    pdb_id = (pdb_id or "").strip().lower()
    emdb_number = (str(emdb_number).strip() if emdb_number else None)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Path] = {}
    for entity in entities:
        iotbx_name, filename_tpl = _ENTITIES[entity]
        filename = filename_tpl.format(pdb=pdb_id, emdb=emdb_number)
        path = work_dir / filename

        if reuse_existing and path.is_file() and path.stat().st_size > 0:
            if progress:
                size = path.stat().st_size
                progress(entity, "cached", size, size)
            if log is not None:
                print(f"reusing {entity} <- {path}", file=log)
            out[entity] = path
            continue

        url = None
        try:
            url = F.get_link(mirror, iotbx_name,
                             pdb_id=pdb_id or _EMDB_ONLY_PDB_ID,
                             emdb_number=emdb_number)
        except Exception:
            pass  # only used to make an error message better

        part = path.with_name(path.name + ".part")
        try:
            response = F.fetch(pdb_id or _EMDB_ONLY_PDB_ID, entity=iotbx_name,
                               mirror=mirror, emdb_number=emdb_number)
            _stream_to_file(response, part, entity=entity, progress=progress)
            # fetch() decompresses model/sf but leaves maps gzipped (see its source), so
            # the maps are the only thing we gunzip here. Streamed rather than held in
            # memory: a 256^3 map is ~64 MB decompressed and there are three of them.
            if entity in _EM_ENTITIES:
                if progress:
                    progress(entity, "decompressing", 0, None)
                unpacked = path.with_name(path.name + ".part2")
                try:
                    with gzip.open(part, "rb") as src, open(unpacked, "wb") as dst:
                        shutil.copyfileobj(src, dst, _CHUNK)
                finally:
                    part.unlink(missing_ok=True)
                part = unpacked
            part.replace(path)   # atomic: a .part is never mistaken for a finished file
        except Exception as exc:   # network, HTTP, gzip and cctbx's Sorry all land here
            part.unlink(missing_ok=True)
            path.with_name(path.name + ".part2").unlink(missing_ok=True)
            if isinstance(exc, FetchError):
                raise
            raise FetchError(_failure_message(entity, exc, url=url, pdb_id=pdb_id,
                                              emdb_number=emdb_number)) from exc

        if progress:
            size = path.stat().st_size
            progress(entity, "done", size, size)
        if log is not None:
            print(f"fetched {entity} -> {path}", file=log)
        out[entity] = path
    return out


def _failure_message(entity, exc, *, url=None, pdb_id=None, emdb_number=None) -> str:
    """Turn whatever the network raised into something with a next step in it."""
    what = describe(entity)
    where = f"{pdb_id.upper()}" if pdb_id and entity not in _EM_ENTITIES else (
        f"EMD-{emdb_number}" if emdb_number else "the entry")
    detail = str(exc).strip() or exc.__class__.__name__

    hint = "check the id and your network connection, then try again"
    code = getattr(exc, "code", None)
    if code in (403, 404):
        hint = (f"the server has no {what} for {where} — check the id, or that this entry "
                f"really has one (half-maps are only deposited for some cryo-EM entries)")
    elif isinstance(exc, urllib.error.URLError) or isinstance(exc, TimeoutError):
        hint = "the download could not reach the server — check your network and retry"
    elif isinstance(exc, OSError):
        hint = "the file could not be written — check free space and the working directory"

    return (f"could not fetch the {what} for {where}: {detail}\n"
            + (f"  url: {url}\n" if url else "") + f"  {hint}")
