"""Fetch models, maps, half-maps and reflections from the PDB/EMDB.

A thin wrapper over cctbx's own :mod:`iotbx.pdb.fetch`: it downloads the requested
pieces of an entry into a working directory and hands back the file paths, so the rest
of the app can load them like any other files. Cryo-EM maps are served gzipped; those are
decompressed on the way in (models and structure factors arrive already decompressed).

The one thing cctbx's fetch does not do is turn a PDB id into the EMDB number its maps
live under — that mapping is looked up from the RCSB REST API (standard library only).
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

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


def fetch_entry(
    *,
    entities: Iterable[str],
    work_dir,
    pdb_id: Optional[str] = None,
    emdb_number: Optional[str] = None,
    mirror: str = "rcsb",
    log=None,
) -> Dict[str, Path]:
    """Download ``entities`` for an entry into ``work_dir``; return ``{entity: Path}``.

    ``entities`` is any subset of ``model``, ``reflections``, ``map``, ``half_map_1``,
    ``half_map_2``. Model/reflections need ``pdb_id``; the maps need ``emdb_number`` — if
    that is not given but a ``pdb_id`` is, it is looked up (see :func:`emdb_for_pdb`).

    Missing prerequisites raise ``ValueError`` before anything is downloaded. Each map is
    gunzipped to a plain ``.map`` so cctbx can read it directly. Files already present are
    overwritten (a re-fetch is a deliberate refresh).
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
        data = F.fetch(pdb_id or "0000", entity=iotbx_name, mirror=mirror,
                       emdb_number=emdb_number)
        raw = data.read()
        # fetch() decompresses model/sf but leaves maps gzipped (see its source), so the
        # maps are the only thing we gunzip here.
        if entity in _EM_ENTITIES:
            raw = gzip.decompress(raw)
        with open(path, "wb") as handle:
            handle.write(raw)
        if log is not None:
            print(f"fetched {entity} -> {path}", file=log)
        out[entity] = path
    return out
