"""A minimal, self-contained BinaryCIF encoder and decoder.

BinaryCIF (https://github.com/molstar/BinaryCIF) is the compact binary form of CIF
that Mol* reads: a MessagePack document wrapping a set of CIF categories, each column
a typed array put through a chain of reversible *encodings*. The full format defines
seven encodings, but six of them (Delta, RunLength, IntegerPacking, FixedPoint, …)
only compress — a decoder reconstructs the same values whether or not they are used.
So this module implements just the two *terminal* encodings, which is all you need to
write a valid file:

  - **ByteArray** — a numeric column as raw little-endian bytes plus a type code.
  - **StringArray** — a string column as the unique strings concatenated, an
    ``offsets`` array delimiting them, and a per-row ``indices`` array. It matches the
    Mol* decoder exactly: strings are rebuilt as ``['', u0, u1, …]`` and each row is
    read as ``strings[index + 1]``, so indices are 0-based into the unique list and
    ``offsets`` has one more entry than there are unique strings.

The document layout is::

    {version, encoder, dataBlocks: [{header, categories: [
        {name, rowCount, columns: [{name, data: {data, encoding}, mask}]}]}]}

Depends only on numpy and msgpack — copy this file into any project as-is.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import msgpack
import numpy as np

__all__ = [
    "INT8", "INT16", "INT32", "UINT8", "UINT16", "UINT32", "FLOAT32", "FLOAT64",
    "number_column", "string_column", "category", "encode", "decode",
]

# BinaryCIF data-type codes (Mol* Encoding.DataType).
INT8, INT16, INT32 = 1, 2, 3
UINT8, UINT16, UINT32 = 4, 5, 6
FLOAT32, FLOAT64 = 32, 33

#: type code -> explicit little-endian numpy dtype. BinaryCIF is little-endian.
_DTYPE: Dict[int, str] = {
    INT8: "<i1", INT16: "<i2", INT32: "<i4",
    UINT8: "<u1", UINT16: "<u2", UINT32: "<u4",
    FLOAT32: "<f4", FLOAT64: "<f8",
}


# -- encoding ----------------------------------------------------------------

def _byte_array(array: Any, type_code: int) -> Dict[str, Any]:
    """A typed array as a ByteArray-encoded ``{data, encoding}`` block."""
    arr = np.ascontiguousarray(array, dtype=_DTYPE[type_code])
    return {"data": arr.tobytes(), "encoding": [{"kind": "ByteArray", "type": type_code}]}


def number_column(name: str, array: Any, type_code: int = FLOAT32) -> Dict[str, Any]:
    """A numeric column. ``array`` is anything array-like; ``type_code`` picks the wire
    type (default float32). No mask — the column is treated as fully present."""
    return {"name": name, "data": _byte_array(array, type_code), "mask": None}


def string_column(name: str, values: Iterable[Any]) -> Dict[str, Any]:
    """A string column, StringArray-encoded.

    Builds the unique-string table in first-appearance order, the per-row 0-based
    indices into it, and the character offsets delimiting the uniques within the
    concatenated ``stringData``. ``None`` becomes the empty string.
    """
    items = ["" if v is None else str(v) for v in values]
    uniques: List[str] = []
    index_of: Dict[str, int] = {}
    indices = np.empty(len(items), dtype="<i4")
    for i, s in enumerate(items):
        j = index_of.get(s)
        if j is None:
            j = len(uniques)
            index_of[s] = j
            uniques.append(s)
        indices[i] = j

    offsets = np.zeros(len(uniques) + 1, dtype="<i4")
    acc = 0
    for k, s in enumerate(uniques):
        acc += len(s)
        offsets[k + 1] = acc

    encoding = {
        "kind": "StringArray",
        "dataEncoding": [{"kind": "ByteArray", "type": INT32}],
        "stringData": "".join(uniques),
        "offsetEncoding": [{"kind": "ByteArray", "type": INT32}],
        "offsets": offsets.tobytes(),
    }
    return {"name": name,
            "data": {"data": indices.tobytes(), "encoding": [encoding]},
            "mask": None}


def category(name: str, row_count: int, columns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One CIF category. ``name`` gets a leading underscore if it lacks one."""
    if not name.startswith("_"):
        name = "_" + name
    return {"name": name, "rowCount": int(row_count), "columns": list(columns)}


def encode(header: str, categories: Sequence[Dict[str, Any]], *,
           encoder: str = "pxviewer-bcif") -> bytes:
    """Serialise one data block of ``categories`` to BinaryCIF bytes."""
    doc = {
        "version": "0.3.0",
        "encoder": encoder,
        "dataBlocks": [{"header": header, "categories": list(categories)}],
    }
    return msgpack.packb(doc, use_bin_type=True)


# -- decoding (for round-trip tests; handles only what this module writes) ----

def _apply(step: Dict[str, Any], data: bytes) -> List[Any]:
    kind = step["kind"]
    if kind == "ByteArray":
        return np.frombuffer(data, dtype=_DTYPE[step["type"]]).tolist()
    if kind == "StringArray":
        offsets = _decode(step["offsets"], step["offsetEncoding"])
        indices = _decode(data, step["dataEncoding"])
        s = step["stringData"]
        strings = [""]
        for i in range(1, len(offsets)):
            strings.append(s[offsets[i - 1]:offsets[i]])
        return [strings[idx + 1] for idx in indices]
    raise ValueError(f"unsupported encoding {kind!r}")


def _decode(data: bytes, encoding: Sequence[Dict[str, Any]]) -> List[Any]:
    """Apply an encoding chain (outermost first) to raw bytes."""
    result: Any = data
    for step in reversed(list(encoding)):
        result = _apply(step, result)
    return result


def decode(data: bytes) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
    """Decode BinaryCIF bytes to ``{block_header: {category: {column: [values]}}}``.

    Understands only the ByteArray and StringArray encodings this module emits, which
    is enough to round-trip anything :func:`encode` produced.
    """
    doc = msgpack.unpackb(data, raw=False)
    out: Dict[str, Any] = {}
    for block in doc["dataBlocks"]:
        cats: Dict[str, Any] = {}
        for cat in block["categories"]:
            cats[cat["name"]] = {
                col["name"]: _decode(col["data"]["data"], col["data"]["encoding"])
                for col in cat["columns"]
            }
        out[block["header"]] = cats
    return out
