# pxviewer

A custom molecular viewer built on top of [Mol*](https://molstar.org) with a Python-driven scene API and a **live coordinate bridge** for in-place updates.

## Structure

- `frontend/` — TypeScript/React frontend using the `molstar` React plugin helpers.
- `python/` — Python package: `molviewspec` to build MVS scenes, `ciftools` to write BinaryCIF, and a `LiveSession` WebSocket server that streams coordinates.

## Quick start

### Frontend

```bash
cd frontend
npm install
npm run build
# serve this directory over http, then open index.html
```

### Python

```bash
cd python
pip install -e '.[live]'      # 'live' pulls in websockets for streaming
```

```python
import pxviewer
mvsj = pxviewer.create_example_view("https://www.ebi.ac.uk/pdbe/entry-files/1cbs.bcif")
print(mvsj)
```

## Live coordinate updates

The live bridge streams a **fixed topology once**, then streams **coordinate-only
frames** that Mol* applies as close to in-place as its data model allows (Level 1:
the parsed topology — hierarchy and bonds — is reused, and only the conformation is
swapped per frame, so representations do a coordinate update rather than a rebuild).

### Try the demo

```bash
# terminal 1 — stream an oscillating structure
cd python
python -m pxviewer serve-demo            # ws://127.0.0.1:8787

# terminal 2 — serve the frontend, then open:
#   index.html?ws=ws://127.0.0.1:8787
```

Clicking an atom prints its identity back in the Python terminal (the duplex path).

### From your own code

```python
from pxviewer import Atom, LiveSession

atoms = [Atom(id=i + 1, element="C", x=float(i), y=0.0, z=0.0) for i in range(10)]
session = LiveSession(atoms)
session.on_pick(lambda info: print("picked", info))
session.start()                          # background thread, ws://127.0.0.1:8787

for coords in my_trajectory:             # coords: (N, 3) array-like, topology order
    session.push(coords)
```

Open the frontend at `index.html?ws=ws://127.0.0.1:8787`. With no `?ws=` param the
page falls back to loading a static PDB.

## Live wire protocol (`pxviewer-live/1`)

WebSocket; binary messages are little-endian and begin with a `uint32` tag.

| Direction | Kind | Layout |
| --- | --- | --- |
| server → client | topology | `[u32 tag=0][BinaryCIF bytes]` (sent once on connect) |
| server → client | frame | `[u32 tag=1][u32 frameIndex][f32 × 3N]` interleaved `x,y,z` |
| client → server | ready | JSON `{"type":"ready"}` |
| client → server | pick | JSON `{"type":"pick","empty":bool,"atom":{id,name,resname,resseq,chain}}` |

### Atom-identity contract

Coordinate frames are **positional**: value triple *i* always refers to the same atom
as row *i* of the topology's `_atom_site` table. Therefore:

- The atom count is fixed for a session; a mismatched frame is rejected, not
  silently mis-assigned.
- Atoms may not be added, removed, or reordered mid-stream — start a new session to
  change the atom set.
- Per-atom identity (`id/name/resname/resseq/chain`) lives only in the topology and
  is never resent; pick events reference atoms by that stable identity.

Read `pxviewer.ATOM_IDENTITY_CONTRACT` for the authoritative statement.

## Next steps

- Batch/queue frames on the client to cap update rate under fast producers.
- Stream a compact selection channel (server → client) so Python can drive highlights.
- Custom GPU visual (Level 3) that re-uploads only a position buffer, for large N.
- Quantize coordinates on the wire (e.g. fixed-point) to cut bandwidth.
