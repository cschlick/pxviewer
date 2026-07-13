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

Build the frontend once (`cd frontend && npm install && npm run build`), then:

```bash
cd python
python -m pxviewer serve-demo            # prints a ready-to-click http:// URL
```

The command serves **both** the WebSocket stream and the frontend page, and prints
a single `http://127.0.0.1:5173/` URL. Open it — the page connects the WebSocket for
you. (Don't open the `ws://` address in a browser directly; it's a WebSocket
endpoint, not a web page, and will just tell you so.)

Clicking an atom prints its identity back in the Python terminal (the duplex path).
Pass `--no-frontend` to stream only, or `--http-port` to change the page port.

### Demos

Narrated, slowed-down scenarios meant to be *watched* in the browser (like tests,
but for human eyes). List them, then run one and open the printed URL:

```bash
cd python
python -m pxviewer demo                   # list available demos
python -m pxviewer demo wave              # then open the http:// URL it prints
```

| Demo | What you see |
| --- | --- |
| `wave` | a chain rippling with a growing travelling wave |
| `breathe` | a sphere of atoms expanding and contracting |
| `orbit` | a rigid body gliding around a square path |
| `morph` | a chain folding into a helix and back |
| `pick` | click atoms to make them pulse — the scene → Python path |

Each demo serves the frontend, waits for the viewer to connect, narrates each step
in the terminal, and loops until Ctrl-C. Use `--fps` to change smoothness within a
motion, or `--no-frontend` to connect a page you're serving yourself.

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

### Selecting atoms (PyMOL syntax)

Drive the viewer's selection from Python using **PyMOL selection syntax**. The
expression is parsed and evaluated by Mol* in the browser (its `mol-script` PyMOL
transpiler), which echoes the matched atoms back — so a selection both *shows* in
the viewer and *returns* what it matched:

```python
sel = session.select("resi 1-20 and chain A")   # highlight + focus; returns a Selection
sel.indices      # [0, 1, 2, ...]  positional atom rows (the identity-contract key)
sel.atoms        # the matching Atom objects
sel.ids          # their _atom_site.id values
sel.mask         # boolean numpy array of length N — handy for coordinate math

session.highlight("elem O")             # just the selection overlay, no camera move
session.focus("id 5")                   # just aim the camera
session.select("name CA", focus=False)  # compose the primitives: highlight only
session.clear_selection()               # remove the highlight
```

`select` composes the `highlight` and `focus` primitives (both on by default).
Each call blocks briefly for the viewer's echo (`timeout=`, default 5 s) and
returns a `Selection`, or `None` if no viewer answered. Supported selectors
include `name`, `elem`, `resn`, `resi`, `chain`, `id`, and `index`, with boolean
`and`/`or`/`not`, parentheses, ranges (`resi 1-10`) and lists (`resi 1+2+3`).

## Live wire protocol (`pxviewer-live/1`)

WebSocket; binary messages are little-endian and begin with a `uint32` tag.

| Direction | Kind | Layout |
| --- | --- | --- |
| server → client | topology | `[u32 tag=0][BinaryCIF bytes]` (sent once on connect) |
| server → client | frame | `[u32 tag=1][u32 frameIndex][f32 × 3N]` interleaved `x,y,z` |
| server → client | select | JSON `{"type":"select","reqId":int,"expression":str,"highlight":bool,"focus":bool}` |
| client → server | ready | JSON `{"type":"ready"}` |
| client → server | pick | JSON `{"type":"pick","empty":bool,"atom":{id,name,resname,resseq,chain}}` |
| client → server | selection-result | JSON `{"type":"selection-result","reqId":int,"indices":[int…],"error":str?}` |

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
- Echo large selections back over a compact binary channel (currently JSON indices).
- Custom GPU visual (Level 3) that re-uploads only a position buffer, for large N.
- Quantize coordinates on the wire (e.g. fixed-point) to cut bandwidth.
