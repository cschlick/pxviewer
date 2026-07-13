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
| `select` | atoms highlighted by index, cycling through subsets |
| `primitives` | angle/distance/dihedral/label measurements tracking a flexing chain |

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

### Selecting atoms (by index)

Atoms are addressed by **positional index** — the row in the topology's
`_atom_site` table, the same stable key the whole live protocol uses. Build a
`Selection` from indices, atom ids, or a boolean mask (all pure Python, no viewer
needed), then show it:

```python
sel = session.select_by(indices=range(20))       # positional rows 0..19
sel = session.select_by(ids=[10, 12, 14])         # by Atom.id
sel = session.select_by(mask=my_bool_array)       # numpy mask of length N
sel.indices   # [0, 1, 2, ...]   sel.atoms   sel.ids   sel.mask

session.highlight(sel)                 # selection overlay
session.focus([4, 5, 6])               # aim the camera (indices coerced to a Selection)
session.select(sel, focus=False)       # compose highlight + focus (both on by default)
session.clear_selection()              # remove the highlight
```

`highlight`/`focus`/`select` accept a `Selection` or anything coercible — an
index, a list of indices, or a boolean mask. Resolution is entirely on the Python
side, so these are **synchronous and viewer-independent** (no round-trip); the
wire carries only indices, run-length-encoded for large contiguous selections.
Highlights re-map onto each streamed frame in O(selected), and are replayed to
viewers that connect later.

### Selecting atoms with the mouse

Let the person at the viewer pick atoms and read their choice back in Python.
Enable click-to-select, then **click** an atom (**shift-click** to add or remove
more); the running selection streams back:

```python
session.enable_mouse_selection()

sel = session.wait_for_selection()      # block until the user clicks; returns a Selection
print(sel.indices, sel.ids)

# …or react to every change:
session.enable_mouse_selection(on_change=lambda sel: print("picked", sel.indices))

session.mouse_selection                  # the current pick set, at any time
session.disable_mouse_selection()
```

Click selects a single atom; shift-click toggles atoms in and out; clicking empty
space clears. Picked atoms are highlighted in the viewer and reported as positional
indices — the same `Selection` you can hand straight to `highlight`, `add_angle`,
and friends.

### Drawing measurements (angles, distances, dihedrals, labels)

Draw Mol\*'s measurement graphics from Python. Atoms are named by a `Selection`
(`select_by(indices=…)` / `select_by(ids=…)` / `select_by(mask=…)`) — or anything
coercible: an index, a list of indices, or a boolean mask. A multi-atom group is
reduced to its **centroid**, so these also work between groups. Each primitive
**tracks the atoms as they move**.

```python
a = session.select_by(indices=[0]); b = session.select_by(ids=[5]); c = session.select_by(indices=[9])

ang  = session.add_angle(a, b, c)              # thin pie-shaped wedge at vertex b
dist = session.add_distance(a, b)              # dashed line + distance
dih  = session.add_dihedral(a, b, c, "resi 7") # torsion across b–c
lbl  = session.add_label(a, "active site")     # floating text

ang.degrees      # measured angle, computed in Python from current coords
dist.distance    # measured distance in Å
session.remove_primitive(ang.id)               # remove one
session.clear_primitives()                     # remove all
```

`add_angle`/`add_dihedral` take `opacity=` (wedge translucency) and `label=`
(toggle the value text). Every `add_*` returns a `Primitive` with an `id` (for
removal), the measured `value` (degrees / Å / `None`), and its `selections`.
Primitives are replayed to viewers that connect later.

## Live wire protocol (`pxviewer-live/1`)

WebSocket; binary messages are little-endian and begin with a `uint32` tag.

| Direction | Kind | Layout |
| --- | --- | --- |
| server → client | topology | `[u32 tag=0][BinaryCIF bytes]` (sent once on connect) |
| server → client | frame | `[u32 tag=1][u32 frameIndex][f32 × 3N]` interleaved `x,y,z` |
| server → client | highlight | JSON `{"type":"highlight","atoms":<index-set>}` (empty clears) |
| server → client | focus | JSON `{"type":"focus","atoms":<index-set>}` |
| server → client | primitive | JSON `{"type":"primitive","action":"add"\|"remove"\|"clear","kind":…,"id":str,"groups":[[int…]…],"options":{…}}` |
| server → client | mouse-selection-mode | JSON `{"type":"mouse-selection-mode","enabled":bool}` |
| client → server | ready | JSON `{"type":"ready"}` |
| client → server | pick | JSON `{"type":"pick","empty":bool,"atom":{id,name,resname,resseq,chain}}` |
| client → server | mouse-selection | JSON `{"type":"mouse-selection","indices":[int…]}` |

An `<index-set>` is `{"list":[int,…]}` or run-length `{"runs":[[start,end],…]}`.
All atom addressing is by positional index; the wire carries no query language.

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
