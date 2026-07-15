# pxviewer

A [Mol*](https://molstar.org)-based viewer for **cctbx objects**: atomic models are
read in Python with cctbx's `DataManager` into an `mmtbx.model.manager`, and
everything the viewer shows is derived from that model's `pdb_hierarchy` and streamed
over a **live coordinate bridge**. Nothing is parsed in the browser — all I/O goes
through cctbx, then maps onto Mol\*'s data on the Python side.

## Structure

- `frontend/` — TypeScript/React frontend using the `molstar` React plugin helpers.
- `python/` — Python package: `cctbx` for model I/O, `ciftools` to write BinaryCIF, `molviewspec` for volume scenes, and a `LiveSession` WebSocket server that streams the topology and coordinates.

## Quick start

### Environment (conda)

Model I/O needs **cctbx**, which ships only on conda-forge (not PyPI), so pxviewer
installs via conda:

```bash
conda env create -f environment.yml   # python, cctbx-base, PySide6, websockets, …
conda activate pxviewer
pip install -e ./python                # the pxviewer package itself
```

(cctbx pins numpy ≤ 2.4, matching numba/ciftools — `environment.yml` handles this.)

### Frontend

```bash
cd frontend
npm install
npm run build
# serve this directory over http, then open index.html
```

### Load a model

```bash
cd python
python -m pxviewer model /path/to/model.pdb   # cctbx reads it; prints a http:// URL
```

Or from code:

```python
from pxviewer.live import LiveSession
session = LiveSession.from_model_file("model.pdb")   # cctbx DataManager -> hierarchy
session.start()                                       # ws://127.0.0.1:8787
# ... open the frontend with ?ws=<that url> ...
```

`LiveSession.from_cctbx_model(model)` builds a session from an existing
`mmtbx.model.manager`; `pxviewer.cctbx_io.model_to_arrays(model)` exposes the
hierarchy's vectorised columns (xyz, element, name, residue/chain labels, B, occ)
if you want the mapping directly.

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
| `interactions` | explicit typed non-covalent contacts stretching as two strands separate |
| `clashes` | steric clashes lighting up red as two clusters interpenetrate |
| `measure` | click atoms to measure distances/angles/dihedrals — the scene → Python path |

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

`Selection` is built on **MolViewSpec**'s `ComponentExpression` — you can select
by chemical identity (chain, residue, atom name, element, ranges, `atom_index`,
`atom_id`) using the same type MVS uses, and pxviewer resolves it to indices:

```python
from pxviewer import ComponentExpression as CE

sel = session.select_by(expression=CE(label_asym_id="A", beg_label_seq_id=1, end_label_seq_id=20))
sel = session.select_by(expression=[CE(label_asym_id="B"), CE(type_symbol="N")])   # a union
session.highlight(CE(label_atom_id="CA"))                                          # coerced like any spec
sel.to_component_expression()                                                       # -> [ComponentExpression, …]
```

The index/id/mask constructors are sugar on top; `ComponentExpression` is the
shared Mol\*/MVS vocabulary.

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

**Measure modes** are a distinct click mode: instead of building a selection,
clicking atoms *draws a measurement*. Click the required number of atoms and the
primitive appears (and tracks the atoms), with the result reported back:

```python
session.enable_measure_mode("angle", on_measure=lambda p: print(p.degrees))
# click 3 atoms -> an angle wedge is drawn
```

`kind` is `"distance"` (2 clicks), `"angle"` (3), `"dihedral"` (4), or `"label"`
(1). Modes are mutually exclusive — `enable_mouse_selection`, `enable_measure_mode`,
and `disable_mouse_selection` switch between select / measure / off.

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

### Representations

Control how the structure is drawn, on the whole structure or a subset:

```python
session.set_representation("cartoon", color="secondary-structure")   # replace with one
rid = session.add_representation("ball_and_stick", color="element-symbol",
                                 on=session.select_by(ids=[101, 102, 103]))  # a subset
session.add_representation("spacefill", color_value="orange", opacity=0.6)   # flat colour
session.remove_representation(rid)
session.clear_representations()                                       # back to default
```

- **`type`** — MolViewSpec's representation types: `ball_and_stick`, `spacefill`
  (alias `sphere`), `cartoon` (alias `ribbon`), `surface`, `carbohydrate`.
- **`color`** — a **uniform** colour (an SVG name like `orange`, or `#ff8800`), or
  a Mol\* **colour theme** name (`element-symbol`, `chain-id`,
  `secondary-structure`, `residue-name`, `hydrophobicity`, …). `color_value=` also
  forces a uniform colour. (Uniform colours are MVS's native `ColorT`; themes are
  the Mol\* colouring mechanism layered on top.)
- **`on`** — a `Selection`, an MVS `ComponentExpression`, or anything coercible, to
  restrict to a subset; omit for the whole structure. `opacity=` and a `params=`
  passthrough are also available.

Representations **track the streamed coordinates**, and the current set is
replayed to viewers that connect later. If you never set any, you get the default
ball-and-stick / element-symbol.

### Non-covalent interactions

Two ways to draw non-covalent (non-bonded) interaction notation — dashed
cylinders, coloured by kind.

**Explicit — you supply the contacts.** Give a typed table of atom-index pairs;
nothing is inferred. This is the usual path when Python owns the atoms (a live
session):

```python
session.set_interactions({
    "hydrogen-bond": [(0, 1), (5, 6)],
    "salt-bridge":   [(3, 8)],       # alias for "ionic"
    "hydrophobic":   [(10, 12)],
})
session.clear_interactions()
```

You can also pass `(kind, a, b)` tuples or `{"kind","a","b","description"}` dicts.
Atom indices are **positional** (the same 0-based identity the rest of the live
protocol uses); an out-of-range index or unknown kind raises `ValueError`. Kinds:
`hydrogen-bond`, `weak-hydrogen-bond`, `ionic`, `hydrophobic`, `pi-stacking`,
`cation-pi`, `halogen-bond`, `metal-coordination`, `water-bridge`, `covalent`,
`unknown` (with aliases like `h-bond`, `salt-bridge`). Because the contacts
reference atoms rather than fixed points, their endpoints **track the streamed
coordinates**. The set is replayed to viewers that connect later. (See the
`interactions` demo.)

**Computed — Mol\* infers them.** When Python doesn't have a contact table — e.g.
a structure loaded and parsed entirely in the browser — let Mol\* compute the
contacts on every structure in the scene:

```python
session.set_computed_interactions(True)   # or show_/hide_computed_interactions()
```

In the desktop app the **Show computed interactions** button under *Display*
toggles this.

### Clashes

Steric clashes — non-bonded atoms overlapping in van der Waals space — drawn as
distinct **red** markers, so they read as "bad contact" next to the interaction
notation. Mol\* has no general clash detector (its only clash support ingests
RCSB validation reports), so pxviewer computes them from the coordinates you own:

```python
pairs = session.detect_clashes(tolerance=0.4)   # vdW overlap, excluding bonds
session.set_clashes(pairs)                       # draw them
session.show_clashes(tolerance=0.4)              # detect + draw in one call
session.clear_clashes()
```

`detect_clashes` flags a pair when its separation is below ``vdw_i + vdw_j -
tolerance`` but above ``cov_i + cov_j + bond_tolerance`` (covalent bonds are not
clashes); radii are looked up by element. You can also pass explicit pairs to
`set_clashes([(i, j), …])` if you compute clashes your own way. Like interactions,
the markers reference atoms, so they **track streamed coordinates** — re-run
`detect_clashes`/`set_clashes` as the structure moves to keep them current (see
the `clashes` demo). This is a live-session feature: it needs the coordinates and
elements Python holds, so it doesn't apply to a structure parsed only in the
browser.

### Secondary structure (for cartoon / ribbon)

Cartoon rendering and the `secondary-structure` color theme need a **polymer**
with secondary-structure assignment. Declare both at session construction — this
puts *your* SS into the topology (`_struct_conf` / `_struct_sheet_range`), so the
ribbon reflects your algorithm rather than Mol\*'s built-in DSSP:

```python
session = LiveSession(
    atoms,                                   # a protein: residues with backbone atoms
    secondary_structure=[                    # (chain, beg_resseq, end_resseq, kind)
        ("A", 1, 14, "helix"),
        ("A", 20, 28, "sheet"),
    ],
)
session.set_representation("cartoon", color="secondary-structure")
```

`secondary_structure=` implies `polymer=True` (you can also pass `polymer=True`
on its own). It's topology-time — sent once with the structure; to change SS,
start a new session.

## Live wire protocol (`pxviewer-live/1`)

WebSocket; binary messages are little-endian and begin with a `uint32` tag.

| Direction | Kind | Layout |
| --- | --- | --- |
| server → client | topology | `[u32 tag=0][BinaryCIF bytes]` (sent once on connect) |
| server → client | frame | `[u32 tag=1][u32 frameIndex][f32 × 3N]` interleaved `x,y,z` |
| server → client | highlight | JSON `{"type":"highlight","atoms":<index-set>}` (empty clears) |
| server → client | focus | JSON `{"type":"focus","atoms":<index-set>}` |
| server → client | primitive | JSON `{"type":"primitive","action":"add"\|"remove"\|"clear","kind":…,"id":str,"groups":[[int…]…],"options":{…}}` |
| server → client | representations | JSON `{"type":"representations","reprs":[{id,type,color?,colorValue?,on?,opacity?,params?},…]}` |
| server → client | interactions | JSON `{"type":"interactions","action":"set","contacts":[{kind,a,b,description?},…]}` or `{"type":"interactions","action":"clear"}` (explicit typed contacts) |
| server → client | computed-interactions | JSON `{"type":"computed-interactions","visible":bool}` (Mol\*-inferred contacts) |
| server → client | clashes | JSON `{"type":"clashes","action":"set","pairs":[{a,b},…]}` or `{"type":"clashes","action":"clear"}` (steric clashes, drawn red) |
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
