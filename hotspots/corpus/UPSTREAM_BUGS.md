# Upstream bugs found during corpus runs

Defects in cctbx/Phenix, not in this directory. Recorded here so they can be fixed or
reported upstream rather than rediscovered.

---

## 1. `probe2._describe_atom_for_debug` crashes on an atom with no parent

**Status:** **fixed and pushed 2026-08-04** to `github.com/cschlick/cctbx_project` —
committed on branch `probe2_parentless_atom` (`6f4056b31f`) and merged into `cschlick-dev`
(`4deeeae817`), both pushed. Not yet sent to the upstream cctbx project. Found 2026-08-02 in
the figure run.
**Component:** `cctbx_project/mmtbx/programs/probe2.py`
**Severity:** low impact, trivial fix — but it converts a clear, actionable error message
into a confusing unrelated one, which is how it cost time here.

### What happens

probe2 detects a genuine data problem — a hydrogen with no bonded neighbour — and raises
`Sorry(...)` to report it. Building that message calls `_describe_atom_for_debug`, which
dereferences the atom's parent without checking it exists. For the offending atom the parent
is `None`, so the error *reporter* raises `AttributeError` and the real diagnostic is lost:

```
File "mmtbx/programs/probe2.py", line 2010, in run
    raise Sorry("Found Hydrogen with no neigbors: " + self._describe_atom_for_debug(a))
File "mmtbx/programs/probe2.py", line 704, in _describe_atom_for_debug
    resName = a.parent().resname.strip().upper()
AttributeError: 'NoneType' object has no attribute 'resname'
```

The caller sees `AttributeError: 'NoneType' object has no attribute 'resname'` — which
suggests a bug in the calling code — instead of "Found Hydrogen with no neighbors", which
names the actual data problem and the residue it is in.

### Reproducer

```bash
source /root/phenix/build/setpaths.sh
libtbx.python -c "
import sys; sys.path.insert(0, 'hotspots')
from events import _load_shared, load_model, add_hydrogens
ve = _load_shared()
m = load_model('/root/data/pdb_mmcif/rv/4rvn.cif.gz')
ve.probe2_dots(add_hydrogens(m))
"
```

`4rvn` (14,520 atoms). It is the only structure of 2,000 that hit this path — the same
underlying "hydrogen with no neighbours" condition occurred 19 other times and reported
itself correctly, so the trigger is the missing parent, not the hydrogen condition.

### Suggested fix

Guard the parent lookups in `_describe_atom_for_debug` (line ~704) so the debug describer
degrades instead of raising — an error formatter should never be able to mask the error it
is formatting:

```python
ag = a.parent()
resName = ag.resname.strip().upper() if ag is not None else "???"
```

with the same treatment for any `ag.parent()` chain-level lookups in that function.

### Why an atom has no parent here — now partly answered

With the guard applied, `4rvn` reports the diagnostic it always should have:

```
Sorry: Found Hydrogen with no neigbors:  ?   ? ???  H2
```

**Every field is a placeholder.** Not just the atom group is missing — the residue group and
chain are gone too, so the atom named `H2` reaches probe2 completely detached from the
hierarchy rather than merely orphaned one level up. That is a stronger result than expected
and supports the suspicion above: something in H placement is emitting an atom that was never
linked in. The guard makes the error legible; it does not make the atom correct.

**So this is still worth chasing.** Next step is to check whether `reduce2` produced `H2` or
whether it survived from the deposited model, and whether probe2 should be rejecting a
detached atom earlier and more explicitly than at a debug-string call site.
