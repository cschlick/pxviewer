"""Read atomic models with cctbx and map them onto pxviewer's transfer format.

pxviewer treats cctbx as the single source of truth for model I/O: a file is read
by cctbx's :class:`~iotbx.data_manager.DataManager` into an
``mmtbx.model.manager``, and everything the viewer shows is derived from that
model's ``pdb_hierarchy`` — no structure is ever parsed in the browser.

The hierarchy exposes its columns as vectorized arrays (``extract_xyz`` etc.), so
we lift them straight into an :class:`~pxviewer.data.AtomArrays` and hand that to
the BinaryCIF encoder, mapping cctbx to the wire with no per-atom Python on the
hot path (only the residue/chain labels need a single ordered walk).

cctbx is imported lazily inside the functions so ``import pxviewer`` stays fast
and the rest of the package works without cctbx installed.
"""

from __future__ import annotations

import itertools
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data import AtomArrays

__all__ = [
    "ModelData",
    "data_manager",
    "read_model",
    "first_model",
    "model_to_arrays",
    "model_secondary_structure",
    "model_is_polymer",
    "load_model",
    "model_from_sites",
    "custom_atom_site_attributes",
    "attributes_from_cif",
    "write_model_with_attributes",
]

# Standard _atom_site columns — everything else numeric is a custom attribute.
# occupancy / B_iso_or_equiv are excluded because they're surfaced as the built-in
# "occupancy" / "bfactor" attributes already.
_STANDARD_ATOM_SITE = frozenset({
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
    "label_asym_id", "label_entity_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
    "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_asym_id", "auth_comp_id",
    "auth_atom_id", "pdbx_PDB_model_num", "pdbx_formal_charge", "pdbx_PDB_ins_code",
    "Cartn_x_esd", "Cartn_y_esd", "Cartn_z_esd", "occupancy_esd", "B_iso_or_equiv_esd",
    "calc_flag", "footnote_id", "footnote", "U_iso_or_equiv", "B_equiv_geom_mean",
})


def _cif_block(model: Any) -> Any:
    """The raw mmCIF block behind a cctbx model, or None (e.g. a PDB-loaded model)."""
    return getattr(model.get_model_input(), "cif_block", None)


def _column_to_float(col: Any) -> Optional[np.ndarray]:
    """Cast a CIF string column to float ('.'/'?' -> nan); None if any real string."""
    out = np.empty(len(col), dtype=float)
    for i, s in enumerate(col):
        s = s.strip()
        if s in (".", "?", ""):
            out[i] = np.nan
        else:
            try:
                out[i] = float(s)
            except ValueError:
                return None
    return out


def _custom_columns_from_block(block: Any, n_expected: int) -> dict:
    """Numeric ``_atom_site.*`` columns beyond the standard set -> {name: float array}."""
    attrs: dict = {}
    for key in block.keys():
        if not key.startswith("_atom_site."):
            continue
        name = key[len("_atom_site."):]
        if name in _STANDARD_ATOM_SITE:
            continue
        col = block[key]
        if len(col) != n_expected:
            continue
        vals = _column_to_float(col)
        if vals is not None:
            attrs[name] = vals
    return attrs


def custom_atom_site_attributes(model: Any) -> dict:
    """Custom per-atom scalar columns from a model's mmCIF, aligned by atom order.

    cctbx preserves ``_atom_site`` row order into the hierarchy, so a custom column
    maps to atoms positionally (== i_seq). Returns ``{name: float array}``; empty for
    a PDB-loaded model (no room for custom columns). The model must not be a
    multi-MODEL ensemble here (reduce to model 1 first).
    """
    block = _cif_block(model)
    if block is None:
        return {}
    return _custom_columns_from_block(block, model.get_hierarchy().atoms().size())


def _identity_keys(model: Any) -> List[tuple]:
    """A hashable identity per atom (chain, resseq, icode, altloc, name) in atom order."""
    return [
        (a.chain_id.strip(), a.resseq_as_int(), a.icode.strip(), a.altloc.strip(), a.name.strip())
        for a in model.get_hierarchy().atoms_with_labels()
    ]


def attributes_from_cif(path: str | Path, target_model: Any, *, data_manager: Any = None) -> dict:
    """Read custom ``_atom_site`` columns from mmCIF ``path`` and align to a model.

    The file is read with cctbx and matched to ``target_model`` **by atom identity**
    (chain, residue number, insertion code, altloc, atom name) — so it need not be in
    the same order, and atoms absent from the file get ``nan``. Returns
    ``{name: float array}`` of length ``target_model``'s atom count.
    """
    source = first_model(read_model(path, data_manager=data_manager))
    block = _cif_block(source)
    if block is None:
        raise ValueError(f"{path} is not mmCIF (no _atom_site cif block to read attributes from)")
    columns = _custom_columns_from_block(block, source.get_hierarchy().atoms().size())
    if not columns:
        raise ValueError(f"{path} has no custom _atom_site columns")
    source_keys = _identity_keys(source)
    target_keys = _identity_keys(target_model)
    result = {}
    for name, values in columns.items():
        by_id = dict(zip(source_keys, values))
        result[name] = np.array([by_id.get(k, np.nan) for k in target_keys], dtype=float)
    return result


def write_model_with_attributes(model: Any, attributes: dict, path: str | Path) -> None:
    """Write ``model`` plus named per-atom attribute arrays to an mmCIF file.

    Each attribute becomes a custom ``_atom_site.<name>`` column, aligned to the
    model's atom order (``nan`` -> the CIF null ``.``). Uses cctbx's mmCIF writer.
    """
    import iotbx.cif
    from cctbx.array_family import flex

    cif_model = iotbx.cif.reader(input_string=model.model_as_mmcif()).model()
    block_name = list(cif_model.keys())[0]
    block = cif_model[block_name]
    loop = block.get_loop("_atom_site")
    n = loop.n_rows()
    for name, values in attributes.items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"attribute {name!r} has {arr.shape[0]} values but the model has {n} atoms")
        col = flex.std_string(["." if not np.isfinite(v) else "%.6g" % v for v in arr])
        loop.add_column(f"_atom_site.{name}", col)
    block.add_loop(loop)
    out = iotbx.cif.model.cif()
    out[block_name] = block
    with open(path, "w") as f:
        f.write(str(out))


def _column(value: Any, n: int, default: Any) -> List[Any]:
    """Broadcast a scalar to length ``n``, or pass a per-atom sequence through."""
    if value is None:
        return [default] * n
    if isinstance(value, str) or not hasattr(value, "__len__"):
        return [value] * n
    seq = list(value)
    if len(seq) != n:
        raise ValueError(f"column length {len(seq)} does not match {n} atoms")
    return seq


def model_from_sites(
    sites: Any,
    *,
    elements: Any = None,
    names: Any = None,
    chains: Any = None,
    resseqs: Any = None,
    resnames: Any = None,
    label: str = "pxviewer",
    data_manager: Any = None,
) -> Any:
    """Build a cctbx model from coordinate + label arrays via a generated mmCIF.

    This is the string route the demos and tests use instead of hand-constructing
    a hierarchy: it writes a minimal ``_atom_site`` loop and loads it through the
    DataManager. ``sites`` is an ``(N, 3)`` array; each label argument is a scalar
    (broadcast) or a per-atom sequence. Defaults make a chain of carbons, one per
    residue (positional index == residue == i_seq), named by element and using the
    ``UNL`` (unknown ligand) residue so cctbx treats synthetic points as
    non-polymer rather than protein.
    """
    sites = np.asarray(sites, dtype=float).reshape(-1, 3)
    n = sites.shape[0]
    elements = _column(elements, n, "C")
    names = list(elements) if names is None else _column(names, n, "C")
    chains = _column(chains, n, "A")
    resnames = _column(resnames, n, "UNL")
    resseqs = _column(resseqs, n, None)
    if resseqs[0] is None:
        resseqs = list(range(1, n + 1))

    cols = [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
        "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv",
        "auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num",
    ]
    out = ["data_" + label, "loop_"] + ["_atom_site." + c for c in cols]
    for i in range(n):
        x, y, z = sites[i]
        out.append(
            "ATOM %d %s %s . %s %s 1 %d %.3f %.3f %.3f 1.00 0.00 %d %s 1"
            % (i + 1, elements[i], names[i], resnames[i], chains[i], int(resseqs[i]),
               x, y, z, int(resseqs[i]), chains[i])
        )
    cif = "\n".join(out) + "\n"

    dm = _dm(data_manager)
    return dm.get_model(dm.process_model_str(label, cif))


def _require_data_manager():
    try:
        from iotbx.data_manager import DataManager
    except ImportError as exc:  # pragma: no cover - exercised only without cctbx
        raise ImportError(
            "pxviewer's model loading needs cctbx. Install it with:\n"
            "    conda install -c conda-forge cctbx-base"
        ) from exc
    return DataManager


def data_manager(existing: Any = None) -> Any:
    """The DataManager for one operation: the caller's, or a fresh one.

    Anything here that needs a DataManager takes one, rather than quietly making its
    own, so the caller owns where the data came from.

    The scope is one operation, not one application, because that is the scope cctbx
    itself uses: ``CCTBXParser`` builds a DataManager per invocation and hands it to
    the program, and ``ProgramTemplate.__init__`` then *mutates* what it is given —
    default output filename, overwrite, and ``set_program(self)``, which binds the
    DataManager to that one program. Two tools sharing one DataManager would fight
    over that binding, so there is deliberately no app-wide instance to pass around.
    """
    if existing is not None:
        return existing
    return _require_data_manager()()


# Callers take a `data_manager` argument, which shadows the function above.
_dm = data_manager


def read_model(path: str | Path, *, data_manager: Any = None) -> Any:
    """Read a model file (PDB or mmCIF) and return its ``mmtbx.model.manager``.

    Reads through ``data_manager`` when given, so the caller keeps the file and the
    model it produced (see :func:`data_manager`).
    """
    dm = _dm(data_manager)
    dm.process_model_file(str(path))
    return dm.get_model(str(path))


def first_model(model: Any) -> Any:
    """Reduce a multi-MODEL (NMR ensemble) to its first model; else return as-is.

    We stream a single fixed topology, so a file with several MODEL records is
    collapsed to model 1 (the common expectation). The reduced model's atoms keep
    a contiguous i_seq order, so it stays consistent with the extracted columns and
    with cctbx selections. (Treating the models as trajectory frames is a possible
    future opt-in — they share a topology.)
    """
    models = model.get_hierarchy().models()
    if len(models) <= 1:
        return model
    return model.select(models[0].atoms().extract_i_seq())


def model_to_arrays(model: Any) -> AtomArrays:
    """Lift a cctbx model's hierarchy into :class:`AtomArrays`.

    Coordinates, element, name, B and occupancy come from the hierarchy's
    vectorized ``extract_*`` arrays; residue name, chain id, residue number and
    altloc need one ordered pass over ``atoms_with_labels`` (cctbx exposes no
    vectorized accessor for those). Both walks follow the same atom order.
    """
    hierarchy = model.get_hierarchy()
    atoms = hierarchy.atoms()

    xyz = atoms.extract_xyz().as_numpy_array()  # (N, 3) float64
    element = [e.strip() for e in atoms.extract_element()]
    name = [n.strip() for n in atoms.extract_name()]
    b = atoms.extract_b().as_numpy_array()
    occ = atoms.extract_occ().as_numpy_array()

    n = len(element)
    resname: List[str] = [""] * n
    chain: List[str] = [""] * n
    resseq = np.empty(n, dtype=np.int32)
    altloc: List[str] = [""] * n
    for i, a in enumerate(hierarchy.atoms_with_labels()):
        resname[i] = a.resname.strip()
        chain[i] = a.chain_id.strip()
        resseq[i] = a.resseq_as_int()
        altloc[i] = a.altloc.strip()

    return AtomArrays(
        element=element,
        name=name,
        resname=resname,
        chain=chain,
        resseq=resseq,
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        altloc=altloc,
        b=b,
        occ=occ,
    )


def model_secondary_structure(model: Any) -> List[Tuple[str, int, int, str]]:
    """Extract secondary structure as ``(chain, beg_resseq, end_resseq, kind)`` rows.

    Reads the model's SS annotation (from HELIX/SHEET records or a stored
    assignment); ``kind`` is ``"helix"`` or ``"sheet"``, ready for the BinaryCIF
    encoder so Mol* can render cartoon. Returns ``[]`` when no annotation exists.
    """
    try:
        annotation = model.get_ss_annotation()
    except Exception:  # pragma: no cover - defensive; some models have no SS machinery
        annotation = None
    if annotation is None:
        return []

    rows: List[Tuple[str, int, int, str]] = []
    for helix in annotation.helices:
        rows.append(
            (helix.start_chain_id.strip(), helix.get_start_resseq_as_int(),
             helix.get_end_resseq_as_int(), "helix")
        )
    for sheet in annotation.sheets:
        for strand in sheet.strands:
            rows.append(
                (strand.start_chain_id.strip(), strand.get_start_resseq_as_int(),
                 strand.get_end_resseq_as_int(), "sheet")
            )

    # Depositors routinely annotate SS on only one member of an NCS-related
    # chain set (e.g. 6CVM declares _struct_conf rows for chain A alone). Mol*
    # honours the annotation literally, so the unannotated copies draw as coil
    # next to a fully-ribboned twin. The annotation is display data here —
    # extend it to any chain whose deposited sequence is identical to an
    # annotated one, since identical geometry gives identical SS. This is
    # deliberately not written back into the model: a copied assignment is a
    # rendering fact, not a restraint source.
    annotated = {r[0] for r in rows}
    if annotated:
        from iotbx.pdb import common_residue_names_get_class as get_class
        polymer = {"common_amino_acid", "modified_amino_acid",
                   "common_rna_dna", "modified_rna_dna"}
        signatures = {}
        for chain in model.get_hierarchy().chains():
            seq = tuple(
                (rg.atom_groups()[0].resname.strip(), rg.resseq_as_int())
                for rg in chain.residue_groups()
                if get_class(rg.atom_groups()[0].resname) in polymer)
            if seq:
                signatures[chain.id.strip()] = seq
        templates = {cid for cid in signatures if cid in annotated}
        base = list(rows)
        for cid, seq in signatures.items():
            if cid in annotated:
                continue
            if any(signatures[t] == seq for t in templates):
                rows.extend((cid, beg, end, kind) for c, beg, end, kind in base
                            if signatures.get(c) == seq)
    return rows


def model_is_polymer(model: Any) -> bool:
    """Whether the model has an actual polymer *chain* — at least two protein or nucleic-acid
    residues in one chain — which is what cartoon can trace. A lone amino acid, or a handful
    of isolated residues (a metal site's coordinating histidines, say), is not a polymer:
    cartoon has nothing to draw, so those default to ball-and-stick."""
    try:
        from iotbx.pdb import common_residue_names_get_class as get_class

        polymer = {"common_amino_acid", "modified_amino_acid",
                   "common_rna_dna", "modified_rna_dna"}
        for chain in model.get_hierarchy().chains():
            n = sum(1 for rg in chain.residue_groups()
                    if get_class(rg.atom_groups()[0].resname) in polymer)
            if n >= 2:
                return True
        return False
    except Exception:  # pragma: no cover - defensive
        return False


class ModelData:
    """The session's atom source: numpy columns plus (optionally) the native model.

    The columns (:class:`AtomArrays`) are kept so we never re-derive them from flex
    on every access. When present, ``model`` is the authority for **identity and
    selection**: `select("...")` goes through cctbx's own atom-selection machinery
    rather than any reimplementation, and `diff()` catches the cached columns
    drifting from the model. cctbx calls are serialized under a lock, since the
    session may touch the model from its WebSocket thread.

    ``polymer`` and ``secondary_structure`` are carried alongside — they're read
    once, when the session encodes the topology (to enable cartoon rendering).
    """

    def __init__(
        self,
        arrays: AtomArrays,
        model: Any = None,
        *,
        polymer: bool = False,
        secondary_structure: Optional[List[Tuple[str, int, int, str]]] = None,
        attributes: Optional[dict] = None,
    ):
        self.arrays = arrays
        self.model = model
        self.polymer = polymer
        self.secondary_structure = secondary_structure or []
        # Custom per-atom scalar columns auto-detected from the model's mmCIF.
        self.attributes: dict = attributes or {}
        self._cache: Any = None
        self._process_for_selection: Optional[bool] = None
        self._lock = threading.Lock()

    @classmethod
    def from_model(cls, model: Any) -> "ModelData":
        """Build from an ``mmtbx.model.manager``: columns, SS, polymer flag, attributes.

        A multi-MODEL ensemble is reduced to model 1 first (columns and any custom
        ``_atom_site`` attribute columns are reduced consistently), so everything
        stays on the same atom set.
        """
        # Extract custom columns before reducing, then apply the same reduction.
        extras = custom_atom_site_attributes(model)
        models = model.get_hierarchy().models()
        if len(models) > 1:
            sel = models[0].atoms().extract_i_seq()
            idx = sel.as_numpy_array()
            model = model.select(sel)
            extras = {k: v[idx] for k, v in extras.items()}
        return cls(
            model_to_arrays(model),
            model=model,
            polymer=model_is_polymer(model),
            secondary_structure=model_secondary_structure(model),
            attributes=extras,
        )

    def __len__(self) -> int:
        return len(self.arrays)

    @property
    def n_atoms(self) -> int:
        return len(self.arrays)

    @property
    def elements(self) -> List[str]:
        return self.arrays.element

    @property
    def coords(self) -> np.ndarray:
        """Base coordinates as ``(N, 3)`` float32 (topology frame)."""
        return self.arrays.xyz

    def has_model(self) -> bool:
        return self.model is not None

    def selection_indices(self, expression: str) -> np.ndarray:
        """Resolve a cctbx atom-selection string to positional (i_seq) indices.

        Routes through ``model.selection()`` — the mmtbx manager's cache is
        built with ``special_position_settings`` (when crystal symmetry is
        present) so spatial keywords like ``within``/``residues_within`` work,
        and processing the model (done lazily below, once) turns on the
        selection manager's extra predicates: ``backbone``, ``sidechain``,
        ``rna``/``dna``/``nucleotide``, ``phosphate``, ``ribose``. The bare
        ``hierarchy.atom_selection_cache()`` silently drops all of that.
        i_seq == our positional/wire index.
        """
        if self.model is None:
            raise ValueError(
                "cctbx selection strings require a model-backed session "
                "(build it via LiveSession.from_model_file / from_cctbx_model)"
            )
        with self._lock:
            if self._process_for_selection is None:
                # Funnel through edits.build_restraints — the serialized entry
                # point that also folds in the user's custom bond/angle edits —
                # not a bare model.process() (a concurrent bare build can race
                # the load-time warm or a minimize build, and would ignore the
                # edits scope). ~0.5 s on 1ubq, scales with size; a failure
                # leaves the plain cache grammar (still has within() when
                # crystal symmetry exists). Attempted once.
                self._process_for_selection = True
                if (hasattr(self.model, "selection")
                        and getattr(self.model, "_all_monomer_mappings", None) is None):
                    try:
                        from .edits import build_restraints
                        build_restraints(self.model)
                    except Exception:
                        pass
            if hasattr(self.model, "selection"):
                bsel = self.model.selection(expression)  # flex.bool
            else:  # raw hierarchy — bare cache, no spatial keywords
                if self._cache is None:
                    self._cache = self.model.get_hierarchy().atom_selection_cache()
                bsel = self._cache.selection(expression)
        return bsel.iselection().as_numpy_array()

    def diff(self, tol: float = 1e-3) -> Optional[str]:
        """None if the cached columns still match the model, else a drift message.

        Guards against the held model being mutated underneath the cache (e.g. a
        refinement step moved atoms): compares atom count and coordinates against
        ``model.get_sites_cart()``. Cheap enough to call before relying on the cache.
        """
        if self.model is None:
            return None
        with self._lock:
            sites = self.model.get_sites_cart().as_numpy_array()
        if sites.shape[0] != self.n_atoms:
            return f"atom-count drift: model has {sites.shape[0]}, cache has {self.n_atoms}"
        cached = np.stack([self.arrays.x, self.arrays.y, self.arrays.z], axis=1)
        dev = float(np.abs(sites - cached).max())
        return None if dev <= tol else f"coordinate drift: max |delta| = {dev:.4f} A"


def _parse_oper_expression(expr: Any) -> List[Tuple[str, ...]]:
    """An mmCIF ``oper_expression`` → op-id tuples to apply in sequence.

    ``'(1-60)'`` is one group of sixty; ``'(1,2,6)(7)'`` is a product — every
    combination of the groups, applied left group first. Bare ``'1'`` means the
    single identity-class op.
    """
    text = str(expr).strip()
    groups = re.findall(r"\(([^()]*)\)", text) or [text]
    expanded = []
    for group in groups:
        ids: List[str] = []
        for tok in group.split(","):
            tok = tok.strip()
            m = re.fullmatch(r"(\w+)-(\w+)", tok)
            if m and m.group(1).isdigit() and m.group(2).isdigit():
                ids.extend(str(i) for i in range(int(m.group(1)),
                                                 int(m.group(2)) + 1))
            elif tok:
                ids.append(tok)
        expanded.append(ids or ["1"])
    return list(itertools.product(*expanded))


def _assembly_plan(path: str) -> Optional[Dict[str, List[Tuple[Any, Any]]]]:
    """The deposited file's preferred biological assembly, as a per-chain list
    of transforms to apply: ``{chain_id: [(r, t), ...]}``.

    mmCIF only. ``None`` when the file declares no ``_pdbx_struct_assembly`` —
    legacy PDB MTRIX expansion still applies in that case via cctbx's default.
    The first declared assembly id is preferred, matching the convention that
    assembly '1' is the entry's primary biological unit.
    """
    try:
        import iotbx.cif
        reader = iotbx.cif.reader(file_path=path)
        blocks = list(reader.model().values())
        if not blocks:
            return None
        block = blocks[0]
        gen_ids = block.get("_pdbx_struct_assembly_gen.assembly_id")
        oper_exprs = block.get("_pdbx_struct_assembly_gen.oper_expression")
        asym_lists = block.get("_pdbx_struct_assembly_gen.asym_id_list")
        if gen_ids is None or oper_exprs is None:
            return None
        rows = list(zip(gen_ids, oper_exprs,
                        asym_lists or [""] * len(gen_ids)))
        chosen = rows[0][0]
        from iotbx.mtrix_biomt import process_BIOMT_records_cif
        ops = process_BIOMT_records_cif(block)
        table = {}
        for sn, r, t in zip(ops.serial_number, ops.r, ops.t):
            table[str(sn[0] if isinstance(sn, tuple) else sn)] = (r, t)
        plan: Dict[str, List[Tuple[Any, Any]]] = {}
        for _asm, expr, asyms in rows:
            if _asm != chosen:
                continue
            chains = [c.strip() for c in str(asyms).split(",") if c.strip()]
            transforms = []
            for combo in _parse_oper_expression(expr):
                r_acc, t_acc = None, None
                for oid in combo:
                    op = table.get(str(oid))
                    if op is None:
                        r_acc = None
                        break
                    r_op, t_op = op
                    if r_acc is None:
                        r_acc, t_acc = r_op, t_op
                    else:
                        r_acc, t_acc = r_op * r_acc, r_op * t_acc + t_op
                if r_acc is not None:
                    transforms.append((r_acc, t_acc))
            for chain in chains:
                plan.setdefault(chain, []).extend(transforms)
        return plan or None
    except Exception:
        return None


def _expand_to_assembly(model: Any, plan: Dict[str, List[Tuple[Any, Any]]]) -> None:
    """Replace the model's hierarchy with its declared biological assembly.

    Each chain listed in the assembly's ``asym_id_list`` emits one copy per
    transform; the first copy keeps the deposited chain id and later copies get
    fresh ids (the same scheme cctbx's own MTRIX expansion uses, so the SS
    annotation multiply still lines up). Chains the assembly does not name are
    kept once, unexpanded — waters and ligands typically absent from
    ``asym_id_list`` stay visible rather than silently dropping atoms.
    """
    import iotbx.pdb.hierarchy
    from iotbx.pdb.utils import all_chain_ids

    hierarchy = model.get_hierarchy()
    src = hierarchy.only_model()
    taken = {c.id for c in src.chains()}
    fresh = iter(cid for cid in all_chain_ids() if cid not in taken)
    result = iotbx.pdb.hierarchy.root()
    out_model = iotbx.pdb.hierarchy.model()
    ids_match: Dict[str, List[str]] = {}
    n_copies = None
    matched = False
    for chain in src.chains():
        ops = plan.get(chain.id.strip()) or plan.get(chain.id)
        if not ops:
            out_model.append_chain(chain.detached_copy())
            continue
        matched = True
        if n_copies is None:
            n_copies = len(ops)
        new_ids = []
        for i, (r, t) in enumerate(ops):
            c = chain.detached_copy()
            c.atoms().set_xyz(r.elems * c.atoms().extract_xyz() + t)
            if i > 0:
                c.id = next(fresh)
                new_ids.append(c.id)
            out_model.append_chain(c)
        ids_match[chain.id] = new_ids
    if not matched:
        return
    result.append_model(out_model)
    result.atoms().reset_i_seq()
    result.atoms().reset_serial()
    model._pdb_hierarchy = result
    if result.models_size() == 1:
        result.only_model().id = ""
    ssa = model.get_ss_annotation()
    if ssa is not None and len({len(v) for v in ids_match.values()}) <= 1:
        try:
            ssa.multiply_to_asu_2(ids_match)
            model.set_ss_annotation(ssa)
        except Exception:
            pass
    model._xray_structure = None
    model._update_atom_selection_cache()
    model.restraints_manager = None


def load_model(path: str | Path, *, data_manager: Any = None) -> ModelData:
    """Read a model file (reduced to model 1) into a :class:`ModelData`.

    When an mmCIF declares a biological assembly, the model is expanded to it
    rather than to the crystallographic NCS records cctbx applies by default.
    The two differ whenever the biological unit is not the ASU — e.g. 1QGT,
    whose 30 ``_struct_ncs_oper`` records produce half a capsid (120 subunits)
    while ``_pdbx_struct_assembly`` declares the complete 240-subunit shell.
    """
    path = str(path)
    plan = _assembly_plan(path)
    dm = _dm(data_manager)
    if plan is not None:
        opts = dm.custom_options
        if not isinstance(opts, list):
            opts = []
        if "model_skip_expand_with_mtrix" not in opts:
            opts.append("model_skip_expand_with_mtrix")
        dm.custom_options = opts
    dm.process_model_file(path)
    model = first_model(dm.get_model(path))
    if plan is not None and not model.mtrix_expanded():
        _expand_to_assembly(model, plan)
    return ModelData.from_model(model)
