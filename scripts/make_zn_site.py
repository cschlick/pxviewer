"""Generate the bundled metal-coordination example, pxviewer/data/zn_site.pdb.

A carbonic-anhydrase-like zinc site: a Zn(II) coordinated by three histidines (a real HIS,
lifted from the bundled 1UBQ and rotated so each NE2 points at the metal) plus one water in
the fourth position. cctbx auto-restrains the Zn-His coordination (metal linking), but not
the Zn-water bond — which is exactly what the "restraint edits" tutorials add by hand.

Run from the repo root:  python scripts/make_zn_site.py
"""

import pathlib

import numpy as np

from iotbx.data_manager import DataManager

from pxviewer import cctbx_io

DATA = pathlib.Path(__file__).resolve().parents[1] / "python" / "pxviewer" / "data"
D = 2.10  # metal-donor coordination distance, Angstrom
TETRAHEDRON = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
HIS_ORDER = ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"]


def _unit(v):
    return v / np.linalg.norm(v)


def _rotation(a, b):
    """Rotation mapping unit vector a onto unit vector b (Rodrigues)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _real_histidine():
    dm = DataManager()
    dm.process_model_file(str(DATA / "1ubq.pdb"))
    ag = next(ag for rg in dm.get_model().get_hierarchy().residue_groups()
              for ag in rg.atom_groups() if ag.resname.strip() == "HIS")
    coord = {a.name.strip(): np.array(a.xyz) for a in ag.atoms()}
    names = [n for n in HIS_ORDER if n in coord]
    return names, np.array([coord[n] for n in names]), coord


def build():
    names, xyz, coord = _real_histidine()
    ne2, ce1, cd2 = coord["NE2"], coord["CE1"], coord["CD2"]
    lone_pair = -_unit(_unit(ce1 - ne2) + _unit(cd2 - ne2))  # imidazole N lone pair
    tet = [_unit(np.array(d)) for d in TETRAHEDRON]

    sites, at_names, elements, chains, resseqs, resnames = [], [], [], [], [], []
    for k, direction in enumerate(tet[:3]):  # three histidines, each NE2 aimed at the metal
        rot = _rotation(lone_pair, -direction)
        placed = (xyz - ne2) @ rot.T + D * direction
        for name, point in zip(names, placed):
            sites.append(point)
            at_names.append(name)
            elements.append(name[0])
            chains.append("ABC"[k])
            resseqs.append(1)
            resnames.append("HIS")
    sites.append(np.zeros(3))  # the metal, at the origin
    at_names.append("ZN"); elements.append("ZN"); chains.append("S")
    resseqs.append(1); resnames.append("ZN")
    sites.append(D * tet[3])  # a coordinating water in the fourth position
    at_names.append("O"); elements.append("O"); chains.append("S")
    resseqs.append(2); resnames.append("HOH")

    model = cctbx_io.model_from_sites(
        np.array(sites), elements=elements, names=at_names,
        chains=chains, resseqs=resseqs, resnames=resnames, label="zn_site")
    (DATA / "zn_site.pdb").write_text(model.model_as_pdb())
    print(f"wrote {DATA / 'zn_site.pdb'} ({len(sites)} atoms)")


if __name__ == "__main__":
    build()
