"""Ablation test on the planted-error corpus.

For each structure and perturbation, compare the change from its own baseline at
the deliberately altered region with changes at spatially remote control
residues. Report AUROC for clash-only, rama-only, rotamer-only, non-clash, and
the combined hotspot field.

Usage:
  libtbx.python analyze_planted.py /root/map_model_validation/corpus/specificity
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from events import extract_all, load_hierarchy
from field import compute_field


CHANNELS = ("clash", "rama", "rota", "nonclash", "combined")
SEGMENT_LENGTH = {"displacement": 3, "register": 4, "overbuild": 1, "rotamer": 1}


def residue_atoms(hierarchy):
    result = {}
    order = {}
    for model in hierarchy.models():
        for chain in model.chains():
            chain_keys = []
            for rg in chain.residue_groups():
                key = (chain.id.strip(), rg.resseq.strip())
                result[key] = [np.asarray(a.xyz, float) for a in rg.atoms()
                               if not a.name.strip().startswith("H")]
                chain_keys.append(key)
            order.setdefault(chain.id.strip(), []).extend(chain_keys)
    return result, order


def fields(path):
    hierarchy = load_hierarchy(path)
    events = extract_all(hierarchy, use_hydrogens=True)["events"]
    by = {name: [e for e in events if e.metric == name]
          for name in ("clash", "rama", "rota")}
    by["nonclash"] = by["rama"] + by["rota"]
    by["combined"] = events
    return hierarchy, {name: compute_field(by[name]) for name in CHANNELS}


def residue_scores(field_set, atoms):
    return {
        name: {key: max(field.sample(xyz) for xyz in xyzs)
               for key, xyzs in atoms.items() if xyzs}
        for name, field in field_set.items()
    }


def affected_keys(site_keys, order, length):
    affected = set()
    for key in site_keys:
        chain_keys = order.get(key[0], [])
        if key not in chain_keys:
            continue
        pos = chain_keys.index(key)
        affected.update(chain_keys[pos:pos + length])
    return affected


def run(root):
    records = []
    for pdb_id in sorted(os.listdir(root)):
        directory = os.path.join(root, pdb_id)
        site_path = os.path.join(directory, "sites.json")
        baseline_path = os.path.join(directory, "baseline.cif")
        if not os.path.isfile(site_path) or not os.path.isfile(baseline_path):
            continue
        sites = {tuple(x) for x in json.load(open(site_path))["sites"]}
        base_h, base_fields = fields(baseline_path)
        base_atoms, order = residue_atoms(base_h)
        base_scores = residue_scores(base_fields, base_atoms)

        for perturbation, seglen in SEGMENT_LENGTH.items():
            pert_path = os.path.join(directory, perturbation + ".cif")
            pert_h, pert_fields = fields(pert_path)
            pert_atoms, _ = residue_atoms(pert_h)
            pert_scores = residue_scores(pert_fields, pert_atoms)
            affected = affected_keys(sites, order, seglen)

            affected_xyz = [xyz for key in affected for xyz in base_atoms.get(key, [])]
            for key in sorted(set(base_atoms) & set(pert_atoms)):
                if key in affected:
                    label = 1
                else:
                    # Avoid calling Gaussian spillover around a planted region a
                    # false positive: controls must be at least 8 A away.
                    if affected_xyz and min(
                            np.linalg.norm(x - y)
                            for x in base_atoms[key] for y in affected_xyz) < 8.0:
                        continue
                    label = 0
                rec = {"pdb": pdb_id, "perturbation": perturbation,
                       "chain": key[0], "resseq": key[1], "label": label}
                for name in CHANNELS:
                    rec[name] = pert_scores[name][key] - base_scores[name][key]
                records.append(rec)
        print("finished", pdb_id, file=sys.stderr)

    report = {"n_records": len(records), "results": {}}
    for perturbation in SEGMENT_LENGTH:
        subset = [r for r in records if r["perturbation"] == perturbation]
        y = [r["label"] for r in subset]
        report["results"][perturbation] = {
            name: {
                "auroc": float(roc_auc_score(y, [r[name] for r in subset])),
                "positive_median_delta":
                    float(np.median([r[name] for r in subset if r["label"]])),
                "positive_fraction_delta_gt_0_5":
                    float(np.mean([r[name] > 0.5 for r in subset if r["label"]])),
            }
            for name in CHANNELS
        }
        report["results"][perturbation]["counts"] = {
            "positive": sum(y), "control": len(y) - sum(y)
        }
    return report


if __name__ == "__main__":
    corpus = (sys.argv[1] if len(sys.argv) > 1
              else "/root/map_model_validation/corpus/specificity")
    print(json.dumps(run(corpus), indent=2, sort_keys=True))
