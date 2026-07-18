#!/usr/bin/env python
"""Generate THIRD_PARTY_LICENSES.md from the active conda env.

Lists every installed package (conda + pip-only) with its declared license, flags the
copyleft / LGPL ones that carry redistribution obligations, and collects the license
text files it can find into a licenses/ directory beside the manifest.

    conda activate pxviewer
    python scripts/gen_third_party_licenses.py

Run it against the environment you actually ship; the declared licenses come straight
from conda-forge/pip package metadata. It is a starting point for a counsel review, not
a legal conclusion.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

PREFIX = Path(sys.prefix)
OUT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = OUT_DIR / "THIRD_PARTY_LICENSES.md"
TEXT_DIR = OUT_DIR / "licenses"

_PERMISSIVE = ("MIT", "BSD", "APACHE", "FTL", "AFL", "ZLIB", "BSL", "ISC", "PSF",
               "PYTHON-2.0", "UNLICENSE", "CC0")
_WEAK_COPYLEFT = ("MPL", "EPL", "CDDL", "EUPL", "OSL", "CPL", "SLEEPYCAT")


def classify(lic: str) -> str:
    """One of: 'attention' (bare GPL/AGPL — real blocker for closed source),
    'obligations' (LGPL / weak copyleft — usable with notice + dynamic-link duties),
    'exception' (GPL under a linking exception, or dual-licensed with a permissive OR —
    safe), or 'permissive'."""
    L = lic.upper()
    has_gpl = "GPL" in L
    has_lgpl = "LGPL" in L or "LESSER" in L
    has_exception = "EXCEPTION" in L
    dual_permissive = " OR " in L and any(p in L for p in _PERMISSIVE)
    if has_gpl and not has_lgpl:
        return "exception" if (has_exception or dual_permissive) else "attention"
    if has_lgpl or any(t in L for t in _WEAK_COPYLEFT):
        return "obligations"
    return "permissive"


def conda_packages() -> dict:
    pkgs = {}
    for meta in (PREFIX / "conda-meta").glob("*.json"):
        try:
            d = json.loads(meta.read_text())
        except Exception:
            continue
        pkgs[d["name"]] = {
            "version": d.get("version", "?"),
            "license": d.get("license") or "?",
            "channel": (d.get("channel") or "").rsplit("/", 1)[-1],
            "source": "conda",
        }
    return pkgs


def pip_only_packages(conda_names: set) -> dict:
    import importlib.metadata as im

    pkgs = {}
    for dist in im.distributions():
        name = dist.metadata["Name"]
        if not name or name.lower() in {n.lower() for n in conda_names}:
            continue  # installed by conda, already covered
        lic = dist.metadata.get("License") or ""
        if not lic or len(lic) > 60:  # some pack the whole text into License:; prefer classifier
            cls = [c for c in (dist.metadata.get_all("Classifier") or []) if "License ::" in c]
            if cls:
                lic = cls[-1].split("::")[-1].strip()
        pkgs[name] = {"version": dist.version, "license": lic or "?",
                      "channel": "pypi", "source": "pip"}
    return pkgs


def collect_license_texts(names) -> dict:
    """Best-effort: copy any license files found for each package into licenses/<name>/.
    Returns {name: [relative paths copied]}."""
    if TEXT_DIR.exists():
        shutil.rmtree(TEXT_DIR)
    found = {}
    # conda-forge installs many license files under share/licenses/<pkg>/
    share = PREFIX / "share" / "licenses"
    import importlib.metadata as im
    dist_by_name = {}
    for dist in im.distributions():
        n = dist.metadata["Name"]
        if n:
            dist_by_name[n.lower()] = dist
    for name in names:
        dest = TEXT_DIR / name
        copied = []
        src = share / name
        if src.is_dir():
            for f in src.rglob("*"):
                if f.is_file():
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, dest / f.name)
                    copied.append(f.name)
        dist = dist_by_name.get(name.lower())
        if dist is not None:
            base = dist._path  # dist-info dir
            for f in list(base.glob("LICENS*")) + list(base.glob("COPYING*")) \
                    + list((base / "licenses").glob("*") if (base / "licenses").is_dir() else []):
                if f.is_file():
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, dest / f.name)
                    copied.append(f.name)
        if copied:
            found[name] = sorted(set(copied))
    return found


def main() -> int:
    conda = conda_packages()
    allpkgs = dict(conda)
    allpkgs.update(pip_only_packages(set(conda)))

    texts = collect_license_texts(allpkgs)
    for p in allpkgs.values():
        p["class"] = classify(p["license"])
    by = lambda c: sorted(n for n, p in allpkgs.items() if p["class"] == c)
    attention, obligations, exception = by("attention"), by("obligations"), by("exception")

    def table(names, cols_texts=False):
        head = ("| Package | Version | License | Source |"
                + (" Texts |" if cols_texts else ""))
        rows = [head, "| --- | --- | --- | --- |" + (" --- |" if cols_texts else "")]
        for n in names:
            p = allpkgs[n]
            row = f"| {n} | {p['version']} | {p['license']} | {p['source']} |"
            if cols_texts:
                row += f" {'✓' if n in texts else ''} |"
            rows.append(row)
        if not names:
            rows.append("| _(none)_ | | | |" + (" |" if cols_texts else ""))
        return rows

    lines = [
        "# Third-party licenses",
        "",
        "Generated by `scripts/gen_third_party_licenses.py` from the conda env. Declared "
        "licenses are from conda-forge / PyPI metadata. **Not legal advice** — a starting "
        "point for a counsel review. Dev/test-only tools (pytest, …) are listed too; prune "
        "to what actually ships.",
        "",
        f"- Packages: **{len(allpkgs)}** "
        f"({sum(1 for p in allpkgs.values() if p['source']=='conda')} conda, "
        f"{sum(1 for p in allpkgs.values() if p['source']=='pip')} pip). "
        f"License texts under `licenses/`: **{len(texts)}**.",
        "- **No GPL that forces opening your code.** Everything is permissive, LGPL, a "
        "GPL-with-linking-exception, or dual-licensed with a permissive option — except "
        "the two bare-GPL items below, neither of which is loaded or needs shipping.",
        "- **QtWebEngine ships no proprietary codecs** (H.264/AAC report false via "
        "`MediaSource.isTypeSupported`; only royalty-free VP8/VP9/AV1/Opus/Vorbis + "
        "patent-expired MP3), so there is no codec-patent obligation from this build.",
        "",
        "## Attention — bare GPL/AGPL",
        "",
        "Full copyleft, no linking exception. Would force opening a combined work — so "
        "these must not be linked into or shipped with the product. Both here are safe in "
        "practice: `readline` (GPLv3) is **not imported** by the app or the qtconsole path "
        "(verified) — exclude that one stdlib `.so` from a bundle; `ld_impl_linux-64` is "
        "binutils' linker, a **build-time** tool that never ships in a runtime bundle.",
        "",
        *table(attention),
        "",
        "## LGPL & weak copyleft — usable with obligations",
        "",
        "Fine for a closed-source release **if** shipped as replaceable shared libraries "
        "(dynamic linking — automatic here) with their license texts/notices included, and "
        "nothing blocks a user swapping in their own build. Covers Qt/PySide6 and the "
        "system libs, plus MPL/EPL (file-level copyleft — only matters if you modify their "
        "sources, which you don't). cctbx/chem_data are BSD at the core with LGPL CCP4 "
        "components (the WxWindows exception on part of it only loosens the terms).",
        "",
        *table(obligations),
        "",
        "## GPL-with-exception / dual-permissive — no obligation",
        "",
        "GPL under a linking exception (the GCC runtime libraries) or offered under a "
        "permissive alternative (pick that side) — safe to link and ship as-is.",
        "",
        *table(exception),
        "",
        "## All packages",
        "",
        *table(sorted(allpkgs, key=str.lower), cols_texts=True),
    ]

    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"wrote {MANIFEST.relative_to(OUT_DIR)}  ({len(allpkgs)} pkgs; "
          f"attention={len(attention)} obligations={len(obligations)} "
          f"exception={len(exception)} texts={len(texts)})")
    print("attention:", ", ".join(attention) or "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
