"""Multi-panel MolProbity validation figure.

Loads the packaged validation-demo structure (1TEC — deliberately imperfect;
see ``pxviewer.loader.SAMPLES``), runs the real validation machinery in the
real app, and photographs each validator's MolProbity markup zoomed on one of
its flagged residues (ball-and-stick), plus the probe2 contact/clash dots on
the hydrogenated sibling model. Panels land in ``fig_validation/``;
``compose`` joins them into ``fig_validation.png`` with Pillow.

    python fig_validation.py                    # capture + compose
    python fig_validation.py --compose-only     # re-grid existing fig_validation/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from pxviewer.regression.visual.harness import (
    _call_session,
    dispose,
    make_app,
    pump,
    settle,
    wait_viewer,
)
from pxviewer.loader import sample_structure_path

OUT = Path(__file__).resolve().parent / "fig_validation"

# One flagged residue per validator, hand-picked from 1TEC's validation tables
# (real outliers, mid-chain; picks tuned interactively through the app's MCP
# endpoint — see the git history of this file). (validator key, chain, resid,
# caption). rama_z is a whole-model metric with no per-residue markup, so it
# has no panel.
PANELS = [
    ("rotamers", "E", "270", "rotamer outlier (Arg 270, score 0.00)"),
    ("ramachandran", "E", "61", "Ramachandran outlier (Asn 61)"),
    ("cablam", "E", "215", "CaBLAM outlier (Thr 215)"),
    ("cbetadev", "E", "156", "C-beta deviation (Val 156, 0.59 A)"),
    ("omegalyze", "E", "172", "cis-Pro peptide (Pro 172)"),
]


def shot(app, name: str):
    png = _call_session(app, "screenshot", 30)
    assert png, f"no screenshot for {name}"
    path = OUT / f"{name}.png"
    path.write_bytes(png)
    print(f"  {path.name}")
    return path


def wait_jobs(app, timeout: float = 300.0):
    """Pump until every background job drains."""
    pump(until=lambda: not app.job_state()["busy"], timeout=timeout)
    settle(1.0)


def markups(app, on: set[str]):
    """Show exactly the ``on`` validators' markup; clear the rest."""
    from pxviewer import validation

    for spec in validation.validators():
        app.set_validation_markers(spec.key, spec.key in on)


def clash_center(hentry, top: int = 8):
    """Aim at the worst-overlap cluster: the centroid of the ``top`` longest
    clash spikes, viewed perpendicular to their mean direction so the spikes
    show length. Returns ``(target, up, direction)`` or None with no clashes."""
    import numpy as np
    from pxviewer.live import PROBE_CLASHES

    clashes = (hentry.get("probe_dots") or {}).get(PROBE_CLASHES) or []
    if not clashes:
        return None
    order = np.argsort([np.linalg.norm(s) for _, s, _ in clashes])[::-1][:top]
    locs = np.array([clashes[i][0] for i in order], dtype=float)
    spk = np.array([clashes[i][1] for i in order], dtype=float)
    mspk = spk.mean(axis=0)
    mspk /= np.linalg.norm(mspk)
    d = np.cross(mspk, [0.0, 1.0, 0.0])
    if np.linalg.norm(d) < 0.1:
        d = np.cross(mspk, [1.0, 0.0, 0.0])
    d /= np.linalg.norm(d)
    return locs.mean(axis=0), np.cross(d, mspk), d


def compose(shots: dict) -> Path:
    """Whitespace-crop every panel, then lay out: overview spanning the top,
    six detail panels in a 2x3 grid under it, lettered (a)-(g)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    def trim(img: Image.Image) -> Image.Image:
        a = np.asarray(img.convert("RGB"))
        mask = (a < 245).any(axis=-1)  # non-white pixels
        if not mask.any():
            return img
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))

    names = ["overview", "zoom_rotamers", "zoom_ramachandran", "zoom_cablam",
             "zoom_cbetadev", "zoom_omegalyze", "zoom_clashes"]
    if shots:
        panels = {n: trim(Image.open(shots[n]).convert("RGB")) for n in names if n in shots}
    else:
        panels = {n: trim(Image.open(OUT / f"{n}.png").convert("RGB"))
                  for n in names if (OUT / f"{n}.png").exists()}

    cell_w, cell_h, pad, label_h = 900, 700, 18, 46
    grid_cols = 3
    width = cell_w * (grid_cols + 1) + pad * (grid_cols + 2)
    ov_h = 2 * (cell_h + label_h) + pad  # left column spans both detail rows
    height = ov_h + label_h + 2 * pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Helvetica.ttc", 30)
    except OSError:
        font = ImageFont.load_default()

    labels = {
        "overview": "(a) 1TEC — all residue-validator markup",
        "zoom_rotamers": "(b) " + PANELS[0][3],
        "zoom_ramachandran": "(c) " + PANELS[1][3],
        "zoom_cablam": "(d) " + PANELS[2][3],
        "zoom_cbetadev": "(e) " + PANELS[3][3],
        "zoom_omegalyze": "(f) " + PANELS[4][3],
        "zoom_clashes": "(g) probe2 contacts + clashes (model + H)",
    }

    def place(img, x, y, w, h, label):
        img = img.copy()
        img.thumbnail((w, h))
        draw.text((x, y), label, fill="black", font=font)
        canvas.paste(img, (x + (w - img.width) // 2,
                           y + label_h + (h - img.height) // 2))

    place(panels["overview"], pad, pad, cell_w + pad, ov_h, labels["overview"])
    order = ["zoom_rotamers", "zoom_ramachandran", "zoom_cablam",
             "zoom_cbetadev", "zoom_omegalyze", "zoom_clashes"]
    for i, name in enumerate(order):
        r, c = divmod(i, grid_cols)
        x = cell_w + 2 * pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        place(panels[name], x, y, cell_w, cell_h, labels[name])

    out = Path(__file__).resolve().parent / "fig_validation.png"
    canvas.save(out)
    print(f"composed -> {out}")
    return out


def capture() -> dict:
    """Drive the real app through every panel; return {name: png path}."""
    app = make_app()
    shots = {}
    try:
        wait_viewer(app)
        app.load_file(str(sample_structure_path("1tec.pdb")))
        pump(until=lambda: len(app._models) >= 1, timeout=60)
        mid = app._models[-1]["id"]
        settle(2.0)
        app._control_session().set_background("#ffffff")

        print("running validators…")
        app.run_validation(keys=None, clashes=False)
        wait_jobs(app)
        results = app._model_entry(mid)["validation"]
        print("  results:", {k: len(r.rows) for k, r in results.items()})

        # (a) overview: cartoon, every residue validator's markup at once.
        session = app._control_session()
        session.set_representation("cartoon")
        markups(app, {k for k, _, _, _ in PANELS})
        session.reset_view()
        settle(1.5)
        shots["overview"] = shot(app, "overview")

        # (b–f) one validator's markup at a time, ball-and-stick, framed on a
        # flagged residue. The selection highlight is cleared after framing:
        # it paints the residue green, which swallows the green Ramachandran
        # flag and competes with every other marker.
        session.set_representation("ball_and_stick")
        for key, chain, resid, caption in PANELS:
            r = results.get(key)
            if r is None or not r.rows or not r.markup:
                print(f"  {key}: no rows/markup, skipped")
                continue
            markups(app, {key})
            app.focus_residue(chain, resid, focus=True, clip=True, context=False)
            session.clear_selection()
            settle(1.5)
            shots[f"zoom_{key}"] = shot(app, f"zoom_{key}")
            print(f"    {key} -> {caption} ({r.summary})")

        # (g) all-atom contacts: reduce2 + probe2 on the hydrogenated sibling.
        # The dots are the subject — aim at the worst-overlap cluster rather
        # than a residue.
        print("running clashes (reduce2 + probe2)…")
        app.run_validation(keys=[], clashes=True)
        wait_jobs(app)
        hmodels = [m for m in app._models if "+ H" in m.get("name", "")]
        hentry = app._model_entry(hmodels[-1]["id"]) if hmodels else None
        if hentry is not None:
            app.set_model_visible(mid, False)
            aim = clash_center(hentry)
            if aim:
                hentry["session"].orient_camera(aim[0].tolist(),
                                                aim[1].tolist(),
                                                aim[2].tolist(), 3.2)
                settle(1.5)
            shots["zoom_clashes"] = shot(app, "zoom_clashes")
            print("    clashes" + ("" if aim else " (no cluster found)"))
    finally:
        dispose(app)
    return shots


def main() -> int:
    OUT.mkdir(exist_ok=True)
    if "--compose-only" in sys.argv:
        compose({})
        return 0
    shots = capture()
    if shots:
        compose(shots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
