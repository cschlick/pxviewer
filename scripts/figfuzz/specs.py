"""The figure-spec corpus for the figfuzz harness.

A spec is a *goal*, not a script — the maker agent must discover the API and
produce the figure through MCP, which is the point: task language a real user
would write is what we want exercising the software. If a spec names a verb,
it is because the semantic ("only the density near the ligand") is the thing
under test, not the method.

Each spec is a dict:

    id       stable run-dir name ("f03-boxed-site")
    title    one line for the report
    task     the paragraph the maker agent is given
    stage    files the runner materializes into the run dir first:
             {"path": "map.mrc", "kind": "blob"|"big"|"diff", "shape": (n,n,n)}
    expect   declarative checks for the programmatic critic (see critic.py):
             models / volumes      -- exact or minimum counts
             volume_props          -- attrs every volume must show
             grid_deliberate       -- drawn grid is full-res or boxed
             regioned              -- some clip/box/mask state exists
             negative_visible      -- a difference map's minus contour is on
             primitives            -- at least N drawing primitives exist
             model_clipped         -- the model carries a clip
             nonblank              -- the screenshot is not empty (default on)
             no_job_errors         -- status() reports none (default on)
    fetch    needs the network (skipped under --no-fetch)
"""

from __future__ import absolute_import, division, print_function

import random

#: Local models, relative to python/pxviewer/data. heme_site carries a ligand;
#: zn_site a metal site; the others are ordinary proteins.
MODELS = ["1ubq.pdb", "1tec.pdb", "3nir.pdb", "heme_site.pdb", "zn_site.pdb"]


def _spec(id_, title, task, *, stage=(), expect=None, fetch=False, anchor=None):
    expect = {"nonblank": True, "no_job_errors": True, **(expect or {})}
    return {"id": id_, "title": title, "task": task,
            "stage": list(stage), "expect": expect, "fetch": fetch,
            "anchor": anchor}


def corpus(data_dir, seed=0):
    """The spec list with its draws resolved under ``seed`` — which model each
    ``{model}`` placeholder lands on, so a full-corpus run is reproducible."""
    rng = random.Random(seed)
    p = lambda name: f"{data_dir}/{name}"
    m = lambda: p(rng.choice(MODELS))
    # Model+map specs name the model once — the maps are staged with their
    # box centered on that model's centroid, or the "density near the site"
    # task would be unreachable (the fixture map has its own frame).
    f03m = m()
    f06m = m()
    f14m = m()
    specs = [
        _spec("f01-ligand-site", "Heme site with its environment",
              f"Load {p('heme_site.pdb')} and make a figure of the heme site: "
              "the protein as a cartoon, the ligand and the residues within "
              "~4 Å of it drawn in more detail, and the camera framed on the "
              "site so it fills the view.",
              expect={"models": 1}),

        _spec("f02-bfactor", "Cartoon colored by B-factor",
              f"Load {p('3nir.pdb')} and produce a figure of the structure "
              "colored by atomic B-factor — something a reader could take a "
              "B-factor ramp from.",
              expect={"models": 1}),

        _spec("f03-boxed-site", "Density carved around a site",
              f"Load {f03m} and the map file sitediff.mrc from this directory. "
              "Make a figure showing the density only within ~15 Å of the "
              "ligand — as a mesh at a sensible contour. The point is the "
              "surrounding density must not appear.",
              stage=[{"path": "sitediff.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              anchor=f03m,
              expect={"models": 1, "volumes": 1, "regioned": True}),

        _spec("f04-bigmap", "Publishable mesh from an oversized map",
              "Load bigmap.mrc from this directory and produce a clean, "
              "publication-quality figure of the density as a mesh at a "
              "sensible contour. Pay attention to what the rendered grid "
              "actually is before you save.",
              stage=[{"path": "bigmap.mrc", "kind": "big", "shape": (256, 256, 256)}],
              expect={"volumes": 1, "grid_deliberate": True}),

        _spec("f05-diffmap", "Difference map, both signs",
              "Load diffmap.mrc from this directory and make a figure of the "
              "difference map — both positive and negative contours must be "
              "visible in the frame.",
              stage=[{"path": "diffmap.mrc", "kind": "diff", "shape": (96, 96, 96)}],
              expect={"volumes": 1, "negative_visible": True}),

        _spec("f06-masked", "Map masked to the model footprint",
              f"Load {f06m} and blob.mrc from this directory. Make a figure of "
              "the map cut down to the model's footprint — density well away "
              "from any atom should not show.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              anchor=f06m,
              expect={"models": 1, "volumes": 1, "regioned": True}),

        _spec("f07-slab", "A slab through the structure",
              f"Load {m()} and make a figure of a thin slab through the "
              "middle of the molecule — the structure outside the slab must "
              "not be drawn.",
              expect={"models": 1, "model_clipped": True}),

        _spec("f08-clip-survives-rep", "Clip held across a rep change",
              f"Load {m()}, restrict the drawn structure to a ~20 Å sphere "
              "around its center, then change the representation to "
              "ball-and-stick and make the figure — the clip must still hold "
              "in the final image.",
              expect={"models": 1, "model_clipped": True}),

        _spec("f09-two-models", "Two models, one hidden",
              f"Load {p('1ubq.pdb')} and {p('1tec.pdb')}. Make a figure that "
              "shows exactly one of them — the other must not appear.",
              expect={"models": 2}),

        _spec("f10-measure", "A labeled distance",
              f"Load {m()} and produce a figure that measures and labels the "
              "distance between two clearly identifiable atoms of your "
              "choice, with the structure in context.",
              expect={"models": 1, "primitives": 1}),

        _spec("f11-fetch-pdb-map", "Fetched model inside its map",
              "Fetch PDB entry 7BV2 together with its map and make a figure "
              "of the model sitting in its electron density at a sensible "
              "contour.",
              expect={"models": 1, "volumes": 1}, fetch=True),

        _spec("f12-fetch-emdb", "Fetched cryo-EM map as a clean mesh",
              "Fetch the EMD-30210 map and produce a clean figure of it as a "
              "mesh at a sensible contour for its resolution.",
              expect={"volumes": 1}, fetch=True),

        _spec("f13-quality", "A final-quality render",
              f"Load {m()} and produce the best-looking still the viewer "
              "can make: supersampled, richer lighting if available. The "
              "frame must show the whole structure.",
              expect={"models": 1}),

        _spec("f14-transparent", "Transparent surface over the model",
              f"Load {f14m} and blob.mrc from this directory. Make a figure "
              "with the map as a transparent surface around the cartoon "
              "model — the model must read through the surface.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              anchor=f14m,
              expect={"models": 1, "volumes": 1}),
    ]
    return specs


def generate(data_dir, count, seed):
    """``count`` specs sampled from the corpus — repeat a spec's draws under
    different seeds and the same task exercises different inputs."""
    rng = random.Random(seed)
    picked = []
    for i in range(count):
        spec = dict(rng.choice(corpus(data_dir, seed=rng.randrange(2 ** 31))))
        spec["id"] = f"{i + 1:02d}-{spec['id']}"
        picked.append(spec)
    return picked
