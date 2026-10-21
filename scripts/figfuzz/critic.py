"""The two-tier critic.

The programmatic tier asks the live app what it actually drew — the checks an
agent "did not think to run" made mechanical: is the page still alive, did the
objects land, is the rendered grid the deliberate one, does the frame have
pixels in it. The agentic tier then judges the screenshot against the spec:
the semantic failures (wrong region, invisible density, model occluded) that
no state check can see.

Both feed one verdict per spec in the run report.
"""

from __future__ import absolute_import, division, print_function

import json
import re


# -- programmatic tier -----------------------------------------------------

def evaluate(client, spec, png_bytes):
    """Run the spec's ``expect`` against the live app. ``client`` is an
    McpClient; ``png_bytes`` the frame captured through screenshot() (may be
    None when the page never answered). Returns a list of findings:
    ``(name, ok, detail)`` — ok None means the check could not run."""
    expect = spec["expect"]
    findings = []

    def find(name, ok, detail=""):
        findings.append((name, ok, detail))
        return ok

    try:
        pong = client.call("ping", timeout=20.0)
    except Exception as exc:
        find("mcp_endpoint", False, f"endpoint did not answer: {exc!r}")
        return findings   # a dead endpoint makes every later check noise
    find("mcp_endpoint", True)

    alive = isinstance(pong, dict) and pong.get("alive")
    detail = "" if alive else (pong.get("detail") if isinstance(pong, dict)
                             else repr(pong)[:200])
    if not find("page_alive", bool(alive), detail):
        # A dead render process is the headline failure — the remaining state
        # is unreached, not untested.
        return findings
    if isinstance(pong, dict) and pong.get("lastStall"):
        findings.append(("apply_chain_stalled", None,
                         str(pong["lastStall"])[:200]))

    try:
        state = client.call("status")
    except Exception as exc:
        find("status", False, repr(exc)[:200])
        state = {}

    models = state.get("models") or []
    volumes = state.get("volumes") or []
    if "models" in expect:
        find("models", len(models) >= expect["models"],
             f"{len(models)} loaded")
    if "volumes" in expect:
        find("volumes", len(volumes) >= expect["volumes"],
             f"{len(volumes)} loaded")

    looks = []
    for v in volumes:
        try:
            out = client.call("python", {
                "code": "import json; "
                        f"print(json.dumps(app.volume_appearance({v['id']!r})))"})
            looks.append(_python_stdout_json(out))
        except Exception as exc:
            looks.append({"_error": repr(exc)[:120]})

    props = expect.get("volume_props") or {}
    for key, want in props.items():
        ok = bool(looks) and all(l.get(key) == want for l in looks)
        find(f"volume.{key}", ok,
             "; ".join(f"{l.get(key)!r}" for l in looks) or "no volumes")

    if expect.get("grid_deliberate"):
        # The rendered grid must be a choice, not inherited decimation: either
        # full resolution (downsample 1) or deliberately carved (box set).
        ok = bool(looks) and all(
            (l.get("downsample") or 1) == 1 or l.get("box") for l in looks)
        find("grid_deliberate", ok,
             "; ".join(f"d{l.get('downsample')} box={bool(l.get('box'))}"
                       for l in looks) or "no volumes")

    if expect.get("regioned"):
        ok = bool(looks) and all(
            l.get("clip") or l.get("box") or l.get("mask_radius")
            or (l.get("sphere") or {}).get("mode") not in (None, "off")
            for l in looks)
        find("regioned", ok, json.dumps(
            [{k: l.get(k) for k in ("clip", "box", "mask_radius", "sphere")}
             for l in looks])[:200])

    if expect.get("negative_visible"):
        ok = bool(looks) and any(l.get("negative_visible") for l in looks)
        find("negative_visible", ok)

    if expect.get("model_clipped"):
        ok = False
        try:
            out = client.call("python", {
                "code": "import json; print(json.dumps("
                        "[len(s._clips) for m in app._models "
                        "for s in [m['session']]]))"})
            counts = _python_stdout_json(out)
            ok = bool(counts) and any(c > 0 for c in counts)
        except Exception as exc:
            findings.append(("model_clipped", None, repr(exc)[:120]))
        else:
            find("model_clipped", ok, str(counts))

    if expect.get("primitives"):
        try:
            out = client.call("python", {
                "code": "import json; print(json.dumps("
                        "[len(s._primitives) for m in app._models "
                        "for s in [m['session']]]))"})
            counts = _python_stdout_json(out)
            ok = bool(counts) and sum(counts) >= expect["primitives"]
            find("primitives", ok, str(counts))
        except Exception as exc:
            findings.append(("primitives", None, repr(exc)[:120]))

    if expect.get("no_job_errors"):
        errors = state.get("job_errors") or []
        find("no_job_errors", not errors,
             "; ".join(str(e)[:80] for e in errors[:3]))

    if expect.get("nonblank"):
        if png_bytes is None:
            find("nonblank", False, "screenshot() returned nothing")
        else:
            frac = _drawn_fraction(png_bytes)
            if frac is None:
                findings.append(("nonblank", None, "PIL unavailable"))
            else:
                find("nonblank", frac > 0.02,
                     f"{frac * 100:.1f}% of pixels differ from the border")

    return findings


def _python_stdout_json(payload):
    """``python(code)`` returns {stdout, ...}; pull the last stdout line and
    parse it."""
    if isinstance(payload, dict):
        stdout = payload.get("stdout") or ""
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line:
                return json.loads(line)
    return json.loads(payload) if isinstance(payload, str) else payload


def _drawn_fraction(png_bytes):
    """Fraction of pixels that differ from the dominant border color — a
    cheap "did anything draw" without a second model call."""
    try:
        import io

        from PIL import Image
    except ImportError:
        return None
    import numpy as np

    img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"),
                     dtype=np.int16)
    border = np.concatenate([img[0].reshape(-1, 3), img[-1].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    return float((np.abs(img - bg).max(axis=2) > 12).mean())


# -- agentic tier ----------------------------------------------------------

JUDGE_PROMPT = """You are judging a figure produced by another agent driving a \
molecular viewer (pxviewer) over MCP.

THE TASK IT WAS GIVEN:
{task}

EVIDENCE:
- The frame the viewer reports now: {critique_png}
- The image the maker itself saved: {figure_png}
- Deterministic checks run against the live app after the maker finished:
{checks}
- The tail of the maker's transcript:
{transcript}

Look at the image and decide whether the figure shows what the task asked \
for. Then say where any failure lies: a defect in the software itself, or \
the maker's own choices.

Answer with EXACTLY these three lines, nothing else:
VERDICT: pass | fail
FAULT: app-bug | agent-error | none
NOTES: <at most two sentences — what the image shows, and what is wrong if anything>"""


def judge_prompt(spec, critique_png, figure_png, checks, transcript_tail):
    checks_text = "\n".join(
        f"  {name}: {'PASS' if ok else ('n/a' if ok is None else 'FAIL')} {detail}"
        for name, ok, detail in checks)
    return JUDGE_PROMPT.format(
        task=spec["task"], critique_png=critique_png, figure_png=figure_png,
        checks=checks_text, transcript=transcript_tail)


VERDICT_RE = re.compile(r"VERDICT:\s*(pass|fail)", re.I)
FAULT_RE = re.compile(r"FAULT:\s*(app-bug|agent-error|none)", re.I)
NOTES_RE = re.compile(r"NOTES:\s*(.+)", re.I | re.S)


def parse_verdict(text):
    """Pull the three verdict lines out of the judge's reply."""
    verdict = VERDICT_RE.search(text or "")
    fault = FAULT_RE.search(text or "")
    notes = NOTES_RE.search(text or "")
    return {
        "verdict": verdict.group(1).lower() if verdict else "unparsed",
        "fault": fault.group(1).lower() if fault else "unknown",
        "notes": " ".join(notes.group(1).split())[:500] if notes else "",
        "raw": (text or "")[-2000:],
    }
