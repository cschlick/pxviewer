#!/usr/bin/env python
"""The figfuzz orchestrator: spec -> maker agent -> critic -> verdict.

For each figure spec this spins up a real pxviewer desktop with an MCP
endpoint, hands the spec to an agent that must produce the figure through
MCP alone, then evaluates the result twice: deterministically against the
live app state (critic.evaluate) and by a second agent reading the frame
against the task (critic.judge). The whole loop is what makes it fuzzing
rather than scripting — the maker finds its own path, so it hits the API in
the orders and combinations a real agent chooses, and the critic judges the
outcome rather than the transcript.

    python scripts/figfuzz/run.py --all
    python scripts/figfuzz/run.py --count 6 --seed 3 --no-fetch
    python scripts/figfuzz/run.py --only bigmap --agent devin

Each run gets a disposable directory under --out holding the staged inputs,
the MCP config, the maker and judge transcripts, the screenshot the critic
took, and report.json. The app is terminated after the critic finishes.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "python"))

import critic  # noqa: E402
import specs as specmod  # noqa: E402
from mcpio import McpClient  # noqa: E402

REPO = HERE.parent.parent
DATA = REPO / "python" / "pxviewer" / "data"

MAKER_PROMPT = """You are producing ONE figure inside pxviewer — a molecular \
viewer you drive through its MCP server. The server (named "pxviewer") is \
configured for this directory; call its tools directly. Start with the \
status tool to see what is loaded, and prefer the python(code) tool for real \
work — the objects it exposes are the API.

TASK
{task}

CONTRACT
- Save the final image to {figure_png} using the viewer's save_screenshot \
tool — not a screen capture of anything else.
- The figure must be legible on its own: the subject framed and filling the \
view, a sensible level for the data, nothing irrelevant cluttering the frame.
- Do not work around the viewer: no edited copies of the inputs, no \
pre-clipped map files. Region, clipping, level and rendering controls exist \
— find and use them.
- Verify before you finish: take a screenshot through the tools and look at \
it; adjust until the frame is right.
- When the saved image is right, end with a line: DONE <what it shows>"""

#: Candidate locations for the devin CLI when it is not on PATH.
_DEVIN_CANDIDATES = [
    "/Applications/Devin.app/Contents/Resources/app/extensions/windsurf/devin/bin/devin",
]


def _agent_bin(requested):
    if requested and shutil.which(requested):
        return requested
    for path in _DEVIN_CANDIDATES:
        if Path(path).exists():
            return path
    return requested or "devin"


def _wait_port(port, timeout=120.0):
    """TCP-connectable as the readiness signal — the MCP endpoint only binds
    once Qt and the server are both up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


def _stage_files(rundir, stage, anchor=None):
    """Materialize a spec's synthetic maps. Shapes are deliberately padded off
    perfect powers of two so a decimated grid is visibly decimated.

    ``anchor`` is a model path the map should land on: the blob's box center is
    shifted to the model's centroid, so "density near the site" tasks actually
    have the density at the site rather than ~70 Å away in the map's own frame.
    """
    import numpy as np

    from pxviewer.volume_io import VolumeData

    centroid = None
    if anchor:
        from iotbx.pdb import input as pdb_input
        xyz = np.asarray(pdb_input(str(anchor)).atoms().extract_xyz())
        centroid = xyz.mean(axis=0)

    for item in stage:
        if item["kind"] == "multiconf":
            # True alternate conformers (altlocs) — what pxviewer's
            # conformer machinery acts on (model_conformers/set_model_conformer
            # address altloc labels; MODEL records are collapsed at load).
            # A stretch of residues gets a second side-chain pose at occupancy
            # 0.5, so "show only conformer A" has atoms to hide.
            lines = []
            src = Path(item["from"]).read_text().splitlines()
            atoms = [l for l in src if l.startswith(("ATOM", "HETATM"))]
            rng = np.random.default_rng(11)
            n_alt = item.get("alt_residues", 8)
            # Pick a contiguous interior run of protein residues.
            resseqs = sorted({int(l[22:26]) for l in atoms
                              if l[17:20].strip() not in ("HOH", "WAT")})
            run = resseqs[len(resseqs) // 3: len(resseqs) // 3 + n_alt]
            for l in atoms:
                resseq = int(l[22:26])
                atom = l[12:16].strip()
                is_protein = l[17:20].strip() not in ("HOH", "WAT")
                if resseq in run and is_protein and atom not in ("N", "CA", "C", "O"):
                    xyz = np.array([float(l[30:38]), float(l[38:46]),
                                    float(l[46:54])])
                    jit = rng.normal(0, 0.6, 3)
                    lines.append(f"{l[:16]}A{l[17:22]}{l[22:30]}"
                                 f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                                 f"  0.50{l[60:]}")
                    xyz2 = xyz + jit
                    lines.append(f"{l[:16]}B{l[17:22]}{l[22:30]}"
                                 f"{xyz2[0]:8.3f}{xyz2[1]:8.3f}{xyz2[2]:8.3f}"
                                 f"  0.50{l[60:]}")
                else:
                    lines.append(l)
            (rundir / item["path"]).write_text("\n".join(lines) + "\n")
            continue
        shape = item["shape"]
        rng = np.random.default_rng(7)
        grid = np.indices(shape, dtype=np.float32)
        center = np.array(shape, dtype=np.float32)[:, None, None, None] / 2
        r2 = ((grid - center) ** 2).sum(axis=0)
        width = min(shape) / 6.0
        if item["kind"] == "diff":
            off = np.array([min(shape) / 4, 0, 0])[:, None, None, None]
            arr = np.exp(-((grid - center - off) ** 2).sum(axis=0) / width ** 2)
            arr -= np.exp(-((grid - center + off) ** 2).sum(axis=0) / width ** 2)
        else:
            arr = np.exp(-r2 / (2 * width ** 2)) * 5.0
        arr = arr.astype(np.float32) + rng.normal(
            0, 0.02, shape).astype(np.float32)
        origin = (0, 0, 0)
        if centroid is not None:
            origin = tuple(int(round(c - s / 2.0))
                           for c, s in zip(centroid, shape))
        vol = VolumeData.from_numpy(arr, name=item["path"], origin=origin)
        vol.write_map(rundir / item["path"])


def _write_mcp_config(rundir, url):
    cfg_dir = rundir / ".devin"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "mcp_config.local.json").write_text(json.dumps(
        {"mcpServers": {"pxviewer": {"url": url}}}, indent=2))


def _run_agent(agent_argv, prompt, cwd, timeout, log_path, permission_mode):
    """One agent run: print mode, disposable workspace, transcript to log."""
    argv = agent_argv + ["--permission-mode", permission_mode,
                         "--respect-workspace-trust", "false",
                         "-p", prompt]
    with open(log_path, "wb") as log:
        try:
            proc = subprocess.run(argv, cwd=str(cwd), stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return code


def _tail(path, lines=60):
    try:
        return "\n".join(Path(path).read_text(errors="replace")
                         .splitlines()[-lines:])
    except OSError:
        return ""


def run_spec(spec, rundir, port, agent_argv, args):
    """One spec end to end; returns the report dict."""
    report = {"id": spec["id"], "title": spec["title"], "task": spec["task"],
              "checks": [], "verdict": None}
    rundir.mkdir(parents=True, exist_ok=True)
    figure_png = rundir / "figure.png"
    critique_png = rundir / "critique.png"

    _stage_files(rundir, spec["stage"], anchor=spec.get("anchor"))
    url = f"http://127.0.0.1:{port}/mcp"
    _write_mcp_config(rundir, url)

    app = subprocess.Popen(
        [sys.executable, "-m", "pxviewer", "desktop",
         "--port", str(port + 1000), "--mcp-port", str(port)],
        stdout=open(rundir / "app.log", "wb"), stderr=subprocess.STDOUT)
    report["app_log"] = "app.log"
    try:
        if not _wait_port(port, timeout=args.app_timeout):
            report["verdict"] = {"verdict": "fail", "fault": "app-bug",
                                 "notes": "the app never served its MCP endpoint"}
            return report

        prompt = MAKER_PROMPT.format(task=spec["task"], figure_png=figure_png)
        (rundir / "prompt.md").write_text(prompt)
        report["maker_exit"] = _run_agent(
            agent_argv, prompt, rundir, args.maker_timeout,
            rundir / "maker.log", args.permission_mode)

        # The critic's own frame — what the page draws *now*, not the file the
        # maker claims it wrote.
        png_bytes = None
        client = McpClient(url)
        try:
            shot = client.call("screenshot", timeout=60.0)
            if isinstance(shot, dict) and shot.get("images"):
                png_bytes = shot["images"][0]
                critique_png.write_bytes(png_bytes)
        except Exception as exc:
            report.setdefault("checks", []).append(
                ["screenshot", False, repr(exc)[:200]])

        checks = critic.evaluate(client, spec, png_bytes)
        report["checks"] = [list(f) for f in checks]
        report["figure_saved"] = figure_png.is_file()

        if not args.no_judge:
            judge_dir = rundir / "judge"
            judge_dir.mkdir(exist_ok=True)
            prompt = critic.judge_prompt(
                spec, critique_png, figure_png, checks,
                _tail(rundir / "maker.log"))
            (judge_dir / "prompt.md").write_text(prompt)
            _run_agent(agent_argv, prompt, judge_dir, args.judge_timeout,
                       judge_dir / "judge.log", args.permission_mode)
            report["verdict"] = critic.parse_verdict(
                _tail(judge_dir / "judge.log", lines=200))
        else:
            failed = [n for n, ok, _ in checks if ok is False]
            report["verdict"] = {
                "verdict": "fail" if failed else "pass",
                "fault": "unknown" if failed else "none",
                "notes": "deterministic checks only: " + (
                    ", ".join(failed) if failed else "all passed")}
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--all", action="store_true",
                     help="run the whole corpus once")
    src.add_argument("--count", type=int, default=6,
                     help="specs to draw (default 6)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--only", metavar="SUBSTR",
                        help="run only specs whose id/title match")
    parser.add_argument("--out", type=Path,
                        default=REPO / "build" / "figfuzz")
    parser.add_argument("--agent", default="devin",
                        help="agent CLI (default: devin, or the app-bundle binary)")
    parser.add_argument("--base-port", type=int, default=28760,
                        help="MCP port for the first spec; each run adds one")
    parser.add_argument("--app-timeout", type=float, default=120.0)
    parser.add_argument("--maker-timeout", type=float, default=900.0)
    parser.add_argument("--judge-timeout", type=float, default=300.0)
    parser.add_argument("--no-fetch", action="store_true",
                        help="skip specs that need the network")
    parser.add_argument("--no-judge", action="store_true",
                        help="deterministic checks only, no judge agent")
    parser.add_argument("--permission-mode", default="dangerous",
                        help="agent permission mode (default dangerous: MCP "
                             "calls must not prompt; the run dir is disposable)")
    parser.add_argument("--list", action="store_true",
                        help="print the resolved specs and exit")
    args = parser.parse_args()

    chosen = (specmod.corpus(str(DATA), seed=args.seed)
              if args.all or args.only
              else specmod.generate(str(DATA), args.count, args.seed))
    if args.no_fetch:
        chosen = [s for s in chosen if not s["fetch"]]
    if args.only:
        chosen = [s for s in chosen
                  if args.only in s["id"] or args.only in s["title"]]
    if not chosen:
        sys.exit("no specs matched")
    if args.list:
        for s in chosen:
            print(f"{s['id']}: {s['title']}\n    {s['task'][:140]}")
        return 0

    agent_argv = [_agent_bin(args.agent)]
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runroot = args.out / f"run-{stamp}"
    runroot.mkdir()
    print(f"figfuzz: {len(chosen)} spec(s) -> {runroot}")

    reports = []
    for i, spec in enumerate(chosen):
        rundir = runroot / spec["id"]
        print(f"[{i + 1}/{len(chosen)}] {spec['id']}: {spec['title']}", flush=True)
        try:
            report = run_spec(spec, rundir, args.base_port + i, agent_argv, args)
        except Exception as exc:  # a harness failure is a finding too
            report = {"id": spec["id"], "title": spec["title"],
                      "verdict": {"verdict": "fail", "fault": "unknown",
                                  "notes": f"harness error: {exc!r}"},
                      "checks": []}
        (rundir / "report.json").parent.mkdir(parents=True, exist_ok=True)
        (rundir / "report.json").write_text(json.dumps(report, indent=2))
        verdict = report["verdict"]
        print(f"    -> {verdict['verdict']} ({verdict['fault']}) {verdict['notes'][:120]}",
              flush=True)
        reports.append(report)

    (runroot / "summary.json").write_text(json.dumps(reports, indent=2))
    lines = ["# figfuzz summary", ""]
    for r in reports:
        v = r["verdict"]
        failed = [n for n, ok, _ in (r.get("checks") or []) if ok is False]
        lines.append(f"- **{r['id']}** {r['title']}: {v['verdict']}"
                     f" ({v['fault']}) {v['notes']}"
                     + (f" — checks: {', '.join(failed)}" if failed else ""))
    (runroot / "summary.md").write_text("\n".join(lines) + "\n")
    n_bad = sum(1 for r in reports if r["verdict"]["verdict"] != "pass")
    print(f"\n{len(reports) - n_bad}/{len(reports)} passed — {runroot}/summary.md")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
