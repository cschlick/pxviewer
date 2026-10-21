# figfuzz — agentic figure fuzzing

A figure is where everything pxviewer does meets at once: loading, selections,
regions, levels, representations, camera, rendering, output. It is also where
agent users actually fail — not on a wrong call, but on a reasonable-looking
result that is wrong.

figfuzz fuzzes that whole loop with *agents in both seats*:

1. **spec** — `specs.py` holds a corpus of figure goals written in task
   language ("density only within ~15 Å of the ligand"), not method calls.
   The point is that the maker must find its own way, the way a real user
   would.
2. **maker** — `run.py` launches a real pxviewer desktop with an MCP
   endpoint per spec, drops an MCP config into a disposable run directory,
   and hands the spec to an agent CLI (`devin -p` by default) that must
   produce `figure.png` through the tools alone.
3. **critic** — two tiers (`critic.py`):
   - *deterministic*: against the live app — `ping()` is the page alive,
     `status()` did the objects land, `volume_appearance` is the rendered
     grid deliberate (not silently decimated), is the frame non-blank;
   - *agentic*: a second agent reads the spec, the frame the critic itself
     captured, the check findings, and the maker transcript tail, then
     returns `VERDICT: pass|fail`, `FAULT: app-bug|agent-error|none`,
     `NOTES: …`. The FAULT field is the deliverable: it separates defects
     in the software from fumbles by the maker.

## Running

```bash
# with the pxviewer env's python:
python scripts/figfuzz/run.py --all                  # whole corpus, once
python scripts/figfuzz/run.py --count 6 --seed 3     # seeded sample
python scripts/figfuzz/run.py --only bigmap          # one spec, for iterating
python scripts/figfuzz/run.py --list                 # specs, no run
python scripts/figfuzz/run.py --no-fetch --no-judge  # offline, checks only
```

Requires: the real desktop (a display), the `mcp` client package, PIL for the
blank-frame check, and an agent CLI — `devin` on PATH or the app-bundle
binary (auto-detected). Network only for the `--fetch` specs.

## Orchestrated mode — no CLI auth needed

Any agent that can launch subprocesses and call MCP tools can play
orchestrator instead of `run.py` — including a Devin session itself, whose
subagents inherit the session's auth and the project's MCP config. The loop
is the same, with every spec's app on the one configured port
(`pxviewer-live` = `127.0.0.1:8753`) since runs are serial:

1. stage the spec's files (`_stage_files`), launch
   `python -m pxviewer desktop --port <webapp> --mcp-port 8753`;
2. maker subagent: `MAKER_PROMPT` with "the pxviewer-live MCP server" for the
   config clause;
3. deterministic checks via `critic.evaluate` against `127.0.0.1:8753/mcp`;
4. judge subagent: `critic.judge_prompt`, reading `critique.png` and
   `figure.png` with its file tools;
5. kill the app, next spec.

`run.py --list` emits the spec texts to feed that loop.

## Output

`build/figfuzz/run-<timestamp>/` holds one directory per spec: staged inputs,
`prompt.md`, `maker.log`, `judge/judge.log`, `figure.png` (what the maker
saved), `critique.png` (what the app actually drew when asked afterwards),
`app.log`, and `report.json`. `summary.md`/`summary.json` aggregate the run.

## Tuning

`--maker-timeout` (default 15 min) bounds a spec; `--base-port` offsets the
MCP/webapp ports so a real pxviewer session can stay up; `--agent` swaps the
agent CLI; `--permission-mode` loosens down from `dangerous` if you want the
agent asked before non-MCP actions. Runs are serial — one desktop at a time.
