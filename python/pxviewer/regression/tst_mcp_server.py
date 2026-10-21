"""The MCP tool surface: one exec tool plus operational plumbing.

The semantic surface collapsed to ``python`` — code executes in the app's own
IPython kernel, the same namespace the Console tab shows (``session``, ``app``,
``api``, ``help``), so the API is the objects rather than a schema of
wrapped verbs. What remains here as named tools is what exec cannot do as well:
viewport liveness, page recovery and binary screenshot output. The contract this
test pins is therefore small on purpose — the kernel bindings are pinned by
``tst_console`` (CONSOLE_NAMES), not by the tool list.
"""

from __future__ import absolute_import, division, print_function

import asyncio
import sys

from pxviewer.regression.tst_utils import _exercise_selected, have, skip

if not have("mcp"):
    skip("mcp not available")


EXPECTED_TOOLS = {
    # the semantic surface: code exec in the app's own kernel
    "python",
    # scene + kernel state
    "status",
    # viewport liveness and recovery
    "ping", "reload_viewport",
    # the rendered frame out (inline and to disk)
    "screenshot", "save_screenshot",
}


def exercise_the_full_tool_surface_is_registered():
    from pxviewer import mcp_server

    server = mcp_server._make_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names, sorted(EXPECTED_TOOLS - names)
    # No strays either — a renamed tool would pass the subset check while
    # leaving the old name dead.
    assert names == EXPECTED_TOOLS, sorted(names - EXPECTED_TOOLS)


def exercise_tools_carry_docstrings():
    """An agent's only documentation is the tool list — every tool must say
    what it does."""
    from pxviewer import mcp_server

    server = mcp_server._make_server()
    tools = asyncio.run(server.list_tools())
    for t in tools:
        assert t.description and t.description.strip(), t.name


def exercise_the_handshake_carries_a_capability_inventory():
    """``instructions`` is the one text every agent reads at connect time, so the
    *existence* of the major capabilities has to be stated there — an agent
    cannot api.find() a verb it never suspects. Pin the task-language inventory:
    clipping, carving, decimation, selection, layered representations, sigma
    levels, coordinate editing, analysis, jobs and output."""
    from pxviewer import mcp_server

    instructions = mcp_server._make_server().instructions
    assert instructions, "no instructions on the server"

    for task_term in (
        "clip", "set_volume_sphere", "set_volume_box", "set_volume_downsample",
        "select_by_expression", "add_representation", "sigma",
        "minimize_model", "run_validation", "run_background", "screenshot",
    ):
        assert task_term in instructions, f"handshake never mentions {task_term!r}"

    # The trap this inventory exists to close: carving new map files instead of
    # using the live region tools — the text must steer away from it.
    assert "never make clipped or" in instructions

    # The figure guidance — the second observed trap: figures saved from an
    # unchecked draw state (a map still decimated from load, a level chosen by
    # habit). The text has to teach check-appearance → iterate-on-screenshot.
    for task_term in ("volume_appearance", "downsample", "max_sigma"):
        assert task_term in instructions, f"handshake never mentions {task_term!r}"


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise") and _exercise_selected(name):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
