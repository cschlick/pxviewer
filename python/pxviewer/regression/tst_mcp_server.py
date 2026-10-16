"""The MCP tool surface: the complete, self-sufficient API an agent gets.

The surface is the contract — an agent driving the viewer through MCP alone must
find every capability there, so the exercise checks the full tool list rather
than a sample. Adding a DesktopApp/LiveSession capability without a tool is a
gap; removing or renaming a tool breaks agents that learned it.
"""

from __future__ import absolute_import, division, print_function

import asyncio
import sys

from pxviewer.regression.tst_utils import have, skip

if not have("mcp"):
    skip("mcp not available")


EXPECTED_TOOLS = {
    # scene objects
    "load", "fetch", "load_view", "save_view", "write_object", "remove",
    "set_active_model", "set_object_visible", "make_maps", "pair_model_with_map",
    "generate_ligand_restraints",
    # appearance
    "set_representation", "set_type_hidden", "set_conformer", "color_by",
    "set_selected_hidden", "set_volume", "set_background", "set_attribute",
    # clipping
    "set_clip_slab", "set_clip_sphere",
    # interactions and overlays
    "set_computed_interactions", "set_model_interactions", "set_interactions",
    "set_clashes", "set_focus_surroundings", "set_localres",
    # selection and navigation
    "select", "clear_selection", "focus_residue", "step", "zoom_view",
    "reset_view", "set_camera", "mouse_selection", "wait_for_selection",
    # markup
    "measure", "add_label", "remove_primitive", "clear_primitives",
    "place_marker", "fit_ligand",
    # validation and editing
    "run_validation", "validation_results", "set_validation_markers",
    "minimize", "list_edits", "add_edit", "remove_edit", "clear_edits",
    "load_edits", "save_edits",
    # introspection (python-side state the view wire does not carry)
    "status", "ping", "selection_state", "atoms", "model_state",
    "volume_state", "reflections", "primitives", "structure_info",
    "secondary_structure",
    # raw data
    "atom_table", "model_text", "map_data", "reflection_data",
    # output
    "screenshot", "save_screenshot",
    # recovery
    "reload_viewport",
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


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
