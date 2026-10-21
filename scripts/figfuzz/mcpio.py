"""A synchronous wrapper around the MCP streamable-http client.

The critic runs after the maker exits — a handful of tool calls against the
still-live endpoint. Async internals, sync surface: the caller wants
``client.call("status")`` returning the tool's parsed payload, not an event
loop of its own.
"""

from __future__ import absolute_import, division, print_function

import asyncio
import json


class McpClient(object):
    def __init__(self, url):
        self.url = url

    def call(self, tool, arguments=None, timeout=30.0):
        """Call a tool, returning its structured content or parsed text.

        Text content is JSON-decoded when it parses (``status`` and friends
        return JSON strings; ``python`` returns a JSON-ish dict as text).
        """
        return asyncio.run(self._call(tool, arguments or {}, timeout))

    async def _call(self, tool, arguments, timeout):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await asyncio.wait_for(
                    session.call_tool(tool, arguments), timeout)
                return _payload(result)


def _payload(result):
    """The useful part of a CallToolResult: structured content when present,
    else image bytes for image blocks, else the joined text (JSON-decoded if
    it parses)."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    import base64

    images = [base64.b64decode(c.data) for c in result.content
              if getattr(c, "type", None) == "image"]
    texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
    if images:
        return {"images": images, "text": "\n".join(texts)}
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except Exception:
        return joined
