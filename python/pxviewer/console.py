"""An embedded IPython console for the desktop app.

Rather than reflect the :class:`~pxviewer.live.LiveSession` API through a wall of
argument widgets, we drop the real thing in front of the user: an in-process
Jupyter kernel (so it shares the app's actual live objects) rendered by a
``qtconsole`` widget. The active session is bound as ``session`` and the desktop
app as ``app``, so the whole Python API — tab-completion, ``obj?`` help, history
and all — is available live against whatever is loaded in the viewport.

This is an optional feature: it needs ``qtconsole`` and ``ipykernel`` (the
``console`` extra). When they are absent the desktop shows an install hint
instead of the console.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

# ipykernel imports debugpy, which prints a frozen-modules warning under a
# debugger; silence it so it never lands in the user's console.
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")

# Bind qtconsole (through qtpy) to PySide6. Left to itself qtpy selects whatever Qt
# binding it finds first — historically PyQt5, which is GPL and so must not be loaded
# into, or shipped with, a closed-source build. Set before qtconsole (hence qtpy) is
# ever imported, which is why it sits at module top rather than inside the functions.
os.environ.setdefault("QT_API", "pyside6")

CONSOLE_MISSING_MESSAGE = (
    "The API console needs qtconsole and ipykernel.\n\n"
    "Install them with:\n    pip install 'pxviewer[console]'"
)


def console_available() -> bool:
    """Whether the optional console dependencies are importable."""
    try:
        import ipykernel  # noqa: F401
        import qtconsole  # noqa: F401

        return True
    except Exception:  # pragma: no cover - import error path
        return False


def _make_widget():
    """A RichJupyterWidget that shows only our banner, not IPython's.

    qtconsole appends the kernel's own banner (Python version, IPython version, a
    random tip) after our frontend banner. That banner is a ``Unicode`` trait set
    asynchronously from the kernel-info reply; observing it and clearing it keeps
    the greeting to just our own lines.
    """
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
    from traitlets import observe

    class _PxJupyterWidget(RichJupyterWidget):
        @observe("kernel_banner")
        def _suppress_kernel_banner(self, change):
            if change["new"]:
                self.kernel_banner = ""

    return _PxJupyterWidget()


class EmbeddedConsole:
    """An in-process IPython kernel wired to a ``RichJupyterWidget``.

    The kernel runs in this very process, so anything pushed into its namespace
    is the *same object* the app holds — evaluating ``session.highlight(...)`` in
    the console drives the live viewport directly.
    """

    def __init__(self, namespace: Optional[Mapping[str, Any]] = None, banner: Optional[str] = None):
        from qtconsole.inprocess import QtInProcessKernelManager

        self._manager = QtInProcessKernelManager()
        self._manager.start_kernel(show_banner=False)
        kernel = self._manager.kernel
        kernel.gui = "qt"
        if namespace:
            kernel.shell.push(dict(namespace))

        self._client = self._manager.client()
        self._client.start_channels()

        self.widget = _make_widget()
        self.widget.set_default_style("lightbg")  # white background
        # Set the banner before attaching the client, which is what triggers the
        # initial prompt (and banner) to be drawn.
        if banner:
            self.widget.banner = banner
        self.widget.kernel_manager = self._manager
        self.widget.kernel_client = self._client

    def push(self, mapping: Mapping[str, Any]) -> None:
        """Bind (or rebind) names in the kernel namespace — e.g. the active session."""
        if not mapping:
            return
        try:
            self._manager.kernel.shell.push(dict(mapping))
        except Exception:  # pragma: no cover - defensive
            pass

    def shutdown(self) -> None:
        """Stop the kernel and its channels. Idempotent."""
        try:
            self._client.stop_channels()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            self._manager.shutdown_kernel()
        except Exception:  # pragma: no cover - defensive
            pass


def default_banner() -> str:
    """The greeting shown at the top of the console: one line, and where to read more.

    It used to be the whole guide, compressed into the pane's width — which is how it
    came to advertise ``app.group_mmm(g)`` without ever saying what ``g`` was or how to
    get one. Four lines cannot both name a thing and explain it, so the banner now
    names one thing (:func:`console_help`) and the explaining happens there, at a
    length that can afford to be clear.
    """
    return "help() — what you can do here\n"


#: The console sits in the controls pane, which is a third of the screen — about 38
#: monospace columns. Anything wider wraps mid-sentence and reads as a mess, which is
#: worse than saying less. Holds for the banner and for help() alike.
BANNER_MAX_COLUMNS = 38


#: What the console binds, and what each one is *for*. Kept as data so the help text
#: and the test that checks it describes reality read the same list. One string per
#: entry: textwrap owns every line break, because hand-broken prose re-wrapped to the
#: pane's width breaks twice and reads like a ransom note.
CONSOLE_NAMES = [
    ("session", "The active model as the viewer sees it: selection, color, "
                "representations, measurements. Re-binds when you switch models."),
    ("app", "The desktop: every object loaded, and every action the panels take — "
            "loading, validation, minimization, maps."),
    ("api", "A categorized map of everything session can do. Type api, or "
            "api.find(\"color\")."),
]

#: The cctbx objects underneath, and the exact expression that reaches each. Every one
#: is runnable as written with no argument to invent -- which the old banner's
#: ``app.group_mmm(g)`` was not, and that is what this replaced. Descriptions are kept
#: short enough to sit on one line at the pane's width: the whole guide has to fit on
#: screen without scrolling, or it is back to being something nobody reads.
CONSOLE_CCTBX = [
    ("session.model", "the mmtbx model"),
    ("app.model_mmm()", "its map_model_manager, or None"),
    ("app.map_for_model()", "the map data, or None"),
    ("session.model.get_sites_cart()", "coordinates, as flex"),
]


def console_help(width: int = BANNER_MAX_COLUMNS) -> str:
    """The console's guide: what is bound, what each thing is, and how to reach cctbx.

    Wrapped to ``width`` because this prints into the controls pane, where prose that
    wraps mid-word is worse than prose that says less.
    """
    import textwrap

    def wrapped(text, indent):
        pad = " " * indent
        return textwrap.wrap(
            text, max(width, indent + 12),
            initial_indent=pad, subsequent_indent=pad,
            # "Re-binds" split across two lines, and an over-long expression would be
            # chopped mid-token: both read as typos rather than as wrapping.
            break_on_hyphens=False, break_long_words=False) or [pad.rstrip()]

    # No preamble: the banner above already said what this is, and every line here
    # costs one the guide needs to stay on a single screen.
    out = []
    for name, description in CONSOLE_NAMES:
        out += ([name] if not out else ["", name]) + wrapped(description, 2)
    out += ["", "cctbx objects underneath:"]
    for expression, description in CONSOLE_CCTBX:
        out += ["  " + expression] + wrapped(description, 4)
    # Nothing trails the list. The guide has to land on one screen alongside the
    # banner, the echoed call and the next prompt -- measured at 34 rows in the pane --
    # and a closing flourish is the first thing worth the two lines it costs.
    return "\n".join(out) + "\n"


class ConsoleHelp:
    """``help`` in the console: the guide when called bare, Python's help otherwise.

    Bound over the builtin deliberately, but without taking anything away — ``help()``
    is what someone types when they want to know what this console *is*, and
    ``help(obj)`` is what they type when they want the docstring of one thing. The
    builtin only answers the second, so this answers the first and hands the second
    straight back to it.
    """

    def __repr__(self) -> str:      # bare `help` at the prompt, no parentheses
        return console_help()

    def __call__(self, *args, **kwargs):
        if not args and not kwargs:
            print(console_help())
            return None
        import pydoc

        return pydoc.help(*args, **kwargs)
