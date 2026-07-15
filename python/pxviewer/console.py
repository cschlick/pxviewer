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
    """The greeting shown at the top of the console."""
    return (
        "pxviewer console.  session = active model · app = desktop · np = numpy\n"
        "Type  api  for all commands · session.name? for help · session.<Tab> to explore.\n"
    )
