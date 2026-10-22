"""An embedded Jupyter kernel for the desktop app.

Rather than reflect the :class:`~pxviewer.live.LiveSession` API through a wall of
argument widgets, we drop the real thing in front of the user: an in-process
Jupyter kernel (so it shares the app's actual live objects). The kernel is a
real ``IPKernelApp`` — it listens on the standard Jupyter wire protocol and
writes a connection file, so *any* Jupyter frontend can attach: the Console tab
is just a built-in qtconsole client; ``jupyter console --existing <file>`` or a
notebook attaches the same way; and the MCP ``python`` tool executes against the
very same namespace. One kernel, many clients — the namespace, not the widget,
is the interface.

Threading: the kernel lives on its own daemon thread, so execution happens off
the Qt thread. ``LiveSession`` is thread-safe by design and is bound raw;
``app`` is bound through :class:`GuiThreadProxy`, which marshals every call
onto the Qt thread — the same context console code ran in before.

This is an optional feature: it needs ``ipykernel`` (and ``qtconsole`` for the
in-app widget) — the ``console`` extra. When absent the desktop shows an
install hint and the MCP ``python`` tool reports the missing extra.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
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


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class GuiThreadProxy:
    """A stand-in for a live object whose methods run on the Qt thread.

    Kernel-side code evaluates off the GUI thread. ``LiveSession`` is
    thread-safe by design, but ``DesktopApp`` methods can touch widgets —
    so ``app`` in the console namespace is wrapped: every callable attribute
    is queued onto the Qt thread through ``invoke`` and the caller blocks
    for its result. Attributes that are themselves QObjects come back
    wrapped too, so ``app.window.title()``-style chains stay marshalled.
    """

    def __init__(self, obj: Any, invoke) -> None:
        object.__setattr__(self, "_px_obj", obj)
        object.__setattr__(self, "_px_invoke", invoke)

    def _px_wrap(self, value):
        try:
            from PySide6.QtCore import QObject

            if isinstance(value, QObject):
                return GuiThreadProxy(value, self._px_invoke)
        except Exception:  # pragma: no cover - PySide6 always present here
            pass
        return value

    def __getattr__(self, name: str):
        attr = getattr(object.__getattribute__(self, "_px_obj"), name)
        invoke = object.__getattribute__(self, "_px_invoke")
        if callable(attr):
            def marshalled(*args, **kwargs):
                return invoke(lambda: attr(*args, **kwargs)).result(timeout=120)

            marshalled.__name__ = name
            marshalled.__doc__ = getattr(attr, "__doc__", None)
            # __wrapped__ is the functools.wraps channel: inspect.signature and
            # help() follow it to the real callable, so a proxied method shows
            # its own signature and docstring instead of (*args, **kwargs).
            marshalled.__wrapped__ = attr
            marshalled.__module__ = getattr(attr, "__module__", None)
            marshalled.__qualname__ = getattr(attr, "__qualname__", name)
            return marshalled
        return self._px_wrap(attr)

    def __setattr__(self, name: str, value: Any) -> None:
        invoke = object.__getattribute__(self, "_px_invoke")
        obj = object.__getattribute__(self, "_px_obj")
        invoke(lambda: setattr(obj, name, value)).result(timeout=30)

    def __dir__(self):
        return dir(object.__getattribute__(self, "_px_obj"))

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_px_obj"))


class AppKernel:
    """The app's in-process Jupyter kernel — one namespace every client shares.

    A real ``IPKernelApp`` on a daemon thread. Its connection file is the
    attach point for external Jupyter frontends (``jupyter console
    --existing``); the Console tab and the MCP ``python`` tool are just
    other clients of the same kernel.

    The whole lifecycle — ``initialize`` through ``start`` — runs on the
    kernel thread, because ipykernel binds its streams to that thread's
    IOLoop. ``push`` is the only mutation allowed from elsewhere.

    ``IPKernelApp`` is a process singleton (its sockets cannot be initialized
    twice), so a second ``AppKernel`` — a fresh ``DesktopApp`` in a test
    process, say — *adopts* the running kernel and rebinds its namespace to
    the new app rather than starting a second kernel.
    """

    def __init__(self, namespace: Optional[Mapping[str, Any]] = None):
        self._ready = threading.Event()
        self._exec_lock = threading.Lock()
        self._exec_client = None
        self._served = False
        self._app = None
        self._error = None
        ns = dict(namespace or {})

        from ipykernel.kernelapp import IPKernelApp

        if IPKernelApp.initialized():
            # The kernel is a process-lifetime resource: adopt it and rebind
            # the namespace — app/session names must mean THIS app.
            self._app = IPKernelApp.instance()
            self._thread = None
            if ns:
                self._app.kernel.shell.push(ns)
            self._ready.set()
            return

        def _run():
            try:
                app = IPKernelApp.instance()
                # init_signal installs a SIGINT handler, which only works on
                # the main thread — and is the app's job anyway, not the
                # kernel's. Without this it logs a harmless-but-scary
                # "signal only works in main thread" traceback on every start.
                app.init_signal = lambda: None
                # init_io wraps sys.stdout in an OutStream that echoes to the
                # real stream — that we want (exec'd print() reaches clients
                # AND the app's stdout). The fd-level capture on top of it
                # (watchfd: dup2 + a reader thread) instead steals the app's
                # own fd-1 writes, and when its reader dies, everything
                # printed through the process stops appearing at all.
                app.capture_fd_output = False
                # IPKernelApp defaults to quiet, which hands OutStream no echo
                # stream — so every print() in the process would stop reaching
                # the real stdout once the kernel starts. The app is not a
                # headless kernel; its prints still belong on its own stdout.
                app.quiet = False
                app.initialize([])
                self._app = app
                # initialize() replaces __main__ with the kernel's module;
                # embed_kernel undoes the same swap so the app's module
                # stays put.
                shell = app.kernel.shell
                main = getattr(shell, "_orig_sys_modules_main_mod", None)
                if main is not None:
                    name = getattr(shell, "_orig_sys_modules_main_name", "__main__")
                    sys.modules[name] = main
                if ns:
                    shell.push(ns)
                self._ready.set()
                app.start()  # blocks on this thread's IOLoop
            except BaseException as exc:  # surfaced to constructors callers
                self._error = exc
                self._ready.set()

        self._thread = threading.Thread(
            target=_run, name="pxviewer-kernel", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("timed out starting the IPython kernel")
        if self._error is not None:
            raise RuntimeError(f"IPython kernel failed to start: {self._error}")

    @property
    def connection_file(self) -> str:
        """The Jupyter connection file external clients attach with."""
        return self._app.abs_connection_file

    @property
    def kernel(self):
        return self._app.kernel

    def push(self, mapping: Mapping[str, Any]) -> None:
        """Bind or rebind names in the kernel namespace (e.g. active session).

        A direct dict update — atomic enough for rebinds, and synchronous, so a
        ``session`` pushed by a model switch is already there when the next
        exec dispatches.
        """
        if not mapping:
            return
        self._app.kernel.shell.push(dict(mapping))

    def client(self):
        """A new BlockingKernelClient attached to this kernel (caller owns it)."""
        from jupyter_client import BlockingKernelClient

        client = BlockingKernelClient()
        client.load_connection_file(self.connection_file)
        client.start_channels()
        return client

    def _exec(self):
        """The shared client for execute(); created on first use."""
        if self._exec_client is None:
            self._exec_client = self.client()
        return self._exec_client

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """True once the kernel answers a kernel_info request."""
        client = self._exec()
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg_id = client.kernel_info()
                inner = time.time() + 2.0
                while time.time() < inner:
                    msg = client.get_shell_msg(timeout=max(inner - time.time(), 0.1))
                    if msg["parent_header"].get("msg_id") == msg_id:
                        return True
            except Exception:
                time.sleep(0.1)
        return False

    def execute(self, code: str, timeout: float = 120.0) -> dict:
        """Run ``code`` in the kernel namespace; return structured output.

        ``result`` is the ``text/plain`` of the last expression; ``data`` is
        the full mimebundle (so an IPython ``Image`` shows up as
        ``image/png``). Errors come back structured, not raised — an exec
        tool's caller needs the traceback, not the exception's loss.
        """
        with self._exec_lock:
            client = self._exec()
            if not self._served:
                # First exec races the kernel's stream wiring — handshake
                # before sending, or the request is silently dropped.
                self._served = self.wait_ready(timeout=30.0)
                if not self._served:
                    return {"status": "error", "stdout": "", "stderr": "",
                            "result": None, "data": {},
                            "error": {"ename": "KernelNotReady",
                                      "evalue": "kernel_info went unanswered "
                                                "for 30s after start",
                                      "traceback": []}}
            msg_id = client.execute(code, allow_stdin=False)
            out = {"status": "ok", "stdout": "", "stderr": "",
                   "result": None, "data": {}, "error": None}
            # The shell reply is authoritative for completion — it is a
            # request/reply channel, so it always arrives. iopub is a SUB
            # socket: a client attached moments ago can miss early messages
            # (slow joiner), so "wait for idle" alone can hang on a finished
            # cell. Collect output until reply *and* idle, with a short
            # grace tail after the reply for output already in flight.
            deadline = time.time() + timeout
            got_reply = False
            saw_idle = False
            reply_at = None
            while True:
                now = time.time()
                if now >= deadline:
                    out["status"] = "timeout"
                    out["error"] = {
                        "ename": "Timeout",
                        "evalue": f"execution exceeded {timeout}s — the kernel is still busy",
                        "traceback": [],
                    }
                    return out
                if got_reply and (saw_idle or (reply_at and now - reply_at > 0.5)):
                    return out
                try:
                    msg = client.get_iopub_msg(timeout=0.2)
                    if msg["parent_header"].get("msg_id") == msg_id:
                        kind = msg["msg_type"]
                        content = msg["content"]
                        if (kind == "status"
                                and content.get("execution_state") == "idle"):
                            saw_idle = True
                        elif kind == "stream":
                            key = ("stderr" if content.get("name") == "stderr"
                                   else "stdout")
                            out[key] += content.get("text", "")
                        elif kind in ("execute_result", "display_data"):
                            out["data"] = content.get("data", {})
                            if "text/plain" in out["data"]:
                                out["result"] = out["data"]["text/plain"]
                        elif kind == "error":
                            out["status"] = "error"
                            out["error"] = {
                                "ename": content.get("ename"),
                                "evalue": content.get("evalue"),
                                "traceback": [_ANSI.sub("", l) for l in
                                              content.get("traceback", [])],
                            }
                except Exception:
                    pass
                if not got_reply:
                    try:
                        reply = client.get_shell_msg(timeout=0.05)
                        if reply["parent_header"].get("msg_id") == msg_id:
                            got_reply = True
                            reply_at = time.time()
                            content = reply["content"]
                            status = content.get("status")
                            if status == "error" and out["error"] is None:
                                out["status"] = "error"
                                out["error"] = {
                                    "ename": content.get("ename"),
                                    "evalue": content.get("evalue"),
                                    "traceback": [_ANSI.sub("", l) for l in
                                                  content.get("traceback", [])],
                                }
                            elif status and status != "ok":
                                out["status"] = status
                    except Exception:
                        pass

    def shutdown(self) -> None:
        """Detach our exec client. The kernel itself is a process-lifetime
        resource — ``IPKernelApp`` is a singleton whose sockets cannot be
        initialized twice, so killing it would strand every later AppKernel
        (a second app in one test process). The daemon thread exits with the
        process; this just drops our channel."""
        if self._exec_client is not None:
            try:
                self._exec_client.stop_channels()
            except Exception:  # pragma: no cover - teardown races
                pass
            self._exec_client = None


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
    """The Console tab: a qtconsole widget attached to the app's ``AppKernel``.

    The widget is an ordinary Jupyter client — it connects to the kernel over
    the same connection file an external ``jupyter console --existing`` would
    use. Execution therefore happens on the kernel thread, not in the widget;
    see :class:`AppKernel` for the threading contract.

    ``shutdown`` only detaches this client — the kernel is the app's shared
    resource (MCP and external frontends may still be attached), so kernel
    teardown belongs to the app, not to the tab.
    """

    def __init__(self, kernel: AppKernel, banner: Optional[str] = None):
        from qtconsole.manager import QtKernelManager

        self._kernel = kernel
        self._manager = QtKernelManager()
        self._manager.load_connection_file(kernel.connection_file)
        self._client = self._manager.client()
        self._client.load_connection_file(kernel.connection_file)
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
        self._kernel.push(mapping)

    def shutdown(self) -> None:
        """Detach this client. The kernel keeps serving other clients."""
        try:
            self._client.stop_channels()
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
                "representations, measurements. Re-binds when you switch "
                "models. Reps layer: set_representation replaces, "
                "add_representation(on=selection) stacks a subset on top."),
    ("app", "The desktop: every object loaded, and every action the panels take — "
            "loading, validation, minimization, maps."),
    ("api", "A categorized map of everything session and app can do. Type api, "
            "or api.find(\"color\")."),
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
