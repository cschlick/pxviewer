"""The embedded IPython console (optional 'console' extra)."""

from __future__ import absolute_import, division, print_function

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from pxviewer.regression.tst_utils import (_exercise_selected, have,
    qt_application, skip)

if not have("PySide6", "qtconsole", "ipykernel"):
    skip("PySide6 / qtconsole / ipykernel not available")


def exercise_console_available():
    from pxviewer import console

    assert console.console_available() is True


def exercise_the_embedded_console_shares_live_objects():
    """The in-process kernel evaluates against the very objects we push in —
    the eval round-trips over the real Jupyter wire now, but the namespace is
    the same one."""
    from pxviewer.console import AppKernel, EmbeddedConsole

    class FakeSession(object):
        marker = "live"

    kernel = AppKernel({"session": FakeSession(), "answer": 21})
    console = EmbeddedConsole(kernel)
    try:
        r = kernel.execute("session.marker", timeout=15)
        assert r["result"] == "'live'"          # the same object we pushed
        r = kernel.execute("answer * 2", timeout=15)
        assert r["result"] == "42"              # a real shell, not a stub
        console.push({"session": "rebound"})    # rebinding takes effect
        r = kernel.execute("session", timeout=15)
        assert r["result"] == "'rebound'"
    finally:
        console.shutdown()
        kernel.shutdown()


def exercise_external_clients_attach_over_the_wire():
    """The whole point of the wire kernel: ``jupyter console --existing`` (and
    the MCP ``python`` tool) see the same namespace the tab shows — proven by
    a bare client reading what it set."""
    import os
    import time

    from pxviewer.console import AppKernel

    kernel = AppKernel({"answer": 21})
    try:
        assert os.path.exists(kernel.connection_file)
        client = kernel.client()
        try:
            msg_id = client.execute("answer + 1")
            seen, deadline = None, time.time() + 30
            while time.time() < deadline:
                try:
                    msg = client.get_iopub_msg(timeout=0.5)
                except Exception:
                    continue
                if (msg["parent_header"].get("msg_id") == msg_id
                        and msg["msg_type"] == "execute_result"):
                    seen = msg["content"]["data"]["text/plain"]
                    break
            assert seen == "22", "external client saw %r" % seen
        finally:
            client.stop_channels()
    finally:
        kernel.shutdown()


def exercise_the_proxy_shows_real_signatures():
    """``inspect.signature(app.*)`` through the kernel proxy used to report
    ``(*args, **kwargs)`` for everything — an agent discovering the API saw
    nothing. ``__wrapped__`` is the channel signature() and help() follow."""
    import concurrent.futures
    import inspect

    from pxviewer.console import GuiThreadProxy

    class FakeApp:
        def focus_residue(self, chain: str, resid: str, *, focus: bool = True):
            """Select + focus a residue."""
            return 1

        attr = "value"

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    proxy = GuiThreadProxy(FakeApp(), pool.submit)
    try:
        sig = inspect.signature(proxy.focus_residue)
        assert "chain" in sig.parameters and "focus" in sig.parameters, sig
        assert proxy.focus_residue.__doc__ == "Select + focus a residue."
        assert proxy.attr == "value"
    finally:
        pool.shutdown(wait=True)


def exercise_the_console_suppresses_the_kernel_banner():
    """The widget squelches IPython's own banner so only our greeting shows."""
    from pxviewer.console import AppKernel, EmbeddedConsole

    kernel = AppKernel()
    console = EmbeddedConsole(kernel)
    try:
        # The kernel-info reply sets this trait; our observer must blank it out.
        console.widget.kernel_banner = "Python 3.12 ... IPython 9 ... Tip: ..."
        assert console.widget.kernel_banner == ""
    finally:
        console.shutdown()
        kernel.shutdown()


def exercise_the_banner_and_the_guide_fit_the_pane():
    """The console sits in the controls pane -- about 38 monospace columns. Anything
    wider wraps mid-sentence, which reads as a mess and is worse than saying less.
    Holds for the one-line banner and for the guide it points at."""
    from pxviewer.console import BANNER_MAX_COLUMNS, console_help, default_banner

    for name, text in (("banner", default_banner()), ("help()", console_help())):
        too_wide = [l for l in text.splitlines() if len(l) > BANNER_MAX_COLUMNS]
        assert not too_wide, "these wrap in the console (%s): %s" % (name, too_wide)


def exercise_the_banner_points_at_the_guide():
    """One line, naming the one thing to type. It used to be the whole guide squeezed
    into the pane's width, which is how it came to advertise app.group_mmm(g) with no
    way to obtain g."""
    from pxviewer.console import default_banner

    banner = default_banner()
    assert "help()" in banner
    assert len(banner.strip().splitlines()) == 1, banner
    assert "group_mmm" not in banner


def exercise_the_guide_runs_what_it_advertises():
    """Every cctbx expression in the guide is runnable *as written* against a real
    session -- no placeholder argument to invent. The old banner's app.group_mmm(g)
    was not, which is the whole reason this exists."""
    from pxviewer.console import CONSOLE_CCTBX, console_help
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.regression.tst_utils import data_path, dispose
    app = DesktopApp(port=0)
    try:
        app._add_model(LiveSession.from_model_file(data_path("1ubq.pdb")), "1ubq")
        session = app.active_model_session()
        scope = {"app": app, "session": session}
        for expression, _description in CONSOLE_CCTBX:
            eval(expression, {}, scope)          # noqa: S307 - our own literals
            assert expression in console_help(), expression
        # And the one the guide leans on hardest actually answers for a lone model:
        # None (no map), not an exception about a missing group.
        assert app.model_mmm() is None
    finally:
        dispose(app)


def exercise_dir_of_the_proxy_is_the_wrapped_surface():
    """dir(app) is a discovery path — if it returned proxy internals an agent
    could never introspect the AppCore surface."""
    from concurrent.futures import Future

    from pxviewer.console import GuiThreadProxy

    class Wrapped(object):
        def semantic_method(self):
            """A verb an agent would look for."""
            return 1

        plain_attribute = "value"

    def invoke(fn):
        fut = Future()
        fut.set_result(fn())
        return fut

    proxy = GuiThreadProxy(Wrapped(), invoke)
    names = dir(proxy)

    assert "semantic_method" in names
    assert "plain_attribute" in names
    # The proxy's own machinery is hidden — the listing is the wrapped object.
    assert "_px_invoke" not in names
    assert len(names) > 10


def exercise_bare_help_is_the_guide_and_help_of_a_thing_is_pythons():
    """Binding over the builtin has to take nothing away: help() is what someone types
    to learn what this console is, help(obj) is what they type for one docstring."""
    import contextlib
    import io

    from pxviewer.console import ConsoleHelp

    helper = ConsoleHelp()
    assert "cctbx objects underneath" in repr(helper)       # bare `help`, no parens

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        helper()
    assert "cctbx objects underneath" in buf.getvalue()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        helper(dict.get)
    text = buf.getvalue()
    assert "cctbx objects underneath" not in text, "help(obj) hijacked by the guide"
    assert "get" in text.lower()


def run():
    qt_application()        # the console widgets need one, created once per process
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise") and _exercise_selected(name):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
