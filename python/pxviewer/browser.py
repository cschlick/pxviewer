"""A Mol* page host backed by headless Chromium (Playwright).

The desktop uses ``QWebEngineView``; a GUI-less ``AppCore`` still wants the
same page — the wire protocol, in-page screenshots and Mol* semantics are
identical — so this hosts it in a browser instead. It is the ``_viewport``
seam: the core only ever calls ``load(url)``/``close()``.

Playwright's sync API owns a driver subprocess and must live on a dedicated
thread (it is a greenlet shim over asyncio; calling it on the Qt/app thread
would stall the event loop). ``load``/``close`` enqueue work there.

Optional dependency: ``pip install playwright && python -m playwright install
chromium`` (or ``chromium --only-shell``). The class raises ImportError at
construction if the package or browser is missing, so callers can degrade to
``_NullViewport`` — everything except pixel output still works.
"""

from __future__ import annotations

import os
import queue
import sys
import threading


def _browser_env() -> dict:
    """Environment for the Chromium process.

    conda-forge provides the shared libraries the browser needs (nspr, nss,
    atk, cups, … — install.sh fetches whatever ldd reports missing), but
    conda never puts the env's lib dir on the loader path: packages find
    their own deps by rpath. An external binary has none, so it needs
    LD_LIBRARY_PATH pointed at $CONDA_PREFIX/lib and the CDT sysroot dirs.
    """
    env = dict(os.environ)
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return env
    import glob
    libdirs = [os.path.join(prefix, "lib")]
    libdirs += glob.glob(
        os.path.join(prefix, "*-conda-linux-gnu/sysroot/lib*"))
    libdirs += glob.glob(
        os.path.join(prefix, "*-conda-linux-gnu/sysroot/usr/lib*"))
    libdirs = [d for d in libdirs if os.path.isdir(d)]
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        libdirs.append(existing)
    if libdirs:
        env["LD_LIBRARY_PATH"] = ":".join(libdirs)
    return env


def try_browser_viewport(**kwargs):
    """A :class:`BrowserViewport`, or None — with the reason printed.

    The headless fallback everywhere (mcp stdio, core entry): no browser is a
    degraded mode, not an error — every semantic operation still works, only
    pixel output (screenshots, saved views) needs the page.
    """
    try:
        return BrowserViewport(**kwargs)
    except Exception as exc:  # noqa: BLE001 — degrade, don't die
        print(f"pxviewer: no headless viewport ({exc}) — load/select/validate "
              "all work; for screenshots: pip install playwright && "
              "python -m playwright install chromium --only-shell",
              file=sys.stderr)
        return None


class BrowserViewport:
    """Headless Chromium running the Mol* webapp; navigated on ``load()``."""

    def __init__(self, width: int = 1600, height: int = 1200,
                 timeout: float = 60.0):
        """Launch the browser and a page; block until it can navigate.

        Bounded by ``timeout`` so a broken playwright install fails at
        startup (where the caller can fall back) rather than on first use.
        """
        self._jobs: "queue.Queue" = queue.Queue()
        self._dead = threading.Event()
        self._ready: "queue.Queue" = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run, args=(width, height), name="pxviewer-browser",
            daemon=True)
        self._thread.start()
        try:
            err = self._ready.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("headless browser did not come up in "
                               f"{timeout:.0f}s")
        if err is not None:
            raise err

    # -- the viewport seam ----------------------------------------------------

    def load(self, url: str) -> None:
        """Navigate the page — fire-and-forget, like ``QWebEngineView.load``.

        The app reloads this URL constantly as the scene is recomposed; the
        page's own ws reconnect drives everything after that. Errors land on
        stderr — a dropped navigation is no worse than a dropped widget load.
        """
        if self._dead.is_set():
            return
        self._jobs.put(url)

    def close(self) -> None:
        """Shut the browser down. Idempotent."""
        self._dead.set()
        self._jobs.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=10)

    # -- browser thread --------------------------------------------------------

    def _run(self, width: int, height: int) -> None:
        page = browser = pw = None
        try:
            from playwright.sync_api import Error, sync_playwright

            pw = sync_playwright().start()
            # Headless WebGL comes from SwANGLE (software ANGLE); the flags
            # keep it enabled where Chromium's blocklist would say no.
            browser = pw.chromium.launch(headless=True, env=_browser_env(),
                                         args=[
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
                "--use-gl=angle",
                "--use-angle=swiftshader",
            ])
            page = browser.new_context(
                viewport={"width": width, "height": height}).new_page()
            self._ready.put(None)
        except Exception as exc:  # noqa: BLE001 — surfaced to the caller
            self._ready.put(exc)
            self._dead.set()
            return

        while True:
            url = self._jobs.get()
            if url is None or self._dead.is_set():
                break
            try:
                # domcontentloaded, not "load": Mol* keeps booting async — the
                # ws handshake (what the app actually waits on) happens after.
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as exc:  # noqa: BLE001 — log and keep serving
                print(f"pxviewer: browser navigation failed: {exc}",
                      file=sys.stderr)

        try:
            browser.close()
            pw.stop()
        except Exception:  # noqa: BLE001 — teardown
            pass
