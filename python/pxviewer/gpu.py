"""Pick a working WebGL backend for the QtWebEngine viewport, gracefully.

The viewport is Chromium (QtWebEngine), which needs a WebGL context to draw. On many
VMs the only GPU is a virtual adapter whose WebGL Chromium blocklists, so the context
fails — and it fails *late*, in Chromium's render process, as an unrecoverable error;
the flags that fix it must be set before ``QApplication`` starts. So the choice has to
be made up front, before we know whether it will work.

The strategy:

  - :func:`configure` runs before ``QApplication`` and decides the backend from an
    explicit choice (``--gpu`` / ``PXVIEWER_GPU``), a remembered verdict, or — by
    default — optimistically starts on hardware with an auto-fix armed.
  - Once the page has loaded, the app asks the *real* viewport whether it got a WebGL
    context (:func:`autofix_enabled` gates the check). If not, :func:`on_webgl_missing`
    remembers the verdict and re-execs the process once with the software flags, so the
    second start renders on SwiftShader (Chromium's CPU WebGL). :func:`mark_hardware_ok`
    remembers a success, so neither the check nor a probe runs next time.

The upshot: it just works — hardware where hardware works, software (slower, but
universal) where it doesn't — with one restart the first time on a bad GPU and none
after. A hand-set ``QTWEBENGINE_CHROMIUM_FLAGS`` is always left alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "SOFTWARE_FLAGS", "configure", "autofix_enabled", "on_webgl_missing",
    "mark_hardware_ok", "webgl_probe_js",
]

#: Chromium flags that route WebGL to the bundled SwiftShader (pure CPU), bypassing the
#: GPU and its blocklist entirely. This exact combination is what renders on VMs whose
#: virtual GPU has WebGL blocklisted.
SOFTWARE_FLAGS = ("--enable-unsafe-swiftshader --disable-gpu --ignore-gpu-blocklist "
                  "--use-gl=angle --use-angle=swiftshader")

#: JS returning whether a WebGL context can be created in the page — the real check,
#: run in the actual viewport once it has loaded.
webgl_probe_js = (
    "(function(){try{var c=document.createElement('canvas');"
    "return !!(c.getContext('webgl2')||c.getContext('webgl'));}"
    "catch(e){return false;}})()"
)

_MODES = ("auto", "hardware", "software")
_RETRY_ENV = "_PXVIEWER_GL_RETRIED"   # sentinel: set on the re-exec, so we never loop
_CACHE_VERSION = "1"                   # bump if SOFTWARE_FLAGS changes, to invalidate

# Module state, set by configure(): whether the app should check WebGL after load and
# fall back. False once a decision is final (custom flags, forced mode, cached verdict).
_STATE = {"autofix": False}


def resolve_mode(mode: Optional[str]) -> str:
    """The effective mode: explicit argument, else ``PXVIEWER_GPU``, else ``auto``."""
    chosen = (mode or os.environ.get("PXVIEWER_GPU") or "auto").strip().lower()
    if chosen not in _MODES:
        raise ValueError(f"gpu mode must be one of {_MODES}, not {chosen!r}")
    return chosen


def configure(mode: Optional[str] = None, *, log: Callable[[str], None] = print) -> str:
    """Decide the WebEngine backend before ``QApplication`` starts.

    Returns what was chosen — ``"hardware"``, ``"software"``, or ``"custom"`` (the user
    set ``QTWEBENGINE_CHROMIUM_FLAGS`` themselves). Sets the software flags when falling
    back, and arms the post-load check when the outcome is not yet known.
    """
    _STATE["autofix"] = False

    if os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS"):
        return "custom"  # the user has taken the wheel; do not second-guess them

    chosen = resolve_mode(mode)
    if chosen == "software":
        _enable_software()
        log("pxviewer: using software WebGL (SwiftShader), as requested.")
        return "software"
    if chosen == "hardware":
        # Trust the GPU and show its raw errors if it fails — the debugging path.
        return "hardware"

    # auto: trust a remembered verdict, else start on hardware with the check armed.
    remembered = _cached_verdict()
    if remembered == "software":
        _enable_software()
        log("pxviewer: using software WebGL (SwiftShader) — this machine had no usable "
            "hardware WebGL last time. Override with PXVIEWER_GPU=hardware.")
        return "software"
    if remembered == "hardware":
        return "hardware"
    _STATE["autofix"] = os.environ.get(_RETRY_ENV) is None
    return "hardware"


def autofix_enabled() -> bool:
    """Whether the app should verify WebGL after load (only when the outcome is unknown
    and we have not already retried)."""
    return _STATE["autofix"]


def mark_hardware_ok() -> None:
    """The viewport got a WebGL context: remember it, so no check runs next time."""
    if not _STATE["autofix"]:
        return
    _STATE["autofix"] = False
    _remember("hardware")


def on_webgl_missing(*, log: Callable[[str], None] = print) -> None:
    """The viewport could not get WebGL: remember it and restart on software rendering.

    Re-execs the same command with the software flags set (via ``PXVIEWER_GPU`` and a
    sentinel so the restart cannot loop). Called at most once per process.
    """
    if not _STATE["autofix"]:
        return
    _STATE["autofix"] = False
    _remember("software")
    log("pxviewer: the GPU could not provide WebGL (common on VMs) — restarting with "
        "software rendering (SwiftShader). Slower, but it works anywhere.")
    log("          Remembered for next time. To see the raw GPU errors instead, run "
        "with PXVIEWER_GPU=hardware.")
    sys.stdout.flush()
    os.environ[_RETRY_ENV] = "1"
    os.environ["PXVIEWER_GPU"] = "software"
    os.execv(sys.executable, _relaunch_argv())


# -- internals ---------------------------------------------------------------

def _enable_software() -> None:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = SOFTWARE_FLAGS


def _relaunch_argv() -> list:
    """Reconstruct the command that started us, for ``os.execv``."""
    if os.path.basename(sys.argv[0]) == "__main__.py":  # `python -m pxviewer …`
        return [sys.executable, "-m", "pxviewer", *sys.argv[1:]]
    return [sys.executable, *sys.argv]


def _signature() -> str:
    """A cheap, stable fingerprint of this machine's GPU, to key the remembered verdict.

    The GPU as the OS enumerates it — no GL init needed — plus OS/arch. If it cannot be
    read (non-Linux, no lspci) it falls back to OS/arch, which is fine: WebGL almost
    always works there, so the verdict is remembered as hardware after one success.
    """
    parts = [platform.system(), platform.machine()]
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=3).stdout
        parts += sorted(line for line in out.splitlines()
                        if re.search(r"VGA|3D controller|Display controller", line, re.I))
    except Exception:  # pragma: no cover - lspci absent / not Linux
        pass
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _cache_file() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "pxviewer" / "gpu.json"


def _cached_verdict() -> Optional[str]:
    """The remembered ``"hardware"``/``"software"`` verdict for this machine, or None."""
    try:
        data = json.loads(_cache_file().read_text())
    except Exception:
        return None
    if (data.get("version") == _CACHE_VERSION
            and data.get("signature") == _signature()
            and data.get("verdict") in ("hardware", "software")):
        return data["verdict"]
    return None


def _remember(verdict: str) -> None:
    try:
        path = _cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"version": _CACHE_VERSION, "signature": _signature(), "verdict": verdict}))
    except Exception:  # pragma: no cover - cache is best-effort
        pass
