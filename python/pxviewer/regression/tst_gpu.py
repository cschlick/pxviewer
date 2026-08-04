"""The WebGL-backend chooser.

The QtWebEngine half cannot be exercised headlessly, but the decision logic can, and it
is what actually governs behaviour: mode precedence, leaving hand-set flags alone,
remembering a verdict across launches, and the one-shot restart that must not loop.

Everything here is real -- real environment variables, a real cache file under a real
temporary ``XDG_CACHE_HOME``. The single exception is ``os.execv``, which is passed in as
``restart`` because it does not return: a test that let it fire would be replaced by the
process it launched, and would never reach its assertions.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import os
import sys

from libtbx.test_utils import raises

from pxviewer import gpu
from pxviewer.regression.tst_utils import tmp_dir

#: Everything the chooser reads. Cleared per exercise so one cannot leak into the next --
#: the module decides from the environment, so a stale variable is a wrong answer.
GPU_ENV = ("QTWEBENGINE_CHROMIUM_FLAGS", "PXVIEWER_GPU", "_PXVIEWER_GL_RETRIED",
           "XDG_CACHE_HOME")


def quiet(*args):
    """A log sink: these functions narrate to the user, which is not under test."""


@contextlib.contextmanager
def clean_environment():
    """A cleared environment and a throwaway cache directory, restored afterwards.

    The cache is a real file in a real directory rather than a stubbed reader, so the
    round trip through ``_remember`` and ``_cached_verdict`` -- including the machine
    signature that keys it -- is part of what these exercises cover.
    """
    saved = dict((name, os.environ.get(name)) for name in GPU_ENV)
    for name in GPU_ENV:
        os.environ.pop(name, None)
    try:
        with tmp_dir() as cache:
            os.environ["XDG_CACHE_HOME"] = cache
            gpu._STATE["autofix"] = False
            yield cache
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        gpu._STATE["autofix"] = False


def flags():
    return os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS")


class Recording_restart(object):
    """Stands in for ``os.execv``, which would replace this process."""

    def __init__(self):
        self.calls = []

    def __call__(self, path, argv):
        self.calls.append((path, argv))


# -- choosing a mode ----------------------------------------------------------


def exercise_resolve_mode_precedence():
    """An explicit choice beats the environment, which beats the default."""
    with clean_environment():
        assert gpu.resolve_mode("software") == "software"
        os.environ["PXVIEWER_GPU"] = "hardware"
        assert gpu.resolve_mode(None) == "hardware"
        assert gpu.resolve_mode("software") == "software"      # explicit still wins
        del os.environ["PXVIEWER_GPU"]
        assert gpu.resolve_mode(None) == "auto"


def exercise_resolve_mode_rejects_garbage():
    with clean_environment():
        with raises(ValueError):
            gpu.resolve_mode("turbo")


def exercise_custom_flags_are_left_alone():
    """A hand-set QTWEBENGINE_CHROMIUM_FLAGS means the user has taken the wheel."""
    with clean_environment():
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--my-own-flags"

        assert gpu.configure("auto", log=quiet) == "custom"
        assert flags() == "--my-own-flags"                     # not appended to, not replaced
        assert not gpu.autofix_enabled()                       # and no second-guessing later


def exercise_software_mode_sets_the_flags():
    with clean_environment():
        assert gpu.configure("software", log=quiet) == "software"
        assert flags() == gpu.SOFTWARE_FLAGS
        assert not gpu.autofix_enabled()                       # the decision is final


def exercise_hardware_mode_sets_nothing_and_does_not_arm():
    """The debugging path: trust the GPU and show its raw errors if it fails."""
    with clean_environment():
        assert gpu.configure("hardware", log=quiet) == "hardware"
        assert flags() is None
        assert not gpu.autofix_enabled()


def exercise_auto_without_a_cached_verdict_arms_the_check():
    """The optimistic default: start on hardware, and verify once the page has loaded."""
    with clean_environment():
        assert gpu.configure("auto", log=quiet) == "hardware"
        assert flags() is None
        assert gpu.autofix_enabled()


def exercise_auto_does_not_arm_after_a_retry():
    """The sentinel the re-exec sets. Without it the child would arm the check again,
    fail it again, and restart forever."""
    with clean_environment():
        os.environ["_PXVIEWER_GL_RETRIED"] = "1"
        assert gpu.configure("auto", log=quiet) == "hardware"
        assert not gpu.autofix_enabled()


# -- remembering the verdict --------------------------------------------------


def exercise_marking_hardware_ok_is_remembered_across_launches():
    """A success is written to the cache, so neither the check nor a probe runs again."""
    with clean_environment():
        gpu.configure("auto", log=quiet)
        assert gpu.autofix_enabled()

        gpu.mark_hardware_ok()
        assert not gpu.autofix_enabled()

        # A fresh configure trusts the remembered verdict: no check, no flags.
        assert gpu.configure("auto", log=quiet) == "hardware"
        assert not gpu.autofix_enabled()
        assert flags() is None


def exercise_marking_hardware_ok_does_nothing_when_not_armed():
    """Only an armed check has an outcome to record. Writing one from a forced hardware
    run would remember a verdict that was never tested."""
    with clean_environment() as cache:
        gpu.configure("hardware", log=quiet)
        gpu.mark_hardware_ok()
        assert not os.path.exists(os.path.join(cache, "pxviewer", "gpu.json"))


def exercise_missing_webgl_remembers_software_and_restarts():
    """The fallback in full: remember the verdict, mark the child so it cannot loop,
    force software for the relaunch, and re-exec."""
    with clean_environment():
        restart = Recording_restart()
        gpu.configure("auto", log=quiet)
        gpu.on_webgl_missing(log=quiet, restart=restart)

        assert len(restart.calls) == 1
        path, argv = restart.calls[0]
        assert path == sys.executable
        assert argv and argv[0] == sys.executable

        assert os.environ["_PXVIEWER_GL_RETRIED"] == "1"       # the child cannot loop
        assert os.environ["PXVIEWER_GPU"] == "software"        # and starts on software

        # And the verdict outlives the process: the *next* launch goes straight there.
        del os.environ["PXVIEWER_GPU"]
        del os.environ["_PXVIEWER_GL_RETRIED"]
        assert gpu.configure("auto", log=quiet) == "software"
        assert flags() == gpu.SOFTWARE_FLAGS


def exercise_missing_webgl_is_a_noop_when_not_armed():
    """A forced hardware run reports its own errors; it must not restart behind them."""
    with clean_environment():
        restart = Recording_restart()
        gpu.configure("hardware", log=quiet)
        gpu.on_webgl_missing(log=quiet, restart=restart)
        assert restart.calls == []


def exercise_the_restart_happens_at_most_once_per_process():
    """``on_webgl_missing`` disarms before it re-execs, so a second call is inert even if
    the viewport reports the failure twice."""
    with clean_environment():
        restart = Recording_restart()
        gpu.configure("auto", log=quiet)
        gpu.on_webgl_missing(log=quiet, restart=restart)
        gpu.on_webgl_missing(log=quiet, restart=restart)
        assert len(restart.calls) == 1


def exercise_a_cache_from_another_machine_is_ignored():
    """The verdict is keyed to a GPU signature, so a cache directory carried to a
    different machine -- an NFS home, a copied image -- is not trusted."""
    import json

    with clean_environment() as cache:
        gpu.configure("auto", log=quiet)
        gpu.mark_hardware_ok()

        path = os.path.join(cache, "pxviewer", "gpu.json")
        data = json.loads(open(path).read())
        assert data["verdict"] == "hardware"
        data["signature"] = "not-this-machine"
        open(path, "w").write(json.dumps(data))

        # Unrecognised: fall back to arming the check rather than trusting it.
        assert gpu.configure("auto", log=quiet) == "hardware"
        assert gpu.autofix_enabled()


def exercise_a_cache_from_an_older_version_is_ignored():
    """``_CACHE_VERSION`` is bumped when SOFTWARE_FLAGS changes, which invalidates every
    remembered verdict -- the flags that were tested are no longer the flags that run."""
    import json

    with clean_environment() as cache:
        gpu.configure("auto", log=quiet)
        gpu.mark_hardware_ok()

        path = os.path.join(cache, "pxviewer", "gpu.json")
        data = json.loads(open(path).read())
        data["version"] = "0"
        open(path, "w").write(json.dumps(data))

        assert gpu.configure("auto", log=quiet) == "hardware"
        assert gpu.autofix_enabled()


def exercise_an_unreadable_cache_is_not_fatal():
    """Remembering is best-effort: garbage in the file must not stop the app starting."""
    with clean_environment() as cache:
        directory = os.path.join(cache, "pxviewer")
        os.makedirs(directory)
        open(os.path.join(directory, "gpu.json"), "w").write("{not json")

        assert gpu.configure("auto", log=quiet) == "hardware"
        assert gpu.autofix_enabled()


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
