"""Downloading an entry: progress reporting, caching, and failing readably.

Offline. ``iotbx.pdb.fetch.fetch`` is replaced with a fake that serves bytes from memory,
so this exercises pxviewer's streaming, gunzipping, ``.part`` handling and error wrapping
without a network -- which also means it cannot flake or spend 100 MB of someone's
bandwidth in CI.

What is worth pinning here is mostly about what the user sees while waiting. A half-map is
~50 MB, so a fetch read in one call is minutes of silence that looks exactly like a hang;
progress has to arrive during the download, not after it. And a download that dies partway
must not leave something that a later ``reuse_existing`` will mistake for a finished file.
"""

from __future__ import absolute_import, division, print_function

import gzip
import os
import sys

from pxviewer.regression.tst_utils import skip, tmp_dir

try:
    import iotbx.pdb.fetch as iotbx_fetch
except ImportError:
    skip("iotbx.pdb.fetch not available")

from pxviewer import fetch as F                                    # noqa: E402


class FakeResponse:
    """The subset of an HTTP response that :func:`fetch._stream_to_file` reads."""

    def __init__(self, payload, *, declare_length=True):
        self._payload = payload
        self._offset = 0
        self.headers = {"Content-Length": str(len(payload))} if declare_length else {}

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


class patched_fetch:
    """Replace ``iotbx.pdb.fetch.fetch`` for the duration of a block."""

    def __init__(self, handler):
        self._handler = handler
        self._saved = None

    def __enter__(self):
        self._saved = iotbx_fetch.fetch
        iotbx_fetch.fetch = self._handler
        return self

    def __exit__(self, *exc):
        iotbx_fetch.fetch = self._saved
        return False


def map_bytes(n=4096):
    """A gzipped payload, the shape a map arrives in (fetch() leaves maps compressed).

    Random content on purpose: repeating bytes compress to a few hundred bytes however
    many are asked for, which is not enough to span the 64 KB read used for streaming --
    so a test of chunked progress would see one chunk and prove nothing.
    """
    return gzip.compress(os.urandom(n))


def collect(events):
    return lambda entity, stage, done, total: events.append((entity, stage, done, total))


def exercise_progress_arrives_during_the_download_not_after():
    """The point of streaming: a caller can tell a slow download from a stalled one."""
    payload = map_bytes(1 << 18)
    events = []
    with patched_fetch(lambda *a, **k: FakeResponse(payload)):
        with tmp_dir() as work:
            F.fetch_entry(entities=["map"], work_dir=work, emdb_number="1234",
                          progress=collect(events))

    downloading = [e for e in events if e[1] == "downloading"]
    assert len(downloading) > 2, "only %d progress reports for %d bytes" % (
        len(downloading), len(payload))
    assert downloading[0][2] == 0, "progress should open at zero, not at the first chunk"
    assert [e[2] for e in downloading] == sorted(e[2] for e in downloading)
    assert all(e[3] == len(payload) for e in downloading), "declared total not reported"
    assert events[-1][1] == "done"
    assert any(e[1] == "decompressing" for e in events), "gunzip stage not announced"


def exercise_a_server_that_declares_no_length_still_reports_bytes():
    """Models arrive through a GzipFile wrapper, which has no headers to read a size from."""
    events = []
    with patched_fetch(lambda *a, **k: FakeResponse(b"CIF" * 5000, declare_length=False)):
        with tmp_dir() as work:
            F.fetch_entry(entities=["model"], work_dir=work, pdb_id="9r04",
                          progress=collect(events))
    downloading = [e for e in events if e[1] == "downloading"]
    assert downloading, "no progress at all without a Content-Length"
    assert all(e[3] is None for e in downloading), "invented a total"
    assert downloading[-1][2] == 15000


def exercise_the_map_lands_decompressed_and_whole():
    payload = map_bytes()
    with patched_fetch(lambda *a, **k: FakeResponse(payload)):
        with tmp_dir() as work:
            out = F.fetch_entry(entities=["map"], work_dir=work, emdb_number="1234")
            path = out["map"]
            assert path.name == "emd_1234.map"
            assert path.read_bytes() == gzip.decompress(payload)
            assert not list(path.parent.glob("*.part*")), "left a partial file behind"


def exercise_a_failed_download_leaves_nothing_to_mistake_for_a_finished_one():
    """The reason .part exists: reuse_existing must never adopt a truncated file."""
    def die(*args, **kwargs):
        raise OSError("connection reset")

    with patched_fetch(die):
        with tmp_dir() as work:
            try:
                F.fetch_entry(entities=["map"], work_dir=work, emdb_number="1234")
            except F.FetchError:
                pass
            else:
                raise AssertionError("a failed download did not raise FetchError")
            leftovers = sorted(p.name for p in work.iterdir()) if hasattr(work, "iterdir") \
                else os.listdir(str(work))
            assert not leftovers, "left %r behind" % (leftovers,)


def exercise_failures_say_which_file_and_what_to_do():
    class NotFound(Exception):
        code = 404

    with patched_fetch(lambda *a, **k: (_ for _ in ()).throw(NotFound("no such file"))):
        with tmp_dir() as work:
            try:
                F.fetch_entry(entities=["half_map_1"], work_dir=work, emdb_number="1234")
            except F.FetchError as exc:
                message = str(exc)
            else:
                raise AssertionError("expected FetchError")

    assert "half-map 1" in message, message      # which piece
    assert "EMD-1234" in message                 # which entry
    assert "half-maps are only deposited" in message, "404 gave no usable advice"


def exercise_reuse_existing_skips_a_file_already_there():
    payload = map_bytes()
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return FakeResponse(payload)

    with patched_fetch(counting):
        with tmp_dir() as work:
            F.fetch_entry(entities=["map"], work_dir=work, emdb_number="1234")
            assert len(calls) == 1

            events = []
            F.fetch_entry(entities=["map"], work_dir=work, emdb_number="1234",
                          reuse_existing=True, progress=collect(events))
            assert len(calls) == 1, "re-downloaded a file that was already present"
            assert [e[1] for e in events] == ["cached"]

            F.fetch_entry(entities=["map"], work_dir=work, emdb_number="1234")
            assert len(calls) == 2, "a plain re-fetch should still refresh"


def exercise_an_emdb_only_fetch_is_possible():
    """A regression: the stand-in PDB id has to satisfy iotbx's own validator.

    fetch_entry used to pass "0000" when the caller gave only an EMDB number, and
    iotbx.pdb.fetch rejects any id whose first character is not 1-9 -- so every
    EMDB-only fetch died with "Invalid pdb id 0000" before reaching the network.
    """
    assert iotbx_fetch.valid_pdb_id(F._EMDB_ONLY_PDB_ID), F._EMDB_ONLY_PDB_ID
    seen = {}

    def capture(pdb_id, **kwargs):
        seen["pdb_id"] = pdb_id
        seen["emdb_number"] = kwargs.get("emdb_number")
        return FakeResponse(map_bytes())

    with patched_fetch(capture):
        with tmp_dir() as work:
            F.fetch_entry(entities=["half_map_2"], work_dir=work, emdb_number="53478")
    assert seen["emdb_number"] == "53478"
    assert iotbx_fetch.get_link("rcsb", "em_half_map_2", pdb_id=seen["pdb_id"],
                                emdb_number="53478").endswith("half_map_2.map.gz")


def exercise_missing_prerequisites_are_refused_before_any_download():
    calls = []

    with patched_fetch(lambda *a, **k: calls.append(1)):
        with tmp_dir() as work:
            for kwargs in ({"entities": ["model"]},          # needs a pdb id
                           {"entities": []}):                # nothing selected
                try:
                    F.fetch_entry(work_dir=work, **kwargs)
                except ValueError:
                    pass
                else:
                    raise AssertionError("accepted %r" % (kwargs,))
    assert not calls, "started downloading before checking its inputs"


def exercise_byte_sizes_read_the_way_a_person_would():
    assert F.format_bytes(None) == "?"
    assert F.format_bytes(512) == "512 B"
    assert F.format_bytes(52035724) == "49.6 MB"
    assert F.describe("half_map_1") == "half-map 1"
    assert F.describe("unheard_of") == "unheard_of"


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
