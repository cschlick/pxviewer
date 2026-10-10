"""Rewrite the build-time env prefix to the installed location.

Conda embeds the env's creation path in config files — qt*.conf, Jupyter
kernelspecs, .pth files, dist-info records, conda-meta. The env was created in
CI at BUILD_PREFIX; the installer drops it wherever the user chose. This runs
once post-install and rewrites every file containing the prefix, in either
slash direction.

Only files that decode as UTF-8 are rewritten. Binary files that embed the
prefix (Scripts\\*.exe shims) are skipped: rewriting a byte string to a
different length would corrupt them, and nothing in the app launches them —
the shortcut calls pythonw.exe directly.

    prefix_fixup.py <old prefix> <env dir> [--write-list]

The env holds ~300k files (5+ GB) and only a few hundred contain the prefix, so
the build runs this once with --write-list against the staged env: it records
those files in LIST_NAME and rewrites nothing. At install time, with the list
present, only the listed files are read; without it, every file is scanned.
"""

import os
import sys

LIST_NAME = ".pxviewer-prefix-files"


def _scan(env_dir, needles):
    """Relative paths of the files under env_dir containing any needle."""
    found = []
    for dirpath, _dirs, files in os.walk(env_dir):
        for name in files:
            path = os.path.join(dirpath, name)
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if any(n in data for n in needles):
                found.append(os.path.relpath(path, env_dir))
    return found


def main() -> int:
    old = sys.argv[1]
    env_dir = sys.argv[2]
    # conda files carry the prefix in both slash directions
    replacements = [
        (old.replace("/", "\\").encode("utf-8"), env_dir.replace("/", "\\").encode("utf-8")),
        (old.replace("\\", "/").encode("utf-8"), env_dir.replace("\\", "/").encode("utf-8")),
    ]
    list_path = os.path.join(env_dir, LIST_NAME)

    if "--write-list" in sys.argv[3:]:
        found = _scan(env_dir, [o for o, _ in replacements])
        with open(list_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(found) + "\n")
        print(f"prefix fixup: listed {len(found)} files containing the build prefix")
        return 0

    if os.path.isfile(list_path):
        with open(list_path, encoding="utf-8") as fh:
            candidates = [os.path.join(env_dir, line) for line in fh.read().splitlines() if line]
    else:
        candidates = [os.path.join(env_dir, rel) for rel in _scan(env_dir, [o for o, _ in replacements])]

    rewritten = 0
    skipped_binary = []
    for path in candidates:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        if not any(o in data for o, _ in replacements):
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary.append(path)
            continue
        for o, n in replacements:
            data = data.replace(o, n)
        try:
            with open(path, "wb") as fh:
                fh.write(data)
            rewritten += 1
        except OSError:
            pass
    marker = os.path.join(env_dir, ".pxviewer-prefix")
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(env_dir)
    print(f"prefix fixup: rewrote {rewritten} files under {env_dir}")
    if skipped_binary:
        print(f"skipped {len(skipped_binary)} binary files containing the old prefix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
