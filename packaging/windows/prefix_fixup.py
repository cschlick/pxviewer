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
"""

import os
import sys


def main() -> int:
    old = sys.argv[1]
    env_dir = sys.argv[2]
    # conda files carry the prefix in both slash directions
    replacements = [
        (old.replace("/", "\\").encode("utf-8"), env_dir.replace("/", "\\").encode("utf-8")),
        (old.replace("\\", "/").encode("utf-8"), env_dir.replace("\\", "/").encode("utf-8")),
    ]
    rewritten = 0
    skipped_binary = []
    for dirpath, _dirs, files in os.walk(env_dir):
        for name in files:
            path = os.path.join(dirpath, name)
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
