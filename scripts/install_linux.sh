#!/usr/bin/env bash
# Kept for compatibility — the installer is now portable (macOS + Linux) and
# lives in install.sh. This shim just forwards.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh" "$@"
