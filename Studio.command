#!/usr/bin/env bash
set -euo pipefail
studio="$(cd "$(dirname "$0")" && pwd -P)"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  cat "$studio/scripts/help.txt"
  exit 0
fi
if [[ ${1:-} == --install || ! -f "$studio/runtime/.installed-v1" ]]; then
  bash "$studio/scripts/install.sh" "$studio"
  [[ ${1:-} != --install ]] || exit 0
fi
exec "$studio/runtime/venv/bin/python" "$studio/scripts/studio.py" "$@"
