#!/usr/bin/env bash
set -euo pipefail
studio="$1"
rt="$studio/runtime"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform=aarch64-apple-darwin; hash=e419bf07533bdbf48795fc793601c834dd79db57a8c49ce8e25cfe3a8639958d ;;
  Linux-x86_64) platform=x86_64-unknown-linux-gnu; hash=bf65ac1a0af10f92c522d91ac096c7c0a39657647b155577e7836c2cbbb4526c ;;
  *) echo 'This build supports Apple Silicon macOS or x86-64 Linux with compatible GPU drivers.' >&2; exit 1 ;;
esac
mkdir -p "$rt/tools/bin" "$rt/downloads"
uv="$rt/tools/bin/uv"
if [[ ! -x "$uv" ]]; then
  archive="$rt/downloads/uv-$platform.tar.gz"
  curl -fL --retry 4 -o "$archive" "https://github.com/astral-sh/uv/releases/download/0.13.10/uv-$platform.tar.gz"
  if command -v sha256sum >/dev/null; then actual=$(sha256sum "$archive"); else actual=$(shasum -a 256 "$archive"); fi
  [[ "${actual%% *}" == "$hash" ]] || { echo 'uv checksum mismatch' >&2; exit 1; }
  tar -xzf "$archive" -C "$rt/tools/bin" --strip-components=1 "uv-$platform/uv"
fi
export UV_PYTHON_INSTALL_DIR="$rt/python" UV_CACHE_DIR="$rt/downloads/uv-cache"
if [[ ! -x "$rt/venv/bin/python" ]]; then
  "$uv" python install 3.12
  "$uv" venv --python "$("$uv" python find --managed-python 3.12)" "$rt/venv"
fi
exec "$rt/venv/bin/python" "$studio/scripts/install.py"
