#!/usr/bin/env bash
set -euo pipefail

VERSION='1.6.0'
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/emmylua-codeformat/$VERSION"
BIN="$CACHE_DIR/CodeFormat"

if [[ ! -x "$BIN" ]]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) ASSET='linux-x64' ;;
        Linux-aarch64) ASSET='linux-aarch64' ;;
        Darwin-x86_64) ASSET='darwin-x64' ;;
        Darwin-arm64) ASSET='darwin-arm64' ;;
        *)
            echo "emmylua-codeformat: unsupported platform $(uname -s)-$(uname -m)" >&2
            exit 1
            ;;
    esac

    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT

    curl -sL -o "$TMP/asset.tar.gz" \
        "https://github.com/CppCXY/EmmyLuaCodeStyle/releases/download/$VERSION/$ASSET.tar.gz"
    tar xzf "$TMP/asset.tar.gz" -C "$TMP"

    mkdir -p "$CACHE_DIR"
    mv "$TMP/$ASSET/bin/CodeFormat" "$BIN"
    chmod +x "$BIN"
fi

exec "$BIN" "$@"
