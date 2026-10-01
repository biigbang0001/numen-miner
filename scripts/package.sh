#!/usr/bin/env bash
set -euo pipefail

VERSION="$1"
LINUX_BIN="$2"
WINDOWS_EXE="$3"
mkdir -p "$4"
OUT="$(cd "$4" && pwd)"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

pack() {
    tar -czf "$OUT/$1" --owner=0 --group=0 -C "$2" "$3"
}

linux="numen-miner-$VERSION-linux-x86_64"
mkdir "$STAGE/$linux"
install -m 755 "$LINUX_BIN" "$STAGE/$linux/numen-miner"
install -m 755 "$ROOT"/packaging/linux/*.sh "$STAGE/$linux"
install -m 644 "$ROOT/README.md" "$ROOT/LICENSE" "$STAGE/$linux"
pack "$linux.tar.gz" "$STAGE" "$linux"

windows="numen-miner-$VERSION-windows-x86_64"
mkdir "$STAGE/$windows"
cp "$WINDOWS_EXE" "$ROOT"/packaging/windows/*.bat "$ROOT/README.md" "$ROOT/LICENSE" "$STAGE/$windows"
(cd "$STAGE" && zip -qr "$OUT/$windows.zip" "$windows")

# HiveOS treats everything after the last dash in the archive name as the version.
rig="numen-miner-${VERSION//-/_}"

mkdir -p "$STAGE/hiveos/numen-miner"
install -m 755 "$LINUX_BIN" "$ROOT"/packaging/hiveos/h-*.sh "$STAGE/hiveos/numen-miner"
install -m 644 "$ROOT/packaging/hiveos/h-manifest.conf" "$STAGE/hiveos/numen-miner"
pack "${rig}_hiveos.tar.gz" "$STAGE/hiveos" numen-miner

mkdir -p "$STAGE/mmpos/numen-miner"
install -m 755 "$LINUX_BIN" "$ROOT/packaging/mmpos/mmp-stats.sh" "$STAGE/mmpos/numen-miner"
pack "${rig}_mmpos.tar.gz" "$STAGE/mmpos" numen-miner
