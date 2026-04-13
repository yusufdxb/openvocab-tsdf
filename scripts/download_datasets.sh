#!/usr/bin/env bash
# Dataset fetch helpers. Nothing here is a dependency for running tests or the
# synthetic smoke demo -- those are self-contained.
#
# usage:
#   ./scripts/download_datasets.sh replica   # fetch the Replica RGB-D dump used by nice-slam
#   ./scripts/download_datasets.sh scannet   # placeholder; ScanNet requires signed terms

set -euo pipefail

DATA_ROOT="${OPENVOCAB_DATASETS_ROOT:-$HOME/data}"
mkdir -p "$DATA_ROOT"

fetch_replica() {
  local target="$DATA_ROOT/replica"
  mkdir -p "$target"
  echo "[replica] destination: $target"
  # The nice-slam project hosts a 12 GB tar with the rendered RGB-D dump used in
  # most Replica-based SLAM papers. Replace the URL when upstream moves it.
  local url="https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
  if [ -d "$target/room_0" ]; then
    echo "[replica] already present; skipping"
    return
  fi
  echo "[replica] downloading $url (~12 GB)..."
  curl -L --fail --continue-at - -o "$target/Replica.zip" "$url"
  echo "[replica] unzipping..."
  unzip -q "$target/Replica.zip" -d "$target"
  echo "[replica] done. scenes:"
  ls "$target"
}

fetch_scannet() {
  echo "[scannet] ScanNet requires signing a terms-of-use form."
  echo "[scannet] See https://github.com/ScanNet/ScanNet for instructions."
  echo "[scannet] After download, point OPENVOCAB_DATASETS_ROOT at the parent dir."
  exit 1
}

case "${1:-}" in
  replica) fetch_replica ;;
  scannet) fetch_scannet ;;
  "")      echo "usage: $0 {replica|scannet}"; exit 2 ;;
  *)       echo "unknown dataset: $1"; exit 2 ;;
esac
