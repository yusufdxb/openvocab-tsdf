#!/usr/bin/env bash
# Download the LSeg minimal checkpoint (~400 MB) from the Intel ISL release.
# Skips if the file already exists locally.

set -euo pipefail

WEIGHTS_DIR="${HOME}/.cache/openvocab_tsdf/weights"
CKPT_PATH="${WEIGHTS_DIR}/lseg_minimal_e200.ckpt"
EXPECTED_SHA256="TODO_FILL_AFTER_FIRST_DOWNLOAD"

if [ -f "${CKPT_PATH}" ]; then
    echo "LSeg weights already exist at ${CKPT_PATH}"
    exit 0
fi

mkdir -p "${WEIGHTS_DIR}"

echo "Downloading LSeg minimal checkpoint (~400 MB)..."
echo ""
echo "The checkpoint comes from the Intel ISL LSeg release."
echo "If the download URL is stale, manually download from:"
echo "  https://github.com/isl-org/lang-seg"
echo "and place the file at: ${CKPT_PATH}"
echo ""

# Google Drive file ID for lseg_minimal_e200.ckpt
FILEID="1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
curl -L "https://drive.google.com/uc?export=download&id=${FILEID}&confirm=t" \
    -o "${CKPT_PATH}"

echo "Downloaded to ${CKPT_PATH}"
echo "Size: $(du -h "${CKPT_PATH}" | cut -f1)"
