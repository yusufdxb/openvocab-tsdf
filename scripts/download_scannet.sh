#!/usr/bin/env bash
# Download ScanNet v2 scenes for openvocab-tsdf evaluation.
#
# Prerequisites:
#   1. Sign the ScanNet Terms of Use at http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf
#   2. Email the signed form to scannet@googlegroups.com
#   3. You'll receive a download script (download-scannet.py) with your credentials
#
# Usage:
#   # Download specific scenes (recommended — full dataset is ~1.3 TB)
#   bash scripts/download_scannet.sh ~/data/scannet scene0011_00 scene0050_00 scene0231_00
#
#   # Or if you already have the official download script:
#   python download-scannet.py -o ~/data/scannet --type .sens --id scene0011_00
#   python download-scannet.py -o ~/data/scannet --type _vh_clean_2.ply --id scene0011_00
#   python download-scannet.py -o ~/data/scannet --type _vh_clean.aggregation.json --id scene0011_00
#   python download-scannet.py -o ~/data/scannet --type .0.010000.segs.json --id scene0011_00
#
# After downloading .sens files, extract them:
#   python SensReader/reader.py --filename <scene>.sens --output_path <scene>/ --export_depth_images --export_color_images --export_poses --export_intrinsics

set -euo pipefail

ROOT="${1:?Usage: $0 <output_root> [scene_id ...]}"
shift
SCENES=("${@}")

if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(scene0011_00 scene0050_00 scene0231_00 scene0568_00 scene0616_00)
fi

mkdir -p "$ROOT/scans"

echo "ScanNet download helper"
echo "======================="
echo ""
echo "Target: $ROOT"
echo "Scenes: ${SCENES[*]}"
echo ""
echo "This script requires the official ScanNet download script."
echo "If you haven't received it yet:"
echo "  1. Download the ToU form: http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf"
echo "  2. Sign and email to: scannet@googlegroups.com"
echo "  3. Place the received download-scannet.py in this directory"
echo ""

DOWNLOADER=""
for candidate in download-scannet.py ../download-scannet.py ~/download-scannet.py; do
    if [ -f "$candidate" ]; then
        DOWNLOADER="$candidate"
        break
    fi
done

if [ -z "$DOWNLOADER" ]; then
    echo "ERROR: download-scannet.py not found."
    echo "Place it in the repo root, parent directory, or home directory."
    exit 1
fi

echo "Found downloader: $DOWNLOADER"
echo ""

for SCENE in "${SCENES[@]}"; do
    echo "--- Downloading $SCENE ---"
    SCENE_DIR="$ROOT/scans/$SCENE"

    # Download .sens (RGB-D stream)
    if [ ! -f "$SCENE_DIR/$SCENE.sens" ]; then
        python3 "$DOWNLOADER" -o "$ROOT" --type .sens --id "$SCENE"
    else
        echo "  .sens already exists, skipping"
    fi

    # Download segmented mesh
    if [ ! -f "$SCENE_DIR/${SCENE}_vh_clean_2.ply" ]; then
        python3 "$DOWNLOADER" -o "$ROOT" --type _vh_clean_2.ply --id "$SCENE"
    else
        echo "  _vh_clean_2.ply already exists, skipping"
    fi

    # Download aggregation JSON
    python3 "$DOWNLOADER" -o "$ROOT" --type _vh_clean.aggregation.json --id "$SCENE" 2>/dev/null || true

    # Download segmentation indices
    python3 "$DOWNLOADER" -o "$ROOT" --type .0.010000.segs.json --id "$SCENE" 2>/dev/null || true

    # Extract .sens -> color/ depth/ pose/ intrinsic/
    if [ ! -d "$SCENE_DIR/color" ]; then
        echo "  Extracting .sens file..."
        if [ -f "scripts/SensReader/reader.py" ]; then
            python3 scripts/SensReader/reader.py \
                --filename "$SCENE_DIR/$SCENE.sens" \
                --output_path "$SCENE_DIR/" \
                --export_depth_images \
                --export_color_images \
                --export_poses \
                --export_intrinsics
        else
            echo "  WARNING: SensReader not found at scripts/SensReader/reader.py"
            echo "  Clone it: git clone https://github.com/ScanNet/ScanNet.git scripts/ScanNet"
            echo "  Then: cp -r scripts/ScanNet/SensReader scripts/"
        fi
    else
        echo "  color/ directory exists, skipping extraction"
    fi

    echo ""
done

echo "Done. Generate eval specs with:"
echo "  python scripts/gen_scannet_eval_specs.py --root $ROOT --scenes ${SCENES[*]} --out-dir eval/specs/"
