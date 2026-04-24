#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SCENES=(scene0011_00 scene0050_00 scene0231_00 scene0568_00 scene0616_00)
SCENES=()
DRY_RUN=0
FORCE=0

usage() {
  cat <<'EOF'
Usage: scripts/eval_all_scannet.sh [--dry-run] [--force] [--scenes scene1,scene2,...]

Runs the 4 cm + block_hash + SAM-dense ScanNet v2 sweep:
  1. Generate eval specs from annotations (if not already present)
  2. Encode each scene
  3. Evaluate grounding
  4. Aggregate results

Requires: ScanNet data at ~/data/scannet (see scripts/download_scannet.sh).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --scenes)
      IFS=',' read -r -a SCENES <<<"$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#SCENES[@]} -eq 0 ]]; then
  SCENES=("${DEFAULT_SCENES[@]}")
fi

cd "$ROOT"
mkdir -p benchmarks/results outputs eval/specs

SCANNET_ROOT="${SCANNET_ROOT:-$HOME/data/scannet}"
if [[ ! -d "$SCANNET_ROOT/scans" ]]; then
  echo "ScanNet data not found at $SCANNET_ROOT/scans" >&2
  echo "Set SCANNET_ROOT or run scripts/download_scannet.sh first" >&2
  exit 1
fi

PAIRS=()
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
AGG_OUT="benchmarks/results/${STAMP}_scannet_aggregate.json"

for scene in "${SCENES[@]}"; do
  cfg="configs/scannet_${scene}_4cm_block_hash_sam.yaml"
  spec="eval/specs/scannet_${scene}.yaml"
  map_out="outputs/scannet_${scene}_4cm_block_hash_sam.npz"

  # Generate config from template if missing.
  if [[ ! -f "$cfg" ]]; then
    echo "- generating config: $cfg"
    cat > "$cfg" <<YAML
dataset:
  name: scannet
  root: $SCANNET_ROOT
  scene: $scene
  max_frames: 100
  stride: 20

camera:
  depth_scale: 1000.0
  depth_trunc_m: 6.0

mapping:
  voxel_size_m: 0.04
  truncation_distance_m: 0.16
  block_size: 8
  backend: block_hash
  device: cuda:0
  store_color: true
  store_features: true
  feature_dim: 512
  auto_bounds_radius_m: 5.0
  max_weight: 32.0
  initial_feat_capacity: 262144
  max_feat_capacity: 8000000

semantics:
  model: ViT-B-16
  pretrained: laion2b_s34b_b88k
  mode: sam_dense
  batch_size: 32
  device: cuda:0
  dtype: fp16

grounding:
  top_k: 5
  score_threshold: 0.22
  cluster_eps_vox: 2
  min_cluster_voxels: 8

logging:
  level: INFO
  rich_tracebacks: true
YAML
  fi

  # Generate eval spec from annotations if missing.
  if [[ ! -f "$spec" ]]; then
    echo "+ generating eval spec: $spec"
    if [[ $DRY_RUN -eq 0 ]]; then
      .venv/bin/python scripts/gen_scannet_eval_specs.py \
        --root "$SCANNET_ROOT" --scene "$scene" --out "$spec"
    fi
  fi

  # Check scene data exists.
  if [[ ! -d "$SCANNET_ROOT/scans/$scene/color" ]]; then
    echo "WARNING: $scene not extracted (no color/ dir), skipping" >&2
    continue
  fi

  if [[ $FORCE -eq 1 || ! -f "$map_out" ]]; then
    echo "+ .venv/bin/openvocab-tsdf encode -c $cfg -o $map_out"
    if [[ $DRY_RUN -eq 0 ]]; then
      .venv/bin/openvocab-tsdf encode -c "$cfg" -o "$map_out"
    fi
  else
    echo "= skip encode (map exists): $map_out"
  fi

  echo "+ .venv/bin/python eval/eval_grounding.py --map $map_out --spec $spec --out-dir benchmarks/results"
  if [[ $DRY_RUN -eq 0 ]]; then
    .venv/bin/python eval/eval_grounding.py --map "$map_out" --spec "$spec" --out-dir benchmarks/results
    latest_json="$(ls -1t benchmarks/results/*_eval_grounding.json | head -n 1)"
    PAIRS+=("--pair" "${scene}=${latest_json}")
  fi
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "+ .venv/bin/python scripts/aggregate_grounding_results.py --pair <scene>=<json> ... --out $AGG_OUT"
  exit 0
fi

if [[ ${#PAIRS[@]} -eq 0 ]]; then
  echo "No scenes were evaluated. Check that ScanNet data is extracted." >&2
  exit 1
fi

echo "+ .venv/bin/python scripts/aggregate_grounding_results.py ${PAIRS[*]} --out $AGG_OUT"
.venv/bin/python scripts/aggregate_grounding_results.py "${PAIRS[@]}" --out "$AGG_OUT"
echo "aggregate json: $AGG_OUT"
