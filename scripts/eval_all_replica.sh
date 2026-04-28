#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
SCENES=()
DRY_RUN=0
FORCE=0

usage() {
  cat <<'EOF'
Usage: scripts/eval_all_replica.sh [--dry-run] [--force] [--scenes scene1,scene2,...]

Runs the 4 cm + block_hash + SAM-dense Replica sweep, then aggregates the
produced eval JSONs into one summary JSON + Markdown table.
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
mkdir -p benchmarks/results outputs

PAIRS=()
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
AGG_OUT="benchmarks/results/${STAMP}_replica_aggregate.json"

for scene in "${SCENES[@]}"; do
  cfg="configs/replica_${scene}_4cm_block_hash_sam.yaml"
  spec="eval/specs/replica_${scene}.yaml"
  map_out="outputs/replica_${scene}_4cm_block_hash_sam.npz"

  if [[ ! -f "$cfg" ]]; then
    echo "missing config: $cfg" >&2
    exit 1
  fi
  if [[ ! -f "$spec" ]]; then
    echo "missing spec: $spec" >&2
    exit 1
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

echo "+ .venv/bin/python scripts/aggregate_grounding_results.py ${PAIRS[*]} --out $AGG_OUT"
.venv/bin/python scripts/aggregate_grounding_results.py "${PAIRS[@]}" --out "$AGG_OUT"
echo "aggregate json: $AGG_OUT"
