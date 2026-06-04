#!/usr/bin/env bash
set -euo pipefail

# Run the merge-friendly GLM-Image focused perf matrix and optional output
# validation from pure JSON inline deploy configs.
#
# Usage:
#   bash benchmarks/glm_image/run_focused_inline_tests.sh
#   bash benchmarks/glm_image/run_focused_inline_tests.sh --skip-validation
#   bash benchmarks/glm_image/run_focused_inline_tests.sh --visible-devices 0,1,2,3
#   bash benchmarks/glm_image/run_focused_inline_tests.sh --allow-busy-gpu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_CONFIG="$ROOT_DIR/tests/dfx/perf/tests/test_glm_image_vllm_omni_focused_inline.json"
WORKLOAD_CONFIG="$ROOT_DIR/tests/dfx/perf/tests/glm_image_512_1024_steps50_t2i_i2i.json"
PERF_SCRIPT="$ROOT_DIR/tests/dfx/perf/scripts/run_diffusion_benchmark.py"
VALIDATION_SCRIPT="$ROOT_DIR/benchmarks/glm_image/validate_parallel_outputs_from_config.py"
TRACE_SCRIPT="$ROOT_DIR/tools/trace_gpu_pids.sh"

VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUTPUT_ROOT="/data/h/test/logs"
RUN_NAME="glm_image_focused_inline_$(date +%Y%m%d_%H%M%S)"
RUN_VALIDATION=1
ALLOW_BUSY_GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --visible-devices)
      VISIBLE_DEVICES="${2:?missing value for --visible-devices}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing value for --output-root}"
      shift 2
      ;;
    --run-name)
      RUN_NAME="${2:?missing value for --run-name}"
      shift 2
      ;;
    --skip-validation)
      RUN_VALIDATION=0
      shift
      ;;
    --validation-only)
      RUN_VALIDATION=2
      shift
      ;;
    --allow-busy-gpu)
      ALLOW_BUSY_GPU=1
      shift
      ;;
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
PERF_DIR="$RUN_DIR/perf_results"
VALIDATION_DIR="$RUN_DIR/output_validation"
mkdir -p "$RUN_DIR" "$PERF_DIR"

echo "ROOT_DIR=$ROOT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES"
echo "TEST_CONFIG=$TEST_CONFIG"
echo "WORKLOAD_CONFIG=$WORKLOAD_CONFIG"

if [[ -x "$TRACE_SCRIPT" ]]; then
  "$TRACE_SCRIPT" --parents 3 > "$RUN_DIR/gpu_before.log" || true
  echo "GPU trace before run: $RUN_DIR/gpu_before.log"
fi

if [[ "$ALLOW_BUSY_GPU" -eq 0 ]]; then
  busy_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"
  if [[ -n "$busy_pids" ]]; then
    echo "Refusing to start because GPU compute processes already exist:" >&2
    if [[ -x "$TRACE_SCRIPT" ]]; then
      "$TRACE_SCRIPT" --parents 3 >&2 || true
    else
      nvidia-smi >&2 || true
    fi
    echo "Rerun with --allow-busy-gpu if this is intentional." >&2
    exit 1
  fi
fi

if [[ -f /data/h/h/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /data/h/h/bin/activate
fi

python - <<PY
import json
from pathlib import Path
for path in [Path("$TEST_CONFIG"), Path("$WORKLOAD_CONFIG")]:
    data = json.loads(path.read_text())
    print(f"json ok: {path} entries={len(data)}")
PY

export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ "$RUN_VALIDATION" -ne 2 ]]; then
  echo "Starting focused perf..."
  export DIFFUSION_BENCHMARK_DIR="$PERF_DIR"
  python -m pytest -s "$PERF_SCRIPT" --test-config-file "$TEST_CONFIG" \
    2>&1 | tee "$RUN_DIR/pytest_glm_image_focused_inline.log"
fi

if [[ "$RUN_VALIDATION" -ne 0 ]]; then
  echo "Starting output validation..."
  python "$VALIDATION_SCRIPT" \
    --test-config-file "$TEST_CONFIG" \
    --output-dir "$VALIDATION_DIR" \
    --sizes 1024 512 \
    --steps 50 \
    --seed 42 \
    2>&1 | tee "$RUN_DIR/validate_parallel_outputs_from_config.log"
fi

if [[ -x "$TRACE_SCRIPT" ]]; then
  "$TRACE_SCRIPT" --parents 3 > "$RUN_DIR/gpu_after.log" || true
  echo "GPU trace after run: $RUN_DIR/gpu_after.log"
fi

echo "Done. Results under: $RUN_DIR"
