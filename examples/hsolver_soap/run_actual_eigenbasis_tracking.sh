#!/usr/bin/env bash
# One independent real SOAP backend trajectory with FP32 EVD measurement.

set -euo pipefail

if (( $# < 4 || $# > 6 )); then
    echo "Usage: $0 <torch|hsolver> <seq128|seq1024> <steps> <targets.json> [data_root] [result_root]" >&2
    exit 2
fi

backend=$1
scenario=$2
steps=$3
targets_file=$4
data_root=${5:-/home/ll/HPC/data/soap-8b}
result_root=${6:-/home/ll/HPC/results/soap-8b/eigenbasis-actual}

if [[ $backend != torch && $backend != hsolver ]]; then
    echo "backend must be 'torch' or 'hsolver', got: $backend" >&2
    exit 2
fi
if [[ ! $steps =~ ^[1-9][0-9]*$ ]]; then
    echo "steps must be a positive integer, got: $steps" >&2
    exit 2
fi
if [[ ! -f $targets_file ]]; then
    echo "targets file does not exist: $targets_file" >&2
    exit 2
fi
targets_file=$(realpath "$targets_file")

case $scenario in
    seq128)
        sequence_length=128
        maximum_position_embeddings=128
        data_split=1,0,0
        cache_name=gpt-seq128-seed42
        original_train_iters=4000
        ;;
    seq1024)
        sequence_length=1024
        maximum_position_embeddings=1024
        data_split=90,5,5
        cache_name=gpt-seq1024-seed42-split90-5-5
        original_train_iters=2400
        ;;
    *)
        echo "scenario must be 'seq128' or 'seq1024', got: $scenario" >&2
        exit 2
        ;;
esac

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_directory/../.." && pwd)
data_prefix="$data_root/processed/nemotron-cc-translated-diverse-qa-eod2/nemotron_cc_translated_diverse_qa_eod2_text_document"
tokenizer_directory="$data_root/tokenizers/nemotron-3-nano-30b-a3b-base-pretrain-eod2"
cache_directory="$data_root/cache/$cache_name"
tokens_per_step=$((sequence_length * 4))
maximum_state_step=$((steps - 1))
if (( steps <= 20 )); then
    burn_in_step=0
else
    burn_in_step=20
fi

job_tag=${SLURM_JOB_ID:-manual}
timestamp=$(date +%Y%m%d_%H%M%S)
experiment="${scenario}_${backend}_${steps}steps"
run_directory="$result_root/$scenario/$backend/job_${job_tag}_${timestamp}"
tracking_directory="$run_directory/tracking"
training_log="$run_directory/train.log"
analysis_log="$run_directory/analysis.log"
summary_json="$run_directory/summary.json"
series_csv="$run_directory/per_target_per_step.csv"
aggregate_csv="$run_directory/target_median_per_step.csv"
spectral_analysis_log="$run_directory/spectral_analysis.log"
spectral_summary_json="$run_directory/spectral_summary.json"
spectral_series_csv="$run_directory/spectral_per_step.csv"
tracking_timeout=${SOAP_TRACKING_TIMEOUT:-12h}
spectral_diagnostics=${SOAP_EIGENBASIS_SPECTRAL_DIAGNOSTICS:-0}

if [[ $spectral_diagnostics != 0 && $spectral_diagnostics != 1 ]]; then
    echo "SOAP_EIGENBASIS_SPECTRAL_DIAGNOSTICS must be 0 or 1" >&2
    exit 2
fi

mkdir -p "$cache_directory" "$tracking_directory"
cd "$repository_root"

python - "$targets_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
targets = payload.get("targets") if isinstance(payload, dict) else payload
assert isinstance(targets, list) and targets, "targets file must contain a non-empty list"
required = {"rank", "group_index", "param_index", "axis_index", "dimension"}
for index, target in enumerate(targets):
    missing = required - set(target)
    assert not missing, f"target {index} missing fields: {sorted(missing)}"
print(f"Validated tracking targets: {len(targets)}")
PY

python - <<'PY'
from pathlib import Path

import emerging_optimizers
import hsolver_torch
import torch
from emerging_optimizers.utils import soap_eigenbasis_tracking

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())
print("Emerging-Optimizers:", Path(emerging_optimizers.__file__).resolve())
print("Hsolver:", Path(hsolver_torch.__file__).resolve())
print("Tracker:", Path(soap_eigenbasis_tracking.__file__).resolve())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
print("ACTUAL EIGENBASIS TRACKING ENVIRONMENT: PASS")
PY

{
    echo "[Experiment]"
    echo "purpose: track one independent actual SOAP trajectory against its own FP32 EVD references"
    echo "experiment: $experiment"
    echo "actual optimizer QR backend: $backend"
    echo "scenario: $scenario"
    echo "steps: $steps"
    echo "sequence length: $sequence_length"
    echo "global batch size: 4"
    echo "tokens per step: $tokens_per_step"
    echo "total tokens: $((tokens_per_step * steps))"
    echo "reference: fresh torch.linalg.eigh in FP32 on each target's own K_t"
    echo "metric dtype: FP64"
    echo "actual-only spectral diagnostics: $spectral_diagnostics"
    echo "alternate QR backend executed: no"
    echo "spectral trace-energy thresholds: 0.5,0.9,0.95,0.99,0.999"
    echo "primary spectral metrics: aggregate relative eigen-residual, leading subspace overlap"
    echo "paper trace-energy threshold: 0.99"
    echo "analysis burn-in state step: $burn_in_step"
    echo "targets file: $targets_file"
    echo "timeout: $tracking_timeout"
    echo "slurm job: ${SLURM_JOB_ID:-not-set}"
    echo "host: $(hostname)"
    echo "run directory: $run_directory"
    echo
    echo "[Targets]"
    python -m json.tool "$targets_file"
} | tee "$run_directory/manifest.log"

common_arguments=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 21504
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length "$sequence_length"
    --max-position-embeddings "$maximum_position_embeddings"
    --position-embedding-type rope
    --rotary-percent 1.0
    --normalization LayerNorm
    --init-method-std 0.02
    --attention-dropout 0.1
    --hidden-dropout 0.1
    --no-rope-fusion
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --no-masked-softmax-fusion
    --no-one-logger
    --micro-batch-size 1
    --global-batch-size 4
    --train-iters "$steps"
    --optimizer soap
    --soap-qr-backend "$backend"
    --soap-qr-hsolver-min-dimension 2048
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr 2e-4
    --min-lr 2e-4
    --lr-decay-style constant
    --lr-decay-iters "$original_train_iters"
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --transformer-impl local
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --distributed-backend nccl
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$tokenizer_directory"
    --data-path "$data_prefix"
    --data-cache-path "$cache_directory"
    --split "$data_split"
    --dataloader-type single
    --num-workers 0
    --seed 42
    --log-interval 1
    --eval-interval 1000
    --eval-iters 0
    --rerun-mode disabled
)

set +e
SOAP_EIGENBASIS_TRACKING=1 \
SOAP_EIGENBASIS_TRACKING_DIR="$tracking_directory" \
SOAP_EIGENBASIS_TRACKING_TARGETS_FILE="$targets_file" \
SOAP_EIGENBASIS_TRACKING_MAX_STATE_STEP="$maximum_state_step" \
SOAP_EIGENBASIS_TRACKING_EXPERIMENT="$experiment" \
SOAP_EIGENBASIS_INVENTORY=0 \
SOAP_NUMERICS_DIAGNOSTICS=0 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_DEBUG=WARN \
OMP_NUM_THREADS=8 \
timeout "$tracking_timeout" \
torchrun \
    --standalone \
    --nproc_per_node=8 \
    pretrain_gpt.py \
    "${common_arguments[@]}" \
    2>&1 | tee "$training_log"
training_exit=${PIPESTATUS[0]}
set -e

echo "ACTUAL_EIGENBASIS_TRACKING_TRAIN_EXIT=$training_exit" | tee "$training_log.exit"
if (( training_exit != 0 )); then
    echo "Training failed; analysis was not run."
    echo "RUN_DIRECTORY=$run_directory"
    exit "$training_exit"
fi

python "$script_directory/analyze_eigenbasis_tracking.py" \
    "$tracking_directory" \
    --burn-in-step "$burn_in_step" \
    --expected-steps "$steps" \
    --tokens-per-step "$tokens_per_step" \
    --json-output "$summary_json" \
    --csv-output "$series_csv" \
    --aggregate-csv-output "$aggregate_csv" \
    2>&1 | tee "$analysis_log"

if [[ $spectral_diagnostics == 1 ]]; then
    python "$script_directory/analyze_spectral_tail_tracking.py" \
        "$tracking_directory" \
        --expected-steps "$steps" \
        --burn-in-state-step 1 \
        --json-output "$spectral_summary_json" \
        --csv-output "$spectral_series_csv" \
        2>&1 | tee "$spectral_analysis_log"
fi

echo "ACTUAL SOAP EIGENBASIS TRACKING: PASS"
echo "RUN_DIRECTORY=$run_directory"
echo "TRAINING_LOG=$training_log"
echo "SUMMARY_JSON=$summary_json"
echo "PER_TARGET_CSV=$series_csv"
echo "TARGET_MEDIAN_CSV=$aggregate_csv"
if [[ $spectral_diagnostics == 1 ]]; then
    echo "SPECTRAL_SUMMARY_JSON=$spectral_summary_json"
    echo "SPECTRAL_PER_STEP_CSV=$spectral_series_csv"
fi
