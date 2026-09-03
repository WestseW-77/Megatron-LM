#!/usr/bin/env bash
# Legacy one-target, 120-step screen of the actual Torch QR basis against FP32 EVD.

set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_directory/../.." && pwd)

data_root=${1:-/home/ll/HPC/data/soap-8b}
result_root=${2:-/home/ll/HPC/results/soap-8b/eigenbasis-evd-comparison}
data_prefix="$data_root/processed/nemotron-cc-translated-diverse-qa-eod2/nemotron_cc_translated_diverse_qa_eod2_text_document"
tokenizer_directory="$data_root/tokenizers/nemotron-3-nano-30b-a3b-base-pretrain-eod2"
cache_directory="$data_root/cache/gpt-seq128-seed42"

job_tag=${SLURM_JOB_ID:-manual}
timestamp=$(date +%Y%m%d_%H%M%S)
run_directory="$result_root/job_${job_tag}_${timestamp}"
tracking_directory="$run_directory/tracking"
training_log="$run_directory/train_torch_driver_120steps.log"
analysis_log="$run_directory/evd_qr_analysis.log"
summary_json="$run_directory/evd_qr_summary.json"
series_csv="$run_directory/evd_qr_per_step.csv"

mkdir -p "$cache_directory" "$tracking_directory"
cd "$repository_root"

python - <<'PY'
from pathlib import Path

import torch
import emerging_optimizers
import hsolver_torch
from emerging_optimizers.utils import soap_eigenbasis_tracking

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())
print("Emerging-Optimizers:", Path(emerging_optimizers.__file__).resolve())
print("Hsolver:", Path(hsolver_torch.__file__).resolve())
print("Tracker:", Path(soap_eigenbasis_tracking.__file__).resolve())

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
assert str(Path(emerging_optimizers.__file__).resolve()).startswith(
    "/home/ll/HPC/Emerging-Optimizers/"
)
assert str(Path(hsolver_torch.__file__).resolve()).startswith(
    "/home/ll/HPC/Hsolver/pytorch/"
)
print("EIGENBASIS TRACKING ENVIRONMENT: PASS")
PY

{
    echo "[Experiment]"
    echo "purpose: track actual SOAP Torch QR residual against fresh FP32 EVD"
    echo "actual training optimizer QR backend: torch"
    echo "reference: fresh torch.linalg.eigh in FP32 at every observed step"
    echo "counterfactual QR branches: none"
    echo "tracked target: rank=0 group=0 parameter=0 axis=1 dimension=2048"
    echo "evaluated columns: all 2048"
    echo "metric computation dtype: FP64"
    echo "steps: 120"
    echo "run directory: $run_directory"
    echo "slurm job: ${SLURM_JOB_ID:-not-set}"
    echo "host: $(hostname)"
    echo
} | tee "$run_directory/manifest.log"

common_arguments=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 21504
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length 128
    --max-position-embeddings 128
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
    --train-iters 120
    --optimizer soap
    --soap-qr-backend torch
    --soap-qr-hsolver-min-dimension 2048
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr 2e-4
    --min-lr 2e-4
    --lr-decay-style constant
    --lr-decay-iters 120
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
    --split 1,0,0
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
SOAP_EIGENBASIS_TRACKING_RANK=0 \
SOAP_EIGENBASIS_TRACKING_GROUP=0 \
SOAP_EIGENBASIS_TRACKING_PARAM=0 \
SOAP_EIGENBASIS_TRACKING_AXIS=1 \
SOAP_EIGENBASIS_TRACKING_DIMENSION=2048 \
SOAP_EIGENBASIS_TRACKING_SAMPLE_COLUMNS=2048 \
SOAP_EIGENBASIS_TRACKING_MAX_STATE_STEP=119 \
SOAP_NUMERICS_DIAGNOSTICS=0 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_DEBUG=WARN \
OMP_NUM_THREADS=8 \
timeout 90m \
torchrun \
    --standalone \
    --nproc_per_node=8 \
    pretrain_gpt.py \
    "${common_arguments[@]}" \
    2>&1 | tee "$training_log"
training_exit=${PIPESTATUS[0]}
set -e

echo "EIGENBASIS_TRACKING_TRAIN_EXIT=$training_exit" | tee "$training_log.exit"
if (( training_exit != 0 )); then
    echo "Training failed; analysis was not run."
    echo "RUN_DIRECTORY=$run_directory"
    exit "$training_exit"
fi

tracking_file="$tracking_directory/rank_00.jsonl"
if [[ ! -s "$tracking_file" ]]; then
    echo "Expected tracking output is missing or empty: $tracking_file" >&2
    exit 1
fi

python "$script_directory/analyze_eigenbasis_tracking.py" \
    "$tracking_directory" \
    --burn-in-step 20 \
    --expected-steps 120 \
    --tokens-per-step 512 \
    --json-output "$summary_json" \
    --csv-output "$series_csv" \
    2>&1 | tee "$analysis_log"

echo "ACTUAL TORCH QR VERSUS EVD SCREEN: PASS"
echo "RUN_DIRECTORY=$run_directory"
echo "TRAINING_LOG=$training_log"
echo "TRACKING_JSONL=$tracking_file"
echo "SUMMARY_JSON=$summary_json"
echo "PER_STEP_CSV=$series_csv"
