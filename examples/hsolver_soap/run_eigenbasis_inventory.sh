#!/usr/bin/env bash
# One-step, all-rank inventory of real SOAP parameters and Kronecker-factor axes.

set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_directory/../.." && pwd)

data_root=${1:-/home/ll/HPC/data/soap-8b}
result_root=${2:-/home/ll/HPC/results/soap-8b/eigenbasis-actual/inventory}
data_prefix="$data_root/processed/nemotron-cc-translated-diverse-qa-eod2/nemotron_cc_translated_diverse_qa_eod2_text_document"
tokenizer_directory="$data_root/tokenizers/nemotron-3-nano-30b-a3b-base-pretrain-eod2"
cache_directory="$data_root/cache/gpt-seq128-seed42"

job_tag=${SLURM_JOB_ID:-manual}
timestamp=$(date +%Y%m%d_%H%M%S)
run_directory="$result_root/job_${job_tag}_${timestamp}"
inventory_directory="$run_directory/raw_inventory"
training_log="$run_directory/inventory_train.log"
analysis_log="$run_directory/inventory_summary.log"
inventory_csv="$run_directory/all_factor_axes.csv"
candidates_csv="$run_directory/hsolver_supported_factor_axes.csv"

mkdir -p "$cache_directory" "$inventory_directory"
cd "$repository_root"

{
    echo "[Experiment]"
    echo "purpose: inventory real SOAP parameter names and factor axes"
    echo "training steps: 1"
    echo "optimizer QR backend: torch"
    echo "slurm job: ${SLURM_JOB_ID:-not-set}"
    echo "host: $(hostname)"
    echo "run directory: $run_directory"
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
    --train-iters 1
    --optimizer soap
    --soap-qr-backend torch
    --soap-qr-hsolver-min-dimension 2048
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr 2e-4
    --min-lr 2e-4
    --lr-decay-style constant
    --lr-decay-iters 4000
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
SOAP_EIGENBASIS_INVENTORY=1 \
SOAP_EIGENBASIS_INVENTORY_DIR="$inventory_directory" \
SOAP_EIGENBASIS_TRACKING=0 \
SOAP_NUMERICS_DIAGNOSTICS=0 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_DEBUG=WARN \
OMP_NUM_THREADS=8 \
timeout 20m \
torchrun \
    --standalone \
    --nproc_per_node=8 \
    pretrain_gpt.py \
    "${common_arguments[@]}" \
    2>&1 | tee "$training_log"
training_exit=${PIPESTATUS[0]}
set -e

echo "SOAP_EIGENBASIS_INVENTORY_TRAIN_EXIT=$training_exit" | tee "$training_log.exit"
if (( training_exit != 0 )); then
    echo "Inventory training failed."
    echo "RUN_DIRECTORY=$run_directory"
    exit "$training_exit"
fi

python "$script_directory/summarize_eigenbasis_inventory.py" \
    "$inventory_directory" \
    --csv-output "$inventory_csv" \
    --candidates-output "$candidates_csv" \
    2>&1 | tee "$analysis_log"

echo "SOAP EIGENBASIS INVENTORY: PASS"
echo "RUN_DIRECTORY=$run_directory"
echo "INVENTORY_CSV=$inventory_csv"
echo "CANDIDATES_CSV=$candidates_csv"
