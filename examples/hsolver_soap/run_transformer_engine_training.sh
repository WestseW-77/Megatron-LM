#!/usr/bin/env bash
# Run one independent SOAP trajectory with Transformer Engine on 8 GPUs.

set -euo pipefail

if (( $# < 2 || $# > 5 )); then
    echo "Usage: $0 <torch|hsolver> <seq128|seq1024> [steps] [data_root] [result_root]" >&2
    exit 2
fi

backend=$1
scenario=$2
requested_steps=${3:-}
data_root=${4:-/home/ll/HPC/data/soap-8b}
result_root=${5:-/home/ll/HPC/results/soap-8b/transformer-engine/training}

if [[ $backend != torch && $backend != hsolver ]]; then
    echo "backend must be 'torch' or 'hsolver', got: $backend" >&2
    exit 2
fi

case $scenario in
    seq128)
        sequence_length=128
        maximum_position_embeddings=128
        formal_steps=4000
        data_split=1,0,0
        cache_name=gpt-seq128-seed42
        eval_interval=1000
        eval_iters=0
        ;;
    seq1024)
        sequence_length=1024
        maximum_position_embeddings=1024
        formal_steps=2400
        data_split=90,5,5
        cache_name=gpt-seq1024-seed42-split90-5-5
        eval_interval=400
        eval_iters=50
        ;;
    *)
        echo "scenario must be 'seq128' or 'seq1024', got: $scenario" >&2
        exit 2
        ;;
esac

steps=${requested_steps:-$formal_steps}
if [[ ! $steps =~ ^[1-9][0-9]*$ ]]; then
    echo "steps must be a positive integer, got: $steps" >&2
    exit 2
fi

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_directory/../.." && pwd)
data_prefix="$data_root/processed/nemotron-cc-translated-diverse-qa-eod2/nemotron_cc_translated_diverse_qa_eod2_text_document"
tokenizer_directory="$data_root/tokenizers/nemotron-3-nano-30b-a3b-base-pretrain-eod2"
cache_directory="$data_root/cache/$cache_name"

for required_file in "${data_prefix}.bin" "${data_prefix}.idx" "$tokenizer_directory/tokenizer.json"; do
    if [[ ! -f $required_file ]]; then
        echo "required input does not exist: $required_file" >&2
        exit 2
    fi
done

if [[ ${CUDA_VISIBLE_DEVICES:-} != "0,1,2,3,4,5,6,7" ]]; then
    echo "expected CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, got: ${CUDA_VISIBLE_DEVICES:-not-set}" >&2
    exit 2
fi

job_tag=${SLURM_JOB_ID:-manual}
timestamp=$(date +%Y%m%d_%H%M%S)
run_name="${scenario}_${backend}_${steps}steps"
run_directory="$result_root/$scenario/$backend/job_${job_tag}_${timestamp}"
training_log="$run_directory/train.log"
manifest_log="$run_directory/manifest.log"
timeout_value=${SOAP_TE_TIMEOUT:-18h}
tokens_per_step=$((sequence_length * 4))

mkdir -p "$cache_directory" "$run_directory"
cd "$repository_root"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

python - "$backend" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

import torch
import hsolver_torch._C as hsolver_C
import transformer_engine
import transformer_engine_torch
from megatron.core.extensions import transformer_engine as megatron_te
from megatron.core.optimizer import USING_PYTORCH_OPTIMIZER

backend = sys.argv[1]

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Transformer Engine:", version("transformer-engine"))
print("Transformer Engine Python:", Path(transformer_engine.__file__).resolve())
print("Transformer Engine extension:", Path(transformer_engine_torch.__file__).resolve())
print("Hsolver extension:", Path(hsolver_C.__file__).resolve())
print("Megatron HAVE_TE:", megatron_te.HAVE_TE)
print("Using PyTorch optimizer fallback:", USING_PYTORCH_OPTIMIZER)
print("Visible GPUs:", torch.cuda.device_count())

assert torch.__version__ == "2.12.1+cu132"
assert torch.version.cuda == "13.2"
assert version("transformer-engine") == "2.18.0+e7c550c5"
assert megatron_te.HAVE_TE
assert not USING_PYTORCH_OPTIMIZER
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
for index in range(8):
    properties = torch.cuda.get_device_properties(index)
    assert (properties.major, properties.minor) == (8, 9)
if backend == "hsolver":
    assert str(Path(hsolver_C.__file__).resolve()).startswith(
        "/home/ll/HPC/Hsolver/pytorch/"
    )
print("TRANSFORMER ENGINE TRAINING ENVIRONMENT: PASS")
PY

{
    echo "[Experiment]"
    echo "name: $run_name"
    echo "purpose: independent SOAP training trajectory with Transformer Engine"
    echo "QR backend: $backend"
    echo "transformer implementation: transformer_engine"
    echo "Transformer Engine version: 2.18.0+e7c550c5"
    echo "Transformer Engine source commit: e7c550c5f80636cf841a8204b1d6f85a5f3f28b7"
    echo "Transformer Engine CUDA architecture: 89"
    echo "precision: BF16 model training; backend-specific SOAP QR precision"
    echo "TE default RoPE, softmax, bias-GELU, and bias-dropout fusions: enabled"
    echo "gradient accumulation fusion: disabled because Apex fused_weight_gradient_mlp_cuda is not installed"
    echo "sequence length: $sequence_length"
    echo "steps: $steps"
    echo "formal scenario steps: $formal_steps"
    echo "global batch size: 4"
    echo "tokens per step: $tokens_per_step"
    echo "total training tokens: $((tokens_per_step * steps))"
    echo "data split: $data_split"
    echo "eval interval: $eval_interval"
    echo "eval iters: $eval_iters"
    echo "seed: 42"
    echo "tensor parallel size: 2"
    echo "pipeline parallel size: 4"
    echo "timeout: $timeout_value"
    echo "host: $(hostname)"
    echo "SLURM job: ${SLURM_JOB_ID:-not-set}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
    echo "Megatron commit: $(git rev-parse HEAD)"
    echo "Emerging-Optimizers commit: $(git -C /home/ll/HPC/Emerging-Optimizers rev-parse HEAD)"
    echo "Hsolver commit: $(git -C /home/ll/HPC/Hsolver rev-parse HEAD)"
    echo "run directory: $run_directory"
} | tee "$manifest_log"

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
    --no-gradient-accumulation-fusion
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
    --lr-decay-iters "$formal_steps"
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --transformer-impl transformer_engine
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
    --eval-interval "$eval_interval"
    --eval-iters "$eval_iters"
    --rerun-mode disabled
)

set +e
SOAP_EIGENBASIS_TRACKING=0 \
SOAP_EIGENBASIS_INVENTORY=0 \
SOAP_NUMERICS_DIAGNOSTICS=0 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_DEBUG=WARN \
OMP_NUM_THREADS=8 \
timeout "$timeout_value" \
torchrun \
    --standalone \
    --nproc_per_node=8 \
    pretrain_gpt.py \
    "${common_arguments[@]}" \
    2>&1 | tee "$training_log"
training_exit=${PIPESTATUS[0]}
set -e

echo "TRANSFORMER_ENGINE_TRAIN_EXIT=$training_exit" | tee "$training_log.exit"
echo "RUN_DIRECTORY=$run_directory"
echo "TRAINING_LOG=$training_log"

if (( training_exit != 0 )); then
    exit "$training_exit"
fi

echo "TRANSFORMER ENGINE SOAP TRAINING: PASS"
