#!/usr/bin/env bash
# Internal worker for one backend of the matched GBS=128 SOAP QR timing pair.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  run_h200_qr_fraction_backend.sh <torch|hsolver> <run_directory> <master_port>

The run always uses the frozen H200 formal-training configuration with:
  GBS=128, train_iters=30, profiling window=iterations 20..30.

Environment:
  SOAP_H200_TIMEOUT  Optional timeout accepted by GNU timeout.

Required GPU visibility:
  CUDA_VISIBLE_DEVICES=0,1,2,3
EOF
    exit 2
}

if (( $# != 3 )); then
    usage
fi

backend=$1
run_directory=$2
master_port=$3
case $backend in
    torch | hsolver) ;;
    *)
        echo "backend must be torch or hsolver, got: $backend" >&2
        exit 2
        ;;
esac

# Frozen formal-training values from the completed 2400-step paired run.
global_batch_size=128
train_iters=30
profile_start_iteration=20
profile_end_iteration=30
formal_train_iters=95338
warmup_iters=953
wsd_decay_iters=9534
data_split=90,9,1
eval_interval=300
eval_iters=10
eval_global_batch_size=64

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd "$script_directory/../.." && pwd -P)
workspace_root=$(cd "$repository_root/.." && pwd -P)
environment_prefix=$(cd "$workspace_root/.." && pwd -P)/.conda/envs/hsolver-torch

if [[ ! $master_port =~ ^[0-9]+$ ]] || (( master_port < 1 || master_port > 65535 )); then
    echo "SOAP_H200_MASTER_PORT must be between 1 and 65535, got: $master_port" >&2
    exit 2
fi
if [[ ${CUDA_VISIBLE_DEVICES:-} != 0,1,2,3 ]]; then
    echo "expected CUDA_VISIBLE_DEVICES=0,1,2,3, got: ${CUDA_VISIBLE_DEVICES:-not-set}" >&2
    exit 2
fi

data_root=/data1/ll/HPC/data/soap-8b-formal
data_args_path=$data_root/processed/nemotron-cc-high-actual-100b-eod2/data_args.txt
tokenizer_directory=$data_root/tokenizers/nemotron-3-nano-30b-a3b-base-pretrain-eod2
cache_directory=$data_root/cache/gpt-seq8192-seed42-split90-9-1
formal_script=$script_directory/run_h200_formal_training.sh
analysis_script=$script_directory/analyze_qr_fraction.py
frozen_formal_script_sha256=0c458c567a5d4f70dd62b1958d6a4b3cbe1dcc5ecc4c4b69a0a3011f8bfcbbc6
frozen_pretrain_gpt_sha256=4755436b97ae18c868b3bfe0468f8d4896b2c20f39d3cc4b6927f9c929df4417

for required_path in \
    "$environment_prefix/bin/python" \
    "$environment_prefix/bin/nvcc" \
    "$environment_prefix/bin/torchrun" \
    "$repository_root/pretrain_gpt.py" \
    "$repository_root/megatron/training/optimizer_fraction_profiler.py" \
    "$workspace_root/Hsolver/pytorch/hsolver_torch" \
    "$workspace_root/Emerging-Optimizers/emerging_optimizers" \
    "$workspace_root/TransformerEngine" \
    "$data_args_path" \
    "$tokenizer_directory/tokenizer.json" \
    "$formal_script" \
    "$analysis_script"; do
    if [[ ! -e $required_path ]]; then
        echo "required path does not exist: $required_path" >&2
        exit 2
    fi
done

actual_formal_script_sha256=$(/usr/bin/sha256sum "$formal_script" | /usr/bin/awk '{print $1}')
actual_pretrain_gpt_sha256=$(/usr/bin/sha256sum "$repository_root/pretrain_gpt.py" | /usr/bin/awk '{print $1}')
if [[ $actual_formal_script_sha256 != "$frozen_formal_script_sha256" ]]; then
    echo "frozen formal script hash mismatch: $actual_formal_script_sha256" >&2
    exit 2
fi
if [[ $actual_pretrain_gpt_sha256 != "$frozen_pretrain_gpt_sha256" ]]; then
    echo "frozen pretrain_gpt.py hash mismatch: $actual_pretrain_gpt_sha256" >&2
    exit 2
fi

mkdir -p "$run_directory" "$cache_directory"

export CONDA_PREFIX=$environment_prefix
export CUDA_HOME=$environment_prefix
export CUDACXX=$environment_prefix/bin/nvcc
export PATH="$environment_prefix/bin:$PATH"

site_packages=$environment_prefix/lib/python3.12/site-packages
cuda_library_paths="$site_packages/nvidia/cu13/lib:$site_packages/nvidia/nccl/lib:$site_packages/nvidia/cudnn/lib:$environment_prefix/lib:$environment_prefix/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="$cuda_library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

manifest=$run_directory/manifest.log
environment_log=$run_directory/environment.log
training_log=$run_directory/train.log
command_log=$run_directory/command.txt
gpu_log=$run_directory/gpu_metrics.csv
exit_file=$run_directory/exit_code.txt
timing_directory=$run_directory/timing
mkdir -p "$timing_directory"
printf 'RUNNING\n' > "$run_directory/status.txt"

monitor_pid=
finalized=0
on_exit() {
    local worker_exit=$?
    if [[ -n ${monitor_pid:-} ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if (( finalized == 0 )); then
        printf '%s\n' "$worker_exit" > "$exit_file.tmp"
        mv "$exit_file.tmp" "$exit_file"
        printf 'FAILED\n' > "$run_directory/status.txt"
    fi
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap on_exit EXIT

git_binary=/data1/ll/.tools/git-env/bin/git
if [[ ! -x $git_binary ]]; then
    git_binary=$(command -v git)
fi
revision() {
    "$git_binary" -C "$1" rev-parse HEAD 2>/dev/null || printf 'unknown\n'
}

{
    echo "[Experiment]"
    echo "name: h200_soap_qr_fraction_gbs${global_batch_size}_${backend}"
    echo "purpose: measure SOAP QR, optimizer, and train-step time for a matched backend pair"
    echo "backend: $backend"
    echo "requested train iterations: $train_iters"
    echo "profiling iterations: $profile_start_iteration..$profile_end_iteration inclusive"
    echo "timing method: rank-local CUDA events around train step, optimizer, and each QR call"
    echo "synchronization: once after iteration $profile_end_iteration"
    echo "rank aggregation: optimizer critical rank retained for each iteration"
    echo "model: 8B dense transformer"
    echo "layers: 32"
    echo "hidden size: 4096"
    echo "FFN hidden size: 21504"
    echo "attention heads: 32"
    echo "query groups: 8"
    echo "KV channels: 128"
    echo "sequence length: 8192"
    echo "maximum position embeddings: 8192"
    echo "micro batch size: 1"
    echo "global batch size: $global_batch_size"
    echo "microbatches per DP rank per optimizer step: $global_batch_size"
    echo "tokens per optimizer step: $((8192 * global_batch_size))"
    echo "tensor parallel size: 1"
    echo "pipeline parallel size: 2"
    echo "context parallel size: 2"
    echo "data parallel size: 1"
    echo "optimizer: SOAP"
    echo "SOAP beta1: 0.9"
    echo "SOAP beta2: 0.95"
    echo "SOAP beta_kron/shampoo_beta: 0.95"
    echo "SOAP epsilon: 1e-8"
    echo "SOAP power iteration steps: 1"
    echo "layer-wise distributed optimizer: enabled through --use-distributed-optimizer"
    echo "maximum learning rate: 1.6e-4"
    echo "minimum learning rate: 1.6e-6"
    echo "learning-rate decay iterations: $formal_train_iters"
    echo "learning-rate decay style: WSD"
    echo "WSD warmup iterations: $warmup_iters"
    echo "WSD final decay iterations: $wsd_decay_iters"
    echo "WSD final decay style: minus_sqrt"
    echo "weight decay: 0.1"
    echo "clip grad: 1.0"
    echo "precision: BF16"
    echo "transformer implementation: transformer_engine"
    echo "data args path: $data_args_path"
    echo "data args SHA256: $(/usr/bin/sha256sum "$data_args_path" | /usr/bin/awk '{print $1}')"
    echo "tokenizer: $tokenizer_directory"
    echo "tokenizer.json SHA256: $(/usr/bin/sha256sum "$tokenizer_directory/tokenizer.json" | /usr/bin/awk '{print $1}')"
    echo "data split: $data_split"
    echo "data cache: $cache_directory"
    echo "seed: 42"
    echo "eval interval: $eval_interval"
    echo "eval iterations: $eval_iters"
    echo "eval global batch size: $eval_global_batch_size"
    echo "checkpointing: disabled"
    echo "frozen baseline tag: h200-soap-2400-baseline-20260917"
    echo "frozen baseline commit: 64d8b83717e004feb57497b140673787319125e3"
    echo "frozen formal script SHA256: $actual_formal_script_sha256"
    echo "pretrain_gpt.py SHA256: $actual_pretrain_gpt_sha256"
    echo "Megatron-LM commit: $(revision "$repository_root")"
    echo "Emerging-Optimizers commit: $(revision "$workspace_root/Emerging-Optimizers")"
    echo "Hsolver commit: $(revision "$workspace_root/Hsolver")"
    echo "Transformer Engine commit: $(revision "$workspace_root/TransformerEngine")"
    echo "distributed master address: 127.0.0.1"
    echo "distributed master port: $master_port"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "CUDA_HOME: $CUDA_HOME"
    echo "CUDACXX: $CUDACXX"
    echo "host: $(hostname)"
    echo "started at: $(date --iso-8601=seconds)"
    echo "run directory: $run_directory"
} > "$manifest"

"$environment_prefix/bin/python" - <<'PY' 2>&1 | tee "$environment_log"
from importlib.metadata import version
from pathlib import Path

import torch
import transformer_engine
import transformer_engine_torch
from megatron.core.extensions import transformer_engine as megatron_te
from megatron.core.optimizer import USING_PYTORCH_OPTIMIZER
from hsolver_torch import linalg as hsolver_linalg

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Transformer Engine:", version("transformer-engine"))
print("Transformer Engine Python:", Path(transformer_engine.__file__).resolve())
print("Transformer Engine extension:", Path(transformer_engine_torch.__file__).resolve())
print("Megatron HAVE_TE:", megatron_te.HAVE_TE)
print("Using PyTorch optimizer fallback:", USING_PYTORCH_OPTIMIZER)
print("Hsolver linalg:", hsolver_linalg)
print("Visible GPUs:", torch.cuda.device_count())

assert torch.__version__ == "2.12.1+cu132"
assert torch.version.cuda == "13.2"
assert version("transformer-engine") == "2.18.0+e7c550c5"
assert megatron_te.HAVE_TE
assert not USING_PYTORCH_OPTIMIZER
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
for index in range(4):
    properties = torch.cuda.get_device_properties(index)
    print(
        f"cuda:{index}: {properties.name}; "
        f"capability={(properties.major, properties.minor)}; "
        f"memory={properties.total_memory / 2**30:.2f} GiB"
    )
    assert "H200" in properties.name
    assert (properties.major, properties.minor) == (9, 0)

print("H200 SOAP QR TIMING ENVIRONMENT CHECK: PASS")
PY
cat "$environment_log" >> "$manifest"

common_arguments=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 21504
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type rope
    --rotary-percent 1.0
    --normalization LayerNorm
    --init-method-std 0.02
    --attention-dropout 0.1
    --hidden-dropout 0.1
    --no-gradient-accumulation-fusion
    --no-one-logger
    --micro-batch-size 1
    --global-batch-size "$global_batch_size"
    --train-iters "$train_iters"
    --optimizer soap
    --soap-qr-hsolver-min-dimension 2048
    --use-distributed-optimizer
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr 1.6e-4
    --min-lr 1.6e-6
    --lr-decay-style WSD
    --lr-decay-iters "$formal_train_iters"
    --lr-warmup-iters "$warmup_iters"
    --lr-wsd-decay-iters "$wsd_decay_iters"
    --lr-wsd-decay-style minus_sqrt
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --transformer-impl transformer_engine
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --context-parallel-size 2
    --cp-comm-type p2p
    --distributed-backend nccl
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$tokenizer_directory"
    --data-args-path "$data_args_path"
    --data-cache-path "$cache_directory"
    --split "$data_split"
    --dataloader-type single
    --num-workers 0
    --seed 42
    --log-interval 1
    --eval-interval "$eval_interval"
    --eval-iters "$eval_iters"
    --eval-global-batch-size "$eval_global_batch_size"
    --rerun-mode disabled
)

torchrun_command=(
    "$environment_prefix/bin/torchrun"
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    --master_port="$master_port"
    --nproc_per_node=4
    "$repository_root/pretrain_gpt.py"
    "${common_arguments[@]}"
    --soap-qr-backend "$backend"
    --profile-optimizer-fraction
    --profile-soap-qr
    --profile-optimizer-start-iteration "$profile_start_iteration"
    --profile-optimizer-end-iteration "$profile_end_iteration"
    --profile-optimizer-output-dir "$timing_directory"
)

{
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf 'SOAP_EIGENBASIS_TRACKING=0 SOAP_EIGENBASIS_INVENTORY=0 SOAP_NUMERICS_DIAGNOSTICS=0 '
    printf 'CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8 '
    printf '%q ' "${torchrun_command[@]}"
    printf '\n'
} > "$command_log"

start_gpu_monitor() {
    {
        echo "sample_time,index,name,memory.used MiB,memory.free MiB,utilization.gpu %,power.draw W,temperature.gpu C,clocks.current.sm MHz,pstate"
        while true; do
            sample_time=$(date --iso-8601=seconds)
            /usr/bin/nvidia-smi \
                --id=0,1,2,3 \
                --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm,pstate \
                --format=csv,noheader,nounits \
                | /usr/bin/sed "s/^/$sample_time,/"
            sleep 10
        done
    } >> "$gpu_log" 2>&1 &
    monitor_pid=$!
}

start_gpu_monitor
launch_command=("${torchrun_command[@]}")
if [[ -n ${SOAP_H200_TIMEOUT:-} ]]; then
    launch_command=(timeout "$SOAP_H200_TIMEOUT" "${torchrun_command[@]}")
fi

set +e
SOAP_EIGENBASIS_TRACKING=0 \
SOAP_EIGENBASIS_INVENTORY=0 \
SOAP_NUMERICS_DIAGNOSTICS=0 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_DEBUG=WARN \
OMP_NUM_THREADS=8 \
    "${launch_command[@]}" 2>&1 | tee "$training_log"
training_exit=${PIPESTATUS[0]}
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=

if (( training_exit == 0 )); then
    timing_files=("$timing_directory"/timing_rank*.csv)
    qr_timing_files=("$timing_directory"/qr_timing_rank*.csv)
    if (( ${#timing_files[@]} != 4 )); then
        echo "expected four rank timing files, found ${#timing_files[@]}" | tee -a "$training_log" >&2
        training_exit=1
    elif (( ${#qr_timing_files[@]} != 4 )); then
        echo "expected four rank QR timing files, found ${#qr_timing_files[@]}" | tee -a "$training_log" >&2
        training_exit=1
    elif ! "$environment_prefix/bin/python" "$analysis_script" "$run_directory" 2>&1 | tee "$run_directory/timing_analysis.log"; then
        training_exit=1
    fi
fi

printf '%s\n' "$training_exit" > "$exit_file.tmp"
mv "$exit_file.tmp" "$exit_file"
if (( training_exit == 0 )); then
    printf 'COMPLETED\n' > "$run_directory/status.txt"
else
    printf 'FAILED\n' > "$run_directory/status.txt"
fi
{
    echo "finished at: $(date --iso-8601=seconds)"
    echo "exit code: $training_exit"
} >> "$manifest"
finalized=1

echo "RUN_DIRECTORY=$run_directory"
echo "EXIT_CODE=$training_exit"
exit "$training_exit"
