#!/usr/bin/env bash
# Run paired Hsolver and PyTorch SOAP trajectories on four NVIDIA H200 GPUs.
#
# Hsolver always runs first. PyTorch starts after Hsolver exits, including when
# Hsolver exits nonzero, so both trajectories retain independent diagnostics.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  run_h200_formal_training.sh [--detach] <warmup_iters> <wsd_decay_iters> [train_iters] [result_root]

Arguments:
  warmup_iters     WSD linear-warmup length. No default is intentionally provided.
  wsd_decay_iters  WSD final minus-sqrt decay length. No default is intentionally provided.
  train_iters      Prefix of the formal 95,338-iteration schedule to run (default: 95338).
  result_root      Result parent directory (default: <workspace>/results/soap-8b-formal/h200).

Environment:
  SOAP_H200_BACKEND           Backend to run: both, hsolver, or torch (default: both).
  SOAP_H200_GLOBAL_BATCH_SIZE  Global batch size shared by both backends (default: 128).
  SOAP_H200_EVAL_ITERS         Validation/test iterations (default: 10; 0 disables evaluation).

Required GPU visibility:
  CUDA_VISIBLE_DEVICES=0,1,2,3

The two WSD lengths are deliberately required so no provisional schedule can
be launched by accident. Pass train_iters=2 or 20 to run an exact prefix of
the formal schedule after those lengths have been frozen.
EOF
    exit 2
}

detach=0
if [[ ${1:-} == --detach ]]; then
    detach=1
    shift
fi

if (( $# < 2 || $# > 4 )); then
    usage
fi

warmup_iters=$1
wsd_decay_iters=$2
train_iters=${3:-95338}

for value_name in warmup_iters wsd_decay_iters train_iters; do
    value=${!value_name}
    if [[ ! $value =~ ^[0-9]+$ ]]; then
        echo "$value_name must be a non-negative integer, got: $value" >&2
        exit 2
    fi
done

formal_train_iters=95338
backend_selection=${SOAP_H200_BACKEND:-both}
global_batch_size=${SOAP_H200_GLOBAL_BATCH_SIZE:-128}
data_split=90,9,1
eval_interval=300
eval_iters=${SOAP_H200_EVAL_ITERS:-10}
eval_global_batch_size=64
case $backend_selection in
    both)
        backends=(hsolver torch)
        ;;
    hsolver | torch)
        backends=("$backend_selection")
        ;;
    *)
        echo "SOAP_H200_BACKEND must be one of: both, hsolver, torch; got: $backend_selection" >&2
        exit 2
        ;;
esac
if [[ ! $global_batch_size =~ ^[1-9][0-9]*$ ]]; then
    echo "SOAP_H200_GLOBAL_BATCH_SIZE must be a positive integer, got: $global_batch_size" >&2
    exit 2
fi
if [[ ! $eval_iters =~ ^[0-9]+$ ]]; then
    echo "SOAP_H200_EVAL_ITERS must be a non-negative integer, got: $eval_iters" >&2
    exit 2
fi
if (( wsd_decay_iters == 0 )); then
    echo "wsd_decay_iters must be greater than zero" >&2
    exit 2
fi
if (( train_iters == 0 || train_iters > formal_train_iters )); then
    echo "train_iters must be between 1 and $formal_train_iters, got: $train_iters" >&2
    exit 2
fi
if (( warmup_iters + wsd_decay_iters >= formal_train_iters )); then
    echo "warmup_iters + wsd_decay_iters must be less than $formal_train_iters" >&2
    exit 2
fi

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd "$script_directory/../.." && pwd -P)
workspace_root=$(cd "$repository_root/.." && pwd -P)
environment_prefix=$(cd "$workspace_root/.." && pwd -P)/.conda/envs/hsolver-torch
result_root=${4:-$workspace_root/results/soap-8b-formal/h200}
master_port=${SOAP_H200_MASTER_PORT:-29500}

if [[ ! $master_port =~ ^[0-9]+$ ]] || (( master_port < 1 || master_port > 65535 )); then
    echo "SOAP_H200_MASTER_PORT must be an integer between 1 and 65535, got: $master_port" >&2
    exit 2
fi

data_root=/data1/ll/HPC/data/soap-8b-formal
data_args_path=$data_root/processed/nemotron-cc-high-actual-100b-eod2/data_args.txt
tokenizer_directory=$data_root/tokenizers/nemotron-3-nano-30b-a3b-base-pretrain-eod2
cache_directory=$data_root/cache/gpt-seq8192-seed42-split90-9-1

if [[ ${CUDA_VISIBLE_DEVICES:-} != 0,1,2,3 ]]; then
    echo "expected CUDA_VISIBLE_DEVICES=0,1,2,3, got: ${CUDA_VISIBLE_DEVICES:-not-set}" >&2
    exit 2
fi

for required_path in \
    "$environment_prefix/bin/python" \
    "$environment_prefix/bin/nvcc" \
    "$environment_prefix/bin/torchrun" \
    "$repository_root/pretrain_gpt.py" \
    "$workspace_root/Hsolver/pytorch/hsolver_torch" \
    "$workspace_root/Emerging-Optimizers/emerging_optimizers" \
    "$workspace_root/TransformerEngine" \
    "$data_args_path" \
    "$tokenizer_directory/tokenizer.json"; do
    if [[ ! -e $required_path ]]; then
        echo "required path does not exist: $required_path" >&2
        exit 2
    fi
done

timestamp=$(date +%Y%m%d_%H%M%S)
job_tag=${SLURM_JOB_ID:-manual}
run_directory=${SOAP_H200_RUN_DIRECTORY:-$result_root/pair_${train_iters}steps_gbs${global_batch_size}/job_${job_tag}_${timestamp}}
mkdir -p "$run_directory" "$cache_directory"
ln -sfn "$run_directory" "$result_root/latest"

if (( detach )); then
    launcher_log=$run_directory/launcher.log
    /usr/bin/nohup /usr/bin/setsid \
        env SOAP_H200_RUN_DIRECTORY="$run_directory" \
        "$script_directory/run_h200_formal_training.sh" \
        "$warmup_iters" "$wsd_decay_iters" "$train_iters" "$result_root" \
        >"$launcher_log" 2>&1 </dev/null &
    launcher_pid=$!
    printf '%s\n' "$launcher_pid" > "$run_directory/launcher.pid"
    printf 'RUNNING\n' > "$run_directory/pair_status.txt"
    echo "PAIR_LAUNCHER_PID=$launcher_pid"
    echo "RUN_DIRECTORY=$run_directory"
    echo "LAUNCHER_LOG=$launcher_log"
    exit 0
fi

export CONDA_PREFIX=$environment_prefix
export CUDA_HOME=$environment_prefix
export CUDACXX=$environment_prefix/bin/nvcc
export PATH="$environment_prefix/bin:$PATH"

site_packages=$environment_prefix/lib/python3.12/site-packages
cuda_library_paths="$site_packages/nvidia/cu13/lib:$site_packages/nvidia/nccl/lib:$site_packages/nvidia/cudnn/lib:$environment_prefix/lib:$environment_prefix/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="$cuda_library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

pair_manifest=$run_directory/manifest.log
environment_log=$run_directory/environment.log
pair_exit_file=$run_directory/exit_code.txt
printf 'RUNNING\n' > "$run_directory/pair_status.txt"

monitor_pid=
pair_finalized=0
on_worker_exit() {
    local worker_exit=$?
    if [[ -n ${monitor_pid:-} ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if (( pair_finalized == 0 )); then
        printf '%s\n' "$worker_exit" > "$pair_exit_file.tmp"
        mv "$pair_exit_file.tmp" "$pair_exit_file"
        printf 'FAILED\n' > "$run_directory/pair_status.txt"
    fi
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap on_worker_exit EXIT

git_binary=/data1/ll/.tools/git-env/bin/git
if [[ ! -x $git_binary ]]; then
    git_binary=$(command -v git)
fi

revision() {
    "$git_binary" -C "$1" rev-parse HEAD 2>/dev/null || printf 'unknown\n'
}

{
    echo "[Experiment]"
    echo "name: h200_soap_pair_${train_iters}steps_gbs${global_batch_size}"
    echo "purpose: SOAP paired training trajectories"
    echo "backend selection: $backend_selection"
    echo "backend order: ${backends[*]}"
    echo "formal train iterations: $formal_train_iters"
    echo "requested train iterations: $train_iters"
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
    echo "transformer layers per pipeline stage: 16"
    echo "local sequence length per context-parallel rank: 4096"
    echo "microbatches per DP rank per optimizer step: $global_batch_size"
    echo "tokens per optimizer step: $((8192 * global_batch_size))"
    echo "requested training tokens: $((8192 * global_batch_size * train_iters))"
    echo "formal training tokens: $((8192 * global_batch_size * formal_train_iters))"
    echo "tensor parallel size: 1"
    echo "pipeline parallel size: 2"
    echo "context parallel size: 2"
    echo "data parallel size: 1"
    echo "SOAP optimizer state sharding group size (DP x CP): 2"
    echo "optimizer: SOAP"
    echo "SOAP beta1: 0.9"
    echo "SOAP beta2: 0.95"
    echo "SOAP beta_kron/shampoo_beta: 0.95"
    echo "SOAP epsilon: 1e-8"
    echo "SOAP power iteration steps: 1"
    echo "SOAP Hsolver minimum QR dimension: 2048"
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
    echo "planned validation points: $((train_iters / eval_interval + 1))"
    echo "planned validation tokens: $(((train_iters / eval_interval + 1) * eval_iters * eval_global_batch_size * 8192))"
    echo "checkpointing: disabled"
    echo "distributed master address: 127.0.0.1"
    echo "distributed master port: $master_port"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "CUDA_HOME: $CUDA_HOME"
    echo "CUDACXX: $CUDACXX"
    echo "host: $(hostname)"
    echo "started at: $(date --iso-8601=seconds)"
    echo "Megatron-LM commit: $(revision "$repository_root")"
    echo "Emerging-Optimizers commit: $(revision "$workspace_root/Emerging-Optimizers")"
    echo "Hsolver commit: $(revision "$workspace_root/Hsolver")"
    echo "Transformer Engine commit: $(revision "$workspace_root/TransformerEngine")"
    echo "run directory: $run_directory"
} > "$pair_manifest"

"$environment_prefix/bin/python" - <<'PY' 2>&1 | tee "$environment_log"
import inspect
from importlib.metadata import version
from pathlib import Path

import torch
import transformer_engine
import transformer_engine_torch
import hsolver_torch
from emerging_optimizers.soap import SOAP
from megatron.core.extensions import transformer_engine as megatron_te
from megatron.core.optimizer import OptimizerConfig, USING_PYTORCH_OPTIMIZER

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Transformer Engine:", version("transformer-engine"))
print("Transformer Engine Python:", Path(transformer_engine.__file__).resolve())
print("Transformer Engine extension:", Path(transformer_engine_torch.__file__).resolve())
print("Hsolver extension:", Path(hsolver_torch.linalg._C.__file__).resolve())
print("Megatron HAVE_TE:", megatron_te.HAVE_TE)
print("Using PyTorch optimizer fallback:", USING_PYTORCH_OPTIMIZER)
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

config = OptimizerConfig(optimizer="soap")
assert config.soap_shampoo_beta == 0.95
assert inspect.signature(SOAP.__init__).parameters["power_iter_steps"].default == 1
print("SOAP beta_kron/shampoo_beta:", config.soap_shampoo_beta)
print("SOAP power_iter_steps:", 1)
print("H200 SOAP ENVIRONMENT CHECK: PASS")
PY

cat "$environment_log" >> "$pair_manifest"

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

stop_gpu_monitor() {
    if [[ -n ${monitor_pid:-} ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=
    fi
}

start_gpu_monitor() {
    local output_path=$1
    {
        echo "sample_time,index,name,memory.used MiB,memory.free MiB,utilization.gpu %,power.draw W"
        while true; do
            sample_time=$(date --iso-8601=seconds)
            /usr/bin/nvidia-smi \
                --id=0,1,2,3 \
                --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw \
                --format=csv,noheader,nounits \
                | /usr/bin/sed "s/^/$sample_time,/"
            sleep 10
        done
    } >> "$output_path" 2>&1 &
    monitor_pid=$!
}

run_backend() {
    local backend=$1
    local backend_directory=$run_directory/$backend
    local training_log=$backend_directory/train.log
    local command_log=$backend_directory/command.txt
    local gpu_log=$backend_directory/gpu_metrics.csv
    local exit_file=$backend_directory/exit_code.txt

    mkdir -p "$backend_directory"
    cp "$pair_manifest" "$backend_directory/manifest.log"
    {
        echo "SOAP QR backend: $backend"
        echo "checkpointing: disabled"
        echo "backend started at: $(date --iso-8601=seconds)"
    } >> "$backend_directory/manifest.log"
    printf 'RUNNING\n' > "$backend_directory/status.txt"

    local torchrun_command=(
        "$environment_prefix/bin/torchrun"
        --nnodes=1
        --node_rank=0
        --master_addr=127.0.0.1
        --master_port="$master_port"
        --nproc_per_node=4
        "$repository_root/pretrain_gpt.py"
        "${common_arguments[@]}"
        --soap-qr-backend "$backend"
    )

    {
        printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
        printf 'SOAP_EIGENBASIS_TRACKING=0 SOAP_EIGENBASIS_INVENTORY=0 SOAP_NUMERICS_DIAGNOSTICS=0 '
        printf 'CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8 '
        printf '%q ' "${torchrun_command[@]}"
        printf '\n'
    } > "$command_log"

    start_gpu_monitor "$gpu_log"

    local launch_command=("${torchrun_command[@]}")
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
    local training_exit=${PIPESTATUS[0]}
    set -e

    stop_gpu_monitor
    printf '%s\n' "$training_exit" > "$exit_file.tmp"
    mv "$exit_file.tmp" "$exit_file"
    if (( training_exit == 0 )); then
        printf 'COMPLETED\n' > "$backend_directory/status.txt"
    else
        printf 'FAILED\n' > "$backend_directory/status.txt"
    fi
    {
        echo "backend finished at: $(date --iso-8601=seconds)"
        echo "exit code: $training_exit"
    } >> "$backend_directory/manifest.log"

    return "$training_exit"
}

cd "$repository_root"

pair_exit=0
hsolver_exit=NOT_RUN
torch_exit=NOT_RUN
for backend in "${backends[@]}"; do
    if run_backend "$backend"; then
        backend_exit=0
    else
        backend_exit=$?
        pair_exit=1
    fi
    if [[ $backend == hsolver ]]; then
        hsolver_exit=$backend_exit
    else
        torch_exit=$backend_exit
    fi
done

{
    echo "hsolver exit code: $hsolver_exit"
    echo "torch exit code: $torch_exit"
    echo "pair exit code: $pair_exit"
    echo "finished at: $(date --iso-8601=seconds)"
} > "$run_directory/pair_summary.log"

printf '%s\n' "$pair_exit" > "$pair_exit_file.tmp"
mv "$pair_exit_file.tmp" "$pair_exit_file"
if (( pair_exit == 0 )); then
    printf 'COMPLETED\n' > "$run_directory/pair_status.txt"
else
    printf 'FAILED\n' > "$run_directory/pair_status.txt"
fi
pair_finalized=1

echo "RUN_DIRECTORY=$run_directory"
echo "HSOLVER_EXIT=$hsolver_exit"
echo "TORCH_EXIT=$torch_exit"
exit "$pair_exit"
