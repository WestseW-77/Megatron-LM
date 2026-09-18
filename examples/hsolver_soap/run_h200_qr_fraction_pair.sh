#!/usr/bin/env bash
# Run matched Torch and Hsolver SOAP QR timing jobs at GBS=128.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  run_h200_qr_fraction_pair.sh [--detach] [result_root]

The script runs two 30-step jobs sequentially: Torch first, then Hsolver.
Both jobs profile iterations 20..30 and differ only in --soap-qr-backend.

Environment:
  SOAP_H200_MASTER_PORT  torchrun master port reused sequentially (default: 30300).
  SOAP_H200_TIMEOUT      Optional per-backend timeout accepted by GNU timeout.

Required GPU visibility:
  CUDA_VISIBLE_DEVICES=0,1,2,3
EOF
    exit 2
}

detach=0
if [[ ${1:-} == --detach ]]; then
    detach=1
    shift
fi
if (( $# > 1 )); then
    usage
fi

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd "$script_directory/../.." && pwd -P)
workspace_root=$(cd "$repository_root/.." && pwd -P)
environment_prefix=$(cd "$workspace_root/.." && pwd -P)/.conda/envs/hsolver-torch
result_root=${1:-$workspace_root/results/soap-8b-formal/h200/qr_fraction_gbs128}
master_port=${SOAP_H200_MASTER_PORT:-30300}
worker_script=$script_directory/run_h200_qr_fraction_backend.sh
comparison_script=$script_directory/compare_qr_fraction_pair.py

if [[ ${CUDA_VISIBLE_DEVICES:-} != 0,1,2,3 ]]; then
    echo "expected CUDA_VISIBLE_DEVICES=0,1,2,3, got: ${CUDA_VISIBLE_DEVICES:-not-set}" >&2
    exit 2
fi
if [[ ! $master_port =~ ^[0-9]+$ ]] || (( master_port < 1 || master_port > 65535 )); then
    echo "SOAP_H200_MASTER_PORT must be between 1 and 65535, got: $master_port" >&2
    exit 2
fi
for required_path in "$worker_script" "$comparison_script" "$environment_prefix/bin/python"; do
    if [[ ! -e $required_path ]]; then
        echo "required path does not exist: $required_path" >&2
        exit 2
    fi
done

timestamp=$(date +%Y%m%d_%H%M%S)
job_tag=${SLURM_JOB_ID:-manual}
pair_directory=${SOAP_H200_PAIR_RUN_DIRECTORY:-$result_root/pair_job_${job_tag}_${timestamp}}
mkdir -p "$pair_directory"
ln -sfn "$pair_directory" "$result_root/latest"

if (( detach )); then
    launcher_log=$pair_directory/launcher.log
    /usr/bin/nohup /usr/bin/setsid \
        env SOAP_H200_PAIR_RUN_DIRECTORY="$pair_directory" \
        "$script_directory/run_h200_qr_fraction_pair.sh" "$result_root" \
        >"$launcher_log" 2>&1 </dev/null &
    launcher_pid=$!
    printf '%s\n' "$launcher_pid" > "$pair_directory/launcher.pid"
    printf 'RUNNING\n' > "$pair_directory/pair_status.txt"
    echo "PAIR_LAUNCHER_PID=$launcher_pid"
    echo "PAIR_DIRECTORY=$pair_directory"
    echo "LAUNCHER_LOG=$launcher_log"
    exit 0
fi

pair_exit_file=$pair_directory/exit_code.txt
pair_manifest=$pair_directory/manifest.log
pair_finalized=0
printf 'RUNNING\n' > "$pair_directory/pair_status.txt"

on_exit() {
    local worker_exit=$?
    if (( pair_finalized == 0 )); then
        printf '%s\n' "$worker_exit" > "$pair_exit_file.tmp"
        mv "$pair_exit_file.tmp" "$pair_exit_file"
        printf 'FAILED\n' > "$pair_directory/pair_status.txt"
    fi
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap on_exit EXIT

{
    echo "[Experiment]"
    echo "name: h200_soap_qr_fraction_pair_gbs128"
    echo "purpose: attribute matched Torch/Hsolver end-to-end savings to SOAP QR"
    echo "backend order: torch,hsolver"
    echo "global batch size: 128"
    echo "train iterations per backend: 30"
    echo "profiling iterations: 20..30 inclusive"
    echo "only backend difference: --soap-qr-backend"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "distributed master port: $master_port"
    echo "started at: $(date --iso-8601=seconds)"
    echo "pair directory: $pair_directory"
    echo "worker SHA256: $(/usr/bin/sha256sum "$worker_script" | /usr/bin/awk '{print $1}')"
    echo "QR analyzer SHA256: $(/usr/bin/sha256sum "$script_directory/analyze_qr_fraction.py" | /usr/bin/awk '{print $1}')"
    echo "pair analyzer SHA256: $(/usr/bin/sha256sum "$comparison_script" | /usr/bin/awk '{print $1}')"
    echo "optimizer profiler SHA256: $(/usr/bin/sha256sum "$repository_root/megatron/training/optimizer_fraction_profiler.py" | /usr/bin/awk '{print $1}')"
    echo "QR dispatch SHA256: $(/usr/bin/sha256sum "$workspace_root/Emerging-Optimizers/emerging_optimizers/utils/eig.py" | /usr/bin/awk '{print $1}')"
} > "$pair_manifest"

pair_exit=0
for backend in torch hsolver; do
    backend_directory=$pair_directory/$backend
    mkdir -p "$backend_directory"
    echo "[$(date --iso-8601=seconds)] starting $backend backend"
    set +e
    "$worker_script" "$backend" "$backend_directory" "$master_port"
    backend_exit=$?
    set -e
    if (( backend_exit != 0 )); then
        echo "$backend backend failed with exit code $backend_exit" >&2
        pair_exit=$backend_exit
        break
    fi
done

if (( pair_exit == 0 )); then
    if ! "$environment_prefix/bin/python" "$comparison_script" "$pair_directory" \
        2>&1 | tee "$pair_directory/comparison.log"; then
        pair_exit=1
    fi
fi

printf '%s\n' "$pair_exit" > "$pair_exit_file.tmp"
mv "$pair_exit_file.tmp" "$pair_exit_file"
if (( pair_exit == 0 )); then
    printf 'COMPLETED\n' > "$pair_directory/pair_status.txt"
else
    printf 'FAILED\n' > "$pair_directory/pair_status.txt"
fi
{
    echo "finished at: $(date --iso-8601=seconds)"
    echo "exit code: $pair_exit"
} >> "$pair_manifest"
pair_finalized=1

echo "PAIR_DIRECTORY=$pair_directory"
echo "EXIT_CODE=$pair_exit"
exit "$pair_exit"
