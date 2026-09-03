#!/usr/bin/env bash
# Run independent Torch and Hsolver SOAP trajectories, then compare their summaries.

set -euo pipefail

if (( $# < 1 || $# > 4 )); then
    echo "Usage: $0 <seq128|seq1024> [steps] [data_root] [result_root]" >&2
    exit 2
fi

scenario=$1
steps=${2:-120}
data_root=${3:-/home/ll/HPC/data/soap-8b}
result_root=${4:-/home/ll/HPC/results/soap-8b/eigenbasis-paper-candidate}

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_directory/../.." && pwd)

case $scenario in
    seq128)
        targets_file="$script_directory/eigenbasis_targets_layer0_layer3_2048.json"
        expected_targets=2
        ;;
    seq1024)
        targets_file="$script_directory/eigenbasis_targets_4stage_2048.json"
        expected_targets=4
        ;;
    *)
        echo "scenario must be 'seq128' or 'seq1024', got: $scenario" >&2
        exit 2
        ;;
esac

if [[ ! $steps =~ ^[1-9][0-9]*$ ]]; then
    echo "steps must be a positive integer, got: $steps" >&2
    exit 2
fi

job_tag=${SLURM_JOB_ID:-manual}
timestamp=$(date +%Y%m%d_%H%M%S)
pair_directory="$result_root/$scenario/${steps}steps/job_${job_tag}_${timestamp}"
run_root="$pair_directory/runs"
torch_driver_log="$pair_directory/torch_driver.log"
hsolver_driver_log="$pair_directory/hsolver_driver.log"
comparison_log="$pair_directory/independent_comparison.log"
comparison_json="$pair_directory/independent_comparison.json"
comparison_csv="$pair_directory/independent_comparison.csv"
tracking_timeout=${SOAP_TRACKING_TIMEOUT:-4h}

mkdir -p "$pair_directory"
cd "$repository_root"

{
    echo "[Paper-candidate eigenbasis experiment]"
    echo "scenario: $scenario"
    echo "steps per independent backend: $steps"
    echo "expected tracked targets per backend: $expected_targets"
    echo "targets file: $targets_file"
    echo "Torch and Hsolver share optimizer state: no"
    echo "alternate/shadow backend executed inside either trajectory: no"
    echo "reference: fresh FP32 EVD of each trajectory's own current factor"
    echo "primary metric 1: aggregate relative eigen-residual"
    echo "primary metric 2: leading subspace overlap at 99% trace energy"
    echo "secondary metric: tail residual energy share"
    echo "result directory: $pair_directory"
    echo "slurm job: ${SLURM_JOB_ID:-not-set}"
    echo "host: $(hostname)"
} | tee "$pair_directory/manifest.log"

set +e
SOAP_EIGENBASIS_SPECTRAL_DIAGNOSTICS=1 \
SOAP_TRACKING_TIMEOUT="$tracking_timeout" \
bash "$script_directory/run_actual_eigenbasis_tracking.sh" \
    torch \
    "$scenario" \
    "$steps" \
    "$targets_file" \
    "$data_root" \
    "$run_root" \
    2>&1 | tee "$torch_driver_log"
torch_exit=${PIPESTATUS[0]}
set -e

echo "PAPER_EIGENBASIS_TORCH_EXIT=$torch_exit" | tee "$torch_driver_log.exit"
if (( torch_exit != 0 )); then
    echo "Torch trajectory failed; Hsolver was not started."
    echo "PAIR_DIRECTORY=$pair_directory"
    exit "$torch_exit"
fi

set +e
SOAP_EIGENBASIS_SPECTRAL_DIAGNOSTICS=1 \
SOAP_TRACKING_TIMEOUT="$tracking_timeout" \
bash "$script_directory/run_actual_eigenbasis_tracking.sh" \
    hsolver \
    "$scenario" \
    "$steps" \
    "$targets_file" \
    "$data_root" \
    "$run_root" \
    2>&1 | tee "$hsolver_driver_log"
hsolver_exit=${PIPESTATUS[0]}
set -e

echo "PAPER_EIGENBASIS_HSOLVER_EXIT=$hsolver_exit" | tee "$hsolver_driver_log.exit"
if (( hsolver_exit != 0 )); then
    echo "Hsolver trajectory failed; comparison was not run."
    echo "PAIR_DIRECTORY=$pair_directory"
    exit "$hsolver_exit"
fi

mapfile -t torch_runs < <(find "$run_root/$scenario/torch" -mindepth 1 -maxdepth 1 -type d | sort)
mapfile -t hsolver_runs < <(find "$run_root/$scenario/hsolver" -mindepth 1 -maxdepth 1 -type d | sort)
if (( ${#torch_runs[@]} != 1 || ${#hsolver_runs[@]} != 1 )); then
    echo "Expected exactly one Torch and one Hsolver run under $run_root" >&2
    exit 1
fi

torch_run=${torch_runs[0]}
hsolver_run=${hsolver_runs[0]}

python "$script_directory/compare_independent_spectral_runs.py" \
    "$torch_run/spectral_summary.json" \
    "$hsolver_run/spectral_summary.json" \
    --json-output "$comparison_json" \
    --csv-output "$comparison_csv" \
    2>&1 | tee "$comparison_log"

echo "PAPER EIGENBASIS PAIR: PASS"
echo "PAIR_DIRECTORY=$pair_directory"
echo "TORCH_RUN_DIRECTORY=$torch_run"
echo "HSOLVER_RUN_DIRECTORY=$hsolver_run"
echo "COMPARISON_JSON=$comparison_json"
echo "COMPARISON_CSV=$comparison_csv"
