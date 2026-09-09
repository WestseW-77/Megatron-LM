#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_root=${1:-}

if [[ -z $data_root ]]; then
    echo "Usage: $0 DATA_ROOT" >&2
    exit 2
fi

workers=${NEMOTRON_CC_WORKERS:-8}
args=(
    --data-root "$data_root"
    --workers "$workers"
    --target-text-tokens 100000000000
)

if [[ ${NEMOTRON_CC_OFFLINE:-0} == 1 ]]; then
    args+=(--offline)
fi
if [[ ${NEMOTRON_CC_IGNORE_PROXY:-0} == 1 ]]; then
    args+=(--ignore-proxy)
fi
if [[ ${NEMOTRON_CC_PLAN_ONLY:-0} == 1 ]]; then
    args+=(--plan-only)
fi
if [[ ${NEMOTRON_CC_REMOVE_COMPRESSED:-0} == 1 ]]; then
    args+=(--remove-compressed-after-indexing)
fi
if [[ ${NEMOTRON_CC_VERIFY_EXISTING_HASHES:-0} == 1 ]]; then
    args+=(--verify-existing-hashes)
fi

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

python "$script_dir/prepare_nemotron_cc_high_actual.py" "${args[@]}"
