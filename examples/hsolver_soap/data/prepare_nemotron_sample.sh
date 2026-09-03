#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_root=${1:-}

if [[ -z "$data_root" ]]; then
  echo "Usage: $0 DATA_ROOT" >&2
  exit 2
fi

workers=${SOAP_DATA_WORKERS:-8}
args=(--data-root "$data_root" --workers "$workers")

if [[ ${SOAP_OFFLINE:-0} == 1 ]]; then
  args+=(--offline)
fi

export TOKENIZERS_PARALLELISM=false

python "$script_dir/prepare_nemotron_sample.py" "${args[@]}"

