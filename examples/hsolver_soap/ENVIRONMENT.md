# Hsolver SOAP environment

This document creates the software environment used by the Hsolver SOAP
experiments without modifying the system CUDA installation. The Conda file
installs the Python stack and CUDA Toolkit 13.2 Update 1. Transformer Engine,
Hsolver, Emerging-Optimizers, and Megatron Core are then installed from their
pinned source checkouts.

## Installation boundary

The host provides:

- the NVIDIA kernel driver and GPUs;
- GCC, G++, and CMake;
- Git and the operating-system runtime.

The dedicated Conda environment provides:

- Python 3.12;
- CUDA Toolkit 13.2 Update 1, including `nvcc`, headers, and development
  libraries;
- PyTorch 2.12.1 built for CUDA 13.2;
- the remaining Python and source-build dependencies.

Do not combine CUDA 13.1 headers with CUDA 13.2 libraries. Once the environment
is active, `CUDA_HOME`, `CUDACXX`, and the leading library search paths must all
refer to the Conda environment. Conda does not install or replace the kernel
driver.

## Reference revisions

The integration was prepared with these source revisions:

```text
Hsolver              5df745ef2525935a79b14c40f9ab7152124dd2a2
Emerging-Optimizers  fdd858259af5d95cf438af9bb511955f88bf5a03
Transformer Engine   e7c550c5f80636cf841a8204b1d6f85a5f3f28b7
```

This document is versioned by the Megatron-LM commit that contains it. Record
the Megatron-LM commit from every experiment manifest rather than duplicating
that revision here.

The expected repository layout is:

```text
<workspace>/
  Emerging-Optimizers/
  Hsolver/
  Megatron-LM/
  TransformerEngine/
```

## 1. Define portable paths

Start from the Megatron-LM checkout. These commands derive all paths from the
checkout location and do not assume that `$HOME/HPC` exists:

```bash
cd /path/to/workspace/Megatron-LM

export MEGATRON_ROOT="$(pwd -P)"
export WORKSPACE_ROOT="$(dirname "$MEGATRON_ROOT")"
export ENV_PREFIX="$(dirname "$WORKSPACE_ROOT")/.conda/envs/hsolver-torch"

printf 'Megatron:  %s\nWorkspace: %s\nEnvironment: %s\n' \
  "$MEGATRON_ROOT" "$WORKSPACE_ROOT" "$ENV_PREFIX"
```

For a checkout at `/home/qf/ll/HPC/Megatron-LM`, the derived environment is
`/home/qf/ll/.conda/envs/hsolver-torch`.

## 2. Check the host

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
gcc --version | head -n 1
g++ --version | head -n 1
cmake --version | head -n 1
```

The reference H200 host used driver 590.48.01. CUDA 13.x minor-version
compatibility accepts drivers from the R580 family onward, but the driver
shipped with CUDA 13.2 Update 1 is 595.58.03. Treat a driver older than
595.58.03 as a compatibility-mode configuration and complete all runtime tests
below before collecting formal results. See the NVIDIA CUDA 13.2 release notes:

<https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html>

## 3. Create the isolated environment

Load Conda into the current shell, then create an environment at the explicit
user-owned prefix. The `--prefix` argument overrides the portable environment
name in `environment.yml`.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create \
  --prefix "$ENV_PREFIX" \
  --file "$MEGATRON_ROOT/examples/hsolver_soap/environment.yml"

conda activate "$ENV_PREFIX"
```

The environment is persistent. A new shell only needs the `source` and
`conda activate` commands; it does not need to recreate the environment.

The YAML intentionally contains no proxy settings. Configure or temporarily
disable a site-specific proxy outside the repository.

## 4. Select the isolated CUDA toolkit

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

command -v nvcc
"$CUDACXX" --version

grep -R -m 1 "cusolverDnXstedc" \
  "$CUDA_HOME/include" \
  "$CUDA_HOME/targets/x86_64-linux/include"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY
```

Expected results are CUDA 13.2 from `nvcc`, PyTorch `2.12.1+cu132`, and compute
capability `(9, 0)` on H100/H200 or `(8, 9)` on RTX 4090.

## 5. Prepare Transformer Engine source

Clone Transformer Engine beside the other repositories if it is not already
present:

```bash
cd "$WORKSPACE_ROOT"
git clone --recursive git@github.com:NVIDIA/TransformerEngine.git
cd TransformerEngine
git switch --detach e7c550c5f80636cf841a8204b1d6f85a5f3f28b7
git submodule update --init --recursive
```

For an existing checkout, do not clone it again. Verify that it is clean before
switching revisions.

PyTorch's CUDA wheel supplies the cuDNN and NCCL headers outside the normal
system include path. Make those paths visible to the Transformer Engine build:

```bash
export SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
export CPATH="$SITE_PACKAGES/nvidia/nccl/include:$SITE_PACKAGES/nvidia/cudnn/include:${CPATH:-}"
export LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cudnn/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

## 6. Build Transformer Engine for the target GPU

H100 and H200 use compute capability 9.0. RTX 4090 uses 8.9. Build only the
architecture used by the target machine:

```bash
cd "$WORKSPACE_ROOT/TransformerEngine"

export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90
export NVTE_WITH_NCCL_EP=0
export MAX_JOBS=8
export NVTE_BUILD_THREADS_PER_JOB=2

python -m pip install -v --no-build-isolation --no-deps .
```

For RTX 4090, replace `NVTE_CUDA_ARCHS=90` with `NVTE_CUDA_ARCHS=89`.
`NVTE_WITH_NCCL_EP=0` matches the reference training environment; SOAP does not
use Transformer Engine's optional NCCL expert-parallel extension.

## 7. Build and install Hsolver

```bash
cd "$WORKSPACE_ROOT/Hsolver/pytorch"

export TORCH_CUDA_ARCH_LIST="9.0"
python -m pip install -v --no-build-isolation --no-deps -e .
```

For RTX 4090, use `TORCH_CUDA_ARCH_LIST="8.9"`. The architecture variable is a
build-time choice and does not need to remain set during training.

## 8. Install the Python source packages

```bash
python -m pip install --no-build-isolation --no-deps \
  -e "$WORKSPACE_ROOT/Emerging-Optimizers"

python -m pip install --no-build-isolation --no-deps \
  -e "$WORKSPACE_ROOT/Megatron-LM"
```

Editable installation keeps imports connected to the Git checkouts. Python
source edits take effect immediately. C++ or CUDA source edits still require a
rebuild.

## 9. Validate the complete stack

Run this import audit from the Megatron-LM checkout:

```bash
cd "$MEGATRON_ROOT"

python - <<'PY'
from importlib.metadata import version
from pathlib import Path

import hsolver_torch
import torch
import transformer_engine
import transformer_engine_torch
from emerging_optimizers import registry
from megatron.core.datasets import helpers_cpp
from megatron.core.optimizer import emerging_optimizers as megatron_eo

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Transformer Engine:", version("transformer-engine"))
print("Transformer Engine Python:", Path(transformer_engine.__file__).resolve())
print("Transformer Engine extension:", Path(transformer_engine_torch.__file__).resolve())
print("Hsolver:", Path(hsolver_torch.__file__).resolve())
print("Megatron helpers:", Path(helpers_cpp.__file__).resolve())
print("SOAP in Emerging-Optimizers:", "soap" in registry.get_optimizer_name_list())
print("SOAP in Megatron:", "soap" in megatron_eo._EMERGING_OPTIMIZERS)
print("Visible GPUs:", torch.cuda.device_count())

assert torch.__version__ == "2.12.1+cu132"
assert torch.version.cuda == "13.2"
assert version("transformer-engine") == "2.18.0+e7c550c5"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) in {(8, 9), (9, 0)}
assert "soap" in registry.get_optimizer_name_list()
assert "soap" in megatron_eo._EMERGING_OPTIMIZERS
print("HSOLVER SOAP ENVIRONMENT: PASS")
PY

python -m pip check
python -m pytest -q "$WORKSPACE_ROOT/Hsolver/pytorch/tests/test_qr.py"
```

The final test requires an allocated GPU. The environment creation and source
builds do not require all training GPUs to be visible.

## 10. Record the resolved environment

Before a formal run, retain the exact resolution in the result directory:

```bash
mkdir -p "$WORKSPACE_ROOT/results/environment"

conda list --explicit \
  > "$WORKSPACE_ROOT/results/environment/conda-explicit.txt"

python -m pip freeze --all \
  > "$WORKSPACE_ROOT/results/environment/pip-freeze.txt"

git -C "$WORKSPACE_ROOT/Hsolver" rev-parse HEAD \
  > "$WORKSPACE_ROOT/results/environment/hsolver-commit.txt"

git -C "$WORKSPACE_ROOT/Emerging-Optimizers" rev-parse HEAD \
  > "$WORKSPACE_ROOT/results/environment/emerging-optimizers-commit.txt"

git -C "$WORKSPACE_ROOT/Megatron-LM" rev-parse HEAD \
  > "$WORKSPACE_ROOT/results/environment/megatron-commit.txt"

git -C "$WORKSPACE_ROOT/TransformerEngine" rev-parse HEAD \
  > "$WORKSPACE_ROOT/results/environment/transformer-engine-commit.txt"
```

These resolved manifests document the exact installed state. They are result
artifacts, not portable replacements for `environment.yml`.
