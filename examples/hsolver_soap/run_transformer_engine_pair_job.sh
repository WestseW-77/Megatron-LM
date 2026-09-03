#!/usr/bin/env bash
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4090:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --exclusive

# Run the Hsolver and Torch QR trajectories for one formal TE scenario.

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: sbatch [options] $0 <seq128|seq1024>" >&2
    exit 2
fi

scenario=$1

case $scenario in
    seq128)
        steps=4000
        backend_timeout=15h
        ;;
    seq1024)
        steps=2400
        backend_timeout=10h
        ;;
    *)
        echo "scenario must be 'seq128' or 'seq1024', got: $scenario" >&2
        exit 2
        ;;
esac

source /opt/conda/etc/profile.d/conda.sh
conda activate hsolver-torch

repository_root=/home/ll/HPC/Megatron-LM
runner="$repository_root/examples/hsolver_soap/run_transformer_engine_training.sh"
data_root=/home/ll/HPC/data/soap-8b
formal_root=/home/ll/HPC/results/soap-8b/transformer-engine/formal

cd "$repository_root"
mkdir -p "$formal_root/slurm"

echo "FORMAL_TE_PAIR_SCENARIO=$scenario"
echo "FORMAL_TE_PAIR_STEPS=$steps"
echo "FORMAL_TE_PAIR_ORDER=hsolver,torch"
echo "FORMAL_TE_PAIR_JOB_ID=${SLURM_JOB_ID:-not-set}"
echo "FORMAL_TE_PAIR_HOST=$(hostname)"
echo "FORMAL_TE_PAIR_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

SOAP_TE_TIMEOUT=$backend_timeout \
bash "$runner" hsolver "$scenario" "$steps" \
    "$data_root" "$formal_root"

SOAP_TE_TIMEOUT=$backend_timeout \
bash "$runner" torch "$scenario" "$steps" \
    "$data_root" "$formal_root"

echo "FORMAL TRANSFORMER ENGINE PAIR: PASS"
