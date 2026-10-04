#!/bin/bash
# Slurm wrapper for run_nd_sweep.sh (one GPU per job).
#
# Usage (from the repository root):
#   sbatch batch_nd_sweep.sh configs/<model>.conf
#
# Add the partition / account directives your cluster requires, e.g.
#   #SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --job-name=nd_sweep
#SBATCH --output=slurm-nd_sweep-%j.out
#SBATCH --time=2-00:00:00
set -euo pipefail

CONFIG="${1:?Usage: sbatch batch_nd_sweep.sh configs/<model>.conf}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

bash run_nd_sweep.sh "${CONFIG}"
