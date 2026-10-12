#!/bin/bash

#SBATCH --partition=Orion
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=simp_trajectory
#SBATCH --output=results/logs/simp_trajectory_%j.log
#SBATCH --error=results/logs/simp_trajectory_%j.log

echo -e "==============================================="
echo -e "  HOST: `hostname -s`
  TIME: `date`
  WDIR: $SLURM_SUBMIT_DIR"
echo -e "==============================================="

cd "$SLURM_SUBMIT_DIR" || exit 1
module load python/3.12
module load openmpi/4.1.6
source ~/.venv/bin/activate

mkdir -p results/logs
ulimit -n 4096
echo "File descriptor limit set to: $(ulimit -n)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python generate_trajectory_dataset.py

echo "Trajectory dataset generation completed"
