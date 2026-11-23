#!/bin/bash
# Kısa deneme koşusu (Adım 2): naive + tek model işi (2 epoch).
# pm25_project/ klasöründen gönderin:  sbatch slurm/smoke.slurm
#SBATCH -p barbun
#SBATCH -A muhammedkaya
#SBATCH -J pm25_smoke
#SBATCH -o logs/smoke_%j.out
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --time=3-00:00:00

cd $SLURM_SUBMIT_DIR

# Conda ortamını etkinleştirme
eval "$(/arf/home/muhammedkaya/miniconda3/bin/conda shell.bash hook)"
conda activate dl-env

export PYTHONPATH=.
export OMP_NUM_THREADS=$SLURM_NTASKS     # torch CPU iş parçacığı sayısı
export MKL_NUM_THREADS=$SLURM_NTASKS

python -m pm25_early_warning.run_experiment --job 1 --out results_smoke --device cpu             # naive
python -m pm25_early_warning.run_experiment --job 2 --out results_smoke --device cpu --epochs 2  # ilk model işi
