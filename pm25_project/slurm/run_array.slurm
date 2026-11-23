#!/bin/bash
# Tam koşu (Adım 3): 81 iş (1 naive + 2 omurga × 2 öznitelik × 4 UQ × 5 tohum).
# Aynı anda en fazla 20 iş (%20) — kuyruk kotasına göre değiştirin.
# pm25_project/ klasöründen gönderin:  sbatch slurm/run_array.slurm
#SBATCH -p barbun
#SBATCH -C weka
#SBATCH -A muhammedkaya
#SBATCH -J pm25_A
#SBATCH -o logs/job_%A_%a.out
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --time=3-00:00:00
#SBATCH --array=1-81%20

cd $SLURM_SUBMIT_DIR

# Conda ortamını etkinleştirme
eval "$(/arf/home/muhammedkaya/miniconda3/bin/conda shell.bash hook)"
conda activate dl-env

export PYTHONPATH=.
export OMP_NUM_THREADS=$SLURM_NTASKS     # torch CPU iş parçacığı sayısı
export MKL_NUM_THREADS=$SLURM_NTASKS

# Her iş kendi sonuç dosyasını results/runs/ altına yazar; dosya varsa atlar → iş kesilirse
# aynı komutla yeniden gönderilebilir (yalnız eksikler koşar).
python -m pm25_early_warning.run_experiment --job $SLURM_ARRAY_TASK_ID --out results --device cpu
