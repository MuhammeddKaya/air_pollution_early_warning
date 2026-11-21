#!/bin/bash
# Tam koşu (Adım 3): 81 iş (1 naive + 2 omurga × 2 öznitelik × 4 UQ × 5 tohum). 
# Aynı anda en fazla 20 iş (%20) — kuyruk kotasına göre değiştirin.
#SBATCH --job-name=pm25_A
#SBATCH --account=<HESAP_ADI>
#SBATCH --partition=<GPU_KUYRUGU>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=1-81%20
#SBATCH --output=logs/job_%A_%a.out

cd $SLURM_SUBMIT_DIR
source venv/bin/activate
export PYTHONPATH=.
mkdir -p logs
# Her iş kendi sonuç dosyasını results/runs/ altına yazar; dosya varsa atlar → iş kesilirse
# aynı komutla yeniden gönderilebilir (yalnız eksikler koşar).
python -m pm25_early_warning.run_experiment --job $SLURM_ARRAY_TASK_ID --out results
