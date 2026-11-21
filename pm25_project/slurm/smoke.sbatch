#!/bin/bash
# Kısa deneme koşusu (Adım 2): tek iş, 2 epoch. <...> alanlarını TRUBA hesabınıza göre doldurun.
#SBATCH --job-name=pm25_smoke
#SBATCH --account=<HESAP_ADI>
#SBATCH --partition=<GPU_KUYRUGU>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out

cd $SLURM_SUBMIT_DIR
source venv/bin/activate          # ortam kurulumu için TRUBA_YONERGE.md Adım 1
export PYTHONPATH=.
mkdir -p logs
python -m pm25_early_warning.run_experiment --job 1 --out results_smoke            # naive
python -m pm25_early_warning.run_experiment --job 2 --out results_smoke --epochs 2 # ilk model işi
