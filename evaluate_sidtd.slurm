#!/bin/bash
#SBATCH --job-name=docverify_eval_sidtd
#SBATCH --partition=gpu-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=eval_sidtd_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uo29067@uniovi.es

# Activar entorno conda
source /soft/miniconda3/bin/activate
conda activate docverify

# Ir al directorio del proyecto
cd ~/mywork_HPC

# === EJECUCIÓN ===
echo "[SLURM] Lanzando evaluación cross-dataset sobre SIDTD"
python evaluate_sidtd.py

status=$?
if [ $status -ne 0 ]; then
    echo "[SLURM] La evaluación falló con código $status"
    exit $status
fi

echo "[SLURM] Evaluación completada. Resultados en exports_hpo_pareto_nested/"
