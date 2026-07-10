#!/bin/bash
#SBATCH --job-name=docverify_eval_sidtd_pareto
#SBATCH --partition=gpu-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=eval_sidtd_pareto_%j.log
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=uo29067@uniovi.es
#SBATCH --requeue

# Activar entorno conda
source /soft/miniconda3/bin/activate
conda activate docverify

# Ir al directorio del proyecto
cd ~/mywork_HPC

# Evaluar los modelos Pareto
export EXPORT_DIR=./exports_hpo_pareto_nested

# === EJECUCIÓN ===
echo "[SLURM] Lanzando evaluación cross-dataset sobre SIDTD (Pareto)"
python evaluate_sidtd.py

status=$?
if [ $status -ne 0 ]; then
    echo "[SLURM] La evaluación falló con código $status"
    exit $status
fi

echo "[SLURM] Evaluación completada. Resultados en exports_hpo_pareto_nested/"
