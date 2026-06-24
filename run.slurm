#!/bin/bash
#SBATCH --job-name=docverify
#SBATCH --partition=gpu-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=salida_%j.log
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=uo29067@uniovi.es
#SBATCH --requeue

#Activar entorno conda
source /soft/miniconda3/bin/activate
conda activate docverify

# Ir al directorio del proyecto
cd ~/IDVerify_HPC

# === EJECUCIÓN ===
echo "[SLURM] Lanzando entrenamiento con reanudación automática"
python main.py

# Si Python devuelve un código de error, pedir requeue explícito
status=$?
if [ $status -ne 0 ]; then
    echo "[SLURM] El job falló con código $status — solicitando requeue"
    scontrol requeue $SLURM_JOB_ID
fi
