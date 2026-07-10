"""
config.py — Configuración global del proyecto DocVerify.
Modifica este archivo para ajustar rutas, hiperparámetros y comportamiento del experimento.
"""

import os
from datetime import datetime
from pathlib import Path

# ============================================================
# RUTAS DEL DATASET
# ============================================================

# Directorio raíz donde está descomprimido FantasyID.
# Puede ser una ruta absoluta o relativa al directorio de ejecución.
# Ejemplos:
#   DATASET_ROOT = Path("./FantasyID")
#   DATASET_ROOT = Path("C:/Datasets/FantasyID")
#   DATASET_ROOT = Path("/home/user/data/FantasyID")
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "./FantasyID"))

# Directorio raíz donde está descomprimido SIDTD.
# Se descarga automáticamente desde http://datasets.cvc.uab.es/SIDTD/
# si no existe y se invoca dataset_sidtd.download_sidtd().
SIDTD_ROOT = Path(os.getenv("SIDTD_ROOT", "./Dataset_SIDTD"))

# ============================================================
# DIRECTORIO DE EXPORTACIÓN DE RESULTADOS
# ============================================================
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", f"./exports_{datetime.now():%Y%m%d_%H%M%S}"))

# ============================================================
# IMAGEN Y MODELO
# ============================================================
PATCH_SIZE = 224              # Resolución de entrada al modelo (alto y ancho)
IMG_EXTS   = {".jpg", ".png"} # Extensiones de imagen aceptadas

# ============================================================
# ARQUITECTURA
# ============================================================
LEAKY_RELU_ALPHA = 0.2   # Fijo — eliminado del HPO (sin señal en 250 trials)
# ============================================================
# lr:           log-uniform [5e-5, 9e-4]  — rango inferior eliminado (lr<5e-5 → Dice≈0 en 10 epochs)
# weight_decay: log-uniform [1e-7, 1e-4]
# dropout_rate: uniform [0.1, 0.4]        — rango reducido (sin señal fuera de ese rango)
# alpha:        fijo 0.2                  — eliminado del HPO (sin impacto en resultados)
# dec_ch:       categórico [96,128,192,256] — añadido 256 (16GB VRAM suficiente)
# loss_w_mask:  uniform [0.5, 3.0]
# Con precarga en RAM, los workers no son necesarios.
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

# pin_memory acelera la transferencia CPU→GPU. Activar solo con GPU.
PIN_MEMORY  = (os.getenv("PIN_MEMORY", "1") == "1")

# persistent_workers mantiene los procesos de carga vivos entre epochs.
# Solo tiene efecto si NUM_WORKERS > 0.
PERSISTENT_WORKERS = NUM_WORKERS > 0

# ============================================================
# ACELERACIÓN GPU
# ============================================================

# Precisión mixta float16/float32 (duplica throughput en GPU moderna).
USE_AMP = (os.getenv("USE_AMP", "1") == "1")

# torch.compile: no disponible en Windows (requiere Triton).
# Activar solo en Linux/Mac.
USE_COMPILE = (os.getenv("USE_COMPILE", "0") == "1")

# Gradient clipping: evita explosión de gradientes.
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "1.0"))

# ============================================================
# MODELOS GUARDADOS
# ============================================================
MODELS_DIR = EXPORT_DIR / "models"

# ============================================================
# ENTRENAMIENTO
# ============================================================
BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "64"))
THR_MASK    = float(os.getenv("THR_MASK",  "0.5"))   # Umbral de binarización de la máscara

# ============================================================
# CHECKPOINTING
# ============================================================
# Carpeta exclusiva para nested CV
CHECKPOINT_DIR_NESTED = "checkpoints_nested"

# Carpeta exclusiva para variantes (multitask, seg_only, etc.)
CHECKPOINT_DIR_VARIANTS = "checkpoints_variants"

CHECKPOINT_EVERY = 1
RESUME_FROM = None

# ============================================================
# NESTED CROSS-VALIDATION Y HPO
# ============================================================
SEED_BASE = 42

N_OUTER  = int(os.getenv("N_OUTER",  "10"))   # Folds del loop externo        (prueba: 2)
N_INNER  = int(os.getenv("N_INNER",  "5"))    # Folds del loop interno (HPO)  (prueba: 2)
N_TRIALS = int(os.getenv("N_TRIALS", "50"))   # Trials de Optuna por fold     (prueba: 2)

# Epochs (subir para runs de producción)
MAX_EPOCHS_TRIAL    = int(os.getenv("MAX_EPOCHS_TRIAL",    "15"))  # prueba: 1-3
MAX_EPOCHS_FINAL    = int(os.getenv("MAX_EPOCHS_FINAL",    "100")) # prueba: 5-15
MAX_EPOCHS_ABLATION = int(os.getenv("MAX_EPOCHS_ABLATION", "50"))  # prueba: 5-10

# Validación durante los trials de HPO
TRIAL_VALIDATION_FREQ  = int(os.getenv("TRIAL_VALIDATION_FREQ",  "2"))
TRIAL_PATIENCE         = int(os.getenv("TRIAL_PATIENCE",          "1"))
TRIAL_STEPS_PER_EPOCH  = int(os.getenv("TRIAL_STEPS_PER_EPOCH",   "0"))  # 0 = sin límite
TRIAL_VAL_STEPS        = int(os.getenv("TRIAL_VAL_STEPS",          "0"))  # 0 = sin límite

# ============================================================
# TEST CIEGO Y ABLACIÓN
# ============================================================
N_FINAL_SEEDS        = int(os.getenv("N_FINAL_SEEDS", "30"))
FINAL_SEEDS          = [SEED_BASE + i for i in range(N_FINAL_SEEDS)]

RUN_FINAL_BLIND_TEST = (os.getenv("RUN_FINAL_BLIND_TEST", "1") == "1")
RUN_ABLATIONS        = (os.getenv("RUN_ABLATIONS",        "1") == "1")
RUN_STATS_TESTS      = (os.getenv("RUN_STATS_TESTS",      "1") == "1")

ABLATION_VARIANTS = ["multitask", "cls_only", "seg_only", "unweighted_losses"]

# ============================================================
# ETIQUETA DE EJECUCIÓN (para distinguir múltiples runs)
# ============================================================
_tag      = os.getenv("RUN_TAG", "").strip()
RUN_TAG   = f"_{_tag}" if _tag else ""

# ============================================================
# RUTAS DE EXPORTACIÓN (derivadas)
# ============================================================
TRIALS_CSV     = EXPORT_DIR / f"optuna_trials_nested{RUN_TAG}.csv"
OUTER_CSV      = EXPORT_DIR / f"nested_outer_results{RUN_TAG}.csv"
PARETO_CSV     = EXPORT_DIR / f"pareto_front_trials{RUN_TAG}.csv"
FINAL_TEST_CSV = EXPORT_DIR / f"final_blind_test_multiseed{RUN_TAG}.csv"
STATS_CSV      = EXPORT_DIR / f"stat_tests{RUN_TAG}.csv"
SQLITE_PATH    = EXPORT_DIR / f"optuna_nested_outer{RUN_TAG}.sqlite3"

# ============================================================
# EXPERIMENTO DE ESCALARIZACIÓN CLÁSICA
# ============================================================

# Subcarpeta dedicada dentro de EXPORT_DIR
SCALAR_EXPORT_DIR = EXPORT_DIR / "scalar_experiment"

# Grid de valores de loss_w_mask a explorar.
# Cubre el mismo rango [0.5, 3.0] que el espacio HPO, con paso uniforme.
SCALAR_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Epochs para cada entrenamiento del grid (igual que ablaciones del blind test)
MAX_EPOCHS_SCALAR = int(os.getenv("MAX_EPOCHS_SCALAR", "50"))

# Rutas de exportación del experimento de escalarización
SCALAR_GRID_CSV     = SCALAR_EXPORT_DIR / "scalar_grid_full.csv"
SCALAR_SELECTED_CSV = SCALAR_EXPORT_DIR / "scalar_grid_selected.csv"
SCALAR_STATS_CSV    = SCALAR_EXPORT_DIR / "scalar_stats.csv"
