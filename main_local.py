"""
main_local.py — Versión reducida para probar el pipeline en local (CPU, sin GPU).

Propósito: verificar que todo el código arranca y fluye correctamente antes
de llevarlo al HPC. Los parámetros son mínimos — NO sirve para producción.

Uso:
    python main_local.py

"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# PERFIL LOCAL — se establecen ANTES de importar config para que surtan efecto.
# Usa setdefault: si la variable ya existe en el entorno, NO se sobreescribe.
# ──────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("N_OUTER",  "2")   # 2 folds externos  (HPC: 10)
os.environ.setdefault("N_INNER",  "2")   # 2 folds internos  (HPC: 5)
os.environ.setdefault("N_TRIALS", "2")   # 2 trials Optuna   (HPC: 50)

os.environ.setdefault("MAX_EPOCHS_TRIAL",    "1")   # (HPC: 15)
os.environ.setdefault("MAX_EPOCHS_FINAL",    "3")   # (HPC: 100)
os.environ.setdefault("MAX_EPOCHS_ABLATION", "2")   # (HPC: 50)
os.environ.setdefault("MAX_EPOCHS_SCALAR",   "2")   # (HPC: 50)

os.environ.setdefault("N_FINAL_SEEDS", "2")   # (HPC: 30)
os.environ.setdefault("BATCH_SIZE",    "8")   # (HPC: 64)

os.environ.setdefault("USE_AMP",     "0")  # Desactivar precisión mixta (sin GPU)
os.environ.setdefault("NUM_WORKERS", "0")  # Sin workers extra

# Carpeta de resultados separada para no mezclar con el run real del HPC
os.environ.setdefault("EXPORT_DIR", "./exports_local_test")

# Solo variante multitask para ahorrar tiempo
os.environ.setdefault("RUN_ABLATIONS",       "0")
os.environ.setdefault("RUN_FINAL_BLIND_TEST","1")
os.environ.setdefault("RUN_STATS_TESTS",     "0")  # Necesita n>=3 para Wilcoxon

# ──────────────────────────────────────────────────────────────────────────────
# A partir de aquí es exactamente igual que main.py
# ──────────────────────────────────────────────────────────────────────────────

from main import main  # noqa: E402  (importar después de fijar env vars)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" MODO LOCAL — parámetros reducidos para smoke test")
    print("=" * 60)
    print("  N_OUTER=2 | N_INNER=2 | N_TRIALS=2")
    print("  MAX_EPOCHS_TRIAL=1 | MAX_EPOCHS_FINAL=3")
    print("  N_FINAL_SEEDS=2 | BATCH_SIZE=8 | AMP=OFF")
    print("  Resultados -> ./exports_local_test/")
    print("=" * 60 + "\n")
    main()
