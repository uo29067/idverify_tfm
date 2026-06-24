"""
main.py — Punto de entrada principal del proyecto DocVerify.

Detecta documentos de identidad falsificados mediante una red neuronal
multi-tarea (clasificación binaria + segmentación de regiones alteradas).

Uso:
    python main.py

Configuración:
    Edita config.py o usa variables de entorno (ver config.py para la lista completa).
    La ruta al dataset se controla con DATASET_ROOT en config.py o con la variable
    de entorno DATASET_ROOT:
        set DATASET_ROOT=C:/Datasets/FantasyID   (Windows)
        export DATASET_ROOT=/data/FantasyID       (Linux/Mac)
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

import config
from dataset import add_json_paths, build_full_doc_df, build_image_dataframe
from train import (
    export_zip,
    get_device,
    print_nested_cv_summary,
    run_blind_test,
    run_nested_cv,
    run_stats_tests,
)


# ============================================================
# 0. VERIFICACIÓN DEL ENTORNO
# ============================================================

def check_environment():
    """Verifica dependencias básicas y disponibilidad de GPU."""
    print(f"[INFO] PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / 1024 ** 3
            print(f"[INFO] ✓ GPU {i}: {props.name} | VRAM: {vram:.1f} GB | "
                  f"CUDA: {torch.version.cuda}")
    else:
        print("[WARN] ✗ No se detectó GPU. Ejecutando en CPU "
              "(el entrenamiento será significativamente más lento).")

    print(f"[INFO] DATASET_ROOT : {config.DATASET_ROOT.resolve()}")
    print(f"[INFO] EXPORT_DIR   : {config.EXPORT_DIR.resolve()}")
    print(f"[INFO] PATCH_SIZE   : {config.PATCH_SIZE}")
    print(f"[INFO] BATCH_SIZE   : {config.BATCH_SIZE}")
    print(f"[INFO] NUM_WORKERS  : {config.NUM_WORKERS}")
    print(f"[INFO] N_OUTER      : {config.N_OUTER}")
    print(f"[INFO] N_INNER      : {config.N_INNER}")
    print(f"[INFO] N_TRIALS     : {config.N_TRIALS}")
    print(f"[INFO] MAX_EPOCHS_TRIAL : {config.MAX_EPOCHS_TRIAL}")
    print(f"[INFO] MAX_EPOCHS_FINAL : {config.MAX_EPOCHS_FINAL}")


# ============================================================
# 1. REPRODUCIBILIDAD BASE
# ============================================================

def set_global_seeds():
    os.environ["PYTHONHASHSEED"] = str(config.SEED_BASE)
    random.seed(config.SEED_BASE)
    np.random.seed(config.SEED_BASE)
    torch.manual_seed(config.SEED_BASE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED_BASE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ============================================================
# 2. CARGA Y PREPARACIÓN DEL DATASET
# ============================================================

def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el dataset desde disco, valida la estructura de directorios,
    empareja imágenes con sus JSON de anotación, realiza el split
    75/10/15 por stem y parsea las máscaras de segmentación.

    Devuelve: (df_dev, df_holdout)
      df_dev     — train + val para Nested CV
      df_holdout — test ciego, nunca visto durante HPO
    """
    root = config.DATASET_ROOT.resolve()

    assert root.exists(), (
        f"\n[ERROR] No se encuentra el dataset en: {root}\n"
        f"  Comprueba que FantasyID está descomprimido en esa ruta\n"
        f"  o cambia DATASET_ROOT en config.py"
    )

    # Indexar imágenes
    df_imgs = build_image_dataframe(root)

    # Emparejar con JSONs de anotación
    df_imgs = add_json_paths(df_imgs)

    # Split 75/10/15 sin data leakage (por stem)
    df_imgs["label"] = (df_imgs["cls_dir"] == "attack").astype(int)
    groups = df_imgs["stem"]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.SEED_BASE)
    idx_rest, idx_test = next(gss1.split(df_imgs, y=df_imgs["label"], groups=groups))

    df_rest      = df_imgs.iloc[idx_rest].reset_index(drop=True)
    df_test_base = df_imgs.iloc[idx_test].reset_index(drop=True)

    val_ratio = 0.10 / 0.85
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=config.SEED_BASE)
    idx_train, idx_val = next(gss2.split(df_rest, y=df_rest["label"], groups=df_rest["stem"]))

    df_train_base = df_rest.iloc[idx_train].reset_index(drop=True)
    df_val_base   = df_rest.iloc[idx_val].reset_index(drop=True)

    print(f"\n[OK] Split 75/10/15 completado (por stem)")
    print(f"  Train: {len(df_train_base)} | Val: {len(df_val_base)} | Test: {len(df_test_base)}")

    # Sanity checks: ningún stem compartido entre splits
    shared_tv = set(df_train_base["stem"]) & set(df_val_base["stem"])
    shared_tt = set(df_train_base["stem"]) & set(df_test_base["stem"])
    shared_vt = set(df_val_base["stem"])   & set(df_test_base["stem"])
    assert not shared_tv, f"Data leakage train/val: {len(shared_tv)} stems compartidos"
    assert not shared_tt, f"Data leakage train/test: {len(shared_tt)} stems compartidos"
    assert not shared_vt, f"Data leakage val/test: {len(shared_vt)} stems compartidos"
    print("  [CHECK] Sin data leakage entre splits ✓")

    # Parsear anotaciones JSON y generar máscaras
    df_train = build_full_doc_df(df_train_base, "train")
    df_val   = build_full_doc_df(df_val_base,   "val")
    df_test  = build_full_doc_df(df_test_base,  "test")

    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        n_attack      = int((d["label"] == 1).sum())
        n_attack_mask = int(((d["label"] == 1) & (d["mask_n_rects"] > 0)).sum())
        print(f"  [{name}] bonafide={int((d['label']==0).sum())} | "
              f"attack={n_attack} | attack con máscara={n_attack_mask}")

    df_dev     = pd.concat([df_train, df_val], ignore_index=True)
    df_holdout = df_test.reset_index(drop=True)

    return df_dev, df_holdout


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print(" DocVerify — Pipeline de Detección de Documentos Falsificados")
    print("=" * 60 + "\n")

    check_environment()
    set_global_seeds()

    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df_dev, df_holdout = load_and_prepare_data()

    outer_rows = run_nested_cv(df_dev)
    print_nested_cv_summary(outer_rows)

    # Selección de HPs finales
    df_outer = pd.DataFrame(outer_rows)
    best_idx = int(df_outer["distance_to_ideal_innercv"].astype(float).idxmin())
    best_hp_row  = df_outer.loc[best_idx]
    final_params = {
        k.replace("hp_", ""): best_hp_row[k]
        for k in df_outer.columns if k.startswith("hp_")
    }
    print(f"\n[FINAL] Hiperparámetros seleccionados (menor distancia inner CV):")
    for k, v in final_params.items():
        print(f"  {k}: {v}")

    if config.RUN_FINAL_BLIND_TEST:
        run_blind_test(df_dev, df_holdout, final_params)

    if config.RUN_STATS_TESTS:
        run_stats_tests()

    export_zip()

    print("\n[OK] Pipeline completado.")
    print(f"[OK] Resultados en: {config.EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
