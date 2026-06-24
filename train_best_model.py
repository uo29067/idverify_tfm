"""
train_best_model.py — Entrena un único modelo con los mejores HPs del Nested CV.

Propósito: recuperar un modelo .pt cuando solo se tienen los CSVs de resultados
(los .pt no se incluyen en el ZIP de exportación).
Una vez generado el modelo, ejecutar evaluate_sidtd.py para la evaluación
cross-dataset.

Uso:
    python train_best_model.py
    python train_best_model.py --outer-csv ruta/nested_outer_results.csv
    python train_best_model.py --seed 42 --epochs 100
"""

import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

import config
from dataset import build_full_doc_df, build_image_dataframe, add_json_paths, make_dataloader
from model import build_model, build_optimizer
from train import (
    _train_with_early_stopping,
    get_device,
    _set_seeds,
)
import evaluate as ev


# ──────────────────────────────────────────────────────────────────────────────
# SELECCIÓN DE MEJORES HPs
# ──────────────────────────────────────────────────────────────────────────────

def load_best_params(outer_csv_path) -> tuple[dict, float]:
    """
    Lee el nested_outer_results.csv y devuelve los HPs del fold con
    menor distance_to_ideal_innercv (criterio usado en main.py).

    Maneja duplicados de outer_fold (ejecuciones interrumpidas y reanudadas)
    quedándose con la última aparición de cada fold.
    """
    df = pd.read_csv(outer_csv_path)

    # Desduplicar: conservar la última ejecución de cada fold
    df = df.drop_duplicates(subset="outer_fold", keep="last").reset_index(drop=True)

    best_idx = int(df["distance_to_ideal_innercv"].astype(float).idxmin())
    best_row = df.loc[best_idx]

    params = {
        k.replace("hp_", ""): best_row[k]
        for k in df.columns if k.startswith("hp_")
    }
    # Asegurar tipos correctos
    params["dec_ch"]       = int(params["dec_ch"])
    params["lr"]           = float(params["lr"])
    params["weight_decay"] = float(params["weight_decay"])
    params["dropout_rate"] = float(params["dropout_rate"])
    params["loss_w_mask"]  = float(params["loss_w_mask"])
    params.setdefault("alpha", config.LEAKY_RELU_ALPHA)

    thr_cal = float(best_row.get("thr_cls_from_val_sel", 0.5))

    print(f"\n[HPs] Fold seleccionado: outer_fold={int(best_row['outer_fold'])} "
          f"| distance={float(best_row['distance_to_ideal_innercv']):.4f}")
    print(f"[HPs] PR-AUC (outer test FantasyID): "
          f"{float(best_row.get('outer_test_pr_auc', float('nan'))):.4f}")
    for k, v in params.items():
        print(f"  {k}: {v}")
    return params, thr_cal


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS (igual que main.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_fantasyid_dev():
    """
    Carga FantasyID y devuelve df_dev (train+val para entrenar el modelo final).
    Mismo split 75/10/15 que en main.py — sin data leakage.
    """
    import random
    import os

    os.environ["PYTHONHASHSEED"] = str(config.SEED_BASE)
    random.seed(config.SEED_BASE)
    np.random.seed(config.SEED_BASE)
    torch.manual_seed(config.SEED_BASE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED_BASE)

    root = config.DATASET_ROOT.resolve()
    assert root.exists(), (
        f"\n[ERROR] Dataset no encontrado: {root}\n"
        f"  Ajusta DATASET_ROOT en config.py"
    )

    df_imgs = build_image_dataframe(root)
    df_imgs = add_json_paths(df_imgs)
    df_imgs["label"] = (df_imgs["cls_dir"] == "attack").astype(int)
    groups = df_imgs["stem"]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.SEED_BASE)
    idx_rest, _ = next(gss1.split(df_imgs, y=df_imgs["label"], groups=groups))
    df_rest = df_imgs.iloc[idx_rest].reset_index(drop=True)

    val_ratio = 0.10 / 0.85
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=config.SEED_BASE)
    idx_train, idx_val = next(gss2.split(df_rest, y=df_rest["label"], groups=df_rest["stem"]))

    df_train_base = df_rest.iloc[idx_train].reset_index(drop=True)
    df_val_base   = df_rest.iloc[idx_val].reset_index(drop=True)

    df_train = build_full_doc_df(df_train_base, "train")
    df_val   = build_full_doc_df(df_val_base,   "val")

    df_dev = pd.concat([df_train, df_val], ignore_index=True)
    print(f"[OK] df_dev: {len(df_dev)} imágenes "
          f"({int((df_dev.label==1).sum())} attacks | {int((df_dev.label==0).sum())} bonafide)")
    return df_dev


# ──────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO Y GUARDADO
# ──────────────────────────────────────────────────────────────────────────────

def train_and_save(params: dict, thr_cal: float, seed: int, max_epochs: int) -> str:
    """
    Entrena un modelo multitask con los HPs dados sobre el conjunto completo
    de desarrollo de FantasyID y lo guarda en MODELS_DIR.
    Devuelve la ruta del modelo guardado.
    """
    device = get_device()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _set_seeds(seed)

    df_dev = load_fantasyid_dev()

    # Split train2 + selection (para early stopping y umbral)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, sel_idx = next(gss.split(df_dev, y=df_dev["label"], groups=df_dev["stem"]))

    loader_tr  = make_dataloader(
        df_dev.iloc[tr_idx].reset_index(drop=True),
        training=True, seed=seed, device=device
    )
    loader_sel = make_dataloader(
        df_dev.iloc[sel_idx].reset_index(drop=True),
        training=False, seed=seed, device=device
    )

    model     = build_model(params, device)
    optimizer = build_optimizer(model, params)
    scaler    = torch.amp.GradScaler(
        "cuda", enabled=config.USE_AMP and device.type == "cuda"
    )

    print(f"\n[INFO] Entrenando modelo final (seed={seed}, max_epochs={max_epochs})...")
    model = _train_with_early_stopping(
        model, optimizer, scaler,
        loader_tr, loader_sel, device,
        params, max_epochs,
        patience=12,
        variant="multitask",
        desc=f"[best_hp seed={seed}]",
    )

    # Calibración del umbral sobre el set de selección
    import torch.nn as nn
    bce_fn = nn.BCEWithLogitsLoss()
    model.eval()
    y_true_sel, y_prob_sel = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader_sel:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                out = model(imgs)
            y_true_sel.append(labels.cpu().numpy().reshape(-1).astype(int))
            y_prob_sel.append(torch.sigmoid(out["cls"].float()).cpu().numpy().reshape(-1))

    y_true_s = np.concatenate(y_true_sel)
    y_prob_s = np.concatenate(y_prob_sel)
    thr_local, best_bacc, _, _ = ev.threshold_sweep(y_true_s, y_prob_s)
    print(f"[OK] Umbral calibrado localmente: {thr_local:.4f} (BaCC={best_bacc:.4f})")
    print(f"[OK] Umbral del Nested CV (referencia): {thr_cal:.4f}")

    # Guardar
    model_path = config.MODELS_DIR / f"model_best_hp_seed{seed}.pt"
    torch.save({
        "variant":    "multitask",
        "seed":       seed,
        "params":     params,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "thr_local":  thr_local,
        "thr_nested_cv": thr_cal,
    }, model_path)
    print(f"\n[OK] Modelo guardado → {model_path}")

    del model, optimizer, loader_tr, loader_sel
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(model_path)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrena un modelo con los mejores HPs del Nested CV"
    )
    parser.add_argument(
        "--outer-csv", type=str,
        default=None,
        help="Ruta a nested_outer_results.csv. Por defecto usa config.OUTER_CSV.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED_BASE,
        help=f"Semilla de entrenamiento. Por defecto: {config.SEED_BASE}",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.MAX_EPOCHS_FINAL,
        help=f"Máximo de epochs. Por defecto: {config.MAX_EPOCHS_FINAL}",
    )
    args = parser.parse_args()

    outer_csv = args.outer_csv or str(config.OUTER_CSV)

    print("\n" + "=" * 60)
    print(" DocVerify — Entrenamiento modelo final (mejor HP)")
    print("=" * 60)
    print(f"[INFO] Leyendo HPs desde: {outer_csv}")

    params, thr_cal = load_best_params(outer_csv)

    model_path = train_and_save(
        params=params,
        thr_cal=thr_cal,
        seed=args.seed,
        max_epochs=args.epochs,
    )

    print("\n" + "=" * 60)
    print(" SIGUIENTE PASO")
    print("=" * 60)
    print("  Ejecuta la evaluación cross-dataset con:")
    print(f"  python evaluate_sidtd.py --model {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
