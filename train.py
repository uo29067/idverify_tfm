"""
train.py — Lógica de entrenamiento: Nested CV, HPO con Optuna,
           test ciego multi-seed, ablación y estadística inferencial (PyTorch).
"""

import gc
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import config
import evaluate as ev
from dataset import VRAMCache, make_dataloader
from model import bce_dice_loss, build_model, build_optimizer

import shutil
import logging

# ============================================================
# UTILIDADES
# ============================================================

def _set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_row_csv(path: Path, row: dict):
    df_row = pd.DataFrame([row])
    df_row.to_csv(path, mode="a", header=not path.exists(), index=False)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_compile(model: nn.Module) -> nn.Module:
    """Aplica torch.compile si está activado y disponible."""
    if not config.USE_COMPILE:
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("  [INFO] torch.compile activado")
        return compiled
    except Exception as e:
        print(f"  [WARN] torch.compile no disponible: {e}")
        return model
  
  
def validate_optuna_study(study):
    """Marca como FAIL cualquier trial que haya quedado en estado RUNNING."""
    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if running:
        for t in running:
            print(f"[WARN] Trial RUNNING detectado: {t.number}. Marcándolo como FAIL.")
        optuna.storages.fail_stale_trials(study)

# ============================================================
# CHECKPOINTING
# ============================================================

def find_latest_checkpoint(desc: str, directory: str):
    """
    Devuelve el checkpoint más reciente que coincide con el 'desc' actual.
    Ejemplo de desc:
        "[outer_fold=3]"
        "[variant=multitask seed=45]"
    """
    if not os.path.exists(directory):
        return None
        
    files = os.listdir(directory)
    ckpts = []

    for f in files:
        # Filtrar solo checkpoints del entrenamiento actual
        if desc in f and f.endswith(".pt") and "epoch" in f:
            try:
                # Extraer número de epoch
                num = int(f.split("epoch")[-1].split(".")[0])
                ckpts.append((num, f))
            except ValueError:
                pass

    if not ckpts:
        return None

    # Ordenar por número de epoch descendente
    ckpts.sort(reverse=True)
    return os.path.join(directory, ckpts[0][1])


def save_checkpoint(path, epoch, model, optimizer, scaler, params):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "params": params,
    }, path)
    print(f"[Checkpoint] Guardado en {path} (epoch {epoch})")


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Cargar modelo
    model.load_state_dict(ckpt["model_state"])

    # Cargar optimizador (si existe)
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        optimizer.zero_grad(set_to_none=True)  # Limpieza

    # Cargar scaler (si existe)
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    print(f"[Checkpoint] Reanudado desde {path} (epoch {ckpt['epoch']})")

    return ckpt["epoch"], ckpt["params"]

def detect_last_completed_outer_fold(checkpoint_dir: Path) -> int:
    """
    Devuelve el último outer_fold que tiene al menos un checkpoint.
    Si no hay ninguno, devuelve 0.
    """
    import re

    pattern = re.compile(r"outer_fold=(\d+)")
    max_fold = 0

    if not os.path.exists(checkpoint_dir):
        return 0

    for fname in os.listdir(checkpoint_dir):
        m = pattern.search(fname)
        if m:
            fold = int(m.group(1))
            max_fold = max(max_fold, fold)

    return max_fold


# ============================================================
# SPLITTERS
# ============================================================

try:
    from sklearn.model_selection import StratifiedGroupKFold

    def _make_sgkf(n_splits: int, seed: int):
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

    if __name__ == "__main__" or not torch.utils.data.get_worker_info():
        print("[INFO] Usando StratifiedGroupKFold")
except ImportError:
    from sklearn.model_selection import GroupKFold

    def _make_sgkf(n_splits: int, seed: int):
        # GroupKFold no soporta shuffle ni seed
        return GroupKFold(n_splits=n_splits)

    if __name__ == "__main__" or not torch.utils.data.get_worker_info():
        print("[INFO] StratifiedGroupKFold no disponible, usando GroupKFold")


# ============================================================
# OPTUNA
# ============================================================

try:
    import optuna
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)


def _make_sampler(seed: int):
    """Sampler robusto: MOTPESampler si está disponible, si no NSGA-II."""
    try:
        return optuna.samplers.MOTPESampler(seed=seed, multivariate=True)
    except Exception:
        return optuna.samplers.NSGAIISampler(seed=seed)


def _get_pareto_trials(study: optuna.Study) -> list:
    """Devuelve los trials del frente de Pareto."""
    if hasattr(study, "get_pareto_front_trials"):
        return study.get_pareto_front_trials()
        
    if hasattr(study, "best_trials"):
        bt = list(study.best_trials)
        if bt:
            return bt
    # Fallback manual
    dirs = getattr(study, "directions", None)
    cand = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]
    
    if dirs is None:
        return cand

    sign = [-1.0 if str(d).lower().endswith("minimize") else 1.0 for d in dirs]

    def vals_max(t):
        return [sign[i] * float(t.values[i]) for i in range(len(sign))]

    def dominates(a, b):
        va, vb = vals_max(a), vals_max(b)
        return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))

    return [
        t for t in cand
        if not any(dominates(u, t) for u in cand if u.number != t.number)
    ]


def _select_best_trial(study: optuna.Study) -> tuple:
    """
    Selecciona el mejor trial del frente de Pareto usando distancia al ideal (1,1).
    """
    pareto = _get_pareto_trials(study)
    if not pareto:
        pareto = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
        ]

    best_t, best_d = None, float("inf")
    
    for t in pareto:
        pr, dc = float(t.values[0]), float(t.values[1])
        dist = math.sqrt((1.0 - pr) ** 2 + (1.0 - dc) ** 2)
        if dist < best_d:
            best_d, best_t = dist, t
    return best_t, best_d


# ============================================================
# LOOP DE ENTRENAMIENTO CON AMP + PROGRESS BAR
# ============================================================

def _train_one_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    scaler:      torch.cuda.amp.GradScaler,
    device:      torch.device,
    lw_cls:      float,
    lw_mask:     float,
    epoch:       int,
    max_epochs:  int,
    desc:        str = "",
) -> float:
    """Entrena una epoch con AMP, gradient clipping y barra de progreso."""
    model.train()
    bce_fn     = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches  = len(loader)

    pbar = tqdm(
        loader,
        desc=f"  Epoch {epoch}/{max_epochs} {desc}",
        leave=False,
        unit="batch"
    )

    for imgs, labels, masks in pbar:
        # Mover datos a GPU si no están ya allí
        if imgs.device != device:
            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
            out       = model(imgs)
            loss_cls  = bce_fn(out["cls"], labels)
            loss_mask = bce_dice_loss(out["mask"], masks)
            loss      = lw_cls * loss_cls + lw_mask * loss_mask

        scaler.scale(loss).backward()

        # Gradient clipping
        if config.GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def _eval_prauc_dice(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluación rápida para HPO: devuelve (PR-AUC, Dice global)."""
    model.eval()
    y_true_cls, y_prob_cls = [], []
    TP = FP = FN = 0

    with torch.no_grad():
        for imgs, labels, masks in loader:
            if imgs.device != device:
                imgs  = imgs.to(device,  non_blocking=True)
                masks = masks.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                out = model(imgs)
                
            # Clasificación
            y_true_cls.append(labels.cpu().numpy().reshape(-1).astype(int))
            y_prob_cls.append(out["cls"].float().cpu().numpy().reshape(-1).astype(float))

            # Segmentación
            yt_m = (masks > 0.5)
            yp_m = (out["mask"].float() > config.THR_MASK)
            
            TP += int(( yt_m &  yp_m).sum().item())
            FP += int((~yt_m &  yp_m).sum().item())
            FN += int(( yt_m & ~yp_m).sum().item())

    y_true = np.concatenate(y_true_cls)
    y_prob = np.concatenate(y_prob_cls)
    
    # Tratar NaN
    nan_count = int(np.isnan(y_prob).sum())
    if nan_count > 0:
        logging.warning(f"NaN en y_prob: {nan_count}/{len(y_prob)} valores")
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    
    pr_auc = float(average_precision_score(y_true, y_prob)) if y_true.size else 0.0
    dice   = float((2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6))
    
    return pr_auc, dice


# ============================================================
# OBJETIVO OPTUNA (inner CV con VRAMCache)
# ============================================================

def _make_inner_objective(
    df_outer_train: pd.DataFrame,
    outer_fold_id: int,
    device: torch.device,
    cache: VRAMCache,          #cache ya cargada en VRAM
):
    """
    Devuelve la función objetivo para Optuna.
    Realiza un inner CV completo usando VRAMCache (sin IO).
    """
    splitter = _make_sgkf(
        config.N_INNER,
        seed=config.SEED_BASE + 100 + outer_fold_id
    )
    
    y = df_outer_train["label"].values
    g = df_outer_train["stem"].values

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        t0 = time.perf_counter()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # ------------------------------------------------------------
        # Hiperparámetros sugeridos por Optuna
        # ------------------------------------------------------------
        params = {
            "lr":           trial.suggest_float("lr", 5e-5, 9e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4),
            "alpha":        0.2,   # fijo — sin señal en HPO anterior
            "dec_ch":       trial.suggest_categorical("dec_ch", [96, 128, 192, 256]),
            "loss_w_mask":  trial.suggest_float("loss_w_mask", 0.5, 3.0),
        }

        fold_metrics = []

        try:
            # --------------------------------------------------------
            # Inner CV
            # --------------------------------------------------------
            for inner_id, (tr_idx, va_idx) in enumerate(
                splitter.split(df_outer_train, y=y, groups=g)
            ):
                seed_fold = (
                    config.SEED_BASE
                    + 1000 * outer_fold_id
                    + 10 * inner_id
                )
                _set_seeds(seed_fold)

                # Loaders instantáneos sobre la cache en VRAMCache
                loader_tr = cache.make_loader(
                    tr_idx, training=True,  seed=seed_fold
                )
                loader_va = cache.make_loader(
                    va_idx, training=False, seed=seed_fold
                )

                # Modelo + optimizador
                model     = _maybe_compile(build_model(params, device))
                optimizer = build_optimizer(model, params)
                scaler    = torch.amp.GradScaler(
                    "cuda",
                    enabled=config.USE_AMP and device.type == "cuda"
                )
                
                # ----------------------------------------------------
                # Entrenamiento del trial
                # ----------------------------------------------------
                for epoch in range(1, config.MAX_EPOCHS_TRIAL + 1):
                    _train_one_epoch(
                        model, loader_tr, optimizer, scaler, device,
                        lw_cls=1.0,
                        lw_mask=float(params["loss_w_mask"]),
                        epoch=epoch,
                        max_epochs=config.MAX_EPOCHS_TRIAL,
                        desc=f"[trial={trial.number} inner={inner_id}]",
                    )

                #----------------------------------------------------
                # Evaluación en inner-val
                # ----------------------------------------------------
                pr_auc, dice = _eval_prauc_dice(model, loader_va, device)
                fold_metrics.append({"prauc": pr_auc, "dice": dice})

                # Limpieza
                del model, optimizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            _append_row_csv(config.TRIALS_CSV, {
                "trial_number": trial.number,
                "outer_fold": outer_fold_id,
                "pruned": True,
                "time_sec": time.perf_counter() - t0,
                **params,
                "notes": "OOM",
            })
            raise optuna.exceptions.TrialPruned("CUDA OOM")
            
        # ------------------------------------------------------------
        # Métricas finales del trial
        # ------------------------------------------------------------
        pr_mean   = float(np.nanmean([m["prauc"] for m in fold_metrics]))
        dc_mean   = float(np.nanmean([m["dice"]  for m in fold_metrics]))
        dist_mean = math.sqrt((1.0 - pr_mean) ** 2 + (1.0 - dc_mean) ** 2)
        
        # Registrar en CSV
        _append_row_csv(config.TRIALS_CSV, {
            "trial_number":         trial.number,
            "outer_fold":           outer_fold_id,
            "pruned":               False,
            "time_sec":             time.perf_counter() - t0,
            **params,
            "val_cls_prauc":        pr_mean,
            "val_mask_dice_global": dc_mean,
            "distance_to_ideal":    dist_mean,
        })

        return pr_mean, dc_mean

    return objective


# ============================================================
# ENTRENAMIENTO FINAL CON EARLY STOPPING + CHECKPOINTING
# ============================================================

def _train_with_early_stopping(
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    scaler:     torch.cuda.amp.GradScaler,
    loader_tr:  torch.utils.data.DataLoader,
    loader_va:  torch.utils.data.DataLoader,
    device:     torch.device,
    params:     dict,
    max_epochs: int,
    patience:   int = 8,
    variant:    str = "multitask",
    desc:       str = "",
) -> nn.Module:
    """Entrena con early stopping, AMP y barra de progreso por epoch."""
    best_dist  = float("inf")
    best_state = None
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    # ------------------------------------------------------------
    # Pesos de pérdidas según variante
    # ------------------------------------------------------------
    lw_mask = {
        "multitask":         float(params.get("loss_w_mask", 1.0)),
        "cls_only":          0.0,
        "seg_only":          float(params.get("loss_w_mask", 1.0)),
        "unweighted_losses": 1.0,
    }.get(variant, float(params.get("loss_w_mask", 1.0)))

    lw_cls = 0.0 if variant == "seg_only" else 1.0
    
    # ------------------------------------------------------------
    # Selección automática de carpeta de checkpoints
    # ------------------------------------------------------------
    if "outer_fold=" in desc:
        checkpoint_dir = config.CHECKPOINT_DIR_NESTED
    else:
        checkpoint_dir = config.CHECKPOINT_DIR_VARIANTS

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Reanudación desde checkpoint (si existe)
    # ------------------------------------------------------------
    ckpt_path = find_latest_checkpoint(desc, checkpoint_dir)
    print("DEBUG — Último checkpoint detectado:", ckpt_path)

    if ckpt_path is not None:
        print(f"[Checkpoint] Detectado último checkpoint: {ckpt_path}")
        start_epoch, params = load_checkpoint(
            ckpt_path, model, optimizer, scaler, device
        )
        start_epoch += 1
        print(f"[Checkpoint] Continuando desde epoch {start_epoch}")
    else:
        print("[Checkpoint] No se encontró ningún checkpoint. Entrenamiento desde cero.")
        start_epoch = 1

    # ------------------------------------------------------------
    # Bucle principal de entrenamiento
    # ------------------------------------------------------------
    outer_pbar = tqdm(
        range(start_epoch, max_epochs + 1),
        desc=f"  Entrenando {desc}",
        unit="epoch",
        leave=True,
    )


    for epoch in outer_pbar:
        #Entrenamiento
        _train_one_epoch(
            model, loader_tr, optimizer, scaler, device,
            lw_cls=lw_cls, lw_mask=lw_mask,
            epoch=epoch, max_epochs=max_epochs, desc=desc,
        )
        
        #Validación
        pr_auc, dice = _eval_prauc_dice(model, loader_va, device)
        
        #Métrica para early stopping
        if variant == "cls_only":
            monitor = 1.0 - pr_auc
        elif variant == "seg_only":
            monitor = 1.0 - dice
        else:
            monitor = math.sqrt((1.0 - pr_auc) ** 2 + (1.0 - dice) ** 2)

        scheduler.step(monitor)

        outer_pbar.set_postfix(
            pr_auc=f"{pr_auc:.4f}",
            dice=f"{dice:.4f}",
            dist=f"{monitor:.4f}",
            best=f"{best_dist:.4f}",
            patience=f"{patience_counter}/{patience}",
        )
        
        #Early stopping
        if monitor < best_dist:
            best_dist = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [EarlyStopping] Epoch {epoch} — sin mejora en {patience} epochs")
                break
        
        # Guardar Checkpoint cada X epochs (rotativo: conservar solo los 2 más recientes)
        if config.CHECKPOINT_EVERY > 0 and epoch % config.CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_{desc}_epoch{epoch}.pt",
            )
            save_checkpoint(
                path=ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                params=params,
            )
            # Eliminar checkpoints de este entrenamiento salvo los 2 más recientes
            _all_ckpts = []
            for _f in os.listdir(checkpoint_dir):
                if desc in _f and _f.endswith(".pt") and "epoch" in _f:
                    try:
                        _ep = int(_f.split("epoch")[-1].split(".")[0])
                        _all_ckpts.append((_ep, _f))
                    except ValueError:
                        pass
            _all_ckpts.sort(reverse=True)
            for _, _old in _all_ckpts[2:]:
                _old_path = os.path.join(checkpoint_dir, _old)
                try:
                    os.remove(_old_path)
                except OSError:
                    pass
    # Cargar mejor estado encontrado
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model


# ============================================================
# NESTED CV COMPLETO
# ============================================================

def run_nested_cv(df_dev: pd.DataFrame) -> list[dict]:
    device = get_device()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    outer_splitter = _make_sgkf(config.N_OUTER, seed=config.SEED_BASE + 7)
    y = df_dev["label"].values
    g = df_dev["stem"].values
    outer_rows = []

    # Reanudación automática del nested CV
    # Prioridad 1 — modelo guardado: si model_outerN.pt existe y hay fila en OUTER_CSV,
    #   el fold está completo y se salta directamente.
    # Prioridad 2 — checkpoint de época: si el fold empezó pero no terminó (sin modelo),
    #   find_latest_checkpoint encontrará el checkpoint más reciente y reanudará desde ahí.
    last_fold = detect_last_completed_outer_fold(config.CHECKPOINT_DIR_NESTED)
    if last_fold > 0:
        print(f"[INFO] Checkpoint de época detectado hasta outer_fold={last_fold}")
        # Limpiar solo el checkpoint del fold incompleto (el último detectado)
        if os.path.exists(config.CHECKPOINT_DIR_NESTED):
            for fname in os.listdir(config.CHECKPOINT_DIR_NESTED):
                if f"outer_fold={last_fold}" in fname:
                    path = os.path.join(config.CHECKPOINT_DIR_NESTED, fname)
                    os.remove(path)
                    print(f"[INFO] Borrado checkpoint parcial: {path}")

    # Precomputar todos los splits
    all_splits = list(outer_splitter.split(df_dev, y=y, groups=g))

    # ------------------------------------------------------------
    # Iterar outer folds (siempre desde 1; skip si ya completado)
    # ------------------------------------------------------------
    for outer_fold in range(1, config.N_OUTER + 1):
        # Saltar folds con modelo guardado + fila en OUTER_CSV
        _model_done = config.MODELS_DIR / f"model_outer{outer_fold}{config.RUN_TAG}.pt"
        if _model_done.exists() and config.OUTER_CSV.exists():
            _df_done = pd.read_csv(config.OUTER_CSV)
            _done_rows = _df_done[_df_done["outer_fold"] == outer_fold]
            if not _done_rows.empty:
                outer_rows.append(_done_rows.iloc[0].to_dict())
                print(f"[Resume] outer_fold={outer_fold} ya completado — saltando")
                continue

        train_idx, test_idx = all_splits[outer_fold - 1]
        
        print(f"\n{'='*60}")
        print(f" OUTER FOLD {outer_fold}/{config.N_OUTER}")
        print(f"{'='*60}")

        df_outer_train = df_dev.iloc[train_idx].reset_index(drop=True)
        df_outer_test  = df_dev.iloc[test_idx].reset_index(drop=True)

        # --------------------------------------------------------
        # VRAMCache para HPO
        # --------------------------------------------------------
        cache_train = VRAMCache(
            df_outer_train, device,
            label=f"outer_fold={outer_fold} train ({len(df_outer_train)} imgs)",
        )
        
        # --------------------------------------------------------
        # INNER CV (Optuna)
        # --------------------------------------------------------
        study_name = f"DOCVERIFY{config.RUN_TAG}_OUTER{outer_fold}"
        storage    = f"sqlite:///{config.SQLITE_PATH}?timeout=300&journal_mode=WAL"

        study = optuna.create_study(
            study_name     = study_name,
            directions     = ["maximize", "maximize"],
            sampler        = _make_sampler(config.SEED_BASE + outer_fold),
            pruner         = optuna.pruners.NopPruner(),
            storage        = storage,
            load_if_exists = True,
        )
        
        validate_optuna_study(study)
        objective = _make_inner_objective(
            df_outer_train, outer_fold, device, cache_train
        )

        t_hpo0 = time.perf_counter()
        study.optimize(objective, n_trials=config.N_TRIALS, gc_after_trial=True)
        t_hpo1 = time.perf_counter()
        
        # --------------------------------------------------------
        # Selección del mejor trial
        # --------------------------------------------------------
        best_trial, best_dist = _select_best_trial(study)
        if best_trial is None:
            raise RuntimeError(f"No hay trials completados en outer fold {outer_fold}.")

        pareto = _get_pareto_trials(study)
        print(f"[HPO] Tiempo: {t_hpo1 - t_hpo0:.1f}s | Pareto: {len(pareto)} trials")
        print(f"[HPO] Trial #{best_trial.number} | dist={best_dist:.4f} | "
              f"(PR-AUC, Dice)={best_trial.values}")

        # Liberar cache del HPO — el entrenamiento final crea sus propios loaders
        cache_train.free()
        del cache_train

        # Guardar Pareto
        for t in pareto:
            pr, dc = (t.values or [np.nan, np.nan])
            _append_row_csv(config.PARETO_CSV, {
                "outer_fold": outer_fold, "trial_number": t.number,
                "val_cls_prauc": float(pr), "val_mask_dice_global": float(dc),
                "distance_to_ideal": math.sqrt((1 - float(pr))**2 + (1 - float(dc))**2),
                **t.params,
            })

        # Entrenamiento final del fold externo
        gss = GroupShuffleSplit(
            n_splits=1, test_size=0.15,
            random_state=config.SEED_BASE + outer_fold
        )
        tr2_idx, sel_idx = next(gss.split(
            df_outer_train,
            y=df_outer_train["label"],
            groups=df_outer_train["stem"]
        ))

        seed_final = config.SEED_BASE + 5000 + outer_fold
        _set_seeds(seed_final)

        loader_tr   = make_dataloader(
            df_outer_train.iloc[tr2_idx].reset_index(drop=True),
            training=True,  seed=seed_final, device=device
        )
        loader_sel  = make_dataloader(
            df_outer_train.iloc[sel_idx].reset_index(drop=True),
            training=False, seed=seed_final, device=device
        )
        loader_test = make_dataloader(
            df_outer_test,
            training=False, seed=seed_final, device=device
        )

        params    = dict(best_trial.params)
        model     = _maybe_compile(build_model(params, device))
        optimizer = build_optimizer(model, params)
        scaler    = torch.amp.GradScaler(
            "cuda",
            enabled=config.USE_AMP and device.type == "cuda"
        )

        t0 = time.perf_counter()
        model = _train_with_early_stopping(
            model, optimizer, scaler,
            loader_tr, loader_sel, device,
            params, config.MAX_EPOCHS_FINAL,
            patience=12,
            desc=f"[outer_fold={outer_fold}]",
        )
        train_time = time.perf_counter() - t0

        # Guardar modelo final del fold
        model_path = config.MODELS_DIR / f"model_outer{outer_fold}{config.RUN_TAG}.pt"
        torch.save({
            "outer_fold":  outer_fold,
            "params":      params,
            "state_dict":  {k: v.cpu() for k, v in model.state_dict().items()},
        }, model_path)
        print(f"  [OK] Modelo guardado: {model_path}")

        # Umbral desde loader_sel
        y_true_sel, y_prob_sel = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels, _ in loader_sel:
                if imgs.device != device:
                    imgs = imgs.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                    out = model(imgs)
                y_true_sel.append(labels.cpu().numpy().reshape(-1).astype(int))
                y_prob_sel.append(out["cls"].float().cpu().numpy().reshape(-1))

        thr_bacc, best_bacc, thr_f1, best_f1m = ev.threshold_sweep(
            np.concatenate(y_true_sel),
            torch.sigmoid(torch.tensor(np.concatenate(y_prob_sel))).numpy(),
        )
        
        # Evaluación final en outer-test
        met = ev.eval_model(model, loader_test, thr_cls=thr_bacc, device=device)

        row = {
            "outer_fold":                outer_fold,
            "selected_trial_number":     best_trial.number,
            "distance_to_ideal_innercv": float(best_dist),
            "hpo_time_sec":              float(t_hpo1 - t_hpo0),
            "train_time_sec":            float(train_time),
            "thr_cls_from_val_sel":      float(thr_bacc),
            "val_sel_best_bacc":         float(best_bacc),
            "val_sel_best_f1m":          float(best_f1m),
            **{f"outer_test_{k}": v for k, v in met.items()},
            **{f"hp_{k}": v for k, v in params.items()},
        }
        _append_row_csv(config.OUTER_CSV, row)
        outer_rows.append(row)

        #limpiar
        del model, optimizer, loader_tr, loader_sel, loader_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outer_rows


# ============================================================
# RESUMEN NESTED CV
# ============================================================

def print_nested_cv_summary(outer_rows: list[dict]):
    df = pd.DataFrame(outer_rows)
    print(f"\n{'='*60}")
    print(" RESUMEN NESTED CV (outer-test)")
    print(f"{'='*60}")
    for k in [
        "outer_test_pr_auc",
        "outer_test_dice_global",
        "outer_test_miou",
        "outer_test_pix_specificity",
        "outer_test_bacc",
        "outer_test_f1_macro"
    ]:
        if k in df.columns:
            m = float(df[k].mean())
            s = float(df[k].std(ddof=1)) if len(df) > 1 else 0.0
            print(f"  {k}: {m:.4f} ± {s:.4f}")


# ============================================================
# ENTRENAMIENTO FINAL (un modelo)
# ============================================================

def _train_final_model(
    params:     dict,
    seed:       int,
    variant:    str,
    df_dev:     pd.DataFrame,
    df_holdout: pd.DataFrame,
    device:     torch.device,
) -> dict:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _set_seeds(seed)
    # ------------------------------------------------------------
    # Split train → train2 + selection
    # ------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, sel_idx = next(gss.split(df_dev, y=df_dev["label"], groups=df_dev["stem"]))

    loader_tr   = make_dataloader(
        df_dev.iloc[tr_idx].reset_index(drop=True),
        training=True,  seed=seed, device=device
    )
    loader_sel  = make_dataloader(
        df_dev.iloc[sel_idx].reset_index(drop=True),
        training=False, seed=seed, device=device
    )
    loader_test = make_dataloader(
        df_holdout,
        training=False, seed=seed, device=device
    )
    
    # ------------------------------------------------------------
    # Modelo + optimizador
    # ------------------------------------------------------------
    model     = _maybe_compile(build_model(params, device))
    optimizer = build_optimizer(model, params)
    scaler    = torch.amp.GradScaler(
        "cuda",
        enabled=config.USE_AMP and device.type == "cuda"
    )
    
    # ------------------------------------------------------------
    # Número de epochs según variante
    # ------------------------------------------------------------
    epochs = (
        config.MAX_EPOCHS_FINAL
        if variant == "multitask"
        else min(config.MAX_EPOCHS_ABLATION, config.MAX_EPOCHS_FINAL)
    )
    
    # ------------------------------------------------------------
    # Entrenamiento final con early stopping + checkpointing
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    model = _train_with_early_stopping(
        model, optimizer, scaler,
        loader_tr, loader_sel, device,
        params, epochs,
        patience=12,
        variant=variant,
        desc=f"[variant={variant} seed={seed}]",
    )
    train_time = time.perf_counter() - t0

    # Guardar modelo final
    model_path = config.MODELS_DIR / f"model_{variant}_seed{seed}{config.RUN_TAG}.pt"
    torch.save({
        "variant": variant,
        "seed": seed,
        "params": params,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }, model_path)

    # ------------------------------------------------------------
    # Umbral desde loader_sel (solo si hay clasificación)
    # ------------------------------------------------------------
    if variant != "seg_only":
        y_true_sel, y_prob_sel = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels, _ in loader_sel:
                if imgs.device != device:
                    imgs = imgs.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                    out = model(imgs)
                y_true_sel.append(labels.cpu().numpy().reshape(-1).astype(int))
                y_prob_sel.append(out["cls"].float().cpu().numpy().reshape(-1))
                
        thr_bacc, best_bacc, thr_f1, best_f1m = ev.threshold_sweep(
            np.concatenate(y_true_sel),
            torch.sigmoid(torch.tensor(np.concatenate(y_prob_sel))).numpy(),
        )
    else:
        thr_bacc, best_bacc, thr_f1, best_f1m = 0.5, float("nan"), 0.5, float("nan")
        
    # ------------------------------------------------------------
    # Evaluación final en test
    # ------------------------------------------------------------
    met = ev.eval_model(model, loader_test, thr_cls=thr_bacc, device=device)

    # ------------------------------------------------------------
    # Registrar resultados
    # ------------------------------------------------------------
    row = {
        "seed": seed,
        "variant": variant,
        "train_time_sec": float(train_time),
        "epochs_budget": epochs,
        "thr_cls_from_val_sel": float(thr_bacc),
        "val_sel_best_bacc": float(best_bacc),
        "val_sel_best_f1m": float(best_f1m),
        **{f"test_{k}": v for k, v in met.items()},
        **{f"hp_{k}": v for k, v in params.items()},
    }
    row["test_cm_TN_FP_FN_TP"] = json.dumps(met["cm_TN_FP_FN_TP"])
    _append_row_csv(config.FINAL_TEST_CSV, row)

    #limpiar
    del model, optimizer, loader_tr, loader_sel, loader_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# ============================================================
# TEST CIEGO MULTI-SEED + ABLACIÓN (multi-seed, multi-variant)
# ============================================================

def run_blind_test(df_dev: pd.DataFrame, df_holdout: pd.DataFrame, final_params: dict):
    device   = get_device()
    variants = config.ABLATION_VARIANTS if config.RUN_ABLATIONS else ["multitask"]

    print(f"\n{'='*60}")
    print(" TEST CIEGO (HOLDOUT)")
    print(f"{'='*60}")
    print(f"  Holdout size : {len(df_holdout)}")
    print(f"  Seeds        : {config.FINAL_SEEDS}")
    print(f"  Variantes    : {variants}")

    for variant in variants:
        for seed in config.FINAL_SEEDS:
            print(f"\n  [FINAL] variant={variant} seed={seed}")
            _train_final_model(
                final_params,
                seed=seed,
                variant=variant,
                df_dev=df_dev,
                df_holdout=df_holdout,
                device=device,
            )

    df_final = pd.read_csv(config.FINAL_TEST_CSV)
    _print_final_summary(df_final)


def _print_final_summary(df_final: pd.DataFrame):
    print(f"\n{'='*60}")
    print(" RESUMEN TEST CIEGO (HOLDOUT)")
    print(f"{'='*60}")
    for variant in df_final["variant"].unique():
        d = df_final[df_final["variant"] == variant]
        
        def ms(col):
            vals = d[col].dropna()
            m = float(vals.mean())
            s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            return f"{m:.4f}±{s:.4f}"
            
        print(
            f"  [{variant}] PR-AUC={ms('test_pr_auc')} | "
            f"Dice(global)={ms('test_dice_global')} | "
            f"Dice(+)={ms('test_dice_pos_mean')} | "
            f"mIoU={ms('test_miou')}"
        )


# ============================================================
# ESTADÍSTICA INFERENCIAL
# ============================================================

def run_stats_tests():
    if not config.FINAL_TEST_CSV.exists():
        print("[STATS] No se encuentra el CSV de resultados. Saltando.")
        return

    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
        from scipy.stats import ttest_rel, wilcoxon

    df_final = pd.read_csv(config.FINAL_TEST_CSV)
    BASE = "multitask"
    METRICS = [
        "test_pr_auc", "test_dice_global", "test_dice_pos_mean",
        "test_miou", "test_pix_specificity", "test_pix_f1",
        "test_bacc", "test_f1_macro",
    ]

    def cohens_d(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))

    def holm_bonferroni(pvals, alpha=0.05):
        pvals  = np.asarray(pvals, dtype=float)
        m      = len(pvals)
        order  = np.argsort(pvals)
        adj    = np.empty_like(pvals)
        reject = np.zeros(m, dtype=bool)
        for i, idx in enumerate(order):
            adj[idx] = min((m - i) * pvals[idx], 1.0)
        for i, idx in enumerate(order):
            if pvals[idx] <= alpha / (m - i):
                reject[idx] = True
            else:
                break
        return adj.tolist(), reject.tolist()

    stats_rows = []
    for comp in [v for v in df_final["variant"].unique() if v != BASE]:
        dfA = df_final[df_final["variant"] == BASE]
        dfB = df_final[df_final["variant"] == comp]
        mrg = dfA.merge(dfB, on="seed", suffixes=("_A", "_B"))

        if len(mrg) < 3:
            print(f"[STATS] n insuficiente para {BASE} vs {comp}. Saltando.")
            continue

        pvals_w, pvals_t, tmp = [], [], []
        for met in METRICS:
            a = mrg[f"{met}_A"].values
            b = mrg[f"{met}_B"].values
            try:
                p_w = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            except Exception:
                p_w = float("nan")
            try:
                p_t = float(ttest_rel(a, b).pvalue)
            except Exception:
                p_t = float("nan")

            pvals_w.append(p_w)
            pvals_t.append(p_t)
            tmp.append({
                "comparison": f"{BASE} vs {comp}", "metric": met,
                "n_paired": len(mrg),
                "mean_A": float(np.mean(a)), "std_A": float(np.std(a, ddof=1)),
                "mean_B": float(np.mean(b)), "std_B": float(np.std(b, ddof=1)),
                "wilcoxon_p": p_w, "ttest_p": p_t,
                "cohens_d_paired": cohens_d(a, b),
            })

        adj_w, rej_w = holm_bonferroni(pvals_w)
        adj_t, rej_t = holm_bonferroni(pvals_t)

        for i, row in enumerate(tmp):
            row.update({
                "wilcoxon_p_holm":      float(adj_w[i]),
                "wilcoxon_reject_holm": bool(rej_w[i]),
                "ttest_p_holm":         float(adj_t[i]),
                "ttest_reject_holm":    bool(rej_t[i]),
            })
            stats_rows.append(row)

    if stats_rows:
        df_stats = pd.DataFrame(stats_rows)
        df_stats.to_csv(config.STATS_CSV, index=False)
        print(f"\n[OK] CSV estadística: {config.STATS_CSV}")
        print(f"\n{'='*60}")
        print(" RESUMEN ESTADÍSTICO (Wilcoxon + Holm)")
        print(f"{'='*60}")
        for _, r in df_stats.sort_values(["comparison", "wilcoxon_p_holm"]).iterrows():
            print(f"  {r['comparison']} | {r['metric']} | "
                  f"p_holm={r['wilcoxon_p_holm']:.4g} | "
                  f"d={r['cohens_d_paired']:.3f} | "
                  f"reject={r['wilcoxon_reject_holm']}")


# ============================================================
# EXPORTACIÓN ZIP
# ============================================================

def export_zip():
    ts       = time.strftime("%Y%m%d_%H%M%S")
    zip_path = config.EXPORT_DIR.parent / f"{config.EXPORT_DIR.name}_{ts}.zip"

    n = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in config.EXPORT_DIR.rglob("*"):
            if p.is_file() and p.suffix != ".pt":  # Excluir modelos del ZIP (son grandes)
                zf.write(p, arcname=p.relative_to(config.EXPORT_DIR.parent))
                n += 1

    print(f"[OK] ZIP exportado: {zip_path} ({n} archivos, modelos .pt excluidos)")
    return zip_path
