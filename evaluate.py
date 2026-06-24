"""
evaluate.py — Métricas de evaluación completas para clasificación y segmentación (PyTorch).
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import config


# ============================================================
# THRESHOLD SWEEP
# ============================================================

def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n: int = 501,
) -> tuple[float, float, float, float]:
    """
    Barre n umbrales en [0, 1] y devuelve el umbral óptimo
    para Balanced Accuracy y para F1-macro, junto con sus valores máximos.

    Devuelve: (thr_bacc, best_bacc, thr_f1, best_f1macro)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    ths    = np.linspace(0.001, 1, n)

    baccs = np.full(n, np.nan)
    f1mac = np.full(n, np.nan)

    has_two_classes = np.unique(y_true).size == 2

    for i, t in enumerate(ths):
        yp = (y_prob >= t).astype(int)
        if has_two_classes:
            baccs[i] = balanced_accuracy_score(y_true, yp)
        f1mac[i] = f1_score(y_true, yp, average="macro", zero_division=0)

    thr_bacc = float(ths[np.nanargmax(baccs)]) if np.isfinite(baccs).any() else 0.5
    thr_f1   = float(ths[np.nanargmax(f1mac)]) if np.isfinite(f1mac).any() else 0.5

    return thr_bacc, float(np.nanmax(baccs)), thr_f1, float(np.nanmax(f1mac))


# ============================================================
# EVALUACIÓN COMPLETA
# ============================================================

def eval_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    thr_cls: float,
    device: torch.device,
    thr_mask: float = None,
) -> dict:
    """
    Evalúa el modelo sobre un DataLoader y devuelve métricas completas
    de clasificación y segmentación.
    """
    if thr_mask is None:
        thr_mask = config.THR_MASK

    model.eval()

    y_true_cls, y_prob_cls = [], []
    TP = FP = TN = FN = 0
    sum_dice_pos = 0.0
    cnt_dice_pos = 0

    with torch.no_grad():
        for imgs, labels, masks in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            masks  = masks.to(device)

            out = model(imgs)

            # Clasificación — aplicar sigmoid a logits
            yt = labels.cpu().numpy().reshape(-1).astype(int)
            yp = torch.sigmoid(out["cls"]).cpu().numpy().reshape(-1).astype(float)
            y_true_cls.append(yt)
            y_prob_cls.append(yp)

            # Segmentación
            yt_m = (masks > 0.5)
            yp_m = (out["mask"] > thr_mask)

            yt_f = yt_m.float()
            yp_f = yp_m.float()

            # Dice solo en positivos (reporting)
            inter_s = (yt_f * yp_f).sum(dim=[1, 2, 3])
            denom_s = yt_f.sum(dim=[1, 2, 3]) + yp_f.sum(dim=[1, 2, 3])
            dice_s  = (2.0 * inter_s + 1e-6) / (denom_s + 1e-6)
            pos_s   = yt_f.sum(dim=[1, 2, 3]) > 0

            if pos_s.any():
                sum_dice_pos += dice_s[pos_s].sum().item()
                cnt_dice_pos += pos_s.sum().item()

            # Pixel-level TP/FP/TN/FN
            TP += int(( yt_m &  yp_m).sum().item())
            FP += int((~yt_m &  yp_m).sum().item())
            TN += int((~yt_m & ~yp_m).sum().item())
            FN += int(( yt_m & ~yp_m).sum().item())

    y_true_cls = np.concatenate(y_true_cls)
    y_prob_cls = np.concatenate(y_prob_cls)
    y_pred_cls = (y_prob_cls >= thr_cls).astype(int)

    has_two = np.unique(y_true_cls).size == 2

    pr_auc  = float(average_precision_score(y_true_cls, y_prob_cls)) if y_true_cls.size else np.nan
    roc_auc = float(roc_auc_score(y_true_cls, y_prob_cls))           if has_two        else np.nan
    bacc    = float(balanced_accuracy_score(y_true_cls, y_pred_cls)) if has_two        else np.nan

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="binary", zero_division=0
    )
    f1m = float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    cm  = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])

    # Métricas de segmentación
    dice_global  = (2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6)
    iou_fg       = (TP + 1e-6) / (TP + FP + FN + 1e-6)
    iou_bg       = (TN + 1e-6) / (TN + FP + FN + 1e-6)
    miou         = 0.5 * (iou_fg + iou_bg)
    pix_spec     = TN / (TN + FP + 1e-9)
    pix_prec     = TP / (TP + FP + 1e-9)
    pix_rec      = TP / (TP + FN + 1e-9)
    pix_f1       = 2 * pix_prec * pix_rec / (pix_prec + pix_rec + 1e-9)
    dice_pos_mean = float(sum_dice_pos / cnt_dice_pos) if cnt_dice_pos > 0 else float("nan")

    return {
        "pr_auc":          pr_auc,
        "roc_auc":         roc_auc,
        "bacc":            bacc,
        "f1_1":            float(f1),
        "f1_macro":        f1m,
        "prec1":           float(prec),
        "rec1":            float(rec),
        "cm_TN_FP_FN_TP":  cm.ravel().tolist() if cm.size == 4 else None,
        "dice_global":     float(dice_global),
        "dice_pos_mean":   float(dice_pos_mean),
        "miou":            float(miou),
        "pix_specificity": float(pix_spec),
        "pix_f1":          float(pix_f1),
        "pix_prec":        float(pix_prec),
        "pix_rec":         float(pix_rec),
    }
