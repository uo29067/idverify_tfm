"""
evaluate_sidtd.py — Evaluación cross-dataset de DocVerify sobre SIDTD.

Carga los modelos entrenados en FantasyID y los evalúa sobre SIDTD sin
reentrenamiento, midiendo la capacidad de generalización cross-dataset.

Nota técnica de formato:
  - FantasyID: (x, y) en las anotaciones es la esquina SUPERIOR-IZQUIERDA
    del rectángulo (formato VIA estándar).
  - SIDTD/MIDV-2020: (x, y) es la esquina INFERIOR-DERECHA del rectángulo.
    Conversión aplicada: x_tl = max(0, x - width), y_tl = max(0, y - height).

Uso:
    python evaluate_sidtd.py
    python evaluate_sidtd.py --model exports/models/model_outer1.pt
    python evaluate_sidtd.py --thr 0.45 --sidtd ./Dataset_SIDTD
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

import config
import evaluate as ev
from dataset import make_dataloader
from model import build_model
from train import get_device

# ──────────────────────────────────────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────────────────────────────────────

# Se intenta el directorio real del repo antes de caer al valor de config.
_DEFAULT_SIDTD = (
    Path("./Dataset_SIDTD")
    if Path("./Dataset_SIDTD").exists()
    else config.SIDTD_ROOT
)

SIDTD_CROSS_CSV          = config.EXPORT_DIR / "sidtd_cross_dataset.csv"
SIDTD_CROSS_PER_MODEL_CSV = config.EXPORT_DIR / "sidtd_cross_per_model.csv"

# ──────────────────────────────────────────────────────────────────────────────
# PARSEO DE ANOTACIONES VIA (SIDTD / MIDV-2020)
# ──────────────────────────────────────────────────────────────────────────────

_via_cache: dict = {}


def _load_via_annotations(via_path: Path) -> dict:
    """
    Parsea un JSON de anotaciones VIA de SIDTD y devuelve un dict:
      { via_filename : { field_name : {"x": x_tl, "y": y_tl, "w": w, "h": h} } }

    Las coordenadas originales de SIDTD usan la esquina inferior-derecha como
    punto de referencia del rectángulo.  Se convierten aquí a esquina
    superior-izquierda para ser compatibles con _mask_from_rects de dataset.py:
        x_tl = max(0, x - width)
        y_tl = max(0, y - height)
    """
    if via_path in _via_cache:
        return _via_cache[via_path]

    with open(via_path, encoding="utf-8") as fh:
        data = json.load(fh)

    result: dict = {}
    for entry in data.get("_via_img_metadata", {}).values():
        fname = entry.get("filename", "")
        field_map: dict = {}
        for region in entry.get("regions", []):
            sa = region.get("shape_attributes", {})
            ra = region.get("region_attributes", {})
            if sa.get("name") != "rect":
                continue
            x = int(sa.get("x", 0))
            y = int(sa.get("y", 0))
            w = int(sa.get("width", 0))
            h = int(sa.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            fn = str(ra.get("field_name", "")).strip().casefold()
            if not fn:
                continue
            # Conversión esquina inferior-derecha → superior-izquierda
            x_tl = max(0, x - w)
            y_tl = max(0, y - h)
            field_map[fn] = {"x": x_tl, "y": y_tl, "w": w, "h": h}
        result[fname] = field_map

    _via_cache[via_path] = result
    return result


def _resolve_mask_rects(fake_meta: dict, reals_ann_dir: Path) -> list:
    """
    Dado el JSON de metadatos de una imagen falsa de SIDTD, devuelve la lista
    de rectángulos alterados {x, y, w, h} buscando en el VIA del real fuente.

    Procesa tanto 'src'/'field' como 'second_src'/'second_field'.
    """
    rects = []
    for src_key, field_key in [("src", "field"), ("second_src", "second_field")]:
        src   = fake_meta.get(src_key,   "None")
        field = fake_meta.get(field_key, "None")
        if not src or src == "None" or not field or field == "None":
            continue

        src_stem  = Path(src).stem                         # e.g. "alb_id_00"
        doc_type  = re.sub(r"_\d+$", "", src_stem)         # e.g. "alb_id"
        # La filename en el VIA es solo la parte numérica + extensión
        via_fname = re.sub(r"^.*_(\d+)$", r"\1.jpg", src_stem)  # e.g. "00.jpg"

        via_path = reals_ann_dir / f"{doc_type}.json"
        if not via_path.exists():
            warnings.warn(f"[SIDTD] Anotación VIA no encontrada: {via_path}", stacklevel=2)
            continue

        via_data  = _load_via_annotations(via_path)
        field_map = via_data.get(via_fname, {})
        field_norm = field.strip().casefold()

        rect = field_map.get(field_norm)
        if rect:
            rects.append(rect)
        else:
            warnings.warn(
                f"[SIDTD] Campo '{field_norm}' no encontrado en "
                f"{doc_type}.json[{via_fname}]. "
                f"Disponibles: {list(field_map.keys())}",
                stacklevel=2,
            )
    return rects


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATAFRAME SIDTD
# ──────────────────────────────────────────────────────────────────────────────

def build_sidtd_df(sidtd_root: Path) -> pd.DataFrame:
    """
    Indexa todas las imágenes SIDTD y resuelve las máscaras de anotación.
    El DataFrame resultante es compatible con make_dataloader / VRAMCache.
    """
    ann_fakes_dir = sidtd_root / "templates" / "Annotations" / "fakes"
    ann_reals_dir = sidtd_root / "templates" / "Annotations" / "reals"
    img_fakes_dir = sidtd_root / "templates" / "Images" / "fakes"
    img_reals_dir = sidtd_root / "templates" / "Images" / "reals"

    for d in [ann_fakes_dir, ann_reals_dir, img_fakes_dir, img_reals_dir]:
        assert d.exists(), f"[ERROR] Directorio no encontrado: {d}"

    rows = []
    n_missing_mask = 0

    # ── FAKES (label = 1) ──────────────────────────────────────
    fake_jsons = sorted(ann_fakes_dir.glob("*.json"))
    for ann_path in tqdm(fake_jsons, desc="Parseando fakes SIDTD"):
        img_path = None
        for ext in (".jpg", ".png"):
            cand = img_fakes_dir / (ann_path.stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            warnings.warn(f"[SIDTD] Imagen no encontrada para: {ann_path.stem}", stacklevel=2)
            continue

        with open(ann_path, encoding="utf-8") as fh:
            fake_meta = json.load(fh)

        mask_rects = _resolve_mask_rects(fake_meta, ann_reals_dir)
        if not mask_rects:
            n_missing_mask += 1

        src_stem = Path(fake_meta.get("src", "")).stem
        doc_type = re.sub(r"_\d+$", "", src_stem) if src_stem else "unknown"

        rows.append({
            "subset":         "sidtd",
            "stem":           ann_path.stem,
            "img_path":       str(img_path),
            "label":          1,
            "mask_rects_abs": json.dumps(mask_rects),
            "mask_n_rects":   len(mask_rects),
            "mask_area_px":   sum(r["w"] * r["h"] for r in mask_rects),
            "attack_type":    fake_meta.get("ctype", "unknown"),
            "doc_type":       doc_type,
        })

    # ── REALS (label = 0) ──────────────────────────────────────
    real_imgs = sorted(
        list(img_reals_dir.rglob("*.jpg")) + list(img_reals_dir.rglob("*.png"))
    )
    for img_path in real_imgs:
        stem = img_path.stem
        rows.append({
            "subset":         "sidtd",
            "stem":           stem,
            "img_path":       str(img_path),
            "label":          0,
            "mask_rects_abs": "[]",
            "mask_n_rects":   0,
            "mask_area_px":   0,
            "attack_type":    None,
            "doc_type":       re.sub(r"_\d+$", "", stem),
        })

    df = pd.DataFrame(rows)
    n_fakes = int((df.label == 1).sum())
    n_reals = int((df.label == 0).sum())

    print(f"\n[SIDTD] {len(df)} imágenes indexadas: {n_fakes} fakes | {n_reals} reals")
    if n_missing_mask > 0:
        print(
            f"[WARN]  {n_missing_mask} / {n_fakes} fakes sin máscara resuelta "
            f"(campo no encontrado en anotación VIA → máscara vacía)"
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────

def _load_model(path: Path, device: torch.device):
    """Carga un checkpoint de DocVerify y devuelve (model, params, ckpt_meta)."""
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    params = ckpt.get("params", {})
    model  = build_model(params, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, params, ckpt


def _get_calibrated_threshold() -> float:
    """
    Devuelve el umbral de clasificación calibrado sobre FantasyID.
    Prioridad: OUTER_CSV → FINAL_TEST_CSV → 0.5.
    """
    if config.OUTER_CSV.exists():
        df = pd.read_csv(config.OUTER_CSV)
        col = "thr_cls_from_val_sel"
        if col in df.columns and df[col].notna().any():
            thr = float(df[col].mean())
            print(f"[INFO] Umbral calibrado desde Nested CV (FantasyID): {thr:.4f}")
            return thr

    if config.FINAL_TEST_CSV.exists():
        df = pd.read_csv(config.FINAL_TEST_CSV)
        col = "thr_cls_from_val_sel"
        if col in df.columns:
            mask = df.get("variant", pd.Series(["multitask"] * len(df))) == "multitask"
            if mask.any() and df.loc[mask, col].notna().any():
                thr = float(df.loc[mask, col].mean())
                print(f"[INFO] Umbral calibrado desde blind test (FantasyID): {thr:.4f}")
                return thr

    print("[WARN] No se encontró umbral calibrado en CSVs. Usando 0.5.")
    return 0.5


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCIA
# ──────────────────────────────────────────────────────────────────────────────

def _collect_preds(model: torch.nn.Module, loader, device: torch.device) -> dict:
    """
    Ejecuta inferencia sobre el DataLoader y recoge predicciones brutas
    de clasificación y estadísticas pixel-level de segmentación.
    """
    model.eval()
    y_true_list: list = []
    y_prob_list: list = []
    TP = FP = TN = FN = 0
    sum_dice_pos = 0.0
    cnt_dice_pos = 0

    with torch.no_grad():
        for imgs, labels, masks in loader:
            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=config.USE_AMP):
                out = model(imgs)

            yt = labels.cpu().numpy().reshape(-1).astype(int)
            yp = torch.sigmoid(out["cls"].float()).cpu().numpy().reshape(-1)
            y_true_list.append(yt)
            y_prob_list.append(yp)

            yt_m = masks > 0.5
            yp_m = out["mask"].float() > config.THR_MASK

            TP += int(( yt_m &  yp_m).sum())
            FP += int((~yt_m &  yp_m).sum())
            TN += int((~yt_m & ~yp_m).sum())
            FN += int(( yt_m & ~yp_m).sum())

            # Dice por imagen positiva
            yt_f = yt_m.float()
            yp_f = yp_m.float()
            inter_s = (yt_f * yp_f).sum(dim=[1, 2, 3])
            denom_s = yt_f.sum(dim=[1, 2, 3]) + yp_f.sum(dim=[1, 2, 3])
            dice_s  = (2.0 * inter_s + 1e-6) / (denom_s + 1e-6)
            pos_s   = yt_f.sum(dim=[1, 2, 3]) > 0
            if pos_s.any():
                sum_dice_pos += dice_s[pos_s].sum().item()
                cnt_dice_pos += int(pos_s.sum().item())

    return {
        "y_true":       np.concatenate(y_true_list),
        "y_prob":       np.concatenate(y_prob_list),
        "seg_TP":       TP,
        "seg_FP":       FP,
        "seg_TN":       TN,
        "seg_FN":       FN,
        "sum_dice_pos": sum_dice_pos,
        "cnt_dice_pos": cnt_dice_pos,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ──────────────────────────────────────────────────────────────────────────────

def _metrics_from_preds(preds: dict, thr_cls: float) -> dict:
    """Calcula el conjunto completo de métricas desde un dict de predicciones."""
    y_true  = preds["y_true"]
    y_prob  = preds["y_prob"]
    y_pred  = (y_prob >= thr_cls).astype(int)
    TP, FP  = preds["seg_TP"], preds["seg_FP"]
    TN, FN  = preds["seg_TN"], preds["seg_FN"]

    has_two = np.unique(y_true).size == 2

    pr_auc  = float(average_precision_score(y_true, y_prob)) if y_true.size else np.nan
    roc_auc = float(roc_auc_score(y_true, y_prob))           if has_two    else np.nan
    bacc    = float(balanced_accuracy_score(y_true, y_pred)) if has_two    else np.nan

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])

    dice_global  = (2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6)
    iou_fg       = (TP + 1e-6) / (TP + FP + FN + 1e-6)
    iou_bg       = (TN + 1e-6) / (TN + FP + FN + 1e-6)
    miou         = 0.5 * (iou_fg + iou_bg)
    pix_spec     = TN / (TN + FP + 1e-9)
    pix_prec_val = TP / (TP + FP + 1e-9)
    pix_rec_val  = TP / (TP + FN + 1e-9)
    pix_f1       = 2 * pix_prec_val * pix_rec_val / (pix_prec_val + pix_rec_val + 1e-9)

    n_pos = preds.get("cnt_dice_pos", 0)
    dice_pos_mean = (
        float(preds["sum_dice_pos"] / n_pos) if n_pos > 0 else float("nan")
    )

    return {
        "thr_cls":         thr_cls,
        "pr_auc":          pr_auc,
        "roc_auc":         roc_auc,
        "bacc":            bacc,
        "f1_macro":        f1m,
        "f1_1":            float(f1),
        "prec1":           float(prec),
        "rec1":            float(rec),
        "cm_TN_FP_FN_TP":  cm.ravel().tolist() if cm.size == 4 else None,
        "dice_global":     float(dice_global),
        "dice_pos_mean":   float(dice_pos_mean),
        "miou":            float(miou),
        "pix_specificity": float(pix_spec),
        "pix_f1":          float(pix_f1),
        "pix_prec":        float(pix_prec_val),
        "pix_rec":         float(pix_rec_val),
    }


def _ensemble_preds(preds_list: list) -> dict:
    """Combina predicciones de múltiples modelos por promedio."""
    y_true  = preds_list[0]["y_true"]
    y_prob  = np.stack([p["y_prob"] for p in preds_list]).mean(axis=0)
    seg_keys = ["seg_TP", "seg_FP", "seg_TN", "seg_FN", "sum_dice_pos", "cnt_dice_pos"]
    seg_avg = {k: float(np.mean([p[k] for p in preds_list])) for k in seg_keys}
    return {
        "y_true": y_true,
        "y_prob": y_prob,
        **{k: int(v) if k in ("seg_TP", "seg_FP", "seg_TN", "seg_FN") else v
           for k, v in seg_avg.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# IMPRESIÓN DE RESULTADOS
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(label: str, met: dict) -> None:
    print(
        f"  [{label}]  "
        f"PR-AUC={met['pr_auc']:.4f} | ROC-AUC={met['roc_auc']:.4f} | "
        f"BaCC={met['bacc']:.4f} | F1-macro={met['f1_macro']:.4f} | "
        f"Dice(global)={met['dice_global']:.4f} | mIoU={met['miou']:.4f} | "
        f"thr={met['thr_cls']:.3f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación cross-dataset de DocVerify sobre SIDTD"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ruta a un .pt específico. Sin este argumento se usan todos los modelos de MODELS_DIR.",
    )
    parser.add_argument(
        "--thr", type=float, default=None,
        help="Umbral de clasificación. Sin este argumento se usa el calibrado de FantasyID.",
    )
    parser.add_argument(
        "--sidtd", type=str, default=None,
        help="Ruta al directorio raíz de SIDTD. Por defecto: ./Dataset_SIDTD o config.SIDTD_ROOT.",
    )
    args = parser.parse_args()

    sidtd_root = Path(args.sidtd) if args.sidtd else _DEFAULT_SIDTD
    device     = get_device()
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(" DocVerify — Evaluación Cross-Dataset (SIDTD)")
    print("=" * 60 + "\n")
    print(f"[INFO] SIDTD root   : {sidtd_root.resolve()}")
    print(f"[INFO] EXPORT_DIR   : {config.EXPORT_DIR.resolve()}")
    print(f"[INFO] Device       : {device}")

    # ── Dataset ───────────────────────────────────────────────
    df_sidtd = build_sidtd_df(sidtd_root)

    # ── Umbral ────────────────────────────────────────────────
    thr_cls = args.thr if args.thr is not None else _get_calibrated_threshold()
    thr_source = "cli" if args.thr is not None else "fantasyid_cv"

    # ── DataLoader ────────────────────────────────────────────
    print("\n[INFO] Cargando imágenes SIDTD en memoria...")
    loader = make_dataloader(
        df_sidtd, training=False, seed=config.SEED_BASE, device=device
    )

    # ── Modelos ───────────────────────────────────────────────
    if args.model:
        model_paths = [Path(args.model)]
    else:
        if not config.MODELS_DIR.exists():
            print(f"[ERROR] MODELS_DIR no existe: {config.MODELS_DIR}")
            print("[INFO]  Entrena el modelo con main.py o usa --model <path>")
            return
        model_paths = sorted(config.MODELS_DIR.glob("*.pt"))
        if not model_paths:
            print(f"[ERROR] No se encontraron .pt en: {config.MODELS_DIR}")
            return

    print(f"\n[INFO] Evaluando {len(model_paths)} modelo(s)...")

    per_model_preds: list = []
    per_model_rows:  list = []

    for mp in model_paths:
        print(f"\n  ▸ {mp.name}")
        try:
            model, params, ckpt_meta = _load_model(mp, device)
        except Exception as exc:
            print(f"    [WARN] No se pudo cargar {mp.name}: {exc}")
            continue

        preds = _collect_preds(model, loader, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        met = _metrics_from_preds(preds, thr_cls)
        _print_summary(mp.stem, met)

        row = {
            "model":       mp.name,
            "outer_fold":  ckpt_meta.get("outer_fold"),
            "seed":        ckpt_meta.get("seed"),
            "variant":     ckpt_meta.get("variant", "multitask"),
            "thr_source":  thr_source,
            **{f"sidtd_{k}": v for k, v in met.items()},
        }
        row["sidtd_cm_TN_FP_FN_TP"] = json.dumps(met["cm_TN_FP_FN_TP"])
        per_model_rows.append(row)
        per_model_preds.append(preds)

    if not per_model_rows:
        print("[ERROR] Ningún modelo evaluado correctamente.")
        return

    # ── Guardar resultados por modelo ─────────────────────────
    df_per = pd.DataFrame(per_model_rows)
    df_per.to_csv(SIDTD_CROSS_PER_MODEL_CSV, index=False)
    print(f"\n[OK] Resultados por modelo → {SIDTD_CROSS_PER_MODEL_CSV}")

    # ── Ensemble + resumen ────────────────────────────────────
    print(f"\n{'='*60}")
    print(" RESUMEN CROSS-DATASET (SIDTD)")
    print(f"{'='*60}")

    summary_rows: list = []
    metric_cols = [
        "pr_auc", "roc_auc", "bacc", "f1_macro",
        "dice_global", "dice_pos_mean", "miou", "pix_specificity",
    ]

    if len(per_model_preds) > 1:
        ens_preds = _ensemble_preds(per_model_preds)

        # Con umbral calibrado en FantasyID
        ens_met = _metrics_from_preds(ens_preds, thr_cls)
        _print_summary(f"Ensemble, thr={thr_cls:.3f} ({thr_source})", ens_met)

        # Oracle: umbral óptimo sobre SIDTD (cota superior)
        thr_oracle, bacc_oracle, _, _ = ev.threshold_sweep(
            ens_preds["y_true"], ens_preds["y_prob"]
        )
        ens_met_oracle = _metrics_from_preds(ens_preds, thr_oracle)
        _print_summary(f"Ensemble, thr={thr_oracle:.3f} (oracle SIDTD)", ens_met_oracle)

        summary_rows.append({
            "type": "ensemble",
            "thr_source": thr_source,
            "n_models": len(per_model_preds),
            **{f"sidtd_{k}": v for k, v in ens_met.items()},
            "sidtd_cm_TN_FP_FN_TP": json.dumps(ens_met["cm_TN_FP_FN_TP"]),
        })
        summary_rows.append({
            "type": "ensemble_oracle",
            "thr_source": "oracle_sidtd",
            "n_models": len(per_model_preds),
            **{f"sidtd_{k}": v for k, v in ens_met_oracle.items()},
            "sidtd_cm_TN_FP_FN_TP": json.dumps(ens_met_oracle["cm_TN_FP_FN_TP"]),
        })
    else:
        ens_met = _metrics_from_preds(per_model_preds[0], thr_cls)

    # Media ± std por modelo
    mean_row: dict = {"type": "per_model_mean", "thr_source": thr_source, "n_models": len(per_model_preds)}
    std_row:  dict = {"type": "per_model_std",  "thr_source": thr_source, "n_models": len(per_model_preds)}
    for mc in metric_cols:
        col_key = f"sidtd_{mc}"
        vals = [r[col_key] for r in per_model_rows if not pd.isna(r.get(col_key, np.nan))]
        mean_row[col_key] = float(np.mean(vals))  if vals          else np.nan
        std_row[col_key]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    summary_rows += [mean_row, std_row]

    if len(per_model_preds) > 1:
        print(f"\n  [Media ±std por modelo]")
        for mc in metric_cols:
            key = f"sidtd_{mc}"
            print(f"    {mc:22s}: {mean_row[key]:.4f} ± {std_row[key]:.4f}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SIDTD_CROSS_CSV, index=False)
    print(f"\n[OK] Resumen cross-dataset → {SIDTD_CROSS_CSV}")
    print(f"[OK] Pipeline completado. Resultados en: {config.EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
