"""
select_demo_images.py — Selecciona la galería de documentos de demo desde el
holdout set de FantasyID (15% del total, separado antes de cualquier
entrenamiento; el 85% restante es el conjunto de desarrollo usado en el NCV).

Criterios de selección:
  - Documentos del holdout (mismo split determinista que en entrenamiento, seed=42).
  - Equilibrio bonafide/attack (~50/50).
  - Diversidad documental: nacionalidad (prefijo del stem) y tipo de captura
    (dispositivo para bonafide, técnica de manipulación para attack).
  - Acuerdo entre los tres encoders (Patel, EfficientNet-B4, ViT-B/16) con la
    etiqueta real, usando el umbral de clasificación calibrado de cada fold
    (columna thr_cls_from_val_sel en nested_outer_results.csv), reservando 1-2
    casos discrepantes (donde los tres no coinciden) como ejemplo ilustrativo
    de diferencias entre arquitecturas.

Uso (desde la raíz del proyecto, con FantasyID/ descomprimido):
    python demo/select_demo_images.py
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent   # mywork/
DEMO_APP = ROOT / "demo" / "demo_app"
DNIS_OUT = DEMO_APP / "DNIs"
N_SELECT      = 10
N_DISCREPANT  = 2   # reservados dentro de N_SELECT para casos donde los 3 encoders no coinciden

sys.path.insert(0, str(ROOT))
import config          # noqa: E402  config.py raíz del proyecto (FantasyID, PATCH_SIZE...)
import main as main_module  # noqa: E402
from dataset import add_json_paths, build_full_doc_df, build_image_dataframe  # noqa: E402


# ============================================================
# Modelos de la demo: encoder -> (checkpoint, clase, kwargs, outer_fold, csv de origen)
# ============================================================
MODEL_SPECS = {
    "patel": dict(
        ckpt="patel_outer9.pt", cls_name="DocVerifyModel", kwargs={},
        outer_fold=9, results_csv=ROOT / "exports_hpo_pareto_nested" / "nested_outer_results.csv",
    ),
    "efficientnet_b4": dict(
        ckpt="efficientnet_outer7.pt", cls_name="DocVerifyEfficientNet", kwargs={"pretrained": False},
        outer_fold=7, results_csv=ROOT / "exports_efficientnet_b4" / "nested_outer_results.csv",
    ),
    "vit": dict(
        ckpt="vit_outer1.pt", cls_name="DocVerifyViT", kwargs={"pretrained": False},
        outer_fold=1, results_csv=ROOT / "exports_vit_b16" / "nested_outer_results.csv",
    ),
}


def _load_demo_model_module():
    """Carga demo_app/model.py como módulo aislado (sin chocar con el model.py raíz)."""
    spec = importlib.util.spec_from_file_location("demo_model", DEMO_APP / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_checkpoint(path: Path, model_cls, kwargs: dict):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    model = model_cls(
        dropout_rate=float(params["dropout_rate"]),
        dec_ch=int(params["dec_ch"]),
        **kwargs,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _threshold_for(spec: dict) -> float:
    df = pd.read_csv(spec["results_csv"])
    row = df[df["outer_fold"] == spec["outer_fold"]].iloc[0]
    return float(row["thr_cls_from_val_sel"])


def _nationality(stem: str) -> str:
    """Prefijo alfabético antes del primer '-' o dígito: 'arabic-024_03' -> 'arabic'."""
    for i, ch in enumerate(stem):
        if ch == "-" or ch.isdigit():
            return stem[:i]
    return stem


def build_holdout_with_device() -> pd.DataFrame:
    """
    Reproduce el split 75/10/15 de main.load_and_prepare_data() (mismo seed,
    mismo GroupShuffleSplit), pero conserva las columnas 'device' y
    'attack_type', que se pierden en build_full_doc_df() y por tanto no están
    disponibles en el df_holdout que devuelve main.load_and_prepare_data().
    """
    root = config.DATASET_ROOT.resolve()
    df_imgs = build_image_dataframe(root)
    df_imgs = add_json_paths(df_imgs)
    df_imgs["label"] = (df_imgs["cls_dir"] == "attack").astype(int)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.SEED_BASE)
    _, idx_test = next(gss1.split(df_imgs, y=df_imgs["label"], groups=df_imgs["stem"]))
    df_test_base = df_imgs.iloc[idx_test].reset_index(drop=True)

    df_holdout = build_full_doc_df(df_test_base, "test")
    df_holdout = df_holdout.merge(
        df_test_base[["img_path", "device", "attack_type"]], on="img_path", how="left"
    )
    df_holdout["capture_type"] = df_holdout.apply(
        lambda r: r["device"] if r["label"] == 0 else r["attack_type"], axis=1
    )
    return df_holdout.reset_index(drop=True)


def load_models():
    demo_model = _load_demo_model_module()
    models, thresholds = {}, {}
    for name, spec in MODEL_SPECS.items():
        cls = getattr(demo_model, spec["cls_name"])
        print(f"  Cargando {name} (outer_fold={spec['outer_fold']})...")
        models[name] = _load_checkpoint(DEMO_APP / "models" / spec["ckpt"], cls, spec["kwargs"])
        thresholds[name] = _threshold_for(spec)
        print(f"    thr_cls_from_val_sel = {thresholds[name]:.4f}")
    return models, thresholds


def run_inference(df_holdout: pd.DataFrame, models: dict, thresholds: dict) -> pd.DataFrame:
    tfm = transforms.Compose([
        transforms.Resize((config.PATCH_SIZE, config.PATCH_SIZE),
                           interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    records = []
    for _, row in tqdm(df_holdout.iterrows(), total=len(df_holdout), desc="Inferencia (3 encoders)"):
        img = Image.open(row["img_path"]).convert("RGB")
        x = tfm(img).unsqueeze(0)

        rec = {
            "stem": row["stem"], "img_path": row["img_path"], "json_path": row["json_path"],
            "ext": Path(row["img_path"]).suffix, "label": int(row["label"]),
            "mask_n_rects": int(row["mask_n_rects"]), "nationality": _nationality(row["stem"]),
            "capture_type": row["capture_type"],
        }
        probs = {}
        for name, model in models.items():
            with torch.no_grad():
                out = model(x)
            p = float(torch.sigmoid(out["cls"]).item())
            probs[name] = p
            rec[f"p_{name}"] = p
            rec[f"pred_{name}"] = int(p >= thresholds[name])
        rec["correct_all"] = all(rec[f"pred_{n}"] == rec["label"] for n in models)
        rec["disagreement"] = max(probs.values()) - min(probs.values())
        records.append(rec)

    return pd.DataFrame(records)


def _pick_diverse(cand: pd.DataFrame, n: int, used_nat: set, used_cap: set) -> list:
    """Selección greedy con nacionalidad y tipo de captura no repetidos (si es posible)."""
    picked = []
    for _, r in cand.iterrows():
        if len(picked) >= n:
            break
        if r["nationality"] in used_nat or r["capture_type"] in used_cap:
            continue
        picked.append(r)
        used_nat.add(r["nationality"])
        used_cap.add(r["capture_type"])
    return picked


def select_gallery(df_pred: pd.DataFrame, n_select: int = N_SELECT,
                    n_discrepant: int = N_DISCREPANT) -> pd.DataFrame:
    """
    Selección equilibrada por clase (target_per_label documentos de cada label,
    calculado ANTES de repartir por clase para no perder el 50/50 si una de las
    dos clases se queda sin candidatos diversos).
    """
    used_nat, used_cap, used_stems = set(), set(), set()
    disagree_pool = df_pred[~df_pred["correct_all"]].sort_values("disagreement", ascending=False)
    agree_pool = df_pred[df_pred["correct_all"]].copy()
    target_per_label = n_select // 2
    n_discrepant_per_label = max(1, n_discrepant // 2)

    selected = []
    for label in (0, 1):
        label_selected = []

        # 1) caso(s) discrepante(s) ilustrativo(s), si hay candidatos
        cand = disagree_pool[(disagree_pool["label"] == label)
                              & (~disagree_pool["stem"].isin(used_stems))]
        picked = _pick_diverse(cand, n_discrepant_per_label, used_nat, used_cap)
        if len(picked) < n_discrepant_per_label:
            for _, r in cand.iterrows():
                if len(picked) >= n_discrepant_per_label:
                    break
                if r["stem"] not in {p["stem"] for p in picked}:
                    picked.append(r)
        for r in picked:
            used_stems.add(r["stem"]); used_nat.add(r["nationality"]); used_cap.add(r["capture_type"])
        label_selected.extend(picked)

        # 2) resto: acuerdo unánime, priorizando diversidad
        remaining = target_per_label - len(label_selected)
        cand = agree_pool[(agree_pool["label"] == label) & (~agree_pool["stem"].isin(used_stems))]
        if label == 1:
            cand = cand[cand["mask_n_rects"] > 0]
        cand = cand.sample(frac=1, random_state=config.SEED_BASE)
        picked = _pick_diverse(cand, remaining, used_nat, used_cap)
        for r in picked:
            used_stems.add(r["stem"])
        label_selected.extend(picked)

        # 3) relleno DENTRO DE LA MISMA CLASE si la diversidad no llegó al target
        remaining = target_per_label - len(label_selected)
        if remaining > 0:
            cand = agree_pool[(agree_pool["label"] == label) & (~agree_pool["stem"].isin(used_stems))]
            if label == 1:
                cand = cand[cand["mask_n_rects"] > 0]
            cand = cand.sample(frac=1, random_state=config.SEED_BASE)
            for _, r in cand.iterrows():
                if remaining <= 0:
                    break
                label_selected.append(r)
                used_stems.add(r["stem"])
                remaining -= 1

        selected.extend(label_selected)

    df_sel = pd.DataFrame(selected).reset_index(drop=True)
    df_sel["selection_type"] = ["discrepant" if not c else "agreement"
                                 for c in df_sel["correct_all"]]
    return df_sel


def copy_gallery(df_selected: pd.DataFrame):
    backup = DEMO_APP / "DNIs_rafael_backup"
    if DNIS_OUT.exists() and not backup.exists():
        shutil.move(str(DNIS_OUT), str(backup))
        print(f"  Galería anterior (Rafael) movida a {backup}")

    if DNIS_OUT.exists():
        shutil.rmtree(DNIS_OUT)  # limpia selecciones de ejecuciones anteriores del script
    DNIS_OUT.mkdir(parents=True, exist_ok=True)

    for _, r in df_selected.iterrows():
        shutil.copy(r["img_path"], DNIS_OUT / f"{r['stem']}{r['ext']}")

        with open(r["json_path"], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["ground_truth"] = "bonafide" if r["label"] == 0 else "attack"
        with open(DNIS_OUT / f"{r['stem']}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("[1/4] Reconstruyendo el holdout set (split determinista, seed=42)...")
    main_module.set_global_seeds()
    df_holdout = build_holdout_with_device()
    print(f"  Holdout: {len(df_holdout)} documentos")

    print("[2/4] Cargando los tres modelos...")
    models, thresholds = load_models()

    print("[3/4] Ejecutando inferencia sobre el holdout...")
    df_pred = run_inference(df_holdout, models, thresholds)
    df_pred.to_csv(ROOT / "demo" / "holdout_predictions.csv", index=False)
    print(f"  Acuerdo entre los 3 encoders (correctos): "
          f"{df_pred['correct_all'].sum()} / {len(df_pred)}")

    print("[4/4] Seleccionando galería y copiando a demo_app/DNIs/...")
    df_selected = select_gallery(df_pred)
    df_selected.to_csv(ROOT / "demo" / "gallery_selection.csv", index=False)
    copy_gallery(df_selected)

    print(f"\n[OK] Galería seleccionada ({len(df_selected)} documentos):")
    cols = ["stem", "label", "nationality", "capture_type", "selection_type",
            "p_patel", "p_efficientnet_b4", "p_vit"]
    print(df_selected[cols].to_string(index=False))


if __name__ == "__main__":
    main()
