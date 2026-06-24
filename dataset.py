"""
dataset.py — Indexación, parsing de anotaciones y construcción de DataLoader (PyTorch).

Estrategia de carga:
  - Al inicio de cada outer fold se cargan TODAS las imágenes en VRAM (una sola vez).
  - Los inner folds hacen slices por índice sobre los tensores ya en VRAM.
  - Coste de IO: 1 carga por outer fold en lugar de N_TRIALS × N_INNER × 2 cargas.
"""

import ast
import json
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import config


# ============================================================
# 1. INDEXACIÓN DE IMÁGENES
# ============================================================

def list_images(split_dir: Path) -> list[Path]:
    paths = [
        p for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMG_EXTS
    ]
    return sorted(paths)


def parse_path_info(img_path: Path, root_dir: Path) -> tuple:
    parts = img_path.relative_to(root_dir).parts
    split = parts[0]
    cls   = parts[1]
    if cls == "attack":
        attack_type = parts[2]
        device      = parts[3]
    else:
        attack_type = None
        device      = parts[2]
    return split, cls, attack_type, device


def build_image_dataframe(root_dir: Path) -> pd.DataFrame:
    train_dir = root_dir / "train"
    test_dir  = root_dir / "test"
    assert train_dir.exists(), f"No se encuentra TRAIN_DIR: {train_dir.resolve()}"
    assert test_dir.exists(),  f"No se encuentra TEST_DIR: {test_dir.resolve()}"

    rows = []
    for img_path in list_images(train_dir) + list_images(test_dir):
        split, cls, attack_type, device = parse_path_info(img_path, root_dir)
        rows.append({
            "split":       split,
            "cls_dir":     cls,
            "attack_type": attack_type,
            "device":      device,
            "img_path":    str(img_path),
            "stem":        img_path.stem,
            "ext":         img_path.suffix.lower(),
        })

    df = pd.DataFrame(rows)
    print(f"[OK] Total imágenes indexadas: {len(df)}")
    print(df.groupby(["split", "cls_dir"]).size().to_string())
    return df


# ============================================================
# 2. EMPAREJAMIENTO IMAGEN ↔ JSON
# ============================================================

def find_json_for_image(img_path_str: str) -> Optional[str]:
    img_p = Path(img_path_str)
    cand  = img_p.with_suffix(".json")
    return str(cand) if cand.exists() else None


def add_json_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["json_path"] = df["img_path"].apply(find_json_for_image)
    missing = df["json_path"].isna().sum()
    print(f"[OK] JSON encontrados: {df['json_path'].notna().sum()} / {len(df)}")
    if missing > 0:
        print("[ERROR] Imágenes sin JSON asociado:")
        print(df[df["json_path"].isna()][["img_path"]].head(20).to_string())
        raise FileNotFoundError(f"Faltan {missing} JSON para imágenes en el dataset.")
    return df


# ============================================================
# 3. PARSING DE ANOTACIONES
# ============================================================

def _norm_field_name(x) -> Optional[str]:
    if x is None:
        return None
    x = unicodedata.normalize("NFKC", str(x)).strip().casefold()
    return x if x else None


def parse_doc_full_image(
    img_path: str, json_path: str, stem: str, subset: str, label: int,
) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions        = data.get("regions", []) or []
    altered_rects  = []
    n_rect         = 0
    n_rect_altered = 0
    mask_area      = 0

    for ridx, r in enumerate(regions):
        sa = r.get("shape_attributes") or {}
        ra = r.get("region_attributes") or {}
        if sa.get("name") != "rect":
            continue
        x = int(sa.get("x", 0)); y = int(sa.get("y", 0))
        w = int(sa.get("width", 0)); h = int(sa.get("height", 0))
        if w <= 0 or h <= 0:
            continue
        n_rect += 1
        prov = ra.get("region_provenance")
        prov = "original" if prov is None else str(prov).strip().casefold()
        if prov == "altered":
            n_rect_altered += 1
            altered_rects.append({
                "x": x, "y": y, "w": w, "h": h,
                "region_index": ridx,
                "field_name": _norm_field_name(ra.get("field_name")),
            })
            mask_area += w * h

    return {
        "subset": subset, "stem": stem, "img_path": img_path, "json_path": json_path,
        "label": int(label), "mask_rects_abs": json.dumps(altered_rects),
        "mask_n_rects": len(altered_rects), "mask_area_px": mask_area,
        "n_rect_regions": n_rect, "n_altered_rect_regions": n_rect_altered,
    }


def build_full_doc_df(df_base: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    rows = []
    for _, r in tqdm(df_base.iterrows(), total=len(df_base),
                     desc=f"Parsing JSON ({subset_name})"):
        rows.append(parse_doc_full_image(
            img_path=r["img_path"], json_path=r["json_path"],
            stem=r["stem"], subset=subset_name, label=r["label"],
        ))
    return pd.DataFrame(rows)


# ============================================================
# 4. MÁSCARA DESDE RECTS
# ============================================================

def _mask_from_rects(rects_json: str, W: int, H: int) -> np.ndarray:
    s = (rects_json or "").strip() or "[]"
    try:
        rects = json.loads(s)
    except Exception:
        try:
            rects = ast.literal_eval(s)
        except Exception:
            rects = []

    mask = np.zeros((H, W), dtype=np.uint8)
    for r in rects:
        rx, ry = int(r.get("x", 0)), int(r.get("y", 0))
        rw, rh = int(r.get("w", 0)), int(r.get("h", 0))
        if rw <= 0 or rh <= 0:
            continue
        x0, y0 = max(0, rx), max(0, ry)
        x1, y1 = min(W, rx + rw), min(H, ry + rh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    return mask


# ============================================================
# 5. CACHE DE VRAM — carga un DataFrame completo en GPU
# ============================================================

class VRAMCache:
    """
    Carga un DataFrame completo de imágenes en VRAM una sola vez.
    Los subconjuntos (inner folds) se obtienen como slices por índice,
    sin ninguna copia ni transferencia adicional.

    Uso:
        cache = VRAMCache(df_outer_train, device)
        loader_tr = cache.make_loader(train_indices, training=True, seed=42)
        loader_va = cache.make_loader(val_indices,   training=False, seed=42)
    """

    def __init__(self, df: pd.DataFrame, device: torch.device, label: str = ""):
        self.device = device
        self.df     = df.copy().reset_index(drop=True)
        self.df["mask_rects_abs"] = self.df["mask_rects_abs"].fillna("[]").astype(str)

        img_transform = transforms.Compose([
            transforms.Resize(
                (config.PATCH_SIZE, config.PATCH_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])

        n      = len(self.df)
        target = "VRAM" if device.type == "cuda" else "RAM"
        print(f"  [Cache] Cargando {n} imágenes en {target}{' (' + label + ')' if label else ''}...")

        imgs, labels, masks = [], [], []
        for _, row in tqdm(self.df.iterrows(), total=n, leave=False):
            img  = Image.open(row["img_path"]).convert("RGB")
            W, H = img.size
            imgs.append(img_transform(img))

            mask_np  = _mask_from_rects(row["mask_rects_abs"], W, H)
            mask_img = Image.fromarray(mask_np, mode="L").resize(
                (config.PATCH_SIZE, config.PATCH_SIZE), resample=Image.NEAREST
            )
            mask_t = torch.from_numpy(np.array(mask_img)).float().unsqueeze(0)
            masks.append((mask_t > 0.5).float())
            labels.append(torch.tensor([float(row["label"])], dtype=torch.float32))

        # Tensores únicos en VRAM
        self.imgs   = torch.stack(imgs).to(device,   non_blocking=True)
        self.labels = torch.stack(labels).to(device, non_blocking=True)
        self.masks  = torch.stack(masks).to(device,  non_blocking=True)

        mb = self.imgs.element_size() * self.imgs.nelement() / 1024 ** 2
        print(f"  [Cache] OK — {mb:.0f} MB en {target}")

    def make_loader(
        self,
        indices: np.ndarray,
        training: bool,
        seed: int,
    ) -> DataLoader:
        """
        Devuelve un DataLoader sobre el subconjunto indicado por `indices`.
        Los datos ya están en VRAM — no hay ninguna transferencia adicional.
        """
        subset = _IndexedVRAMDataset(
            self.imgs[indices],
            self.labels[indices],
            self.masks[indices],
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            subset,
            batch_size = config.BATCH_SIZE,
            shuffle    = training,
            num_workers = 0,      # Datos ya en VRAM, workers innecesarios
            pin_memory  = False,  # Ídem
            generator  = generator if training else None,
            drop_last  = False,
        )

    def free(self):
        """Libera la VRAM explícitamente."""
        del self.imgs, self.labels, self.masks
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class _IndexedVRAMDataset(Dataset):
    """Dataset ligero sobre slices de tensores ya en VRAM."""

    def __init__(self, imgs: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor):
        self.imgs   = imgs
        self.labels = labels
        self.masks  = masks

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple:
        return self.imgs[idx], self.labels[idx], self.masks[idx]


# ============================================================
# 6. DATALOADER CLÁSICO (para outer test y holdout)
# ============================================================

def make_dataloader(
    df: pd.DataFrame,
    training: bool,
    seed: int,
    device: torch.device = None,
) -> DataLoader:
    """
    Carga un DataFrame en VRAM/RAM y devuelve un DataLoader.
    Usar solo para conjuntos que no se reutilizan (outer test, holdout).
    Para inner folds usar VRAMCache.make_loader().
    """
    cache = VRAMCache(df, device or torch.device("cpu"))
    indices = np.arange(len(df))
    return cache.make_loader(indices, training=training, seed=seed)
