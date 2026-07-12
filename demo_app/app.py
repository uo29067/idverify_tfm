"""
app.py — Backend FastAPI de la demo DocVerify.

Variante PyTorch de Fantasy_AP/app.py (Rafael González).
Cambios principales:
  - TensorFlow → PyTorch (torch-cpu)
  - Parámetro `encoder` ("patel" | "efficientnet_b4" | "vit") en vez de `mode`
  - Carga modelos DocVerify .pt (state_dict + params)
  - Todos los endpoints /dnis/* y la lógica de heatmap se mantienen igual
"""

import os
import json
import re
import io
import base64
from typing import Optional, Literal

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# DocVerify model classes — model.py vive en la misma carpeta
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DocVerifyModel, DocVerifyEfficientNet, DocVerifyViT  # noqa: E402

# ==================================================
#           VARIABLES GLOBALES
# ==================================================

PATCH_SIZE = 224
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DEVICE     = torch.device("cpu")  # CPU para el tier gratuito de HF Spaces

DNIS_DIR = os.getenv("DNIS_DIR", os.path.join(BASE_DIR, "DNIs"))

# Modelos: mejor fold de cada encoder según distancia al punto ideal (NCV)
#   Patel outer9        → PR-AUC=0.9987, Dice=0.871
#   EfficientNet outer7 → distancia=0.059, PR-AUC=0.9987, Dice=0.941
#   ViT outer1          → distancia=0.094, PR-AUC=0.9968, Dice=0.906
MODEL_SPECS = {
    "patel": dict(
        cls=DocVerifyModel, kwargs={},
        path=os.getenv("PATEL_MODEL_PATH", os.path.join(BASE_DIR, "models", "patel_outer9.pt")),
    ),
    "efficientnet_b4": dict(
        cls=DocVerifyEfficientNet, kwargs={"pretrained": False},
        path=os.getenv("EFFICIENTNET_MODEL_PATH",
                        os.path.join(BASE_DIR, "models", "efficientnet_outer7.pt")),
    ),
    "vit": dict(
        cls=DocVerifyViT, kwargs={"pretrained": False},
        path=os.getenv("VIT_MODEL_PATH", os.path.join(BASE_DIR, "models", "vit_outer1.pt")),
    ),
}

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
API_KEY         = os.getenv("API_KEY", "")

MODELS: dict[str, Optional[torch.nn.Module]] = {name: None for name in MODEL_SPECS}


@asynccontextmanager
async def lifespan(app):
    load_models()
    yield


app = FastAPI(title="DocVerify API", docs_url=None, redoc_url=None, lifespan=lifespan)

# Permitir peticiones desde Netlify y desarrollo local
_CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173",  # sobrescribir con URL de Netlify en producción
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ==================================================
#                       UTILS
# ==================================================

def check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _load_docverify_checkpoint(path: str, model_cls, **model_kwargs) -> torch.nn.Module:
    """
    Carga un checkpoint DocVerify guardado con:
      torch.save({"outer_fold": ..., "params": {...}, "state_dict": {...}}, path)
    """
    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    params = ckpt["params"]

    model = model_cls(
        dropout_rate = float(params["dropout_rate"]),
        dec_ch       = int(params["dec_ch"]),
        **model_kwargs,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _preprocess(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """
    Decodifica bytes → tensor PyTorch (1, 3, 224, 224) float32 [0,1]
    y devuelve también la imagen como numpy (224, 224, 3) [0,1] para el overlay.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3) [0,1]
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return t, arr


# ==================================================
#           GENERACIÓN DEL MAPA DE CALOR
# (lógica idéntica a Fantasy_AP/app.py — puro numpy)
# ==================================================

def _simple_colormap(hm: np.ndarray) -> np.ndarray:
    """Colormap azul→rojo sin matplotlib. hm (H,W) en [0,1] → (H,W,3)."""
    hm = np.clip(hm, 0.0, 1.0)
    r  = hm
    g  = np.clip(1.0 - np.abs(hm - 0.5) * 2.0, 0.0, 1.0)
    b  = 1.0 - hm
    return np.stack([r, g, b], axis=-1)


def _png_base64_from_rgb(rgb01: np.ndarray) -> str:
    """rgb01 (H,W,3) float [0,1] o uint8 → base64 PNG."""
    if rgb01.dtype != np.uint8:
        arr = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        arr = rgb01
    im  = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _normalize01_robust(hm: np.ndarray, pmin: float = 1.0, pmax: float = 99.0,
                         eps: float = 1e-8) -> np.ndarray:
    """Normalización robusta por percentiles (mejor contraste que min-max)."""
    hm = hm.astype(np.float32)
    lo = np.percentile(hm, pmin)
    hi = np.percentile(hm, pmax)
    return np.clip((hm - lo) / (hi - lo + eps), 0.0, 1.0)


def _heatmap_and_overlay(img_np: np.ndarray, mask_logits: torch.Tensor,
                          overlay_alpha: float = 0.45,
                          orig_size: tuple | None = None) -> dict:
    """
    Genera heatmap coloreado y overlay a partir de los logits de máscara.

    img_np     — (H, W, 3) numpy float32 [0,1]  (versión 224×224)
    orig_size  — (orig_w, orig_h) del documento original, para restaurar el aspect ratio
    """
    pm = torch.sigmoid(mask_logits).squeeze().cpu().numpy()

    if pm.shape[0] != PATCH_SIZE or pm.shape[1] != PATCH_SIZE:
        pm_t = torch.from_numpy(pm[None, None])
        pm   = F.interpolate(pm_t, size=(PATCH_SIZE, PATCH_SIZE),
                              mode="bilinear", align_corners=False).squeeze().numpy()

    hm      = _normalize01_robust(np.clip(pm, 0.0, 1.0))
    hm_rgb  = _simple_colormap(hm)
    overlay = np.clip((1.0 - overlay_alpha) * img_np + overlay_alpha * hm_rgb, 0.0, 1.0)

    if orig_size is not None:
        # Redimensionar a las proporciones originales del documento (deshace el squash a 224×224)
        orig_w, orig_h = orig_size
        hm_pil  = Image.fromarray((hm_rgb  * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
        ov_pil  = Image.fromarray((overlay * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
        buf_hm  = io.BytesIO(); hm_pil.save(buf_hm, format="PNG")
        buf_ov  = io.BytesIO(); ov_pil.save(buf_ov, format="PNG")
        return {
            "heatmap_png_base64": base64.b64encode(buf_hm.getvalue()).decode("ascii"),
            "overlay_png_base64": base64.b64encode(buf_ov.getvalue()).decode("ascii"),
        }

    return {
        "heatmap_png_base64": _png_base64_from_rgb(hm_rgb),
        "overlay_png_base64": _png_base64_from_rgb(overlay),
    }


# ==================================================
#           Startup: CARGAR MODELOS
# ==================================================

def load_models():
    for name, spec in MODEL_SPECS.items():
        path = spec["path"]
        if not os.path.exists(path):
            raise RuntimeError(f"Modelo {name} no encontrado: {path}")
        model = _load_docverify_checkpoint(path, spec["cls"], **spec["kwargs"]).to(DEVICE)
        with torch.no_grad():
            model(torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE))  # warmup
        MODELS[name] = model


# ==================================================
#                      LÓGICA DE INFERENCIA
# ==================================================

async def _run_inference(image_bytes: bytes, encoder: str) -> JSONResponse:
    try:
        # Guardar dimensiones originales para restaurar aspect ratio en la salida
        with Image.open(io.BytesIO(image_bytes)) as _orig:
            orig_size = _orig.size  # (width, height)

        x, img_np = _preprocess(image_bytes)
        x = x.to(DEVICE)

        if encoder not in MODELS:
            raise HTTPException(status_code=400, detail=f"Encoder desconocido: {encoder}")
        model = MODELS[encoder]
        if model is None:
            raise RuntimeError(f"Modelo '{encoder}' no cargado")

        with torch.no_grad():
            out = model(x)

        cls_logit  = out["cls"]
        mask_logit = out["mask"]

        p    = float(torch.sigmoid(cls_logit).item())
        cams = _heatmap_and_overlay(img_np, mask_logit, overlay_alpha=0.45, orig_size=orig_size)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return JSONResponse({
        "probability_fake":     float(np.clip(p, 0.0, 1.0)),
        "heatmap":              cams["heatmap_png_base64"],
        "heatmap_png_base64":   cams["heatmap_png_base64"],
        "overlay_png_base64":   cams["overlay_png_base64"],
    })


# ==================================================
#                      ENDPOINTS
# ==================================================

@app.get("/analyze-demo/{doc_id}")
async def analyze_demo(
    doc_id: str,
    encoder: Literal["patel", "efficientnet_b4", "vit"] = "patel",
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Analiza un documento precargado por su ID. Evita la descarga del frontend."""
    check_api_key(x_api_key)
    _safe_doc_id(doc_id)

    img_path = _find_image_for_doc(doc_id)
    if img_path is None:
        raise HTTPException(status_code=404, detail=f"Imagen no encontrada para doc_id: {doc_id}")

    with open(img_path, "rb") as fh:
        image_bytes = fh.read()

    return await _run_inference(image_bytes, encoder)


@app.post("/analyze")
async def analyze(
    encoder: Literal["patel", "efficientnet_b4", "vit"] = Form(...),
    image:   UploadFile | None = File(None),
    file:    UploadFile | None = File(None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    check_api_key(x_api_key)

    up = image or file
    if up is None:
        raise HTTPException(status_code=400, detail="Missing file field: send 'image' or 'file'")
    if up.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Only image/jpeg or image/png supported")

    image_bytes = await up.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_IMAGE_BYTES} bytes)")

    return await _run_inference(image_bytes, encoder)


# ==================================================
# Demo DNIs endpoints (idénticos a Fantasy_AP/app.py)
# ==================================================

if os.path.isdir(DNIS_DIR):
    app.mount("/DNIs", StaticFiles(directory=DNIS_DIR), name="dnis")

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_doc_id(doc_id: str) -> str:
    if not _SAFE_ID_RE.fullmatch(doc_id or ""):
        raise HTTPException(status_code=400, detail="Invalid doc_id")
    return doc_id


def _find_image_for_doc(doc_id: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(DNIS_DIR, doc_id + ext)
        if os.path.exists(p):
            return p
    try:
        for fn in os.listdir(DNIS_DIR):
            base, ext = os.path.splitext(fn)
            if base == doc_id and ext.lower() in (".png", ".jpg", ".jpeg"):
                return os.path.join(DNIS_DIR, fn)
    except Exception:
        pass
    return None


def _norm_provenance(v) -> Optional[str]:
    if v is None:
        return "original"
    s = str(v).strip().lower()
    if not s:
        return "original"
    if s in ("original", "orig", "bonafide", "bona_fide", "genuine", "real", "authentic", "clean"):
        return "original"
    if s in ("altered", "attack", "fake", "tampered", "manipulated", "forged", "forgery", "synthetic"):
        return "altered"
    if s in ("0", "false", "no"):
        return "original"
    if s in ("1", "true", "yes"):
        return "altered"
    if any(k in s for k in ("alter", "tamper", "manip", "fake", "attack", "forg", "synth")):
        return "altered"
    if any(k in s for k in ("orig", "bona", "genu", "real", "auth")):
        return "original"
    return None


def _iter_regions_any_schema(data, doc_id: str):
    if isinstance(data, dict):
        regs = data.get("regions")
        if isinstance(regs, list):
            return regs

        md = data.get("_via_img_metadata")
        if isinstance(md, dict):
            doc_base = doc_id
            chosen   = None
            for k, v in md.items():
                if not isinstance(v, dict):
                    continue
                fn   = v.get("filename") or k
                base = os.path.splitext(os.path.basename(str(fn)))[0]
                if base == doc_base:
                    chosen = v
                    break
            if chosen is None and len(md) == 1:
                chosen = next(iter(md.values()))
            if chosen is None:
                for k, v in md.items():
                    if not isinstance(v, dict):
                        continue
                    fn   = v.get("filename") or k
                    base = os.path.splitext(os.path.basename(str(fn)))[0]
                    if doc_base in base:
                        chosen = v
                        break
            if isinstance(chosen, dict):
                regs = chosen.get("regions")
                if isinstance(regs, list):
                    return regs

        for key in ("annotations", "objects", "boxes"):
            regs = data.get(key)
            if isinstance(regs, list):
                return regs

        return []

    if isinstance(data, list):
        return data
    return []


def _get_num(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _parse_region_generic(reg: dict, idx: int):
    if not isinstance(reg, dict):
        return None

    sa = reg.get("shape_attributes") or reg.get("shape") or reg.get("shapeAttributes") or {}
    ra = reg.get("region_attributes") or reg.get("attributes") or reg.get("regionAttributes") or {}
    if not isinstance(sa, dict):
        sa = {}
    if not isinstance(ra, dict):
        ra = {}

    prov_raw = (
        ra.get("region_provenance") or ra.get("provenance")
        or reg.get("region_provenance") or reg.get("provenance")
        or reg.get("label_provenance") or reg.get("type")
    )
    prov = _norm_provenance(prov_raw)
    if prov not in ("original", "altered"):
        return None

    x = _get_num(sa, ["x", "left", "xmin", "X"])
    y = _get_num(sa, ["y", "top", "ymin", "Y"])
    w = _get_num(sa, ["width", "w", "W"])
    h = _get_num(sa, ["height", "h", "H"])

    if x is None: x = _get_num(reg, ["x", "left", "xmin", "X"])
    if y is None: y = _get_num(reg, ["y", "top", "ymin", "Y"])
    if w is None: w = _get_num(reg, ["width", "w", "W"])
    if h is None: h = _get_num(reg, ["height", "h", "H"])

    x2 = _get_num(sa, ["x2", "right", "xmax"])
    y2 = _get_num(sa, ["y2", "bottom", "ymax"])
    if w is None and x is not None and x2 is not None:
        w = float(x2) - float(x)
    if h is None and y is not None and y2 is not None:
        h = float(y2) - float(y)

    bbox = reg.get("bbox")
    if isinstance(bbox, dict):
        x = x if x is not None else _get_num(bbox, ["x", "left", "xmin"])
        y = y if y is not None else _get_num(bbox, ["y", "top", "ymin"])
        w = w if w is not None else _get_num(bbox, ["width", "w"])
        h = h if h is not None else _get_num(bbox, ["height", "h"])
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        if x is None: x = bbox[0]
        if y is None: y = bbox[1]
        if w is None: w = bbox[2]
        if h is None: h = bbox[3]

    try:
        x = int(round(float(x)))
        y = int(round(float(y)))
        w = int(round(float(w)))
        h = int(round(float(h)))
    except Exception:
        return None

    if w <= 0 or h <= 0:
        return None

    field_name = (
        ra.get("field_name") or ra.get("name")
        or reg.get("field_name") or reg.get("label") or reg.get("category")
    )
    val      = ra.get("val") or ra.get("text") or reg.get("val") or reg.get("text")
    language = ra.get("language") or reg.get("language")

    return {
        "id": f"r{idx}", "x": x, "y": y, "w": w, "h": h,
        "field_name": field_name, "val": val, "language": language,
        "region_provenance": prov,
    }


@app.get("/dnis")
def list_demo_dnis():
    if not os.path.isdir(DNIS_DIR):
        return []

    items = []
    for fn in sorted(os.listdir(DNIS_DIR)):
        if not fn.lower().endswith(".json"):
            continue
        doc_id = os.path.splitext(fn)[0]
        if not _SAFE_ID_RE.fullmatch(doc_id):
            continue
        img_path = _find_image_for_doc(doc_id)
        img_url  = ("/DNIs/" + os.path.basename(img_path)) if img_path else None
        items.append({
            "id": doc_id, "image_url": img_url, "meta_url": f"/dnis/{doc_id}/meta",
            "ground_truth": _read_ground_truth(doc_id),
        })

    return items


def _read_ground_truth(doc_id: str) -> Optional[str]:
    """'bonafide' | 'attack' | None, leído del campo ground_truth del JSON del documento."""
    json_path = os.path.join(DNIS_DIR, doc_id + ".json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("ground_truth") if isinstance(data, dict) else None
    except Exception:
        return None


@app.get("/dnis/{doc_id}/meta")
def get_demo_dni_meta(doc_id: str):
    doc_id = _safe_doc_id(doc_id)

    if not os.path.isdir(DNIS_DIR):
        raise HTTPException(status_code=404, detail="DNIs directory not found")

    json_path = os.path.join(DNIS_DIR, doc_id + ".json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Metadata JSON not found")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_path = _find_image_for_doc(doc_id)
    img_url  = img_w = img_h = None

    if img_path:
        img_url = "/DNIs/" + os.path.basename(img_path)
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            pass

    ci = data.get("cropping_info", {}) if isinstance(data, dict) else {}
    if (img_w is None or img_h is None) and isinstance(ci, dict):
        img_w = img_w or ci.get("resulted_cropped_image_width")
        img_h = img_h or ci.get("resulted_cropped_image_height")

    regions_out  = []
    raw_regions  = _iter_regions_any_schema(data, doc_id)
    for i, reg in enumerate(raw_regions):
        parsed = _parse_region_generic(reg, i)
        if parsed:
            regions_out.append(parsed)

    faces_all = [r for r in regions_out if str(r.get("field_name") or "").strip().lower() == "face"]
    if faces_all:
        keep = []
        for prov in ("original", "altered"):
            faces = [r for r in faces_all if r.get("region_provenance") == prov]
            if faces:
                keep.append(max(faces, key=lambda r: int(r.get("w", 0)) * int(r.get("h", 0))))
        regions_out = [r for r in regions_out
                       if str(r.get("field_name") or "").strip().lower() != "face"] + keep

    regions_out.sort(key=lambda r: (int(r.get("y", 0)), int(r.get("x", 0))))

    return {
        "id": doc_id, "image_url": img_url,
        "image_width": img_w, "image_height": img_h,
        "regions": regions_out,
        "ground_truth": data.get("ground_truth") if isinstance(data, dict) else None,
    }


# ==================================================
# Frontend (sirve / → frontend/index.html)
# IMPORTANTE: al final para no tapar /analyze ni /dnis
# ==================================================
if os.path.isdir(os.path.join(BASE_DIR, "frontend")):
    app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "frontend"), html=True),
              name="frontend")
