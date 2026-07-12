"""
Fixtures compartidas para los tests de la demo DocVerify.

Estrategia de mock de modelos:
  - Se reemplaza la lista de handlers de startup del router por una función
    que inyecta modelos mock (MockDocVerifyModel) en lugar de cargar los .pt
    reales.  Así los tests no dependen de los ficheros de pesos (89 MB) y
    son rápidos.
"""

import io
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

# Asegurar que demo_app/ está en el path para importar app y model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock model ────────────────────────────────────────────────────────────────

class MockDocVerifyModel(nn.Module):
    """
    Devuelve tensores de ceros con las formas correctas.
    cls logit = 0  →  sigmoid(0) = 0.5 (probabilidad de falsificación)
    mask logit = 0 →  heatmap uniforme gris
    """
    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]
        return {
            "cls":  torch.zeros(B, 1),
            "mask": torch.zeros(B, 1, x.shape[2], x.shape[3]),
        }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_jpeg_bytes() -> bytes:
    """Imagen JPEG 120×80 RGB con contenido aleatorio."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_png_bytes() -> bytes:
    """Imagen PNG 64×64 RGB."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """
    TestClient de FastAPI con modelos mockeados y directorio DNIs temporal.
    Se crea una sola vez por sesión de test (scope='session').
    """
    from fastapi.testclient import TestClient
    import app as app_module

    mock = MockDocVerifyModel().eval()

    # Directorio temporal con un DNI de prueba
    dnis_dir = tmp_path_factory.mktemp("DNIs")
    _create_test_dni(dnis_dir, "test_doc")

    app_module.DNIS_DIR = str(dnis_dir)

    # Remontar el endpoint estático con el directorio temporal
    from fastapi.staticfiles import StaticFiles
    app_module.app.routes[:] = [
        r for r in app_module.app.routes
        if not (hasattr(r, "name") and r.name == "dnis")
    ]
    app_module.app.mount("/DNIs", StaticFiles(directory=str(dnis_dir)), name="dnis")

    # Parchear load_models para que el lifespan inyecte los mocks
    def _mock_load_models():
        for name in app_module.MODEL_SPECS:
            app_module.MODELS[name] = mock

    with patch.object(app_module, "load_models", _mock_load_models):
        with TestClient(app_module.app) as c:
            yield c


def _create_test_dni(dnis_dir, doc_id: str):
    """Crea un par imagen+JSON de DNI para las pruebas de endpoints."""
    import json

    # Imagen 100×60
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (60, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(dnis_dir / f"{doc_id}.jpg")

    # JSON con dos regiones (original + alterada)
    metadata = {
        "regions": [
            {
                "shape_attributes":  {"x": 5, "y": 5, "width": 30, "height": 20},
                "region_attributes": {"region_provenance": "original", "field_name": "nombre"},
            },
            {
                "shape_attributes":  {"x": 50, "y": 10, "width": 25, "height": 15},
                "region_attributes": {"region_provenance": "altered", "field_name": "fecha"},
            },
        ]
    }
    (dnis_dir / f"{doc_id}.json").write_text(json.dumps(metadata), encoding="utf-8")
