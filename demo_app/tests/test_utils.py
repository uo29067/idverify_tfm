"""
Pruebas unitarias de las funciones de utilidad de app.py:
  - _preprocess
  - _normalize01_robust
  - _simple_colormap
  - _heatmap_and_overlay
  - _norm_provenance
  - _parse_region_generic
  - _iter_regions_any_schema
  - _safe_doc_id
"""

import io

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from PIL import Image

from app import (
    _heatmap_and_overlay,
    _iter_regions_any_schema,
    _norm_provenance,
    _normalize01_robust,
    _parse_region_generic,
    _preprocess,
    _safe_doc_id,
    _simple_colormap,
)


# ── _preprocess ───────────────────────────────────────────────────────────────

class TestPreprocess:

    def test_returns_tensor_and_numpy(self, sample_jpeg_bytes):
        t, arr = _preprocess(sample_jpeg_bytes)
        assert isinstance(t, torch.Tensor)
        assert isinstance(arr, np.ndarray)

    def test_tensor_shape(self, sample_jpeg_bytes):
        t, _ = _preprocess(sample_jpeg_bytes)
        assert t.shape == (1, 3, 224, 224)

    def test_tensor_range(self, sample_jpeg_bytes):
        t, _ = _preprocess(sample_jpeg_bytes)
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0

    def test_numpy_shape(self, sample_jpeg_bytes):
        _, arr = _preprocess(sample_jpeg_bytes)
        assert arr.shape == (224, 224, 3)

    def test_numpy_range(self, sample_jpeg_bytes):
        _, arr = _preprocess(sample_jpeg_bytes)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_accepts_png(self, sample_png_bytes):
        t, arr = _preprocess(sample_png_bytes)
        assert t.shape == (1, 3, 224, 224)

    def test_dtype_float32(self, sample_jpeg_bytes):
        t, arr = _preprocess(sample_jpeg_bytes)
        assert t.dtype == torch.float32
        assert arr.dtype == np.float32


# ── _normalize01_robust ───────────────────────────────────────────────────────

class TestNormalize01Robust:

    def test_output_in_range(self):
        rng = np.random.default_rng(1)
        hm  = rng.random((50, 50)).astype(np.float32)
        out = _normalize01_robust(hm)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_uniform_input_returns_zeros(self):
        hm  = np.ones((10, 10), dtype=np.float32) * 0.5
        out = _normalize01_robust(hm)
        # Cuando min≈max el denominador es eps → resultado ≈ 0
        assert out.max() < 1e-3

    def test_preserves_shape(self):
        hm  = np.zeros((30, 40), dtype=np.float32)
        out = _normalize01_robust(hm)
        assert out.shape == (30, 40)

    def test_output_dtype_float32(self):
        hm  = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float64)
        out = _normalize01_robust(hm)
        assert out.dtype == np.float32


# ── _simple_colormap ──────────────────────────────────────────────────────────

class TestSimpleColormap:

    def test_output_shape(self):
        hm  = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        out = _simple_colormap(hm)
        assert out.shape == (10, 10, 3)

    def test_output_range(self):
        hm  = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        out = _simple_colormap(hm)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_zero_is_blue(self):
        hm  = np.zeros((1, 1), dtype=np.float32)
        out = _simple_colormap(hm)
        # hm=0 → R=0, B=1
        assert out[0, 0, 0] == pytest.approx(0.0)
        assert out[0, 0, 2] == pytest.approx(1.0)

    def test_one_is_red(self):
        hm  = np.ones((1, 1), dtype=np.float32)
        out = _simple_colormap(hm)
        # hm=1 → R=1, B=0
        assert out[0, 0, 0] == pytest.approx(1.0)
        assert out[0, 0, 2] == pytest.approx(0.0)

    def test_clips_out_of_range(self):
        hm  = np.array([[-0.5, 1.5]], dtype=np.float32)
        out = _simple_colormap(hm)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ── _heatmap_and_overlay ──────────────────────────────────────────────────────

class TestHeatmapAndOverlay:

    def _make_inputs(self):
        img_np     = np.zeros((224, 224, 3), dtype=np.float32)
        mask_logit = torch.zeros(1, 1, 224, 224)
        return img_np, mask_logit

    def test_returns_dict_with_required_keys(self):
        img_np, mask = self._make_inputs()
        out = _heatmap_and_overlay(img_np, mask)
        assert "heatmap_png_base64"  in out
        assert "overlay_png_base64"  in out

    def test_base64_strings_are_nonempty(self):
        img_np, mask = self._make_inputs()
        out = _heatmap_and_overlay(img_np, mask)
        assert len(out["heatmap_png_base64"]) > 0
        assert len(out["overlay_png_base64"]) > 0

    def test_base64_decodes_to_valid_png(self):
        import base64
        img_np, mask = self._make_inputs()
        out  = _heatmap_and_overlay(img_np, mask)
        data = base64.b64decode(out["heatmap_png_base64"])
        img  = Image.open(io.BytesIO(data))
        assert img.format == "PNG"

    def test_non_square_mask_is_resized(self):
        img_np = np.zeros((224, 224, 3), dtype=np.float32)
        mask   = torch.zeros(1, 1, 14, 14)   # resolución de ViT
        out    = _heatmap_and_overlay(img_np, mask)
        assert len(out["heatmap_png_base64"]) > 0


# ── _norm_provenance ──────────────────────────────────────────────────────────

class TestNormProvenance:

    @pytest.mark.parametrize("v,expected", [
        (None,        "original"),
        ("",          "original"),
        ("original",  "original"),
        ("Original",  "original"),
        ("bonafide",  "original"),
        ("genuine",   "original"),
        ("real",      "original"),
        ("0",         "original"),
        ("altered",   "altered"),
        ("Altered",   "altered"),
        ("attack",    "altered"),
        ("fake",      "altered"),
        ("tampered",  "altered"),
        ("1",         "altered"),
        ("true",      "altered"),
    ])
    def test_known_values(self, v, expected):
        assert _norm_provenance(v) == expected

    def test_unknown_returns_none(self):
        assert _norm_provenance("unknown_xyz") is None


# ── _parse_region_generic ─────────────────────────────────────────────────────

class TestParseRegionGeneric:

    def _region(self, x=10, y=20, w=50, h=30, prov="original", field="nombre"):
        return {
            "shape_attributes":  {"x": x, "y": y, "width": w, "height": h},
            "region_attributes": {"region_provenance": prov, "field_name": field},
        }

    def test_valid_original_region(self):
        r = _parse_region_generic(self._region(), 0)
        assert r is not None
        assert r["x"] == 10
        assert r["y"] == 20
        assert r["w"] == 50
        assert r["h"] == 30
        assert r["region_provenance"] == "original"
        assert r["field_name"] == "nombre"

    def test_valid_altered_region(self):
        r = _parse_region_generic(self._region(prov="altered"), 1)
        assert r["region_provenance"] == "altered"

    def test_zero_width_returns_none(self):
        assert _parse_region_generic(self._region(w=0), 0) is None

    def test_negative_height_returns_none(self):
        assert _parse_region_generic(self._region(h=-5), 0) is None

    def test_unknown_provenance_returns_none(self):
        assert _parse_region_generic(self._region(prov="unknown"), 0) is None

    def test_non_dict_returns_none(self):
        assert _parse_region_generic("not a dict", 0) is None

    def test_missing_coords_returns_none(self):
        reg = {"region_attributes": {"region_provenance": "original"}}
        assert _parse_region_generic(reg, 0) is None

    def test_id_format(self):
        r = _parse_region_generic(self._region(), 3)
        assert r["id"] == "r3"

    def test_float_coords_rounded(self):
        # Python 3 usa redondeo bancario: round(0.5)=0, round(1.5)=2, etc.
        r = _parse_region_generic(self._region(x=10.7, y=20.3, w=50.5, h=30.1), 0)
        assert r["x"] == 11               # round(10.7) = 11
        assert r["y"] == 20               # round(20.3) = 20
        assert r["w"] == round(50.5)      # redondeo bancario: 50
        assert r["h"] == 30               # round(30.1) = 30


# ── _iter_regions_any_schema ──────────────────────────────────────────────────

class TestIterRegionsAnySchema:

    def test_list_format(self):
        data = [{"shape_attributes": {}}]
        assert _iter_regions_any_schema(data, "doc") == data

    def test_dict_with_regions_key(self):
        regions = [{"r": 1}, {"r": 2}]
        assert _iter_regions_any_schema({"regions": regions}, "doc") == regions

    def test_via_multi_image_format(self):
        regions = [{"shape_attributes": {"x": 0}}]
        data = {
            "_via_img_metadata": {
                "abc123": {
                    "filename": "test_doc.jpg",
                    "regions":  regions,
                }
            }
        }
        result = _iter_regions_any_schema(data, "test_doc")
        assert result == regions

    def test_annotations_fallback(self):
        anns = [{"bbox": [0, 0, 10, 10]}]
        assert _iter_regions_any_schema({"annotations": anns}, "doc") == anns

    def test_empty_dict_returns_empty_list(self):
        assert _iter_regions_any_schema({}, "doc") == []

    def test_non_list_non_dict_returns_empty(self):
        assert _iter_regions_any_schema("invalid", "doc") == []

    def test_via_single_entry_fallback(self):
        regions = [{"shape_attributes": {}}]
        data = {
            "_via_img_metadata": {
                "some_key": {"filename": "other.jpg", "regions": regions}
            }
        }
        # Solo hay una entrada → la devuelve aunque no coincida el nombre
        result = _iter_regions_any_schema(data, "test_doc")
        assert result == regions


# ── _safe_doc_id ──────────────────────────────────────────────────────────────

class TestSafeDocId:

    @pytest.mark.parametrize("doc_id", [
        "french-097_03",
        "usa-016_03",
        "doc123",
        "A.B-C_D",
    ])
    def test_valid_ids(self, doc_id):
        assert _safe_doc_id(doc_id) == doc_id

    @pytest.mark.parametrize("doc_id", [
        "../etc/passwd",
        "doc/id",
        "doc id",
        "",
        None,
        "doc;rm",
    ])
    def test_invalid_ids_raise_400(self, doc_id):
        with pytest.raises(HTTPException) as exc:
            _safe_doc_id(doc_id)
        assert exc.value.status_code == 400
