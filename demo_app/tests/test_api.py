"""
Pruebas de integración de los endpoints de la API DocVerify.

Usa el fixture `client` de conftest.py (TestClient con modelos mockeados).
"""

import io
import json

import numpy as np
import pytest
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg(width=100, height=80) -> bytes:
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _make_png(width=64, height=64) -> bytes:
    rng = np.random.default_rng(55)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ── GET /dnis ─────────────────────────────────────────────────────────────────

class TestListDnis:

    def test_returns_200(self, client):
        r = client.get("/dnis")
        assert r.status_code == 200

    def test_returns_list(self, client):
        r = client.get("/dnis")
        assert isinstance(r.json(), list)

    def test_contains_test_doc(self, client):
        ids = [d["id"] for d in client.get("/dnis").json()]
        assert "test_doc" in ids

    def test_each_item_has_required_fields(self, client):
        items = client.get("/dnis").json()
        for item in items:
            assert "id"       in item
            assert "meta_url" in item

    def test_meta_url_format(self, client):
        items = client.get("/dnis").json()
        doc   = next(d for d in items if d["id"] == "test_doc")
        assert doc["meta_url"] == "/dnis/test_doc/meta"


# ── GET /dnis/{doc_id}/meta ───────────────────────────────────────────────────

class TestDniMeta:

    def test_existing_doc_returns_200(self, client):
        r = client.get("/dnis/test_doc/meta")
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.get("/dnis/test_doc/meta").json()
        assert "id"      in data
        assert "regions" in data

    def test_regions_contain_provenance(self, client):
        data    = client.get("/dnis/test_doc/meta").json()
        provs   = {r["region_provenance"] for r in data["regions"]}
        assert "original" in provs
        assert "altered"  in provs

    def test_region_has_bbox_fields(self, client):
        data    = client.get("/dnis/test_doc/meta").json()
        for r in data["regions"]:
            for field in ("x", "y", "w", "h"):
                assert field in r

    def test_nonexistent_doc_returns_404(self, client):
        r = client.get("/dnis/nonexistent_doc/meta")
        assert r.status_code == 404

    def test_path_traversal_returns_400(self, client):
        r = client.get("/dnis/../etc/passwd/meta")
        assert r.status_code in (400, 404)

    def test_id_with_slash_rejected(self, client):
        r = client.get("/dnis/foo%2Fbar/meta")
        assert r.status_code in (400, 404)

    def test_id_with_spaces_rejected(self, client):
        r = client.get("/dnis/foo bar/meta")
        assert r.status_code in (400, 404)


# ── POST /analyze ─────────────────────────────────────────────────────────────

class TestAnalyze:

    def _post(self, client, encoder, image_bytes, content_type="image/jpeg"):
        return client.post(
            "/analyze",
            data={"encoder": encoder},
            files={"image": ("test.jpg", io.BytesIO(image_bytes), content_type)},
        )

    # ── Casos válidos ──

    def test_patel_returns_200(self, client):
        r = self._post(client, "patel", _make_jpeg())
        assert r.status_code == 200

    def test_efficientnet_returns_200(self, client):
        r = self._post(client, "efficientnet_b4", _make_jpeg())
        assert r.status_code == 200

    def test_vit_returns_200(self, client):
        r = self._post(client, "vit", _make_jpeg())
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        r    = self._post(client, "patel", _make_jpeg())
        data = r.json()
        assert "probability_fake"    in data
        assert "heatmap_png_base64"  in data
        assert "overlay_png_base64"  in data

    def test_probability_in_range(self, client):
        r = self._post(client, "patel", _make_jpeg())
        p = r.json()["probability_fake"]
        assert 0.0 <= p <= 1.0

    def test_mock_model_returns_05(self, client):
        # MockDocVerifyModel devuelve logit=0 → sigmoid(0)=0.5
        r = self._post(client, "patel", _make_jpeg())
        assert r.json()["probability_fake"] == pytest.approx(0.5, abs=1e-4)

    def test_accepts_png(self, client):
        r = self._post(client, "patel", _make_png(), content_type="image/png")
        assert r.status_code == 200

    def test_heatmap_is_nonempty_base64(self, client):
        import base64
        r    = self._post(client, "patel", _make_jpeg())
        data = base64.b64decode(r.json()["heatmap_png_base64"])
        assert len(data) > 0

    # ── Casos de error ──

    def test_missing_file_returns_400(self, client):
        r = client.post("/analyze", data={"encoder": "patel"})
        assert r.status_code == 400

    def test_wrong_content_type_returns_415(self, client):
        r = self._post(client, "patel", b"fake data", content_type="image/gif")
        assert r.status_code == 415

    def test_empty_file_returns_400(self, client):
        r = client.post(
            "/analyze",
            data={"encoder": "patel"},
            files={"image": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert r.status_code == 400

    def test_unknown_encoder_returns_422(self, client):
        r = self._post(client, "unknown_encoder", _make_jpeg())
        assert r.status_code == 422

    def test_missing_encoder_returns_422(self, client):
        r = client.post(
            "/analyze",
            files={"image": ("test.jpg", io.BytesIO(_make_jpeg()), "image/jpeg")},
        )
        assert r.status_code == 422


# ── GET /analyze-demo/{doc_id} ────────────────────────────────────────────────

class TestAnalyzeDemo:

    # ── Casos válidos ──

    def test_existing_doc_patel_returns_200(self, client):
        r = client.get("/analyze-demo/test_doc?encoder=patel")
        assert r.status_code == 200

    def test_existing_doc_efficientnet_returns_200(self, client):
        r = client.get("/analyze-demo/test_doc?encoder=efficientnet_b4")
        assert r.status_code == 200

    def test_existing_doc_vit_returns_200(self, client):
        r = client.get("/analyze-demo/test_doc?encoder=vit")
        assert r.status_code == 200

    def test_default_encoder_is_patel(self, client):
        r = client.get("/analyze-demo/test_doc")
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.get("/analyze-demo/test_doc?encoder=patel").json()
        assert "probability_fake"   in data
        assert "heatmap_png_base64" in data
        assert "overlay_png_base64" in data

    def test_probability_in_range(self, client):
        p = client.get("/analyze-demo/test_doc?encoder=patel").json()["probability_fake"]
        assert 0.0 <= p <= 1.0

    def test_mock_model_returns_05(self, client):
        p = client.get("/analyze-demo/test_doc?encoder=patel").json()["probability_fake"]
        assert p == pytest.approx(0.5, abs=1e-4)

    def test_heatmap_is_nonempty_base64(self, client):
        import base64
        data = client.get("/analyze-demo/test_doc?encoder=patel").json()
        assert len(base64.b64decode(data["heatmap_png_base64"])) > 0

    # ── Casos de error ──

    def test_nonexistent_doc_returns_404(self, client):
        r = client.get("/analyze-demo/nonexistent_doc?encoder=patel")
        assert r.status_code == 404

    def test_unknown_encoder_returns_400(self, client):
        r = client.get("/analyze-demo/test_doc?encoder=unknown")
        assert r.status_code in (400, 422)

    def test_path_traversal_returns_400(self, client):
        r = client.get("/analyze-demo/../etc/passwd")
        assert r.status_code in (400, 404)
