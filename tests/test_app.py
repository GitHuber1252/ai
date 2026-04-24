import io
import os
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app as ai_app


def _image_bytes() -> bytes:
    img = Image.new("RGB", (16, 16), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_reports_missing_model_or_etalons():
    client = TestClient(ai_app.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "ok" in response.json()


def test_list_etalons_from_temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        open(os.path.join(temp_dir, "a.jpg"), "wb").write(_image_bytes())
        open(os.path.join(temp_dir, "note.txt"), "wb").write(b"bad")

        with patch.object(ai_app, "ETALON_DIR", temp_dir):
            client = TestClient(ai_app.app)
            response = client.get("/etalons")
            assert response.status_code == 200
            payload = response.json()
            assert payload["count"] == 1
            assert payload["items"] == ["a.jpg"]


def test_upload_and_delete_etalon():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(ai_app, "ETALON_DIR", temp_dir):
            client = TestClient(ai_app.app)

            upload = client.post(
                "/etalons",
                files={"file": ("new_etalon.png", _image_bytes(), "image/png")},
            )
            assert upload.status_code == 200

            listing = client.get("/etalons")
            assert listing.status_code == 200
            assert "new_etalon.png" in listing.json()["items"]

            delete = client.post("/etalons/delete", data={"filename": "new_etalon.png"})
            assert delete.status_code == 200

            listing_after = client.get("/etalons")
            assert "new_etalon.png" not in listing_after.json()["items"]
