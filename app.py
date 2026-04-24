import io
import os
import re
from uuid import uuid4

import cv2
import numpy as np
import psycopg
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms


class SiameseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision import models

        self.resnet = models.resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_once(x1), self.forward_once(x2)


app = FastAPI(title="Handwriting AI", version="2.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/handwriting_expert_epoch_5.pth")
THRESHOLD = float(os.getenv("AI_THRESHOLD", "0.3"))
DEVICE = torch.device("cpu")

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "autor_recognition")
DB_USER = os.getenv("DB_USER", "autor_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "autor_pass")

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = SiameseNetwork().to(DEVICE)
model_loaded = False
model_load_error = ""


def get_conn():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
    )


def _try_load_model() -> None:
    global model_loaded, model_load_error
    if not os.path.exists(MODEL_PATH):
        model_load_error = f"Model file not found: {MODEL_PATH}"
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        model_loaded = True
    except Exception as exc:
        model_load_error = str(exc)


def dist_to_prob(distance: float, threshold: float = 0.3) -> float:
    k = 10.0
    prob = 1.0 / (1.0 + np.exp(k * (distance - threshold)))
    return float(prob)


def preprocess_image(img: Image.Image) -> Image.Image:
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


def transform_image(img: Image.Image) -> torch.Tensor:
    img_preprocessed = preprocess_image(img.convert("RGB"))
    return transform(img_preprocessed).unsqueeze(0).to(DEVICE)


def validate_image_file(contents: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(contents))
        return img.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")


def safe_name_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", value).strip("_")
    return cleaned or "unknown"


def build_storage_filename(fio: str, original_name: str) -> tuple[str, str]:
    ext = os.path.splitext(original_name)[1].lower().lstrip(".")
    image_id = str(uuid4())
    fio_part = safe_name_component(fio)
    filename = f"{fio_part}_photo_{image_id}" + (f".{ext}" if ext else "")
    return image_id, filename


def list_etalons_rows() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, storage_filename, full_name, mime_type, content
                FROM images
                WHERE kind = 'etalon'
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()
    result = []
    for row in rows:
        result.append(
            {
                "id": str(row[0]),
                "storage_filename": row[1],
                "full_name": row[2],
                "mime_type": row[3],
                "content": bytes(row[4]),
            }
        )
    return result


@app.on_event("startup")
def startup_event() -> None:
    _try_load_model()


@app.get("/")
def root() -> dict:
    return {"message": "Handwriting AI API is running.", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM images WHERE kind = 'etalon'")
                etalon_count = int(cur.fetchone()[0])
    except Exception as exc:
        return {"ok": False, "error": f"DB error: {exc}", "etalon_count": 0}
    return {"ok": model_loaded and etalon_count > 0, "error": model_load_error, "etalon_count": etalon_count}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if not model_loaded:
        raise HTTPException(status_code=500, detail=f"Model is not loaded: {model_load_error}")

    etalons = list_etalons_rows()
    if not etalons:
        raise HTTPException(status_code=500, detail="No etalon images found in database")

    contents = await image.read()
    test_image = validate_image_file(contents)
    test_tensor = transform_image(test_image)

    with torch.no_grad():
        best_chance = 0.0
        best_etalon_name = None
        best_etalon_person = None
        best_etalon_id = None
        for etalon in etalons:
            etalon_image = validate_image_file(etalon["content"])
            etalon_tensor = transform_image(etalon_image)
            out1, out2 = model(etalon_tensor, test_tensor)
            distance = F.pairwise_distance(out1, out2).item()
            chance = dist_to_prob(distance, THRESHOLD)
            if chance > best_chance:
                best_chance = chance
                best_etalon_name = etalon["storage_filename"]
                best_etalon_person = etalon["full_name"]
                best_etalon_id = etalon["id"]

    return {
        "chance": round(best_chance, 4),
        "best_etalon": best_etalon_name,
        "best_etalon_person": best_etalon_person,
        "best_etalon_id": best_etalon_id,
    }


@app.get("/etalons")
def list_etalons() -> dict:
    rows = list_etalons_rows()
    items = [{"id": row["id"], "filename": row["storage_filename"], "full_name": row["full_name"]} for row in rows]
    return {"items": items, "count": len(items)}


@app.post("/etalons")
async def upload_etalon(file: UploadFile = File(...), fio: str = Form(...)) -> dict:
    contents = await file.read()
    validate_image_file(contents)
    image_id, storage_filename = build_storage_filename(fio, file.filename or "etalon")
    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    mime_type = file.content_type or "application/octet-stream"

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO images (id, kind, full_name, original_filename, storage_filename, mime_type, extension, content)
                    VALUES (%s, 'etalon', %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        image_id,
                        fio,
                        file.filename or "etalon",
                        storage_filename,
                        mime_type,
                        ext,
                        contents,
                    ),
                )
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save etalon to DB: {exc}")

    return {"ok": True, "id": image_id, "filename": storage_filename}


@app.post("/etalons/delete")
def delete_etalon(filename: str = Form(...)) -> dict:
    safe_name = os.path.basename(filename).strip()
    if safe_name == "":
        raise HTTPException(status_code=400, detail="Empty filename")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM images WHERE kind = 'etalon' AND storage_filename = %s", (safe_name,))
            deleted = cur.rowcount
        conn.commit()

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Etalon file not found")
    return {"ok": True, "filename": safe_name}
