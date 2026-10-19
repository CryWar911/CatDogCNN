"""API FastAPI para clasificar imágenes de gatos y perros.

Endpoints:
- ``GET /``        -> interfaz web (HTML/CSS/JS).
- ``GET /health``  -> estado del servicio y del modelo.
- ``POST /predict``-> recibe una imagen multipart y devuelve clase y confianza.
- ``GET /stats``   -> estadísticas de monitoreo.

El modelo se carga UNA sola vez al iniciar la aplicación (lifespan).
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import monitoring  # noqa: E402
from src.data_prep import IMG_SIZE, preprocess_pil_image  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "model.keras"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# El modelo se carga una sola vez y se guarda en este diccionario global.
STATE: dict = {"model": None, "model_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar y lo libera al apagar."""
    try:
        import tensorflow as tf

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
        STATE["model"] = tf.keras.models.load_model(MODEL_PATH)
        STATE["model_error"] = None
        print(f"[api] Modelo cargado una sola vez desde {MODEL_PATH}")
    except Exception as exc:  # noqa: BLE001
        STATE["model"] = None
        STATE["model_error"] = str(exc)
        print(f"[api][ERROR] No se pudo cargar el modelo: {exc}")
    yield
    STATE["model"] = None


app = FastAPI(
    title="Clasificador de perros y gatos",
    description="API + interfaz web para clasificar imágenes de gatos y perros.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """Sirve la interfaz web."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="No se encontró la interfaz web.")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    """Estado del servicio y del modelo."""
    return {
        "status": "ok" if STATE["model"] is not None else "degradado",
        "modelo_cargado": STATE["model"] is not None,
        "modelo": MODEL_PATH.name,
        "ruta_modelo": str(MODEL_PATH),
        "error_modelo": STATE["model_error"],
        "hora": datetime.now().isoformat(timespec="seconds"),
    }


def _validate_upload(file: UploadFile, raw: bytes) -> None:
    """Valida extensión, tipo declarado, tamaño y contenido real de la imagen.

    Lanza ``HTTPException`` con códigos claros:
    - 413: archivo demasiado grande.
    - 415: extensión o tipo MIME no admitido.
    - 400: contenido que no es una imagen válida.
    """
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo supera el límite de 5 MB ({len(raw) / (1024 * 1024):.2f} MB).",
        )
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Extensión no permitida. Usa JPG, JPEG, PNG o WebP.",
        )

    declared = (file.content_type or "").lower()
    if declared not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo declarado no permitido: {declared or 'desconocido'}.",
        )

    # Comprobación del contenido real, independientemente de la extensión declarada.
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw)) as img2:
            img2.load()
            if img2.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail=f"Formato real no permitido: {img2.format}.",
                )
            if img2.width < 16 or img2.height < 16:
                raise HTTPException(status_code=400, detail="La imagen es demasiado pequeña.")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"El contenido del archivo no es una imagen válida: {exc}",
        ) from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Clasifica una imagen y devuelve ``{"clase", "confianza"}``."""
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"El modelo no está cargado: {STATE['model_error'] or 'error desconocido'}",
        )

    raw = await file.read()
    _validate_upload(file, raw)

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.") from exc

    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype="float32")
    brightness = float(gray.mean())  # 0-255

    # Preprocesamiento idéntico al de entrenamiento/evaluación: RGB, 224x224 y /255.
    arr = preprocess_pil_image(image, IMG_SIZE)
    batch = np.expand_dims(arr, axis=0).astype("float32")

    t0 = time.perf_counter()
    prob = float(STATE["model"].predict(batch, verbose=0).ravel()[0])
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # sigmoid >= 0.5 -> perro (1); en caso contrario gato (0).
    if prob >= 0.5:
        clase = "perro"
        confianza = prob
    else:
        clase = "gato"
        confianza = 1.0 - prob
    confianza = round(float(confianza), 4)
    incierta = 0.40 <= confianza <= 0.60

    try:
        monitoring.log_prediction(
            filename=file.filename or "imagen",
            clase=clase,
            confianza=confianza,
            tiempo_inferencia_ms=inference_ms,
            ancho=width,
            alto=height,
            brillo_medio=brightness,
            incierta=incierta,
            image_bytes=raw if incierta else None,
            extension=Path(file.filename or ".jpg").suffix.lower() or ".jpg",
        )
    except Exception as exc:  # noqa: BLE001 - el monitoreo nunca debe romper la predicción
        print(f"[api][AVISO] No se pudo registrar el monitoreo: {exc}")

    result = {"clase": clase, "confianza": confianza}
    if incierta:
        result["incierta"] = True
    return JSONResponse(content=result)


@app.get("/stats")
async def stats():
    """Estadísticas agregadas de las predicciones registradas."""
    return monitoring.get_stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
