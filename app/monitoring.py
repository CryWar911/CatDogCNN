"""Monitoreo de predicciones para la API gato/perro.

Registra cada predicción en ``reports/predictions_log.csv`` y guarda las imágenes
consideradas inciertas en ``reports/uncertain/`` con nombres únicos.

Campos registrados: fecha, nombre de archivo, clase, confianza, tiempo de inferencia,
ancho, alto y brillo medio. No se registran direcciones IP ni otros datos personales.
"""

from __future__ import annotations

import csv
import threading
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_PATH = REPORTS_DIR / "predictions_log.csv"
UNCERTAIN_DIR = REPORTS_DIR / "uncertain"

FIELDS = [
    "fecha",
    "nombre_archivo",
    "clase",
    "confianza",
    "tiempo_inferencia_ms",
    "ancho",
    "alto",
    "brillo_medio",
    "incierta",
]

_lock = threading.Lock()


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    UNCERTAIN_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_log() -> None:
    """Crea el CSV con cabecera si todavía no existe."""
    _ensure_dirs()
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()


def _safe_stem(filename: str) -> str:
    """Sanitiza el nombre de archivo para usarlo en disco (sin rutas ni caracteres raros)."""
    stem = Path(filename or "imagen").stem
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60]
    return cleaned or "imagen"


def log_prediction(
    filename: str,
    clase: str,
    confianza: float,
    tiempo_inferencia_ms: float,
    ancho: int,
    alto: int,
    brillo_medio: float,
    incierta: bool,
    image_bytes: bytes | None = None,
    extension: str = ".jpg",
) -> str | None:
    """Añade una fila al CSV y, si la predicción es incierta, guarda la imagen.

    Devuelve la ruta de la imagen guardada (o ``None`` si no se guardó).
    """
    _ensure_log()
    saved_path = None
    timestamp = datetime.now()
    with _lock:
        with LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow({
                "fecha": timestamp.isoformat(timespec="seconds"),
                "nombre_archivo": Path(filename or "imagen").name,
                "clase": clase,
                "confianza": f"{confianza:.4f}",
                "tiempo_inferencia_ms": f"{tiempo_inferencia_ms:.2f}",
                "ancho": int(ancho),
                "alto": int(alto),
                "brillo_medio": f"{brillo_medio:.2f}",
                "incierta": bool(incierta),
            })

        if incierta and image_bytes:
            ext = extension if extension.startswith(".") else f".{extension}"
            unique = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
            dest = UNCERTAIN_DIR / f"{unique}_{_safe_stem(filename)}{ext}"
            dest.write_bytes(image_bytes)
            saved_path = str(dest.relative_to(PROJECT_ROOT))
    return saved_path


def get_stats() -> dict:
    """Calcula las estadísticas agregadas del monitoreo a partir del CSV real."""
    _ensure_log()
    total = 0
    gatos = 0
    perros = 0
    sum_conf = 0.0
    sum_lat = 0.0
    inciertas = 0

    with _lock:
        with LOG_PATH.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    conf = float(row.get("confianza", 0) or 0)
                    lat = float(row.get("tiempo_inferencia_ms", 0) or 0)
                except (TypeError, ValueError):
                    continue
                total += 1
                if row.get("clase") == "perro":
                    perros += 1
                elif row.get("clase") == "gato":
                    gatos += 1
                sum_conf += conf
                sum_lat += lat
                if str(row.get("incierta", "")).strip().lower() in {"true", "1", "si", "sí"}:
                    inciertas += 1

    if total == 0:
        return {
            "total_predicciones": 0,
            "proporcion_gato": 0.0,
            "proporcion_perro": 0.0,
            "confianza_media": 0.0,
            "latencia_media_ms": 0.0,
            "porcentaje_inciertas": 0.0,
            "conteo": {"gato": 0, "perro": 0},
        }

    return {
        "total_predicciones": total,
        "proporcion_gato": round(gatos / total, 4),
        "proporcion_perro": round(perros / total, 4),
        "confianza_media": round(sum_conf / total, 4),
        "latencia_media_ms": round(sum_lat / total, 2),
        "porcentaje_inciertas": round(100.0 * inciertas / total, 2),
        "conteo": {"gato": gatos, "perro": perros},
        "imagenes_inciertas_guardadas": len(list(UNCERTAIN_DIR.glob("*"))) if UNCERTAIN_DIR.exists() else 0,
    }
