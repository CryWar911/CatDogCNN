"""Evaluación del modelo de producción sobre el conjunto de prueba.

Genera:
- ``reports/metrics.json``: accuracy, precision, recall, F1, AUC y tiempo medio de inferencia.
- ``reports/confusion_matrix.png``
- ``reports/roc_curve.png``
- ``reports/training_curves.png`` (se regenera a partir del histórico real de entrenamiento)

No reentrena ni ajusta nada: solo mide el modelo guardado en ``models/model.keras``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_prep  # noqa: E402

REPORTS_DIR = data_prep.REPORTS_DIR
MODELS_DIR = data_prep.MODELS_DIR
MODEL_PATH = MODELS_DIR / "model.keras"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CM_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_PATH = REPORTS_DIR / "roc_curve.png"
CURVES_PATH = REPORTS_DIR / "training_curves.png"
HISTORY_PATH = REPORTS_DIR / "training_history.json"

OBJETIVO_ACCURACY = 0.92
OBJETIVO_LATENCIA_MS = 200.0
N_IMAGENES_LATENCIA = 100


def plot_training_curves_from_history() -> bool:
    """Regenera ``training_curves.png`` a partir del histórico real, si existe."""
    if not HISTORY_PATH.exists():
        print("[evaluate][AVISO] No hay training_history.json; no se regeneran las curvas.")
        return False
    record = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    hist = record.get("historico", {})
    if not hist:
        return False
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, hist["loss"], "o-", label="Entrenamiento")
    axes[0].plot(epochs, hist["val_loss"], "s-", label="Validación")
    axes[0].set_title("Pérdida (binary crossentropy)")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, hist["accuracy"], "o-", label="Entrenamiento")
    axes[1].plot(epochs, hist["val_accuracy"], "s-", label="Validación")
    axes[1].set_title("Exactitud (accuracy)")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("MobileNetV2 - curvas de entrenamiento y validación")
    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=120)
    plt.close(fig)
    print(f"[evaluate] Curvas de entrenamiento -> {CURVES_PATH}")
    return True


def measure_latency(model, test_ds, n_images: int = N_IMAGENES_LATENCIA) -> dict:
    """Mide el tiempo medio de inferencia por imagen (ms) sobre imágenes reales."""
    # Calentamiento (evita medir la inicialización de la primera llamada).
    warm = np.zeros((1, *data_prep.IMG_SIZE, 3), dtype="float32")
    for _ in range(3):
        model(warm, training=False)

    times = []
    count = 0
    for images, _labels in test_ds:
        for i in range(images.shape[0]):
            if count >= n_images:
                break
            x = tf.expand_dims(images[i], axis=0)
            t0 = time.perf_counter()
            _ = model(x, training=False)
            times.append((time.perf_counter() - t0) * 1000.0)
            count += 1
        if count >= n_images:
            break
    times = np.array(times, dtype="float64")
    return {
        "imagenes_medidas": int(len(times)),
        "latencia_media_ms": float(round(float(times.mean()), 2)),
        "latencia_mediana_ms": float(round(float(np.median(times)), 2)),
        "latencia_p95_ms": float(round(float(np.percentile(times, 95)), 2)),
        "latencia_min_ms": float(round(float(times.min()), 2)),
        "latencia_max_ms": float(round(float(times.max()), 2)),
    }


def main() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe {MODEL_PATH}. Ejecuta primero 'python -m src.train'."
        )

    print(f"[evaluate] Cargando modelo {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    manifest = data_prep.load_manifest()
    _train_ds, _val_ds, test_ds = data_prep.build_datasets(
        manifest=manifest, img_size=data_prep.IMG_SIZE,
        batch_size=data_prep.BATCH_SIZE, augment=False,
    )

    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_true.extend(labels.numpy().ravel().tolist())
        y_prob.extend(probs.tolist())
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = float(auc(fpr, tpr))
    cm = confusion_matrix(y_true, y_pred)

    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["gato", "perro"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["gato", "perro"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión - conjunto de prueba")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=120)
    plt.close(fig)
    print(f"[evaluate] Matriz de confusión -> {CM_PATH}")

    # Curva ROC
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC - conjunto de prueba")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=120)
    plt.close(fig)
    print(f"[evaluate] Curva ROC -> {ROC_PATH}")

    plot_training_curves_from_history()

    print("[evaluate] Midiendo latencia de inferencia...")
    latency = measure_latency(model, test_ds)

    cm_list = cm.tolist()
    metrics = {
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "modelo": str(MODEL_PATH.name),
        "conjunto": "test",
        "n_imagenes": int(len(y_true)),
        "positivo": "perro (1)",
        "negativo": "gato (0)",
        "umbral": 0.5,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc_value, 4),
        "matriz_confusion": {
            "valores": cm_list,
            "formato": "filas=real [gato,perro], columnas=predicción [gato,perro]",
        },
        "latencia": latency,
        "objetivos": {
            "accuracy_objetivo": OBJETIVO_ACCURACY,
            "accuracy_cumplido": bool(accuracy >= OBJETIVO_ACCURACY),
            "latencia_objetivo_ms": OBJETIVO_LATENCIA_MS,
            "latencia_cumplido": bool(latency["latencia_media_ms"] < OBJETIVO_LATENCIA_MS),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate] Métricas -> {METRICS_PATH}")
    print(f"[evaluate] accuracy={accuracy:.4f} precision={precision:.4f} "
          f"recall={recall:.4f} f1={f1:.4f} auc={auc_value:.4f}")
    print(f"[evaluate] latencia media={latency['latencia_media_ms']} ms "
          f"(objetivo < {OBJETIVO_LATENCIA_MS} ms: "
          f"{'CUMPLIDO' if metrics['objetivos']['latencia_cumplido'] else 'NO CUMPLIDO'})")
    print(f"[evaluate] accuracy objetivo {OBJETIVO_ACCURACY:.0%}: "
          f"{'CUMPLIDO' if metrics['objetivos']['accuracy_cumplido'] else 'NO CUMPLIDO'}")
    return metrics


if __name__ == "__main__":
    main()
