"""Entrenamiento del modelo de producción (MobileNetV2 + cabeza de clasificación).

Decisiones:
- MobileNetV2 preentrenada en ImageNet con ``include_top=False`` y base congelada.
- Cabeza: GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid).
- Adam(lr=1e-4), binary crossentropy, métricas accuracy/precision/recall.
- Máximo 5 épocas, EarlyStopping(paciencia=2, restore_best_weights=True).
- Sin fine-tuning.
- El preprocesamiento de datos (resize 224x224 y normalización a [0,1]) lo aporta
  ``src.data_prep``; el modelo NO vuelve a normalizar, evitando la doble normalización.

Salidas:
- ``models/model.keras``
- ``reports/training_curves.png``
- ``reports/training_history.json``
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
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_prep  # noqa: E402

REPORTS_DIR = data_prep.REPORTS_DIR
MODELS_DIR = data_prep.MODELS_DIR
MODEL_PATH = MODELS_DIR / "model.keras"
HISTORY_PATH = REPORTS_DIR / "training_history.json"
CURVES_PATH = REPORTS_DIR / "training_curves.png"

EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = data_prep.BATCH_SIZE
IMG_SIZE = data_prep.IMG_SIZE
SEED = data_prep.SEED


def set_seeds() -> None:
    """Fija todas las semillas posibles para hacer el entrenamiento reproducible."""
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def build_mobilenetv2_base() -> keras.Model:
    """Carga MobileNetV2 preentrenada, con degradación elegante si no hay Internet."""
    try:
        base = keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        print("[train] Pesos ImageNet de MobileNetV2 cargados correctamente.")
    except Exception as exc:  # noqa: BLE001
        print(f"[train][AVISO] No se pudieron descargar los pesos ImageNet ({exc}).")
        print("[train][AVISO] Se entrena con pesos inicializados aleatoriamente.")
        base = keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights=None,
        )
    base.trainable = False
    return base


def build_model() -> keras.Model:
    """Construye el modelo de producción completo."""
    base = build_mobilenetv2_base()
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="entrada")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="densa_128")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="salida")(x)
    model = keras.Model(inputs, outputs, name="gato_perro_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def plot_curves(history: dict, path: Path) -> None:
    """Genera la figura con las curvas de pérdida y accuracy (train vs validación)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], "o-", label="Entrenamiento")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Validación")
    axes[0].set_title("Pérdida (binary crossentropy)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["accuracy"], "o-", label="Entrenamiento")
    axes[1].plot(epochs, history["val_accuracy"], "s-", label="Validación")
    axes[1].set_title("Exactitud (accuracy)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("MobileNetV2 - curvas de entrenamiento y validación")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[train] Curvas -> {path}")


def main() -> dict:
    set_seeds()
    print("[train] Construyendo particiones de datos...")
    manifest = data_prep.build_manifest(force=False)
    train_ds, val_ds, test_ds = data_prep.build_datasets(
        manifest=manifest, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=True
    )
    print(f"[train] Tamanos -> train={manifest['tamanos']['train']} "
          f"val={manifest['tamanos']['val']} test={manifest['tamanos']['test']}")

    model = build_model()
    model.summary()
    total_params = int(model.count_params())
    trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable_params = total_params - trainable_params

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    duration = time.perf_counter() - start

    model.save(MODEL_PATH)
    print(f"[train] Modelo guardado -> {MODEL_PATH}")

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(min(range(len(hist["val_loss"])), key=lambda i: hist["val_loss"][i])) + 1
    record = {
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "duracion_segundos": round(duration, 2),
        "epocas_ejecutadas": len(hist["loss"]),
        "epocas_maximas": EPOCHS,
        "mejor_epoca": best_epoch,
        "optimizador": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "binary_crossentropy",
        "batch_size": BATCH_SIZE,
        "img_size": list(IMG_SIZE),
        "semilla": SEED,
        "parametros_totales": total_params,
        "parametros_entrenables": trainable_params,
        "parametros_no_entrenables": non_trainable_params,
        "early_stopping": {"monitor": "val_loss", "paciencia": 2, "restore_best_weights": True},
        "historico": hist,
        "mejor_val_loss": float(min(hist["val_loss"])),
        "mejor_val_accuracy": float(max(hist["val_accuracy"])),
    }
    HISTORY_PATH.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_curves(hist, CURVES_PATH)

    print(f"[train] Duración total: {duration:.1f} s | mejor época: {best_epoch}")
    return record


if __name__ == "__main__":
    main()
