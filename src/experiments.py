"""Experimentos académicos del Taller CNN.

Se implementa una CNN base independiente del modelo de producción, con:
- Convoluciones 3x3 + ReLU.
- MaxPooling.
- Salida Softmax de dos clases (gato / perro).
- Comparación de Dropout, de optimizadores (Adam frente a SGD)
  y del aumento de datos (sin aumento frente a rotación + zoom).

El modelo de producción sigue siendo MobileNetV2; esta CNN es solo para el estudio
comparativo exigido por el taller.

Para que los experimentos sean asequibles en CPU:
- Se reduce la resolución a 96x96.
- Se usa un subconjunto reproducible del conjunto de entrenamiento/validación
  (``MAX_TRAIN_POR_CLASE`` y ``MAX_VAL_POR_CLASE``).

Todos los resultados se guardan en ``reports/experiments.csv`` y el detalle por
época en ``reports/experiments_detail.json``. No se inventa ningún valor: el CSV se
construye exclusivamente a partir de los entrenamientos realmente ejecutados.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_prep  # noqa: E402

REPORTS_DIR = data_prep.REPORTS_DIR
EXP_CSV = REPORTS_DIR / "experiments.csv"
EXP_DETAIL = REPORTS_DIR / "experiments_detail.json"
EXP_PLOT = REPORTS_DIR / "experiments_comparison.png"

SEED = data_prep.SEED
IMG_SIZE_EXP = (96, 96)
BATCH_SIZE_EXP = 32
MAX_TRAIN_POR_CLASE = 1500
MAX_VAL_POR_CLASE = 300
EPOCHS_EXP = 5


def set_seeds() -> None:
    """Fija las semillas para que cada experimento sea reproducible."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def subsample_manifest(manifest: dict) -> dict:
    """Crea una copia del manifiesto con un subconjunto estratificado por clase."""
    rng = random.Random(SEED)
    new_splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    limits = {"train": MAX_TRAIN_POR_CLASE, "val": MAX_VAL_POR_CLASE, "test": 0}
    label_to_clase = {0: "Cat", 1: "Dog"}

    for split in ("train", "val"):
        for label, clase in label_to_clase.items():
            items = [it for it in manifest["splits"][split] if it["label"] == label]
            items = list(items)
            rng.shuffle(items)
            new_splits[split].extend(items[: limits[split]])
        rng.shuffle(new_splits[split])

    return {
        "semilla": SEED,
        "splits": new_splits,
        "tamanos": {s: len(v) for s, v in new_splits.items()},
        "nota": "Subconjunto reproducible usado SOLO en los experimentos académicos.",
    }


def build_academic_cnn(input_shape=(96, 96, 3), dropout: float | None = 0.5) -> keras.Model:
    """CNN académica: Conv3x3 + ReLU, MaxPooling y softmax de 2 clases."""
    model = keras.Sequential(name="cnn_academica")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    if dropout:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(2, activation="softmax"))
    return model


EXPERIMENTS = [
    {"nombre": "exp1_adam_dropout_aumento", "dropout": 0.5, "optimizador": "adam", "aumento": True},
    {"nombre": "exp2_adam_sin_dropout_aumento", "dropout": None, "optimizador": "adam", "aumento": True},
    {"nombre": "exp3_sgd_dropout_aumento", "dropout": 0.5, "optimizador": "sgd", "aumento": True},
    {"nombre": "exp4_adam_dropout_sin_aumento", "dropout": 0.5, "optimizador": "adam", "aumento": False},
    {"nombre": "exp5_sgd_sin_dropout_sin_aumento", "dropout": None, "optimizador": "sgd", "aumento": False},
]


def get_optimizer(name: str):
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=1e-3)
    return keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)


def run_experiment(cfg: dict, manifest: dict) -> dict:
    """Entrena una configuración y devuelve sus métricas reales."""
    print(f"\n[experiments] === {cfg['nombre']} ===")
    set_seeds()

    # Sin aumento -> dataset sin capas de aumento; con aumento -> rotación + zoom.
    aug_layers = data_prep.AUGMENTATION_LAYERS_ACADEMIC if cfg["aumento"] else None
    train_ds, val_ds, test_ds = data_prep.build_datasets(
        manifest=manifest,
        img_size=IMG_SIZE_EXP,
        batch_size=BATCH_SIZE_EXP,
        augment=cfg["aumento"],
        one_hot=True,
        augmentation_layers=aug_layers if cfg["aumento"] else None,
    )

    model = build_academic_cnn(input_shape=(*IMG_SIZE_EXP, 3), dropout=cfg["dropout"])
    model.compile(
        optimizer=get_optimizer(cfg["optimizador"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    n_params = int(model.count_params())

    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_EXP, verbose=2)
    duration = time.perf_counter() - start

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(min(range(len(hist["val_loss"])), key=lambda i: hist["val_loss"][i])) + 1

    return {
        "nombre": cfg["nombre"],
        "dropout": cfg["dropout"] if cfg["dropout"] else 0.0,
        "optimizador": cfg["optimizador"],
        "aumento": cfg["aumento"],
        "epocas": EPOCHS_EXP,
        "parametros": n_params,
        "duracion_segundos": round(duration, 2),
        "mejor_epoca": best_epoch,
        "train_accuracy_final": hist["accuracy"][-1],
        "val_accuracy_final": hist["val_accuracy"][-1],
        "mejor_val_accuracy": float(max(hist["val_accuracy"])),
        "train_loss_final": hist["loss"][-1],
        "val_loss_final": hist["val_loss"][-1],
        "mejor_val_loss": float(min(hist["val_loss"])),
        "img_size": IMG_SIZE_EXP[0],
        "historico": hist,
    }


def save_csv(rows: list[dict]) -> None:
    fields = [
        "nombre", "dropout", "optimizador", "aumento", "epocas", "parametros",
        "duracion_segundos", "mejor_epoca", "train_accuracy_final", "val_accuracy_final",
        "mejor_val_accuracy", "train_loss_final", "val_loss_final", "mejor_val_loss", "img_size",
    ]
    with EXP_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    print(f"[experiments] CSV -> {EXP_CSV}")


def plot_comparison(rows: list[dict]) -> None:
    names = [r["nombre"] for r in rows]
    train_acc = [r["train_accuracy_final"] for r in rows]
    val_acc = [r["val_accuracy_final"] for r in rows]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, train_acc, width, label="Train")
    ax.bar(x + width / 2, val_acc, width, label="Validación")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparación de experimentos (CNN académica, 96x96)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP_PLOT, dpi=120)
    plt.close(fig)
    print(f"[experiments] Gráfica comparativa -> {EXP_PLOT}")


def main() -> list[dict]:
    manifest_full = data_prep.build_manifest(force=False)
    manifest = subsample_manifest(manifest_full)
    print(f"[experiments] Subconjunto -> {manifest['tamanos']}")

    rows = []
    for cfg in EXPERIMENTS:
        try:
            rows.append(run_experiment(cfg, manifest))
        except Exception as exc:  # noqa: BLE001
            print(f"[experiments][ERROR] {cfg['nombre']} falló: {exc}")
            rows.append({
                "nombre": cfg["nombre"], "dropout": cfg["dropout"] or 0.0,
                "optimizador": cfg["optimizador"], "aumento": cfg["aumento"],
                "epocas": 0, "parametros": 0, "duracion_segundos": 0.0, "mejor_epoca": 0,
                "train_accuracy_final": None, "val_accuracy_final": None, "mejor_val_accuracy": None,
                "train_loss_final": None, "val_loss_final": None, "mejor_val_loss": None,
                "img_size": IMG_SIZE_EXP[0], "historico": {}, "error": str(exc),
            })

    save_csv(rows)
    EXP_DETAIL.write_text(
        json.dumps({
            "generado_en": datetime.now().isoformat(timespec="seconds"),
            "semilla": SEED,
            "img_size": IMG_SIZE_EXP[0],
            "max_train_por_clase": MAX_TRAIN_POR_CLASE,
            "max_val_por_clase": MAX_VAL_POR_CLASE,
            "epocas": EPOCHS_EXP,
            "experimentos": rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[experiments] Detalle -> {EXP_DETAIL}")
    try:
        plot_comparison([r for r in rows if r.get("val_accuracy_final") is not None])
    except Exception as exc:  # noqa: BLE001
        print(f"[experiments][AVISO] No se pudo generar la gráfica: {exc}")
    return rows


if __name__ == "__main__":
    main()
