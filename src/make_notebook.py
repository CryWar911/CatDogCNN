"""Genera ``notebooks/taller_cnn.ipynb`` a partir de los artefactos reales.

El notebook NO reentrena: carga los resultados ya generados en ``reports/`` para
que su ejecución sea rápida y sus salidas coincidan con los archivos reales.
Las conclusiones se construyen dinámicamente con las métricas obtenidas.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUT_PATH = NOTEBOOKS_DIR / "taller_cnn.ipynb"


def _md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def _code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def _load_json(name: str, default=None):
    path = REPORTS_DIR / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_conclusions() -> str:
    metrics = _load_json("metrics.json", {}) or {}
    history = _load_json("training_history.json", {}) or {}
    quality = _load_json("data_quality.json", {}) or {}
    lines = ["## 7. Análisis y conclusiones (generadas con los resultados reales)\n"]

    if metrics:
        obj = metrics.get("objetivos", {})
        lines.append(
            f"- **Resultado en test:** accuracy = {metrics['accuracy']:.4f}, "
            f"precision = {metrics['precision']:.4f}, recall = {metrics['recall']:.4f}, "
            f"F1 = {metrics['f1']:.4f}, AUC = {metrics['auc']:.4f}."
        )
        lines.append(
            f"- **Objetivo de accuracy ≥ 92 %:** "
            f"{'CUMPLIDO' if obj.get('accuracy_cumplido') else 'NO CUMPLIDO'}."
        )
        lines.append(
            f"- **Latencia media:** {metrics['latencia']['latencia_media_ms']} ms "
            f"(objetivo < 200 ms: "
            f"{'CUMPLIDO' if obj.get('latencia_cumplido') else 'NO CUMPLIDO'})."
        )
        cm = metrics["matriz_confusion"]["valores"]
        tn, fp = cm[0]
        fn, tp = cm[1]
        lines.append(
            f"- **Matriz de confusión:** VP={tp}, VN={tn}, FP={fp}, FN={fn}. "
            f"Se observan {fp + fn} errores sobre {tn + fp + fn + tp} imágenes de prueba."
        )
    else:
        lines.append("- `reports/metrics.json` NO DISPONIBLE: no se pudo evaluar el modelo.")

    if history:
        hist = history["historico"]
        gap = hist["accuracy"][-1] - hist["val_accuracy"][-1]
        lines.append(
            f"- **Entrenamiento:** {history['epocas_ejecutadas']} épocas, "
            f"{history['duracion_segundos']} s, mejor época = {history['mejor_epoca']} "
            f"(val_loss = {history['mejor_val_loss']:.4f})."
        )
        lines.append(
            f"- **Brecha train–validación (última época):** {gap:+.4f}. "
            + ("No se aprecia un sobreajuste severo con los datos observados."
               if abs(gap) < 0.05 else
               "La brecha es apreciable; conviene revisar las curvas antes de concluir.")
        )
    else:
        lines.append("- `reports/training_history.json` NO DISPONIBLE.")

    if quality:
        tot = quality.get("totales", {})
        lines.append(
            f"- **Calidad de datos:** {tot.get('validas', 'NO DISPONIBLE')} imágenes válidas "
            f"de {tot.get('examinadas', 'NO DISPONIBLE')} examinadas "
            f"({tot.get('invalidas', 'NO DISPONIBLE')} inválidas descartadas lógicamente)."
        )

    lines.append(
        "\n> Estas conclusiones se generaron automáticamente a partir de los archivos "
        "`metrics.json`, `training_history.json` y `data_quality.json`. "
        "No se han introducido valores manuales."
    )
    return "\n".join(lines)


def main() -> Path:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }

    cells = [
        _md("""
# Taller CNN: clasificación de imágenes de perros y gatos

**Objetivo:** construir y comparar modelos de visión por computador que distingan
gatos de perros, evaluar el efecto de Dropout, del optimizador (Adam vs. SGD) y del
aumento de datos, y documentar los resultados reales.

- **Clases:** gato = 0, perro = 1.
- **Dataset:** `PetImages/` (Cat/Dog).
- **Modelo de producción:** MobileNetV2 preentrenada + cabeza densa (ver `src/train.py`).
- **CNN académica:** convoluciones 3×3 + ReLU, MaxPooling y softmax de 2 clases.

> Este notebook carga los artefactos generados por los scripts. No inventa métricas:
> todas las cifras provienen de `reports/*.json` y `reports/experiments.csv`.
        """),
        _md("## 1. Entorno y configuración"),
        _code("""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

REPORTS = ROOT / "reports"

import tensorflow as tf
import sklearn
import PIL
print("TensorFlow:", tf.__version__)
print("scikit-learn:", sklearn.__version__)
print("Pillow:", PIL.__version__)
print("GPU disponible:", bool(tf.config.list_physical_devices("GPU")))
print("Raíz del proyecto:", ROOT)
        """),
        _md("## 2. Calidad y partición de los datos\n\nLa validación se hizo con "
            "`PIL.Image.open()` + `verify()` + `load()`; los archivos inválidos se "
            "descartaron lógicamente sin modificar `PetImages/`."),
        _code("""
quality = json.loads((REPORTS / "data_quality.json").read_text(encoding="utf-8"))
rows = []
for clase, datos in quality["clases"].items():
    rows.append({"Clase": clase, **datos})
display(pd.DataFrame(rows))
print("Totales:", quality["totales"])
manifest = json.loads((REPORTS / "manifest.json").read_text(encoding="utf-8"))
print("Semilla:", manifest["semilla"])
print("Tamaños de las particiones:", manifest["tamanos"])
print("Tamaños por clase:", manifest["tamanos_por_clase"])
        """),
        _md("## 3. Arquitectura del modelo de producción\n\n"
            "`MobileNetV2(include_top=False, weights='imagenet')` congelada, seguida de "
            "`GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid)`."),
        _code("""
modelo = tf.keras.models.load_model(ROOT / "models" / "model.keras")
modelo.summary()
print("Parámetros totales:", modelo.count_params())
        """),
        _md("## 4. Curvas de entrenamiento y validación"),
        _code("""
historial = json.loads((REPORTS / "training_history.json").read_text(encoding="utf-8"))
h = historial["historico"]
df_hist = pd.DataFrame(h)
df_hist.index = np.arange(1, len(df_hist) + 1)
df_hist.index.name = "época"
display(df_hist.round(4))
print("Duración (s):", historial["duracion_segundos"])
print("Mejor época:", historial["mejor_epoca"])
display(Image(filename=str(REPORTS / "training_curves.png")))
        """),
        _md("## 5. Experimentos de la CNN académica\n\n"
            "Se comparan Dropout vs. sin Dropout, Adam vs. SGD y aumento "
            "(rotación + zoom) vs. sin aumento. Los experimentos usan 96×96 y un "
            "subconjunto reproducible por límites de cómputo."),
        _code("""
df_exp = pd.read_csv(REPORTS / "experiments.csv")
display(df_exp)
        """),
        _code("""
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(df_exp))
ax.bar(x - 0.2, df_exp["train_accuracy_final"], width=0.4, label="Train")
ax.bar(x + 0.2, df_exp["val_accuracy_final"], width=0.4, label="Validación")
ax.set_xticks(x)
ax.set_xticklabels(df_exp["nombre"], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Comparación de experimentos")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
        """),
        _md("## 6. Evaluación final sobre el conjunto de prueba"),
        _code("""
metricas = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
cm = np.array(metricas["matriz_confusion"]["valores"])
tn, fp = cm[0]
fn, tp = cm[1]
total = cm.sum()
accuracy_calculada = (tp + tn) / total

print(f"Accuracy reportada : {metricas['accuracy']:.4f}")
print(f"Accuracy calculada : {accuracy_calculada:.4f}")
print(f"Precision: {metricas['precision']:.4f} | Recall: {metricas['recall']:.4f} | "
      f"F1: {metricas['f1']:.4f} | AUC: {metricas['auc']:.4f}")
print(f"VP={tp}, VN={tn}, FP={fp}, FN={fn}, total={total}")
print("Latencia media (ms):", metricas["latencia"]["latencia_media_ms"])
print("Objetivos:", metricas["objetivos"])
        """),
        _code("""
display(Image(filename=str(REPORTS / "confusion_matrix.png")))
display(Image(filename=str(REPORTS / "roc_curve.png")))
        """),
        _md(build_conclusions()),
    ]

    nb["cells"] = cells
    nbf.write(nb, str(OUT_PATH))
    print(f"[make_notebook] Notebook generado -> {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
