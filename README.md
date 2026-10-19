# Clasificador de imágenes de perros y gatos

Proyecto completo de *Machine Learning* y *full-stack* que clasifica imágenes de
perros y gatos. Incluye preparación y validación de datos, un modelo de producción
basado en **MobileNetV2**, una **CNN académica** para experimentos comparativos,
evaluación sobre el conjunto de prueba, una **API FastAPI** con interfaz web y
monitoreo de predicciones.

> Las métricas, tiempos y resultados deben consultarse en los archivos generados
> dentro de `reports/`. Este README no inventa resultados.

---

## 1. Estructura

```
├── app/
│   ├── main.py                 # API FastAPI y carga única del modelo
│   ├── monitoring.py           # Registro CSV, imágenes inciertas y /stats
│   └── static/
│       ├── index.html          # Interfaz en español
│       ├── style.css
│       └── app.js
├── src/
│   ├── data_prep.py            # Validación PIL, partición 80/10/10 y tf.data
│   ├── train.py                # MobileNetV2 (modelo de producción)
│   ├── experiments.py          # CNN académica y experimentos
│   ├── evaluate.py             # Métricas de prueba, matriz, ROC, latencia
│   ├── make_notebook.py        # Genera notebooks/taller_cnn.ipynb
│   └── make_pdf.py             # Genera reports/informe_cnn.pdf
├── notebooks/taller_cnn.ipynb  # Notebook ejecutado con salidas
├── models/model.keras          # Modelo de producción entrenado
├── reports/                    # Métricas, gráficas, CSV, PDF y TXT
├── PetImages/                  # Dataset original (NO se modifica)
├── requirements.txt
└── README.md
```

## 2. Requisitos

- Python 3.10–3.12 (TensorFlow no publica ruedas para todas las versiones de 3.13/3.14).
- ~2 GB de espacio libre y varios GB de RAM.
- No se requiere GPU: el proyecto funciona en CPU.

## 3. Instalación

```powershell
# Desde la raíz del proyecto
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si TensorFlow no encuentra los pesos de ImageNet por falta de Internet, `src/train.py`
lo avisa y entrena la base con pesos aleatorios (el resto del flujo continúa).

## 4. Flujo de ejecución

```powershell
# 1) Validación + partición estratificada (no borra nada del dataset)
python -m src.data_prep

# 2) Entrenamiento de MobileNetV2 (máx. 5 épocas)
python -m src.train

# 3) Experimentos de la CNN académica
python -m src.experiments

# 4) Evaluación sobre test
python -m src.evaluate

# 5) Notebook ejecutado
python -m src.make_notebook
jupyter nbconvert --to notebook --execute --inplace notebooks/taller_cnn.ipynb

# 6) Informe PDF
python -m src.make_pdf

# 7) API + web
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abrir <http://127.0.0.1:8000/>.

## 5. Preparación de datos

- Se validan **todas** las imágenes con `PIL.Image.open()` + `verify()` + `load()`.
- Los archivos inválidos se descartan **lógicamente** (no se borran).
- Conteos en `reports/data_quality.json`.
- Semilla fija `42`; hasta **3000 imágenes válidas por clase**.
- Partición **estratificada 80/10/10** en `reports/manifest.json` (sin solapamientos).
- Etiquetas: **gato = 0**, **perro = 1**.
- Redimensionado **224×224×3**, normalización a **[0,1]** una sola vez.
- Aumento de datos **solo en entrenamiento**: rotación ≤ 20°, volteo horizontal,
  desplazamiento horizontal/vertical y zoom.
- Pipeline `tf.data` con `cache()`, `batch(32)` y `prefetch()`.

## 6. Modelo de producción

- `MobileNetV2(weights="imagenet", include_top=False)`, base congelada.
- Cabeza: `GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid)`.
- Adam (`lr=1e-4`), `binary_crossentropy`, métricas accuracy/precision/recall.
- Máximo 5 épocas; `EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)`.
- **Sin fine-tuning**.
- La normalización se hace en el pipeline de datos; el modelo no vuelve a normalizar.
  En inferencia se aplica exactamente el mismo preprocesamiento (RGB, 224×224, `/255`).

## 7. Experimentos académicos

CNN independiente con `Conv2D(3×3)+ReLU` + `MaxPooling`, `Flatten`,
`Dense(64, ReLU)`, `Dropout` opcional y `Dense(2, softmax)`.
Se comparan 5 configuraciones: Dropout vs. sin Dropout, Adam vs. SGD y
aumento (rotación + zoom) vs. sin aumento. Resultados en `reports/experiments.csv`.

## 8. API

| Método | Ruta      | Descripción |
|--------|-----------|-------------|
| GET    | `/`       | Interfaz web |
| GET    | `/health` | Estado del servicio y del modelo |
| POST   | `/predict`| Imagen multipart (campo `file`) |
| GET    | `/stats`  | Estadísticas de monitoreo |

- Formatos: JPG, JPEG, PNG y WebP. Límite: **5 MB**.
- Se valida extensión, tipo MIME declarado y contenido real con PIL.
- Códigos: `400` imagen inválida, `413` demasiado grande, `415` formato no admitido,
  `503` modelo no disponible.
- Respuesta: `{"clase":"perro","confianza":0.97}` y `"incierta":true` cuando
  la confianza está entre 0.40 y 0.60.
- CORS habilitado y modelo cargado una sola vez al iniciar.

## 9. Monitoreo

`reports/predictions_log.csv` guarda fecha, archivo, clase, confianza, latencia,
ancho, alto y brillo medio. Las predicciones inciertas se copian a
`reports/uncertain/` con nombre único. `/stats` agrega total, proporciones,
confianza media, latencia media y porcentaje de inciertas.

## 10. Limitaciones

- Dataset con posibles imágenes mal etiquetadas o corruptas (se filtran las corruptas).
- Entrenamiento acotado a 5 épocas y 3000 imágenes por clase.
- Los experimentos académicos usan 96×96 y un subconjunto por recursos.
- Sin GPU disponible: los tiempos dependen de la CPU.
- No se implementa Docker, Grad-CAM, pytest, fine-tuning ni detección de drift.
- Las métricas reales están en `reports/metrics.json`, `reports/experiments.csv`
  y `reports/informe_completo.txt`.
