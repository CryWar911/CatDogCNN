# 🐱 vs 🐶 — Clasificador CNN de gatos y perros

Red neuronal convolucional entrenada desde cero con el dataset *PetImages* (25 000 fotos) que se ejecuta **en el navegador** con ONNX Runtime Web. La página permite subir una imagen y ver:

- La predicción (probabilidad gato / perro).
- El **mapa de atención (CAM)**: dónde se fijó la red.
- Los **mapas de activación** de cada bloque convolucional.
- El vector de *Global Average Pooling* y su contribución a la decisión.
- Los filtros aprendidos por la primera capa, la curva de entrenamiento y la matriz de confusión.

**Resultado en prueba:** 96.80 % de accuracy (2 499 imágenes que la red nunca vio).

## Estructura del repositorio

```
CatDogCNN/
├── docs/                          ← sitio web publicado con GitHub Pages
│   ├── index.html                 página interactiva (clasificador y visualizaciones)
│   ├── model/
│   │   ├── catdog_cnn.onnx        modelo entrenado que usa la página
│   │   └── metrics.json           métricas, historial de entrenamiento y filtros
│   └── examples/                  12 fotos de prueba (gatos y perros) + index.json
│
├── train/                         ← código del modelo
│   ├── ModeloB_CNN_web.ipynb      notebook ejecutado: datos, modelo, entrenamiento,
│   │                              gráficas, matriz de confusión, accuracy y exportación
│   ├── prepare_data.py            script: valida y redimensiona las imágenes a 128×128
│   └── train.py                   script: entrena la CNN y la exporta a ONNX
│
├── Informe_CNN_tralaleros_corregido.docx   informe del taller
├── README.md                      este archivo
└── .gitignore                     archivos que NO se suben (ver abajo)
```

### Qué no se sube a GitHub

Estas carpetas están en `.gitignore` porque pesan mucho o se pueden regenerar:

| Carpeta / archivo | Motivo |
|---|---|
| `PetImages/` | dataset original (≈ 800 MB) |
| `gatosperros.zip` | el mismo dataset comprimido |
| `.venv/` | entorno de Python con PyTorch (varios GB) |
| `train/cache/` | caché de imágenes que crea `prepare_data.py` (1.2 GB) |
| `train/*.pt` | pesos de PyTorch (la web usa el `.onnx`) |

> Si subes los archivos arrastrándolos a la web de GitHub, el `.gitignore` no se aplica: sube solo `docs/`, los archivos de `train/` listados arriba, el informe, `README.md` y `.gitignore`.

## Arquitectura (Modelo B)

| Capa | Salida |
|---|---|
| Entrada RGB | 3×128×128 |
| Bloque 1: (Conv 3×3, stride 1, padding 1 → BN → ReLU) ×2 → MaxPool 2×2 | 32×64×64 |
| Bloque 2 | 64×32×32 |
| Bloque 3 | 128×16×16 |
| Bloque 4 | 256×8×8 |
| Global Average Pooling → Dropout(0.3) | 256 |
| Densa + Softmax | 2 |

1 173 730 parámetros. Entrenado 30 épocas con AdamW, One-Cycle, label smoothing y aumento de datos (rotación, zoom, volteo, brillo/contraste).

## Reentrenar

Coloca el dataset en `PetImages/Cat` y `PetImages/Dog` (junto a la carpeta `train/`) y crea el entorno:

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install onnx onnxscript pillow numpy matplotlib jupyter
```

Luego, una de dos opciones:

- **Notebook:** abre `train/ModeloB_CNN_web.ipynb` y ejecuta todas las celdas.
- **Scripts:**

  ```bash
  .venv/Scripts/python train/prepare_data.py
  .venv/Scripts/python train/train.py --epochs 30
  ```

Ambos sobrescriben `docs/model/` con el modelo nuevo.

## Publicar en GitHub Pages

1. Sube el repositorio a GitHub.
2. En **Settings → Pages**, elige *Deploy from a branch*, rama `main`, carpeta **`/docs`**.
3. La web quedará en `https://<usuario>.github.io/<repositorio>/`.

Para probarla en local (no funciona abriendo el archivo directamente por `file://`):

```bash
python -m http.server -d docs 8000
```
