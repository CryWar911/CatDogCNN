# 🐱 vs 🐶 — Clasificador CNN de gatos y perros

Red neuronal convolucional entrenada desde cero con el dataset *PetImages* (25 000 fotos) que se ejecuta **en el navegador** con ONNX Runtime Web. La página permite subir una imagen y ver:

- La predicción (probabilidad gato / perro).
- El **mapa de atención (CAM)**: dónde se fijó la red.
- Los **mapas de activación** de cada bloque convolucional.
- El vector de *Global Average Pooling* y su contribución a la decisión.
- Los filtros aprendidos por la primera capa, la curva de entrenamiento y la matriz de confusión.

## Estructura

```
docs/               ← sitio web (GitHub Pages)
  index.html
  model/catdog_cnn.onnx   modelo exportado
  model/metrics.json      métricas, historial, filtros
  examples/               imágenes de prueba
train/
  prepare_data.py   limpia el dataset y lo cachea a 128×128
  train.py          entrena la CNN (PyTorch) y exporta a ONNX
```

## Arquitectura

| Capa | Salida |
|---|---|
| Entrada RGB | 3×128×128 |
| Bloque 1: (Conv3×3 → BN → ReLU) ×2 → MaxPool | 32×64×64 |
| Bloque 2 | 64×32×32 |
| Bloque 3 | 128×16×16 |
| Bloque 4 | 256×8×8 |
| Global Average Pooling → Dropout(0.3) | 256 |
| Densa + Softmax | 2 |

## Reentrenar

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install onnx onnxscript pillow numpy
# Coloca el dataset en PetImages/Cat y PetImages/Dog
.venv/Scripts/python train/prepare_data.py
.venv/Scripts/python train/train.py --epochs 30
```

## Publicar en GitHub Pages

1. Sube el repositorio a GitHub (el dataset está en `.gitignore`).
2. En **Settings → Pages**, elige *Deploy from a branch*, rama `main`, carpeta **`/docs`**.
3. La web quedará en `https://<usuario>.github.io/<repositorio>/`.

Para probarla en local (no funciona abriendo el archivo directamente por `file://`):

```bash
python -m http.server -d docs 8000
```
