"""Limpia el dataset PetImages y lo cachea como arrays uint8 de 128x128.

El dataset de Microsoft (Kaggle "Cats and Dogs") trae algunas imagenes corruptas
o que no son JPEG reales; se descartan aqui.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

IMG_SIZE = 128
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "PetImages")
CACHE = os.path.join(os.path.dirname(__file__), "cache")
CLASSES = ["Cat", "Dog"]  # 0 = gato, 1 = perro


def load(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            if min(im.size) < 32:
                return None
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def main():
    os.makedirs(CACHE, exist_ok=True)
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".jpg"):
                paths.append(os.path.join(folder, name))
                labels.append(label)

    with Pool() as pool:
        arrays = pool.map(load, paths, chunksize=64)

    keep = [i for i, a in enumerate(arrays) if a is not None]
    print(f"Imagenes validas: {len(keep)} / {len(paths)} (descartadas {len(paths) - len(keep)})")
    x = np.stack([arrays[i] for i in keep])
    y = np.array([labels[i] for i in keep], dtype=np.int64)
    np.save(os.path.join(CACHE, "x.npy"), x)
    np.save(os.path.join(CACHE, "y.npy"), y)
    print("Guardado:", x.shape, "gatos:", int((y == 0).sum()), "perros:", int((y == 1).sum()))


if __name__ == "__main__":
    sys.exit(main())
