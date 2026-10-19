"""Preparación, validación y partición del dataset de perros y gatos.

Este módulo cumple varias funciones:

1. Validar con PIL todas las imágenes del dataset (``Image.open()`` + ``verify()``).
   Los archivos inválidos se descartan *lógicamente*: nunca se borran del disco.
2. Guardar los conteos de calidad en ``reports/data_quality.json``.
3. Seleccionar de forma aleatoria y reproducible hasta 3000 imágenes válidas por clase.
4. Construir una partición estratificada 80/10/10 (train/val/test) sin fugas de datos
   y guardarla en ``reports/manifest.json``.
5. Construir objetos ``tf.data.Dataset`` con resize 224x224x3, normalización a [0,1],
   aumento de datos solo en entrenamiento, ``cache``, ``batch(32)`` y ``prefetch``.

Etiquetas: gato = 0, perro = 1.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------------------
# Rutas y constantes globales
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "PetImages"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_DIRS = {"Cat": 0, "Dog": 1}  # gato = 0, perro = 1
CLASS_NAMES = {0: "gato", 1: "perro"}

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_PER_CLASS = 3000
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE

# Capas de aumento de datos (se instancian una sola vez).
AUGMENTATION_LAYERS = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(20.0 / 360.0),  # rotación de hasta 20 grados
        tf.keras.layers.RandomFlip("horizontal"),      # volteo horizontal
        tf.keras.layers.RandomTranslation(0.1, 0.1),   # desplazamiento horizontal/vertical
        tf.keras.layers.RandomZoom(0.1),               # zoom
    ],
    name="data_augmentation",
)

# Aumento reducido exigido por los experimentos académicos: rotación + zoom.
AUGMENTATION_LAYERS_ACADEMIC = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(20.0 / 360.0),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="academic_augmentation",
)


# --------------------------------------------------------------------------------------
# Validación de imágenes
# --------------------------------------------------------------------------------------
def is_valid_image(path: Path) -> tuple[bool, str]:
    """Comprueba con PIL que el archivo sea una imagen legible.

    Devuelve ``(es_valida, motivo)``. No lanza excepciones: cualquier fallo se
    traduce en un motivo de descarte.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # verifica la integridad del archivo
        # ``verify()`` invalida el objeto, por eso se reabre y se fuerza la carga
        # completa de los píxeles para detectar archivos truncados.
        with Image.open(path) as img2:
            img2.load()
            if img2.width < 16 or img2.height < 16:
                return False, "imagen demasiado pequena"
        return True, ""
    except Exception as exc:  # noqa: BLE001 - se guarda el motivo real
        return False, f"{type(exc).__name__}: {exc}"


def validate_dataset(force: bool = False) -> dict:
    """Recorre el dataset, valida cada archivo y escribe ``reports/data_quality.json``.

    Devuelve un diccionario ``{"valid": {clase: [rutas]}, "report": {...}}``.
    Si ya existe el informe y ``force`` es falso, se reutiliza el informe guardado
    para no revalidar 25.000 imágenes en cada ejecución.
    """
    report_path = REPORTS_DIR / "data_quality.json"
    valid_lists_path = REPORTS_DIR / "valid_files.json"

    if not force and report_path.exists() and valid_lists_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        valid_lists = json.loads(valid_lists_path.read_text(encoding="utf-8"))
        valid_lists = {cls: [Path(p) for p in paths] for cls, paths in valid_lists.items()}
        print("[data_prep] Reutilizando validacion previa (reports/data_quality.json).")
        return {"valid": valid_lists, "report": report}

    print("[data_prep] Validando imagenes con PIL (puede tardar unos minutos)...")
    valid_lists: dict[str, list[Path]] = {}
    report: dict = {
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(DATASET_DIR),
        "criterio": "Image.open() + verify() + load() + min 16x16 px",
        "clases": {},
        "totales": {},
        "archivos_invalidos": [],
    }
    total_exam = total_valid = total_invalid = 0

    for class_name, _label in CLASS_DIRS.items():
        class_dir = DATASET_DIR / class_name
        files = sorted([p for p in class_dir.iterdir() if p.is_file()])
        valid, invalid = [], []
        for path in files:
            ok, reason = is_valid_image(path)
            if ok:
                valid.append(path)
            else:
                invalid.append({"archivo": str(path.relative_to(PROJECT_ROOT)), "motivo": reason})
        valid_lists[class_name] = valid
        report["clases"][class_name] = {
            "examinadas": len(files),
            "validas": len(valid),
            "invalidas": len(invalid),
        }
        report["archivos_invalidos"].extend(invalid)
        total_exam += len(files)
        total_valid += len(valid)
        total_invalid += len(invalid)
        print(f"[data_prep] {class_name}: {len(valid)} validas / {len(files)} examinadas")

    report["totales"] = {
        "examinadas": total_exam,
        "validas": total_valid,
        "invalidas": total_invalid,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    valid_lists_path.write_text(
        json.dumps({cls: [str(p) for p in paths] for cls, paths in valid_lists.items()},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[data_prep] Informe de calidad -> {report_path}")
    return {"valid": valid_lists, "report": report}


# --------------------------------------------------------------------------------------
# Partición estratificada
# --------------------------------------------------------------------------------------
def build_manifest(force: bool = False) -> dict:
    """Crea (o reutiliza) la partición estratificada 80/10/10 y la guarda en JSON."""
    manifest_path = REPORTS_DIR / "manifest.json"
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("[data_prep] Reutilizando particion previa (reports/manifest.json).")
        return manifest

    validation = validate_dataset(force=False)
    rng = random.Random(SEED)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    sizes: dict[str, dict] = {}

    for class_name, label in CLASS_DIRS.items():
        valid = list(validation["valid"][class_name])
        rng.shuffle(valid)
        selected = valid[:MAX_PER_CLASS]
        n = len(selected)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train = selected[:n_train]
        val = selected[n_train:n_train + n_val]
        test = selected[n_train + n_val:]
        for split_name, paths in (("train", train), ("val", val), ("test", test)):
            for p in paths:
                splits[split_name].append(
                    {"path": str(p), "label": label, "clase": class_name}
                )
        sizes[class_name] = {"seleccionadas": n, "train": len(train),
                             "val": len(val), "test": len(test)}

    # Se mezclan las clases dentro de cada partición (sin mezclar particiones entre sí).
    train_rng = random.Random(SEED)
    for split_name, items in splits.items():
        train_rng.shuffle(items)

    manifest = {
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "semilla": SEED,
        "max_por_clase": MAX_PER_CLASS,
        "proporciones": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "etiquetas": {"gato": 0, "perro": 1},
        "tamanos": {s: len(v) for s, v in splits.items()},
        "tamanos_por_clase": sizes,
        "splits": splits,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[data_prep] Particion -> {manifest_path}  tamanos={manifest['tamanos']}")
    return manifest


def load_manifest() -> dict:
    """Carga el manifiesto existente o lo construye si no existe."""
    manifest_path = REPORTS_DIR / "manifest.json"
    if not manifest_path.exists():
        return build_manifest(force=False)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Pipeline tf.data
# --------------------------------------------------------------------------------------
def _pil_load_resize(path, img_size):
    """Decodifica con PIL y convierte SIEMPRE a RGB.

    Se usa PIL también en el pipeline (no ``tf.io.decode_image``) porque el dataset
    contiene imágenes válidas con 2 canales (p. ej. escala de grises + alfa) que el
    decodificador de TensorFlow rechaza. Así el entrenamiento y la inferencia usan
    exactamente la misma conversión a RGB.
    """
    if hasattr(path, "numpy"):
        path = path.numpy()
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    with Image.open(path) as im:
        im = im.convert("RGB").resize((int(img_size[1]), int(img_size[0])), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _decode_and_resize(path, label, img_size):
    """Lee y redimensiona una imagen. Devuelve uint8 para poder cachear."""
    img = tf.py_function(
        func=lambda p: _pil_load_resize(p, img_size),
        inp=[path],
        Tout=tf.uint8,
    )
    img.set_shape([int(img_size[0]), int(img_size[1]), 3])
    return img, label


def _apply_augmentation(img, label, layers):
    img = tf.cast(img, tf.float32)
    img = layers(img, training=True)
    return img, label


def _normalize(img, label):
    """Normaliza a [0,1]. Es la ÚNICA normalización aplicada (no se duplica)."""
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _to_one_hot(img, label):
    # Se fuerza int32 porque algunas transformaciones del pipeline pueden
    # propagar la etiqueta como float32.
    return img, tf.one_hot(tf.cast(label, tf.int32), depth=2)


def build_datasets(
    manifest: dict | None = None,
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    augment: bool = True,
    one_hot: bool = False,
    use_cache: bool = True,
    augmentation_layers=None,
):
    """Devuelve ``(train_ds, val_ds, test_ds)`` listos para Keras.

    - Aumento de datos SOLO en el conjunto de entrenamiento.
    - ``cache`` + ``batch`` + ``prefetch``.
    - La normalización se aplica después del aumento para no duplicarla.
    - ``one_hot=True`` genera etiquetas categóricas (para la CNN académica con softmax).
    """
    if manifest is None:
        manifest = load_manifest()
    if augmentation_layers is None:
        augmentation_layers = AUGMENTATION_LAYERS

    splits = manifest["splits"]

    def make_split(split_name: str, training: bool):
        items = splits[split_name]
        paths = [it["path"] for it in items]
        labels = [it["label"] for it in items]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(
            lambda p, l: _decode_and_resize(p, l, img_size),
            num_parallel_calls=AUTOTUNE,
        )
        if use_cache:
            ds = ds.cache()
        if training:
            ds = ds.shuffle(
                buffer_size=min(len(items), 10000), seed=SEED,
                reshuffle_each_iteration=True,
            )
            if augment:
                ds = ds.map(
                    lambda i, l: _apply_augmentation(i, l, augmentation_layers),
                    num_parallel_calls=AUTOTUNE,
                )
        if one_hot:
            ds = ds.map(_to_one_hot, num_parallel_calls=AUTOTUNE)
        ds = ds.map(_normalize, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    return (
        make_split("train", True),
        make_split("val", False),
        make_split("test", False),
    )


def preprocess_pil_image(image: Image.Image, img_size: tuple[int, int] = IMG_SIZE):
    """Preprocesamiento para inferencia, idéntico al de evaluación.

    Convierte a RGB, redimensiona y normaliza a [0,1] (una sola vez).
    """
    import numpy as np

    img = image.convert("RGB").resize(img_size, Image.BILINEAR)
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr


if __name__ == "__main__":
    val = validate_dataset(force=False)
    man = build_manifest(force=False)
    print(json.dumps(man["tamanos"], indent=2, ensure_ascii=False))
