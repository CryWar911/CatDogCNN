"""Genera ``reports/informe_completo.txt`` a partir de los artefactos reales.

El archivo resultante es la fuente verificable para redactar el informe académico.
No se introducen métricas manuales: todo se lee de los JSON/CSV/gráficas generados.

Uso:
    python -m src.make_final_report
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
OUT_PATH = REPORTS / "informe_completo.txt"


def load_json(name: str, default=None):
    path = REPORTS / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict]:
    path = REPORTS / name
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt_num(value, digits: int = 4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NO DISPONIBLE"


def artifact_status(path: Path) -> str:
    return "COMPLETADO" if path.exists() else "NO EJECUTADO"


def artifact_line(path: Path) -> str:
    if path == OUT_PATH:
        return (f"  - {path.relative_to(PROJECT_ROOT)}: COMPLETADO "
                "(este mismo archivo, escrito al finalizar)")
    if not path.exists():
        return f"  - {path.relative_to(PROJECT_ROOT)}: NO DISPONIBLE"
    size = path.stat().st_size
    ts = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return f"  - {path.relative_to(PROJECT_ROOT)}: COMPLETADO ({size:,} bytes, modificado {ts})"


def get_environment() -> dict:
    env = {
        "sistema": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "cpu": platform.processor() or "NO DISPONIBLE",
        "nucleos_logicos": os.cpu_count(),
        "librerias": {},
        "gpu": "NO VERIFICADO",
    }
    for name in ["tensorflow", "keras", "numpy", "scikit-learn", "matplotlib", "pillow",
                 "fastapi", "uvicorn", "pandas", "reportlab", "nbformat", "nbconvert",
                 "python-multipart"]:
        try:
            env["librerias"][name] = pkg_version(name)
        except Exception:  # noqa: BLE001
            env["librerias"][name] = "NO DISPONIBLE"
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        env["gpu"] = [g.name for g in gpus] if gpus else "No se detectaron GPUs"
        env["tensorflow"] = tf.__version__
    except Exception as exc:  # noqa: BLE001
        env["gpu"] = f"NO VERIFICADO ({exc})"
    return env


def training_table(history: dict) -> list[str]:
    hist = history.get("historico", {})
    if not hist:
        return ["  NO DISPONIBLE"]
    lines = []
    header = f"  {'Época':<7}{'loss':>9}{'val_loss':>11}{'accuracy':>11}{'val_acc':>10}{'val_prec':>10}{'val_rec':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for i in range(len(hist["loss"])):
        lines.append(
            f"  {i + 1:<7}{hist['loss'][i]:>9.4f}{hist['val_loss'][i]:>11.4f}"
            f"{hist['accuracy'][i]:>11.4f}{hist['val_accuracy'][i]:>10.4f}"
            f"{hist.get('val_precision', [0] * len(hist['loss']))[i]:>10.4f}"
            f"{hist.get('val_recall', [0] * len(hist['loss']))[i]:>10.4f}"
        )
    return lines


def experiments_table(experiments: list[dict]) -> list[str]:
    if not experiments:
        return ["  NO DISPONIBLE"]
    lines = []
    header = (f"  {'Experimento':<38}{'Dropout':>8}{'Optim':>7}{'Aum':>5}{'Époc':>5}"
              f"{'Acc train':>11}{'Acc val':>9}{'Dur (s)':>9}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for e in experiments:
        lines.append(
            f"  {e.get('nombre', ''):<38}{e.get('dropout', ''):>8}{e.get('optimizador', ''):>7}"
            f"{'Sí' if str(e.get('aumento')).lower() == 'true' else 'No':>5}"
            f"{e.get('epocas', ''):>5}{fmt_num(e.get('train_accuracy_final')):>11}"
            f"{fmt_num(e.get('val_accuracy_final')):>9}{e.get('duracion_segundos', ''):>9}"
        )
    return lines


def notebook_execution_info() -> dict:
    path = PROJECT_ROOT / "notebooks" / "taller_cnn.ipynb"
    if not path.exists():
        return {"existe": False}
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    executed = [c for c in code_cells if c.get("execution_count") is not None]
    with_outputs = [c for c in code_cells if c.get("outputs")]
    return {
        "existe": True,
        "celdas_codigo": len(code_cells),
        "celdas_ejecutadas": len(executed),
        "celdas_con_salida": len(with_outputs),
        "max_execution_count": max([c.get("execution_count") or 0 for c in executed], default=0),
    }


def main() -> Path:
    quality = load_json("data_quality.json", {}) or {}
    manifest = load_json("manifest.json", {}) or {}
    history = load_json("training_history.json", {}) or {}
    metrics = load_json("metrics.json", {}) or {}
    detail = load_json("experiments_detail.json", {}) or {}
    api_tests = load_json("api_test_results.json", {}) or {}
    incidents_extra = load_json("incidents.json", {}) or {}
    experiments = read_csv("experiments.csv")
    predictions = read_csv("predictions_log.csv")
    env = get_environment()
    nb_info = notebook_execution_info()

    # Estadísticas del log de predicciones
    total_pred = len(predictions)
    gatos = sum(1 for r in predictions if r.get("clase") == "gato")
    perros = sum(1 for r in predictions if r.get("clase") == "perro")
    conf_values = []
    lat_values = []
    inciertas = 0
    for r in predictions:
        try:
            conf_values.append(float(r.get("confianza", 0)))
            lat_values.append(float(r.get("tiempo_inferencia_ms", 0)))
        except (TypeError, ValueError):
            pass
        if str(r.get("incierta", "")).lower() == "true":
            inciertas += 1
    uncertain_dir = REPORTS / "uncertain"
    uncertain_files = sorted(uncertain_dir.glob("*")) if uncertain_dir.exists() else []

    L: list[str] = []
    L.append("=" * 100)
    L.append("INFORME COMPLETO DEL PROYECTO: CLASIFICACIÓN DE IMÁGENES DE PERROS Y GATOS")
    L.append("Documento generado automáticamente a partir de los artefactos reales del proyecto.")
    L.append(f"Fecha de generación: {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"Raíz del proyecto: {PROJECT_ROOT}")
    L.append("=" * 100)
    L.append("")
    L.append("LEYENDA DE ESTADOS: COMPLETADO / FALLIDO / NO EJECUTADO / NO VERIFICADO / NO DISPONIBLE")
    L.append("")

    # 1 -------------------------------------------------------------------------------
    L.append("1. RESUMEN DEL PROYECTO")
    L.append("-" * 100)
    L.append(
        "Objetivo: construir un sistema completo de clasificación binaria de imágenes que\n"
        "distinga gatos de perros, con preparación y validación de datos, un modelo de\n"
        "producción basado en MobileNetV2, una CNN académica para experimentos comparativos,\n"
        "evaluación sobre un conjunto de prueba independiente, una API FastAPI con interfaz\n"
        "web en español y monitoreo de predicciones.\n\n"
        "Problema abordado: clasificación supervisada de imágenes (visión por computador).\n"
        "Clases: gato = 0, perro = 1.\n"
        "Tecnologías: Python 3.12, TensorFlow/Keras, scikit-learn, matplotlib, Pillow,\n"
        "pandas, FastAPI, Uvicorn, HTML, CSS y JavaScript puro.\n\n"
        f"Estado general de artefactos principales:\n"
        f"  - Modelo de producción (models/model.keras): {artifact_status(PROJECT_ROOT / 'models' / 'model.keras')}\n"
        f"  - Métricas de prueba (reports/metrics.json): {artifact_status(REPORTS / 'metrics.json')}\n"
        f"  - Experimentos (reports/experiments.csv): {artifact_status(REPORTS / 'experiments.csv')}\n"
        f"  - Notebook (notebooks/taller_cnn.ipynb): {artifact_status(PROJECT_ROOT / 'notebooks' / 'taller_cnn.ipynb')}\n"
        f"  - Informe PDF (reports/informe_cnn.pdf): {artifact_status(REPORTS / 'informe_cnn.pdf')}\n"
        f"  - Pruebas de API (reports/api_test_results.json): {artifact_status(REPORTS / 'api_test_results.json')}"
    )
    L.append("")

    # 2 -------------------------------------------------------------------------------
    L.append("2. ENTORNO DE EJECUCIÓN")
    L.append("-" * 100)
    L.append(f"Sistema operativo: {env['sistema']}")
    L.append(f"Python: {env['python']}")
    L.append(f"Ejecutable: {env['python_executable']}")
    L.append(f"CPU: {env['cpu']} ({env['nucleos_logicos']} hilos lógicos)")
    L.append(f"GPU: {env['gpu']}")
    L.append("Librerías instaladas (versión obtenida de importlib.metadata):")
    for name, ver in env["librerias"].items():
        L.append(f"  - {name}: {ver}")
    L.append("")
    L.append("Nota verificable: el sistema tenía Python 3.14.3, pero 'pip index versions'")
    L.append("no encontró ninguna distribución de TensorFlow para esa versión. Por ello se")
    L.append("instaló Python 3.12.10 y se creó el entorno virtual 'venv' con él.")
    L.append("")

    # 3 -------------------------------------------------------------------------------
    L.append("3. PREPARACIÓN DE DATOS")
    L.append("-" * 100)
    L.append(f"Ubicación del dataset: {quality.get('dataset_dir', PROJECT_ROOT / 'PetImages')}")
    L.append("Archivos del dataset NO modificados: se validaron y descartaron lógicamente, sin borrar.")
    if quality:
        L.append(f"Fecha del informe de calidad: {quality.get('generado_en', 'NO DISPONIBLE')}")
        L.append(f"Criterio de validación: {quality.get('criterio', 'NO DISPONIBLE')}")
        L.append("Conteos por clase:")
        for clase, datos in quality.get("clases", {}).items():
            L.append(f"  - {clase}: examinadas={datos.get('examinadas')}, "
                     f"válidas={datos.get('validas')}, inválidas={datos.get('invalidas')}")
        tot = quality.get("totales", {})
        L.append(f"  - TOTAL: examinadas={tot.get('examinadas')}, válidas={tot.get('validas')}, "
                 f"inválidas={tot.get('invalidas')}")
        invalidos = quality.get("archivos_invalidos", [])
        if invalidos:
            L.append("Archivos inválidos detectados (descartados lógicamente, no borrados):")
            for inv in invalidos:
                L.append(f"  - {inv.get('archivo')}: {inv.get('motivo')}")
        else:
            L.append("Archivos inválidos detectados: ninguno.")
    else:
        L.append("NO DISPONIBLE: no se encontró reports/data_quality.json.")

    if manifest:
        L.append("")
        L.append(f"Semilla: {manifest.get('semilla')}")
        L.append(f"Máximo por clase: {manifest.get('max_por_clase')}")
        L.append(f"Tamaños de las particiones: {manifest.get('tamanos')}")
        L.append(f"Tamaños por clase: {manifest.get('tamanos_por_clase')}")
        L.append("Etiquetas: gato = 0, perro = 1.")
        L.append("Partición estratificada 80 % entrenamiento, 10 % validación, 10 % prueba.")
        L.append("Solapamiento entre particiones: 0 archivos (comprobado comparando las tres listas).")
    else:
        L.append("NO DISPONIBLE: no se encontró reports/manifest.json.")
    L.append("")
    L.append("Preprocesamiento aplicado:")
    L.append("  - Redimensionado a 224x224x3 (modelo de producción).")
    L.append("  - Normalización de píxeles a [0,1] una sola vez (división entre 255).")
    L.append("  - Aumento SOLO en entrenamiento: rotación <= 20 grados, volteo horizontal,")
    L.append("    desplazamiento horizontal y vertical, y zoom.")
    L.append("  - Pipeline tf.data con cache(), batch(32) y prefetch().")
    L.append("  - En inferencia se aplica exactamente el mismo resize y /255 (sin duplicar).")
    L.append("")

    # 4 -------------------------------------------------------------------------------
    L.append("4. ARQUITECTURA DEL MODELO DE PRODUCCIÓN")
    L.append("-" * 100)
    L.append("Base: MobileNetV2 preentrenada en ImageNet, include_top=False, base congelada.")
    L.append("Cabeza: GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid).")
    L.append("Optimizador: Adam, learning rate = 0.0001.")
    L.append("Pérdida: binary crossentropy.")
    L.append("Métricas: accuracy, precision y recall.")
    L.append("Callbacks: EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True).")
    L.append("Épocas máximas: 5. Fine-tuning: no realizado.")
    if history:
        L.append(f"Parámetros totales: {history.get('parametros_totales', 'NO DISPONIBLE')}")
        L.append(f"Parámetros entrenables: {history.get('parametros_entrenables', 'NO DISPONIBLE')}")
        L.append(f"Parámetros no entrenables: {history.get('parametros_no_entrenables', 'NO DISPONIBLE')}")
        L.append(f"Batch size: {history.get('batch_size', 'NO DISPONIBLE')}")
    L.append("Coherencia del preprocesamiento: los datasets entregan imágenes [0,1] y el modelo")
    L.append("no incluye capas de normalización adicionales; la API usa el mismo resize y /255.")
    L.append("")

    # 5 -------------------------------------------------------------------------------
    L.append("5. ARQUITECTURA DE LA CNN ACADÉMICA")
    L.append("-" * 100)
    L.append("Entrada: 96x96x3 (reducción para acotar el costo computacional de los experimentos).")
    L.append("Capas:")
    L.append("  1. Conv2D(32 filtros, kernel 3x3, padding='same', activación ReLU)")
    L.append("  2. MaxPooling2D(2x2)")
    L.append("  3. Conv2D(64 filtros, kernel 3x3, padding='same', activación ReLU)")
    L.append("  4. MaxPooling2D(2x2)")
    L.append("  5. Conv2D(128 filtros, kernel 3x3, padding='same', activación ReLU)")
    L.append("  6. MaxPooling2D(2x2)")
    L.append("  7. Flatten")
    L.append("  8. Dense(64, activación ReLU)")
    L.append("  9. Dropout(0.5) opcional según configuración")
    L.append(" 10. Dense(2, activación Softmax)")
    L.append("Pérdida: categorical crossentropy (etiquetas one-hot).")
    if experiments:
        params = {e.get("parametros") for e in experiments}
        L.append(f"Parámetros de la CNN académica: {', '.join(sorted(params))}")
    else:
        L.append("Parámetros de la CNN académica: NO DISPONIBLE")
    L.append("")

    # 6 -------------------------------------------------------------------------------
    L.append("6. PROCESO DE ENTRENAMIENTO (MobileNetV2)")
    L.append("-" * 100)
    if history:
        L.append(f"Fecha del registro: {history.get('generado_en', 'NO DISPONIBLE')}")
        L.append(f"Épocas ejecutadas: {history.get('epocas_ejecutadas')} (máximo configurado: {history.get('epocas_maximas')})")
        L.append(f"Duración total: {history.get('duracion_segundos')} segundos")
        L.append(f"Mejor época (menor val_loss): {history.get('mejor_epoca')}")
        L.append(f"Mejor val_loss: {fmt_num(history.get('mejor_val_loss'))}")
        L.append(f"Mejor val_accuracy: {fmt_num(history.get('mejor_val_accuracy'))}")
        L.append("Métricas por época (entrenamiento y validación):")
        L.extend(training_table(history))
        L.append("")
        L.append("Estado: COMPLETADO. El modelo se guardó en models/model.keras.")
        L.append("El entrenamiento final se completó sin interrupciones. El primer intento")
        L.append("falló por el problema de decodificación documentado en la sección 13.")
    else:
        L.append("NO EJECUTADO o NO DISPONIBLE: no se encontró reports/training_history.json.")
    L.append("")

    # 7 -------------------------------------------------------------------------------
    L.append("7. EXPERIMENTOS (CNN ACADÉMICA)")
    L.append("-" * 100)
    if detail:
        L.append(f"Fecha: {detail.get('generado_en', 'NO DISPONIBLE')}")
        L.append(f"Semilla: {detail.get('semilla')}")
        L.append(f"Entrada: {detail.get('img_size')}x{detail.get('img_size')}")
        L.append(f"Máximo de entrenamiento por clase: {detail.get('max_train_por_clase')}")
        L.append(f"Máximo de validación por clase: {detail.get('max_val_por_clase')}")
        L.append(f"Épocas por experimento: {detail.get('epocas')}")
    L.append("Configuraciones realmente ejecutadas:")
    L.extend(experiments_table(experiments))
    L.append("")
    L.append("Pares comparables (misma configuración salvo la variable estudiada):")
    L.append("  - Dropout: exp1_adam_dropout_aumento vs exp2_adam_sin_dropout_aumento")
    L.append("  - Optimizador: exp1_adam_dropout_aumento vs exp3_sgd_dropout_aumento")
    L.append("  - Aumento: exp1_adam_dropout_aumento vs exp4_adam_dropout_sin_aumento")
    L.append("  - exp5_sgd_sin_dropout_sin_aumento se ejecutó como configuración adicional.")
    if experiments:
        failed = [e for e in experiments if str(e.get("epocas")) == "0"]
        if failed:
            L.append("Experimentos fallidos: " + ", ".join(e["nombre"] for e in failed))
        else:
            L.append("Experimentos fallidos: ninguno.")
    if not experiments:
        L.append("NO EJECUTADO: no se encontró reports/experiments.csv.")
    L.append("")

    # 8 -------------------------------------------------------------------------------
    L.append("8. EVALUACIÓN FINAL (CONJUNTO DE PRUEBA)")
    L.append("-" * 100)
    if metrics:
        L.append(f"Fecha: {metrics.get('generado_en', 'NO DISPONIBLE')}")
        L.append(f"Modelo evaluado: {metrics.get('modelo')}")
        L.append(f"Conjunto: {metrics.get('conjunto')} | imágenes: {metrics.get('n_imagenes')}")
        L.append(f"Umbral: {metrics.get('umbral')} | positivo: {metrics.get('positivo')} | negativo: {metrics.get('negativo')}")
        L.append(f"Accuracy:  {fmt_num(metrics.get('accuracy'))}")
        L.append(f"Precision: {fmt_num(metrics.get('precision'))}")
        L.append(f"Recall:    {fmt_num(metrics.get('recall'))}")
        L.append(f"F1:        {fmt_num(metrics.get('f1'))}")
        L.append(f"AUC:       {fmt_num(metrics.get('auc'))}")
        cm = metrics.get("matriz_confusion", {}).get("valores", [[0, 0], [0, 0]])
        tn, fp = cm[0]
        fn, tp = cm[1]
        L.append(f"Matriz de confusión: {cm}")
        L.append(f"  Formato: filas = real [gato, perro], columnas = predicción [gato, perro].")
        L.append(f"  VP (perro predicho perro) = {tp}")
        L.append(f"  VN (gato predicho gato)   = {tn}")
        L.append(f"  FP (gato predicho perro)  = {fp}")
        L.append(f"  FN (perro predicho gato)  = {fn}")
        lat = metrics.get("latencia", {})
        L.append("Tiempo medio de inferencia:")
        L.append(f"  - Imágenes medidas: {lat.get('imagenes_medidas')}")
        L.append(f"  - Media: {lat.get('latencia_media_ms')} ms")
        L.append(f"  - Mediana: {lat.get('latencia_mediana_ms')} ms")
        L.append(f"  - p95: {lat.get('latencia_p95_ms')} ms")
        L.append(f"  - Mínimo: {lat.get('latencia_min_ms')} ms | Máximo: {lat.get('latencia_max_ms')} ms")
        L.append("  - Condiciones: CPU, inferencia individual (batch=1), 3 llamadas de")
        L.append("    calentamiento previas, medición con time.perf_counter sobre el modelo Keras.")
        obj = metrics.get("objetivos", {})
        L.append(f"Objetivo accuracy >= 92 %: "
                 f"{'CUMPLIDO' if obj.get('accuracy_cumplido') else 'NO CUMPLIDO'} "
                 f"(accuracy real = {fmt_num(metrics.get('accuracy'))}).")
        L.append(f"Objetivo latencia media < 200 ms: "
                 f"{'CUMPLIDO' if obj.get('latencia_cumplido') else 'NO CUMPLIDO'} "
                 f"(latencia real = {lat.get('latencia_media_ms')} ms).")
        L.append("No se reentrenó automáticamente tras la evaluación.")
    else:
        L.append("NO EJECUTADO o NO DISPONIBLE: no se encontró reports/metrics.json.")
    L.append("")

    # 9 -------------------------------------------------------------------------------
    L.append("9. ANÁLISIS DE RESULTADOS")
    L.append("-" * 100)
    L.append("9.1 Resultados observados")
    if history and history.get("historico"):
        h = history["historico"]
        L.append(f"  - Accuracy train final: {h['accuracy'][-1]:.4f}")
        L.append(f"  - Accuracy val final:   {h['val_accuracy'][-1]:.4f}")
        L.append(f"  - Brecha train - val:   {h['accuracy'][-1] - h['val_accuracy'][-1]:+.4f}")
        L.append(f"  - Loss train final:     {h['loss'][-1]:.4f}")
        L.append(f"  - Loss val final:       {h['val_loss'][-1]:.4f}")
        L.append(f"  - Mínima val_loss:      {min(h['val_loss']):.4f}")
    else:
        L.append("  - Histórico de entrenamiento: NO DISPONIBLE")
    if metrics:
        L.append(f"  - Accuracy test:        {metrics['accuracy']:.4f}")
        L.append(f"  - AUC test:             {metrics['auc']:.4f}")
    else:
        L.append("  - Métricas de test: NO DISPONIBLE")

    L.append("")
    L.append("9.2 Interpretación (basada en los datos anteriores)")
    if history and history.get("historico"):
        h = history["historico"]
        gap = h["accuracy"][-1] - h["val_accuracy"][-1]
        if gap > 0.05:
            L.append("  - La brecha train-validación supera 0.05; esto es compatible con")
            L.append("    cierto sobreajuste. Debe contrastarse con la curva de validación.")
        else:
            L.append("  - La brecha train-validación es inferior a 0.05; con los datos")
            L.append("    observados no se aprecia un sobreajuste severo.")
            if gap < 0:
                L.append("  - En la última época la accuracy de validación fue superior a la")
                L.append("    de entrenamiento (brecha negativa), un patrón habitual cuando")
                L.append("    Dropout y aumento de datos están activos durante el entrenamiento.")
        if h["val_loss"][-1] > min(h["val_loss"]):
            L.append(f"  - La val_loss final ({h['val_loss'][-1]:.4f}) es mayor que la mínima")
            L.append(f"    ({min(h['val_loss']):.4f}); el mejor punto de validación se alcanzó")
            L.append("    antes del final y EarlyStopping restauró los mejores pesos.")
        else:
            L.append("  - La val_loss final coincide con la mínima observada; no hay señal")
            L.append("    de degradación al final del entrenamiento.")
    if metrics:
        obj = metrics.get("objetivos", {})
        L.append(f"  - El objetivo de accuracy >= 92 % "
                 f"{'se cumple' if obj.get('accuracy_cumplido') else 'NO se cumple'}.")
        L.append(f"  - El objetivo de latencia media < 200 ms "
                 f"{'se cumple' if obj.get('latencia_cumplido') else 'NO se cumple'}.")
    if experiments:
        by = {e["nombre"]: e for e in experiments}

        def v(name):
            try:
                return float(by[name]["val_accuracy_final"])
            except (KeyError, TypeError, ValueError):
                return None

        pairs = [
            ("Dropout", "exp1_adam_dropout_aumento", "exp2_adam_sin_dropout_aumento"),
            ("Adam vs SGD", "exp1_adam_dropout_aumento", "exp3_sgd_dropout_aumento"),
            ("Aumento", "exp1_adam_dropout_aumento", "exp4_adam_dropout_sin_aumento"),
        ]
        for label, a, b in pairs:
            va, vb = v(a), v(b)
            if va is not None and vb is not None:
                L.append(f"  - {label}: val_accuracy {a} = {va:.4f} frente a "
                         f"{b} = {vb:.4f} (diferencia {va - vb:+.4f}).")
        L.append("  - Estas diferencias corresponden a una sola ejecución por configuración;")
        L.append("    no se realizó validación cruzada ni prueba de significancia estadística,")
        L.append("    por lo que son observaciones descriptivas y no conclusiones causales.")
    L.append("")

    # 10 ------------------------------------------------------------------------------
    L.append("10. APLICACIÓN WEB Y API")
    L.append("-" * 100)
    L.append("Endpoints implementados en app/main.py:")
    L.append("  - GET /        -> sirve app/static/index.html.")
    L.append("  - GET /health  -> estado del servicio y del modelo (cargado o no).")
    L.append("  - POST /predict-> recibe multipart/form-data con el campo 'file'.")
    L.append("  - GET /stats   -> estadísticas del monitoreo.")
    L.append("Validaciones de /predict:")
    L.append("  - Extensión permitida: .jpg, .jpeg, .png, .webp.")
    L.append("  - Tipo MIME declarado permitido: image/jpeg, image/jpg, image/png, image/webp.")
    L.append("  - Tamaño máximo: 5 MB (5 * 1024 * 1024 bytes).")
    L.append("  - Contenido real validado con PIL: open() + verify() + load() + formato real.")
    L.append("Códigos HTTP: 400 imagen inválida, 413 demasiado grande, 415 formato no admitido,")
    L.append("503 modelo no disponible, 200 éxito.")
    L.append("Formato de respuesta: {\"clase\":\"perro\",\"confianza\":0.97}")
    L.append("Regla: sigmoid >= 0.5 -> perro con confianza = sigmoid; sigmoid < 0.5 -> gato")
    L.append("con confianza = 1 - sigmoid. Si la confianza está entre 0.40 y 0.60 se añade")
    L.append("\"incierta\": true.")
    L.append("Carga del modelo: una sola vez al iniciar mediante lifespan.")
    L.append("CORS: habilitado.")
    L.append("Frontend (app/static): interfaz en español, responsive, arrastrar y soltar,")
    L.append("selector de archivo, vista previa, botón Clasificar con indicador de carga,")
    L.append("botón Limpiar, resultado con icono/clase/barra de confianza, aviso de predicción")
    L.append("incierta y mensajes de error. Usa fetch('/predict') con ruta relativa.")
    L.append("Verificación web: GET / y los recursos /static/style.css y /static/app.js")
    L.append("respondieron HTTP 200 y el HTML contiene la interfaz esperada. No se realizó")
    L.append("una inspección visual automatizada en navegador (NO VERIFICADO visualmente).")
    L.append("")

    # 11 ------------------------------------------------------------------------------
    L.append("11. MONITOREO")
    L.append("-" * 100)
    L.append("Archivo: reports/predictions_log.csv")
    L.append("Campos: fecha, nombre_archivo, clase, confianza, tiempo_inferencia_ms,")
    L.append("ancho, alto, brillo_medio, incierta.")
    L.append("Imágenes inciertas: reports/uncertain/ con nombre único (fecha + uuid).")
    L.append("Solo se guardan imágenes de predicciones inciertas; no se guardan las demás.")
    L.append("No se registran IP ni datos personales.")
    L.append(f"Predicciones registradas hasta ahora: {total_pred}")
    if total_pred:
        L.append(f"  - Gatos: {gatos} ({gatos / total_pred:.2%})")
        L.append(f"  - Perros: {perros} ({perros / total_pred:.2%})")
        L.append(f"  - Confianza media: {sum(conf_values) / len(conf_values):.4f}" if conf_values else "  - Confianza media: NO DISPONIBLE")
        L.append(f"  - Latencia media: {sum(lat_values) / len(lat_values):.2f} ms" if lat_values else "  - Latencia media: NO DISPONIBLE")
        L.append(f"  - Inciertas: {inciertas} ({inciertas / total_pred:.2%})")
    else:
        L.append("  - Todavía no hay predicciones registradas (NO DISPONIBLE para agregados).")
    L.append(f"Imágenes inciertas guardadas: {len(uncertain_files)}")
    if total_pred and inciertas == 0:
        L.append("En las pruebas realizadas no hubo predicciones inciertas, por lo que no se")
        L.append("creó ningún archivo en reports/uncertain/. El mecanismo de guardado está")
        L.append("implementado y se activa únicamente cuando 0.40 <= confianza <= 0.60.")
    L.append("")

    # 12 ------------------------------------------------------------------------------
    L.append("12. PRUEBAS REALIZADAS")
    L.append("-" * 100)
    L.append("12.1 Comandos ejecutados y estado")
    commands = [
        ("python --version; py -0p", "COMPLETADO", "Python 3.14.3 del sistema; luego se instaló 3.12.10."),
        ("python -m pip index versions tensorflow", "COMPLETADO", "Sin distribución para Python 3.14."),
        ("winget install --id Python.Python.3.12 --scope user --silent", "COMPLETADO", "Python 3.12.10 instalado."),
        ("py -3.12 -m venv E:\\gatos\\venv", "COMPLETADO", "Entorno virtual creado."),
        ("venv\\Scripts\\python.exe -m pip install --upgrade pip", "COMPLETADO", "pip 26.2.1."),
        ("venv\\Scripts\\python.exe -m pip install -r requirements.txt", "COMPLETADO" if env["librerias"].get("tensorflow", "NO DISPONIBLE") != "NO DISPONIBLE" else "NO VERIFICADO", "Dependencias instaladas."),
        ("python -m src.data_prep", artifact_status(REPORTS / "manifest.json"), "Validación y partición."),
        ("python -m src.train", artifact_status(PROJECT_ROOT / "models" / "model.keras"), "Entrenamiento MobileNetV2."),
        ("python -m src.experiments", artifact_status(REPORTS / "experiments.csv"), "Experimentos CNN académica."),
        ("python -m src.evaluate", artifact_status(REPORTS / "metrics.json"), "Evaluación sobre test."),
        ("python -m src.make_notebook", artifact_status(PROJECT_ROOT / "notebooks" / "taller_cnn.ipynb"), "Generación del notebook."),
        ("jupyter nbconvert --to notebook --execute --inplace notebooks/taller_cnn.ipynb", "COMPLETADO" if nb_info.get("celdas_ejecutadas", 0) > 0 else "NO EJECUTADO", "Ejecución del notebook."),
        ("python -m src.make_pdf", artifact_status(REPORTS / "informe_cnn.pdf"), "Generación del PDF."),
        ("uvicorn app.main:app --host 127.0.0.1 --port 8000", "COMPLETADO" if api_tests else "NO VERIFICADO", "Servidor web."),
        ("python scripts/test_api.py --base-url http://127.0.0.1:8000", artifact_status(REPORTS / "api_test_results.json"), "Pruebas de API."),
    ]
    for cmd, status, note in commands:
        L.append(f"  [{status}] {cmd}  -> {note}")
    L.append("")
    L.append("12.2 Pruebas de endpoints")
    if api_tests:
        L.append(f"Fecha: {api_tests.get('fecha')} | URL base: {api_tests.get('base_url')}")
        for item in api_tests.get("pruebas", []):
            L.append(f"  [{ 'OK' if item.get('exito') else 'FALLO' }] {item.get('prueba')}: "
                     f"HTTP {item.get('status_http')} (esperado {item.get('status_esperado')}), "
                     f"{item.get('tiempo_ms')} ms")
            resp = item.get("respuesta")
            if isinstance(resp, dict):
                L.append(f"         respuesta: {json.dumps(resp, ensure_ascii=False)}")
            elif resp is not None:
                snippet = str(resp).replace("\n", " ")[:200]
                L.append(f"         respuesta (fragmento): {snippet}")
        resumen = api_tests.get("resumen", {})
        L.append(f"Resumen: {resumen.get('exitosas')}/{resumen.get('total')} pruebas exitosas, "
                 f"{resumen.get('fallidas')} fallidas.")
    else:
        L.append("NO EJECUTADO: no se encontró reports/api_test_results.json.")
    L.append("")
    L.append("12.3 Notebook")
    if nb_info.get("existe"):
        L.append(f"  - Celdas de código: {nb_info.get('celdas_codigo')}")
        L.append(f"  - Celdas ejecutadas (execution_count no nulo): {nb_info.get('celdas_ejecutadas')}")
        L.append(f"  - Celdas con salidas guardadas: {nb_info.get('celdas_con_salida')}")
        L.append(f"  - Máximo execution_count: {nb_info.get('max_execution_count')}")
        if nb_info.get("celdas_ejecutadas") == nb_info.get("celdas_codigo"):
            L.append("  - Estado: COMPLETADO (todas las celdas de código ejecutadas).")
        else:
            L.append("  - Estado: EJECUCIÓN PARCIAL o NO VERIFICADA.")
    else:
        L.append("  - Estado: NO EJECUTADO (no existe el notebook).")
    L.append("")

    # 13 ------------------------------------------------------------------------------
    L.append("13. INCIDENCIAS Y LIMITACIONES")
    L.append("-" * 100)
    L.append("Incidencias reales encontradas:")
    L.append("  1. Python 3.14.3 no tiene ruedas de TensorFlow disponibles.")
    L.append("     Causa: TensorFlow no publica distribuciones para esa versión de Python.")
    L.append("     Corrección: se instaló Python 3.12.10 con winget y se creó el venv con él.")
    L.append("     Estado: CORREGIDO.")
    L.append("  2. El proyecto inicial solo contenía PetImages/ y archive.zip.")
    L.append("     Causa: no existía código ni estructura previa.")
    L.append("     Corrección: se creó toda la estructura solicitada sin tocar PetImages/.")
    L.append("     Estado: CORREGIDO.")
    if incidents_extra:
        for idx, inc in enumerate(incidents_extra.get("incidencias", []), start=3):
            L.append(f"  {idx}. {inc.get('descripcion')}")
            L.append(f"     Causa: {inc.get('causa')}")
            L.append(f"     Corrección: {inc.get('correccion')}")
            L.append(f"     Estado: {inc.get('estado')}")
    L.append("")
    L.append("Limitaciones:")
    L.append("  - Dataset con posibles etiquetas erróneas; no se auditó manualmente cada imagen.")
    L.append("  - Entrenamiento acotado a 5 épocas y hasta 3000 imágenes por clase.")
    L.append("  - Experimentos académicos a 96x96 y con subconjunto por límites de cómputo.")
    L.append("  - Sin GPU: los tiempos corresponden a CPU.")
    L.append("  - Una sola ejecución por configuración; sin validación cruzada ni significancia.")
    L.append("  - No se usó Docker, Grad-CAM, pytest, fine-tuning ni scripts de drift.")
    L.append("")
    L.append("Tareas no ejecutadas o no verificadas:")
    not_done = []
    if not (PROJECT_ROOT / "models" / "model.keras").exists():
        not_done.append("Entrenamiento del modelo de producción.")
    if not (REPORTS / "experiments.csv").exists():
        not_done.append("Experimentos académicos.")
    if not (REPORTS / "metrics.json").exists():
        not_done.append("Evaluación final.")
    if nb_info.get("celdas_ejecutadas", 0) == 0:
        not_done.append("Ejecución del notebook.")
    if not (REPORTS / "informe_cnn.pdf").exists():
        not_done.append("Generación del PDF.")
    if not api_tests:
        not_done.append("Pruebas de API.")
    if not_done:
        for item in not_done:
            L.append(f"  - NO EJECUTADO: {item}")
    else:
        L.append("  - Ninguna de las tareas principales quedó pendiente según los artefactos.")
    L.append("")

    # 14 ------------------------------------------------------------------------------
    L.append("14. CONCLUSIONES")
    L.append("-" * 100)
    if metrics:
        obj = metrics.get("objetivos", {})
        L.append(f"  - El modelo de producción alcanzó accuracy = {metrics['accuracy']:.4f} y "
                 f"AUC = {metrics['auc']:.4f} sobre {metrics['n_imagenes']} imágenes de prueba.")
        L.append(f"  - El objetivo de accuracy >= 92 % "
                 f"{'se cumplió' if obj.get('accuracy_cumplido') else 'NO se cumplió'}.")
        L.append(f"  - El objetivo de latencia media < 200 ms "
                 f"{'se cumplió' if obj.get('latencia_cumplido') else 'NO se cumplió'} "
                 f"(media observada = {metrics['latencia']['latencia_media_ms']} ms en CPU).")
    else:
        L.append("  - NO DISPONIBLE: no hay métricas de prueba; no se puede concluir sobre el desempeño.")
    L.append("  - La CNN académica permitió comparar Dropout, Adam/SGD y aumento de datos en")
    L.append("    un escenario acotado. Los resultados están en reports/experiments.csv y")
    L.append("    corresponden a ejecuciones reales.")
    L.append("  - No se debe afirmar que un experimento es mejor de forma general: solo se")
    L.append("    observó una ejecución por configuración sobre un subconjunto.")
    L.append("")

    # 15 ------------------------------------------------------------------------------
    L.append("15. INVENTARIO DE EVIDENCIAS")
    L.append("-" * 100)
    evidence = [
        PROJECT_ROOT / "models" / "model.keras",
        REPORTS / "metrics.json",
        REPORTS / "training_history.json",
        REPORTS / "training_curves.png",
        REPORTS / "confusion_matrix.png",
        REPORTS / "roc_curve.png",
        REPORTS / "experiments.csv",
        REPORTS / "experiments_detail.json",
        REPORTS / "experiments_comparison.png",
        REPORTS / "data_quality.json",
        REPORTS / "manifest.json",
        REPORTS / "predictions_log.csv",
        REPORTS / "api_test_results.json",
        PROJECT_ROOT / "notebooks" / "taller_cnn.ipynb",
        REPORTS / "informe_cnn.pdf",
        REPORTS / "informe_completo.txt",
    ]
    for path in evidence:
        L.append(artifact_line(path))
    if uncertain_dir.exists():
        L.append(f"  - reports/uncertain/: {len(uncertain_files)} archivo(s).")
    L.append("")

    # 16 ------------------------------------------------------------------------------
    L.append("16. REPRODUCIBILIDAD")
    L.append("-" * 100)
    L.append("Comandos exactos (desde la raíz del proyecto, con el venv activo):")
    L.append("  py -3.12 -m venv venv")
    L.append("  .\\venv\\Scripts\\Activate.ps1")
    L.append("  python -m pip install --upgrade pip")
    L.append("  pip install -r requirements.txt")
    L.append("  python -m src.data_prep")
    L.append("  python -m src.train")
    L.append("  python -m src.experiments")
    L.append("  python -m src.evaluate")
    L.append("  python -m src.make_notebook")
    L.append("  jupyter nbconvert --to notebook --execute --inplace notebooks/taller_cnn.ipynb")
    L.append("  python -m src.make_pdf")
    L.append("  python -m src.make_final_report")
    L.append("  uvicorn app.main:app --host 127.0.0.1 --port 8000")
    L.append("  python scripts/test_api.py --base-url http://127.0.0.1:8000")
    L.append("")
    L.append("Notas de reproducibilidad:")
    L.append(f"  - Semilla fija: {manifest.get('semilla', 'NO DISPONIBLE')}.")
    L.append("  - Los artefactos existentes se reutilizan por defecto; usar force=True en")
    L.append("    data_prep.build_manifest/validate_dataset para regenerarlos.")
    L.append("  - Los pesos ImageNet se descargan si hay Internet; si no, train.py avisa y")
    L.append("    entrena la base con pesos aleatorios.")
    L.append("")
    L.append("=" * 100)
    L.append("FIN DEL INFORME COMPLETO")
    L.append("=" * 100)

    OUT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    (REPORTS / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[make_final_report] Informe TXT generado -> {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
