"""Genera ``reports/informe_cnn.pdf`` en español con los resultados reales.

Todas las cifras se leen de ``reports/metrics.json``, ``reports/training_history.json``,
``reports/experiments.csv``, ``reports/data_quality.json`` y ``reports/manifest.json``.
No se introducen métricas manuales.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
OUT_PATH = REPORTS / "informe_cnn.pdf"


def load_json(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiments() -> list[dict]:
    path = REPORTS / "experiments.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "Titulo", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        "Subtitulo", parent=styles["Normal"], fontSize=11, leading=15,
        alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "H1c", parent=styles["Heading1"], fontSize=14, leading=18,
        spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#3730a3"),
    ))
    styles.add(ParagraphStyle(
        "H2c", parent=styles["Heading2"], fontSize=11.5, leading=15,
        spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#4338ca"),
    ))
    styles.add(ParagraphStyle(
        "Cuerpo", parent=styles["BodyText"], fontSize=9.6, leading=13.5,
        alignment=TA_JUSTIFY, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "BulletC", parent=styles["BodyText"], fontSize=9.6, leading=13.5,
        leftIndent=14, bulletIndent=4, spaceAfter=3, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        "Pie", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#777777"),
        alignment=TA_CENTER,
    ))
    return styles


def p(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str], styles) -> list:
    return [Paragraph(f"• {item}", styles["BulletC"]) for item in items]


def add_image(story: list, path: Path, width_cm: float = 15.5, caption: str | None = None, styles=None):
    if not path.exists():
        story.append(p(f"<i>Figura no disponible: {path.name}</i>", styles["Cuerpo"]))
        return
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    width = width_cm * cm
    height = width * h / w
    max_height = 10.5 * cm
    if height > max_height:
        height = max_height
        width = height * w / h
    story.append(Image(str(path), width=width, height=height))
    if caption:
        story.append(p(f"<i>{caption}</i>", styles["Cuerpo"]))
    story.append(Spacer(1, 6))


def main() -> Path:
    metrics = load_json("metrics.json")
    history = load_json("training_history.json")
    quality = load_json("data_quality.json")
    manifest = load_json("manifest.json")
    experiments = load_experiments()
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(OUT_PATH), pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="Informe CNN gato/perro", author="Proyecto académico",
    )
    story: list = []

    # ------------------------------ Portada ------------------------------
    story.append(Spacer(1, 3.5 * cm))
    story.append(p("Clasificación de imágenes de perros y gatos", styles["Titulo"]))
    story.append(p("MobileNetV2 como modelo de producción y CNN académica comparativa", styles["Subtitulo"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(p(f"Informe generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                   styles["Subtitulo"]))
    story.append(p("Todas las métricas provienen de los artefactos reales del proyecto.",
                   styles["Subtitulo"]))
    story.append(Spacer(1, 1.5 * cm))
    story.append(p(
        "Este documento describe la arquitectura, el proceso de entrenamiento, los "
        "experimentos comparativos y la evaluación del modelo. Las cifras se obtuvieron "
        "de <b>reports/metrics.json</b>, <b>reports/training_history.json</b>, "
        "<b>reports/experiments.csv</b> y <b>reports/data_quality.json</b>.",
        styles["Cuerpo"],
    ))
    story.append(PageBreak())

    # ------------------------------ Resumen ------------------------------
    story.append(p("1. Resumen del proyecto", styles["H1c"]))
    tot = quality.get("totales", {}) if quality else {}
    story.append(p(
        "Se construyó un sistema completo de clasificación binaria de imágenes de gatos "
        "y perros. El flujo incluye validación de imágenes, partición estratificada "
        "80/10/10, aumento de datos, entrenamiento de MobileNetV2, experimentos con una "
        "CNN académica, evaluación sobre test, API FastAPI, interfaz web y monitoreo.",
        styles["Cuerpo"],
    ))
    story.append(p(
        f"El dataset examinado contiene {tot.get('examinadas', 'NO DISPONIBLE')} archivos, "
        f"de los cuales {tot.get('validas', 'NO DISPONIBLE')} resultaron válidos y "
        f"{tot.get('invalidas', 'NO DISPONIBLE')} inválidos. Los inválidos se descartaron "
        "lógicamente sin borrar ni modificar archivos de PetImages/.",
        styles["Cuerpo"],
    ))

    # ------------------------------ Datos ------------------------------
    story.append(p("2. Datos y preprocesamiento", styles["H1c"]))
    if manifest:
        sizes = manifest.get("tamanos", {})
        story.append(p(
            f"Semilla fija: <b>{manifest.get('semilla')}</b>. Selección: hasta "
            f"{manifest.get('max_por_clase')} imágenes válidas por clase. Particiones: "
            f"entrenamiento = {sizes.get('train', 'NO DISPONIBLE')}, "
            f"validación = {sizes.get('val', 'NO DISPONIBLE')}, "
            f"prueba = {sizes.get('test', 'NO DISPONIBLE')}.",
            styles["Cuerpo"],
        ))
    story.extend(bullets([
        "Etiquetas: gato = 0, perro = 1.",
        "Redimensionado a 224×224×3 y normalización a [0,1] una sola vez.",
        "Aumento solo en entrenamiento: rotación ≤ 20°, volteo horizontal, "
        "desplazamientos horizontal y vertical y zoom.",
        "Pipeline tf.data con cache(), batch(32) y prefetch().",
        "Partición estratificada y sin solapamiento entre conjuntos.",
    ], styles))
    if quality:
        rows = [["Clase", "Examinadas", "Válidas", "Inválidas"]]
        for clase, datos in quality.get("clases", {}).items():
            rows.append([clase, datos.get("examinadas"), datos.get("validas"), datos.get("invalidas")])
        table = Table(rows, colWidths=[3 * cm, 4 * cm, 4 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7cbe0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(Spacer(1, 6))
        story.append(table)
        story.append(Spacer(1, 8))

    # ------------------------------ MobileNetV2 ------------------------------
    story.append(p("3. Arquitectura del modelo de producción", styles["H1c"]))
    story.extend(bullets([
        "Base: MobileNetV2 preentrenada en ImageNet, include_top=False, congelada.",
        "Cabeza: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid).",
        "Optimizador: Adam con learning rate 0.0001.",
        "Pérdida: binary crossentropy. Métricas: accuracy, precision y recall.",
        "Máximo 5 épocas; EarlyStopping con paciencia 2 y restauración de los mejores pesos.",
        "Sin fine-tuning.",
        "La normalización se aplica en el pipeline tf.data; el modelo no normaliza de nuevo.",
    ], styles))
    if history:
        story.append(p(
            f"Configuración registrada: batch = {history.get('batch_size')}, "
            f"entrada = {history.get('img_size')}, épocas máximas = {history.get('epocas_maximas')}.",
            styles["Cuerpo"],
        ))

    # ------------------------------ CNN académica ------------------------------
    story.append(p("4. Arquitectura de la CNN académica", styles["H1c"]))
    params = experiments[0].get("parametros") if experiments else "NO DISPONIBLE"
    story.extend(bullets([
        "Conv2D de 32 filtros, kernel 3×3, padding same, activación ReLU.",
        "MaxPooling2D de 2×2.",
        "Conv2D de 64 filtros, kernel 3×3, ReLU, seguida de MaxPooling2D 2×2.",
        "Conv2D de 128 filtros, kernel 3×3, ReLU, seguida de MaxPooling2D 2×2.",
        "Flatten → Dense(64, ReLU) → Dropout opcional (0.5) → Dense(2, softmax).",
        f"Parámetros de la CNN académica: {params}.",
        "Entrada de los experimentos: 96×96×3 (reducción por límites de cómputo).",
        "Pérdida: categorical crossentropy.",
    ], styles))

    # ------------------------------ Entrenamiento ------------------------------
    if history:
        story.append(p("5. Proceso de entrenamiento", styles["H1c"]))
        story.append(p(
            f"El modelo de producción se entrenó durante "
            f"<b>{history.get('epocas_ejecutadas')} épocas</b> en "
            f"<b>{history.get('duracion_segundos')} segundos</b>. La mejor época según "
            f"val_loss fue la <b>{history.get('mejor_epoca')}</b>, con val_loss = "
            f"{history.get('mejor_val_loss'):.4f} y val_accuracy = "
            f"{history.get('mejor_val_accuracy'):.4f}.",
            styles["Cuerpo"],
        ))
        hist = history.get("historico", {})
        if hist:
            rows = [["Época", "Loss train", "Loss val", "Acc train", "Acc val",
                     "Precision val", "Recall val"]]
            for i in range(len(hist["loss"])):
                rows.append([
                    i + 1,
                    f"{hist['loss'][i]:.4f}", f"{hist['val_loss'][i]:.4f}",
                    f"{hist['accuracy'][i]:.4f}", f"{hist['val_accuracy'][i]:.4f}",
                    f"{hist.get('val_precision', [0])[i]:.4f}",
                    f"{hist.get('val_recall', [0])[i]:.4f}",
                ])
            table = Table(rows, colWidths=[1.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm,
                                           2.4 * cm, 2.6 * cm, 2.4 * cm])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7cbe0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]))
            story.append(Spacer(1, 6))
            story.append(table)
            story.append(Spacer(1, 8))
        add_image(story, REPORTS / "training_curves.png",
                  caption="Figura 1. Curvas de pérdida y accuracy de entrenamiento frente a validación.",
                  styles=styles)

    # ------------------------------ Evaluación ------------------------------
    story.append(p("6. Evaluación final sobre el conjunto de prueba", styles["H1c"]))
    if metrics:
        story.append(p(
            f"Se evaluaron <b>{metrics.get('n_imagenes')}</b> imágenes de prueba con umbral "
            f"{metrics.get('umbral')} (positivo = perro). Accuracy = "
            f"<b>{metrics.get('accuracy'):.4f}</b>, precision = "
            f"<b>{metrics.get('precision'):.4f}</b>, recall = "
            f"<b>{metrics.get('recall'):.4f}</b>, F1 = <b>{metrics.get('f1'):.4f}</b> y "
            f"AUC = <b>{metrics.get('auc'):.4f}</b>.",
            styles["Cuerpo"],
        ))
        cm_val = metrics.get("matriz_confusion", {}).get("valores", [[0, 0], [0, 0]])
        tn, fp = cm_val[0]
        fn, tp = cm_val[1]
        total = tn + fp + fn + tp
        story.append(p("Fórmula y cálculo de accuracy", styles["H2c"]))
        story.append(p(
            "accuracy = (VP + VN) / (VP + VN + FP + FN)", styles["Cuerpo"],
        ))
        story.append(p(
            f"accuracy = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = "
            f"{tp + tn} / {total} = <b>{(tp + tn) / total:.4f}</b>",
            styles["Cuerpo"],
        ))
        lat = metrics.get("latencia", {})
        story.append(p("Tiempo de inferencia", styles["H2c"]))
        story.append(p(
            f"Medición sobre {lat.get('imagenes_medidas', 'NO DISPONIBLE')} imágenes "
            f"individuales del conjunto de prueba, tras un calentamiento de 3 llamadas, "
            f"en CPU. Media = <b>{lat.get('latencia_media_ms')} ms</b>, mediana = "
            f"{lat.get('latencia_mediana_ms')} ms, p95 = {lat.get('latencia_p95_ms')} ms "
            f"(mín. {lat.get('latencia_min_ms')} ms, máx. {lat.get('latencia_max_ms')} ms).",
            styles["Cuerpo"],
        ))
        obj = metrics.get("objetivos", {})
        story.append(p(
            f"<b>Objetivo accuracy ≥ 92 %:</b> "
            f"{'CUMPLIDO' if obj.get('accuracy_cumplido') else 'NO CUMPLIDO'}. "
            f"<b>Objetivo latencia media < 200 ms:</b> "
            f"{'CUMPLIDO' if obj.get('latencia_cumplido') else 'NO CUMPLIDO'}.",
            styles["Cuerpo"],
        ))
        rows = [["Métrica", "Valor"],
                ["Accuracy", f"{metrics.get('accuracy'):.4f}"],
                ["Precision", f"{metrics.get('precision'):.4f}"],
                ["Recall", f"{metrics.get('recall'):.4f}"],
                ["F1", f"{metrics.get('f1'):.4f}"],
                ["AUC", f"{metrics.get('auc'):.4f}"],
                ["Latencia media (ms)", f"{lat.get('latencia_media_ms')}"]]
        table = Table(rows, colWidths=[7 * cm, 5 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7cbe0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(Spacer(1, 6))
        story.append(table)
        story.append(Spacer(1, 10))
        add_image(story, REPORTS / "confusion_matrix.png", width_cm=10,
                  caption="Figura 2. Matriz de confusión sobre el conjunto de prueba.", styles=styles)
        add_image(story, REPORTS / "roc_curve.png", width_cm=10,
                  caption="Figura 3. Curva ROC y AUC sobre el conjunto de prueba.", styles=styles)
    else:
        story.append(p("NO DISPONIBLE: no se encontró reports/metrics.json.", styles["Cuerpo"]))

    # ------------------------------ Experimentos ------------------------------
    story.append(PageBreak())
    story.append(p("7. Experimentos comparativos", styles["H1c"]))
    if experiments:
        story.append(p(
            "La CNN académica se entrenó con 5 épocas por configuración sobre un "
            "subconjunto reproducible y entrada de 96×96. La tabla contiene únicamente "
            "resultados de ejecuciones reales.",
            styles["Cuerpo"],
        ))
        rows = [["Experimento", "Dropout", "Optim.", "Aumento", "Épocas",
                 "Acc train", "Acc val", "Duración (s)"]]
        for e in experiments:
            def fmt(x):
                try:
                    return f"{float(x):.4f}"
                except (TypeError, ValueError):
                    return "NO DISPONIBLE"
            rows.append([
                e.get("nombre", ""), e.get("dropout", ""), e.get("optimizador", ""),
                "Sí" if str(e.get("aumento")).lower() == "true" else "No",
                e.get("epocas", ""), fmt(e.get("train_accuracy_final")),
                fmt(e.get("val_accuracy_final")), e.get("duracion_segundos", ""),
            ])
        table = Table(rows, colWidths=[4.0 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm,
                                       1.3 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7cbe0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.2),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))
        add_image(story, REPORTS / "experiments_comparison.png",
                  caption="Figura 4. Comparación de accuracy de entrenamiento y validación por experimento.",
                  styles=styles)

        # Comparaciones directas (pares limpios)
        by_name = {e["nombre"]: e for e in experiments}

        def val(name):
            try:
                return float(by_name[name]["val_accuracy_final"])
            except (KeyError, TypeError, ValueError):
                return None

        story.append(p("Análisis de los pares comparables", styles["H2c"]))
        analysis = []
        d_drop = val("exp1_adam_dropout_aumento")
        d_nodrop = val("exp2_adam_sin_dropout_aumento")
        if d_drop is not None and d_nodrop is not None:
            analysis.append(
                f"Dropout (Adam + aumento): con Dropout val_accuracy = {d_drop:.4f}; "
                f"sin Dropout = {d_nodrop:.4f}; diferencia = {d_drop - d_nodrop:+.4f}."
            )
        d_adam = val("exp1_adam_dropout_aumento")
        d_sgd = val("exp3_sgd_dropout_aumento")
        if d_adam is not None and d_sgd is not None:
            analysis.append(
                f"Optimizador (Dropout + aumento): Adam val_accuracy = {d_adam:.4f}; "
                f"SGD = {d_sgd:.4f}; diferencia = {d_adam - d_sgd:+.4f}."
            )
        d_aug = val("exp1_adam_dropout_aumento")
        d_noaug = val("exp4_adam_dropout_sin_aumento")
        if d_aug is not None and d_noaug is not None:
            analysis.append(
                f"Aumento (Adam + Dropout): con aumento val_accuracy = {d_aug:.4f}; "
                f"sin aumento = {d_noaug:.4f}; diferencia = {d_aug - d_noaug:+.4f}."
            )
        story.extend(bullets(analysis, styles))
        valid_vals = [(e["nombre"], val(e["nombre"])) for e in experiments]
        valid_vals = [(n, v) for n, v in valid_vals if v is not None]
        if valid_vals:
            best_name, best_val = max(valid_vals, key=lambda x: x[1])
            story.append(p(
                f"La mejor val_accuracy observada fue <b>{best_val:.4f}</b> en "
                f"<b>{best_name}</b>. Estas diferencias corresponden a una sola ejecución "
                "por configuración; no se realizó validación cruzada ni pruebas de "
                "significancia estadística, por lo que deben interpretarse como "
                "observaciones descriptivas.",
                styles["Cuerpo"],
            ))
    else:
        story.append(p("NO DISPONIBLE: no se encontró reports/experiments.csv.", styles["Cuerpo"]))

    # ------------------------------ Overfitting ------------------------------
    story.append(p("8. Análisis de overfitting", styles["H1c"]))
    if history and history.get("historico"):
        hist = history["historico"]
        train_acc = hist["accuracy"][-1]
        val_acc = hist["val_accuracy"][-1]
        gap = train_acc - val_acc
        min_val_loss = min(hist["val_loss"])
        last_val_loss = hist["val_loss"][-1]
        story.append(p("Resultados observados", styles["H2c"]))
        story.extend(bullets([
            f"Accuracy final de entrenamiento: {train_acc:.4f}.",
            f"Accuracy final de validación: {val_acc:.4f}.",
            f"Brecha train − validación en la última época: {gap:+.4f}.",
            f"Mejor val_loss: {min_val_loss:.4f}; val_loss final: {last_val_loss:.4f}.",
        ], styles))
        story.append(p("Interpretación", styles["H2c"]))
        if gap > 0.05:
            text = ("La brecha supera 0.05, lo que es compatible con cierto sobreajuste. "
                    "Debe contrastarse con las curvas: si la pérdida de validación deja de "
                    "mejorar mientras la de entrenamiento sigue bajando, la señal es más clara.")
        else:
            text = ("La brecha es inferior a 0.05, por lo que con estos datos no se aprecia "
                    "un sobreajuste severo. Esto no descarta sobreajuste leve ni garantiza "
                    "generalización a dominios distintos.")
        if last_val_loss > min_val_loss:
            text += (f" La val_loss final ({last_val_loss:.4f}) es superior a la mínima "
                     f"({min_val_loss:.4f}), lo que indica que el mejor punto de validación "
                     "se alcanzó antes del final del entrenamiento; EarlyStopping restauró "
                     "los mejores pesos.")
        story.append(p(text, styles["Cuerpo"]))
        story.append(p(
            "Los experimentos académicos usan un subconjunto y 96×96, por lo que sus "
            "resultados no son directamente comparables con el modelo de producción.",
            styles["Cuerpo"],
        ))
    else:
        story.append(p("NO DISPONIBLE: no hay histórico de entrenamiento.", styles["Cuerpo"]))

    # ------------------------------ Conclusiones ------------------------------
    story.append(p("9. Conclusiones", styles["H1c"]))
    if metrics:
        obj = metrics.get("objetivos", {})
        concl = [
            f"El modelo de producción obtuvo accuracy = {metrics.get('accuracy'):.4f} "
            f"y AUC = {metrics.get('auc'):.4f} sobre {metrics.get('n_imagenes')} imágenes de prueba.",
            f"El objetivo de accuracy ≥ 92 % "
            f"{'se cumplió' if obj.get('accuracy_cumplido') else 'NO se cumplió'}.",
            f"El objetivo de latencia media < 200 ms "
            f"{'se cumplió' if obj.get('latencia_cumplido') else 'NO se cumplió'} "
            f"(media observada: {metrics['latencia']['latencia_media_ms']} ms en CPU).",
            "La comparación de Dropout, Adam/SGD y aumento de datos se realizó con una "
            "CNN independiente y un subconjunto acotado; sus conclusiones son descriptivas.",
        ]
    else:
        concl = ["NO DISPONIBLE: no se pudo evaluar el modelo."]
    story.extend(bullets(concl, styles))

    story.append(p("10. Limitaciones", styles["H1c"]))
    story.extend(bullets([
        "Entrenamiento acotado a 5 épocas y hasta 3000 imágenes por clase.",
        "Posibles etiquetas erróneas en el dataset original (no se auditó manualmente cada imagen).",
        "Los experimentos académicos usan 96×96 y un subconjunto por límites de cómputo.",
        "Ejecución en CPU; los tiempos de inferencia dependen del hardware.",
        "Una sola ejecución por configuración; sin validación cruzada ni pruebas de significancia.",
        "No se implementaron Docker, Grad-CAM, pytest, fine-tuning ni detección de drift.",
    ], styles))

    doc.build(story)
    print(f"[make_pdf] Informe PDF generado -> {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
