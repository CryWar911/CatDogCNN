"""Prueba de humo de la API (sin pytest).

Realiza peticiones reales a un servidor uvicorn en ejecución y guarda el resultado
individual de cada prueba en ``reports/api_test_results.json``.

Uso:
    python scripts/test_api.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
OUT_PATH = REPORTS / "api_test_results.json"


def multipart_body(filename: str, content: bytes, content_type: str):
    boundary = f"----prueba{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def request(method: str, url: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            elapsed = (time.perf_counter() - start) * 1000
            return resp.status, raw, elapsed, None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed = (time.perf_counter() - start) * 1000
        return exc.code, raw, elapsed, None
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - start) * 1000
        return None, b"", elapsed, str(exc)


def post_image(base_url: str, path: Path, filename: str | None = None,
               content_type: str | None = None, content: bytes | None = None):
    if content is None:
        content = path.read_bytes()
    if filename is None:
        filename = path.name
    if content_type is None:
        ext = path.suffix.lower()
        content_type = {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    body, ctype = multipart_body(filename, content, content_type)
    return request("POST", f"{base_url}/predict", body, {"Content-Type": ctype})


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    manifest = json.loads((REPORTS / "manifest.json").read_text(encoding="utf-8"))
    cat_tests = [it for it in manifest["splits"]["test"] if it["label"] == 0][:3]
    dog_tests = [it for it in manifest["splits"]["test"] if it["label"] == 1][:3]

    results = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "base_url": base,
        "pruebas": [],
    }

    def add(nombre, status, raw, elapsed, error, esperado=200):
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except Exception:  # noqa: BLE001
            body = raw.decode("utf-8", errors="replace")[:500] if raw else None
        item = {
            "prueba": nombre,
            "status_http": status,
            "status_esperado": esperado,
            "exito": bool(status == esperado and error is None),
            "error_transporte": error,
            "tiempo_ms": round(elapsed, 2),
            "respuesta": body,
        }
        results["pruebas"].append(item)
        print(f"[test_api] {nombre}: HTTP {status} -> {'OK' if item['exito'] else 'FALLO'}")
        return item

    # GET /
    status, raw, elapsed, error = request("GET", f"{base}/")
    item = add("GET / (interfaz web)", status, raw, elapsed, error)
    texto = raw.decode("utf-8", errors="ignore") if raw else ""
    item["contiene_html"] = "<!DOCTYPE html" in texto and "Clasificador" in texto
    item["exito"] = item["exito"] and item["contiene_html"]

    # GET /health
    status, raw, elapsed, error = request("GET", f"{base}/health")
    add("GET /health", status, raw, elapsed, error)

    # POST /predict con 3 gatos y 3 perros
    for i, item in enumerate(cat_tests, start=1):
        path = Path(item["path"])
        status, raw, elapsed, error = post_image(base, path)
        add(f"POST /predict gato {i} ({path.name})", status, raw, elapsed, error)
    for i, item in enumerate(dog_tests, start=1):
        path = Path(item["path"])
        status, raw, elapsed, error = post_image(base, path)
        add(f"POST /predict perro {i} ({path.name})", status, raw, elapsed, error)

    # Archivo inválido: texto plano con extensión .jpg
    invalid = b"esto no es una imagen, es texto plano"
    status, raw, elapsed, error = post_image(
        base, Path("archivo_invalido.jpg"), filename="archivo_invalido.jpg",
        content_type="image/jpeg", content=invalid,
    )
    add("POST /predict archivo inválido (texto con extensión .jpg)", status, raw, elapsed, error, esperado=400)

    # Extensión no permitida
    status, raw, elapsed, error = post_image(
        base, Path("archivo.txt"), filename="archivo.txt",
        content_type="text/plain", content=b"texto", 
    )
    add("POST /predict extensión no permitida (.txt)", status, raw, elapsed, error, esperado=415)

    # Archivo demasiado grande (> 5 MB)
    oversized = b"\x00" * (5 * 1024 * 1024 + 1024)
    status, raw, elapsed, error = post_image(
        base, Path("grande.jpg"), filename="grande.jpg",
        content_type="image/jpeg", content=oversized,
    )
    add("POST /predict archivo > 5 MB", status, raw, elapsed, error, esperado=413)

    # GET /stats
    status, raw, elapsed, error = request("GET", f"{base}/stats")
    add("GET /stats", status, raw, elapsed, error)

    results["resumen"] = {
        "total": len(results["pruebas"]),
        "exitosas": sum(1 for x in results["pruebas"] if x["exito"]),
        "fallidas": sum(1 for x in results["pruebas"] if not x["exito"]),
    }
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[test_api] Resultados -> {OUT_PATH}")
    print(f"[test_api] Resumen: {results['resumen']}")
    return results


if __name__ == "__main__":
    main()
