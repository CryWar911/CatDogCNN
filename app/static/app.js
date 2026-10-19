// Frontend del clasificador de perros y gatos.
// Usa exclusivamente rutas relativas ("/predict", "/stats") para funcionar
// tanto en local como detrás de un proxy.

const MAX_BYTES = 5 * 1024 * 1024;
const EXTENSIONES = [".jpg", ".jpeg", ".png", ".webp"];

const zonaSoltar = document.getElementById("zonaSoltar");
const inputArchivo = document.getElementById("inputArchivo");
const botonSeleccionar = document.getElementById("botonSeleccionar");
const botonClasificar = document.getElementById("botonClasificar");
const botonLimpiar = document.getElementById("botonLimpiar");
const zonaPreview = document.getElementById("zonaPreview");
const imagenPreview = document.getElementById("imagenPreview");
const nombreArchivo = document.getElementById("nombreArchivo");
const cargando = document.getElementById("cargando");
const mensajeError = document.getElementById("mensajeError");
const resultado = document.getElementById("resultado");
const iconoResultado = document.getElementById("iconoResultado");
const claseResultado = document.getElementById("claseResultado");
const barraConfianza = document.getElementById("barraConfianza");
const textoConfianza = document.getElementById("textoConfianza");
const avisoIncierta = document.getElementById("avisoIncierta");

let archivoActual = null;
let urlPreview = null;

// ------------------------------ Utilidades de UI ------------------------------
function mostrarError(mensaje) {
  mensajeError.textContent = mensaje;
  mensajeError.classList.remove("oculto");
}

function limpiarError() {
  mensajeError.textContent = "";
  mensajeError.classList.add("oculto");
}

function limpiarResultado() {
  resultado.classList.add("oculto");
  avisoIncierta.classList.add("oculto");
  barraConfianza.style.width = "0%";
}

function validarArchivo(archivo) {
  if (!archivo) return "No se seleccionó ningún archivo.";
  const nombre = (archivo.name || "").toLowerCase();
  const extension = EXTENSIONES.find((ext) => nombre.endsWith(ext));
  if (!extension) return "Formato no admitido. Usa JPG, JPEG, PNG o WebP.";
  if (!["image/jpeg", "image/png", "image/webp", "image/jpg"].includes(archivo.type)) {
    return "El tipo de archivo no es una imagen admitida.";
  }
  if (archivo.size > MAX_BYTES) {
    return "La imagen supera los 5 MB. Elige un archivo más pequeño.";
  }
  if (archivo.size === 0) return "El archivo está vacío.";
  return null;
}

function mostrarPreview(archivo) {
  if (urlPreview) URL.revokeObjectURL(urlPreview);
  urlPreview = URL.createObjectURL(archivo);
  imagenPreview.src = urlPreview;
  nombreArchivo.textContent = `${archivo.name} · ${(archivo.size / 1024).toFixed(1)} KB`;
  zonaPreview.classList.remove("oculto");
  zonaSoltar.classList.add("oculto");
  botonClasificar.disabled = false;
  botonLimpiar.disabled = false;
}

function seleccionarArchivo(archivo) {
  limpiarError();
  limpiarResultado();
  const error = validarArchivo(archivo);
  if (error) {
    archivoActual = null;
    mostrarError(error);
    return;
  }
  archivoActual = archivo;
  mostrarPreview(archivo);
}

function limpiarTodo() {
  archivoActual = null;
  if (urlPreview) {
    URL.revokeObjectURL(urlPreview);
    urlPreview = null;
  }
  imagenPreview.removeAttribute("src");
  nombreArchivo.textContent = "";
  zonaPreview.classList.add("oculto");
  zonaSoltar.classList.remove("oculto", "arrastrando");
  botonClasificar.disabled = true;
  botonLimpiar.disabled = true;
  inputArchivo.value = "";
  limpiarError();
  limpiarResultado();
}

// ------------------------------ Eventos de selección ------------------------------
botonSeleccionar.addEventListener("click", (evento) => {
  evento.stopPropagation();
  inputArchivo.click();
});

zonaSoltar.addEventListener("click", () => inputArchivo.click());

inputArchivo.addEventListener("change", () => {
  if (inputArchivo.files && inputArchivo.files[0]) {
    seleccionarArchivo(inputArchivo.files[0]);
  }
});

["dragenter", "dragover"].forEach((evento) => {
  zonaSoltar.addEventListener(evento, (e) => {
    e.preventDefault();
    e.stopPropagation();
    zonaSoltar.classList.add("arrastrando");
  });
});

["dragleave", "drop"].forEach((evento) => {
  zonaSoltar.addEventListener(evento, (e) => {
    e.preventDefault();
    e.stopPropagation();
    zonaSoltar.classList.remove("arrastrando");
  });
});

zonaSoltar.addEventListener("drop", (e) => {
  const archivos = e.dataTransfer && e.dataTransfer.files;
  if (archivos && archivos.length > 0) {
    seleccionarArchivo(archivos[0]);
  }
});

// ------------------------------ Clasificación ------------------------------
async function clasificar() {
  if (!archivoActual) {
    mostrarError("Primero selecciona una imagen.");
    return;
  }
  limpiarError();
  limpiarResultado();
  cargando.classList.remove("oculto");
  botonClasificar.disabled = true;
  botonLimpiar.disabled = true;

  try {
    const datos = new FormData();
    datos.append("file", archivoActual, archivoActual.name);

    const respuesta = await fetch("/predict", { method: "POST", body: datos });
    let cuerpo = null;
    try {
      cuerpo = await respuesta.json();
    } catch (_) {
      cuerpo = null;
    }

    if (!respuesta.ok) {
      const detalle = cuerpo && cuerpo.detail ? cuerpo.detail : `Error HTTP ${respuesta.status}`;
      throw new Error(detalle);
    }
    if (!cuerpo || !cuerpo.clase) {
      throw new Error("La respuesta del servidor no tiene el formato esperado.");
    }

    const esPerro = cuerpo.clase === "perro";
    const confianza = Number(cuerpo.confianza) || 0;
    const porcentaje = Math.round(confianza * 100);

    iconoResultado.textContent = esPerro ? "🐶" : "🐱";
    claseResultado.textContent = esPerro ? "Perro" : "Gato";
    barraConfianza.style.width = `${Math.max(0, Math.min(100, porcentaje))}%`;
    textoConfianza.textContent = `Confianza: ${porcentaje}%`;
    avisoIncierta.classList.toggle("oculto", cuerpo.incierta !== true);
    resultado.classList.remove("oculto");

    actualizarEstadisticas();
  } catch (error) {
    mostrarError(`No se pudo clasificar la imagen. ${error.message || "Inténtalo de nuevo."}`);
  } finally {
    cargando.classList.add("oculto");
    botonClasificar.disabled = false;
    botonLimpiar.disabled = false;
  }
}

botonClasificar.addEventListener("click", clasificar);
botonLimpiar.addEventListener("click", limpiarTodo);

// ------------------------------ Estadísticas ------------------------------
async function actualizarEstadisticas() {
  try {
    const respuesta = await fetch("/stats");
    if (!respuesta.ok) return;
    const datos = await respuesta.json();
    const total = datos.total_predicciones || 0;
    const conteo = datos.conteo || { gato: 0, perro: 0 };
    document.getElementById("statTotal").textContent = total;
    document.getElementById("statGato").textContent = `${conteo.gato ?? 0} (${Math.round((datos.proporcion_gato || 0) * 100)}%)`;
    document.getElementById("statPerro").textContent = `${conteo.perro ?? 0} (${Math.round((datos.proporcion_perro || 0) * 100)}%)`;
    document.getElementById("statConfianza").textContent = `${Math.round((datos.confianza_media || 0) * 100)}%`;
    document.getElementById("statLatencia").textContent = `${(datos.latencia_media_ms || 0).toFixed(1)} ms`;
    document.getElementById("statInciertas").textContent = `${(datos.porcentaje_inciertas || 0).toFixed(1)}%`;
  } catch (_) {
    // Las estadísticas son informativas: un fallo no debe afectar a la clasificación.
  }
}

actualizarEstadisticas();
