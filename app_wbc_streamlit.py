# ==========================================
# Clasificador de leucocitos (WBC) - Streamlit
# ==========================================
# Carga de modelo .keras/.h5 o SavedModel(.zip) desde:
# - Subida (≤200 MB) o
# - URL directa (GitHub Release/HF/S3/Drive) con caché y validación.
# ------------------------------------------

import io, os, re, json, time, zipfile, hashlib, requests
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ==============================
# Configuración general
# ==============================
st.set_page_config(page_title="Clasificador WBC", layout="wide")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Utilidades: hash y descarga robusta
# ==============================
def _sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def _auth_headers_for_url(url: str) -> dict:
    """Cabeceras adecuadas según dominio y secretos disponibles."""
    headers = {"User-Agent": "wbc-streamlit/1.0"}
    if "github.com" in url or "raw.githubusercontent.com" in url:
        token = st.secrets.get("GITHUB_TOKEN", "")
        if token:
            headers["Authorization"] = f"token {token}"
        headers["Accept"] = "application/octet-stream"
    # Puedes añadir aquí más dominios si fuese necesario.
    return headers

def _looks_like_html(path: Path) -> bool:
    try:
        head = open(path, "rb").read(512).lower()
        return (b"<html" in head) or (b"<!doctype html" in head)
    except Exception:
        return False

def _download_with_progress(url: str, dst: Path, chunk=1 << 20, max_retries: int = 3):
    """Descarga con progreso, redirecciones, cabeceras y reintentos."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            headers = _auth_headers_for_url(url)
            with requests.get(
                url, stream=True, timeout=90, allow_redirects=True, headers=headers
            ) as r:
                final_url = r.url
                total = int(r.headers.get("content-length", 0)) if r.headers.get("content-length") else 0
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code} al descargar.\nURL final: {final_url}")

                prog = st.progress(0.0, text=f"Descargando modelo…\n{final_url}")
                written = 0
                with open(dst, "wb") as f:
                    for data in r.iter_content(chunk_size=chunk):
                        if not data:
                            continue
                        f.write(data)
                        written += len(data)
                        if total:
                            prog.progress(min(written / total, 1.0))
                prog.empty()

                # Validaciones básicas
                if dst.stat().st_size < 1024 * 10 or _looks_like_html(dst):
                    raise RuntimeError(
                        "Se descargó contenido no válido (HTML o demasiado pequeño). "
                        "Revise si la URL es directa o requiere autenticación."
                    )
                return  # éxito
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"Fallo de descarga tras {max_retries} intentos: {last_err}")

# ==============================
# Carga de modelos (local/URL) con caché
# ==============================
def _load_keras_or_savedmodel(path: Path) -> tf.keras.Model:
    # .keras/.h5
    if path.suffix.lower() in [".keras", ".h5", ".hdf5"]:
        return tf.keras.models.load_model(path)

    # Carpeta SavedModel
    if path.is_dir() and (path / "saved_model.pb").exists():
        return tf.keras.models.load_model(path)

    # SavedModel.zip
    if path.suffix.lower() == ".zip":
        target_dir = MODEL_DIR / (path.stem + "_extracted")
        if not target_dir.exists():
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(target_dir)
        return tf.keras.models.load_model(target_dir)

    raise ValueError(f"Formato no soportado: {path}")

@st.cache_resource(show_spinner=False)
def load_model_from_local(path: Path) -> tf.keras.Model:
    return _load_keras_or_savedmodel(path)

@st.cache_resource(show_spinner=False)
def ensure_model_by_url(url: str, expected_sha256: str = "") -> tf.keras.Model:
    filename = url.split("/")[-1] or "modelo.bin"
    local_path = MODEL_DIR / filename
    if not local_path.exists():
        _download_with_progress(url, local_path)
    if expected_sha256:
        calc = _sha256sum(local_path)
        if calc.lower() != expected_sha256.lower():
            local_path.unlink(missing_ok=True)
            raise ValueError("SHA256 no coincide. Verifique la URL o el asset.")
    return _load_keras_or_savedmodel(local_path)

# ==============================
# Gestión de etiquetas
# ==============================
def parse_labels_text(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]

@st.cache_data(show_spinner=False)
def load_labels_from_json_bytes(b: bytes) -> List[str]:
    data = json.loads(b.decode("utf-8"))
    if isinstance(data, dict) and "labels" in data:
        return [str(x) for x in data["labels"]]
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("Formato de labels.json no reconocido.")

# ==============================
# Preprocesamiento de imagen
# ==============================
def preprocess_image(img: Image.Image, target_size: Tuple[int, int],
                     rescale_255: bool = True, center_crop: bool = False) -> np.ndarray:
    img = img.convert("RGB")
    if center_crop:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
    img = img.resize(target_size, Image.BILINEAR)
    x = np.asarray(img).astype("float32")
    if rescale_255:
        x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# ==============================
# Interfaz de usuario
# ==============================
st.title("Clasificador de leucocitos (WBC)")
st.caption("Docencia e investigación")

with st.sidebar:
    st.subheader("Configuración del modelo")
    load_mode = st.radio(
        "Fuente del modelo",
        ["Subir archivo (≤200 MB)", "URL directa / GitHub Release"],
        index=1
    )

    model: Optional[tf.keras.Model] = None
    model_ready = False

    if load_mode == "Subir archivo (≤200 MB)":
        up = st.file_uploader(
            "Selecciona .keras / .h5 / SavedModel .zip",
            type=["keras", "h5", "hdf5", "zip"],
            accept_multiple_files=False
        )
        if up:
            tmp_path = MODEL_DIR / up.name
            with open(tmp_path, "wb") as f:
                f.write(up.read())
            with st.spinner("Cargando modelo…"):
                model = load_model_from_local(tmp_path)
                model_ready = True
    else:
        st.markdown("**GitHub Release recomendado para >200 MB.**")
        owner = st.text_input("Owner", value="tu_usuario")
        repo = st.text_input("Repo", value="tu_repo")
        tag = st.text_input("Tag (release)", value="v1.0.0")
        asset = st.text_input("Nombre del asset", value="modelo_final.keras")
        expected = st.text_input("SHA-256 (opcional)", value="")
        url_default = f"https://github.com/{owner}/{repo}/releases/download/{tag}/{asset}"
        url = st.text_input("URL del modelo", value=url_default,
                            help="También acepta raw.githubusercontent.com, S3, Hugging Face o Drive (uc?export=download&id=...).")
        if st.button("Descargar y cargar modelo"):
            with st.spinner("Descargando y cargando…"):
                model = ensure_model_by_url(url, expected_sha256=expected.strip())
                model_ready = True
        if st.secrets.get("GITHUB_TOKEN", ""):
            st.caption("Se encontró GITHUB_TOKEN en Secrets: se usará para repos/releases privados.")
        else:
            st.caption("Si tu release es privada, añade GITHUB_TOKEN en Secrets.")

    st.divider()
    st.subheader("Etiquetas de clase")
    labels_source = st.radio("Origen de etiquetas", ["Manual", "labels.json"])
    labels: List[str] = []
    if labels_source == "labels.json":
        lj = st.file_uploader("Sube labels.json", type=["json"], accept_multiple_files=False)
        if lj:
            try:
                labels = load_labels_from_json_bytes(lj.read())
                st.success(f"{len(labels)} etiquetas cargadas.")
            except Exception as e:
                st.error(f"Error en labels.json: {e}")
    else:
        default_labels = "Neutrófilo,Eosinófilo,Basófilo,Linfocito,Monocito"
        raw = st.text_input("Listado separado por comas", value=default_labels)
        labels = parse_labels_text(raw)

    st.divider()
    st.subheader("Preprocesamiento")
    rescale = st.checkbox("Dividir por 255", value=True)
    center_crop = st.checkbox("Recorte central cuadrado", value=True)

# Panel principal
colA, colB = st.columns([1, 1.2])

with colA:
    st.markdown("### Imagen del leucocito")
    image_file = st.file_uploader("Arrastra una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
    img: Optional[Image.Image] = None
    if image_file:
        img = Image.open(io.BytesIO(image_file.read()))
        st.image(img, caption="Imagen cargada", use_container_width=True)

with colB:
    st.markdown("### Predicción")
    if model_ready:
        # Inferencia del tamaño de entrada
        try:
            in_shape = model.inputs[0].shape  # (None, H, W, C)
            H = int(in_shape[1]); W = int(in_shape[2])
            C = int(in_shape[3]) if len(in_shape) > 3 and in_shape[3] is not None else 3
        except Exception:
            H, W, C = 224, 224, 3
        st.info(f"Tamaño de entrada del modelo: {(H, W, C)}")

        if img is not None and labels:
            x = preprocess_image(img, (W, H), rescale_255=rescale, center_crop=center_crop)
            with st.spinner("Calculando predicción…"):
                preds = model.predict(x, verbose=0)

            if preds.ndim == 2 and preds.shape[0] == 1:
                preds = preds[0]
            probs = preds
            if np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0, atol=1e-3):
                probs = softmax(probs)

            n_classes = len(labels)
            if len(probs) != n_classes:
                st.warning(
                    f"El modelo devuelve {len(probs)} logits, pero hay {n_classes} etiquetas. "
                    "Verifique el orden y longitud de 'labels'."
                )
            m = min(len(probs), n_classes)
            pairs = list(zip(labels[:m], probs[:m].tolist()))
            pairs.sort(key=lambda t: t[1], reverse=True)

            if pairs:
                st.success(f"**Top-1:** {pairs[0][0]}  ({pairs[0][1]*100:.1f}%)")
                st.markdown("**Top-5**")
                for name, p in pairs[:5]:
                    st.write(f"- {name}: {p*100:.2f}%")
            else:
                st.info("Sin predicción disponible.")
        elif img is None:
            st.info("Cargue una imagen para predecir.")
        else:
            st.info("Configure las etiquetas antes de predecir.")
    else:
        st.info("Cargue el modelo en la barra lateral.")

# Pie de página
st.caption("© 2025 — Clasificador WBC para docencia e investigación.")
