# ==========================================
# Clasificador de leucocitos (WBC) - Streamlit
# ==========================================
# Compat: TensorFlow/Keras modelo .keras o SavedModel empaquetado
# Autor: Sabino + asistente de docencia
# ------------------------------------------

import os, io, json, hashlib, zipfile, requests
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
# Utilidades de descarga/caché
# ==============================
def _sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def _download_with_progress(url: str, dst: Path, chunk=1 << 20):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    prog = st.progress(0, text=f"Descargando modelo desde {url} …")
    written = 0
    with open(dst, "wb") as f:
        for data in r.iter_content(chunk_size=chunk):
            f.write(data)
            written += len(data)
            if total:
                prog.progress(min(written / total, 1.0))
    prog.empty()

@st.cache_resource(show_spinner=False)
def load_model_from_local(path: Path) -> tf.keras.Model:
    # Carga .keras / .h5 / SavedModel directory
    if path.suffix in [".keras", ".h5", ".hdf5"]:
        return tf.keras.models.load_model(path)
    # SavedModel descomprimido (carpeta con saved_model.pb)
    if (path / "saved_model.pb").exists():
        return tf.keras.models.load_model(path)
    # SavedModel .zip -> descomprimir y cargar
    if path.suffix == ".zip":
        target_dir = MODEL_DIR / (path.stem + "_extracted")
        if not target_dir.exists():
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(target_dir)
        return tf.keras.models.load_model(target_dir)
    # Fallback
    raise ValueError(f"Formato no soportado en: {path}")

@st.cache_resource(show_spinner=False)
def ensure_model_by_url(url: str, expected_sha256: str = "") -> tf.keras.Model:
    filename = url.split("/")[-1] or "modelo.bin"
    local_path = MODEL_DIR / filename
    if not local_path.exists():
        _download_with_progress(url, local_path)
    if expected_sha256:
        calc = _sha256sum(local_path)
        if calc != expected_sha256.lower():
            raise ValueError(
                f"SHA256 no coincide.\nEsperado: {expected_sha256}\nCalculado: {calc}"
            )
    return load_model_from_local(local_path)

# ==============================
# Gestión de etiquetas
# ==============================
def parse_labels_text(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]

@st.cache_data(show_spinner=False)
def load_labels_from_json_bytes(b: bytes) -> List[str]:
    data = json.loads(b.decode("utf-8"))
    # Admite {"labels":[...]} o lista simple
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
    """Devuelve un tensor (1, H, W, 3) float32."""
    img = img.convert("RGB")
    if center_crop:
        # recorte cuadrado central
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
# UI
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
    model_info = {}

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
        st.markdown("**GitHub Release (recomendado para >200 MB)**")
        owner = st.text_input("Owner", value="tu_usuario")
        repo = st.text_input("Repo", value="tu_repo")
        tag = st.text_input("Tag (release)", value="v1.0.0")
        asset = st.text_input("Nombre del asset", value="modelo_final.keras")
        expected = st.text_input("SHA-256 (opcional)", value="")
        url_default = f"https://github.com/{owner}/{repo}/releases/download/{tag}/{asset}"
        url = st.text_input("URL del modelo", value=url_default)
        if st.button("Descargar y cargar modelo"):
            with st.spinner("Descargando y cargando…"):
                model = ensure_model_by_url(url, expected_sha256=expected.strip())
                model_ready = True
        st.caption("También puedes pegar cualquier URL directa (S3/HF/Drive export=download).")

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
        # Inferir tamaño de entrada (H, W) del primer tensor
        try:
            # Typical Keras: model.inputs[0].shape = (None, H, W, C)
            in_shape = model.inputs[0].shape
            H = int(in_shape[1]); W = int(in_shape[2])
            C = int(in_shape[3]) if len(in_shape) > 3 and in_shape[3] is not None else 3
        except Exception:
            # Fallback razonable
            H, W, C = 224, 224, 3
        st.info(f"Tamaño de entrada del modelo: {(H, W, C)}")

        if img is not None and labels:
            x = preprocess_image(img, (W, H), rescale_255=rescale, center_crop=center_crop)
            with st.spinner("Calculando predicción…"):
                preds = model.predict(x, verbose=0)
            # Asegurar vector 1D
            if preds.ndim == 2 and preds.shape[0] == 1:
                preds = preds[0]
            # Si la salida no es softmax, aplicar para mostrar probabilidades
            if np.any(preds < 0) or not np.isclose(np.sum(preds), 1.0, atol=1e-3):
                probs = softmax(preds)
            else:
                probs = preds

            # Alinear etiquetas
            n_classes = len(labels)
            if len(probs) != n_classes:
                st.warning(
                    f"El modelo devuelve {len(probs)} logits, pero hay {n_classes} etiquetas. "
                    "Verifica el orden/longitud de labels."
                )
            m = min(len(probs), n_classes)
            pairs = list(zip(labels[:m], probs[:m].tolist()))
            pairs.sort(key=lambda t: t[1], reverse=True)

            st.success(f"**Predicción top-1:** {pairs[0][0]}  "
                       f"({pairs[0][1]*100:.1f}%)" if pairs else "Sin predicción.")
            st.markdown("**Top-5**")
            for name, p in pairs[:5]:
                st.write(f"- {name}: {p*100:.2f}%")

        elif img is None:
            st.info("Carga una imagen para predecir.")
        else:
            st.info("Configura las etiquetas antes de predecir.")
    else:
        st.info("Carga el modelo en la barra lateral.")

# Pie
st.caption("© 2025 — Clasificador WBC para docencia e investigación.")
