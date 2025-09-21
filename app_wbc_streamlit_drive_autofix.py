# -*- coding: utf-8 -*-
# Clasificador de leucocitos — Carga de modelo robusta
# Autor: Sabino + asistente
# Requisitos (para Py 3.13 en Streamlit Cloud):
#   streamlit==1.36.0
#   tensorflow-cpu==2.20.0
#   numpy==2.1.1
#   pillow==10.4.0
#   h5py==3.12.1
#   protobuf>=5.28.0,<6
#   typing-extensions==4.12.2

from __future__ import annotations
import io
import os
import zipfile
import tempfile
import shutil
import time
from pathlib import Path
from typing import Tuple, Dict, Optional

import requests
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# -----------------------------------------------------------------------------
# Configuración de página
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Clasificador de leucocitos — modelo en prueba",
                   page_icon="🧫",
                   layout="wide")

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:,.0f} {unit}".replace(",", " ")
        n /= 1024.0
    return f"{n:.1f} TB"

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------------------------------------------------------
# Descarga de archivos (URL / Google Drive ID)
# -----------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "wbc-streamlit/1.0"})

@st.cache_data(show_spinner=False)
def fetch_url_cached(url: str, token: Optional[str] = None) -> Tuple[str, Dict]:
    """
    Descarga un recurso binario y devuelve:
      - ruta local al archivo
      - metadata útil (final_url, bytes, ts)
    Se cachea en disco del contenedor mientras viva el proceso.
    """
    t0 = time.time()
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    with SESSION.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = 0
        suffix = Path(url).name or "download.bin"
        fd, tmp_path = tempfile.mkstemp(prefix="wbc_", suffix=f"__{suffix}")
        os.close(fd)
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)

    info = {
        "final_url": r.url,
        "bytes": total,
        "elapsed_s": time.time() - t0,
        "ts": int(t0),
    }
    return tmp_path, info

def build_drive_direct_url(file_id: str) -> str:
    # URL directa oficial de descarga para Drive (sin librerías extra)
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download"

# -----------------------------------------------------------------------------
# Carga del modelo
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_path(model_path: str):
    """
    Carga .keras / .h5 o SavedModel contenido en .zip
    """
    p = Path(model_path)
    ext = p.suffix.lower()

    if ext in {".keras", ".h5"}:
        model = tf.keras.models.load_model(str(p), compile=False)
        return model

    if ext == ".zip":
        # Descomprimir en un directorio temporal y cargar SavedModel
        tmpdir = Path(tempfile.mkdtemp(prefix="wbc_savedmodel_"))
        with zipfile.ZipFile(str(p), "r") as zf:
            zf.extractall(tmpdir)
        # Buscar carpeta con saved_model.pb
        candidates = list(tmpdir.rglob("saved_model.pb"))
        if not candidates:
            raise ValueError("El .zip no contiene un SavedModel válido (saved_model.pb).")
        model_dir = candidates[0].parent
        model = tf.keras.models.load_model(str(model_dir), compile=False)
        return model

    raise ValueError(f"Extensión no soportada: {ext}. Usa .keras, .h5 o SavedModel .zip")

# -----------------------------------------------------------------------------
# Preprocesamiento de imágenes para pruebas rápidas (opcional)
# -----------------------------------------------------------------------------
IMG_SIZE = (224, 224)

def load_rgb_from_upload(uploaded) -> np.ndarray:
    img = Image.open(uploaded).convert("RGB")
    img = img.resize(IMG_SIZE, resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)

def preprocess_rescale(x_uint8: np.ndarray) -> np.ndarray:
    x = x_uint8.astype("float32") / 255.0
    return np.expand_dims(x, 0)  # (1, 224, 224, 3)

# -----------------------------------------------------------------------------
# Autocarga al arrancar
# -----------------------------------------------------------------------------
DEFAULT_MODEL_URL = (
    "https://github.com/Spor195/wbc_streamlit2/releases/download/v1.0.0/modelo_final.keras"
)

# Permitir override por query param ?model_url=
params = st.query_params
if "model_url" in params:
    st.session_state["last_url"] = params["model_url"]

auto_url = st.session_state.get("last_url") or DEFAULT_MODEL_URL

if "model" not in st.session_state:
    try:
        path, info = fetch_url_cached(auto_url, None)
        st.session_state["model"] = load_model_from_path(path)
        st.session_state["fetch_info"] = info
        st.session_state["last_url"] = info.get("final_url", auto_url)
    except Exception as e:
        # Arranca sin bloquear; el usuario puede cargar desde la barra
        st.info("Cargue el modelo desde la barra lateral.")
        st.caption(f"Autocarga fallida: {e}")

# -----------------------------------------------------------------------------
# UI — barra lateral (cargar modelo)
# -----------------------------------------------------------------------------
st.sidebar.header("Modelo y opciones")

origin = st.sidebar.radio(
    "Origen del modelo",
    ["Subir archivo", "URL directa", "Google Drive (ID)", "Ruta local (servidor)"],
    index=1,
)

token_direct = None
model_loaded_now = False
fetch_info_now: Optional[Dict] = None

if origin == "Subir archivo":
    up = st.sidebar.file_uploader(
        "Modelo (.keras/.h5/.zip)", type=["keras", "h5", "zip"]
    )
    if st.sidebar.button("Cargar ahora", disabled=(up is None)):
        if up is None:
            st.warning("Suba un archivo de modelo.")
        else:
            # Guardar a disco temporal y cargar
            fd, tmp = tempfile.mkstemp(prefix="wbc_upload_", suffix=f"__{up.name}")
            os.close(fd)
            with open(tmp, "wb") as f:
                shutil.copyfileobj(up, f)
            model = load_model_from_path(tmp)
            st.session_state["model"] = model
            fetch_info_now = {"final_url": f"upload://{up.name}", "bytes": up.size}
            st.session_state["fetch_info"] = fetch_info_now
            st.session_state["last_url"] = fetch_info_now["final_url"]
            model_loaded_now = True

elif origin == "URL directa":
    url_direct = st.sidebar.text_input(
        "URL del modelo (.keras/.h5/.zip)",
        value=auto_url,
        help="Pega la URL completa del asset (GitHub Releases o equivalente).",
    )
    token_direct = st.sidebar.text_input(
        "Token (opcional si es privado)", type="password"
    )
    if st.sidebar.button("Cargar ahora"):
        try:
            path, fetch_info_now = fetch_url_cached(url_direct, token_direct or None)
            model = load_model_from_path(path)
            st.session_state["model"] = model
            st.session_state["fetch_info"] = fetch_info_now
            st.session_state["last_url"] = fetch_info_now.get("final_url", url_direct)
            model_loaded_now = True
        except Exception as e:
            st.sidebar.error(f"Error al cargar: {e}")

elif origin == "Google Drive (ID)":
    file_id = st.sidebar.text_input("ID de archivo de Google Drive")
    if st.sidebar.button("Cargar desde Drive"):
        try:
            drive_url = build_drive_direct_url(file_id.strip())
            path, fetch_info_now = fetch_url_cached(drive_url, None)
            model = load_model_from_path(path)
            st.session_state["model"] = model
            st.session_state["fetch_info"] = fetch_info_now
            st.session_state["last_url"] = fetch_info_now.get("final_url", drive_url)
            model_loaded_now = True
        except Exception as e:
            st.sidebar.error(f"Descarga desde Drive falló: {e}")

else:  # Ruta local (servidor)
    local_path = st.sidebar.text_input("Ruta absoluta del archivo .keras/.h5/.zip")
    if st.sidebar.button("Cargar (ruta local)"):
        try:
            if not local_path:
                raise ValueError("Ingrese una ruta válida.")
            model = load_model_from_path(local_path)
            st.session_state["model"] = model
            fetch_info_now = {"final_url": f"file://{local_path}"}
            st.session_state["fetch_info"] = fetch_info_now
            st.session_state["last_url"] = fetch_info_now["final_url"]
            model_loaded_now = True
        except Exception as e:
            st.sidebar.error(f"Error al cargar desde ruta local: {e}")

# Mostrar resultado de la barra lateral
fi = st.session_state.get("fetch_info")
if fi and ("bytes" in fi):
    st.sidebar.success(
        f"Descarga OK · HTTP 200 · bytes: {fi['bytes']}"
    )
    st.sidebar.caption(f"URL final: {fi.get('final_url','')}")


#  Modificación para etiquetas
st.sidebar.divider()
labels_str = st.sidebar.text_input(
    "Etiquetas de clases (orden del modelo)",
    "Neutrófilo,Linfocito,Monocito,Eosinófilo,Basófilo"
)
CLASS_NAMES = [s.strip() for s in labels_str.split(",") if s.strip()]
thresh = st.sidebar.slider("Umbral de 'Indeterminado'", 0.0, 1.0, 0.50, 0.01)
# 


# -----------------------------------------------------------------------------
# UI — contenido principal
# -----------------------------------------------------------------------------
st.title("Clasificador de leucocitos — modelo en prueba")

model = st.session_state.get("model")
if model is None:
    st.info("Cargue el modelo desde la barra lateral.")
else:
    st.success("Modelo cargado correctamente.")
    # Mostrar input_shape
    try:
        if hasattr(model, "inputs") and model.inputs:
            shape = tuple(int(d) if d is not None else None for d in model.inputs[0].shape)
        else:
            # Keras 3 (Functional/Sequential) – fallback
            shape = tuple(model.get_layer(index=0).input_shape)
        st.write("**Input shape:**", shape)
    except Exception:
        st.write("**Input shape:** (no disponible)")

    # Prueba rápida de inferencia con imagen
    st.subheader("Prueba rápida (opcional)")
    col1, col2 = st.columns([1, 2])
    with col1:
        up_img = st.file_uploader("Sube una imagen para probar", type=["png", "jpg", "jpeg"])
    with col2:
        if up_img is not None:
            rgb = load_rgb_from_upload(up_img)
            st.image(rgb, caption="Entrada (RGB, 224x224)", width=256)
            x = preprocess_rescale(rgb)
            # predict (evitar training=True)
            y = model(x, training=False).numpy()
            # aplicar softmax si es necesario
            if y.ndim == 2 and not np.allclose(np.sum(y, axis=1), 1.0, atol=1e-3):
                y = tf.nn.softmax(y, axis=-1).numpy()
            probs = y[0]
            topk = np.argsort(-probs)[:5]
            st.write("Top-K:")
            for i in topk:
                p = float(probs[i])
                nombre = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Clase {int(i)}"
                if p < thresh:
                    nombre = f"{nombre} (indeterminado)"
                st.write(f"{nombre} → {p:.3f}")
                st.progress(min(max(p, 0.0), 1.0))

            pred = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else f"Clase {pred}"
            st.caption(f"Predicción principal: {pred_name} (p={probs[pred]:.3f})")
              


# Mensajes informativos del evento actual
if model_loaded_now and fi:
    st.toast("Modelo cargado y listo.", icon="✅")
