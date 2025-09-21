
# -*- coding: utf-8 -*-
import io, os, sys, json, tempfile
import numpy as np
import streamlit as st
from urllib.parse import urlparse
import urllib.request as urlreq
from typing import List, Dict, Any

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from PIL import Image

st.set_page_config(page_title="Clasificador de leucocitos — modelo en prueba", layout="wide")

# ======================================================================
# CONFIGURACIÓN
# ======================================================================
# 1) Puedes fijar aquí una URL estable (GitHub Releases, etc.).
#    También puedes definir una variable de entorno DEFAULT_MODEL_URL.
DEFAULT_MODEL_URL = os.environ.get("DEFAULT_MODEL_URL", "")  # <-- coloca tu URL si deseas.

# 2) Nombre del archivo JSON con etiquetas en el repo (lista o dict).
LABELS_JSON_FILENAME = "labels.json"

# ======================================================================
# Utilidades
# ======================================================================
def drive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

@st.cache_data(show_spinner=False)
def fetch_url(url: str, token: str | None = None) -> Dict[str, Any]:
    """Descarga bytes de una URL. Si token, agrega cabecera Authorization.
    Retorna dict con ok, data, final_url, error.
    """
    try:
        req = urlreq.Request(url)
        if token:
            req.add_header("Authorization", f"token {token}")
        with urlreq.urlopen(req, timeout=60) as resp:
            data = resp.read()
            final_url = resp.geturl()
        return {"ok": True, "data": data, "final_url": final_url, "error": None}
    except Exception as e:
        return {"ok": False, "data": None, "final_url": "", "error": str(e)}

def save_tempfile(data: bytes, suffix: str = ".keras") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path

@st.cache_resource(show_spinner=True)
def load_model_from_bytes(b: bytes):
    tmp = save_tempfile(b, suffix=".keras")
    try:
        model = keras.models.load_model(tmp, compile=False)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass
    return model

def load_rgb_from_upload(file) -> np.ndarray:
    im = Image.open(file).convert("RGB")
    im = im.resize((224, 224))
    return np.array(im)

def preprocess_rescale(rgb: np.ndarray) -> tf.Tensor:
    x = tf.convert_to_tensor(rgb, dtype=tf.float32) / 255.0
    x = tf.expand_dims(x, axis=0)  # (1,224,224,3)
    return x

# ----------------------------------------------------------------------
# Etiquetas desde labels.json
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_labels_json(path: str) -> List[str] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Aceptar varios formatos:
        # a) ["Neutrófilo", "Linfocito", ...]
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return obj
        # b) { "0": "Neutrófilo", "1": "Linfocito", ... }  (claves dígitos)
        if isinstance(obj, dict):
            # si todas las claves son dígitos, ordenar por índice
            if all(str(k).isdigit() for k in obj.keys()):
                pairs = sorted(((int(k), v) for k, v in obj.items()), key=lambda p: p[0])
                return [v for _, v in pairs]
            # c) {"Neutrófilo": 0, "Linfocito": 1, ...}  (valores dígitos)
            if all(isinstance(v, int) for v in obj.values()):
                pairs = sorted(((int(v), k) for k, v in obj.items()), key=lambda p: p[0])
                return [k for _, k in pairs]
    except Exception:
        pass
    return None

def get_query_param(param: str) -> str | None:
    try:
        qp = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
        if isinstance(qp, dict):
            v = qp.get(param)
            if isinstance(v, list):
                return v[0]
            return v
    except Exception:
        pass
    return None

# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.header("Modelo y opciones")
origin = st.sidebar.radio(
    "Origen del modelo",
    ["Subir archivo", "URL directa", "Google Drive (ID)", "Ruta local (servidor)"],
    index=1,
    key="origin_choice",
)

model_url_input = st.sidebar.text_input(
    "URL del modelo (keras/h5/zip)",
    "",
    key="model_url_input",
    help="Ej.: https://github.com/<owner>/<repo>/releases/download/<tag>/modelo_final.keras",
)
gh_token = st.sidebar.text_input("Token (opcional si es privado)", value="", type="password", key="gh_token")
drive_id = st.sidebar.text_input("ID de Google Drive (si aplica)", value="", key="drive_id")
local_path = st.sidebar.text_input("Ruta local (solo servidor)", value="", key="local_path")

st.sidebar.divider()
use_labels_json = False
labels_from_json = None
labels_path = Path(__file__).with_name(LABELS_JSON_FILENAME)
if labels_path.exists():
    labels_from_json = read_labels_json(str(labels_path))
    if labels_from_json:
        with st.sidebar.expander("Etiquetas desde labels.json", expanded=True):
            st.caption(f"Se detectó {LABELS_JSON_FILENAME} en el repo.")
            st.code(json.dumps(labels_from_json, ensure_ascii=False, indent=2))
            use_labels_json = st.checkbox("Usar etiquetas de labels.json", value=True, key="use_labels_json")

labels_str_default = "Neutrófilo,Linfocito,Monocito,Eosinófilo,Basófilo"
labels_str = st.sidebar.text_input(
    "Etiquetas de clases (orden del modelo)",
    ",".join(labels_from_json) if (labels_from_json and use_labels_json) else labels_str_default,
    key="class_labels",
)
CLASS_NAMES = [s.strip() for s in labels_str.split(",") if s.strip()]
thresh = st.sidebar.slider("Umbral de 'Indeterminado'", 0.0, 1.0, 0.50, 0.01, key="conf_threshold")

# ======================================================================
# RESOLVER URL EFECTIVA
# ======================================================================
query_model_url = get_query_param("model_url")
effective_url = ""

if origin == "URL directa":
    effective_url = model_url_input or query_model_url or DEFAULT_MODEL_URL
elif origin == "Google Drive (ID)":
    if drive_id.strip():
        effective_url = drive_direct_url(drive_id.strip())
elif origin == "Ruta local (servidor)":
    effective_url = local_path
else:
    effective_url = ""  # Upload de archivo

# ======================================================================
# CARGA MODELO
# ======================================================================
st.header("Clasificador de leucocitos — modelo en prueba")

banner = st.empty()
model = None

if origin != "Subir archivo":
    if not effective_url:
        banner.info("Cargue el modelo desde la barra lateral o defina DEFAULT_MODEL_URL.")
    else:
        fi = fetch_url(effective_url, gh_token if gh_token else None)
        if not fi["ok"]:
            st.warning(f"Autocarga fallida: {fi['error']}")
        else:
            banner.success("Modelo cargado correctamente desde URL.")
            model = load_model_from_bytes(fi["data"])
else:
    up_model = st.sidebar.file_uploader("Subir modelo (.keras/.h5/.zip)", type=["keras", "h5", "zip"], key="model_uploader")
    if up_model is not None:
        b = up_model.read()
        model = load_model_from_bytes(b)
        banner.success("Modelo cargado correctamente desde archivo.")

if model is None:
    st.info("Cargue un modelo para habilitar la prueba rápida.")
    st.stop()

# Mostrar forma de entrada si está disponible
try:
    ishape = model.inputs[0].shape
    st.caption(f"Input shape: {tuple(ishape)}")
except Exception:
    pass

# ======================================================================
# PRUEBA RÁPIDA (ÚNICA, CORRECTAMENTE INDENTADA)
# ======================================================================
st.subheader("Prueba rápida (opcional)")
col1, col2 = st.columns([1, 2])

with col1:
    up_img = st.file_uploader(
        "Sube una imagen para probar",
        type=["png", "jpg", "jpeg"],
        key="test_image_uploader",
    )

with col2:
    if up_img is not None:
        rgb = load_rgb_from_upload(up_img)
        st.image(rgb, caption="Entrada (RGB, 224x224)", width=256)

        x = preprocess_rescale(rgb)
        y = model(x, training=False).numpy()

        # Softmax solo si hace falta
        if y.ndim == 2 and not np.allclose(np.sum(y, axis=1), 1.0, atol=1e-3):
            y = tf.nn.softmax(y, axis=-1).numpy()

        probs = y[0]

        # Alinear número de etiquetas y clases del modelo
        num_classes = probs.shape[-1]
        if len(CLASS_NAMES) < num_classes:
            CLASS_NAMES += [f"Clase {i}" for i in range(len(CLASS_NAMES), num_classes)]
        elif len(CLASS_NAMES) > num_classes:
            CLASS_NAMES = CLASS_NAMES[:num_classes]
        st.sidebar.caption(f"Salidas del modelo: {num_classes} clases")

        # Top-K
        topk = np.argsort(-probs)[:5]
        st.write("Top-K:")
        for i in topk:
            p = float(probs[i])
            nombre = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Clase {int(i)}"
            if p < thresh:
                nombre = f"{nombre} (indeterminado)"
            st.write(f"{nombre} → {p:.3f}")
            st.progress(min(max(p, 0.0), 1.0))

        # Top-1
        pred = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else f"Clase {pred}"
        st.caption(f"Predicción principal: {pred_name} (p={probs[pred]:.3f})")
