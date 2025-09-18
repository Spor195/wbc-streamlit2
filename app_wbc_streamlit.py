# app_wbc_streamlit.py
# ------------------------------------------------------
# Clasificador de leucocitos con carga de modelo flexible
# (c) Sabino – versión con botón "Cargar ahora" y depuración
# ------------------------------------------------------

import os, io, json, zipfile, tempfile
import urllib.request, urllib.error
import numpy as np
from PIL import Image
import streamlit as st

# Intento perezoso de TF (si no está instalado, se informa claramente)
try:
    import tensorflow as tf
except Exception as e:
    st.error(f"No se pudo importar TensorFlow: {e}")
    st.stop()

# ------------------------------------------------------
# Configuración de página
# ------------------------------------------------------
st.set_page_config(page_title="Clasificación de leucocitos", layout="centered")
st.title("Clasificación de leucocitos")
st.caption("Sube tu modelo y una imagen. Ajusta el preprocesamiento hasta reproducir tu entrenamiento.")

# ------------------------------------------------------
# Utilidades
# ------------------------------------------------------
def _fetch_bytes(url: str, bearer_token: str | None = None) -> tuple[bytes, dict]:
    """Descarga bytes de una URL con cabeceras adecuadas (útil para GitHub Releases)."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "streamlit-wbc/1.0")
    req.add_header("Accept", "application/octet-stream")
    if bearer_token:
        req.add_header("Authorization", f"Bearer {bearer_token}")
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        info = {
            "status": getattr(resp, "status", 200),
            "final_url": resp.geturl(),
            "length": len(data),
        }
        return data, info

@st.cache_resource(show_spinner=False)
def load_model_from_bytes(model_bytes: bytes, filename: str):
    """Carga .keras/.h5 o SavedModel.zip desde bytes a un dir temporal."""
    suffix = os.path.splitext(filename)[1].lower()
    tmpdir = tempfile.mkdtemp(prefix="wbc_model_")

    if suffix in (".keras", ".h5"):
        path = os.path.join(tmpdir, filename)
        with open(path, "wb") as f:
            f.write(model_bytes)
        model = tf.keras.models.load_model(path)
        return model

    if suffix == ".zip":
        zpath = os.path.join(tmpdir, filename)
        with open(zpath, "wb") as f:
            f.write(model_bytes)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmpdir)
        # buscar carpeta con saved_model.pb
        sm_dir = None
        for root, _, files in os.walk(tmpdir):
            if "saved_model.pb" in files:
                sm_dir = root
                break
        if sm_dir is None:
            raise ValueError("El ZIP no contiene un SavedModel válido (falta saved_model.pb).")
        model = tf.keras.models.load_model(sm_dir)
        return model

    raise ValueError("Formato de modelo no soportado. Usa .keras, .h5 o SavedModel .zip.")

def infer_target_size(model):
    ishape = getattr(model, "input_shape", None)
    if isinstance(ishape, (list, tuple)) and len(ishape) == 4:
        _, h, w, c = ishape
        if isinstance(h, int) and isinstance(w, int) and c in (1, 3):
            return (h, w), c
    return (224, 224), 3

def preprocess_image(img: Image.Image, size=(224, 224), mode="1/255", channels=3):
    # asegurar canales
    img = img.convert("RGB") if channels == 3 else img.convert("L")
    img = img.resize(size, Image.BILINEAR)
    x = np.array(img).astype("float32")
    if channels == 1:
        x = np.expand_dims(x, axis=-1)

    if mode == "1/255":
        x = x / 255.0
    elif mode == "EfficientNet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        x = preprocess_input(x)
    elif mode == "VGG/ResNet (caffe)":
        x = x[:, :, ::-1]  # RGB->BGR
        mean = np.array([103.939, 116.779, 123.68], dtype="float32")
        x = x - mean
    elif mode == "Sin normalizar":
        pass
    else:
        raise ValueError("Modo de preprocesamiento desconocido.")
    x = np.expand_dims(x, axis=0)
    return x

def load_class_names(file) -> list:
    """Lee etiquetas desde .txt (una por línea) o .json (lista)."""
    name = file.name.lower()
    data = file.read()
    try:
        if name.endswith(".json"):
            return list(json.loads(data.decode("utf-8")))
        else:
            lines = data.decode("utf-8").strip().splitlines()
            return [ln.strip() for ln in lines if ln.strip()]
    except Exception:
        return []

# ------------------------------------------------------
# Barra lateral — Modelo y opciones
# ------------------------------------------------------
st.sidebar.header("Modelo y opciones")

src = st.sidebar.radio(
    "Origen del modelo",
    ["Subir archivo", "URL directa", "GitHub Release (público)", "Ruta local (servidor)"],
    index=2
)

mfile = None
model = None
model_bytes = None
model_name = None
fetch_info = None
err_loading = None

try:
    if src == "Subir archivo":
        mfile = st.sidebar.file_uploader("Modelo (.keras, .h5 o SavedModel .zip)", type=["keras", "h5", "zip"])
        if mfile is not None and st.sidebar.button("Cargar ahora", use_container_width=True):
            model_bytes = mfile.read()
            model_name = mfile.name

    elif src == "URL directa":
        url = st.sidebar.text_input("URL del modelo (.keras/.h5/.zip)", placeholder="https://...")
        token = st.sidebar.text_input("Token (opcional si es privado)", type="password")
        if url and st.sidebar.button("Cargar ahora", use_container_width=True):
            model_bytes, fetch_info = _fetch_bytes(url, bearer_token=token or None)
            model_name = os.path.basename(url.split("?")[0])

    elif src == "GitHub Release (público)":
        gh_user = st.sidebar.text_input("Usuario/Org", placeholder="miusuario")
        gh_repo = st.sidebar.text_input("Repositorio", placeholder="wbc_streamlit")
        gh_tag  = st.sidebar.text_input("Tag de release", placeholder="v1.0.0")
        gh_asset = st.sidebar.text_input("Nombre del asset", placeholder="modelo.keras")
        url = None
        if gh_user and gh_repo and gh_tag and gh_asset:
            url = f"https://github.com/{gh_user}/{gh_repo}/releases/download/{gh_tag}/{gh_asset}"
            st.sidebar.caption(f"URL generada: {url}")
        if url and st.sidebar.button("Cargar ahora", use_container_width=True):
            model_bytes, fetch_info = _fetch_bytes(url)
            model_name = gh_asset

    elif src == "Ruta local (servidor)":
        local_path = st.sidebar.text_input("Ruta absoluta en el servidor", placeholder="/ruta/a/modelo.keras")
        if local_path and st.sidebar.button("Cargar ahora", use_container_width=True):
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    model_bytes = f.read()
                model_name = os.path.basename(local_path)
            else:
                st.sidebar.warning("La ruta indicada no existe en este entorno.")

    # Informes de descarga (si aplican)
    if fetch_info:
        st.sidebar.info(f"Descarga OK · HTTP {fetch_info.get('status')} · bytes: {fetch_info.get('length'):,}")
        st.sidebar.caption(f"URL final: {fetch_info.get('final_url')}")

    # Cargar modelo si tenemos bytes y nombre
    if model_bytes and model_name:
        try:
            model = load_model_from_bytes(model_bytes, model_name)
            st.sidebar.success(f"Modelo cargado: {model_name}")
            st.session_state["_model_loaded_ok"] = True
        except Exception as e:
            err_loading = f"{e}"

except Exception as e:
    err_loading = str(e)

if err_loading:
    st.sidebar.error(f"Error al cargar el modelo: {err_loading}")

# Resto de opciones
labels_file = st.sidebar.file_uploader("Etiquetas (.txt o .json)", type=["txt", "json"])
pp_mode = st.sidebar.selectbox("Preprocesamiento", ["1/255", "EfficientNet", "VGG/ResNet (caffe)", "Sin normalizar"])
threshold = st.sidebar.slider("Umbral de confianza para 'detectar'", 0.0, 0.99, 0.0, 0.01)
topk = st.sidebar.slider("Top-K a mostrar", 1, 10, 5, 1)
show_shapes = st.sidebar.checkbox("Mostrar shapes y depuración", value=True)

# ------------------------------------------------------
# Área principal — Imagen e inferencia
# ------------------------------------------------------
img_file = st.file_uploader("Imagen de leucocito (JPG/PNG)", type=["jpg", "jpeg", "png"])

class_names = []
if labels_file is not None:
    class_names = load_class_names(labels_file)
    if class_names:
        st.caption(f"Etiquetas cargadas: {len(class_names)}")

if model is not None:
    target_size, channels = infer_target_size(model)
    if show_shapes:
        st.write(f"**Input del modelo**: {model.input_shape} → tamaño inferido: {target_size}, canales: {channels}")

    if img_file is not None:
        try:
            img_bytes = img_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, caption="Imagen cargada", use_column_width=True)

            x = preprocess_image(img, size=target_size, mode=pp_mode, channels=channels)
            if show_shapes:
                st.write(f"**Shape del tensor de entrada**: {x.shape}")

            preds = model.predict(x, verbose=0)
            if preds.ndim == 2 and preds.shape[0] == 1:
                preds = preds[0]
            if preds.ndim > 1:
                st.warning(f"Salida con forma inusual: {preds.shape}. Se tomará argmax del último eje.")
                preds = preds.reshape(-1)

            # Convertir a probabilidades (softmax) si no parecen ya normalizadas
            if np.any(preds < 0) or np.sum(preds) <= 0.99 or np.sum(preds) >= 1.01:
                try:
                    from scipy.special import softmax
                    probs = softmax(preds)
                except Exception:
                    exps = np.exp(preds - np.max(preds))
                    probs = exps / np.sum(exps)
            else:
                probs = preds

            k = int(min(topk, probs.shape[0]))
            idxs = np.argsort(probs)[::-1][:k]
            top_labels = [class_names[i] if i < len(class_names) else f"clase_{i}" for i in idxs]
            top_probs = [float(probs[i]) for i in idxs]

            st.subheader("Resultados")
            for i, (lab, p) in enumerate(zip(top_labels, top_probs), 1):
                st.write(f"{i}. **{lab}** — {p:.3f}")

            best_i = int(idxs[0])
            best_p = float(probs[best_i])
            best_lab = top_labels[0]
            if best_p >= threshold:
                st.success(f"Predicción: **{best_lab}** (confianza {best_p:.3f} ≥ umbral {threshold:.2f})")
            else:
                st.warning(f"Sin clase sobre el umbral ({best_p:.3f} < {threshold:.2f}). Ajusta preprocesamiento/umbral o verifica etiquetas.")

            if show_shapes:
                st.write("— Depuración —")
                st.write({"target_size": target_size, "channels": channels, "preprocess": pp_mode, "num_clases": probs.shape[0]})

        except Exception as e:
            st.error(f"Error durante la inferencia: {e}")
else:
    st.info("Sube o carga tu modelo en la barra lateral para habilitar la predicción.")
