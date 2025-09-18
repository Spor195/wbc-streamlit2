# app_wbc_streamlit.py
# ------------------------------------------------------------
# Clasificador de leucocitos (WBC) con un modelo ya entrenado
# Publicar con:  streamlit run app_wbc_streamlit.py
# ------------------------------------------------------------
import io
import os
import zipfile
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import streamlit as st

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Clasificador de leucocitos (WBC)", layout="centered")

# ---------------------------
# Utilidades
# ---------------------------
def center_square_crop(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return pil_img.crop((left, top, left + s, top + s))

def preprocess_image(pil_img: Image.Image, img_size: int, to_rgb: bool = True, rescale_255: bool = True):
    """Recorte cuadrado centrado, resize y normalización simple (x/255)."""
    if pil_img.mode not in ("RGB", "RGBA", "L", "I;16"):
        pil_img = pil_img.convert("RGB")
    if to_rgb and pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img = center_square_crop(pil_img)
    img = img.resize((img_size, img_size), Image.BICUBIC)
    arr = np.array(img).astype("float32")
    if rescale_255:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def save_uploaded_to_tmp(uploaded, suffix=""):
    """Guarda un archivo subido en un archivo temporal y retorna la ruta."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def try_load_savedmodel_from_zip(zip_path: str):
    """Intenta extraer un SavedModel .zip y retorna el directorio extraído."""
    dest_dir = tempfile.mkdtemp(prefix="savedmodel_")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    # buscar carpeta con saved_model.pb
    for root, dirs, files in os.walk(dest_dir):
        if "saved_model.pb" in files:
            return root
    return None

def load_keras_model(uploaded_file):
    """Carga .keras/.h5 o SavedModel (zip)."""
    if uploaded_file is None:
        return None, "Cargue su modelo (.keras/.h5) o SavedModel (.zip)."
    name = uploaded_file.name.lower()
    if name.endswith((".keras", ".h5", ".hdf5")):
        path = save_uploaded_to_tmp(uploaded_file, suffix=os.path.splitext(name)[1])
        model = load_model(path, compile=False)
        return model, f"Modelo Keras cargado desde: {Path(path).name}"
    elif name.endswith(".zip"):
        path = save_uploaded_to_tmp(uploaded_file, suffix=".zip")
        savedmodel_dir = try_load_savedmodel_from_zip(path)
        if savedmodel_dir is None:
            return None, "No se encontró 'saved_model.pb' dentro del zip."
        model = tf.keras.models.load_model(savedmodel_dir, compile=False)
        return model, f"SavedModel cargado desde carpeta: {savedmodel_dir}"
    else:
        return None, "Formato no soportado. Use .keras, .h5 o un SavedModel .zip."

def load_labels(uploaded_json, manual_text):
    """Lee etiquetas desde JSON o desde un cuadro de texto (una por línea)."""
    if uploaded_json is not None:
        try:
            labels = list(json.loads(uploaded_json.read()))
            return labels, "Etiquetas cargadas desde JSON."
        except Exception as e:
            return None, f"Error al leer JSON: {e}"
    # si no hay JSON, usar manual
    manual_text = manual_text.strip()
    if manual_text:
        labels = [line.strip() for line in manual_text.splitlines() if line.strip()]
        return labels, "Etiquetas cargadas desde el listado manual."
    return None, "Defina las etiquetas (JSON o listado)."

def softmax_if_needed(x):
    """Si la última capa no es softmax, aplica softmax para probabilidades legibles."""
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    # Intentar detectar si ya es softmax (suma ~1 y >=0)
    if np.all(x >= -1e-6) and np.isclose(np.sum(x), 1.0, atol=1e-3):
        return x
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

def grad_cam(input_img, model, last_conv_name=None, class_index=None):
    """
    Grad-CAM genérico. Puede fallar según arquitectura; se maneja con try/except en UI.
    Devuelve un heatmap en formato PIL.Image del mismo tamaño que input_img.
    """
    # inferir última conv si no se provee nombre
    if last_conv_name is None:
        # Heurística: último layer con 4D output
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    last_conv_name = layer.name
                    break
            except Exception:
                continue
    if last_conv_name is None:
        raise RuntimeError("No se halló una capa convolucional válida.")
    conv_layer = model.get_layer(last_conv_name)

    # Preparar gradiente
    img_arr = np.array(input_img.resize(model.input_shape[1:3], Image.BICUBIC)).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, 0)

    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_arr)
        if class_index is None:
            class_index = int(np.argmax(predictions[0]))
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        heatmap = np.zeros_like(heatmap)
    else:
        heatmap /= np.max(heatmap)

    # Redimensionar al tamaño de la imagen original
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(input_img.size, Image.BICUBIC).convert("L")
    return heatmap_img

def overlay_heatmap_on_image(pil_img, heatmap_img, alpha=0.35):
    heatmap_rgb = ImageOps.colorize(heatmap_img, black="black", white="red")
    return Image.blend(pil_img.convert("RGB"), heatmap_rgb.convert("RGB"), alpha=alpha)

# ---------------------------
# Barra lateral (config)
# ---------------------------
st.sidebar.header("Configuración del modelo")
uploaded_model = st.sidebar.file_uploader("Modelo (.keras / .h5) o SavedModel (.zip)", type=["keras", "h5", "hdf5", "zip"])

with st.sidebar.expander("Etiquetas de clases", expanded=True):
    uploaded_labels = st.file_uploader("labels.json (opcional)", type=["json"], key="labels_json")
    manual_labels = st.text_area("O pegue un listado (una etiqueta por línea):", value="Neutrófilo\nLinfocito\nMonocito\nEosinófilo\nBasófilo")

with st.sidebar.expander("Preprocesamiento", expanded=True):
    img_size = st.slider("Tamaño de entrada (px)", 64, 512, 224, step=16)
    rescale = st.checkbox("Dividir por 255 (normalización simple)", value=True)
    show_gradcam = st.checkbox("Generar Grad‑CAM (experimental)", value=False)

# ---------------------------
# Título
# ---------------------------
st.title("Clasificador de leucocitos (WBC)")

st.markdown("""
Cargue su **modelo entrenado** y una **imagen de un leucocito** para obtener la predicción.
- Modelos soportados: **Keras** (`.keras`, `.h5`) o **SavedModel** comprimido (`.zip`).
- Defina las **etiquetas** en `labels.json` (lista JSON) o manualmente.
""")

# Cargar modelo
model, model_msg = load_keras_model(uploaded_model) if uploaded_model else (None, "Modelo aún no cargado.")
st.info(model_msg)

# Etiquetas
labels, labels_msg = load_labels(uploaded_labels, manual_labels)
if labels is None:
    st.warning(labels_msg)
else:
    st.success(labels_msg + f"  ({len(labels)} clases)")

# ---------------------------
# Carga de imagen y predicción
# ---------------------------
uploaded_image = st.file_uploader("Imagen del leucocito (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    pil_img = Image.open(uploaded_image).convert("RGB")
    st.image(pil_img, caption="Imagen cargada", use_container_width=True)

    if model is not None and labels is not None:
        # Preprocesar
        arr = preprocess_image(pil_img, img_size=img_size, to_rgb=True, rescale_255=rescale)

        # Predicción
        preds = model.predict(arr, verbose=0)
        probs = softmax_if_needed(preds)
        if probs.ndim == 2:
            probs = probs[0]
        # Ajustar etiquetas si dimensiones no coinciden
        if len(labels) != probs.shape[-1]:
            st.error(f"El modelo devuelve {probs.shape[-1]} salidas, pero hay {len(labels)} etiquetas.")
        else:
            top_idx = int(np.argmax(probs))
            top_label = labels[top_idx]
            top_prob = float(probs[top_idx])

            st.subheader("Diagnóstico")
            st.write(f"**{top_label}** (probabilidad estimada: **{top_prob:.3f}**)")

            # Top‑k tabla
            k = min(5, len(labels))
            order = np.argsort(probs)[::-1][:k]
            st.markdown("**Top‑5**")
            st.dataframe({
                "Clase": [labels[i] for i in order],
                "Probabilidad": [float(probs[i]) for i in order]
            })

            # Grad‑CAM (opcional)
            if show_gradcam:
                try:
                    heat = grad_cam(pil_img, model, last_conv_name=None, class_index=top_idx)
                    overlay = overlay_heatmap_on_image(pil_img, heat, alpha=0.35)
                    st.markdown("**Grad‑CAM (regiones de contribución):**")
                    st.image(overlay, use_container_width=True)
                except Exception as e:
                    st.warning(f"No fue posible generar Grad‑CAM: {e}")
    else:
        st.info("Cargue el modelo y las etiquetas para habilitar la predicción.")

# ---------------------------
# Pie de página
# ---------------------------
st.caption("© 2025 — Clasificador WBC para docencia e investigación.")
