# app_wbc_streamlit.py
# ------------------------------------------------------
# Clasificador de leucocitos (Streamlit)
# Carga de modelo flexible: archivo, URL directa, GitHub Release (API), ruta local
# Robusto para assets grandes (.keras/.h5 o SavedModel.zip)
# ------------------------------------------------------

import os, io, json, zipfile, tempfile
import numpy as np
from PIL import Image
import streamlit as st

# =========================
# TensorFlow (carga perezosa)
# =========================
try:
    import tensorflow as tf
except Exception as e:
    st.error(f"No se pudo importar TensorFlow: {e}")
    st.stop()

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="Clasificación de leucocitos", layout="centered")
st.title("Clasificación de leucocitos")
st.caption("Sube tu modelo y una imagen. Ajusta el preprocesamiento hasta reproducir tu entrenamiento.")

# =========================
# Utilidades generales
# =========================
import urllib.request as _urlreq
import urllib.error as _urlerr

def _fetch_to_path(url: str, bearer_token: str | None = None) -> tuple[str, dict]:
    """
    Descarga en streaming a archivo temporal y devuelve (ruta, info).
    Robusto para archivos grandes. Usa cabeceras adecuadas (GitHub).
    """
    req = _urlreq.Request(url)
    req.add_header("User-Agent", "streamlit-wbc/1.0")
    # Si el asset es público, Accept octet-stream funciona bien; si no, se ignora.
    req.add_header("Accept", "application/octet-stream")
    if bearer_token:
        req.add_header("Authorization", f"Bearer {bearer_token}")

    tmpdir = tempfile.mkdtemp(prefix="wbc_dl_")
    fname = os.path.basename(url.split("?")[0]) or "model.bin"
    fpath = os.path.join(tmpdir, fname)

    total = 0
    try:
        with _urlreq.urlopen(req) as resp, open(fpath, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
            info = {
                "status": getattr(resp, "status", 200),
                "final_url": resp.geturl(),
                "length": total,
            }
        return fpath, info
    except _urlerr.HTTPError as e:
        raise RuntimeError(f"HTTPError {e.code}: {e.reason}")
    except _urlerr.URLError as e:
        raise RuntimeError(f"URLError: {e.reason}")

@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str):
    """
    Carga un modelo desde ruta:
    - .keras / .h5 -> load_model directo
    - .zip (SavedModel) -> descomprime y busca saved_model.pb
    """
    low = path.lower()
    if low.endswith((".keras", ".h5")):
        return tf.keras.models.load_model(path)
    if low.endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="wbc_sm_")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmpdir)
        sm_dir = None
        for root, _, files in os.walk(tmpdir):
            if "saved_model.pb" in files:
                sm_dir = root
                break
        if sm_dir is None:
            raise ValueError("El ZIP no contiene un SavedModel válido (falta saved_model.pb).")
        return tf.keras.models.load_model(sm_dir)
    raise ValueError("Extensión no soportada. Usa .keras, .h5 o SavedModel .zip.")

def infer_target_size(model):
    ishape = getattr(model, "input_shape", None)
    if isinstance(ishape, (list, tuple)) and len(ishape) == 4:
        _, h, w, c = ishape
        if isinstance(h, int) and isinstance(w, int) and c in (1, 3):
            return (h, w), c
    return (224, 224), 3

def preprocess_image(img: Image.Image, size=(224, 224), mode="1/255", channels=3):
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




# --- Descarga robusta desde Google Drive (maneja archivos grandes con confirmación) ---

# ======================================================
# Utilidades Google Drive (ID o URL) — normalización y descarga robusta
# ======================================================
import urllib.request as _urlreq, urllib.error as _urlerr, urllib.parse as _urlparse
import http.cookiejar as _cookielib
import re as _re

def _gdrive_extract_id(x: str) -> str:
    """Acepta ID o URL y devuelve el file_id (solo [A-Za-z0-9_-])."""
    s = (x or "").strip()
    s = s.strip().strip("{}[]() \n\r\t")
    m = _re.search(r"/file/d/([A-Za-z0-9_-]{20,})", s)
    if not m:
        m = _re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    if m:
        s = m.group(1)
    if not _re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        raise ValueError("El ID/URL de Drive no es válido. Pegue el ID exacto o una URL de Drive.")
    return s

def _fetch_gdrive_to_path(file_id: str) -> tuple[str, dict]:
    """Descarga un archivo grande de Google Drive con confirm token y reintentos."""
    import tempfile, os
    cj = _cookielib.CookieJar()
    opener = _urlreq.build_opener(_urlreq.HTTPCookieProcessor(cj))

    def _do(url):
        req = _urlreq.Request(url, headers={"User-Agent":"wbc-streamlit/1.1"})
        return opener.open(req)

    def _stream_to_tmp(resp, fname_hint="model.bin"):
        tmpdir = tempfile.mkdtemp(prefix="wbc_gd_")
        fname = fname_hint or "model.bin"
        fpath = os.path.join(tmpdir, fname)
        total = 0
        with open(fpath, "wb") as out:
            while True:
                chunk = resp.read(1024*1024)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
        # Auto-detección y renombrado de extensión si falta o no es soportada
        try:
            low = fpath.lower()
            supported = (low.endswith('.keras') or low.endswith('.h5') or low.endswith('.zip'))
            if not supported:
                with open(fpath, 'rb') as _fh:
                    head = _fh.read(8)
                new_ext = None
                # HDF5 signature -> .h5
                if head.startswith(b"\x89HDF"):
                    new_ext = '.h5'
                # ZIP signature -> distinguir SavedModel vs .keras
                elif head.startswith(b"PK\x03\x04"):
                    import zipfile
                    with zipfile.ZipFile(fpath, 'r') as z:
                        names = z.namelist()
                        if any(n.endswith('saved_model.pb') for n in names):
                            new_ext = '.zip'
                        else:
                            new_ext = '.keras'
                if new_ext:
                    new_path = fpath + new_ext
                    import os
                    os.rename(fpath, new_path)
                    fpath = new_path
        except Exception:
            pass
        return fpath, total

    # 1) Primer intento: uc?export=download
    base1 = "https://docs.google.com/uc?export=download&id=" + file_id
    last_err = None
    try:
        r1 = _do(base1)
        cookie_token = None
        for c in cj:
            if c.name.startswith("download_warning"):
                cookie_token = c.value
                break
        if cookie_token:
            url2 = base1 + "&confirm=" + cookie_token
            r2 = _do(url2)
            fname = r2.headers.get("Content-Disposition","" ).split("filename=")[-1].strip('"; ') or "model.bin"
            p, n = _stream_to_tmp(r2, fname)
        else:
            fname = r1.headers.get("Content-Disposition","" ).split("filename=")[-1].strip('"; ') or "model.bin"
            p, n = _stream_to_tmp(r1, fname)
        if n > 0:
            return p, {"status":200, "final_url": "gdrive:"+file_id, "length": n}
    except Exception as e:
        last_err = e

    # 2) Segundo intento: drive.usercontent.google.com
    base2 = "https://drive.usercontent.google.com/download?id=" + file_id + "&export=download"
    try:
        r3 = _do(base2)
        fname = r3.headers.get("Content-Disposition","" ).split("filename=")[-1].strip('"; ') or "model.bin"
        p, n = _stream_to_tmp(r3, fname)
        if n > 0:
            return p, {"status":200, "final_url": "gdrive-usercontent:"+file_id, "length": n}
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Descarga desde Drive falló (bytes=0). ID: {file_id}. Detalle: {last_err}")


def _github_api_json(url: str, token: str | None = None) -> dict:
    req = _urlreq.Request(url)
    req.add_header("User-Agent", "streamlit-wbc/1.0")
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with _urlreq.urlopen(req) as resp:
        return _json.loads(resp.read().decode("utf-8"))

def _resolve_release_asset_download_url(user, repo, tag, asset_name, token=None) -> str:
    """
    Devuelve la browser_download_url exacta del asset (nombre exacto) en la release {tag}.
    """
    meta = _github_api_json(
        f"https://api.github.com/repos/{user}/{repo}/releases/tags/{tag}",
        token=token
    )
    for a in meta.get("assets", []):
        if a.get("name") == asset_name:
            return a.get("browser_download_url")
    raise RuntimeError(f"Asset '{asset_name}' no se encontró en la release '{tag}'.")

# ======================================================
# Barra lateral — Modelo y opciones
# ======================================================
st.sidebar.header("Modelo y opciones")

src = st.sidebar.radio(
    "Origen del modelo",
    ["Subir archivo", "URL directa", "GitHub Release (público)", "Google Drive (ID)", "Ruta local (servidor)"],
    index=3
)

mfile = None
model = None
model_path = None
fetch_info = None
err_loading = None

try:
    # ---------- Subir archivo ----------
    if src == "Subir archivo":
        mfile = st.sidebar.file_uploader("Modelo (.keras, .h5 o SavedModel .zip)", type=["keras", "h5", "zip"], key="up_model")
        if mfile is not None and st.sidebar.button("Cargar ahora", use_container_width=True, key="btn_upload"):
            # Guardar a disco y cargar desde ruta (evita picos de RAM)
            tmpdir = tempfile.mkdtemp(prefix="wbc_up_")
            model_path = os.path.join(tmpdir, mfile.name)
            with open(model_path, "wb") as f:
                f.write(mfile.read())
            model = load_model_from_path(model_path)
            fetch_info = {"status": 200, "final_url": "archivo_subido", "length": os.path.getsize(model_path)}

    # ---------- URL directa ----------
    elif src == "URL directa":
        url_direct = st.sidebar.text_input("URL del modelo (.keras/.h5/.zip)", placeholder="https://...", key="url_direct").strip()
        token_direct = st.sidebar.text_input("Token (opcional si es privado)", type="password", key="tok_direct").strip()
        if url_direct and st.sidebar.button("Cargar ahora", use_container_width=True, key="btn_direct"):
            model_path, fetch_info = _fetch_to_path(url_direct, bearer_token=token_direct or None)
            model = load_model_from_path(model_path)



     ###### AÑADIDO ELIF
    elif src == "Google Drive (ID)":
        gdid_raw = st.sidebar.text_input("ID de archivo de Google Drive", placeholder="pegar ID o URL de Drive", key="gdrive_id").strip()
        if gdid_raw and st.sidebar.button("Cargar desde Drive", use_container_width=True, key="btn_gdrive"):
            try:
                gdid = _gdrive_extract_id(gdid_raw)
                model_path, fetch_info = _fetch_gdrive_to_path(gdid)
                model = load_model_from_path(model_path)
            except Exception as e:
                err_loading = f"Descarga desde Drive falló: {e}"

    ####################################
    





    
    # ---------- GitHub Release (público) ----------
    elif src == "GitHub Release (público)":
        gh_user  = st.sidebar.text_input("Usuario/Org", value="Spor195", key="gh_user").strip()
        gh_repo  = st.sidebar.text_input("Repositorio", value="wbc_streamlit2", key="gh_repo").strip()
        gh_tag   = st.sidebar.text_input("Tag de release", value="v1.0.0", key="gh_tag").strip()
        gh_asset = st.sidebar.text_input("Nombre del asset", value="modelo_final.keras", key="gh_asset").strip()
        gh_token = st.sidebar.text_input("Token (opcional: privado o rate limit)", type="password", key="gh_tok").strip()

        built_url = f"https://github.com/{gh_user}/{gh_repo}/releases/download/{gh_tag}/{gh_asset}"
        st.sidebar.caption("URL construida (exacta):")
        st.sidebar.code(repr(built_url), language="text")

    # ---- descarga robusta: usa Accept */* y muestra URL final ----
    def _fetch_to_path_any(url: str, bearer_token: str | None = None) -> tuple[str, dict]:
        import urllib.request as _rq, urllib.error as _er
        req = _rq.Request(url)
        req.add_header("User-Agent", "streamlit-wbc/1.0")
        req.add_header("Accept", "*/*")  # <- más permisivo que octet-stream para algunos mirrors
        if bearer_token:
            req.add_header("Authorization", f"Bearer {bearer_token}")
        tmpdir = tempfile.mkdtemp(prefix="wbc_dl_")
        fpath = os.path.join(tmpdir, os.path.basename(url.split("?")[0]) or "model.bin")
        total = 0
        with _rq.urlopen(req) as resp, open(fpath, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk: break
                out.write(chunk); total += len(chunk)
            info = {"status": getattr(resp, "status", 200), "final_url": resp.geturl(), "length": total}
        return fpath, info

    # ---- flujo con cadena de fallbacks ----
    if st.sidebar.button("Cargar desde Release (API)", use_container_width=True, key="btn_load_api_chain"):
        try:
            # 1) Resolver URL exacta del asset por API
            dl_url = _resolve_release_asset_download_url(gh_user, gh_repo, gh_tag, gh_asset, token=gh_token or None)
            st.sidebar.caption("URL resuelta por API (exacta):")
            st.sidebar.code(repr(dl_url), language="text")

            # 2) Intento principal
            try:
                model_path, fetch_info = _fetch_to_path_any(dl_url, bearer_token=gh_token or None)
                model = load_model_from_path(model_path)
            except Exception as e1:
                # 3) Fallback directo fijo (solo si coincide con tus valores)
                try:
                    fixed_url = "https://github.com/Spor195/wbc_streamlit2/releases/download/v1.0.0/modelo_final.keras"
                    model_path, fetch_info = _fetch_to_path_any(fixed_url, bearer_token=gh_token or None)
                    model = load_model_from_path(model_path)
                except Exception as e2:
                    # 4) Último intento: URL construida por los campos
                    model_path, fetch_info = _fetch_to_path_any(built_url, bearer_token=gh_token or None)
                    model = load_model_from_path(model_path)

        except Exception as e:
            st.sidebar.error(f"No se pudo resolver/descargar el asset: {e}")

    st.sidebar.divider()

    # Botón directo (si quieres forzar manualmente la URL de los campos)
    if st.sidebar.button("Cargar ahora (usar URL de campos)", use_container_width=True, key="btn_load_fields_only"):
        try:
            model_path, fetch_info = _fetch_to_path_any(built_url, bearer_token=gh_token or None)
            model = load_model_from_path(model_path)
        except Exception as e:
            st.sidebar.error(f"Descarga por URL construida falló: {e}")

    # ---------- Ruta local ----------
    elif src == "Ruta local (servidor)":
        local_path = st.sidebar.text_input("Ruta absoluta en el servidor", placeholder="/ruta/a/modelo.keras", key="local_path").strip()
        if local_path and st.sidebar.button("Cargar ahora", use_container_width=True, key="btn_local"):
            if os.path.exists(local_path):
                model = load_model_from_path(local_path)
                model_path = local_path
                fetch_info = {"status": 200, "final_url": local_path, "length": os.path.getsize(local_path)}
            else:
                err_loading = "La ruta indicada no existe en este entorno."

except Exception as e:
    err_loading = str(e)

# Informe de descarga/carga
if fetch_info:
    st.sidebar.info(f"Descarga OK · HTTP {fetch_info.get('status')} · bytes: {fetch_info.get('length'):,}")
    st.sidebar.caption(f"URL final: {fetch_info.get('final_url')}")
if err_loading:
    st.sidebar.error(f"Error al cargar el modelo: {err_loading}")
elif model is not None:
    st.sidebar.success("Modelo cargado correctamente.")

# =========================
# Resto de opciones
# =========================
labels_file = st.sidebar.file_uploader("Etiquetas (.txt o .json)", type=["txt", "json"], key="labels_up")
pp_mode = st.sidebar.selectbox("Preprocesamiento", ["1/255", "EfficientNet", "VGG/ResNet (caffe)", "Sin normalizar"], key="pp_mode")
threshold = st.sidebar.slider("Umbral de confianza para 'detectar'", 0.0, 0.99, 0.0, 0.01, key="thr")
topk = st.sidebar.slider("Top-K a mostrar", 1, 10, 5, 1, key="topk")
show_shapes = st.sidebar.checkbox("Mostrar shapes y depuración", value=True, key="dbg")

# ======================================================
# Área principal — Imagen e inferencia
# ======================================================
img_file = st.file_uploader("Imagen de leucocito (JPG/PNG)", type=["jpg", "jpeg", "png"], key="img_up")

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

            # Normalización a probabilidades si son logits
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
