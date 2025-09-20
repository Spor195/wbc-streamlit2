# app_wbc_streamlit_drive_autofix.py
# ------------------------------------------------------
# Clasificador WBC – Carga de modelo en prueba (Drive/URL)
# - Uso en docencia o investigación
# - Acepta ID o URL de Google Drive
# - Maneja confirm-token y endpoint alterno (drive.usercontent)
# - Auto-detecta y renombra .keras / .h5 / SavedModel .zip
# ------------------------------------------------------

import os, io, json, zipfile, tempfile, re
import numpy as np
from PIL import Image
import streamlit as st

# =========================
# TensorFlow (carga perezosa)
# =========================
try:
    import tensorflow as tf
except Exception:
    tf = None

# =========================
# Utilidades genéricas
# =========================
import urllib.request as _urlreq, urllib.error as _urlerr, urllib.parse as _urlparse
import http.cookiejar as _cookielib

def _ensure_tf():
    if tf is None:
        raise RuntimeError("TensorFlow no está disponible. Instálalo e inténtalo nuevamente.")

@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str):
    """
    Carga un modelo desde ruta:
    - .keras / .h5 -> load_model directo
    - .zip (SavedModel) -> descomprime y busca saved_model.pb
    """
    _ensure_tf()
    low = path.lower()
    if low.endswith((".keras", ".h5")):
        return tf.keras.models.load_model(path)

    if low.endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="wbc_sm_")
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmpdir)
        # SavedModel
        sm = os.path.join(tmpdir, "saved_model.pb")
        if os.path.exists(sm):
            return tf.keras.models.load_model(tmpdir)
        # Si no hay saved_model.pb, asumir Keras empaquetado
        raise RuntimeError("El .zip no contiene SavedModel (saved_model.pb).")

    raise RuntimeError("Extensión no soportada. Usa .keras, .h5 o SavedModel .zip")

def _guess_and_fix_extension(fpath: str) -> str:
    """
    Si el archivo llegó sin extensión o con extensión genérica,
    detecta el tipo real y renombra a .keras / .h5 / .zip.
    """
    def _is_hdf5(fp):
        try:
            with open(fp, "rb") as f:
                head = f.read(8)
            # Firma HDF5: \x89HDF\r\n\x1a\n
            return head.startswith(b"\x89HDF")
        except Exception:
            return False

    def _is_zip(fp):
        try:
            with open(fp, "rb") as f:
                head = f.read(4)
            return head == b"PK\x03\x04"
        except Exception:
            return False

    new_ext = None
    if _is_hdf5(fpath):
        new_ext = ".h5"
    elif _is_zip(fpath):
        # Verificar si es SavedModel.zip
        try:
            with zipfile.ZipFile(fpath, "r") as z:
                names = z.namelist()
            if any(n.endswith("saved_model.pb") for n in names):
                new_ext = ".zip"
            else:
                # Podría ser un .keras (también es zip); preferimos .keras
                new_ext = ".keras"
        except Exception:
            new_ext = ".zip"

    if new_ext:
        base, ext = os.path.splitext(fpath)
        if ext.lower() not in (".keras", ".h5", ".zip"):
            new_path = base + new_ext
            os.rename(fpath, new_path)
            return new_path
    return fpath

# ======================================================
# Google Drive – extracción de ID y descarga robusta
# ======================================================
def _gdrive_extract_id(x: str) -> str:
    """Acepta ID o URL y devuelve el file_id (solo [A-Za-z0-9_-])."""
    s = (x or "").strip()
    s = s.strip().strip("{}[]() \n\r\t")
    m = re.search(r"/file/d/([A-Za-z0-9_-]{20,})", s)
    if not m:
        m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    if m:
        s = m.group(1)
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        raise ValueError("El ID/URL de Drive no es válido. Pegue el ID exacto o una URL de Drive.")
    return s

def _stream_to_tmp(resp, fname_hint="model.bin"):
    tmpdir = tempfile.mkdtemp(prefix="wbc_dl_")
    fname = fname_hint or "model.bin"
    fpath = os.path.join(tmpdir, fname)
    total = 0
    with open(fpath, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    return fpath, total

def _fetch_gdrive_to_path(file_id: str) -> tuple[str, dict]:
    """
    Descarga un archivo grande de Google Drive; maneja confirm-token y reintentos.
    Devuelve (ruta_local, info).
    """
    cj = _cookielib.CookieJar()
    opener = _urlreq.build_opener(_urlreq.HTTPCookieProcessor(cj))

    def _do(url):
        req = _urlreq.Request(url, headers={"User-Agent": "wbc-streamlit/1.1", "Accept": "*/*"})
        return opener.open(req)

    # 1) Primer intento: uc?export=download con token
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
            fname = r2.headers.get("Content-Disposition", "").split("filename=")[-1].strip('"; ') or "model.bin"
            p, n = _stream_to_tmp(r2, fname)
        else:
            fname = r1.headers.get("Content-Disposition", "").split("filename=")[-1].strip('"; ') or "model.bin"
            p, n = _stream_to_tmp(r1, fname)
        if n > 1024 * 100:  # >100 KB como sanity check
            return _guess_and_fix_extension(p), {"status": 200, "final_url": "gdrive:"+file_id, "length": n}
    except Exception as e:
        last_err = e

    # 2) Segundo intento: drive.usercontent.google.com
    base2 = "https://drive.usercontent.google.com/download?id=" + file_id + "&export=download"
    try:
        r3 = _do(base2)
        fname = r3.headers.get("Content-Disposition", "").split("filename=")[-1].strip('"; ') or "model.bin"
        p, n = _stream_to_tmp(r3, fname)
        if n > 1024 * 100:
            return _guess_and_fix_extension(p), {"status": 200, "final_url": "gdrive-usercontent:"+file_id, "length": n}
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Descarga desde Drive falló (bytes<=100KB). ID: {file_id}. Detalle: {last_err}")

# ======================================================
# Descarga genérica por URL (con soporte Drive)
# ======================================================
def _fetch_to_path_any(url: str, bearer_token: str | None = None) -> tuple[str, dict]:
    # Si es una URL de Drive, derivar al flujo Drive robusto
    low = url.lower()
    # usercontent (directo) -> úsalo tal cual (es el endpoint alterno)
    if "drive.google.com" in low:
        # extraer ID y reutilizar flujo Drive
        fid = _gdrive_extract_id(url)
        return _fetch_gdrive_to_path(fid)

    req = _urlreq.Request(url)
    req.add_header("User-Agent", "streamlit-wbc/1.1")
    req.add_header("Accept", "application/octet-stream")
    if bearer_token:
        req.add_header("Authorization", f"Bearer {bearer_token}")

    tmpdir = tempfile.mkdtemp(prefix="wbc_dl_")
    fname = os.path.basename(url.split("?")[0]) or "model.bin"
    fpath = os.path.join(tmpdir, fname)

    total = 0
    with _urlreq.urlopen(req) as resp, open(fpath, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)

    # sanity check
    if total <= 1024 * 100:
        # Intento de rescate: si era drive.usercontent y algo falló, reintentar
        if "drive.usercontent.google.com" in low:
            fid = _gdrive_extract_id(url)
            return _fetch_gdrive_to_path(fid)
        raise RuntimeError(f"Descarga muy pequeña ({total} bytes). ¿La URL es binaria directa?")

    fpath = _guess_and_fix_extension(fpath)
    return fpath, {"status": 200, "final_url": url, "length": total}

# ======================================================
# UI — Sidebar
# ======================================================
st.sidebar.header("Modelo y opciones")

src = st.sidebar.radio(
    "Origen del modelo",
    ["Subir archivo", "URL directa", "Google Drive (ID)", "Ruta local (servidor)"],
    index=2
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
        if mfile is not None:
            tmpdir = tempfile.mkdtemp(prefix="wbc_up_")
            model_path = os.path.join(tmpdir, mfile.name)
            with open(model_path, "wb") as out:
                out.write(mfile.read())
            model_path = _guess_and_fix_extension(model_path)
            model = load_model_from_path(model_path)
            fetch_info = {"status": 200, "final_url": "archivo_subido", "length": os.path.getsize(model_path)}

    # ---------- URL directa ----------
    elif src == "URL directa":
        url_direct = st.sidebar.text_input("URL del modelo (.keras/.h5/.zip)", placeholder="https://...", key="url_direct").strip()
        token_direct = st.sidebar.text_input("Token (opcional si es privado)", type="password", key="tok_direct").strip()
        if url_direct and st.sidebar.button("Cargar ahora", use_container_width=True, key="btn_direct"):
            model_path, fetch_info = _fetch_to_path_any(url_direct, bearer_token=token_direct or None)
            model = load_model_from_path(model_path)

    # ---------- Google Drive (ID) ----------
    elif src == "Google Drive (ID)":
        gdid_raw = st.sidebar.text_input("ID de archivo de Google Drive", placeholder="pegar ID o URL de Drive", key="gdrive_id").strip()
        if gdid_raw and st.sidebar.button("Cargar desde Drive", use_container_width=True, key="btn_gdrive"):
            try:
                gdid = _gdrive_extract_id(gdid_raw)
                model_path, fetch_info = _fetch_gdrive_to_path(gdid)
                model = load_model_from_path(model_path)
            except Exception as e:
                err_loading = f"Descarga desde Drive falló: {e}"

    # ---------- Ruta local (servidor) ----------
    elif src == "Ruta local (servidor)":
        local_path = st.sidebar.text_input("Ruta absoluta en el servidor", placeholder="/path/to/model.keras|.h5|.zip", key="local_path").strip()
        if local_path and st.sidebar.button("Cargar desde ruta local", use_container_width=True, key="btn_local"):
            if not os.path.exists(local_path):
                raise RuntimeError(f"No existe la ruta: {local_path}")
            model_path = _guess_and_fix_extension(local_path)
            model = load_model_from_path(model_path)
            fetch_info = {"status": 200, "final_url": f"file://{local_path}", "length": os.path.getsize(local_path)}

except Exception as e:
    err_loading = str(e)

# ======================================================
# Cuerpo — Reporte de carga y predicción de prueba
# ======================================================
st.title("Clasificador de leucocitos — Carga de modelo robusta")

if fetch_info:
    with st.sidebar:
        st.info(f"Descarga OK · HTTP 200 · bytes: {fetch_info.get('length', '—')}")
        st.caption("URL final:")
        st.code(fetch_info.get("final_url", "—"), language="text")

if err_loading:
    st.error(f"Error al cargar el modelo: {err_loading}")

if model is not None:
    st.success("Modelo cargado correctamente.")
    try:
        # Mostrar input_shape si es posible
        sig = getattr(model, "inputs", None)
        if sig:
            st.write("**Input shape:**", model.inputs[0].shape)
    except Exception:
        pass

    # Demo mínima de inferencia
    img = st.file_uploader("Sube una imagen para probar (opcional)", type=["png", "jpg", "jpeg"])
    if img is not None:
        im = Image.open(img).convert("RGB").resize((224, 224))
        x = np.array(im, dtype=np.float32)[None] / 255.0
        try:
            preds = model.predict(x, verbose=0)
            if preds.ndim == 2 and preds.shape[0] == 1:
                preds = preds[0]
            if preds.ndim > 1:
                st.warning(f"Salida con forma inusual: {preds.shape}. Se tomará argmax del último eje.")
                preds = preds.reshape(-1)
            # Normalización a probabilidades si parecen logits
            if np.any(preds < 0) or np.sum(preds) <= 0.99 or np.sum(preds) >= 1.01:
                try:
                    from scipy.special import softmax
                    probs = softmax(preds)
                except Exception:
                    exps = np.exp(preds - np.max(preds))
                    probs = exps / np.sum(exps)
            else:
                probs = preds
            topk = int(st.slider("Top-K a mostrar", 1, min(10, len(probs)), value=min(3, len(probs))))
            idxs = np.argsort(probs)[::-1][:topk]
            st.write({int(i): float(probs[i]) for i in idxs})
        except Exception as e:
            st.error(f"Inferencia falló: {e}")
else:
    st.info("Cargue el modelo desde la barra lateral.")
