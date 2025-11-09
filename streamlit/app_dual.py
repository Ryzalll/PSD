
# app_dual.py
import streamlit as st
import numpy as np
import pandas as pd
import tsfel
import librosa
import soundfile as sf
import io, tempfile, shutil, subprocess, joblib
from pathlib import Path

st.set_page_config(page_title="Deteksi BUKA/TUTUP & AZHARI/RIZAL", page_icon="🎙️", layout="centered")

MODELS_DIR = Path("./models")
MODEL_TARGET = MODELS_DIR / "model_all_features.joblib"
MODEL_NAME   = MODELS_DIR / "model_name_all_features.joblib"

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False

@st.cache_resource
def load_model_pkg(path: Path):
    return joblib.load(path.as_posix())

@st.cache_resource
def load_cfg_all():
    try:
        cfg = tsfel.get_features_by_domain("all")
        if isinstance(cfg, dict) and cfg:
            return cfg
    except Exception:
        pass
    temp = tsfel.get_features_by_domain("temporal")
    spec = tsfel.get_features_by_domain("spectral")
    stat = tsfel.get_features_by_domain("statistical")
    cfg = {}
    for c in (temp, spec, stat):
        for d, feats in c.items():
            if d not in cfg: cfg[d] = feats
            else: cfg[d].update(feats)
    return cfg

def is_ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def decode_with_ffmpeg_to_wav_bytes(input_bytes: bytes, target_sr: int = None) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as fin:
        fin.write(input_bytes)
        in_path = fin.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
        out_path = fout.name
    try:
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-vn", "-f", "wav", "-acodec", "pcm_s16le"]
        if target_sr:
            cmd += ["-ar", str(int(target_sr))]
        cmd += [out_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return Path(out_path).read_bytes()
    finally:
        try: Path(in_path).unlink(missing_ok=True)
        except Exception: pass

def read_audio_bytes_to_mono(bytes_data: bytes):
    try:
        y, sr = sf.read(io.BytesIO(bytes_data), dtype="float32", always_2d=False)
        if y.ndim == 2: y = np.mean(y, axis=1)
        return y, int(sr)
    except Exception:
        pass
    if is_ffmpeg_available():
        try:
            wav_bytes = decode_with_ffmpeg_to_wav_bytes(bytes_data)
            y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
            if y.ndim == 2: y = np.mean(y, axis=1)
            return y, int(sr)
        except Exception:
            pass
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
        tmp.write(bytes_data); tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        return y, int(sr)
    finally:
        try: Path(tmp_path).unlink(missing_ok=True)
        except Exception: pass

def extract_all_features(y, sr, feature_cols, cfg_all):
    feats = tsfel.time_series_features_extractor(cfg_all, y, fs=sr, verbose=0)
    feats = feats.select_dtypes(include=["number"]).copy()
    X = feats.reindex(columns=feature_cols)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]
    return X

st.title("🎙️ Deteksi Perintah & Nama (Dual Model)")
st.caption("Satu input audio → dua prediksi: **BUKA/TUTUP** dan **AZHARI/RIZAL**.")

cfg_all = load_cfg_all()

if not MODEL_TARGET.exists():
    st.error(f"Model target tidak ditemukan: {MODEL_TARGET.as_posix()}")
    st.stop()
pkg_t = load_model_pkg(MODEL_TARGET)
pipe_t = pkg_t['pipeline']
cols_t = pkg_t['feature_columns']
classes_t = [str(c) for c in pkg_t.get('target_classes', [])]

pkg_n = None
if MODEL_NAME.exists():
    pkg_n = load_model_pkg(MODEL_NAME)
    pipe_n = pkg_n['pipeline']
    cols_n = pkg_n['feature_columns']
    classes_n = [str(c) for c in pkg_n.get('target_classes', [])]
else:
    st.info("Model nama tidak ditemukan. App tetap memprediksi perintah.")

st.sidebar.header("Input")
mode = st.sidebar.radio("Metode input", ["Rekam Mikrofon", "Upload File"])
force_mono = st.sidebar.checkbox("Paksa mono", value=True)

audio_bytes = None
if mode == "Rekam Mikrofon":
    if MIC_AVAILABLE:
        st.info("Klik Start → ucapkan 'buka' / 'tutup' → Stop.")
        mic_audio = mic_recorder(key="mic", start_prompt="Start", stop_prompt="Stop", just_once=False)
        if mic_audio and "bytes" in mic_audio and mic_audio["bytes"]:
            audio_bytes = mic_audio["bytes"]
            st.audio(audio_bytes, format="audio/wav")
    else:
        st.warning("streamlit_mic_recorder tidak tersedia. Gunakan Upload File.")
else:
    up = st.file_uploader("Unggah audio (wav/mp3/ogg/m4a)", type=["wav","mp3","ogg","m4a"])
    if up is not None:
        audio_bytes = up.read()
        st.audio(audio_bytes)

if st.button("🔍 Prediksi", type="primary"):
    if not audio_bytes:
        st.warning("Belum ada audio.")
        st.stop()
    try:
        y, sr = read_audio_bytes_to_mono(audio_bytes)
        if y.ndim == 2 and force_mono:
            y = np.mean(y, axis=1)

        X_t = extract_all_features(y, sr, cols_t, cfg_all)
        pred_t_idx = pipe_t.predict(X_t)[0]
        label_t = classes_t[pred_t_idx] if classes_t else str(pred_t_idx)
        label_t_upper = str(label_t).upper().replace("OPEN","BUKA").replace("CLOSE","TUTUP")

        label_n_upper = None
        if pkg_n is not None:
            X_n = extract_all_features(y, sr, cols_n, cfg_all)
            pred_n_idx = pipe_n.predict(X_n)[0]
            label_n = classes_n[pred_n_idx] if classes_n else str(pred_n_idx)
            label_n_upper = str(label_n).upper()

        if label_n_upper:
            st.success(f"**Hasil:** {label_t_upper.capitalize()} dari {label_n_upper.capitalize()}")
        else:
            st.success(f"**Hasil:** {label_t_upper.capitalize()}")

    except Exception as e:
        if not is_ffmpeg_available():
            st.error("Backend decoding tidak tersedia. Pasang FFmpeg (tambahkan ke PATH) atau unggah WAV.")
        st.error(f"Gagal memproses audio: {e}")
