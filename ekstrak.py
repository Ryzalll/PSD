# Ekstraksi Fitur Suara
import os
import glob
import traceback
import librosa
import tsfel
import pandas as pd

# =======================
# KONFIGURASI OUTPUT
# =======================
BASE_AUDIO_DIR = "./audio"  # struktur: ./audio/<nama>/{buka|tutup}/*.ext
OUTPUT_DIR = "./dataset/new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_STAT = os.path.join(OUTPUT_DIR, "audio_features_statistical.csv")
OUT_ALL  = os.path.join(OUTPUT_DIR, "audio_features_all.csv")

# Ekstensi audio yang dipindai
AUDIO_EXTS = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a")


def list_audio_files(base_dir: str):
    """
    Mengumpulkan path audio: audio/<nama>/{buka|tutup}/*.(ext)
    Mengembalikan list path absolute.
    """
    paths = []
    # Telusuri nama (rizal, azhari, dst) lalu kelas (buka/tutup)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Folder tidak ditemukan: {base_dir}")
    for nama in sorted(next(os.walk(base_dir))[1]):  # subfolder di bawah ./audio
        for target in ("buka", "tutup"):
            target_dir = os.path.join(base_dir, nama, target)
            if not os.path.isdir(target_dir):
                continue
            for ext in AUDIO_EXTS:
                paths.extend(glob.glob(os.path.join(target_dir, ext)))
    return sorted(paths)


def parse_label_from_path(path: str):
    """
    Mengambil (nama, target) dari path: audio/<nama>/<target>/file.xxx
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    # .../<audio>/<nama>/<target>/<file>
    # pastikan minimal panjangnya cukup
    nama = None
    target = None
    try:
        # cari index 'audio' paling kanan
        if "audio" in parts:
            i = len(parts) - 1 - parts[::-1].index("audio")
            nama   = parts[i+1] if i + 1 < len(parts) else None
            target = parts[i+2] if i + 2 < len(parts) else None
        else:
            # fallback: ambil dua folder di atas file
            nama   = parts[-3] if len(parts) >= 3 else None
            target = parts[-2] if len(parts) >= 2 else None
    except Exception:
        pass
    return nama, target


def load_cfg_statistical():
    return tsfel.get_features_by_domain("statistical")


def load_cfg_all():
    return tsfel.get_features_by_domain()
    """
    Coba dapatkan konfigurasi 'all'.
    Jika gagal (beberapa versi TSFEL), gabungkan manual tiga domain.
    """
    try:
        cfg_all = tsfel.get_features_by_domain("all")
        if isinstance(cfg_all, dict) and cfg_all:
            return cfg_all
    except Exception:
        pass

    # Fallback: gabungkan tiga domain
    cfg_temp  = tsfel.get_features_by_domain("temporal")
    cfg_spec  = tsfel.get_features_by_domain("spectral")
    cfg_stat  = tsfel.get_features_by_domain("statistical")

    # Struktur TSFEL config adalah dict bertingkat per domain,
    # gabungkan dengan hati-hati (tanpa menimpa isinya).
    cfg_all = {}
    for cfg in (cfg_temp, cfg_spec, cfg_stat):
        for dom, feats in cfg.items():
            if dom not in cfg_all:
                cfg_all[dom] = feats
            else:
                # gabungkan fitur di domain yang sama
                cfg_all[dom].update(feats)
    return cfg_all


def extract_for_one_file(path: str, cfg) -> pd.DataFrame:
    """
    Ekstraksi fitur TSFEL untuk 1 file audio -> DataFrame 1 baris (+meta).
    """
    # 1) Load audio mono, sr asli
    signal, sr = librosa.load(path, sr=None, mono=True)

    # 2) Ekstraksi fitur
    feats_df = tsfel.time_series_features_extractor(
        cfg,
        signal,
        fs=sr,
        verbose=0
    )

    # 3) Tambah metadata dasar
    feats_df.insert(0, "filename", os.path.basename(path))
    feats_df.insert(1, "sr", sr)
    feats_df.insert(2, "duration_sec", (len(signal) / float(sr)) if sr else None)

    # 4) Tambah label dari path: nama, target
    nama, target = parse_label_from_path(path)
    feats_df.insert(3, "nama", nama)
    feats_df.insert(4, "target", target)

    return feats_df


def run_pipeline(paths, cfg, out_csv: str, label: str):
    """
    Jalankan ekstraksi untuk sekumpulan path dengan config tertentu.
    """
    rows = []
    for idx, p in enumerate(paths, 1):
        try:
            df_row = extract_for_one_file(p, cfg)
            rows.append(df_row)
            print(f"[{label}] {idx:05d}/{len(paths)} OK  -> {os.path.basename(p)}  shape={df_row.shape}")
        except Exception as e:
            print(f"[{label}] Gagal: {p}\n  {e}")
            # Jika butuh debug rinci:
            # traceback.print_exc()

    if rows:
        out_df = pd.concat(rows, ignore_index=True)
        out_df.to_csv(out_csv, index=False)
        print(f"\n[{label}] Sukses simpan: {out_csv} | Total file={len(rows)} | Shape={out_df.shape}")
    else:
        print(f"[{label}] Tidak ada baris berhasil diekstraksi. CSV tidak dibuat.")


def main():
    # Kumpulkan semua file
    audio_paths = list_audio_files(BASE_AUDIO_DIR)
    if not audio_paths:
        raise FileNotFoundError(
            f"Tidak ada file audio ditemukan di: {BASE_AUDIO_DIR}\n"
            f"Pastikan struktur: ./audio/<nama>/(buka|tutup)/*.mp3|wav|flac|ogg|m4a"
        )

    print(f"Total file ditemukan: {len(audio_paths)}")

    # Siapkan config
    cfg_stat = load_cfg_statistical()
    cfg_all  = load_cfg_all()

    # Jalankan 2 ekstraksi (statistical & all)
    run_pipeline(audio_paths, cfg_stat, OUT_STAT, label="STAT")
    run_pipeline(audio_paths, cfg_all,  OUT_ALL,  label="ALL")

if __name__ == "__main__":
    main()
