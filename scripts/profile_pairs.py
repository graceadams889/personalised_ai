"""
Phase B: profile Grace's mastering style across all paired tracks.

For each (pre, master) pair, this measures four things:
  - Integrated loudness (LUFS) — what loudness she masters TO
  - Crest factor (peak - RMS in dB) — how compressed her masters are
  - L-R stereo correlation — how wide her masters are
  - Per-octave-band EQ delta — the shape of her tonal mastering move
    (computed on RMS-normalised audio so the curve is independent of level)

Saves: outputs/profile_per_track.csv
Prints: per-track values and a per-genre summary.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRE_DIR = ROOT / "data" / "raw" / "pre"
MASTER_DIR = ROOT / "data" / "raw" / "master"
OUT_CSV = ROOT / "outputs" / "profile_per_track.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Octave-band centres for the EQ-shape analysis
BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
BAND_LABELS = [f"{b}Hz" for b in BANDS]


def genre_from_filename(name: str) -> str:
    n = name.lower()
    if "acoustic" in n:
        return "Acoustic"
    if "altrock" in n:
        return "AltRock"
    if "classical" in n:
        return "Classical"
    if "electr" in n or "rbpop" in n:
        return "ElectronicPop"
    return "Unknown"


def load_stereo(path: Path):
    audio, sr = sf.read(str(path), always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.astype(np.float64), sr


def lufs(audio, sr):
    return float(pyln.Meter(sr).integrated_loudness(audio))


def peak_db(audio):
    p = float(np.max(np.abs(audio)))
    return 20 * np.log10(p) if p > 0 else -np.inf


def rms_db(audio):
    r = float(np.sqrt(np.mean(audio ** 2)))
    return 20 * np.log10(r) if r > 0 else -np.inf


def stereo_corr(audio):
    L, R = audio[:, 0], audio[:, 1]
    if np.std(L) == 0 or np.std(R) == 0:
        return 1.0
    return float(np.corrcoef(L, R)[0, 1])


def band_energy_db(audio, sr, bands=BANDS):
    """Per-octave-band energy (dB) via Welch's PSD."""
    mono = audio.mean(axis=1)
    f, psd = signal.welch(mono, fs=sr, nperseg=8192)
    out = []
    for fc in bands:
        lo, hi = fc / np.sqrt(2), fc * np.sqrt(2)
        mask = (f >= lo) & (f < hi)
        power = psd[mask].sum() if mask.any() else 0.0
        out.append(10 * np.log10(power) if power > 0 else -np.inf)
    return np.array(out)


def rms_normalise(audio, rdb):
    if rdb == -np.inf:
        return audio
    return audio / (10 ** (rdb / 20))


# ---- main loop ----
rows = []
files = sorted([p.name for p in PRE_DIR.glob("*.wav")])
print(f"Profiling {len(files)} pairs...\n")

for filename in files:
    pre_path = PRE_DIR / filename
    master_path = MASTER_DIR / filename
    if not master_path.exists():
        print(f"  skip {filename}: no master")
        continue

    genre = genre_from_filename(filename)
    pre, pre_sr = load_stereo(pre_path)
    mas, mas_sr = load_stereo(master_path)
    if pre_sr != mas_sr:
        print(f"  skip {filename}: sample rate mismatch")
        continue

    # align to same length
    n = min(len(pre), len(mas))
    pre, mas = pre[:n], mas[:n]

    pre_lufs, mas_lufs = lufs(pre, pre_sr), lufs(mas, mas_sr)
    pre_peak, mas_peak = peak_db(pre), peak_db(mas)
    pre_rms, mas_rms = rms_db(pre), rms_db(mas)
    pre_crest, mas_crest = pre_peak - pre_rms, mas_peak - mas_rms
    pre_corr, mas_corr = stereo_corr(pre), stereo_corr(mas)

    # EQ shape: difference on RMS-normalised audio, so level changes don't pollute
    pre_n = rms_normalise(pre, pre_rms)
    mas_n = rms_normalise(mas, mas_rms)
    eq_delta = band_energy_db(mas_n, pre_sr) - band_energy_db(pre_n, pre_sr)

    row = {
        "file": filename,
        "genre": genre,
        "pre_lufs": pre_lufs,
        "master_lufs": mas_lufs,
        "lufs_delta": mas_lufs - pre_lufs,
        "master_peak_dbfs": mas_peak,
        "master_crest_db": mas_crest,
        "crest_delta_db": mas_crest - pre_crest,
        "master_lr_corr": mas_corr,
        "corr_delta": mas_corr - pre_corr,
    }
    for label, val in zip(BAND_LABELS, eq_delta):
        row[f"eq_delta_{label}"] = float(val)

    rows.append(row)
    print(f"  [{genre:14s}] {filename[:46]:46s}  "
          f"master={mas_lufs:6.1f} LUFS, crest={mas_crest:4.1f} dB, corr={mas_corr:+.2f}")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

print("\n=== Per-genre averages ===")
cols = ["master_lufs", "master_crest_db", "master_lr_corr", "lufs_delta"] \
    + [f"eq_delta_{b}" for b in BAND_LABELS]
summary = df.groupby("genre")[cols].mean().round(2)
print(summary.T.to_string())
print(f"\nTracks per genre:\n{df['genre'].value_counts().to_string()}")
