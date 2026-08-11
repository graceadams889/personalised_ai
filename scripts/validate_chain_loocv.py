"""Leave-one-out cross-validation (LOOCV) of the parametric mastering chain.

For each of the 13 paired tracks:
  1. Rebuild the per-genre profile from the OTHER tracks in its genre (i.e.
     exclude the held-out track from the averaging).
  2. Apply the chain with that held-out profile to the track's pre-master file.
  3. Measure LUFS / crest / L-R correlation of the output and compare against
     the actual master.

The chain therefore never "sees" the track it is being evaluated on — this
tests generalisation rather than in-sample fit (Stone, 1974; James, Witten,
Hastie, & Tibshirani, 2021).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import soundfile as sf

from src.chain import apply_chain
from src.chain.measure import lufs, crest_db, stereo_corr, BANDS

ROOT = Path(__file__).resolve().parent.parent
PRE_DIR = ROOT / "data" / "raw" / "pre"
MASTER_DIR = ROOT / "data" / "raw" / "master"
PROFILE_CSV = ROOT / "outputs" / "profile_per_track.csv"
OUT_DIR = ROOT / "outputs" / "loocv_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_CSV = ROOT / "outputs" / "loocv_results.csv"


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


def measure_all(a, sr):
    return {"lufs": lufs(a, sr), "crest_db": crest_db(a), "lr_corr": stereo_corr(a)}


def held_out_profile(df: pd.DataFrame, genre: str, held_out_file: str) -> dict:
    """Average every track in `genre` EXCEPT `held_out_file` to form a
    leave-one-out per-genre profile."""
    grp = df[(df["genre"] == genre) & (df["file"] != held_out_file)]
    if len(grp) == 0:
        raise ValueError(f"No other tracks in {genre} to fit from")
    avg = grp.mean(numeric_only=True)
    return {
        "genre": genre,
        "n_tracks": int(len(grp)),
        "master_lufs": float(avg["master_lufs"]),
        "master_crest_db": float(avg["master_crest_db"]),
        "master_lr_corr": float(avg["master_lr_corr"]),
        "eq_gains_db": [float(avg[f"eq_delta_{b}Hz"]) for b in BANDS],
    }


df = pd.read_csv(PROFILE_CSV)
files = sorted([p.name for p in PRE_DIR.glob("*.wav")])
print(f"Running LOOCV on {len(files)} tracks (each held out once)...\n")

rows = []
for filename in files:
    pre_path = PRE_DIR / filename
    mas_path = MASTER_DIR / filename
    if not mas_path.exists():
        continue

    genre = genre_from_filename(filename)
    try:
        profile = held_out_profile(df, genre, filename)
    except ValueError as e:
        print(f"  [{genre:14s}] {filename}  -- SKIP ({e})")
        continue

    pre_audio, sr = load_stereo(pre_path)
    mas_audio, _ = load_stereo(mas_path)
    n = min(len(pre_audio), len(mas_audio))
    pre_audio, mas_audio = pre_audio[:n], mas_audio[:n]

    print(f"  [{genre:14s}] {filename}  "
          f"(fitted from {profile['n_tracks']} other track"
          f"{'s' if profile['n_tracks'] != 1 else ''})")

    chained = apply_chain(pre_audio, sr, genre, profile=profile, verbose=False)

    out_wav = OUT_DIR / f"{filename.replace('.wav', '')}_loocv.wav"
    sf.write(out_wav, np.clip(chained, -1.0, 1.0), sr, subtype="PCM_24")

    t = measure_all(mas_audio, sr)
    a = measure_all(chained, sr)
    rows.append({
        "file": filename,
        "genre": genre,
        "n_fitted_from": profile["n_tracks"],
        "target_lufs": t["lufs"],
        "achieved_lufs": a["lufs"],
        "lufs_error": a["lufs"] - t["lufs"],
        "target_crest_db": t["crest_db"],
        "achieved_crest_db": a["crest_db"],
        "crest_error_db": a["crest_db"] - t["crest_db"],
        "target_lr_corr": t["lr_corr"],
        "achieved_lr_corr": a["lr_corr"],
        "corr_error": a["lr_corr"] - t["lr_corr"],
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(RESULT_CSV, index=False)
print(f"\nSaved LOOCV results CSV: {RESULT_CSV}")
print(f"Held-out mastered WAVs in: {OUT_DIR}\n")

print("=== Per-track held-out error (achieved − target) ===")
show_cols = ["file", "genre", "n_fitted_from",
             "lufs_error", "crest_error_db", "corr_error"]
print(
    results_df[show_cols].to_string(
        index=False,
        formatters={
            "lufs_error": lambda x: f"{x:+.2f}",
            "crest_error_db": lambda x: f"{x:+.2f}",
            "corr_error": lambda x: f"{x:+.3f}",
        },
    )
)

print("\n=== Per-genre held-out MAE (mean absolute error) ===")
errs = ["lufs_error", "crest_error_db", "corr_error"]
mae = results_df.groupby("genre")[errs].apply(lambda x: x.abs().mean()).round(3)
counts = results_df.groupby("genre").size().rename("n")
print(pd.concat([counts, mae], axis=1).to_string())
