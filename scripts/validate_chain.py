"""Apply the chain to every pre file with its known-genre profile, then compare
the result's LUFS / crest / correlation against Grace's actual master.

Pass --no-width to bypass the stereo widening stage; outputs go to a separate
folder so v1 (with width) and v2 (without) can be A/B-compared in Reaper.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import soundfile as sf

from src.chain import apply_chain
from src.chain.measure import lufs, crest_db, stereo_corr

parser = argparse.ArgumentParser()
parser.add_argument("--with-width", action="store_true",
                    help="Re-enable the M/S width stage (off by default after v2 finding)")
parser.add_argument("--no-width", action="store_true",
                    help=argparse.SUPPRESS)  # deprecated, default behaviour now
args = parser.parse_args()
USE_WIDTH = args.with_width

ROOT = Path(__file__).resolve().parent.parent
PRE_DIR = ROOT / "data" / "raw" / "pre"
MASTER_DIR = ROOT / "data" / "raw" / "master"
suffix = "_with_width" if USE_WIDTH else "_nowidth"
OUT_DIR = ROOT / "outputs" / f"chain_validation{suffix}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_CSV = ROOT / "outputs" / f"chain_validation{suffix}.csv"


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


rows = []
files = sorted([p.name for p in PRE_DIR.glob("*.wav")])
print(f"Validating chain on {len(files)} tracks...\n")

for filename in files:
    pre_path = PRE_DIR / filename
    mas_path = MASTER_DIR / filename
    if not mas_path.exists():
        continue
    genre = genre_from_filename(filename)
    pre_audio, sr = load_stereo(pre_path)
    mas_audio, _ = load_stereo(mas_path)
    n = min(len(pre_audio), len(mas_audio))
    pre_audio, mas_audio = pre_audio[:n], mas_audio[:n]

    print(f"  [{genre:14s}] {filename}")
    chained = apply_chain(pre_audio, sr, genre, use_width=USE_WIDTH, verbose=True)

    out_wav = OUT_DIR / f"{filename.replace('.wav', '')}_chain.wav"
    sf.write(out_wav, np.clip(chained, -1.0, 1.0), sr, subtype="PCM_24")

    p_meas = measure_all(pre_audio, sr)
    t = measure_all(mas_audio, sr)
    a = measure_all(chained, sr)
    rows.append({
        "file": filename, "genre": genre,
        "target_lufs": t["lufs"], "achieved_lufs": a["lufs"],
        "lufs_error": a["lufs"] - t["lufs"],
        "target_crest_db": t["crest_db"], "achieved_crest_db": a["crest_db"],
        "crest_error_db": a["crest_db"] - t["crest_db"],
        "pre_lr_corr": p_meas["lr_corr"],
        "target_lr_corr": t["lr_corr"], "achieved_lr_corr": a["lr_corr"],
        "corr_error": a["lr_corr"] - t["lr_corr"],
    })

df = pd.DataFrame(rows)
df.to_csv(COMPARE_CSV, index=False)
print(f"\nSaved comparison CSV: {COMPARE_CSV}")
print(f"Chained WAVs in: {OUT_DIR}\n")

print("=== Per-track error (achieved − target) ===")
cols = ["file", "genre", "lufs_error", "crest_error_db", "corr_error"]
print(df[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

print("\n=== Per-genre mean absolute error ===")
errs = ["lufs_error", "crest_error_db", "corr_error"]
mae = df.groupby("genre")[errs].apply(lambda x: x.abs().mean()).round(2)
print(mae.to_string())
