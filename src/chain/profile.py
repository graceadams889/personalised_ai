"""Load the per-genre mastering profile averaged from profile_per_track.csv."""
from pathlib import Path
import pandas as pd

from .measure import BANDS

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "outputs" / "profile_per_track.csv"


def get_profile(genre: str, csv_path: Path | None = None) -> dict:
    csv_path = csv_path or DEFAULT_CSV
    df = pd.read_csv(csv_path)
    grp = df[df["genre"] == genre]
    if len(grp) == 0:
        raise ValueError(
            f"No tracks for genre '{genre}'. Available: {sorted(df['genre'].unique())}"
        )
    avg = grp.mean(numeric_only=True)
    return {
        "genre": genre,
        "n_tracks": int(len(grp)),
        "master_lufs": float(avg["master_lufs"]),
        "master_crest_db": float(avg["master_crest_db"]),
        "master_lr_corr": float(avg["master_lr_corr"]),
        "eq_gains_db": [float(avg[f"eq_delta_{b}Hz"]) for b in BANDS],
    }


def available_genres(csv_path: Path | None = None) -> list[str]:
    csv_path = csv_path or DEFAULT_CSV
    df = pd.read_csv(csv_path)
    return sorted(df["genre"].unique())
