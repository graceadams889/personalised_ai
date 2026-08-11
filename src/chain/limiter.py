"""Brick-wall peak limiter + gain-to-LUFS targeting."""
import numpy as np
import numba
import pyloudnorm as pyln

from .measure import lufs as measure_lufs


@numba.njit(cache=True)
def _gain_smooth(needed: np.ndarray, alpha_r: float) -> np.ndarray:
    """One-pole release smoothing on the limiter gain (instant attack)."""
    g = np.empty_like(needed)
    prev = needed[0]
    g[0] = prev
    for i in range(1, len(needed)):
        if needed[i] < prev:
            prev = needed[i]  # instant attack
        else:
            prev = alpha_r * prev + (1 - alpha_r) * needed[i]
        g[i] = prev
    return g


def peak_limit(
    audio: np.ndarray,
    sr: int,
    ceiling_dbfs: float = -1.0,
    release_ms: float = 50.0,
) -> np.ndarray:
    """Brick-wall peak limiter with smoothed gain release."""
    ceiling = 10 ** (ceiling_dbfs / 20)
    peak = np.maximum(np.max(np.abs(audio), axis=1).astype(np.float64), 1e-10)
    needed = np.minimum(1.0, ceiling / peak)
    alpha_r = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    g = _gain_smooth(needed, alpha_r)
    return audio * g[:, np.newaxis]


def fit_loudness(
    audio: np.ndarray,
    sr: int,
    target_lufs: float,
    ceiling_dbfs: float = -1.0,
    tolerance_db: float = 0.3,
    max_iter: int = 5,
) -> np.ndarray:
    """Iteratively gain + limit until integrated LUFS hits the target."""
    meter = pyln.Meter(sr)
    for _ in range(max_iter):
        current = meter.integrated_loudness(audio)
        if not np.isfinite(current):
            return audio
        if abs(current - target_lufs) < tolerance_db:
            break
        audio = audio * 10 ** ((target_lufs - current) / 20)
        audio = peak_limit(audio, sr, ceiling_dbfs)
    return audio
