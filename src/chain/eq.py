"""Nine-band peaking EQ — one biquad per octave band, RBJ Audio Cookbook formulas."""
import numpy as np
from scipy.signal import sosfilt

from .measure import BANDS


def peaking_biquad(f0: float, Q: float, gain_db: float, sr: int) -> np.ndarray:
    """One peaking-EQ biquad as an SOS row [b0, b1, b2, 1, a1, a2]."""
    A = 10 ** (gain_db / 40)
    omega = 2 * np.pi * f0 / sr
    alpha = np.sin(omega) / (2 * Q)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(omega)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(omega)
    a2 = 1 - alpha / A
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def apply_eq(
    audio: np.ndarray,
    sr: int,
    band_gains_db: list[float],
    bands: list[int] = BANDS,
    Q: float = 1.414,
) -> np.ndarray:
    """Apply cascaded peaking EQ to stereo audio (samples, channels)."""
    sos = np.stack(
        [peaking_biquad(f0, Q, g, sr) for f0, g in zip(bands, band_gains_db)]
    )
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = sosfilt(sos, audio[:, ch])
    return out
