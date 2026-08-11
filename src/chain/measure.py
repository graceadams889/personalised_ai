"""Measurement helpers used by both the profiler and the chain validator."""
import numpy as np
import pyloudnorm as pyln
from scipy import signal

BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
BAND_LABELS = [f"{b}Hz" for b in BANDS]


def lufs(audio, sr):
    return float(pyln.Meter(sr).integrated_loudness(audio))


def peak_db(audio):
    p = float(np.max(np.abs(audio)))
    return 20 * np.log10(p) if p > 0 else -np.inf


def rms_db(audio):
    r = float(np.sqrt(np.mean(audio ** 2)))
    return 20 * np.log10(r) if r > 0 else -np.inf


def crest_db(audio):
    return peak_db(audio) - rms_db(audio)


def stereo_corr(audio):
    L, R = audio[:, 0], audio[:, 1]
    if np.std(L) == 0 or np.std(R) == 0:
        return 1.0
    return float(np.corrcoef(L, R)[0, 1])


def band_energy_db(audio, sr, bands=BANDS):
    mono = audio.mean(axis=1)
    f, psd = signal.welch(mono, fs=sr, nperseg=8192)
    out = []
    for fc in bands:
        lo, hi = fc / np.sqrt(2), fc * np.sqrt(2)
        mask = (f >= lo) & (f < hi)
        power = psd[mask].sum() if mask.any() else 0.0
        out.append(10 * np.log10(power) if power > 0 else -np.inf)
    return np.array(out)
