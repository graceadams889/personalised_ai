import numpy as np
import torch
from typing import Dict


def signal_to_noise_ratio(predicted: np.ndarray, target: np.ndarray) -> float:
    noise = target - predicted
    signal_power = np.mean(target ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return np.inf
    return float(10 * np.log10(signal_power / noise_power))


def spectral_convergence(predicted: np.ndarray, target: np.ndarray, fft_size: int = 2048) -> float:
    if predicted.ndim > 1:
        predicted = predicted.mean(axis=1)
    if target.ndim > 1:
        target = target.mean(axis=1)

    pred_stft = np.abs(np.fft.rfft(predicted, n=fft_size))
    target_stft = np.abs(np.fft.rfft(target, n=fft_size))

    diff_norm = np.linalg.norm(target_stft - pred_stft)
    target_norm = np.linalg.norm(target_stft)

    if target_norm < 1e-10:
        return 0.0
    return float(diff_norm / target_norm)


def lufs_difference(predicted: np.ndarray, target: np.ndarray, sr: int = 44100) -> float:
    from ..data_prep.normalise import measure_lufs
    pred_lufs = measure_lufs(predicted, sr)
    target_lufs = measure_lufs(target, sr)
    if np.isinf(pred_lufs) or np.isinf(target_lufs):
        return np.inf
    return float(abs(pred_lufs - target_lufs))


def per_band_energy(audio: np.ndarray, sr: int = 44100) -> Dict[str, float]:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    n_fft = 4096
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    bands = {
        "sub_bass": (20, 60),
        "bass": (60, 250),
        "low_mid": (250, 500),
        "mid": (500, 2000),
        "high_mid": (2000, 6000),
        "presence": (6000, 12000),
        "brilliance": (12000, 20000),
    }

    energies = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_energy = np.mean(spectrum[mask] ** 2) if mask.any() else 0.0
        energies[name] = float(10 * np.log10(band_energy + 1e-20))

    return energies


def per_band_energy_difference(predicted: np.ndarray, target: np.ndarray, sr: int = 44100) -> Dict[str, float]:
    pred_bands = per_band_energy(predicted, sr)
    target_bands = per_band_energy(target, sr)
    return {band: abs(pred_bands[band] - target_bands[band]) for band in pred_bands}


def crest_factor(audio: np.ndarray) -> float:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    if rms < 1e-10:
        return 0.0
    return float(20 * np.log10(peak / rms))


def crest_factor_difference(predicted: np.ndarray, target: np.ndarray) -> float:
    return abs(crest_factor(predicted) - crest_factor(target))


def compute_all_metrics(predicted: np.ndarray, target: np.ndarray, sr: int = 44100) -> Dict[str, float]:
    results = {
        "snr_db": signal_to_noise_ratio(predicted, target),
        "spectral_convergence": spectral_convergence(predicted, target),
        "lufs_difference": lufs_difference(predicted, target, sr),
        "crest_factor_diff_db": crest_factor_difference(predicted, target),
    }

    band_diffs = per_band_energy_difference(predicted, target, sr)
    for band, diff in band_diffs.items():
        results[f"band_diff_{band}_db"] = diff

    return results
