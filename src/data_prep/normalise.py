import numpy as np
from typing import Tuple


def measure_lufs(audio: np.ndarray, sr: int = 44100) -> float:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    block_size = int(0.4 * sr)
    hop = int(0.1 * sr)

    if len(audio) < block_size:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return -np.inf
        return 20 * np.log10(rms) - 0.691

    blocks = []
    for start in range(0, len(audio) - block_size + 1, hop):
        block = audio[start : start + block_size]
        block_power = np.mean(block ** 2, axis=0).sum()
        blocks.append(block_power)

    blocks = np.array(blocks)
    if len(blocks) == 0 or np.max(blocks) < 1e-20:
        return -np.inf

    abs_threshold = -70.0
    abs_gate = 10 ** ((abs_threshold + 0.691) / 10)
    above_abs = blocks[blocks > abs_gate]

    if len(above_abs) == 0:
        return -np.inf

    rel_threshold_power = np.mean(above_abs) * 10 ** (-10 / 10)
    above_rel = above_abs[above_abs > rel_threshold_power]

    if len(above_rel) == 0:
        return -np.inf

    loudness = -0.691 + 10 * np.log10(np.mean(above_rel))
    return float(loudness)


def loudness_normalise(
    audio: np.ndarray, sr: int = 44100, target_lufs: float = -23.0
) -> Tuple[np.ndarray, float, float]:
    current_lufs = measure_lufs(audio, sr)

    if np.isinf(current_lufs):
        print("  Warning: audio is silent, skipping normalisation")
        return audio, current_lufs, 0.0

    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)

    normalised = audio * gain_linear

    peak = np.max(np.abs(normalised))
    if peak > 1.0:
        normalised = normalised / peak
        print(f"  Warning: clipping prevented, peak was {peak:.3f}")

    return normalised, current_lufs, gain_db
