"""Stereo-linked broadband compressor with crest-factor fitting."""
import numpy as np
import numba

from .measure import crest_db


@numba.njit(cache=True)
def _env_follow(x_abs: np.ndarray, alpha_a: float, alpha_r: float) -> np.ndarray:
    """Peak envelope follower with separate attack/release time constants."""
    env = np.empty_like(x_abs)
    prev = x_abs[0]
    env[0] = prev
    for i in range(1, len(x_abs)):
        if x_abs[i] > prev:
            prev = alpha_a * prev + (1 - alpha_a) * x_abs[i]
        else:
            prev = alpha_r * prev + (1 - alpha_r) * x_abs[i]
        env[i] = prev
    return env


def compress(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float = 2.0,
    knee_db: float = 6.0,
    attack_ms: float = 30.0,
    release_ms: float = 100.0,
    max_gr_db: float = 6.0,
) -> np.ndarray:
    """Feed-forward compressor with soft knee, linked across stereo.
    max_gr_db caps the per-sample gain reduction to avoid audible pumping
    on dynamic input (Grace feedback 2026-05-15)."""
    peak = np.maximum(np.max(np.abs(audio), axis=1).astype(np.float64), 1e-10)
    alpha_a = np.exp(-1.0 / (attack_ms * 1e-3 * sr))
    alpha_r = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    env = _env_follow(peak, alpha_a, alpha_r)
    env_db = 20 * np.log10(np.maximum(env, 1e-10))

    over = env_db - threshold_db
    # Soft-knee gain reduction (dB)
    below = over < -knee_db / 2
    above = over > knee_db / 2
    in_knee = ~below & ~above
    reduction_db = np.zeros_like(over)
    reduction_db[above] = over[above] * (1 - 1 / ratio)
    reduction_db[in_knee] = (
        (over[in_knee] + knee_db / 2) ** 2 / (2 * knee_db) * (1 - 1 / ratio)
    )
    reduction_db = np.minimum(reduction_db, max_gr_db)
    gain = 10 ** (-reduction_db / 20)
    return audio * gain[:, np.newaxis]


def fit_compressor(
    audio: np.ndarray,
    sr: int,
    target_crest_db: float,
    ratio: float = 2.0,
    knee_db: float = 6.0,
    attack_ms: float = 30.0,
    release_ms: float = 100.0,
    max_gr_db: float = 6.0,
    tolerance_db: float = 0.2,
    max_iter: int = 14,
) -> tuple[np.ndarray, float]:
    """Bisection-search threshold until output crest matches target."""
    lo, hi = -40.0, 0.0
    best, best_diff, best_thr = audio, abs(crest_db(audio) - target_crest_db), 0.0
    for _ in range(max_iter):
        thr = (lo + hi) / 2
        out = compress(audio, sr, thr, ratio, knee_db, attack_ms, release_ms, max_gr_db)
        c = crest_db(out)
        diff = abs(c - target_crest_db)
        if diff < best_diff:
            best, best_diff, best_thr = out, diff, thr
        if c > target_crest_db:
            hi = thr  # need more compression -> lower threshold
        else:
            lo = thr  # need less compression -> higher threshold
        if best_diff < tolerance_db:
            break
    return best, best_thr
