"""M/S stereo width with closed-form fitting to a target L-R correlation.

corr(L, R) = (A - k²B) / (A + k²B), where A = E[mid²], B = E[side²], k = side gain.
Solving for k: k = sqrt(A(1-corr) / (B(1+corr))).
"""
import numpy as np


def apply_width(audio: np.ndarray, side_gain: float) -> np.ndarray:
    mid = (audio[:, 0] + audio[:, 1]) / 2
    side = (audio[:, 0] - audio[:, 1]) / 2 * side_gain
    L = mid + side
    R = mid - side
    return np.stack([L, R], axis=1)


def fit_width(audio: np.ndarray, target_corr: float) -> tuple[np.ndarray, float]:
    mid = (audio[:, 0] + audio[:, 1]) / 2
    side = (audio[:, 0] - audio[:, 1]) / 2
    A = float(np.mean(mid ** 2))
    B = float(np.mean(side ** 2))
    if B < 1e-12:
        return audio, 1.0  # already mono, can't widen
    target_corr = float(np.clip(target_corr, -0.999, 0.999))
    k_sq = A * (1 - target_corr) / (B * (1 + target_corr))
    k = float(np.sqrt(max(k_sq, 0.0)))
    return apply_width(audio, k), k
