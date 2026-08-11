"""The full Grace-style mastering chain: EQ -> compressor -> [width] -> limiter.

Width stage is OFF by default. Iteration on 2026-05-15 found that fitting a
single side-gain to a single per-genre correlation target produced extreme
gain values (0.75-2.68x) and audible phasey artifacts, while the mix's
existing stereo image was already close to what the master preserves.
Pass use_width=True to re-enable for comparison.
"""
import numpy as np

from .profile import get_profile
from .eq import apply_eq
from .compressor import fit_compressor
from .width import fit_width
from .limiter import fit_loudness


def apply_chain(
    audio: np.ndarray,
    sr: int,
    genre: str,
    *,
    profile: dict | None = None,
    use_width: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    # If a pre-computed profile is passed (e.g. from LOOCV), use it verbatim;
    # otherwise load the per-genre profile from the profile CSV as usual.
    if profile is None:
        profile = get_profile(genre)
    if verbose:
        print(f"  profile: {genre} (n={profile['n_tracks']}) "
              f"LUFS={profile['master_lufs']:.1f}, crest={profile['master_crest_db']:.1f} dB, "
              f"corr={profile['master_lr_corr']:+.2f}")

    audio = apply_eq(audio, sr, profile["eq_gains_db"])
    audio, thr = fit_compressor(audio, sr, profile["master_crest_db"])
    if verbose:
        print(f"  compressor threshold fitted to {thr:.1f} dBFS")
    if use_width:
        audio, k = fit_width(audio, profile["master_lr_corr"])
        if verbose:
            print(f"  side gain fitted to {k:.2f}")
    elif verbose:
        print("  width stage: bypassed (mix stereo image preserved)")
    audio = fit_loudness(audio, sr, profile["master_lufs"])
    return audio
