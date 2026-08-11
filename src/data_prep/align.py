import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple


def cross_correlate_offset(pre: np.ndarray, master: np.ndarray, max_offset_samples: int = 48000) -> int:
    if pre.ndim > 1:
        pre_mono = pre.mean(axis=1)
    else:
        pre_mono = pre

    if master.ndim > 1:
        master_mono = master.mean(axis=1)
    else:
        master_mono = master

    n = min(len(pre_mono), len(master_mono), max_offset_samples * 4)
    pre_chunk = pre_mono[:n]
    master_chunk = master_mono[:n]

    correlation = np.correlate(master_chunk, pre_chunk, mode="full")
    mid = len(pre_chunk) - 1
    search_range = min(max_offset_samples, mid)
    search_slice = correlation[mid - search_range : mid + search_range + 1]
    offset = np.argmax(search_slice) - search_range

    return int(offset)


def apply_offset(pre: np.ndarray, master: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    if offset > 0:
        master = master[offset:]
        pre = pre[: len(master)]
    elif offset < 0:
        pre = pre[-offset:]
        master = master[: len(pre)]

    min_len = min(len(pre), len(master))
    return pre[:min_len], master[:min_len]


def align_pair(pre_path: Path, master_path: Path, sr: int = 44100) -> Tuple[np.ndarray, np.ndarray, int]:
    pre, pre_sr = sf.read(str(pre_path), dtype="float32")
    master, master_sr = sf.read(str(master_path), dtype="float32")

    assert pre_sr == master_sr == sr, f"Sample rate mismatch: pre={pre_sr}, master={master_sr}, expected={sr}"

    offset = cross_correlate_offset(pre, master)
    pre_aligned, master_aligned = apply_offset(pre, master, offset)

    if abs(offset) > 0:
        print(f"  {pre_path.stem}: aligned with offset={offset} samples ({offset/sr*1000:.1f} ms)")

    return pre_aligned, master_aligned, offset
