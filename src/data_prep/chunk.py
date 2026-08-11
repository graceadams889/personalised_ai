import numpy as np
from typing import List, Tuple


def chunk_pair(
    pre: np.ndarray,
    master: np.ndarray,
    sr: int = 44100,
    segment_length_sec: float = 5.0,
    segment_hop_sec: float = 2.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    segment_length = int(segment_length_sec * sr)
    segment_hop = int(segment_hop_sec * sr)

    min_len = min(len(pre), len(master))
    pre = pre[:min_len]
    master = master[:min_len]

    if min_len < segment_length:
        pad_len = segment_length - min_len
        if pre.ndim > 1:
            pre = np.pad(pre, ((0, pad_len), (0, 0)))
            master = np.pad(master, ((0, pad_len), (0, 0)))
        else:
            pre = np.pad(pre, (0, pad_len))
            master = np.pad(master, (0, pad_len))
        return [(pre, master)]

    chunks = []
    for start in range(0, min_len - segment_length + 1, segment_hop):
        end = start + segment_length
        pre_chunk = pre[start:end]
        master_chunk = master[start:end]

        chunk_rms = np.sqrt(np.mean(pre_chunk ** 2))
        if chunk_rms < 1e-6:
            continue

        chunks.append((pre_chunk, master_chunk))

    return chunks
