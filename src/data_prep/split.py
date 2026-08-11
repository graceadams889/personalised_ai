import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple


def create_splits(
    track_chunks: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    splits_dir: str,
    sr: int = 44100,
    ratios: Dict[str, float] = None,
    seed: int = 42,
) -> Dict[str, int]:
    if ratios is None:
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    splits_path = Path(splits_dir)
    for split in ratios:
        (splits_path / split / "pre").mkdir(parents=True, exist_ok=True)
        (splits_path / split / "master").mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    track_names = sorted(track_chunks.keys())
    rng.shuffle(track_names)

    n_tracks = len(track_names)
    n_train = max(1, int(n_tracks * ratios["train"]))
    n_val = max(1, int(n_tracks * ratios["val"]))

    train_tracks = track_names[:n_train]
    val_tracks = track_names[n_train : n_train + n_val]
    test_tracks = track_names[n_train + n_val :]

    if not test_tracks and len(val_tracks) > 1:
        test_tracks = [val_tracks.pop()]

    assignment = {}
    for t in train_tracks:
        assignment[t] = "train"
    for t in val_tracks:
        assignment[t] = "val"
    for t in test_tracks:
        assignment[t] = "test"

    counts = {"train": 0, "val": 0, "test": 0}
    for track_name, split in assignment.items():
        chunks = track_chunks[track_name]
        for i, (pre_chunk, master_chunk) in enumerate(chunks):
            chunk_name = f"{track_name}_chunk{i:04d}.wav"
            sf.write(
                str(splits_path / split / "pre" / chunk_name),
                pre_chunk, sr, subtype="FLOAT",
            )
            sf.write(
                str(splits_path / split / "master" / chunk_name),
                master_chunk, sr, subtype="FLOAT",
            )
            counts[split] += 1

    print(f"Split complete: train={counts['train']}, val={counts['val']}, test={counts['test']} chunks")
    print(f"Tracks: train={len(train_tracks)}, val={len(val_tracks)}, test={len(test_tracks)}")
    return counts
