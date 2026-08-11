import yaml
from pathlib import Path
from typing import Dict

from .validate import find_pairs, validate_pairs
from .align import align_pair
from .normalise import loudness_normalise
from .chunk import chunk_pair
from .split import create_splits


def run_data_pipeline(config_path: str = "configs/default.yaml") -> Dict[str, int]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    sr = data_cfg["sample_rate"]

    print("=" * 60)
    print("Phase 1: Data Preparation Pipeline")
    print("=" * 60)

    print("\n[1/5] Finding pairs...")
    pairs = find_pairs(data_cfg["raw_dir"])
    print(f"  Found {len(pairs)} pairs")

    print("\n[2/5] Validating pairs...")
    valid_pairs = validate_pairs(pairs, expected_sr=sr, expected_bit_depth=data_cfg["bit_depth"])

    print("\n[3/5] Aligning and normalising...")
    track_chunks = {}
    for pre_path, master_path in valid_pairs:
        track_name = pre_path.stem
        print(f"  Processing: {track_name}")

        pre_aligned, master_aligned, offset = align_pair(pre_path, master_path, sr=sr)

        pre_norm, original_lufs, gain_db = loudness_normalise(
            pre_aligned, sr=sr, target_lufs=data_cfg["target_lufs"]
        )
        print(f"    Pre-master: {original_lufs:.1f} LUFS -> {data_cfg['target_lufs']} LUFS (gain: {gain_db:+.1f} dB)")

        print("\n[4/5] Chunking...")
        chunks = chunk_pair(
            pre_norm,
            master_aligned,
            sr=sr,
            segment_length_sec=data_cfg["segment_length_sec"],
            segment_hop_sec=data_cfg["segment_hop_sec"],
        )
        print(f"    {track_name}: {len(chunks)} chunks")
        track_chunks[track_name] = chunks

    total_chunks = sum(len(c) for c in track_chunks.values())
    print(f"\n  Total chunks: {total_chunks} from {len(track_chunks)} tracks")

    print("\n[5/5] Creating train/val/test splits...")
    counts = create_splits(
        track_chunks,
        splits_dir=data_cfg["splits_dir"],
        sr=sr,
        ratios=data_cfg["split_ratios"],
        seed=config["project"]["seed"],
    )

    print("\n" + "=" * 60)
    print("Data preparation complete")
    print("=" * 60)
    return counts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data preparation pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    run_data_pipeline(args.config)
