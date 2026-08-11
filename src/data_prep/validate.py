import os
import soundfile as sf
from pathlib import Path
from typing import List, Tuple


def find_pairs(raw_dir: str) -> List[Tuple[Path, Path]]:
    raw_path = Path(raw_dir)
    pre_dir = raw_path / "pre"
    master_dir = raw_path / "master"

    if not pre_dir.exists():
        raise FileNotFoundError(f"Expected directory: {pre_dir}")
    if not master_dir.exists():
        raise FileNotFoundError(f"Expected directory: {master_dir}")

    pre_files = {f.stem: f for f in sorted(pre_dir.glob("*.wav"))}
    master_files = {f.stem: f for f in sorted(master_dir.glob("*.wav"))}

    common = sorted(set(pre_files.keys()) & set(master_files.keys()))
    if not common:
        raise ValueError(
            f"No matching pairs found between {pre_dir} and {master_dir}. "
            "Ensure filenames match (e.g., track_001.wav in both directories)."
        )

    pairs = [(pre_files[name], master_files[name]) for name in common]

    orphan_pre = set(pre_files.keys()) - set(master_files.keys())
    orphan_master = set(master_files.keys()) - set(pre_files.keys())
    if orphan_pre:
        print(f"Warning: {len(orphan_pre)} pre-master files have no matching master: {sorted(orphan_pre)[:5]}")
    if orphan_master:
        print(f"Warning: {len(orphan_master)} master files have no matching pre-master: {sorted(orphan_master)[:5]}")

    return pairs


def validate_pairs(
    pairs: List[Tuple[Path, Path]],
    expected_sr: int = 44100,
    expected_bit_depth: int = 24,
) -> List[Tuple[Path, Path]]:
    valid = []
    for pre_path, master_path in pairs:
        pre_info = sf.info(str(pre_path))
        master_info = sf.info(str(master_path))

        issues = []

        if pre_info.samplerate != master_info.samplerate:
            issues.append(
                f"Sample rate mismatch: pre={pre_info.samplerate}, master={master_info.samplerate}"
            )

        if pre_info.channels != master_info.channels:
            issues.append(
                f"Channel mismatch: pre={pre_info.channels}, master={master_info.channels}"
            )

        if pre_info.samplerate != expected_sr:
            issues.append(
                f"Unexpected sample rate: {pre_info.samplerate} (expected {expected_sr})"
            )

        subtype = pre_info.subtype
        if expected_bit_depth == 24 and "24" not in subtype:
            issues.append(f"Pre-master bit depth: {subtype} (expected 24-bit)")

        if issues:
            print(f"Skipping {pre_path.stem}: {'; '.join(issues)}")
            continue

        valid.append((pre_path, master_path))

    print(f"Validated {len(valid)}/{len(pairs)} pairs")
    return valid
