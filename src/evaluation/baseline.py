import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from typing import Tuple


def run_matchering_baseline(
    pre_path: str,
    reference_master_path: str,
    output_path: str,
) -> np.ndarray:
    try:
        import matchering as mg
    except ImportError:
        raise ImportError("matchering not installed. Run: pip install matchering")

    mg.process(
        target=pre_path,
        reference=reference_master_path,
        results=[
            mg.pcm24(output_path),
        ],
    )

    result, sr = sf.read(output_path, dtype="float32")
    return result


def run_matchering_on_test_set(
    test_dir: str,
    reference_master_path: str,
    output_dir: str,
) -> dict:
    test_path = Path(test_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pre_files = sorted((test_path / "pre").glob("*.wav"))
    results = {}

    for pre_file in pre_files:
        output_file = out_path / pre_file.name
        print(f"  Matchering: {pre_file.stem}")
        try:
            run_matchering_baseline(
                str(pre_file),
                reference_master_path,
                str(output_file),
            )
            results[pre_file.stem] = str(output_file)
        except Exception as e:
            print(f"    Error: {e}")
            results[pre_file.stem] = None

    return results
