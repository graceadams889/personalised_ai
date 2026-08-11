#!/usr/bin/env python3
"""Step 4: Run inference on new audio files."""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.inference import MasteringInference

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--input", required=True, help="Input WAV file or directory")
    parser.add_argument("--output", required=True, help="Output WAV file or directory")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    engine = MasteringInference(args.checkpoint, args.config)

    if Path(args.input).is_dir():
        engine.process_directory(args.input, args.output)
    else:
        engine.process_file(args.input, args.output)
