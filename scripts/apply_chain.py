#!/usr/bin/env python3
"""Apply Grace's parametric mastering chain to a single mix."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from src.chain import apply_chain, available_genres


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input WAV (unmastered mix)")
    p.add_argument("--output", required=True, help="Output WAV (mastered)")
    p.add_argument("--genre", required=True,
                   help=f"Profile to use. Available: {available_genres()}")
    p.add_argument("--with-width", action="store_true",
                   help="Enable M/S width stage (off by default; see chain.py docstring)")
    args = p.parse_args()

    audio, sr = sf.read(args.input, always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    print(f"Mastering {args.input} as {args.genre}...")
    mastered = apply_chain(audio.astype(np.float64), sr, args.genre,
                           use_width=args.with_width, verbose=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, np.clip(mastered, -1.0, 1.0), sr, subtype="PCM_24")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
