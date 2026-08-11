#!/usr/bin/env python3
"""Step 1: Run the data preparation pipeline."""
import sys
sys.path.insert(0, ".")

from src.data_prep.pipeline import run_data_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_data_pipeline(args.config)
