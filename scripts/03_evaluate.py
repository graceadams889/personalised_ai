#!/usr/bin/env python3
"""Step 3: Evaluate model predictions against ground truth and Matchering baseline."""
import sys
sys.path.insert(0, ".")

from src.evaluation.evaluate import run_evaluation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", help="Directory of model prediction WAVs")
    parser.add_argument("--reference", help="Reference master WAV for Matchering baseline")
    args = parser.parse_args()
    run_evaluation(args.config, args.predictions, args.reference)
