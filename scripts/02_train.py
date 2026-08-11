#!/usr/bin/env python3
"""Step 2: Train the mastering model."""
import sys
sys.path.insert(0, ".")

from src.training.trainer import Trainer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.train()
