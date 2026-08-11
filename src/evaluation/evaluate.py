import json
import yaml
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from .metrics import compute_all_metrics
from .baseline import run_matchering_on_test_set


def evaluate_predictions(
    predictions_dir: str,
    test_dir: str,
    sr: int = 44100,
) -> pd.DataFrame:
    pred_path = Path(predictions_dir)
    test_path = Path(test_dir)
    master_dir = test_path / "master"

    pred_files = sorted(pred_path.glob("*.wav"))
    all_metrics = []

    for pred_file in pred_files:
        master_file = master_dir / pred_file.name
        if not master_file.exists():
            print(f"  Warning: no ground truth for {pred_file.stem}, skipping")
            continue

        predicted, _ = sf.read(str(pred_file), dtype="float32")
        target, _ = sf.read(str(master_file), dtype="float32")

        min_len = min(len(predicted), len(target))
        predicted = predicted[:min_len]
        target = target[:min_len]

        metrics = compute_all_metrics(predicted, target, sr)
        metrics["track"] = pred_file.stem
        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


def run_evaluation(
    config_path: str = "configs/default.yaml",
    model_predictions_dir: Optional[str] = None,
    matchering_reference_path: Optional[str] = None,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    sr = data_cfg["sample_rate"]
    test_dir = str(Path(data_cfg["splits_dir"]) / "test")
    output_dir = config["inference"]["output_dir"]

    print("=" * 60)
    print("Evaluation Pipeline")
    print("=" * 60)

    if model_predictions_dir:
        print("\n[1/3] Evaluating model predictions...")
        model_df = evaluate_predictions(model_predictions_dir, test_dir, sr)
        print("\nModel metrics (mean):")
        numeric_cols = model_df.select_dtypes(include=[np.number]).columns
        print(model_df[numeric_cols].mean().to_string())
    else:
        model_df = None
        print("\n[1/3] No model predictions provided, skipping.")

    if config["evaluation"].get("matchering_baseline") and matchering_reference_path:
        print("\n[2/3] Running Matchering baseline...")
        matchering_dir = str(Path(output_dir) / "matchering_baseline")
        matchering_results = run_matchering_on_test_set(
            test_dir, matchering_reference_path, matchering_dir
        )

        print("\n[3/3] Evaluating Matchering baseline...")
        baseline_df = evaluate_predictions(matchering_dir, test_dir, sr)
        print("\nMatchering baseline metrics (mean):")
        numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns
        print(baseline_df[numeric_cols].mean().to_string())
    else:
        baseline_df = None
        print("\n[2/3] Matchering baseline skipped.")

    results = {}
    if model_df is not None:
        results["model"] = model_df.to_dict(orient="records")
    if baseline_df is not None:
        results["matchering_baseline"] = baseline_df.to_dict(orient="records")

    results_path = Path(output_dir) / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    if model_df is not None and baseline_df is not None:
        print("\n" + "=" * 60)
        print("Comparison: Model vs Matchering Baseline")
        print("=" * 60)
        model_means = model_df[numeric_cols].mean()
        baseline_means = baseline_df[numeric_cols].mean()
        comparison = pd.DataFrame({
            "Model": model_means,
            "Matchering": baseline_means,
            "Difference": model_means - baseline_means,
        })
        print(comparison.to_string())

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", help="Directory of model prediction WAVs")
    parser.add_argument("--reference", help="Reference master WAV for Matchering baseline")
    args = parser.parse_args()
    run_evaluation(args.config, args.predictions, args.reference)
