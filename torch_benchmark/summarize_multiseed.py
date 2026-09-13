#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


METRICS = (
    "max_abs",
    "p99_abs",
    "max_rel",
    "max_normalized_l2",
    "mean_normalized_l2",
    "global_l2",
    "min_cosine",
    "global_cosine",
    "tensor_allclose_pct",
    "element_allclose_pct",
    "min_cosine_fwd",
    "min_cosine_act_grads",
    "min_cosine_param_grads",
    "min_cosine_updated_params",
)


def read_trials(results_dir):
    trials = []
    for path in sorted(results_dir.glob("**/summary.json")):
        relative = path.relative_to(results_dir)
        parts = relative.parts
        if len(parts) < 3 or not parts[-2].startswith("seed_"):
            continue
        family = parts[0]
        seed = int(parts[-2][len("seed_"):])
        with path.open(encoding="utf-8") as f:
            aggregate = json.load(f)["aggregate"]
        trials.append({"family": family, "seed": seed, **aggregate})
    return trials


def mean_std(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return mean, float("nan")
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return mean, math.sqrt(variance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    trials = read_trials(args.results_dir)
    if not trials:
        raise SystemExit("No summary.json trial files found")

    grouped = {}
    for trial in trials:
        grouped.setdefault(trial["family"], []).append(trial)

    fields = ["family", "seed", *METRICS]
    for family, family_trials in sorted(grouped.items()):
        trial_path = args.results_dir / f"{family}_trial_metrics.csv"
        with trial_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: trial.get(field, "") for field in fields} for trial in family_trials)
        print(f"Wrote {trial_path}")

    summary_fields = ["family", "trials", "seeds"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])

    for family, family_trials in sorted(grouped.items()):
        summary_path = args.results_dir / f"{family}_summary_metrics.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            row = {
                "family": family,
                "trials": len(family_trials),
                "seeds": " ".join(str(t["seed"]) for t in family_trials),
            }
            for metric in METRICS:
                mean, std = mean_std([t.get(metric, float("nan")) for t in family_trials])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            writer.writerow(row)
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()