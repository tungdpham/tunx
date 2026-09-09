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
    "min_cosine",
    "tensor_allclose_pct",
    "element_allclose_pct",
)


def read_trials(results_dir):
    trials = []
    for path in sorted(results_dir.glob("**/summary.json")):
        relative = path.relative_to(results_dir)
        parts = relative.parts
        if len(parts) < 3 or not parts[-2].startswith("seed_"):
            continue
        family = "resnet50" if parts[0] == "resnet" else parts[1]
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

    trial_path = args.results_dir / "trial_metrics.csv"
    fields = ["family", "seed", *METRICS]
    with trial_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: trial.get(field, "") for field in fields} for trial in trials)

    summary_path = args.results_dir / "summary_metrics.csv"
    summary_fields = ["family", "trials", "seeds"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    grouped = {}
    for trial in trials:
        grouped.setdefault(trial["family"], []).append(trial)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for family, family_trials in sorted(grouped.items()):
            row = {
                "family": family,
                "trials": len(family_trials),
                "seeds": " ".join(str(t["seed"]) for t in family_trials),
            }
            for metric in METRICS:
                mean, std = mean_std([t[metric] for t in family_trials])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            writer.writerow(row)

    print(f"Wrote {trial_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()