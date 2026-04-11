"""
Analyze evaluation results and produce summary statistics + visualizations.

Usage:
    python scripts/analyze_results.py --evaluations data/evaluations/evaluations_*.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_evaluations(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_dataframe(evaluations: list[dict]) -> pd.DataFrame:
    rows = []
    dimensions = ["accuracy", "suitability", "completeness", "harm_avoidance", "disclosure", "actionability"]

    for ev in evaluations:
        if ev.get("error") or ev.get("aggregate_score") is None:
            continue

        row = {
            "model": ev.get("display_name", ev.get("model_key")),
            "model_key": ev["model_key"],
            "scenario": ev["scenario_id"],
            "prompt_key": ev.get("prompt_key", "default"),
            "category": ev.get("category"),
            "sophistication": ev.get("sophistication"),
            "emotional_state": ev.get("emotional_state"),
            "pressure": ev.get("pressure"),
            "run_index": ev.get("run_index", 0),
            "aggregate": ev["aggregate_score"],
        }

        for dim in dimensions:
            if isinstance(ev.get(dim), dict):
                row[dim] = ev[dim]["score"]

        row["red_flags_count"] = len(ev.get("rubric_red_flags_triggered", []))
        row["must_mention_hits"] = len(ev.get("rubric_must_mention_hits", []))
        row["must_mention_misses"] = len(ev.get("rubric_must_mention_misses", []))

        rows.append(row)

    return pd.DataFrame(rows)


def print_variation_breakdown(df: pd.DataFrame):
    """Core equity analysis: how does advice quality vary by prompt framing?"""
    if df["sophistication"].isna().all():
        return

    print("\n" + "=" * 90)
    print("PROMPT VARIATION EFFECTS (the equity angle)")
    print("=" * 90)

    for axis in ["sophistication", "emotional_state", "pressure"]:
        if df[axis].isna().all():
            continue
        print(f"\n--- By {axis} ---")
        tab = df.groupby([axis])["aggregate"].agg(["mean", "std", "count"]).round(2)
        print(tab.to_string())

    print("\n--- Per-model score drop by sophistication (naive vs sophisticated) ---")
    if not df["sophistication"].isna().all():
        piv = df.groupby(["model", "sophistication"])["aggregate"].mean().unstack()
        if "naive" in piv.columns and "sophisticated" in piv.columns:
            piv["delta (soph - naive)"] = (piv["sophisticated"] - piv["naive"]).round(2)
            print(piv.round(2).to_string())


def print_model_rankings(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("OVERALL MODEL RANKINGS")
    print("=" * 90)

    agg = df.groupby("model").agg(
        avg_score=("aggregate", "mean"),
        std_score=("aggregate", "std"),
        min_score=("aggregate", "min"),
        max_score=("aggregate", "max"),
        n_scenarios=("aggregate", "count"),
        avg_red_flags=("red_flags_count", "mean"),
    ).sort_values("avg_score", ascending=False)

    print(f"\n{'Model':<25} {'Avg':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'N':>4} {'Avg Red Flags':>14}")
    print("-" * 75)
    for model, row in agg.iterrows():
        print(
            f"{model:<25} {row['avg_score']:>6.2f} {row['std_score']:>6.2f} "
            f"{row['min_score']:>6.1f} {row['max_score']:>6.1f} {int(row['n_scenarios']):>4} "
            f"{row['avg_red_flags']:>14.2f}"
        )


def print_dimension_breakdown(df: pd.DataFrame):
    dimensions = ["accuracy", "suitability", "completeness", "harm_avoidance", "disclosure", "actionability"]

    print("\n" + "=" * 90)
    print("SCORES BY DIMENSION (model averages)")
    print("=" * 90)

    header = f"{'Model':<25}" + "".join(f"{d[:8]:>10}" for d in dimensions)
    print(header)
    print("-" * (25 + 10 * len(dimensions)))

    model_avgs = df.groupby("model")[dimensions].mean().sort_values("accuracy", ascending=False)
    for model, row in model_avgs.iterrows():
        line = f"{model:<25}" + "".join(f"{row[d]:>10.2f}" for d in dimensions)
        print(line)


def print_scenario_breakdown(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("SCORES BY SCENARIO (averaged across models)")
    print("=" * 90)

    agg = df.groupby("scenario").agg(
        avg_score=("aggregate", "mean"),
        avg_red_flags=("red_flags_count", "mean"),
        avg_must_mention_misses=("must_mention_misses", "mean"),
    ).sort_values("avg_score", ascending=False)

    print(f"\n{'Scenario':<35} {'Avg Score':>10} {'Avg Red Flags':>14} {'Avg Misses':>12}")
    print("-" * 75)
    for scenario, row in agg.iterrows():
        print(
            f"{scenario:<35} {row['avg_score']:>10.2f} "
            f"{row['avg_red_flags']:>14.2f} {row['avg_must_mention_misses']:>12.2f}"
        )


def print_worst_performers(df: pd.DataFrame, n: int = 10):
    print(f"\n{'=' * 90}")
    print(f"WORST {n} MODEL x PROMPT COMBINATIONS")
    print("=" * 90)

    cols = ["model", "scenario", "prompt_key", "aggregate", "red_flags_count"]
    worst = df.nsmallest(n, "aggregate")[cols]
    for _, row in worst.iterrows():
        label = f"{row['scenario']}/{row['prompt_key']}"
        print(f"  {row['model']:<25} x {label:<45} score={row['aggregate']:.1f} red_flags={int(row['red_flags_count'])}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--evaluations", required=True, help="Path to evaluations JSON file")
    parser.add_argument("--output-csv", default=None, help="Optional: save results as CSV")
    args = parser.parse_args()

    evaluations = load_evaluations(args.evaluations)
    df = build_dataframe(evaluations)

    if df.empty:
        print("No valid evaluations found.")
        return

    print(f"Loaded {len(df)} evaluations across {df['model'].nunique()} models and {df['scenario'].nunique()} scenarios")

    print_model_rankings(df)
    print_dimension_breakdown(df)
    print_scenario_breakdown(df)
    print_variation_breakdown(df)
    print_worst_performers(df)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
