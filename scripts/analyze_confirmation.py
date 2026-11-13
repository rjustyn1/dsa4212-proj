"""Summarise the frozen five-seed Moonlight Muon / SOAP-Muon comparison.

The script consumes the saved per-run ``metrics.json`` files rather than
hand-entered results.  It implements the endpoints defined in
``research/analysis/analysis_plan.md`` and writes a Markdown report suitable
for inclusion in the project report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (201, 202, 203, 204, 205)
NOMINAL_TRAINING_TOKENS = 200_000_000


def load_metrics(metrics_dir: Path, seed: int, method: str) -> dict[str, Any]:
    """Load one locally downloaded confirmation metrics file."""
    path = metrics_dir / f"seed{seed}-{method}.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing {path}. Download the saved Modal metric files before analysis."
        ) from error


def curve_area(metrics: dict[str, Any]) -> float:
    """Return the prespecified sampled-curve area divided by 200M tokens."""
    curve = metrics["training"]["curve"]
    if len(curve) < 2:
        raise ValueError("A confirmation curve needs at least two evaluation points.")

    area = 0.0
    for left, right in zip(curve, curve[1:]):
        width = right["tokens_processed"] - left["tokens_processed"]
        if width <= 0:
            raise ValueError("Curve tokens must be strictly increasing.")
        area += width * (left["mean_nll"] + right["mean_nll"]) / 2
    return area / NOMINAL_TRAINING_TOKENS


def first_crossing(metrics: dict[str, Any], target_nll: float) -> int | None:
    """Return the first logged token count at or below ``target_nll``."""
    for point in metrics["training"]["curve"]:
        if point["mean_nll"] <= target_nll:
            return point["tokens_processed"]
    return None


def first_crossing_elapsed_seconds(
    metrics: dict[str, Any], target_nll: float
) -> float | None:
    """Return elapsed wall-clock seconds at the first logged threshold crossing."""
    for point in metrics["training"]["curve"]:
        if point["mean_nll"] <= target_nll:
            return float(point["elapsed_wall_clock_seconds"])
    return None


def mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return sum(values) / len(values)


def percentile(values: list[float], probability: float) -> float:
    """Use a reproducible nearest-rank percentile for bootstrap reporting."""
    index = round((len(values) - 1) * probability)
    return sorted(values)[index]


def bootstrap_mean_interval(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    bootstrap_means = [mean(rng.choice(values) for _ in values) for _ in range(samples)]
    return percentile(bootstrap_means, 0.025), percentile(bootstrap_means, 0.975)


def format_tokens(value: int | None) -> str:
    return "not reached" if value is None else f"{value / 1_000_000:.1f}M"


def format_difference(value: float, places: int = 6) -> str:
    return f"{value:+.{places}f}"


def analyse_pair(seed: int, baseline: dict[str, Any], soap: dict[str, Any]) -> dict[str, Any]:
    baseline_curve = baseline["training"]["curve"]
    soap_curve = soap["training"]["curve"]
    baseline_tokens = [point["tokens_processed"] for point in baseline_curve]
    soap_tokens = [point["tokens_processed"] for point in soap_curve]
    if baseline_tokens != soap_tokens:
        raise ValueError(f"Seed {seed} methods do not share the same curve-token grid.")

    baseline_area = curve_area(baseline)
    soap_area = curve_area(soap)
    baseline_final = baseline["validation"]["mean_nll"]
    soap_final = soap["validation"]["mean_nll"]
    return {
        "seed": seed,
        "baseline_area": baseline_area,
        "soap_area": soap_area,
        "area_delta": soap_area - baseline_area,
        "baseline_final": baseline_final,
        "soap_final": soap_final,
        "final_delta": soap_final - baseline_final,
        "baseline_throughput": baseline["training"]["tokens_per_second"],
        "soap_throughput": soap["training"]["tokens_per_second"],
        "baseline_memory_gib": baseline["training"]["peak_memory_bytes"] / 2**30,
        "soap_memory_gib": soap["training"]["peak_memory_bytes"] / 2**30,
        "baseline_425": first_crossing(baseline, 4.25),
        "soap_425": first_crossing(soap, 4.25),
        "baseline_420": first_crossing(baseline, 4.20),
        "soap_420": first_crossing(soap, 4.20),
        "baseline_425_seconds": first_crossing_elapsed_seconds(baseline, 4.25),
        "soap_425_seconds": first_crossing_elapsed_seconds(soap, 4.25),
        "baseline_420_seconds": first_crossing_elapsed_seconds(baseline, 4.20),
        "soap_420_seconds": first_crossing_elapsed_seconds(soap, 4.20),
    }


def paired_t_interval(values: list[float]) -> tuple[float, float]:
    """Return the descriptive two-sided 95% paired t interval for five pairs."""
    if len(values) != 5:
        raise ValueError("The fixed confirmation design requires five paired values.")
    average = mean(values)
    standard_deviation = math.sqrt(
        sum((value - average) ** 2 for value in values) / (len(values) - 1)
    )
    critical_value = 2.776445105  # t_{0.975, 4}
    half_width = critical_value * standard_deviation / math.sqrt(len(values))
    return average - half_width, average + half_width


def exact_two_sided_sign_p_value(values: list[float]) -> float:
    """Return the exact two-sided binomial sign-test p-value, omitting ties."""
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    tail = min(positive, len(nonzero) - positive)
    probability = sum(math.comb(len(nonzero), count) for count in range(tail + 1))
    return min(1.0, 2 * probability / 2 ** len(nonzero))


def render_report(rows: list[dict[str, Any]], bootstrap_samples: int) -> str:
    area_deltas = [row["area_delta"] for row in rows]
    final_deltas = [row["final_delta"] for row in rows]
    area_interval = bootstrap_mean_interval(area_deltas, bootstrap_samples, seed=4212)
    final_interval = bootstrap_mean_interval(final_deltas, bootstrap_samples, seed=4212)
    area_t_interval = paired_t_interval(area_deltas)
    final_t_interval = paired_t_interval(final_deltas)

    baseline_area = mean(row["baseline_area"] for row in rows)
    soap_area = mean(row["soap_area"] for row in rows)
    baseline_final = mean(row["baseline_final"] for row in rows)
    soap_final = mean(row["soap_final"] for row in rows)
    baseline_throughput = mean(row["baseline_throughput"] for row in rows)
    soap_throughput = mean(row["soap_throughput"] for row in rows)
    baseline_memory = mean(row["baseline_memory_gib"] for row in rows)
    soap_memory = mean(row["soap_memory_gib"] for row in rows)

    lines = [
        "# Moonlight Muon / SOAP-Muon confirmation results",
        "",
        "The methods are paired by seed (initialisation, data order, and batch regime). "
        "All figures below are generated from saved run metrics.",
        "",
        "## Prespecified primary endpoint",
        "",
        "The primary endpoint is the trapezoidal sampled validation-NLL curve area, "
        "normalised by the nominal 200M-token budget. Lower is better.",
        "",
        "| Seed | Moonlight Muon | SOAP-Muon | SOAP − baseline |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['seed']} | {row['baseline_area']:.6f} | {row['soap_area']:.6f} | "
        f"{format_difference(row['area_delta'])} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            f"Mean paired difference: **{format_difference(mean(area_deltas))}** "
            f"({mean(area_deltas) / baseline_area:.3%}); range "
            f"[{min(area_deltas):+.6f}, {max(area_deltas):+.6f}].",
            f"Deterministic {bootstrap_samples:,}-resample bootstrap 95% interval: "
            f"[{area_interval[0]:+.6f}, {area_interval[1]:+.6f}].",
            f"Descriptive paired t 95% interval: [{area_t_interval[0]:+.6f}, {area_t_interval[1]:+.6f}]; "
            f"exact two-sided sign-test p-value: {exact_two_sided_sign_p_value(area_deltas):.4f}.",
            f"SOAP-Muon has lower area in {sum(delta < 0 for delta in area_deltas)}/{len(rows)} pairs.",
            "",
            "## Secondary endpoints",
            "",
            "| Outcome | Moonlight Muon mean | SOAP-Muon mean | SOAP − baseline |",
            "|---|---:|---:|---:|",
            f"| Final 5M-token validation NLL | {baseline_final:.6f} | {soap_final:.6f} | {format_difference(mean(final_deltas))} |",
            f"| Training throughput (tokens/s) | {baseline_throughput:,.0f} | {soap_throughput:,.0f} | {((soap_throughput / baseline_throughput) - 1):+.2%} |",
            f"| Peak training memory (GiB) | {baseline_memory:.3f} | {soap_memory:.3f} | {soap_memory - baseline_memory:+.3f} |",
            "",
            f"Final-NLL bootstrap 95% interval: [{final_interval[0]:+.6f}, {final_interval[1]:+.6f}].",
            f"Final-NLL paired t 95% interval: [{final_t_interval[0]:+.6f}, {final_t_interval[1]:+.6f}]; "
            f"exact two-sided sign-test p-value: {exact_two_sided_sign_p_value(final_deltas):.4f}.",
            "",
            "| Seed | NLL ≤ 4.25: baseline / SOAP | NLL ≤ 4.20: baseline / SOAP |",
            "|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['seed']} | {format_tokens(row['baseline_425'])} / "
        f"{format_tokens(row['soap_425'])} | {format_tokens(row['baseline_420'])} / "
        f"{format_tokens(row['soap_420'])} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "| Seed | Time to NLL ≤ 4.25: baseline / SOAP (s) | Time to NLL ≤ 4.20: baseline / SOAP (s) |",
            "|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['seed']} | {row['baseline_425_seconds']:.1f} / {row['soap_425_seconds']:.1f} | "
        f"{row['baseline_420_seconds']:.1f} / {row['soap_420_seconds']:.1f} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The primary endpoint favours SOAP-Muon consistently across these five "
            "held-out pairs. Final NLL is mixed, and SOAP-Muon has lower training "
            "throughput; this is a sample-efficiency trade-off rather than an "
            "unqualified final-quality improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/confirmation_metrics"),
        help="Directory containing seed<seed>-<method>.json metric files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional Markdown output path. Defaults to standard output.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100_000,
        help="Number of deterministic bootstrap resamples (default: 100000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive.")

    rows = [
        analyse_pair(
            seed,
            load_metrics(args.metrics_dir, seed, "baseline"),
            load_metrics(args.metrics_dir, seed, "soap"),
        )
        for seed in DEFAULT_SEEDS
    ]
    report = render_report(rows, args.bootstrap_samples)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
