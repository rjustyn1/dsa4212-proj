"""Generate the report figure and results tables from saved confirmation metrics.

The saved per-run metrics are the only numeric input.  The figure uses
matplotlib from the ``report`` dependency group; run this script with
``uv run`` after ``uv sync --all-groups``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_confirmation import DEFAULT_SEEDS, analyse_pair, load_metrics


MUON = "#2a78d6"
SOAP = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
SEED_LINE = "#b9b8b1"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NOMINAL_TOKENS = 200_000_000


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def configure_style() -> None:
    """Match the IEEE body text: Times at 8 pt with hairline axes and grid."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def add_grid(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)


def area_split(tokens: list[int], differences: list[float]) -> tuple[float, float]:
    """Split the area between the mean difference curve and zero by sign."""
    negative = positive = 0.0
    for (t0, d0), (t1, d1) in zip(zip(tokens, differences), zip(tokens[1:], differences[1:])):
        if d0 * d1 >= 0:
            pieces = [(t1 - t0) * (d0 + d1) / 2]
        else:
            crossing = t0 + (t1 - t0) * d0 / (d0 - d1)
            pieces = [(crossing - t0) * d0 / 2, (t1 - crossing) * d1 / 2]
        for piece in pieces:
            negative += min(piece, 0.0)
            positive += max(piece, 0.0)
    return negative / NOMINAL_TOKENS, positive / NOMINAL_TOKENS


def render_training_figure(metrics_dir: Path, output: Path, seeds: tuple[int, ...]) -> None:
    """Plot mean validation loss and the per-seed SOAP-minus-baseline difference."""
    baseline = [load_metrics(metrics_dir, seed, "baseline")["training"]["curve"] for seed in seeds]
    soap = [load_metrics(metrics_dir, seed, "soap")["training"]["curve"] for seed in seeds]
    tokens = [point["tokens_processed"] for point in baseline[0]]
    millions = [token / 1_000_000 for token in tokens]
    baseline_mean = [mean([curve[index]["mean_nll"] for curve in baseline]) for index in range(len(tokens))]
    soap_mean = [mean([curve[index]["mean_nll"] for curve in soap]) for index in range(len(tokens))]
    per_seed = [
        [soap_curve[index]["mean_nll"] - baseline_curve[index]["mean_nll"] for index in range(len(tokens))]
        for baseline_curve, soap_curve in zip(baseline, soap)
    ]
    difference = [soap_value - baseline_value for baseline_value, soap_value in zip(baseline_mean, soap_mean)]
    ahead, behind = area_split(tokens, difference)

    configure_style()
    figure, (loss_axis, difference_axis) = plt.subplots(
        1, 2, figsize=(7.16, 2.45), constrained_layout=True
    )

    loss_axis.plot(millions, baseline_mean, color=MUON, linewidth=1.3, label="Moonlight Muon")
    loss_axis.plot(millions, soap_mean, color=SOAP, linewidth=1.3, label="SOAP–Muon")
    loss_axis.set_title("(a) Validation loss, mean of five seeds", loc="left")
    loss_axis.set_xlabel("Training tokens (millions)")
    loss_axis.set_ylabel("Validation loss")
    loss_axis.set_xlim(0, 200)
    loss_axis.legend(frameon=False, loc="upper right")
    add_grid(loss_axis)

    for values in per_seed:
        difference_axis.plot(millions, values, color=SEED_LINE, linewidth=0.6)
    difference_axis.fill_between(
        millions, difference, 0, where=[value <= 0 for value in difference],
        interpolate=True, color=SOAP, alpha=0.28, linewidth=0,
    )
    difference_axis.fill_between(
        millions, difference, 0, where=[value >= 0 for value in difference],
        interpolate=True, color=MUON, alpha=0.28, linewidth=0,
    )
    difference_axis.axhline(0, color=SECONDARY_INK, linewidth=0.7)
    difference_axis.plot(millions, difference, color=INK, linewidth=1.3)
    difference_axis.set_title("(b) Loss difference, SOAP–Muon minus Moonlight Muon", loc="left")
    difference_axis.set_xlabel("Training tokens (millions)")
    difference_axis.set_ylabel("Loss difference")
    difference_axis.set_xlim(0, 200)
    difference_axis.set_ylim(-0.06, 0.02)
    add_grid(difference_axis)

    def signed(value: float) -> str:
        return f"{value:+.4f}".replace("-", "\N{MINUS SIGN}")

    difference_axis.text(27, -0.0575, "SOAP–Muon ahead", color=INK, fontsize=7.5)
    difference_axis.text(198, 0.0148, "Moonlight Muon ahead", color=INK, fontsize=7.5, ha="right")
    difference_axis.text(
        198, -0.0575,
        f"Shaded area ÷ 200M tokens:\n{signed(ahead)} {signed(behind)[0]} {signed(behind)[1:]}"
        f" = {signed(ahead + behind)}",
        color=SECONDARY_INK, fontsize=7, ha="right", va="bottom", linespacing=1.35,
    )

    figure.savefig(output, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    plt.close(figure)


def describe_scope(seeds: tuple[int, ...], learning_rate: float) -> str:
    """Return a caption fragment naming the seeds and the shared learning rate."""
    ordered = sorted(seeds)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        seed_text = f"seeds {ordered[0]}--{ordered[-1]}"
    else:
        seed_text = "seeds " + ", ".join(str(seed) for seed in ordered)
    mantissa, exponent = f"{learning_rate:.1e}".split("e")
    return f"{seed_text}, learning rate ${mantissa}\\times10^{{{int(exponent)}}}$"


def render_results_table(rows: list[dict[str, float | int]], output: Path, scope: str) -> None:
    lines = [
        "% Generated by scripts/generate_confirmation_figures.py; do not edit manually.",
        "\\begin{table*}[t]",
        f"\\caption{{Results for each seed ({scope}): area under the validation-loss curve (the main measure) and final validation loss. Muon denotes Moonlight Muon. Differences are SOAP--Muon minus Muon; lower is better, so a negative difference favors SOAP--Muon.}}",
        "\\label{tab:confirmation-results}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{c r r r r r r}",
        "\\hline",
        "Seed & Muon area & SOAP--Muon area & Difference & Muon final loss & SOAP--Muon final loss & Difference \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['seed']} & {row['baseline_area']:.4f} & {row['soap_area']:.4f} & "
            f"{row['area_delta']:+.4f} & {row['baseline_final']:.4f} & "
            f"{row['soap_final']:.4f} & {row['final_delta']:+.4f} \\\\")
    averages = {key: mean([float(row[key]) for row in rows]) for key in (
        "baseline_area", "soap_area", "area_delta", "baseline_final", "soap_final", "final_delta")}
    lines.extend([
        "\\hline",
        f"Mean & {averages['baseline_area']:.4f} & {averages['soap_area']:.4f} & "
        f"{averages['area_delta']:+.4f} & {averages['baseline_final']:.4f} & "
        f"{averages['soap_final']:.4f} & {averages['final_delta']:+.4f} \\\\",
    ])
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    output.write_text("\n".join(lines))


def render_systems_table(rows: list[dict[str, float | int]], output: Path, scope: str) -> None:
    lines = [
        "% Generated by scripts/generate_confirmation_figures.py; do not edit manually.",
        "\\begin{table*}[t]",
        f"\\caption{{Training speed and peak GPU memory (GiB) for each seed ({scope}). Higher speed and lower memory are better; the mean change in speed compares the two mean speeds.}}",
        "\\label{tab:confirmation-systems}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{c r r r r r r}",
        "\\hline",
        "Seed & Muon tokens/s & SOAP--Muon tokens/s & Change & Muon GiB & SOAP--Muon GiB & Change \\\\",
        "\\hline",
    ]
    for row in rows:
        baseline_throughput = float(row["baseline_throughput"])
        soap_throughput = float(row["soap_throughput"])
        baseline_memory = float(row["baseline_memory_gib"])
        soap_memory = float(row["soap_memory_gib"])
        throughput_delta = (soap_throughput / baseline_throughput) - 1
        memory_delta = soap_memory - baseline_memory
        throughput_delta_text = f"{throughput_delta:+.2%}".replace("%", r"\%")
        lines.append(
            f"{row['seed']} & {baseline_throughput:,.0f} & {soap_throughput:,.0f} & "
            f"{throughput_delta_text} & {baseline_memory:.3f} & {soap_memory:.3f} & "
            f"{memory_delta:+.3f} \\\\")
    baseline_throughput = mean([float(row["baseline_throughput"]) for row in rows])
    soap_throughput = mean([float(row["soap_throughput"]) for row in rows])
    baseline_memory = mean([float(row["baseline_memory_gib"]) for row in rows])
    soap_memory = mean([float(row["soap_memory_gib"]) for row in rows])
    throughput_delta_text = f"{soap_throughput / baseline_throughput - 1:+.2%}".replace("%", r"\%")
    lines.extend([
        "\\hline",
        f"Mean & {baseline_throughput:,.0f} & {soap_throughput:,.0f} & "
        f"{throughput_delta_text} & {baseline_memory:.3f} & {soap_memory:.3f} & "
        f"{soap_memory - baseline_memory:+.3f} \\\\",
    ])
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    output.write_text("\n".join(lines))


def render_wall_clock_table(rows: list[dict[str, float | int]], output: Path, scope: str) -> None:
    lines = [
        "% Generated by scripts/generate_confirmation_figures.py; do not edit manually.",
        "\\begin{table*}[t]",
        f"\\caption{{Time in seconds, including loss measurements, until the measured validation loss first reaches 4.25 and 4.20 ({scope}). Lower is better.}}",
        "\\label{tab:confirmation-wall-clock}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{c r r r r r r}",
        "\\hline",
        "Seed & Muon, 4.25 & SOAP--Muon, 4.25 & Difference & Muon, 4.20 & SOAP--Muon, 4.20 & Difference \\\\",
        "\\hline",
    ]
    for row in rows:
        baseline_425 = float(row["baseline_425_seconds"])
        soap_425 = float(row["soap_425_seconds"])
        baseline_420 = float(row["baseline_420_seconds"])
        soap_420 = float(row["soap_420_seconds"])
        lines.append(
            f"{row['seed']} & {baseline_425:.1f} & {soap_425:.1f} & {soap_425 - baseline_425:+.1f} & "
            f"{baseline_420:.1f} & {soap_420:.1f} & {soap_420 - baseline_420:+.1f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    output.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/confirmation_metrics"))
    parser.add_argument("--figures-dir", type=Path, default=Path("report/Figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("report/Tables"))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Five paired seed identifiers (default: 201 202 203 204 205).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        raise ValueError("--seeds must contain exactly five distinct seeds.")
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        analyse_pair(
            seed,
            load_metrics(args.metrics_dir, seed, "baseline"),
            load_metrics(args.metrics_dir, seed, "soap"),
        )
        for seed in args.seeds
    ]
    learning_rate = float(
        load_metrics(args.metrics_dir, args.seeds[0], "baseline")["optimizer"]["moonlight_muon_learning_rate"]
    )
    scope = describe_scope(tuple(args.seeds), learning_rate)
    render_training_figure(
        args.metrics_dir, args.figures_dir / "fig_01_validation_nll_difference.pdf", tuple(args.seeds)
    )
    render_results_table(rows, args.tables_dir / "tab_03_confirmation_results.tex", scope)
    render_systems_table(rows, args.tables_dir / "tab_04_confirmation_systems.tex", scope)
    render_wall_clock_table(rows, args.tables_dir / "tab_05_confirmation_wall_clock.tex", scope)


if __name__ == "__main__":
    main()
