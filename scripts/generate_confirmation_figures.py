"""Generate report figures and a results table from saved confirmation metrics.

The source metrics are the only numeric input.  This script deliberately uses
SVG plus rsvg-convert so that report figures can be reproduced without adding
an undeclared plotting dependency to the project.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from analyze_confirmation import DEFAULT_SEEDS, analyse_pair, load_metrics


WIDTH = 720
HEIGHT = 440
LEFT = 82
RIGHT = 26
TOP = 34
BOTTOM = 68
BASELINE = "#D55E00"
SOAP = "#0072B2"
GRID = "#D9D9D9"
TEXT = "#202020"


def svg_start(title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>"
        ".label { font: 15px Arial, Helvetica, sans-serif; fill: #202020; }"
        ".tick { font: 13px Arial, Helvetica, sans-serif; fill: #202020; }"
        ".legend { font: 13px Arial, Helvetica, sans-serif; fill: #202020; }"
        ".title { font: 600 16px Arial, Helvetica, sans-serif; fill: #202020; }"
        "</style>",
        f"<title>{escape(title)}</title>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
    ]


def finish_svg(lines: list[str], path: Path) -> None:
    path.write_text("\n".join([*lines, "</svg>", ""]))


def plot_area() -> tuple[float, float, float, float]:
    return LEFT, TOP, WIDTH - RIGHT, HEIGHT - BOTTOM


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def render_curve_figure(metrics_dir: Path, output: Path) -> None:
    baseline = [load_metrics(metrics_dir, seed, "baseline") for seed in DEFAULT_SEEDS]
    soap = [load_metrics(metrics_dir, seed, "soap") for seed in DEFAULT_SEEDS]
    curve_tokens = [point["tokens_processed"] for point in baseline[0]["training"]["curve"]]
    assert all(
        [point["tokens_processed"] for point in run["training"]["curve"]] == curve_tokens
        for run in [*baseline, *soap]
    )
    baseline_nll = [[point["mean_nll"] for point in run["training"]["curve"]] for run in baseline]
    soap_nll = [[point["mean_nll"] for point in run["training"]["curve"]] for run in soap]
    baseline_mean = [mean([run[index] for run in baseline_nll]) for index in range(len(curve_tokens))]
    soap_mean = [mean([run[index] for run in soap_nll]) for index in range(len(curve_tokens))]
    values = [value for run in [*baseline_nll, *soap_nll] for value in run]
    y_min = 3.95
    y_max = max(values) + 0.15
    x0, y0, x1, y1 = plot_area()

    def x(value: float) -> float:
        return x0 + (value / 200_000_000) * (x1 - x0)

    def y(value: float) -> float:
        return y1 - ((value - y_min) / (y_max - y_min)) * (y1 - y0)

    lines = svg_start("Validation NLL over training tokens")
    lines.append('<text class="title" x="82" y="22">Validation NLL across held-out confirmation seeds</text>')
    for tick in range(4, 8):
        value = float(tick)
        yy = y(value)
        lines.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>')
        lines.append(f'<text class="tick" x="{x0 - 12:.1f}" y="{yy + 4:.1f}" text-anchor="end">{value:.1f}</text>')
    for tick in (0, 50, 100, 150, 200):
        xx = x(tick * 1_000_000)
        lines.append(f'<line x1="{xx:.1f}" y1="{y0:.1f}" x2="{xx:.1f}" y2="{y1:.1f}" stroke="{GRID}"/>')
        lines.append(f'<text class="tick" x="{xx:.1f}" y="{y1 + 23:.1f}" text-anchor="middle">{tick}</text>')
    lines.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{TEXT}"/>')
    lines.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" stroke="{TEXT}"/>')
    lines.append(f'<text class="label" x="{(x0 + x1) / 2:.1f}" y="{HEIGHT - 18}" text-anchor="middle">Training tokens (millions)</text>')
    lines.append(f'<text class="label" transform="translate(22 {(y0 + y1) / 2:.1f}) rotate(-90)" text-anchor="middle">Validation NLL (lower is better)</text>')

    for runs, color in ((baseline_nll, BASELINE), (soap_nll, SOAP)):
        upper = [max(run[index] for run in runs) for index in range(len(curve_tokens))]
        lower = [min(run[index] for run in runs) for index in range(len(curve_tokens))]
        upper_points = " ".join(f"{x(token):.1f},{y(value):.1f}" for token, value in zip(curve_tokens, upper))
        lower_points = " ".join(f"{x(token):.1f},{y(value):.1f}" for token, value in reversed(list(zip(curve_tokens, lower))))
        lines.append(f'<polygon points="{upper_points} {lower_points}" fill="{color}" opacity="0.16"/>')
    for values_for_method, color in ((baseline_mean, BASELINE), (soap_mean, SOAP)):
        points = " ".join(f"{x(token):.1f},{y(value):.1f}" for token, value in zip(curve_tokens, values_for_method))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    legend_x = x0 + 12
    legend_y = y0 + 18
    for index, (label, color) in enumerate((("Moonlight Muon mean", BASELINE), ("SOAP-Muon mean", SOAP))):
        yy = legend_y + index * 20
        lines.append(f'<line x1="{legend_x:.1f}" y1="{yy:.1f}" x2="{legend_x + 24:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text class="legend" x="{legend_x + 31:.1f}" y="{yy + 4:.1f}">{label}</text>')
    lines.append(f'<text class="legend" x="{legend_x:.1f}" y="{legend_y + 53:.1f}">Shaded band: min--max over five seeds</text>')
    finish_svg(lines, output)


def render_difference_figure(metrics_dir: Path, output: Path) -> None:
    """Plot the paired SOAP-minus-baseline NLL difference over training."""
    baseline = [load_metrics(metrics_dir, seed, "baseline") for seed in DEFAULT_SEEDS]
    soap = [load_metrics(metrics_dir, seed, "soap") for seed in DEFAULT_SEEDS]
    tokens = [point["tokens_processed"] for point in baseline[0]["training"]["curve"]]
    differences = [
        [soap_run["training"]["curve"][index]["mean_nll"] - baseline_run["training"]["curve"][index]["mean_nll"]
         for baseline_run, soap_run in zip(baseline, soap)]
        for index in range(len(tokens))
    ]
    means = [mean(values) for values in differences]
    extrema = max(abs(value) for values in differences for value in values)
    y_limit = max(0.01, extrema * 1.15)
    x0, y0, x1, y1 = plot_area()

    def x(value: float) -> float:
        return x0 + (value / 200_000_000) * (x1 - x0)

    def y(value: float) -> float:
        return y1 - ((value + y_limit) / (2 * y_limit)) * (y1 - y0)

    lines = svg_start("Paired validation-NLL difference over training")
    lines.append('<text class="title" x="82" y="22">Paired validation-NLL difference across training</text>')
    for tick in (-y_limit, -y_limit / 2, 0.0, y_limit / 2, y_limit):
        yy = y(tick)
        stroke = TEXT if tick == 0 else GRID
        width = "1.5" if tick == 0 else "1"
        lines.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="{width}"/>')
        lines.append(f'<text class="tick" x="{x0 - 12:.1f}" y="{yy + 4:.1f}" text-anchor="end">{tick:+.3f}</text>')
    for tick in (0, 50, 100, 150, 200):
        xx = x(tick * 1_000_000)
        lines.append(f'<line x1="{xx:.1f}" y1="{y0:.1f}" x2="{xx:.1f}" y2="{y1:.1f}" stroke="{GRID}"/>')
        lines.append(f'<text class="tick" x="{xx:.1f}" y="{y1 + 23:.1f}" text-anchor="middle">{tick}</text>')
    upper = [max(values) for values in differences]
    lower = [min(values) for values in differences]
    upper_points = " ".join(f"{x(token):.1f},{y(value):.1f}" for token, value in zip(tokens, upper))
    lower_points = " ".join(f"{x(token):.1f},{y(value):.1f}" for token, value in reversed(list(zip(tokens, lower))))
    lines.append(f'<polygon points="{upper_points} {lower_points}" fill="{SOAP}" opacity="0.16"/>')
    points = " ".join(f"{x(token):.1f},{y(value):.1f}" for token, value in zip(tokens, means))
    lines.append(f'<polyline points="{points}" fill="none" stroke="{SOAP}" stroke-width="3"/>')
    lines.append(f'<text class="label" x="{(x0 + x1) / 2:.1f}" y="{HEIGHT - 18}" text-anchor="middle">Training tokens (millions)</text>')
    lines.append(f'<text class="label" transform="translate(22 {(y0 + y1) / 2:.1f}) rotate(-90)" text-anchor="middle">SOAP minus Moonlight Muon validation NLL (negative favours SOAP)</text>')
    lines.append(f'<text class="legend" x="{x0 + 12:.1f}" y="{y0 + 18:.1f}">Line: paired-seed mean; band: min--max across five pairs</text>')
    finish_svg(lines, output)


def render_results_table(rows: list[dict[str, float | int]], output: Path) -> None:
    lines = [
        "% Generated by scripts/generate_confirmation_figures.py; do not edit manually.",
        "\\begin{table*}[t]",
        "\\caption{Held-out paired confirmation results. Lower is better for both NLL columns and the trajectory area.}",
        "\\label{tab:confirmation-results}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{c r r r r r r}",
        "\\hline",
        "Seed & Muon area & SOAP area & $\\Delta$ area & Muon final NLL & SOAP final NLL & $\\Delta$ final NLL \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['seed']} & {row['baseline_area']:.6f} & {row['soap_area']:.6f} & "
            f"{row['area_delta']:+.6f} & {row['baseline_final']:.6f} & "
            f"{row['soap_final']:.6f} & {row['final_delta']:+.6f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    output.write_text("\n".join(lines))


def render_systems_table(rows: list[dict[str, float | int]], output: Path) -> None:
    lines = [
        "% Generated by scripts/generate_confirmation_figures.py; do not edit manually.",
        "\\begin{table*}[t]",
        "\\caption{Per-seed systems diagnostics for the held-out confirmation. Higher throughput and lower peak memory are better. All runs completed with finite metrics.}",
        "\\label{tab:confirmation-systems}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{c r r r r r r}",
        "\\hline",
        "Seed & Muon tokens/s & SOAP tokens/s & $\\Delta$ throughput & Muon GiB & SOAP GiB & $\\Delta$ GiB \\\\",
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
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    output.write_text("\n".join(lines))


def render_wall_clock_table(rows: list[dict[str, float | int]], output: Path) -> None:
    lines = [
        "% Generated by scripts/generate_confirmation_figures.py; do not edit manually.",
        "\\begin{table*}[t]",
        "\\caption{Elapsed wall-clock seconds at the first logged threshold crossing, including evaluation. Lower is better.}",
        "\\label{tab:confirmation-wall-clock}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{c r r r r r r}",
        "\\hline",
        "Seed & Muon $\\leq4.25$ & SOAP $\\leq4.25$ & $\\Delta$s & Muon $\\leq4.20$ & SOAP $\\leq4.20$ & $\\Delta$s \\\\",
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


def convert_to_pdf(svg_path: Path) -> None:
    subprocess.run(
        ["rsvg-convert", "--format=pdf", "--output", str(svg_path.with_suffix(".pdf")), str(svg_path)],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("runs/confirmation_metrics"))
    parser.add_argument("--figures-dir", type=Path, default=Path("report/Figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("report/Tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        analyse_pair(
            seed,
            load_metrics(args.metrics_dir, seed, "baseline"),
            load_metrics(args.metrics_dir, seed, "soap"),
        )
        for seed in DEFAULT_SEEDS
    ]
    curve = args.figures_dir / "fig_01_validation_nll_vs_tokens.svg"
    difference = args.figures_dir / "fig_02_validation_nll_difference.svg"
    render_curve_figure(args.metrics_dir, curve)
    render_difference_figure(args.metrics_dir, difference)
    render_results_table(rows, args.tables_dir / "tab_03_confirmation_results.tex")
    render_systems_table(rows, args.tables_dir / "tab_04_confirmation_systems.tex")
    render_wall_clock_table(rows, args.tables_dir / "tab_05_confirmation_wall_clock.tex")
    convert_to_pdf(curve)
    convert_to_pdf(difference)


if __name__ == "__main__":
    main()
