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


def render_pair_figure(rows: list[dict[str, float | int]], output: Path) -> None:
    values = [float(row[key]) for row in rows for key in ("baseline_area", "soap_area")]
    x_min = min(values) - 0.003
    x_max = max(values) + 0.003
    x0, y0, x1, y1 = plot_area()
    y0 += 36
    x1 -= 20

    def x(value: float) -> float:
        return x0 + ((value - x_min) / (x_max - x_min)) * (x1 - x0)

    lines = svg_start("Paired primary endpoint")
    lines.append('<text class="title" x="82" y="22">Paired primary endpoint: normalised validation-NLL area</text>')
    ticks = (4.148, 4.152, 4.156, 4.160, 4.164)
    for tick in ticks:
        xx = x(tick)
        lines.append(f'<line x1="{xx:.1f}" y1="{y0:.1f}" x2="{xx:.1f}" y2="{y1:.1f}" stroke="{GRID}"/>')
        lines.append(f'<text class="tick" x="{xx:.1f}" y="{y1 + 23:.1f}" text-anchor="middle">{tick:.3f}</text>')
    for index, row in enumerate(rows):
        yy = y0 + 40 + index * ((y1 - y0 - 80) / (len(rows) - 1))
        baseline = float(row["baseline_area"])
        soap = float(row["soap_area"])
        lines.append(f'<line x1="{x(baseline):.1f}" y1="{yy:.1f}" x2="{x(soap):.1f}" y2="{yy:.1f}" stroke="#777777" stroke-width="2"/>')
        lines.append(f'<circle cx="{x(baseline):.1f}" cy="{yy:.1f}" r="5" fill="{BASELINE}"/>')
        lines.append(f'<circle cx="{x(soap):.1f}" cy="{yy:.1f}" r="5" fill="{SOAP}"/>')
        lines.append(f'<text class="label" x="{x0 - 12:.1f}" y="{yy + 5:.1f}" text-anchor="end">Seed {row["seed"]}</text>')
    lines.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{TEXT}"/>')
    lines.append(f'<text class="label" x="{(x0 + x1) / 2:.1f}" y="{HEIGHT - 18}" text-anchor="middle">Normalised validation-NLL area (lower is better)</text>')
    legend_x = x0 + 12
    legend_y = 45
    for index, (label, color) in enumerate((("Moonlight Muon", BASELINE), ("SOAP-Muon", SOAP))):
        xx = legend_x + index * 180
        lines.append(f'<circle cx="{xx:.1f}" cy="{legend_y:.1f}" r="5" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{xx + 10:.1f}" y="{legend_y + 4:.1f}">{label}</text>')
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
    pairs = args.figures_dir / "fig_03_primary_endpoint_pairs.svg"
    render_curve_figure(args.metrics_dir, curve)
    render_pair_figure(rows, pairs)
    render_results_table(rows, args.tables_dir / "tab_03_confirmation_results.tex")
    render_systems_table(rows, args.tables_dir / "tab_04_confirmation_systems.tex")
    convert_to_pdf(curve)
    convert_to_pdf(pairs)


if __name__ == "__main__":
    main()
