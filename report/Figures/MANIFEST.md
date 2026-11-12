# Generated asset manifest

All assets below are generated from the saved, held-out confirmation metrics.
Regenerate them from the repository root with:

```bash
python3 scripts/generate_confirmation_figures.py
```

| Asset | Inputs | Purpose |
|---|---|---|
| `fig_01_validation_nll_vs_tokens.{svg,pdf}` | `runs/confirmation_metrics/seed{201..205}-{baseline,soap}.json` | Mean validation-NLL curves with min--max seed bands. |
| `fig_03_primary_endpoint_pairs.{svg,pdf}` | Same five paired metric files | Per-seed paired primary endpoint contrasts. |
| `../Tables/tab_03_confirmation_results.tex` | Same five paired metric files | Per-seed primary and final-NLL table. |
| `../Tables/tab_04_confirmation_systems.tex` | Same five paired metric files | Per-seed throughput and peak-memory table. |
