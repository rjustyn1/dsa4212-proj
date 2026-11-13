# Generated asset manifest

All assets below are generated from the saved, held-out confirmation metrics.
Regenerate them from the repository root with:

```bash
python3 scripts/generate_confirmation_figures.py
```

| Asset | Inputs | Purpose |
|---|---|---|
| `fig_01_validation_nll_difference.{svg,pdf}` | `results/confirmation_metrics/seed{201..205}-{baseline,soap}.json` | Mean paired SOAP-minus-Muon NLL difference with min--max band. |
| `../Tables/tab_03_confirmation_results.tex` | Same five paired metric files | Per-seed primary and final-NLL table. |
| `../Tables/tab_04_confirmation_systems.tex` | Same five paired metric files | Per-seed throughput and peak-memory table. |
| `../Tables/tab_05_confirmation_wall_clock.tex` | Same five paired metric files | Per-seed wall-clock threshold-crossing table. |
