# Committed result artifacts

This directory contains lightweight, immutable JSON metrics needed to
regenerate the report's figures and tables. Checkpoints and token shards are
deliberately excluded because of their size; the data-preparation command and
all configurations are versioned separately.

- `confirmation_metrics/` contains the ten held-out, paired confirmation
  metrics used by the main report results.
- `screening_metrics/` contains the complete seed-101 route/LR screen,
  including the Moonlight Muon extension and the matched shared-rate
  Moonlight-Muon/SOAP--Muon (both MLP projections) extension.
- `exploratory_metrics/` contains earlier single-seed pilots retained only for
  the exploratory appendix.

Regenerate the report evidence from a fresh clone with:

```bash
uv sync --all-groups
python3 scripts/analyze_confirmation.py --metrics-dir results/confirmation_metrics
python3 scripts/generate_confirmation_figures.py --metrics-dir results/confirmation_metrics
python3 scripts/generate_exploratory_tables.py --results-dir results
uv run python scripts/verify_confirmation_artifacts.py
```
