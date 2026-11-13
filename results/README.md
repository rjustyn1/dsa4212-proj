# Committed result artifacts

This directory contains lightweight, immutable JSON metrics needed to
regenerate the report's figures and tables. Checkpoints and token shards are
deliberately excluded because of their size; the data-preparation command and
all configurations are versioned separately.

- `confirmation_metrics/` contains the ten held-out, paired confirmation
  metrics for the initially selected configuration.
- `shared_rate_confirmation_metrics/` contains the ten fresh held-out paired
  metrics at the shared, screen-selected $4.8\times10^{-3}$ rate, with seeds
  301--305 and a generated summary. Every published run is labelled NVIDIA A10;
  the superseded NVIDIA A10G seed-305 treatment measurement is retained under
  the metrics directory's superseded/ folder for auditability.
- `screening_metrics/` contains the complete seed-101 route/LR screen,
  including the Moonlight Muon extension and the matched shared-rate
  Moonlight-Muon/SOAP--Muon (both MLP projections) extension.
- `exploratory_metrics/` contains earlier single-seed pilots retained only for
  the exploratory appendix.
- `fineweb_edu_75m_gpt2_manifest.json` is the manifest that
  `scripts/prepare_data.py` wrote for the study's token shards: the dataset
  revision, token counts, and SHA-256 hash of each shard.

Regenerate the report evidence from a fresh clone with:

```bash
uv sync --all-groups
python3 scripts/analyze_confirmation.py --metrics-dir results/confirmation_metrics
python3 scripts/analyze_confirmation.py \
  --metrics-dir results/shared_rate_confirmation_metrics \
  --seeds 301 302 303 304 305 \
  --report results/shared_rate_confirmation_metrics/summary.md
uv run python scripts/generate_confirmation_figures.py \
  --metrics-dir results/shared_rate_confirmation_metrics \
  --seeds 301 302 303 304 305
python3 scripts/generate_exploratory_tables.py --results-dir results
python3 scripts/generate_sensitivity_table.py --results-dir results
uv run python scripts/verify_confirmation_artifacts.py
uv run python scripts/verify_shared_rate_confirmation_artifacts.py
```
