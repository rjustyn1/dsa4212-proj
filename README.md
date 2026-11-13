# DSA4212: Selective One-Sided SOAP--Muon

This repository evaluates whether selective one-sided SOAP preconditioning
improves a Moonlight Muon update in a compact decoder-only Transformer. The
main evidence is a five-seed, paired 200M-token confirmation study. Its bounded
conclusion is an early token-indexed trajectory difference with a systems cost;
it is not a final-quality or wall-clock improvement claim. The report and its
claim-to-artifact map are in [`report/`](report/) and
[`research/analysis/figure_table_map.md`](research/analysis/figure_table_map.md).

## Reproduce the committed report evidence

The committed JSON metrics under [`results/`](results/) are sufficient to
regenerate every reported figure and generated table; no Modal account, GPU, or
token shard is needed for this step.

```bash
uv sync --all-groups
uv run pytest
python3 scripts/analyze_confirmation.py \
  --metrics-dir results/shared_rate_confirmation_metrics \
  --seeds 301 302 303 304 305 \
  --report results/shared_rate_confirmation_metrics/summary.md
python3 scripts/generate_confirmation_figures.py \
  --metrics-dir results/shared_rate_confirmation_metrics \
  --seeds 301 302 303 304 305 \
  --figures-dir report/Figures \
  --tables-dir report/Tables
python3 scripts/generate_exploratory_tables.py --results-dir results
uv run python scripts/verify_confirmation_artifacts.py
uv run python scripts/verify_shared_rate_confirmation_artifacts.py
```

Compile the IEEE report with a TeX engine such as Tectonic:

```bash
cd report
tectonic main.tex
```

## Project code and tests

The implementation is a normal Python package under `src/soap_muon/`:

- `data/` loads deterministic token shards and prepares immutable FineWeb-Edu
  shards.
- `models/` defines the decoder-only GPT used by the experiments.
- `optim/`, `routing.py`, and `training/` define the optimizer routes, training
  loop, evaluation, metrics, and checkpointing.
- `configs/` records model, data, optimizer, and experiment settings.
- `tests/` contains deterministic optimizer, routing, data, model, training,
  checkpoint, and evaluation tests.

Run the full local test suite with `uv run pytest`.

## Frozen pretraining data

Token shards are intentionally not committed because they are large. A fresh
training reproduction must create them once, retain the resulting manifest,
and use the same shards for every optimizer:

```bash
uv run python scripts/prepare_data.py \
  --config configs/data/fineweb_edu_75m.yaml \
  --output-dir data/processed/fineweb_edu_75m_gpt2 \
  --seed 2026 \
  --selection-modulus 4
```

The command uses seeded document-ID selection, writes train/validation/test
shards plus SHA-256 metadata, and supports `--resume`. Do not regenerate the
directory after an experiment begins.

## Modal GPU entry points

`modal_app.py` contains bounded remote training functions. It needs a user
owned Modal account and the prepared local token shards. Authenticate only on
your own machine:

```bash
uv run modal setup
```

Do not launch a new GPU run merely to reproduce the report: the reported
metrics are already versioned. Every new experiment needs a committed protocol,
a dedicated experiment branch, an explicit spend decision, and saved metrics.

## Repository layout

- `authors.md` — required submission metadata; replace its placeholders before
  submission.
- `results/` — committed lightweight confirmation, screening, and exploratory
  metrics.
- `report/` — IEEE source, generated assets, and review checklist.
- `research/` — analysis plan, experiment matrix, and claim-to-artifact map.
- `scripts/` — data preparation, analysis, validation, and report generation.
