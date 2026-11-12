# Report claim and asset map

This map makes the generated evidence behind the IEEE report explicit.  It is
not a second protocol: the experiment configuration remains authoritative.
Paths below are relative to the repository root.

## Reproduce the numerical summary

The held-out primary and secondary summaries are regenerated from the ten
saved JSON metric files, not copied into the paper by hand.

```bash
python3 scripts/analyze_confirmation.py \
  --metrics-dir runs/confirmation_metrics \
  --report runs/confirmation_metrics/summary.md
python3 scripts/generate_confirmation_figures.py \
  --metrics-dir runs/confirmation_metrics \
  --figures-dir report/Figures \
  --tables-dir report/Tables
```

The metric inputs are
`runs/confirmation_metrics/seed{201,202,203,204,205}-{baseline,soap}.json`.
`scripts/analyze_confirmation.py` fixes the bootstrap seed (`4212`) and uses
100,000 resamples by default.  The raw `runs/` directory is intentionally a
generated artifact; retain these files when transferring or archiving the
report package.

Units and labels are fixed by the generator: Figure~1 uses training tokens
(millions) on the horizontal axis and validation NLL on the vertical axis;
Figure~2 uses normalised validation-NLL area and names both method endpoints.
Table~III reports NLL/area to six decimal places.  Lower is better for all NLL
and area quantities; throughput is the sole higher-is-better diagnostic.

## Report asset provenance

| Paper item | Evidence inputs | Generator or derivation | Report use |
|---|---|---|---|
| Table I, `tab_01_experimental_protocol.tex` | `configs/experiments/confirm_moonlight_soap_200m.yaml`; `configs/data/fineweb_edu_75m.yaml`; `configs/model/gpt_35m.yaml`; `data/processed/fineweb_edu_75m_gpt2/manifest.json`; all ten confirmation metrics | Protocol values are frozen from configurations.  Completed tokens and updates are checked against the metrics. | Section IV specifies pairing, data, model, budget, and evaluation cadence. |
| Table II, `tab_02_optimizer_definitions.tex` | `configs/optimizer/moonlight_muon_reference.yaml`; `configs/optimizer/moonlight_soap_muon.yaml`; `configs/experiments/confirm_moonlight_soap_200m.yaml` | Frozen routing and optimizer settings transcribed from versioned configuration. | Section III defines the compared methods. |
| Figure 1, `Figures/fig_01_validation_nll_vs_tokens.{svg,pdf}` | All ten confirmation metrics | `scripts/generate_confirmation_figures.py` | Section V-A shows the mean curve and the min--max band across paired seeds. |
| Figure 2, `Figures/fig_03_primary_endpoint_pairs.{svg,pdf}` | All ten confirmation metrics | `scripts/generate_confirmation_figures.py` | Section V-A shows every paired primary-endpoint contrast.  The report labels this as Fig.~2 because there is no wall-clock figure. |
| Table III, `tab_03_confirmation_results.tex` | All ten confirmation metrics | `scripts/generate_confirmation_figures.py`, which imports `analyse_pair` from `scripts/analyze_confirmation.py` | Section V-B reports every seed's primary area and final NLL. |
| Mean contrasts, bootstrap intervals, threshold crossings, throughput, and memory in Sections V--VII | All ten confirmation metrics | `scripts/analyze_confirmation.py` -> `runs/confirmation_metrics/summary.md` | Text is constrained to the generated summary; lower is better for NLL/area and higher is better for throughput. |

There is deliberately no wall-clock figure: the stored results contain curve
wall-clock measurements, but the final report does not claim an end-to-end
time-to-threshold comparison with evaluation included.  Training throughput is
therefore reported only as a systems diagnostic.

## Scope and integrity checks

- The analysis compares paired seeds 201--205 only.  Screening seed 101 is
  selection evidence, not confirmation evidence.
- The primary endpoint is the sampled validation-NLL trapezoidal area divided
  by the nominal 200M-token budget.  The script rejects unequal curve grids.
- The data manifest pins FineWeb-Edu revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, selection-manifest SHA-256
  `f8c277f1430f4503a3b846a99dc4571d3803b2706e7670ef9712c6d60a590228`, and
  SHA-256 hashes for all three token shards.
- All ten metrics report device, seed, batch regime, optimizer settings,
  sampled curves, endpoint validation, throughput, peak memory, and finite
  completion.  They do **not** embed a Git source commit or Modal artifact ID.
  Those two identifiers must be recovered from the run/archive metadata before
  a submission that requires run-level code provenance.
- `configs/experiments/confirm_moonlight_soap_200m.yaml` is the compact frozen
  confirmation record used by the report.  It does not presently conform to
  the component-path schema expected by `scripts/validate_config.py`; do not
  represent it as a directly runnable current-code configuration until that
  compatibility issue is resolved.

