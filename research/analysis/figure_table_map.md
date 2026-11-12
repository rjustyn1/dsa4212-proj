# Report claim and asset map

This map makes the generated evidence behind the IEEE report explicit.  It is
not a second protocol: the experiment configuration remains authoritative.
Paths below are relative to the repository root.

## Version record

The generated assets are derived from the archived metric inputs listed below.
Their generator, input paths, and revision history are retained in this
repository so a report revision can be reproduced without transcribing values
by hand.

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
uv run python scripts/verify_confirmation_artifacts.py
```

The metric inputs are
`runs/confirmation_metrics/seed{201,202,203,204,205}-{baseline,soap}.json`.
`scripts/analyze_confirmation.py` fixes the bootstrap seed (`4212`) and uses
100,000 resamples by default.  The raw `runs/` directory is intentionally a
generated artifact; retain these files when transferring or archiving the
report package.

## Modal archive

The ten metric inputs are archived in Modal volume
`dsa4212-optimizer-results`, under the active `rjustyn1` profile.  They were
downloaded read-only and SHA-256-checked against the local inputs on
2026-11-12.  For any row below, retrieve the saved input with:

```bash
uv run modal volume get dsa4212-optimizer-results \
  /<remote-directory>/metrics.json \
  runs/confirmation_metrics/<local-file>
```

| Local file | Remote directory | SHA-256 |
|---|---|---|
| `seed201-baseline.json` | `confirmation-moonlight-baseline-a10-seed201-200m` | `681863de2af44ea928dbe6a457a114cc0331f02332e0fb412a5a8011ce29e073` |
| `seed201-soap.json` | `confirmation-moonlight-soap-a10-seed201-200m` | `2abed36686c6d57ee59a090cd7535647eeec60503b177d43c995ddb314d9a670` |
| `seed202-baseline.json` | `confirmation-moonlight-baseline-a10-seed202-200m` | `828b99baca6e282d8f6788aa58ea6c4b43b78333007468366c17880ae786dfe1` |
| `seed202-soap.json` | `confirmation-moonlight-soap-a10-seed202-200m` | `d30ad0c36d8985fe22063dbf7f45c1bb91067d1a0e39a5588f7568339fe06f4e` |
| `seed203-baseline.json` | `confirmation-moonlight-baseline-a10-seed203-200m` | `d8405b165a7cb160ddc76de0deccfb6b48ce59ccfd73a7497097f9fecdb181a5` |
| `seed203-soap.json` | `confirmation-moonlight-soap-a10-seed203-200m` | `a771567a5d98256c1220cbaf1ffabb63a0464e59588c8a44cae62c1a99af31e2` |
| `seed204-baseline.json` | `confirmation-moonlight-baseline-a10-seed204-200m` | `b203493afbb6847738017c1d3169f235ab99d6c3b968db131e0dac60c70000b5` |
| `seed204-soap.json` | `confirmation-moonlight-soap-a10-seed204-200m` | `809b26d3cae7ad93d9ea1e327bb1e8b2f487ae39e4067d396713082b93ce6b17` |
| `seed205-baseline.json` | `confirmation-moonlight-baseline-a10-seed205-200m` | `aefe36a17c4703a701bc736cae22504066d27f0be0b104c8b5facfa2c249858a` |
| `seed205-soap.json` | `confirmation-moonlight-soap-a10-seed205-200m` | `8fd8cae4ff83055652c13ede60d498bc912706e06902bb69702168498bc33d9b` |

Units and labels are fixed by the generator: Figures~1--2 use training tokens
(millions) on the horizontal axis; Figure~1 plots validation NLL and Figure~2
plots paired SOAP-minus-Muon validation-NLL differences. Table~III reports
NLL/area to six decimal places. Lower is better for all NLL and area
quantities; throughput is the sole higher-is-better diagnostic.

## Report asset provenance

| Paper item | Evidence inputs | Generator or derivation | Report use |
|---|---|---|---|
| Table I, `tab_01_experimental_protocol.tex` | `configs/experiments/confirm_moonlight_soap_200m.yaml`; `configs/data/fineweb_edu_75m.yaml`; `configs/model/gpt_35m.yaml`; `data/processed/fineweb_edu_75m_gpt2/manifest.json`; all ten confirmation metrics | Protocol values are frozen from configurations. Completed tokens and updates are checked against the metrics. | Section IV specifies pairing, data, model, budget, and evaluation cadence. |
| Table II, `tab_02_optimizer_definitions.tex` | `configs/optimizer/moonlight_muon_reference.yaml`; `configs/optimizer/moonlight_soap_muon.yaml`; `configs/experiments/confirm_moonlight_soap_200m.yaml` | Frozen routing and optimizer settings transcribed from versioned configuration. | Section III defines the compared methods. |
| Figure 1, `Figures/fig_01_validation_nll_vs_tokens.{svg,pdf}` | All ten confirmation metrics | `scripts/generate_confirmation_figures.py` | Section V-A shows the mean curve and the min--max band across paired seeds. |
| Figure 2, `Figures/fig_02_validation_nll_difference.{svg,pdf}` | All ten confirmation metrics | `scripts/generate_confirmation_figures.py` | Section V-A shows the paired NLL difference over training, including its late reversal. |
| Table III, `tab_03_confirmation_results.tex` | All ten confirmation metrics | `scripts/generate_confirmation_figures.py`, which imports `analyse_pair` from `scripts/analyze_confirmation.py` | Section V-B reports every seed's primary area and final NLL. |
| Table IV, `tab_04_confirmation_systems.tex` | All ten confirmation metrics | `scripts/generate_confirmation_figures.py`, which imports `analyse_pair` from `scripts/analyze_confirmation.py` | Section V-B reports every seed's throughput and peak-memory diagnostics. |
| Table V, `tab_05_confirmation_wall_clock.tex` | All ten confirmation metrics | `scripts/generate_confirmation_figures.py`, which imports `analyse_pair` from `scripts/analyze_confirmation.py` | Section V-B reports elapsed wall-clock time to first logged NLL-threshold crossings, including evaluation. |
| Mean contrasts, bootstrap intervals, threshold crossings, throughput, and memory in Sections V--VII | All ten confirmation metrics | `scripts/analyze_confirmation.py` -> `runs/confirmation_metrics/summary.md` | Text is constrained to the generated summary; lower is better for NLL/area and higher is better for throughput. |

Wall-clock efficiency is reported as a threshold-crossing table rather than a
curve: each value is the elapsed time, including evaluation, at the first
logged 10M-token cadence checkpoint meeting the stated NLL target. Training
throughput remains a separate, evaluation-excluded systems diagnostic.

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
  completion. They do **not** embed a Git source commit. The Modal artifact
  paths and hashes are recorded above; the source commit must still be
  recovered before a submission that requires run-level code provenance.
- The hardware field is paired within each seed: seed 202 is labelled NVIDIA
  A10G in both methods and seeds 201 and 203--205 are labelled NVIDIA A10 in
  both methods. The report therefore describes the hardware as A10-class and
  does not compare methods across differently labelled seed pairs.
- `configs/experiments/confirm_moonlight_soap_200m.yaml` is the compact frozen
  confirmation record used by the report.  It does not presently conform to
  the component-path schema expected by `scripts/validate_config.py`; do not
  represent it as a directly runnable current-code configuration until that
  compatibility issue is resolved.

