# Report production checklist

This file is the control list for the IEEE report. It intentionally contains
tasks, evidence locations, and completion gates rather than draft prose.

## Folder contract

- [x] `main.tex` is the only report entry point for Overleaf.
- [x] `IEEEtran.cls` is the supplied IEEE class, retained locally for a
  self-contained transfer.
- [x] `template/` preserves the supplied template source and guidance PDF;
  do not edit these files while drafting.
- [x] `Figures/` is the only directory for report figure assets.
- [x] `Tables/` is reserved for table fragments if they become too large to
  keep beside their section source.
- [x] `sections/` holds one content file per report section.
- [x] `references.bib` is the single bibliography database.

## Phase 0 - report freeze

- [ ] Confirm the assignment's page limit, marking rubric, required sections,
  and required citation style.
- [ ] Confirm final author names, affiliations, and author order.
- [ ] Confirm the report title and 3--5 keywords.
- [x] Create a claim ledger: every result claim must name its metric file,
  analysis command, and figure/table destination.
- [ ] Keep the result scope fixed: five held-out paired 200M-token runs are
  the primary evidence; route/LR screening is exploratory selection only.

## Phase 1 - reproducibility package

- [ ] Archive the ten confirmation `metrics.json` files with their Modal
  artifact names and download command.
- [ ] Record the exact source commit used for runs and the report branch commit.
- [x] Re-run `scripts/analyze_confirmation.py` from saved metrics and retain
  its generated Markdown summary as a provenance artefact.
- [ ] Verify the experiment manifest in
  `configs/experiments/confirm_moonlight_soap_200m.yaml` against the report.
- [ ] Record hardware, precision, batch tokens, actual completed tokens,
  evaluation cadence, and validation-set sizes from saved metrics/config.

## Phase 2 - tables and figures

- [x] Create `Figures/fig_01_validation_nll_vs_tokens.pdf` from all five
  paired curves, with a clearly labelled aggregate and uncertainty treatment.
- [ ] Create `Figures/fig_02_validation_nll_vs_wall_clock.pdf` only if the
  plotted clock definition includes evaluation time and is documented.
- [x] Create `Figures/fig_03_primary_endpoint_pairs.pdf` showing all five
  paired curve-area contrasts, not just the mean.
- [x] Create `Tables/tab_01_experimental_protocol.tex`: model/data/budget,
  validation schedules, hardware, and seed pairing.
- [x] Create `Tables/tab_02_optimizer_definitions.tex`: baseline versus
  SOAP-Muon routing and frozen hyperparameters.
- [x] Create `Tables/tab_03_confirmation_results.tex`: every seed, primary
  endpoint, final NLL, throughput, memory, and status.
- [ ] For each asset, record: source file paths, generator command, commit,
  units, legend labels, and the report section that cites it.
- [ ] Do not create an ablation figure unless a separately planned experiment
  is completed; do not relabel the route/LR screen as confirmation.

## Phase 3 - section drafting order

- [x] `00_abstract.tex`: draft last, after all claims and numbers are frozen.
- [x] `01_introduction.tex`: motivation, precise question, and bounded
  contributions; avoid claiming SOAP-Muon novelty.
- [x] `02_background.tex`: cite Muon, SOAP, and directly relevant optimiser
  studies; define terminology once.
- [x] `03_method.tex`: describe only implemented routing and update choices;
  cross-check every detail against source/configuration.
- [x] `04_experimental_protocol.tex`: explain screening/confirmation
  separation, pairing, evaluation, metrics, and analysis rules.
- [x] `05_results.tex`: lead with the prespecified primary endpoint, then
  final NLL and systems trade-offs; show every confirmation seed.
- [x] `06_discussion.tex`: state the sample-efficiency result and its cost;
  discuss small-scale, finite-seed, and fixed-budget limitations.
- [x] `07_conclusion.tex`: answer the research question without extending the
  claim beyond the observed 200M-token regime.

## Phase 4 - evidence and writing checks

- [ ] Verify every numerical value against a generated table, never a manual
  transcription.
- [ ] Label methods consistently: unpreconditioned Moonlight Muon baseline
  and selected one-sided SOAP-Muon route.
- [ ] Use lower-is-better direction markers for NLL/area and higher-is-better
  markers for throughput.
- [ ] State that the final-NLL contrast is mixed and its bootstrap interval
  crosses zero.
- [ ] State the primary area contrast is consistently lower across all five
  pairs, with its reported bootstrap interval.
- [ ] Separate observed results, interpretation, and proposed future work.
- [x] Add citations only after verifying the cited source supports the claim.

## Phase 5 - build and Overleaf handoff

- [x] Compile from `report/main.tex` with the local `IEEEtran.cls`.
- [ ] Resolve all undefined references, citations, overfull boxes, and missing
  figure/table warnings.
- [x] Check the compiled PDF visually: two-column flow, captions, tables,
  axis labels, fonts, and page limit.
- [ ] Upload the contents of `report/` to Overleaf, preserving case-sensitive
  paths such as `Figures/` and `Tables/`.
- [ ] Set `main.tex` as the Overleaf main document.
- [ ] Perform one final clean compile in Overleaf before submission.
