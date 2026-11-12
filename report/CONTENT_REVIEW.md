# Content review handoff

This checklist is for the final human review of the report on the recorded
five-seed confirmation study. It separates facts already checked against saved
artifacts from choices that require the author or course brief. It is not a
source of experimental results; `research/analysis/figure_table_map.md` is the
claim-to-artifact record.

## Review in reading order

| Item | Checked factual content | Reviewer decision still needed |
|---|---|---|
| Title and front matter | The title describes a bounded evaluation rather than a new optimizer. Keywords match the compared methods and setting. | Replace the `TODO` author line in `main.tex`; confirm the final title and 3--5 keywords. |
| Abstract | States the five paired seeds, 200M-token nominal budget, area result, late reversal, mixed endpoint, and systems cost. | Decide whether the numerical detail is appropriate for the assignment abstract limit. |
| Introduction | States one focused question, explains the fixed data-order pairing, and limits the conclusion to this model/data/budget. | Check that the stated motivation and contribution style matches the rubric. |
| Background | Uses primary sources for Transformer, Adam/AdamW, Shampoo, SOAP, Moonlight, and FineWeb-Edu. The Muon technical note is explicitly labelled as an implementation source. | Confirm that citing the Muon technical note is acceptable for the course. The non-formal SOAP--Muon technical report is deliberately not cited. |
| Method | Equations, routing, state updates, hyperparameters, and parameter counts were checked against the committed optimizer code and frozen configuration. | Check notation and desired mathematical depth. |
| Protocol | Records fixed data order, seed role, hardware labels, BF16/TF32, tokens, update count, validation sizes, endpoints, and screen/confirmation separation. | Confirm any rubric-specific reproducibility wording or required ethics/data statement. |
| Results | Every numeric result is generated from the ten archived metrics files. Figures and tables show all five seed pairs, trajectory, endpoint, throughput, memory, and threshold wall-clock values. | Confirm that the chosen primary endpoint and the explicit screen limitation are acceptable framing. |
| Discussion and conclusion | Claims only an early token-indexed difference; they explicitly reject general, final-quality, and wall-clock claims. | Decide whether to retain the proposed equal-budget follow-up as the final sentence. |
| References | All bibliography keys resolve in the local IEEE build; SOAP is cited as the ICLR 2025 paper. | Apply the course's required bibliography or citation style if it differs from IEEE. |

## Evidence already checked

- `uv run python scripts/verify_confirmation_artifacts.py` validates the ten
  saved metric files, five paired seeds, 199,983,104 completed tokens per run,
  and 6,103 updates.
- `python3 scripts/analyze_confirmation.py --metrics-dir
  runs/confirmation_metrics --report runs/confirmation_metrics/summary.md`
  regenerates the reported area, endpoint, threshold, throughput, and memory
  summaries.
- `python3 scripts/generate_confirmation_figures.py --metrics-dir
  runs/confirmation_metrics --figures-dir report/Figures --tables-dir
  report/Tables` regenerates the report figures and generated tables.
- The report was compiled locally from `main.tex`; references, citations,
  figures, and tables resolve without undefined-reference or overflow errors.

## Author-supplied inputs before submission

1. Assignment rubric, page limit, mandatory sections, and required citation
   style.
2. Final author name(s), affiliation(s), email/contact information, and order.
3. Confirmation of the final title and keywords.
4. If required by the assignment, the exact source commit that produced the
   archived GPU runs. The saved run metrics do not include a Git commit, so it
   cannot be recovered from the artifacts alone.

## Scope that must not be overstated

- The baseline learning-rate screen selected its high boundary and had fewer
  route/learning-rate candidates than the treatment; the comparison does not
  prove an equally tuned baseline.
- Seeds vary model initialization only. The token stream and order are fixed,
  so the seed spread is conditional on that data order.
- The 10M-token curve cadence limits threshold timing precision.
- There is no completed architecture study, routing ablation, or mechanism
  diagnostic. No statement about those factors belongs in the results.
