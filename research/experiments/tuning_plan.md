# Equal-budget tuning plan

## Principle

Tuning is part of the experimental budget. A method is not allowed a larger
search simply because it has more hyperparameters. All selection happens on a
screening seed and a fixed short horizon; confirmation seeds remain untouched.

## Before tuning

1. Complete E0 and commit the chosen reference-Muon specification.
2. Record the available A10 spend from Modal and reserve enough budget for all
   confirmation runs before launching a screening run.
3. Create a manifest listing every candidate, seed, branch commit, planned
   tokens, and stop condition.

## Screening protocol

| Item | Rule |
|---|---|
| Horizon | 40M training tokens, with validation every 10M tokens |
| Screening seed | Seed 101 only; never reused in confirmation |
| Objective | Lowest token-normalised validation-loss area through 40M tokens |
| Candidate count | Three learning-rate candidates per method and batch regime |
| Required baseline | Screen and freeze unpreconditioned Moonlight Muon before comparing any SOAP–Muon route |
| Weight decay | One shared, predeclared value unless E0 identifies a source-mandated difference |
| SOAP basis frequency | Use the pinned source default for faithful SOAP–Muon; do not tune it in the main screen |
| Failure | Non-finite loss, a sustained divergence flag, or an out-of-memory run is retained and ranked ineligible |

The LR grid is multiplicative around the source/reference value: `0.5×`,
`1×`, and `2×`. If the source has no appropriate value for the selected Muon
definition, define the centre using a short AdamW/Muon calibration and commit
it before hybrid screening.

## Selection and confirmation

1. Select one LR per method/batch with the screening objective.
2. Freeze the selected unpreconditioned Moonlight Muon settings as the primary
   baseline; every hybrid confirmation run is paired against that baseline.
3. In E2, select one hybrid route by the rule in `run_matrix.md`.
4. Freeze all selected settings in versioned YAML files.
5. Run confirmation seeds 201, 202, and 203 at 200M tokens for both the
   baseline and the selected hybrid route.
6. Do not retune after inspecting any confirmation loss curve. A failed
   confirmation run is reported, not silently replaced.

## Budget guardrail

The confirmation set has priority over optional architecture work. Do not
start E4 until E3 is complete and the remaining Modal spend supports all
declared E4 runs. Record GPU, CPU, memory, and setup charges from Modal rather
than inferring final spend from training throughput.
