# SOAP–Muon: rigorous experiment matrix

## Status and scope

This is the pre-registered plan for the main study. It replaces the earlier
single-endpoint framing. Existing 200M-token runs are **exploratory pilots**:
they motivate this plan, but will not be used as confirmatory evidence or to
select a winner after the fact.

The question is:

> Under controlled gradient conditioning, matrix orientation, and batch-noise
> conditions, when does one-sided SOAP preconditioning improve the selected
> reference Muon update enough to offset its added cost?

The project does **not** claim to invent SOAP–Muon. It independently studies
the published idea's boundary conditions, orientation sensitivity, and
practical efficiency.

## Mandatory decision gate: define Muon once

No new main-result training run may start before this table is resolved in a
committed optimizer specification and deterministic update fixture.

| Candidate | Definition | Role |
|---|---|---|
| `muon_original` | The pinned original/reference Muon convention matched to the public SOAP–Muon report | Required for a faithful SOAP–Muon replication |
| `muon_moonlight` | The later Moonlight RMS-matched convention, including its published update scale and decay rule | Optional proposed extension; never label it as the published SOAP–Muon method |

The primary study uses exactly one choice. The other may only be an explicitly
named follow-up, never a silent implementation substitution.

## Shared controls

- Fixed tokenizer, corpus split, model initialization rule, sequence length,
  precision, schedule shape, clipping, validation shards, and seed set within
  each comparison.
- AdamW handles the same excluded embeddings, output head, normalisation, and
  one-dimensional parameters in every matrix-aware method.
- Every method receives equal tuning **GPU-token budget**, equal candidate
  count, and the same screening horizon.
- Main confirmation seeds are held out from tuning. A common seed is paired
  across methods.
- Each run logs a fixed 1M-token validation slice every 10M training tokens,
  a full 5M-token validation once at the endpoint, elapsed time, peak memory,
  and method-specific update diagnostics.
- Each material experiment has its own `experiment/<topic>` branch. Only the
  user authorises a merge to `main`.

## Experiment stages

### E0 — source and numerical audit

**Branch:** `experiment/optimizer-spec`

**Purpose:** establish that method names map to one exact update rule.

| Check | Required evidence |
|---|---|
| Reference Muon | Deterministic one- and two-step fixtures for square, 4:1, 1:4, 8:1, and 1:8 matrices |
| Faithful SOAP–Muon | Fixture matched to the pinned public source, including basis refresh, moments, stabilisation, and decay order |
| Route audit | `up_proj`, `down_proj`, both, attention, embedding/head, and vectors each have exactly one route |
| Invariance diagnostic | A rotation-only ablation is numerically equivalent to Muon; this guards against mistaking a basis rotation for preconditioning |

This stage is correctness work, not a GPU result.

### E1 — controlled matrix/MLP mechanism study

**Branch:** `experiment/conditioning-mechanism`

**Purpose:** test the explanation before interpreting language-model results.

Use a synthetic regression or classification MLP with known feature covariance.
The task, target, optimizer schedule, and generated data seed are fixed across
methods.

| Factor | Levels |
|---|---|
| Matrix shape | 1:1, 4:1, 1:4, 8:1, 1:8 |
| Conditioning | isotropic (`κ≈1`), anisotropic (`κ≈100`) |
| Gradient noise | full/large batch, noisy/small batch |
| Method | AdamW, selected reference Muon, one-sided SOAP, faithful SOAP–Muon |

Run an exploratory one-seed screen across the full grid. Confirm the four
orientation contrasts (4:1 versus 1:4; 8:1 versus 1:8 under anisotropic data)
with three held-out seeds. Record objective value, steps and time to target,
raw-gradient spectrum, SOAP-preconditioned spectrum, post-Muon spectrum,
update RMS, and failures.

This study supplies the mechanism result: whether a benefit depends on
conditioning, orientation, or reduced gradient noise.

### E2 — Transformer orientation screening

**Branch:** `experiment/route-orientation`

**Purpose:** transfer the mechanism question to one fixed GPT architecture
without confounding it with a multi-architecture study.

The existing MLP already contains both orientations:

| Route | Stored shape | Orientation |
|---|---:|---|
| `up_proj` | `(4d, d)` | 4:1 |
| `down_proj` | `(d, 4d)` | 1:4 |

At the standard global batch, screen the following methods at the equal tuning
horizon specified in `tuning_plan.md`:

1. reference Muon;
2. faithful SOAP–Muon on `up_proj` only;
3. faithful SOAP–Muon on `down_proj` only;
4. faithful SOAP–Muon on both MLP projections.

AdamW remains a contextual baseline and is not used to select a hybrid route.
The selection rule is fixed: choose the hybrid route with the lowest screening
validation-loss area, subject to finite training and no worse than 5% peak
memory increase beyond the stated hardware limit. The screening seed is never
reused for confirmation.

### E3 — Transformer batch-noise confirmation

**Branch:** `experiment/batch-interaction`

**Purpose:** test the published claim's most relevant condition: whether any
hybrid benefit changes with global batch size.

Use the same model, frozen corpus, and selected Muon definition at:

- standard batch: 32,768 global tokens;
- large batch: the largest predeclared batch that fits the A10 memory limit,
  initially targeted at 131,072 global tokens.

For each batch, compare reference Muon with the route selected in E2 using
three held-out paired seeds and the full 200M-token budget. If E2 selects no
stable hybrid route, E3 stops and reports that result; it does not search new
routes after the fact.

### E4 — optional 8:1 Transformer follow-up

**Branch:** `experiment/aspect-ratio-followup`

This is optional, not part of the required claim. It is permitted only if E1
shows a reproducible orientation/conditioning effect and E3 shows a batch
interaction. Maintain approximately constant parameter count if changing MLP
ratio; otherwise label the result as an architecture-confounded sensitivity
check rather than an aspect-ratio causal result.

## Planned evidence and stopping rules

| Claim | Required evidence |
|---|---|
| A method improves sample efficiency | Lower token-normalised validation-loss area and a consistent paired result across three confirmation seeds |
| A method improves practical efficiency | Same or lower wall-clock time to a predeclared NLL target after including optimiser overhead |
| An orientation matters | Up-only and down-only routes differ consistently in E1 and are tested as separate routes in E2 |
| Batch interaction exists | Effect size differs between the two predeclared E3 batch regimes; report uncertainty, not a binary significance claim |
| No practical benefit | Hybrid fails to improve both loss and time, is unstable, or its effect lies within seed variability |

No result is extrapolated to larger language models. The report must retain
negative and failed runs, including tuning candidates.
