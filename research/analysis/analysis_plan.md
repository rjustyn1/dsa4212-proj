# Analysis plan

## Primary endpoint

For each confirmation run, evaluate validation NLL on the same fixed 1M-token
slice every 10M training tokens. The primary statistic is the trapezoidal area
under that validation-NLL-versus-training-tokens curve through 200M tokens,
divided by 200M. Lower is better.

This metric measures the complete learning trajectory without choosing one
post-hoc endpoint or target threshold.

## Secondary endpoints

| Outcome | Definition | Better direction |
|---|---|---:|
| Final validation NLL | Full fixed 5M-token validation at 200M training tokens | Lower |
| Token efficiency | First training-token count reaching NLL 4.25 and NLL 4.20 on the curve slice; mark unreached rather than extrapolating | Lower |
| Wall-clock efficiency | Elapsed wall-clock time to the same target crossing, including optimiser and evaluation time | Lower |
| Throughput | Training tokens per second, excluding validation; systems diagnostic only | Higher |
| Memory | Peak training-device memory | Lower |
| Stability | Non-finite events, divergence, and completed-run rate | Fewer failures |

The two NLL targets are fixed before confirmation. They are secondary aids to
interpretation, not the primary basis for declaring a result.

## Mechanism diagnostics

On a fixed subset of MLP layers and fixed logged steps, save:

1. raw momentum/gradient singular values;
2. SOAP-preconditioned direction singular values;
3. post-Muon update singular values;
4. update RMS and cosine similarity between Muon and hybrid updates;
5. SOAP basis-refresh age and covariance-spectrum summary.

Diagnostics run outside timed training steps. They support an explanation only
when their pattern agrees with the loss and efficiency results; they do not by
themselves establish optimisation benefit.

## Uncertainty and comparison rules

- The central contrast is each SOAP–Muon method minus its paired,
  unpreconditioned Moonlight Muon baseline; neither AdamW nor an older Muon
  variant substitutes for that comparison.
- Pair methods by seed, initialization, data order, and batch regime.
- Report every seed, the paired differences, their mean, range, and a bootstrap
  interval where useful. With three seeds, avoid treating a p-value as a
  decisive result.
- A positive result requires a consistent direction across seeds and a stated
  practical trade-off. A lower final NLL that takes materially longer is a
  trade-off, not an unconditional win.
- A result that fails to reach a target is retained as `not reached`; no curve
  extrapolation is permitted.
- Report failed, stopped, and excluded runs with their reason and incurred
  budget.

## Reporting boundary

The final report distinguishes:

- faithful replication of published SOAP–Muon;
- the project's proposed Muon-aligned extension, if run;
- exploratory pilots;
- held-out confirmation results.

Every table and figure must be generated from saved run manifests and metrics,
not manually transcribed.
