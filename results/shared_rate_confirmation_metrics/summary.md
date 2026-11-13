# Moonlight Muon / SOAP-Muon confirmation results

The methods are paired by seed (initialisation, data order, and batch regime). All figures below are generated from saved run metrics.

## Prespecified primary endpoint

The primary endpoint is the trapezoidal sampled validation-NLL curve area, normalised by the nominal 200M-token budget. Lower is better.

| Seed | Moonlight Muon | SOAP-Muon | SOAP − baseline |
|---:|---:|---:|---:|
| 301 | 4.118841 | 4.123700 | +0.004859 |
| 302 | 4.128683 | 4.123271 | -0.005412 |
| 303 | 4.122858 | 4.122185 | -0.000673 |
| 304 | 4.127905 | 4.125505 | -0.002400 |
| 305 | 4.126504 | 4.118473 | -0.008031 |

Mean paired difference: **-0.002332** (-0.057%); range [-0.008031, +0.004859].
Deterministic 100,000-resample bootstrap 95% interval: [-0.005936, +0.001540].
Descriptive paired t 95% interval: [-0.008428, +0.003765]; exact two-sided sign-test p-value: 0.3750.
SOAP-Muon has lower area in 4/5 pairs.

## Secondary endpoints

| Outcome | Moonlight Muon mean | SOAP-Muon mean | SOAP − baseline |
|---|---:|---:|---:|
| Final 5M-token validation NLL | 3.970229 | 3.975755 | +0.005525 |
| Training throughput (tokens/s) | 98,570 | 96,927 | -1.67% |
| Peak training memory (GiB) | 11.454 | 11.509 | +0.055 |

Final-NLL bootstrap 95% interval: [+0.001551, +0.009055].
Final-NLL paired t 95% interval: [-0.000697, +0.011748]; exact two-sided sign-test p-value: 0.3750.

| Seed | NLL ≤ 4.25: baseline / SOAP | NLL ≤ 4.20: baseline / SOAP |
|---:|---:|---:|
| 301 | 100.0M / 100.0M | 110.0M / 120.0M |
| 302 | 100.0M / 100.0M | 120.0M / 120.0M |
| 303 | 100.0M / 100.0M | 120.0M / 120.0M |
| 304 | 100.0M / 100.0M | 120.0M / 120.0M |
| 305 | 100.0M / 100.0M | 120.0M / 120.0M |

| Seed | Time to NLL ≤ 4.25: baseline / SOAP (s) | Time to NLL ≤ 4.20: baseline / SOAP (s) |
|---:|---:|---:|
| 301 | 1069.4 / 1082.6 | 1177.6 / 1299.7 |
| 302 | 1068.1 / 1081.1 | 1282.5 / 1298.2 |
| 303 | 1067.4 / 1102.2 | 1281.6 / 1323.2 |
| 304 | 1068.6 / 1091.8 | 1283.0 / 1314.1 |
| 305 | 1060.7 / 1060.7 | 1273.5 / 1272.6 |

## Interpretation boundary

SOAP-Muon has lower trajectory area in 4/5 held-out pairs. Final NLL and training throughput are reported separately; this is not an unqualified final-quality improvement.
