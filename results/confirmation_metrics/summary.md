# Moonlight Muon / SOAP-Muon confirmation results

The methods are paired by seed (initialisation, data order, and batch regime). All figures below are generated from saved run metrics.

## Prespecified primary endpoint

The primary endpoint is the trapezoidal sampled validation-NLL curve area, normalised by the nominal 200M-token budget. Lower is better.

| Seed | Moonlight Muon | SOAP-Muon | SOAP − baseline |
|---:|---:|---:|---:|
| 201 | 4.159223 | 4.153635 | -0.005588 |
| 202 | 4.161580 | 4.152614 | -0.008966 |
| 203 | 4.153476 | 4.148413 | -0.005064 |
| 204 | 4.150768 | 4.148460 | -0.002307 |
| 205 | 4.155791 | 4.147575 | -0.008216 |

Mean paired difference: **-0.006028** (-0.145%); range [-0.008966, -0.002307].
Deterministic 100,000-resample bootstrap 95% interval: [-0.008035, -0.004040].
Descriptive paired t 95% interval: [-0.009336, -0.002721]; exact two-sided sign-test p-value: 0.0625.
SOAP-Muon has lower area in 5/5 pairs.

## Secondary endpoints

| Outcome | Moonlight Muon mean | SOAP-Muon mean | SOAP − baseline |
|---|---:|---:|---:|
| Final 5M-token validation NLL | 3.997182 | 3.998108 | +0.000926 |
| Training throughput (tokens/s) | 99,109 | 97,420 | -1.70% |
| Peak training memory (GiB) | 11.454 | 11.509 | +0.055 |

Final-NLL bootstrap 95% interval: [-0.001369, +0.003359].
Final-NLL paired t 95% interval: [-0.002821, +0.004673]; exact two-sided sign-test p-value: 1.0000.

| Seed | NLL ≤ 4.25: baseline / SOAP | NLL ≤ 4.20: baseline / SOAP |
|---:|---:|---:|
| 201 | 100.0M / 100.0M | 120.0M / 120.0M |
| 202 | 110.0M / 100.0M | 120.0M / 120.0M |
| 203 | 100.0M / 100.0M | 120.0M / 120.0M |
| 204 | 100.0M / 100.0M | 120.0M / 120.0M |
| 205 | 100.0M / 100.0M | 120.0M / 120.0M |

| Seed | Time to NLL ≤ 4.25: baseline / SOAP (s) | Time to NLL ≤ 4.20: baseline / SOAP (s) |
|---:|---:|---:|
| 201 | 1065.6 / 1081.7 | 1279.3 / 1299.1 |
| 202 | 1152.3 / 1062.0 | 1257.2 / 1275.8 |
| 203 | 1059.3 / 1081.9 | 1271.7 / 1298.6 |
| 204 | 1071.8 / 1079.1 | 1287.0 / 1295.6 |
| 205 | 1063.7 / 1092.9 | 1277.1 / 1312.1 |

## Interpretation boundary

The primary endpoint favours SOAP-Muon consistently across these five held-out pairs. Final NLL is mixed, and SOAP-Muon has lower training throughput; this is a sample-efficiency trade-off rather than an unqualified final-quality improvement.
