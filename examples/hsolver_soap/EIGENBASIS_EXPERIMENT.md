# SOAP eigenbasis paper-candidate experiment

This experiment is an explanatory numerical diagnostic for the Hsolver SOAP
case study. It is not a convergence proof and is not used for the throughput
measurement.

## Fixed experiment matrix

| Scenario | Steps per backend | Tracked factors | Actual runs |
| --- | ---: | ---: | --- |
| `seq128` | 120 | layer 0 and layer 3, 2048-dimensional right factors | independent Torch, then independent Hsolver |
| `seq1024` | 120 | layers 3, 11, 19, and 27, 2048-dimensional right factors | independent Torch, then independent Hsolver |

Each actual trajectory computes fresh FP32 EVD references only for its own
current Kronecker factors. No alternate QR backend is executed inside that
trajectory, and optimizer state is never shared between the Torch and Hsolver
runs.

## Primary reported quantities

1. Aggregate relative eigen-residual:

   `||KQ - Q diag(Q^T KQ)||_F / ||K||_F`

2. Leading subspace overlap at the smallest `k` covering at least 99% of the
   FP32 EVD spectral trace:

   `||V_k^T Q_k||_F^2 / k`

The tail residual energy share is retained as a secondary explanatory metric.
Thresholds other than 99% are retained for sensitivity analysis but are not
additional primary claims.

## Interpretation boundary

The independent runs provide end-to-end descriptive evidence. Because their
optimizer states and factors evolve independently, subtracting their spectral
metrics is not a same-input solver-error measurement. Long-run loss and
throughput claims must continue to come from the existing independent 4000-step
`seq128` and 2400-step `seq1024` training runs.

## Entrypoint

Run one scenario with:

```bash
bash examples/hsolver_soap/run_paper_eigenbasis_pair.sh <seq128|seq1024>
```

The default result root is:

```text
/home/ll/HPC/results/soap-8b/eigenbasis-paper-candidate
```
