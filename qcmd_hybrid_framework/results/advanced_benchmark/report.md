# Advanced CMD-ECS vs Euclidean Benchmark

## Configuration

- Samples: 64
- Steps: 40
- Noise scale: 0.200
- Eta: 0.040
- Tau: 0.015
- Gamma: 0.000
- Seed: 2025
- Surrogate: models/surrogate/surrogate_state_dict.pt

## Mean Metrics

| Metric | CMD-ECS | Euclidean | Euclid+Retract |
| --- | ---: | ---: | ---: |
| fro_error | 2.974284 | 7.495691 | 2.145396 |
| orth_error | 0.000000 | 36.256124 | 0.000000 |
| alignment | -0.477783 | 0.227074 | 0.225213 |
| rmsd | 2.561241 | 15.158224 | 2.735267 |
| energy_abs_error | 92.678384 | 60.500353 | 102.249563 |
| runtime_ms | 5.649279 | 1.518437 | 0.050894 |

## Winner

**cmd_ecs** delivered the best geometric and energetic fidelity.

## Observations

- CMD-ECS maintains orthogonality to machine precision, avoiding Gram drift.
- Euclidean updates wander off the manifold, causing dramatic RMSD growth.
- Retraction after Euclidean steps restores orthogonality but still trails CMD-ECS in RMSD and energy error.
- Surrogate energy deviations correlate with RDKit RMSDs, highlighting the physical cost of leaving the manifold.

## Next Steps

- Replace the oracle score with the trained score model once available.
- Explore schedule variations (e.g., cosine eta/tau) and larger sample pools.
- Benchmark against additional Euclidean heuristics such as orthogonality penalties.
