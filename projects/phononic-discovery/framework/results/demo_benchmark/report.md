# CMD-ECS vs Euclidean Benchmark

## Configuration

- Samples: 5
- Steps: 20
- Noise scale: 0.1000
- Eta: 0.0500
- Tau: 0.0200
- Gamma: 0.0000
- Seed: 1234

## Mean Metrics

| Metric | Manifold | Euclidean |
| --- | ---: | ---: |
| Frobenius error | 1.863547 | 2.395518 |
| Orthogonality error | 0.000000 | 3.794945 |
| Cosine alignment | 0.420751 | 0.574624 |
| Runtime (ms) | 2.615 | 0.659 |

## Notes

- Oracle score pulls states toward the dataset frame; no trained score model is required.
- Euclidean baseline omits tangent projection and retraction, highlighting orthogonality drift.
