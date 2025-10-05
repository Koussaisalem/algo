# Copilot Instructions for QCMD-ECS

## Architecture recap
- `qcmd_ecs/core/` is the mathematically verified heart: `manifold.py` implements the Stiefel operators, `dynamics.py` wires them into `run_reverse_diffusion`, and `types.py` enforces `torch.float64` semantics via `DTYPE`.
- Scripts `1_prepare_dataset.py`→`6_analyze_results.py` form the research pipeline: prepare QM9 slices, enrich them with xTB energies, train a GNN surrogate, fit a SchNet score model, run manifold diffusion, then benchmark results.
- `models/` holds neural components. `score_model.py` wraps SchNetPack to output per-atom 3D vectors; `surrogate.py` currently exposes a bare SchNet instance (`model`) but no class—`test_surrogate.py` expects `Surrogate`/`SimpleSurrogate`, so add or import them before using that test.

## Data & shape conventions
- Everything that touches `qcmd_ecs.core` must live in double precision (`torch.float64`). Cast external tensors before calling `project_to_tangent_space`, `retract_to_manifold`, or `run_reverse_diffusion`.
- `run_reverse_diffusion` expects callables returning `(m,k)` tensors on the Stiefel manifold. The callback signature is `callback(t: int, U_t: Tensor)` with `t` counting down.
- Surrogate/score models work with PyG `Data` objects; include `batch` (even if zeros) so SchNetPack’s `NeuralNetworkPotential` can read it. When orbitals are needed, stash them under `data.orbitals` as a `(n_atoms, orbital_dim)` tensor.

## Critical workflows
- Dataset prep assumes QM9 via `torch_geometric.datasets.QM9` and writes to `data/`. Run sequentially: `python 1_prepare_dataset.py` → `python 2_enrich_dataset.py --input_path data/qm9_micro_raw.pt`.
- Surrogate training (`3_train_surrogate.py`) loads `data/qm9_micro_enriched.pt` onto CUDA, validates entries, and expects a usable model exported from `models/surrogate.py` as `model`. Update that module to return a `torch.nn.Module` with `.to(dtype=DTYPE, device=DEVICE)` support before training.
- Generative training (`4_run_generative_training.py`) still uses `DummyDataset` scaffolding. Replace it with real loaders but preserve the Langevin-style noise schedule and the wrappers that adapt score/surrogate outputs to the Stiefel update API.
- Molecule generation (`5_generate_molecules.py`) writes `.xyz` files and needs both trained weights (`models/score_model_*.pt`, `models/surrogate_model.pt`). Make sure wrappers reshape U-vectors (flattened `(num_atoms*3,)`) back to `(num_atoms,3)` before invoking PyG radius graphs.
- Result analysis (`6_analyze_results.py`) loads `.xyz`, filters valid RDKit molecules, then calls a TorchScript surrogate via `torch.jit.load`. Export the surrogate appropriately, or adapt the loader if you keep a state-dict model.

## Tests & diagnostics
- The authoritative regression suite is `pytest -v qcmd_ecs/tests/test_core.py`. It enforces manifold invariants and energy-gradient projection—keep it green after every change to `core/`.
- `qcmd_ecs_tests.py` is a standalone high-precision sanity script; run it when tweaking schedule math or tangent projections.
- `diagnose_pyg.py` prints available PyG convolutions if you hit import/runtime issues.

## Implementation tips & pitfalls
- Always seed torch (`torch.manual_seed`) before stochastic diffusion loops to keep determinism aligned with `run_reverse_diffusion`’s internal seed call.
- Clamp QR retraction signs exactly as in `retract_to_manifold`; altering that logic will break orthonormality assertions at 1e-9 tolerance.
- `models/score_model.ScoreModel` returns a dict keyed by `spk.properties`; unwrap `['score']` and ensure gradients flow by keeping tensors on the same device/dtype as the model.
- Many scripts hard-code `torch.device('cuda')`. Add graceful CPU fallbacks or checks if you intend to run in CI.
- External tools: `xtb` often needs Conda (`conda install -c conda-forge xtb-python`), and SchNetPack 2.x may require pinning PyTorch ≤2.1—see `requirements.txt` notes before upgrading.

## When extending the repo
- Document any new manifold ops in `qcmd_ecs/core` and expand `TestQCMDECSMechanics` with edge cases (e.g., larger `m,k` or stress tests on schedule functions).
- Keep pipeline scripts stateless: read inputs from `data/`, write derived artifacts to `models/` or `results/`, and avoid hidden globals so downstream steps remain reproducible.
- Prefer small helper modules over bloating the procedural scripts; e.g., move dataset transforms into `dataset.py` once its placeholder code is replaced.
