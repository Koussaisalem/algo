# QCMD Hybrid Framework

The hybrid framework wraps the verified QCMD-ECS manifold engine with data preparation,
enrichment, and model tooling aimed at NequIP-based neural components. Every script in
this directory assumes double-precision (`torch.float64`) tensors and is compatible with
the Stiefel operators in `qcmd_hybrid_framework/qcmd_ecs/core`.

## Directory layout

```
qcmd_hybrid_framework/
├── data/                 # Cached datasets and enrichment artifacts
├── manifold_utils.py     # Mass-weighted frames and unit conversions
├── models/               # Surrogate and score-model definitions
└── scripts/              # End-to-end pipeline stages
    ├── 01_prepare_data.py
    ├── 02_enrich_dataset.py
    ├── 03_train_surrogate.py
    └── 05_advanced_benchmark.py
```

## Pipeline stages

### 1. Prepare a QM9 micro-dataset

`01_prepare_data.py` downloads the QM9 dataset with PyTorch Geometric, selects a seeded
subset (`NUM_SAMPLES`) of molecules, and stores an `AtomicDataDict`-compatible payload in
`data/qm9_micro_5k.pt`.

Run it from the repository root or the `qcmd_hybrid_framework` directory:

```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/01_prepare_data.py
```

### 2. Enrich with GFN2-xTB metadata

`02_enrich_dataset.py` adds physics-aware annotations needed for manifold diffusion:

- Single-point GFN2-xTB energy (Hartree and eV)
- Forces converted to eV/Å
- Molecular orbital coefficients
- Mass-weighted orthonormal frames (via `manifold_utils.compute_manifold_frame`)

The script checkpoints progress and logs failures so long jobs can resume safely.

```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/02_enrich_dataset.py \
  --input-path data/qm9_micro_5k.pt \
  --output-path data/qm9_micro_5k_enriched.pt \
  --checkpoint-every 25 \
  --overwrite
```

Use `--resume` to append to an existing enrichment file, and `--max-molecules` to process
smaller batches while debugging.

### 3. Train the NequIP surrogate

`03_train_surrogate.py` consumes the enriched dataset to fit the NequIP surrogate defined
in `models/surrogate.py`. The script performs a reproducible train/val/test split,
trains with Adam in double precision, checkpoints the best validation weights, and
writes JSON metrics for future reporting.

```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/03_train_surrogate.py \
  --dataset-path data/qm9_micro_5k_enriched.pt \
  --output-dir models/surrogate_runs/default \
  --epochs 100 \
  --batch-size 32 \
  --lr 5e-4
```

The default target is the xTB energy in eV (`energy_ev`). Switch to Hartree supervision
with `--target energy_hartree` if you prefer to postpone the unit conversion.

### 4. Train the score model

`04_train_score_model.py` trains a NequIP-based neural network to predict
score functions (directions toward the data manifold) from noisy samples.
The model learns to denoise manifold frames using multiple noise scales
for robust generalization.

```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/04_train_score_model.py \
  --dataset-path data/qm9_micro_5k_enriched.pt \
  --output-dir models/score_model \
  --epochs 100 \
  --batch-size 16 \
  --lr 5e-4 \
  --noise-levels 0.1 0.2 0.3 0.5
```

The trained weights are saved to `models/score_model/score_model_state_dict.pt`
along with training metrics. This replaces the oracle score used in earlier
benchmarks with a learned neural predictor.

### 5. Advanced CMD-ECS vs Euclidean benchmarking

`05_advanced_benchmark.py` replays the reverse-diffusion process with three
strategies—true CMD-ECS updates, an unconstrained Euclidean walk, and a
post-hoc retracted Euclidean walk—using oracle scores and the trained surrogate
for evaluation. RDKit is used to compare generated geometries against the
ground-truth frames.

```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/05_advanced_benchmark.py \
  --num-samples 128 \
  --num-steps 40 \
  --noise-scale 0.2 \
  --gamma 0.1 \
  --surrogate-path models/surrogate/surrogate_state_dict.pt \
  --output-dir results/advanced_benchmark
```

Setting `--gamma 0.1` activates MAECS (energy-guided diffusion). Use `--gamma 0.0`
to test pure manifold diffusion without energy steering.

Artifacts include per-sample metrics, a summary JSON, and a Markdown report
(`results/advanced_benchmark/report.md`) that names the winning method (currently
CMD-ECS) and highlights where Euclidean baselines fail.

### 6. Generate novel molecules

`06_generate_molecules.py` implements full CMD-ECS inference to generate
molecular geometries from scratch. It uses the trained score and surrogate
models to perform reverse diffusion, optionally with energy-based guidance.

```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/06_generate_molecules.py \
  --num-samples 20 \
  --num-steps 50 \
  --noise-scale 0.3 \
  --gamma 0.1 \
  --score-model-path models/score_model/score_model_state_dict.pt \
  --surrogate-path models/surrogate/surrogate_state_dict.pt \
  --output-dir results/generated_molecules
```

The script generates:
- XYZ files for initial and final molecular geometries
- An HTML visualization report with statistics
- Energy predictions from the surrogate model

Use `--gamma > 0` to enable MAECS energy steering, which biases generation
toward low-energy configurations.

## Customising training data

To swap the default QM9 micro-sample for your own molecules while preserving the
workflow:

1. **Dataset preparation** – tweak `scripts/01_prepare_data.py` to point at your
  source dataset or loader. The script expects PyTorch Geometric `Data` objects
  with atomic numbers (`z`) and coordinates (`pos`).
2. **Manifold enrichment** – adjust `scripts/02_enrich_dataset.py` so that it
  computes—or imports—energies, forces, and orthonormal frames for your
  structures. Downstream stages rely on the keys `atom_types`, `pos`,
  `forces_ev_per_angstrom` (or `gradient_hartree_per_bohr`), `orbitals`, and the
  `manifold_frame` dictionary containing `frame`, `centroid`, `mass_weights`, and
  `rank`.
3. **Configuration** – pass the new artifact path via each script’s
  `--dataset-path`. All intermediate files live under `data/`, so you can keep
  multiple datasets side by side.

> **Tip:** if new atomic species appear, extend `DEFAULT_TYPE_NAMES` and
> `DEFAULT_ATOMIC_NUMBERS` inside `models/nequip_common.py`, and add their masses
> to `manifold_utils.ATOMIC_MASSES_AMU` before re-running enrichment.

## Fine-tuning the neural components

Both neural models share helper utilities in `models/nequip_common.py`, making
fine-tuning consistent:

- **Surrogate:** re-run `scripts/03_train_surrogate.py` with altered hyperparameters
  (batch size, learning rate, epochs) or a different target field using
  `--target`. The script saves the best validation checkpoint to
  `models/surrogate/surrogate_state_dict.pt`; you can resume training by loading
  the state dict manually and continuing the loop.
- **Score model:** the NequIP-based score predictor in `models/score_model.py`
  mirrors the surrogate’s setup. Wrap diffusion training samples in a PyG loader,
  feed batches through the model, and optimise against force/score supervision in
  float64. Once weights are learned, export them alongside the surrogate so
  inference can consume both modules.

After any modification, validate core manifold operations with:

```bash
cd /workspaces/algo/qcmd_hybrid_framework
pytest -q qcmd_ecs/tests/test_core.py
```

## CMD-ECS inference workflow

When a trained score model is available, molecule generation proceeds as:

1. Load the surrogate and score checkpoints (ensure they run in `torch.float64`).
2. Convert the dataset’s `manifold_frame` for the target molecule into a Stiefel
  state `U_T`, then inject noise matching your diffusion schedule.
3. Call `qcmd_ecs.core.dynamics.run_reverse_diffusion` with:
  - the noisy Stiefel state,
  - your score model `(U_t, t) -> score`,
  - an energy-gradient callback derived from the surrogate,
  - gamma/eta/tau schedules,
  - and an optional callback for logging or assertion checks.
4. Reconstruct Cartesian coordinates via the mass-weighted frame (see
  `frame_to_positions` inside `scripts/05_advanced_benchmark.py`), export to XYZ,
  and post-process with RDKit to validate chemistry.

The advanced benchmark script implements steps 2–4 with oracle scores; replacing
the oracle with your trained score model turns the setup into full-fledged
inference for CMD-ECS.

## Next steps

This README will stay the blueprint as the architecture evolves. Completed features:

- ✅ Training the score model to replace oracle gradients in benchmarking and inference
- ✅ Full molecule generation pipeline with XYZ export and HTML visualization
- ✅ MAECS energy-guided diffusion with configurable gamma parameter

Future work includes:

- Scaling to larger datasets (10k, 50k molecules) for improved model accuracy
- Advanced score model architectures (SE(3)-equivariant, attention mechanisms)
- Exploring schedule sweeps and alternative noise regimes for CMD-ECS
- Expanding benchmarks with additional baselines and molecular property prediction
- Integration with molecular docking and optimization workflows

Every new stage will ship with a script under `scripts/` and an accompanying
documentation update here.
