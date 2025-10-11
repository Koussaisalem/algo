<div align="center"># QCMD Hybrid Framework



# Quantum Materials Discovery PlatformThe hybrid framework wraps the verified QCMD-ECS manifold engine with data preparation,

enrichment, and model tooling aimed at NequIP-based neural components. Every script in

<p align="center">this directory assumes double-precision (`torch.float64`) tensors and is compatible with

  <i>AI-driven discovery of materials with exotic quantum properties</i>the Stiefel operators in `qcmd_hybrid_framework/qcmd_ecs/core`.

</p>

## Directory layout

<p align="center">

  <a href="https://github.com/Koussaisalem/algo/actions">```

    <img src="https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square&logo=github" alt="Tests">qcmd_hybrid_framework/

  </a>├── data/                 # Cached datasets and enrichment artifacts

  <a href="https://github.com/Koussaisalem/algo/blob/main/LICENSE">├── manifold_utils.py     # Mass-weighted frames and unit conversions

    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License">├── models/               # Surrogate and score-model definitions

  </a>└── scripts/              # End-to-end pipeline stages

  <a href="https://github.com/Koussaisalem/algo">    ├── 01_prepare_data.py

    <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python" alt="Python">    ├── 02_enrich_dataset.py

  </a>    ├── 03_train_surrogate.py

  <a href="https://github.com/Koussaisalem/algo/stargazers">    └── 05_advanced_benchmark.py

    <img src="https://img.shields.io/github/stars/Koussaisalem/algo?style=flat-square&logo=github&color=yellow" alt="Stars">```

  </a>

</p>## Pipeline stages



<p align="center">### 1. Prepare a QM9 micro-dataset

  <a href="#overview">Overview</a> •

  <a href="#key-achievements">Achievements</a> •`01_prepare_data.py` downloads the QM9 dataset with PyTorch Geometric, selects a seeded

  <a href="#installation">Installation</a> •subset (`NUM_SAMPLES`) of molecules, and stores an `AtomicDataDict`-compatible payload in

  <a href="#projects">Projects</a> •`data/qm9_micro_5k.pt`.

  <a href="#documentation">Documentation</a> •

  <a href="#citation">Citation</a>Run it from the repository root or the `qcmd_hybrid_framework` directory:

</p>

```bash

</div>cd /workspaces/algo/qcmd_hybrid_framework

python scripts/01_prepare_data.py

---```



## Overview### 2. Enrich with GFN2-xTB metadata



This platform combines **manifold-constrained generative AI** with **ab-initio quantum chemistry** to systematically discover materials with targeted topological and electronic properties.`02_enrich_dataset.py` adds physics-aware annotations needed for manifold diffusion:



**Core Innovation**: Diffusion models operating on the Stiefel manifold, enabling physics-aware crystal structure generation with built-in symmetry constraints and orthonormality preservation.- Single-point GFN2-xTB energy (Hartree and eV)

- Forces converted to eV/Å

<details>- Molecular orbital coefficients

<summary><b>Technical Architecture</b> (click to expand)</summary>- Mass-weighted orthonormal frames (via `manifold_utils.compute_manifold_frame`)



<br>The script checkpoints progress and logs failures so long jobs can resume safely.



**Key Components**:```bash

- **Stiefel Manifold Diffusion**: Constrained generation preserving geometric structurecd /workspaces/algo/qcmd_hybrid_framework

- **Equivariant GNN Surrogates**: Fast property prediction respecting E(3) symmetriespython scripts/02_enrich_dataset.py \

- **Automated DFT Pipeline**: GPAW-based validation with phonon analysis  --input-path data/qm9_micro_5k.pt \

- **Synthesis Design**: MBE/CVD parameter optimization via AIMD  --output-path data/qm9_micro_5k_enriched.pt \

  --checkpoint-every 25 \

**Workflow**:  --overwrite

``````

Generative Model → Candidate Structures → GNN Surrogate → DFT Validation → Synthesis Protocols

```Use `--resume` to append to an existing enrichment file, and `--max-molecules` to process

smaller batches while debugging.

</details>

### 3. Train the NequIP surrogate

---

`03_train_surrogate.py` consumes the enriched dataset to fit the NequIP surrogate defined

## Key Achievementsin `models/surrogate.py`. The script performs a reproducible train/val/test split,

trains with Adam in double precision, checkpoints the best validation weights, and

<table>writes JSON metrics for future reporting.

  <tr>

    <td align="center" width="33%">```bash

      <h3>Novel 2D Material</h3>cd /workspaces/algo/qcmd_hybrid_framework

      <p><b>CrCuSe₂</b></p>python scripts/03_train_surrogate.py \

      <p>Hetero-metallic TMD with 0.6 eV bandgap</p>  --dataset-path data/qm9_micro_5k_enriched.pt \

      <p><i>Dynamically stable (0 imaginary phonons)</i></p>  --output-dir models/surrogate_runs/default \

      <a href="docs/discoveries/CrCuSe2/">Read more →</a>  --epochs 100 \

    </td>  --batch-size 32 \

    <td align="center" width="33%">  --lr 5e-4

      <h3>Computational Validation</h3>```

      <p><b>Multi-scale verification</b></p>

      <p>xTB → DFT → Phonon dispersion</p>The default target is the xTB energy in eV (`energy_ev`). Switch to Hartree supervision

      <p><i>97% consultant validation accuracy</i></p>with `--target energy_hartree` if you prefer to postpone the unit conversion.

      <a href="docs/discoveries/CrCuSe2/VALIDATION.md">Validation report →</a>

    </td>### 4. Train the score model

    <td align="center" width="33%">

      <h3>Synthesis Protocol</h3>`04_train_score_model.py` trains a NequIP-based neural network to predict

      <p><b>MBE growth design</b></p>score functions (directions toward the data manifold) from noisy samples.

      <p>Temperature screening via AIMD</p>The model learns to denoise manifold frames using multiple noise scales

      <p><i>Optimal conditions computed</i></p>for robust generalization.

      <a href="projects/phononic-discovery/framework/synthesis_lab/">Protocols →</a>

    </td>```bash

  </tr>cd /workspaces/algo/qcmd_hybrid_framework

</table>python scripts/04_train_score_model.py \

  --dataset-path data/qm9_micro_5k_enriched.pt \

---  --output-dir models/score_model \

  --epochs 100 \

## Installation  --batch-size 16 \

  --lr 5e-4 \

```bash  --noise-levels 0.1 0.2 0.3 0.5

# Clone repository```

git clone https://github.com/Koussaisalem/algo.git

cd algoThe trained weights are saved to `models/score_model/score_model_state_dict.pt`

along with training metrics. This replaces the oracle score used in earlier

# Install dependenciesbenchmarks with a learned neural predictor.

pip install -r requirements.txt

### 5. Advanced CMD-ECS vs Euclidean benchmarking

# Verify installation (tests coming soon)

python -c "from core.qcmd_ecs.core import manifold; print('Installation successful!')"`05_advanced_benchmark.py` replays the reverse-diffusion process with three

```strategies—true CMD-ECS updates, an unconstrained Euclidean walk, and a

post-hoc retracted Euclidean walk—using oracle scores and the trained surrogate

<details>for evaluation. RDKit is used to compare generated geometries against the

<summary><b>Requirements</b></summary>ground-truth frames.



<br>```bash

cd /workspaces/algo/qcmd_hybrid_framework

- Python ≥ 3.10python scripts/05_advanced_benchmark.py \

- PyTorch ≥ 2.0  --num-samples 128 \

- PyTorch Geometric  --num-steps 40 \

- ASE (Atomic Simulation Environment)  --noise-scale 0.2 \

- GPAW (optional, for DFT calculations)  --gamma 0.1 \

- xTB (optional, for semi-empirical methods)  --surrogate-path models/surrogate/surrogate_state_dict.pt \

  --output-dir results/advanced_benchmark

See [`requirements.txt`](requirements.txt) for complete dependencies.```



</details>Setting `--gamma 0.1` activates MAECS (energy-guided diffusion). Use `--gamma 0.0`

to test pure manifold diffusion without energy steering.

---

Artifacts include per-sample metrics, a summary JSON, and a Markdown report

## Projects(`results/advanced_benchmark/report.md`) that names the winning method (currently

CMD-ECS) and highlights where Euclidean baselines fail.

### [Phononic Materials for Analog Gravity](projects/phononic-discovery/)

### 6. Generate novel molecules

Discovery of materials where phonons exhibit relativistic dispersion and effective spacetime curvature—creating laboratory analogs of gravitational physics.

`06_generate_molecules.py` implements full CMD-ECS inference to generate

**Status**: Active  molecular geometries from scratch. It uses the trained score and surrogate

**Key Result**: CrCuSe₂ hetero-metallic TMD discovered and validated  models to perform reverse diffusion, optionally with energy-based guidance.

**Collaboration**: Université Le Mans (LAUM) - Acoustics Laboratory

```bash

<details>cd /workspaces/algo/qcmd_hybrid_framework

<summary><b>Project Highlights</b></summary>python scripts/06_generate_molecules.py \

  --num-samples 20 \

<br>  --num-steps 50 \

  --noise-scale 0.3 \

**Objectives**:  --gamma 0.1 \

- Discover materials with Dirac/Weyl phonon topology  --score-model-path models/score_model/score_model_state_dict.pt \

- Design strain-tunable phononic systems (effective curved spacetime)  --surrogate-path models/surrogate/surrogate_state_dict.pt \

- Engineer synthesis protocols via computational prediction  --output-dir results/generated_molecules

```

**Deliverables**:

- AI-generated candidate structures with target propertiesThe script generates:

- DFT-validated stability and electronic structure- XYZ files for initial and final molecular geometries

- MBE growth protocols (temperature, substrate, flux ratios)- An HTML visualization report with statistics

- Experimental collaboration proposal- Energy predictions from the surrogate model



**Impact**:Use `--gamma > 0` to enable MAECS energy steering, which biases generation

- First tunable quantum-gravity testbed at room temperaturetoward low-energy configurations.

- Applications in topological thermal management

- Foundation for phononic quantum computing isolation## Customising training data



</details>To swap the default QM9 micro-sample for your own molecules while preserving the

workflow:

---

1. **Dataset preparation** – tweak `scripts/01_prepare_data.py` to point at your

## Core Framework  source dataset or loader. The script expects PyTorch Geometric `Data` objects

  with atomic numbers (`z`) and coordinates (`pos`).

### Stiefel Manifold Operations2. **Manifold enrichment** – adjust `scripts/02_enrich_dataset.py` so that it

  computes—or imports—energies, forces, and orthonormal frames for your

The mathematical foundation enabling physics-aware structure generation:  structures. Downstream stages rely on the keys `atom_types`, `pos`,

  `forces_ev_per_angstrom` (or `gradient_hartree_per_bohr`), `orbitals`, and the

```python  `manifold_frame` dictionary containing `frame`, `centroid`, `mass_weights`, and

from core.qcmd_ecs.core import manifold, dynamics  `rank`.

3. **Configuration** – pass the new artifact path via each script’s

# Project gradient to tangent space (preserve orthonormality)  `--dataset-path`. All intermediate files live under `data/`, so you can keep

tangent_vector = manifold.project_to_tangent_space(U, gradient)  multiple datasets side by side.



# Retract to manifold (geodesic step)> **Tip:** if new atomic species appear, extend `DEFAULT_TYPE_NAMES` and

U_new = manifold.retract_to_manifold(U + step_size * tangent_vector)> `DEFAULT_ATOMIC_NUMBERS` inside `models/nequip_common.py`, and add their masses

> to `manifold_utils.ATOMIC_MASSES_AMU` before re-running enrichment.

# Run reverse diffusion (generative sampling)

structures = dynamics.run_reverse_diffusion(## Fine-tuning the neural components

    score_model=score_model,

    energy_model=surrogate,Both neural models share helper utilities in `models/nequip_common.py`, making

    n_samples=100fine-tuning consistent:

)

```- **Surrogate:** re-run `scripts/03_train_surrogate.py` with altered hyperparameters

  (batch size, learning rate, epochs) or a different target field using

**Key Properties**:  `--target`. The script saves the best validation checkpoint to

- Preserves orthonormality constraints ($U^T U = I$)  `models/surrogate/surrogate_state_dict.pt`; you can resume training by loading

- Enforces physical symmetries (translation, rotation, permutation)  the state dict manually and continuing the loop.

- Enables gradient-based optimization on curved manifolds- **Score model:** the NequIP-based score predictor in `models/score_model.py`

  mirrors the surrogate’s setup. Wrap diffusion training samples in a PyG loader,

See [Architecture Documentation](docs/architecture/) for mathematical details.  feed batches through the model, and optimise against force/score supervision in

  float64. Once weights are learned, export them alongside the surrogate so

---  inference can consume both modules.



## DocumentationAfter any modification, validate core manifold operations with:



| Resource | Description |```bash

|----------|-------------|cd /workspaces/algo/qcmd_hybrid_framework

| [Architecture Overview](docs/architecture/OVERVIEW.md) | System design and mathematical framework |pytest -q qcmd_ecs/tests/test_core.py

| [Quick Start Guide](docs/guides/QUICK_START.md) | Tutorial for new users |```

| [CrCuSe₂ Discovery](docs/discoveries/CrCuSe2/) | Complete discovery documentation |

| [Synthesis Lab](projects/phononic-discovery/framework/synthesis_lab/) | MBE protocol design tools |## CMD-ECS inference workflow

| [Contributing](docs/guides/CONTRIBUTING.md) | Guidelines for contributors |

When a trained score model is available, molecule generation proceeds as:

---

1. Load the surrogate and score checkpoints (ensure they run in `torch.float64`).

## Repository Structure2. Convert the dataset’s `manifold_frame` for the target molecule into a Stiefel

  state `U_T`, then inject noise matching your diffusion schedule.

```3. Call `qcmd_ecs.core.dynamics.run_reverse_diffusion` with:

algo/  - the noisy Stiefel state,

├── core/                           # Shared infrastructure  - your score model `(U_t, t) -> score`,

│   ├── qcmd_ecs/                  # Manifold operations  - an energy-gradient callback derived from the surrogate,

│   ├── models/                    # Neural architectures  - gamma/eta/tau schedules,

│   └── legacy_models/             # Original implementations  - and an optional callback for logging or assertion checks.

├── projects/4. Reconstruct Cartesian coordinates via the mass-weighted frame (see

│   └── phononic-discovery/        # Active research project  `frame_to_positions` inside `scripts/05_advanced_benchmark.py`), export to XYZ,

│       └── framework/  and post-process with RDKit to validate chemistry.

│           ├── scripts/           # Discovery pipeline

│           ├── dft_validation/    # Quantum chemistry validationThe advanced benchmark script implements steps 2–4 with oracle scores; replacing

│           ├── synthesis_lab/     # Experimental designthe oracle with your trained score model turns the setup into full-fledged

│           └── results/           # Discovered materialsinference for CMD-ECS.

├── docs/                          # Documentation

│   ├── architecture/              # Technical specifications## Next steps

│   ├── guides/                    # User tutorials

│   └── discoveries/               # Material reportsThis README will stay the blueprint as the architecture evolves. Completed features:

└── requirements.txt               # Dependencies

```- ✅ Training the score model to replace oracle gradients in benchmarking and inference

- ✅ Full molecule generation pipeline with XYZ export and HTML visualization

---- ✅ MAECS energy-guided diffusion with configurable gamma parameter



## CitationFuture work includes:



If you use this work in your research, please cite:- Scaling to larger datasets (10k, 50k molecules) for improved model accuracy

- Advanced score model architectures (SE(3)-equivariant, attention mechanisms)

```bibtex- Exploring schedule sweeps and alternative noise regimes for CMD-ECS

@misc{koussai2025qcmd,- Expanding benchmarks with additional baselines and molecular property prediction

  title={AI-Driven Discovery of Quantum Materials via Manifold-Constrained Diffusion},- Integration with molecular docking and optimization workflows

  author={Koussai Salem},

  year={2025},Every new stage will ship with a script under `scripts/` and an accompanying

  publisher={GitHub},documentation update here.

  howpublished={\url{https://github.com/Koussaisalem/algo}}
}
```

**Discoveries**:
```bibtex
@article{koussai2025crcuse2,
  title={CrCuSe₂: A Novel Hetero-Metallic 2D Semiconductor with Topological Phonon Modes},
  author={Koussai Salem},
  journal={In preparation},
  year={2025}
}
```

---

## Collaboration

We welcome collaborations in:
- **Experimental Synthesis**: MBE/CVD growth of discovered materials
- **Characterization**: Raman, XRD, ARPES, transport measurements
- **Theory**: Phonon topology, analog gravity, quantum field theory

**Current Partnerships**:
- Université Le Mans (LAUM) - Phononic materials and acoustics

For collaboration inquiries, please open an issue or contact via email.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><sub>Built with rigor. Driven by discovery.</sub></p>
  <p>
    <a href="https://github.com/Koussaisalem/algo/issues">Report Bug</a> •
    <a href="https://github.com/Koussaisalem/algo/issues">Request Feature</a> •
    <a href="https://github.com/Koussaisalem/algo/discussions">Discussions</a>
  </p>
</div>
