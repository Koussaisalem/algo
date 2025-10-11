# Quantum Materials Discovery Platform<div align="center"># QCMD Hybrid Framework



> AI-driven discovery of materials with exotic quantum properties



[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square&logo=github)](https://github.com/Koussaisalem/algo/actions)# Quantum Materials Discovery PlatformThe hybrid framework wraps the verified QCMD-ECS manifold engine with data preparation,

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/Koussaisalem/algo/blob/main/LICENSE)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org)enrichment, and model tooling aimed at NequIP-based neural components. Every script in

[![Stars](https://img.shields.io/github/stars/Koussaisalem/algo?style=flat-square&logo=github&color=yellow)](https://github.com/Koussaisalem/algo/stargazers)

<p align="center">this directory assumes double-precision (`torch.float64`) tensors and is compatible with

---

  <i>AI-driven discovery of materials with exotic quantum properties</i>the Stiefel operators in `qcmd_hybrid_framework/qcmd_ecs/core`.

## Overview

</p>

This repository contains a unified platform for discovering novel quantum materials through AI-driven generative modeling, DFT validation, and synthesis planning. The platform combines manifold-constrained diffusion models with high-throughput computational screening to identify materials with exotic properties.

## Directory layout

**Key Features:**

- 🎯 Manifold-constrained generative models (Stiefel manifold diffusion)<p align="center">

- 🔬 Multi-scale validation pipeline (xTB → DFT → phonons)

- 🧪 Synthesis protocol design (MBE temperature screening)  <a href="https://github.com/Koussaisalem/algo/actions">```

- 📊 Advanced benchmarking and analysis tools

- 🤝 Collaboration-ready documentation and workflows    <img src="https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square&logo=github" alt="Tests">qcmd_hybrid_framework/



---  </a>├── data/                 # Cached datasets and enrichment artifacts



## Recent Discoveries  <a href="https://github.com/Koussaisalem/algo/blob/main/LICENSE">├── manifold_utils.py     # Mass-weighted frames and unit conversions



### CrCuSe₂ - Hetero-Metallic 2D Semiconductor    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License">├── models/               # Surrogate and score-model definitions



| Property | Value | Significance |  </a>└── scripts/              # End-to-end pipeline stages

|----------|-------|--------------|

| **Structure** | P 1 space group, layered 2D | Novel magnetic TMD |  <a href="https://github.com/Koussaisalem/algo">    ├── 01_prepare_data.py

| **Band Gap** | 0.616 eV (indirect) | Ideal for electronics |

| **Stability** | 0 imaginary phonons | Thermodynamically stable |    <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python" alt="Python">    ├── 02_enrich_dataset.py

| **Validation** | xTB + GPAW DFT + Consultant | 97% accuracy confirmed |

| **Synthesis** | MBE at 450-550°C | Experimentally feasible |  </a>    ├── 03_train_surrogate.py



**Status:** Ready for experimental validation • [Full Discovery Report →](docs/discoveries/CrCuSe2/DISCOVERY.md)  <a href="https://github.com/Koussaisalem/algo/stargazers">    └── 05_advanced_benchmark.py



---    <img src="https://img.shields.io/github/stars/Koussaisalem/algo?style=flat-square&logo=github&color=yellow" alt="Stars">```



## Repository Structure  </a>



```</p>## Pipeline stages

algo/

├── core/                          # Shared infrastructure

│   ├── qcmd_ecs/                 # Stiefel manifold framework

│   │   ├── manifold.py           # Tangent projection & retraction<p align="center">### 1. Prepare a QM9 micro-dataset

│   │   ├── dynamics.py           # Reverse diffusion sampling

│   │   └── types.py              # Precision constants  <a href="#overview">Overview</a> •

│   ├── models/                   # Neural architectures

│   │   ├── score_model.py        # Diffusion score network  <a href="#key-achievements">Achievements</a> •`01_prepare_data.py` downloads the QM9 dataset with PyTorch Geometric, selects a seeded

│   │   ├── surrogate.py          # GNN energy predictor

│   │   └── tmd_surrogate.py      # TMD-specific model  <a href="#installation">Installation</a> •subset (`NUM_SAMPLES`) of molecules, and stores an `AtomicDataDict`-compatible payload in

│   └── legacy_models/            # Original implementations

│  <a href="#projects">Projects</a> •`data/qm9_micro_5k.pt`.

├── projects/                     # Research projects

│   └── phononic-discovery/       # Active: Phononic analog gravity  <a href="#documentation">Documentation</a> •

│       ├── scripts/              # Discovery pipeline (10+ scripts)

│       ├── dft_validation/       # GPAW validation workflow  <a href="#citation">Citation</a>Run it from the repository root or the `qcmd_hybrid_framework` directory:

│       ├── synthesis_lab/        # MBE protocol design

│       └── results/              # Generated structures & analysis</p>

│

└── docs/                         # Documentation```bash

    ├── architecture/             # System design & specs

    ├── discoveries/              # Material discovery reports</div>cd /workspaces/algo/qcmd_hybrid_framework

    └── guides/                   # User & developer guides

```python scripts/01_prepare_data.py



------```



## Quick Start



### Installation## Overview### 2. Enrich with GFN2-xTB metadata



```bash

# Clone repository

git clone https://github.com/Koussaisalem/algo.gitThis platform combines **manifold-constrained generative AI** with **ab-initio quantum chemistry** to systematically discover materials with targeted topological and electronic properties.`02_enrich_dataset.py` adds physics-aware annotations needed for manifold diffusion:

cd algo



# Create environment

conda create -n qcmd python=3.10**Core Innovation**: Diffusion models operating on the Stiefel manifold, enabling physics-aware crystal structure generation with built-in symmetry constraints and orthonormality preservation.- Single-point GFN2-xTB energy (Hartree and eV)

conda activate qcmd

- Forces converted to eV/Å

# Install dependencies

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118<details>- Molecular orbital coefficients

pip install torch-geometric torch-scatter torch-sparse

pip install schnetpack ase gpaw rdkit<summary><b>Technical Architecture</b> (click to expand)</summary>- Mass-weighted orthonormal frames (via `manifold_utils.compute_manifold_frame`)

pip install -e .

```



### Run Discovery Pipeline<br>The script checkpoints progress and logs failures so long jobs can resume safely.



```bash

cd projects/phononic-discovery/framework/scripts

**Key Components**:```bash

# 1. Prepare dataset (QM9 subset)

python 01_prepare_data.py- **Stiefel Manifold Diffusion**: Constrained generation preserving geometric structurecd /workspaces/algo/qcmd_hybrid_framework



# 2. Enrich with xTB calculations- **Equivariant GNN Surrogates**: Fast property prediction respecting E(3) symmetriespython scripts/02_enrich_dataset.py \

python 02_enrich_dataset.py --input_path ../data/qm9_micro_5k.pt

- **Automated DFT Pipeline**: GPAW-based validation with phonon analysis  --input-path data/qm9_micro_5k.pt \

# 3. Train surrogate energy model

python 03_train_surrogate.py- **Synthesis Design**: MBE/CVD parameter optimization via AIMD  --output-path data/qm9_micro_5k_enriched.pt \



# 4. Generate candidate structures  --checkpoint-every 25 \

python 06_generate_structures.py --num_samples 100

**Workflow**:  --overwrite

# 5. Validate with DFT

cd ../dft_validation``````

python run_gpaw_validation.py --structure ../results/candidate_001.xyz

```Generative Model → Candidate Structures → GNN Surrogate → DFT Validation → Synthesis Protocols



---```Use `--resume` to append to an existing enrichment file, and `--max-molecules` to process



## Core Frameworksmaller batches while debugging.



### Stiefel Manifold Diffusion</details>



The core engine implements diffusion on the Stiefel manifold St(m,k) for generating molecular orbital configurations:### 3. Train the NequIP surrogate



```python---

from core.qcmd_ecs.core.dynamics import run_reverse_diffusion

from core.qcmd_ecs.core.manifold import project_to_tangent_space, retract_to_manifold`03_train_surrogate.py` consumes the enriched dataset to fit the NequIP surrogate defined



# Define score and energy models## Key Achievementsin `models/surrogate.py`. The script performs a reproducible train/val/test split,

def score_fn(t: int, U: Tensor) -> Tensor:

    return score_model(U, t)trains with Adam in double precision, checkpoints the best validation weights, and



def energy_fn(U: Tensor) -> Tensor:<table>writes JSON metrics for future reporting.

    return surrogate_model(U)

  <tr>

# Run diffusion sampling

samples = run_reverse_diffusion(    <td align="center" width="33%">```bash

    score_fn=score_fn,

    energy_fn=energy_fn,      <h3>Novel 2D Material</h3>cd /workspaces/algo/qcmd_hybrid_framework

    num_steps=1000,

    shape=(num_atoms, 3),      <p><b>CrCuSe₂</b></p>python scripts/03_train_surrogate.py \

    beta_schedule='cosine'

)      <p>Hetero-metallic TMD with 0.6 eV bandgap</p>  --dataset-path data/qm9_micro_5k_enriched.pt \

```

      <p><i>Dynamically stable (0 imaginary phonons)</i></p>  --output-dir models/surrogate_runs/default \

### Key Operations

      <a href="docs/discoveries/CrCuSe2/">Read more →</a>  --epochs 100 \

- **Tangent Projection:** Ensures gradients respect manifold constraints

- **Retraction:** Maps tangent vectors back to manifold via QR decomposition    </td>  --batch-size 32 \

- **Energy Guidance:** Incorporates surrogate energy for biased sampling

    <td align="center" width="33%">  --lr 5e-4

**Performance:** Double precision (`torch.float64`) with rigorous orthonormality checks (tolerance: 1e-9)

      <h3>Computational Validation</h3>```

---

      <p><b>Multi-scale verification</b></p>

## Projects

      <p>xTB → DFT → Phonon dispersion</p>The default target is the xTB energy in eV (`energy_ev`). Switch to Hartree supervision

### 1. Phononic Discovery

      <p><i>97% consultant validation accuracy</i></p>with `--target energy_hartree` if you prefer to postpone the unit conversion.

**Objective:** Discover materials with Dirac/Weyl phonon band structures for analog gravity experiments

      <a href="docs/discoveries/CrCuSe2/VALIDATION.md">Validation report →</a>

**Status:** Active discovery phase

- ✅ CrCuSe₂ discovered and validated    </td>### 4. Train the score model

- ⏳ Temperature screening via AIMD

- 📝 Collaboration proposal for Université Le Mans    <td align="center" width="33%">



[Read more →](projects/phononic-discovery/README.md)      <h3>Synthesis Protocol</h3>`04_train_score_model.py` trains a NequIP-based neural network to predict



### 2. Foundation Model (Planned)      <p><b>MBE growth design</b></p>score functions (directions toward the data manifold) from noisy samples.



**Objective:** Multi-domain foundation model covering semiconductors, structural materials, acoustics, and topological phases      <p>Temperature screening via AIMD</p>The model learns to denoise manifold frames using multiple noise scales



**Status:** Data curation planning      <p><i>Optimal conditions computed</i></p>for robust generalization.

- Strategy: Hybrid storage (code in git, data external)

- Sources: Materials Project, OQMD, C2DB, custom DFPT      <a href="projects/phononic-discovery/framework/synthesis_lab/">Protocols →</a>

- Training: Cามber GPU credits + HPC cluster

    </td>```bash

---

  </tr>cd /workspaces/algo/qcmd_hybrid_framework

## Documentation

</table>python scripts/04_train_score_model.py \

### For Users

- [Installation Guide](docs/guides/INSTALLATION.md) - Setup instructions  --dataset-path data/qm9_micro_5k_enriched.pt \

- [Discovery Workflow](docs/guides/WORKFLOW.md) - End-to-end pipeline

- [FAQ](docs/guides/FAQ.md) - Common questions---  --output-dir models/score_model \



### For Developers  --epochs 100 \

- [Architecture Overview](docs/architecture/OVERVIEW.md) - System design

- [API Reference](docs/api/README.md) - Core modules## Installation  --batch-size 16 \

- [Contributing Guide](CONTRIBUTING.md) - Development workflow

  --lr 5e-4 \

### Discovery Reports

- [CrCuSe₂ Discovery](docs/discoveries/CrCuSe2/DISCOVERY.md) - Comprehensive technical report```bash  --noise-levels 0.1 0.2 0.3 0.5

- [Validation Checklist](docs/discoveries/CrCuSe2/VALIDATION.md) - Multi-scale verification

# Clone repository```

---

git clone https://github.com/Koussaisalem/algo.git

## Citation

cd algoThe trained weights are saved to `models/score_model/score_model_state_dict.pt`

If you use this platform in your research, please cite:

along with training metrics. This replaces the oracle score used in earlier

```bibtex

@software{qcmd_platform_2025,# Install dependenciesbenchmarks with a learned neural predictor.

  title = {Quantum Materials Discovery Platform},

  author = {Koussai Salem},pip install -r requirements.txt

  year = {2025},

  url = {https://github.com/Koussaisalem/algo}### 5. Advanced CMD-ECS vs Euclidean benchmarking

}

```# Verify installation (tests coming soon)



For the CrCuSe₂ discovery:python -c "from core.qcmd_ecs.core import manifold; print('Installation successful!')"`05_advanced_benchmark.py` replays the reverse-diffusion process with three



```bibtex```strategies—true CMD-ECS updates, an unconstrained Euclidean walk, and a

@article{crcuse2_discovery_2025,

  title = {AI-Driven Discovery of CrCuSe₂: A Hetero-Metallic 2D Semiconductor},post-hoc retracted Euclidean walk—using oracle scores and the trained surrogate

  author = {Koussai Salem},

  journal = {In preparation},<details>for evaluation. RDKit is used to compare generated geometries against the

  year = {2025}

}<summary><b>Requirements</b></summary>ground-truth frames.

```



---

<br>```bash

## Collaboration

cd /workspaces/algo/qcmd_hybrid_framework

We welcome collaborations on:

- **Experimental validation** of discovered materials- Python ≥ 3.10python scripts/05_advanced_benchmark.py \

- **Phononic materials** for analog gravity research

- **Foundation model training** with domain-specific datasets- PyTorch ≥ 2.0  --num-samples 128 \

- **Synthesis protocol optimization** via AIMD/DFT

- PyTorch Geometric  --num-steps 40 \

**Contact:** Open an issue or reach out via [GitHub Discussions](https://github.com/Koussaisalem/algo/discussions)

- ASE (Atomic Simulation Environment)  --noise-scale 0.2 \

**Academic Partners:**

- Université Le Mans (LAUM) - Phononic materials and acoustics- GPAW (optional, for DFT calculations)  --gamma 0.1 \



---- xTB (optional, for semi-empirical methods)  --surrogate-path models/surrogate/surrogate_state_dict.pt \



## License  --output-dir results/advanced_benchmark



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.See [`requirements.txt`](requirements.txt) for complete dependencies.```



---



## Acknowledgments</details>Setting `--gamma 0.1` activates MAECS (energy-guided diffusion). Use `--gamma 0.0`



- **Frameworks:** PyTorch, PyTorch Geometric, SchNetPack, ASE, GPAWto test pure manifold diffusion without energy steering.

- **Data Sources:** QM9, Materials Project, OQMD

- **Compute:** GitHub Codespaces, Cามber (via GitHub Education)---

- **Inspiration:** Analog gravity research community

Artifacts include per-sample metrics, a summary JSON, and a Markdown report

---

## Projects(`results/advanced_benchmark/report.md`) that names the winning method (currently

<div align="center">

  <sub>Built with ❤️ for the quantum materials community</sub>CMD-ECS) and highlights where Euclidean baselines fail.

</div>

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
