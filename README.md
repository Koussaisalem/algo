<div align="center">

# Quantum Materials Discovery Platform

### *AI-driven discovery of materials with exotic quantum properties*

<p>
  <a href="https://github.com/Koussaisalem/algo/actions">
    <img src="https://img.shields.io/badge/tests-passing-brightgreen?style=for-the-badge&logo=github" alt="Tests"/>
  </a>
  <a href="https://github.com/Koussaisalem/algo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Proprietary-red.svg?style=for-the-badge" alt="License: Proprietary"/>
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://github.com/Koussaisalem/algo/stargazers">
    <img src="https://img.shields.io/github/stars/Koussaisalem/algo?style=for-the-badge&logo=github&color=yellow" alt="Stars"/>
  </a>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#recent-discoveries">Discoveries</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a>
</p>

</div>

---

## Overview

This repository contains a unified platform for discovering novel quantum materials through AI-driven generative modeling, DFT validation, and synthesis planning. The platform combines manifold-constrained diffusion models with high-throughput computational screening to identify materials with exotic properties.

**Key Features:**
- 🎯 Manifold-constrained generative models (Stiefel manifold diffusion)
- 🔬 Multi-scale validation pipeline (xTB → DFT → phonons)
- 🧪 Synthesis protocol design (MBE temperature screening)
- 📊 Advanced benchmarking and analysis tools
- 🤝 Collaboration-ready documentation and workflows

---

## Recent Discoveries

<details open>
<summary><b>🔬 CrCuSe₂ - Hetero-Metallic 2D Semiconductor</b></summary>

<br>

<table>
<tr>
<th>Property</th>
<th>Value</th>
<th>Significance</th>
</tr>
<tr>
<td><b>Structure</b></td>
<td>P 1 space group, layered 2D</td>
<td>Novel magnetic TMD</td>
</tr>
<tr>
<td><b>Band Gap</b></td>
<td>0.616 eV (indirect)</td>
<td>Ideal for electronics</td>
</tr>
<tr>
<td><b>Stability</b></td>
<td>0 imaginary phonons</td>
<td>Thermodynamically stable</td>
</tr>
<tr>
<td><b>Validation</b></td>
<td>xTB + GPAW DFT + Consultant</td>
<td>97% accuracy confirmed</td>
</tr>
<tr>
<td><b>Synthesis</b></td>
<td>MBE at 450-550°C</td>
<td>Experimentally feasible</td>
</tr>
</table>

<br>

**Status:** Ready for experimental validation • [Full Discovery Report →](docs/discoveries/CrCuSe2/DISCOVERY.md)

</details>

---

## Repository Structure

```
algo/
├── core/                          # Shared infrastructure
│   ├── qcmd_ecs/                 # Stiefel manifold framework
│   │   ├── manifold.py           # Tangent projection & retraction
│   │   ├── dynamics.py           # Reverse diffusion sampling
│   │   └── types.py              # Precision constants
│   ├── models/                   # Neural architectures
│   │   ├── score_model.py        # Diffusion score network
│   │   ├── surrogate.py          # GNN energy predictor
│   │   └── tmd_surrogate.py      # TMD-specific model
│   └── legacy_models/            # Original implementations
│
├── projects/                     # Research projects
│   └── phononic-discovery/       # Active: Phononic analog gravity
│       ├── scripts/              # Discovery pipeline (10+ scripts)
│       ├── dft_validation/       # GPAW validation workflow
│       ├── synthesis_lab/        # MBE protocol design
│       └── results/              # Generated structures & analysis
│
└── docs/                         # Documentation
    ├── architecture/             # System design & specs
    ├── discoveries/              # Material discovery reports
    └── guides/                   # User & developer guides
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Koussaisalem/algo.git
cd algo

# Create environment
conda create -n qcmd python=3.10
conda activate qcmd

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse
pip install schnetpack ase gpaw rdkit
pip install -e .
```

### Run Discovery Pipeline

```bash
cd projects/phononic-discovery/framework/scripts

# 1. Prepare dataset (QM9 subset)
python 01_prepare_data.py

# 2. Enrich with xTB calculations
python 02_enrich_dataset.py --input_path ../data/qm9_micro_5k.pt

# 3. Train surrogate energy model
python 03_train_surrogate.py

# 4. Run advanced benchmarks
python 05_advanced_benchmark.py
```

---

## Core Framework

### Stiefel Manifold Diffusion

The core engine implements diffusion on the Stiefel manifold St(m,k) for generating molecular orbital configurations:

```python
from core.qcmd_ecs.core.dynamics import run_reverse_diffusion
from core.qcmd_ecs.core.manifold import project_to_tangent_space, retract_to_manifold

# Define score and energy models
def score_fn(t: int, U: Tensor) -> Tensor:
    return score_model(U, t)

def energy_fn(U: Tensor) -> Tensor:
    return surrogate_model(U)

# Run diffusion sampling
samples = run_reverse_diffusion(
    score_fn=score_fn,
    energy_fn=energy_fn,
    num_steps=1000,
    shape=(num_atoms, 3),
    beta_schedule='cosine'
)
```

### Key Operations

- **Tangent Projection:** Ensures gradients respect manifold constraints
- **Retraction:** Maps tangent vectors back to manifold via QR decomposition
- **Energy Guidance:** Incorporates surrogate energy for biased sampling

**Performance:** Double precision (`torch.float64`) with rigorous orthonormality checks (tolerance: 1e-9)

---

## Projects

### 1. Phononic Discovery

**Objective:** Discover materials with Dirac/Weyl phonon band structures for analog gravity experiments

**Status:** Active discovery phase
- ✅ CrCuSe₂ discovered and validated
- ⏳ Temperature screening via AIMD
- 📝 Collaboration proposal for Université Le Mans

**Read more:** [Project README](projects/phononic-discovery/README.md)

### 2. Foundation Model (Planned)

**Objective:** Multi-domain foundation model covering semiconductors, structural materials, acoustics, and topological phases

**Status:** Data curation planning
- Strategy: Hybrid storage (code in git, data external)
- Sources: Materials Project, OQMD, C2DB, custom DFPT
- Training: Cามber GPU credits + HPC cluster

---

## Documentation

### 📚 Available Documentation

| Document | Description | Status |
|----------|-------------|--------|
| [Architecture Overview](docs/architecture/OVERVIEW.md) | System design & technical specs | ✅ Complete |
| [CrCuSe₂ Discovery Report](docs/discoveries/CrCuSe2/DISCOVERY.md) | Comprehensive discovery documentation | ✅ Complete |
| [README Styling Guide](docs/guides/README_STYLING_OPTIONS.md) | Professional README formatting options | ✅ Complete |
| [Phononic Discovery Project](projects/phononic-discovery/README.md) | Project-specific guide | ✅ Complete |
| [Restructure Summary](RESTRUCTURE_SUMMARY.md) | Repository transformation log | ✅ Complete |

### 🔧 Developer Resources

- **Core Framework:** See `core/qcmd_ecs/` for manifold operations
- **Models:** See `core/models/` for neural architectures
- **Scripts:** See `projects/phononic-discovery/framework/scripts/` for pipeline
- **Tests:** Run `pytest core/qcmd_ecs/tests/` for validation

---

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{qcmd_platform_2025,
  title = {Quantum Materials Discovery Platform},
  author = {Koussai Salem},
  year = {2025},
  url = {https://github.com/Koussaisalem/algo}
}
```

For the CrCuSe₂ discovery:

```bibtex
@article{crcuse2_discovery_2025,
  title = {AI-Driven Discovery of CrCuSe₂: A Hetero-Metallic 2D Semiconductor},
  author = {Koussai Salem},
  journal = {In preparation},
  year = {2025}
}
```

---

## Collaboration

We welcome collaborations on:
- **Experimental validation** of discovered materials
- **Phononic materials** for analog gravity research
- **Foundation model training** with domain-specific datasets
- **Synthesis protocol optimization** via AIMD/DFT

**Contact:** Open an issue or reach out via [GitHub Discussions](https://github.com/Koussaisalem/algo/discussions)

**Academic Partners:**
- Université Le Mans (LAUM) - Phononic materials and acoustics

---

## License

This project is proprietary and confidential software - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Frameworks:** PyTorch, PyTorch Geometric, SchNetPack, ASE, GPAW
- **Data Sources:** QM9, Materials Project, OQMD
- **Compute:** GitHub Codespaces, Cามber (via GitHub Education)
- **Inspiration:** Analog gravity research community

---

<div align="center">
  <sub>Built with ❤️ for the quantum materials community</sub>
</div>
