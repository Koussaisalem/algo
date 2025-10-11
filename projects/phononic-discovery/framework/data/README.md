# Data Directory

This directory contains datasets used for training and validation. Large data files are **not tracked in git** - they should be stored locally or in external storage.

---

## Data Files (Not in Repository)

The following large files are excluded from git tracking:

### QM9 Dataset
- `qm9/raw/gdb9.sdf` (224 MB) - Original QM9 structures
- `qm9/raw/gdb9.sdf.csv` (26 MB) - QM9 molecular properties
- `qm9/processed/data_v3.pt` (varies) - Processed PyTorch Geometric data

### Enriched Datasets
- `qm9_micro_5k_enriched.pt` - xTB-enriched subset (formation energies)
- `qm9_micro_5k.pt` - Original micro dataset
- `*.pt.bak` - Backup files

---

## Obtaining Data

### QM9 Dataset

**Download automatically:**
```python
from torch_geometric.datasets import QM9
dataset = QM9(root='projects/phononic-discovery/framework/data/qm9')
```

**Or download manually:**
- Source: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
- Files: `gdb9.sdf` and `gdb9.sdf.csv`
- Place in: `projects/phononic-discovery/framework/data/qm9/raw/`

### Enriched Datasets

**Generate via pipeline:**
```bash
cd projects/phononic-discovery/framework/scripts

# Prepare base dataset
python 01_prepare_data.py

# Enrich with xTB calculations
python 02_enrich_dataset.py --input_path ../data/qm9_micro_5k.pt
```

---

## Data Storage Best Practices

### For Local Development
- Keep data files in this directory
- They're ignored by git (see `.gitignore`)
- Share via external storage if needed

### For Collaboration
- **Small datasets** (<10 MB): Can commit to git
- **Medium datasets** (10-100 MB): Use Git LFS
- **Large datasets** (>100 MB): External storage (S3, institutional server)

### Recommended External Storage
- AWS S3 (with academic credits)
- Institutional HPC storage
- Zenodo (for published datasets)
- Materials Project API (regenerate on demand)

---

## Current Dataset Status

### Tracked in Git
- `tmd/dataset_summary.json` - Metadata only
- `qm9_micro_5k_enriched.failures.json` - Failure log
- Small metadata files

### Locally Available (Not Tracked)
- QM9 raw data (if downloaded)
- Processed datasets
- Training checkpoints

---

## Reproducing Results

To reproduce published results:

1. **Download QM9**: Run PyTorch Geometric download (automated)
2. **Enrich dataset**: Run scripts 01-02 (generates formation energies)
3. **Train models**: Run scripts 03-04 (creates surrogates)
4. **Generate structures**: Run script 06 (produces candidates)

All scripts handle data caching automatically.

---

## Data Access for External Users

If you need access to our processed datasets:

1. Check if data is on Materials Project (public)
2. Request via GitHub issue (we can share via Zenodo)
3. Regenerate using our scripts (fully reproducible)

---

<div align="center">
  <p><sub>Data management following open science best practices</sub></p>
</div>
