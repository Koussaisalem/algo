# Nextflow Pipeline

This directory contains the Nextflow implementation of the end-to-end quantum materials discovery pipeline.

## Overview

The pipeline automates the complete workflow from data preparation to synthesis design:

1. **Data Preparation**: Extract and preprocess molecular datasets
2. **xTB Enrichment**: Compute formation energies using GFN2-xTB
3. **Surrogate Training**: Train GNN-based energy surrogate models
4. **Score Training**: Train diffusion score models
5. **Generative Sampling**: Generate novel molecular structures using Stiefel manifold diffusion
6. **DFT Validation**: Validate structures with GPAW quantum chemistry calculations
7. **Analysis & Visualization**: Generate publication-ready figures and reports
8. **Benchmarking**: Compare manifold vs Euclidean approaches

## Quick Start

### Prerequisites

- Nextflow (>=21.04.0)
- Python 3.10+
- PyTorch with CUDA support
- All project dependencies installed

### Running the Pipeline

```bash
# Basic run with default parameters
nextflow run main.nf

# Run with custom parameters
nextflow run main.nf \
    --raw_dataset /path/to/dataset.pt \
    --output_dir results/my_run \
    --num_samples 100

# Run with test profile (faster, for development)
nextflow run main.nf -profile test

# Run with Docker
nextflow run main.nf -profile docker

# Run with Conda environment
nextflow run main.nf -profile conda

# Run on HPC cluster
nextflow run main.nf -profile cluster
```

## Configuration

Edit `nextflow.config` to customize:

- **Resources**: CPU, memory, time limits per process
- **Parameters**: Training epochs, sample counts, paths
- **Profiles**: Execution environments (local, docker, conda, cluster)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `raw_dataset` | `data/qm9/qm9_micro_raw.pt` | Input dataset path |
| `output_dir` | `results/pipeline_output` | Output directory |
| `num_epochs_surrogate` | 50 | Surrogate training epochs |
| `num_epochs_score` | 100 | Score model training epochs |
| `num_samples` | 100 | Number of molecules to generate |

## Profiles

- **standard**: Local execution (default)
- **docker**: Run in Docker container with GPU support
- **conda**: Use Conda environment
- **cluster**: Submit to SLURM cluster
- **test**: Fast testing with reduced parameters

## Pipeline Outputs

The pipeline generates:

```
results/pipeline_output/
├── data/
│   └── processed_dataset.pt
├── enriched/
│   └── enriched_dataset.pt
├── models/
│   ├── surrogate/
│   │   ├── surrogate_model.pt
│   │   └── training_metrics.json
│   └── score/
│       ├── score_model.pt
│       └── training_metrics.json
├── generated/
│   └── generated_molecules/*.xyz
├── validated/
│   ├── validation_results/*.json
│   └── validated_structures/*.xyz
├── analysis/
│   ├── figures/*.png
│   └── analysis_report.html
├── benchmark/
│   ├── benchmark_results.json
│   └── benchmark_plots/*.png
├── timeline.html
├── report.html
├── trace.txt
└── dag.svg
```

## Monitoring

Nextflow generates several reports:

- **timeline.html**: Execution timeline
- **report.html**: Resource usage report
- **trace.txt**: Detailed execution trace
- **dag.svg**: Pipeline dependency graph

## Resume Failed Runs

If a pipeline fails, resume from the last successful checkpoint:

```bash
nextflow run main.nf -resume
```

## Examples

### Quick Test Run

```bash
# Fast test with minimal resources
nextflow run main.nf -profile test --num_samples 10
```

### Production Run

```bash
# Full pipeline with optimal settings
nextflow run main.nf \
    --num_epochs_surrogate 100 \
    --num_epochs_score 200 \
    --num_samples 1000 \
    --output_dir results/production_run
```

### HPC Cluster Run

```bash
# Submit to SLURM cluster
nextflow run main.nf -profile cluster \
    --output_dir /scratch/user/algo_run
```

## Troubleshooting

### Pipeline Fails

- Check logs in `work/` directory
- View detailed trace: `cat results/pipeline_output/trace.txt`
- Resume from checkpoint: `nextflow run main.nf -resume`

### Out of Memory

- Reduce batch sizes in training scripts
- Adjust memory limits in `nextflow.config`
- Use smaller dataset or fewer samples

### Process Timeout

- Increase time limits in `nextflow.config`
- Use faster hardware (GPU cluster)

## Development

### Adding New Processes

1. Add process definition to `main.nf`
2. Connect process in workflow section
3. Update `nextflow.config` with resource requirements
4. Test with `-profile test`

### Customizing Workflows

Create custom workflow by modifying the `workflow` block in `main.nf`.

## Support

For issues or questions:
- Check documentation: `docs/guides/SCRIPTS_REFERENCE.md`
- Open GitHub issue
- Contact: Koussai Salem

## License

See main repository LICENSE file.
