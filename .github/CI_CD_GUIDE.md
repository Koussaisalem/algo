# CI/CD Pipeline Documentation

## Overview

This repository includes a comprehensive CI/CD pipeline with GitHub Actions workflows that ensure code quality, test coverage, and reproducibility.

## Workflows

### 1. **Tests** (`.github/workflows/test.yml`)

**Triggers:** Push/PR to `main` or `develop` branches

**Jobs:**
- **Unit Tests**: Runs pytest on core manifold operations
- **Integration Tests**: Validates end-to-end diffusion pipeline
- **Style Checks**: Black, isort, flake8 (non-blocking)
- **Coverage Upload**: Sends coverage to Codecov

**Matrix Testing:**
- Python 3.10, 3.11
- Ensures compatibility across versions

### 2. **Weekly Benchmarks** (`.github/workflows/benchmark.yml`)

**Triggers:** Every Monday at 3 AM UTC (cron) + manual

**Jobs:**
- **Manifold Accuracy**: Tests orthogonality preservation across sizes
- **Diffusion Convergence**: Validates energy optimization
- **Model Import Check**: Ensures all architectures load
- **Report Generation**: Creates benchmark artifacts

**Metrics Tracked:**
- Projection/retraction speed (ms)
- Orthogonality error (< 1e-9)
- Energy convergence

### 3. **Dependency Review** (`.github/workflows/dependency-review.yml`)

**Triggers:** Pull requests to `main`

**Jobs:**
- **Security Scan**: pip-audit for vulnerabilities
- **Conflict Detection**: Checks for version conflicts
- **PyTorch Compatibility**: Verifies torch + PyG integration

### 4. **Documentation** (`.github/workflows/docs.yml`)

**Triggers:** Push to docs/ or README.md

**Jobs:**
- **Link Validation**: Checks for broken markdown links
- **Structure Check**: Ensures required docs exist
- **Code Examples**: Validates Python syntax in docs

### 5. **Release** (`.github/workflows/release.yml`)

**Triggers:** Version tags (`v*.*.*`) + manual

**Jobs:**
- **Full Test Suite**: Comprehensive validation
- **Archive Creation**: Bundles core, tools, docs
- **GitHub Release**: Creates release with artifacts
- **Docker Build**: Containerized distribution

## Local Testing

### Quick Validation
```bash
# Make script executable
chmod +x scripts/validate_ci.sh

# Run local checks
./scripts/validate_ci.sh
```

### Run Tests Locally
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run full test suite
pytest core/qcmd_ecs/tests/ -v --cov=core/qcmd_ecs/core

# Run specific test
pytest core/qcmd_ecs/tests/test_core.py::TestQCMDECSMechanics::test_tangent_space_projection -v

# Run with coverage report
pytest --cov=core --cov-report=html
open htmlcov/index.html
```

### Code Quality Checks
```bash
# Format code
black core/ --line-length 100

# Sort imports
isort core/ --profile black

# Lint
flake8 core/ --max-line-length=100 --extend-ignore=E203,E501,W503
```

## Status Badges

Add these to your README.md:

```markdown
![Tests](https://github.com/Koussaisalem/algo/actions/workflows/test.yml/badge.svg)
![Benchmarks](https://github.com/Koussaisalem/algo/actions/workflows/benchmark.yml/badge.svg)
[![codecov](https://codecov.io/gh/Koussaisalem/algo/branch/main/graph/badge.svg)](https://codecov.io/gh/Koussaisalem/algo)
```

## Continuous Monitoring

### View Results
- **Actions Tab**: https://github.com/Koussaisalem/algo/actions
- **Coverage**: https://codecov.io/gh/Koussaisalem/algo
- **Benchmark History**: Artifacts section in Actions

### Notifications
- Failed workflows trigger GitHub notifications
- Configure Slack/email in repository settings

## Best Practices

### Before Committing
1. Run local tests: `pytest core/qcmd_ecs/tests/ -v`
2. Format code: `black core/ && isort core/`
3. Check for errors: `flake8 core/`

### Pull Requests
- All tests must pass (green checkmarks)
- Dependency review must succeed
- Maintain coverage > 80% (target)

### Releases
1. Ensure all tests pass
2. Update version in relevant files
3. Create tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. GitHub Actions creates release automatically

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure dependencies installed
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Coverage Not Uploading:**
- Check Codecov token in repository secrets
- Ensure coverage.xml generated

**Workflow Not Triggering:**
- Check branch names match workflow triggers
- Verify YAML syntax is correct

## Future Enhancements

- [ ] Add performance regression detection
- [ ] Integrate DFT validation benchmarks
- [ ] Add model accuracy tracking (MAE vs. MP/QM9)
- [ ] Deploy documentation site (GitHub Pages)
- [ ] Add GPU-enabled test runners

## References

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Integration](https://about.codecov.io/)
