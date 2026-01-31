#!/bin/bash
# Quick validation script to check CI/CD setup

set -e

echo "================================================"
echo "CI/CD Setup Validation"
echo "================================================"
echo ""

# Check workflow files
echo "✓ Checking workflow files..."
for workflow in .github/workflows/*.yml; do
    if [ -f "$workflow" ]; then
        echo "  - $(basename $workflow)"
    fi
done
echo ""

# Test Python imports
echo "✓ Testing core imports..."
python -c "
from core.qcmd_ecs.core import manifold, dynamics, types
from core.models import score_model, surrogate
print('  - All imports successful')
"
echo ""

# Run quick test
echo "✓ Running quick manifold test..."
python -c "
import torch
from core.qcmd_ecs.core.manifold import project_to_tangent_space, retract_to_manifold
from core.qcmd_ecs.core.types import DTYPE

m, k = 16, 4
U = torch.randn(m, k, dtype=DTYPE)
U, _ = torch.linalg.qr(U)

gradient = torch.randn(m, k, dtype=DTYPE)
U_tangent = project_to_tangent_space(U, gradient)
U_new = retract_to_manifold(U + 0.01 * U_tangent)

eye_k = torch.eye(k, dtype=DTYPE)
assert torch.allclose(U_new.T @ U_new, eye_k, atol=1e-9)
print('  - Manifold operations validated')
"
echo ""

# Check if pytest works
echo "✓ Checking pytest installation..."
if command -v pytest &> /dev/null; then
    echo "  - pytest installed"
    pytest --version
else
    echo "  ⚠ pytest not found, install with: pip install -r requirements-dev.txt"
fi
echo ""

echo "================================================"
echo "✅ CI/CD setup validation complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Commit and push: git add .github/ && git commit -m 'Add CI/CD workflows'"
echo "  2. GitHub Actions will run automatically on push/PR"
echo "  3. Check: https://github.com/$(git config remote.origin.url | sed 's/.*github.com[:/]\(.*\).git/\1/')/actions"
echo ""
