#!/bin/bash
###############################################################################
# Fast-Track Validation for Mo₂Te₄
# 
# This script implements the two-phase validation strategy:
#   Phase 1: xTB pre-relaxation (~5 min)
#   Phase 2: DFT single-point calculation (~20-30 min)
#
# Total time: ~30-35 minutes
###############################################################################

set -e  # Exit on any error

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../.."

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate qcmd_nequip

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║         🚀 Fast-Track Validation for Mo₂Te₄                 ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║  Phase 1: xTB Pre-Relaxation         (~5 min)               ║${NC}"
echo -e "${BLUE}║  Phase 2: DFT Single-Point           (~20-30 min)           ║${NC}"
echo -e "${BLUE}║                                       ──────────             ║${NC}"
echo -e "${BLUE}║  Total Expected Time:                 ~30-35 min            ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

echo ""

###############################################################################
# Phase 1: xTB Pre-Relaxation
###############################################################################

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Phase 1: xTB Pre-Relaxation (5-Minute Cleanup)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

PHASE1_START=$(date +%s)

if [ ! -f "dft_validation/priority/Mo2Te4_candidate.xyz" ]; then
    echo -e "${RED}❌ ERROR: Mo2Te4_candidate.xyz not found!${NC}"
    echo "Expected location: dft_validation/priority/Mo2Te4_candidate.xyz"
    exit 1
fi

echo "Running xTB pre-relaxation..."
python scripts/tmd/prerelax_mo2te4.py

PHASE1_END=$(date +%s)
PHASE1_DURATION=$((PHASE1_END - PHASE1_START))

if [ ! -f "dft_validation/priority/Mo2Te4_prerelaxed.xyz" ]; then
    echo -e "${RED}❌ Phase 1 FAILED: Pre-relaxed structure not created!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Phase 1 Complete!${NC}"
echo -e "   Duration: ${PHASE1_DURATION} seconds"
echo -e "   Output: dft_validation/priority/Mo2Te4_prerelaxed.xyz"
echo ""

###############################################################################
# Phase 2: DFT Single-Point Calculation
###############################################################################

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Phase 2: DFT Single-Point (30-Minute Litmus Test)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

PHASE2_START=$(date +%s)

echo "Running DFT single-point calculation..."
echo "(This will take 20-30 minutes - be patient!)"
echo ""

python scripts/tmd/05_validate_with_dft.py \
    dft_validation/priority/Mo2Te4_prerelaxed.xyz \
    --mode fast \
    --single-point \
    --force

PHASE2_END=$(date +%s)
PHASE2_DURATION=$((PHASE2_END - PHASE2_START))

echo ""
echo -e "${GREEN}✅ Phase 2 Complete!${NC}"
echo -e "   Duration: ${PHASE2_DURATION} seconds (~$((PHASE2_DURATION / 60)) min)"
echo ""

###############################################################################
# Final Report
###############################################################################

TOTAL_DURATION=$((PHASE1_DURATION + PHASE2_DURATION))

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║                    🎉 VALIDATION COMPLETE                    ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "⏱️  Total Time:"
echo "   Phase 1 (xTB):        ${PHASE1_DURATION}s (~$((PHASE1_DURATION / 60)) min)"
echo "   Phase 2 (DFT):        ${PHASE2_DURATION}s (~$((PHASE2_DURATION / 60)) min)"
echo "   ────────────────────────────────────"
echo "   Total:                ${TOTAL_DURATION}s (~$((TOTAL_DURATION / 60)) min)"
echo ""
echo "📁 Results Location:"
echo "   dft_validation/results/Mo2Te4_prerelaxed_results.json"
echo ""

# Check results file and extract key metrics
RESULTS_FILE="dft_validation/results/Mo2Te4_prerelaxed_results.json"
if [ -f "$RESULTS_FILE" ]; then
    echo "📊 Quick Summary:"
    
    # Extract key metrics using Python
    python3 << EOF
import json
from pathlib import Path

results_file = Path("$RESULTS_FILE")
if results_file.exists():
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    energy = data.get('energy_per_atom_eV', float('nan'))
    converged = data.get('converged', False)
    max_force = data['properties'].get('max_force', float('nan'))
    
    print(f"   Energy per atom:  {energy:.4f} eV")
    print(f"   Max Force:        {max_force:.4f} eV/Å")
    print(f"   Converged:        {'✅ Yes' if converged else '❌ No'}")
    print()
    
    # Stability assessment
    if max_force < 1.5:
        print("   🎯 VERDICT: ✅ GO SIGNAL")
        print("      Structure is STABLE (max force < 1.5 eV/Å)")
        print("      → Safe to proceed with full optimization!")
    elif max_force < 3.0:
        print("   🎯 VERDICT: ⚠️  CAUTION")
        print("      Moderate forces (1.5-3.0 eV/Å)")
        print("      → May need careful optimization settings")
    else:
        print("   🎯 VERDICT: ❌ NO-GO SIGNAL")
        print("      High forces (> 3.0 eV/Å)")
        print("      → Structure likely unstable")
EOF
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Next Steps:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "If GO SIGNAL (✅):"
echo "  Run full optimization:"
echo "  $ python scripts/tmd/05_validate_with_dft.py \\"
echo "      dft_validation/priority/Mo2Te4_prerelaxed.xyz \\"
echo "      --mode production"
echo ""
echo "If NO-GO SIGNAL (❌):"
echo "  Structure is not viable. Consider:"
echo "  - Trying a different generated structure"
echo "  - Adjusting generation parameters"
echo "  - Analyzing what went wrong"
echo ""
