#!/bin/bash
################################################################################
# Parallel DFT Validation Launcher
#
# This script launches 3 independent DFT calculations in parallel,
# utilizing all available CPU cores for maximum speedup.
#
# Expected speedup: 3× faster than serial execution
# Time: ~20-30 min (vs 60-90 min serial)
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🚀 PARALLEL DFT VALIDATION LAUNCHER                         ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║  This will launch 3 independent DFT calculations             ║${NC}"
echo -e "${BLUE}║  in parallel, using 3 of your 4 CPU cores.                   ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║  Structures:                                                  ║${NC}"
echo -e "${BLUE}║    1. Mo₂Te₄ - Novel phase candidate                        ║${NC}"
echo -e "${BLUE}║    2. CrCuSe₂ - Hetero-metallic alloy                       ║${NC}"
echo -e "${BLUE}║    3. VTe₂ - Magnetic TMD                                    ║${NC}"
echo -e "${BLUE}║                                                               ║${NC}"
echo -e "${BLUE}║  Expected time: ~20-30 minutes                                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Navigate to project root
cd /workspaces/algo/qcmd_hybrid_framework

# Create log directory
LOG_DIR="dft_validation/parallel_logs"
mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${YELLOW}📁 Logs will be saved to: $LOG_DIR${NC}"
echo ""

# Structure files
STRUCTURE_1="dft_validation/priority/Mo2Te4_candidate.xyz"
STRUCTURE_2="dft_validation/priority/CrCuSe2_candidate.xyz"
STRUCTURE_3="dft_validation/priority/VTe2_candidate.xyz"

# Check that structures exist
for struct in "$STRUCTURE_1" "$STRUCTURE_2" "$STRUCTURE_3"; do
    if [ ! -f "$struct" ]; then
        echo -e "${RED}❌ ERROR: Structure not found: $struct${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ All 3 structures found${NC}"
echo ""

# Launch parallel jobs
echo -e "${BLUE}🚀 Launching parallel DFT calculations...${NC}"
echo ""

# Job 1: Mo2Te4
echo -e "${YELLOW}[1/3] Launching Mo₂Te₄ validation...${NC}"
python scripts/tmd/05_validate_with_dft.py "$STRUCTURE_1" \
    > "$LOG_DIR/Mo2Te4_${TIMESTAMP}.log" 2>&1 &
PID1=$!
echo -e "      PID: $PID1"

sleep 2  # Small delay to avoid initialization conflicts

# Job 2: CrCuSe2
echo -e "${YELLOW}[2/3] Launching CrCuSe₂ validation...${NC}"
python scripts/tmd/05_validate_with_dft.py "$STRUCTURE_2" \
    > "$LOG_DIR/CrCuSe2_${TIMESTAMP}.log" 2>&1 &
PID2=$!
echo -e "      PID: $PID2"

sleep 2

# Job 3: VTe2
echo -e "${YELLOW}[3/3] Launching VTe₂ validation...${NC}"
python scripts/tmd/05_validate_with_dft.py "$STRUCTURE_3" \
    > "$LOG_DIR/VTe2_${TIMESTAMP}.log" 2>&1 &
PID3=$!
echo -e "      PID: $PID3"

echo ""
echo -e "${GREEN}✅ All 3 jobs launched successfully!${NC}"
echo ""
echo -e "${BLUE}📊 Process IDs:${NC}"
echo -e "   Mo₂Te₄:  $PID1"
echo -e "   CrCuSe₂: $PID2"
echo -e "   VTe₂:    $PID3"
echo ""
echo -e "${YELLOW}⏳ Waiting for all calculations to complete...${NC}"
echo -e "   (This will take ~20-30 minutes)"
echo ""
echo -e "${BLUE}💡 TIP: Monitor progress in another terminal:${NC}"
echo -e "   tail -f $LOG_DIR/Mo2Te4_${TIMESTAMP}.log"
echo -e "   tail -f $LOG_DIR/CrCuSe2_${TIMESTAMP}.log"
echo -e "   tail -f $LOG_DIR/VTe2_${TIMESTAMP}.log"
echo ""

# Wait for all jobs to complete
wait $PID1
STATUS1=$?
echo -e "${GREEN}✅ Mo₂Te₄ completed (exit code: $STATUS1)${NC}"

wait $PID2
STATUS2=$?
echo -e "${GREEN}✅ CrCuSe₂ completed (exit code: $STATUS2)${NC}"

wait $PID3
STATUS3=$?
echo -e "${GREEN}✅ VTe₂ completed (exit code: $STATUS3)${NC}"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🎉 PARALLEL DFT VALIDATION COMPLETE                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check results
RESULTS_DIR="dft_validation/results"
echo -e "${YELLOW}📊 Checking results...${NC}"
echo ""

if [ -f "$RESULTS_DIR/Mo2Te4_candidate_results.json" ]; then
    echo -e "${GREEN}✅ Mo₂Te₄ results saved${NC}"
else
    echo -e "${RED}❌ Mo₂Te₄ results missing (check logs)${NC}"
fi

if [ -f "$RESULTS_DIR/CrCuSe2_candidate_results.json" ]; then
    echo -e "${GREEN}✅ CrCuSe₂ results saved${NC}"
else
    echo -e "${RED}❌ CrCuSe₂ results missing (check logs)${NC}"
fi

if [ -f "$RESULTS_DIR/VTe2_candidate_results.json" ]; then
    echo -e "${GREEN}✅ VTe₂ results saved${NC}"
else
    echo -e "${RED}❌ VTe₂ results missing (check logs)${NC}"
fi

echo ""
echo -e "${BLUE}🔍 Next steps:${NC}"
echo -e "   1. Check results: ls -lh $RESULTS_DIR/"
echo -e "   2. View summary: cat $RESULTS_DIR/validation_summary.json"
echo -e "   3. Visualize structures: view $RESULTS_DIR/*_relaxed.xyz"
echo ""

# Calculate total time
if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ] && [ $STATUS3 -eq 0 ]; then
    echo -e "${GREEN}🎉 All validations succeeded!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some validations failed. Check logs for details.${NC}"
    exit 1
fi
