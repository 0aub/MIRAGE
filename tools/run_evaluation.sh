#!/bin/bash
# MIRAGE Evaluation Test Runner
# Usage: ./run_evaluation.sh [options]
#
# Run from host: ./run_evaluation.sh
# Run inside docker: docker exec mirage-api python /app/tools/evaluation_test_cases.py

set -e

API_URL="${API_URL:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../evaluation_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   MIRAGE Evaluation Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running inside docker or from host
if [ -f /.dockerenv ]; then
    # Running inside docker
    EXEC_PREFIX=""
    INTERNAL_API="http://localhost:8000"
else
    # Running on host - use docker exec
    EXEC_PREFIX="docker exec mirage-api"
    INTERNAL_API="http://localhost:8000"

    # Copy latest test file to container
    echo -e "${YELLOW}Copying evaluation script to container...${NC}"
    docker cp "${SCRIPT_DIR}/evaluation_test_cases.py" mirage-api:/app/tools/evaluation_test_cases.py
fi

# Check if API is available
echo -e "${YELLOW}Checking API availability...${NC}"
if ! curl -s "${API_URL}/db/stats" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: API not available at ${API_URL}${NC}"
    echo "Please ensure the MIRAGE backend is running."
    exit 1
fi
echo -e "${GREEN}API is available${NC}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run evaluation
echo ""
echo -e "${YELLOW}Running evaluation tests...${NC}"
echo ""

OUTPUT_FILE="${OUTPUT_DIR}/evaluation_${TIMESTAMP}.json"
CONTAINER_OUTPUT="/tmp/evaluation_${TIMESTAMP}.json"

${EXEC_PREFIX} python /app/tools/evaluation_test_cases.py \
    --api "${INTERNAL_API}" \
    --modes naive local global hybrid \
    --output "${CONTAINER_OUTPUT}"

# Copy results from container if running on host
if [ -z "$EXEC_PREFIX" ]; then
    cp "${CONTAINER_OUTPUT}" "${OUTPUT_FILE}"
else
    docker cp "mirage-api:${CONTAINER_OUTPUT}" "${OUTPUT_FILE}"
fi

# Check if output was created
if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo -e "${GREEN}Evaluation complete!${NC}"
    echo -e "Results saved to: ${OUTPUT_FILE}"

    # Generate summary
    echo ""
    echo -e "${BLUE}--- QUICK SUMMARY ---${NC}"
    python3 -c "
import json
with open('${OUTPUT_FILE}', 'r') as f:
    data = json.load(f)
    summary = data['summary']
    print(f'Total tests: {summary[\"total_tests\"]}')
    print()
    for mode, stats in summary['mode_scores'].items():
        total = stats['passed'] + stats['failed']
        pct = (stats['passed'] / total * 100) if total > 0 else 0
        print(f'{mode:>8}: {stats[\"passed\"]}/{total} passed ({pct:.0f}%) - avg score: {stats[\"avg_score\"]:.2f}')
"
else
    echo -e "${RED}ERROR: Evaluation failed to produce output${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Done!${NC}"
