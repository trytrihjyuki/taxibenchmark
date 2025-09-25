#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Default values
IMAGE_NAME="taxi-benchmark"
IMAGE_TAG="latest"
CONTAINER_NAME="taxi-exp-$(date +%Y%m%d-%H%M%S)"

# Show usage
if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ $# -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  🚕 Taxi Benchmark - Run Experiments${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required Options:"
    echo "  --processing-date DATE    Date to process (YYYY-MM-DD)"
    echo ""
    echo "Optional Options:"
    echo "  --vehicle-type TYPE       Vehicle type (green/yellow, default: green)"
    echo "  --boroughs BOROUGH...     Boroughs (default: Manhattan)"
    echo "  --methods METHOD...       Methods (default: MinMaxCostFlow MAPS)"
    echo "  --num-iter N             Number of iterations (default: 100)"
    echo "  --start-hour H           Start hour 0-23 (default: 10)"
    echo "  --end-hour H             End hour 0-23 (default: 11)"
    echo "  --time-delta M           Interval between window starts in minutes (default: 5)"
    echo "  --time-window-size S     Window duration in seconds (default: 30)"
    echo "  --num-workers N          Parallel workers (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0 --processing-date 2019-10-06"
    echo "  $0 --processing-date 2019-10-06 --boroughs Manhattan Brooklyn --methods LP MinMaxCostFlow MAPS LinUCB"
    echo "  $0 --processing-date 2019-10-06 --time-window-size 300  # 5-minute windows"
    echo "  $0 --processing-date 2019-10-06 --time-window-size 30   # 30-second windows"
    echo ""
    exit 0
fi

# Check if Docker image exists
if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker image not found. Run ./build.sh first${NC}"
    exit 1
fi

# Load environment variables if available
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run container
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Starting Experiment${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "${YELLOW}Args: $@${NC}"
echo ""

# Create experiments directory on host
mkdir -p experiments

# Run Docker container with volume mount for results
docker run \
    --rm \
    --name ${CONTAINER_NAME} \
    -v $(pwd)/experiments:/app/experiments \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_REGION="${AWS_REGION:-eu-north-1}" \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    python run_experiment.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Experiment completed successfully!${NC}"
    echo -e "${GREEN}Results saved in: experiments/${NC}"
else
    echo ""
    echo -e "${RED}❌ Experiment failed (exit code: $EXIT_CODE)${NC}"
fi

exit $EXIT_CODE 