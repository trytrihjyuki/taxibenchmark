#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  🚕 Taxi Benchmark - Docker Build${NC}"
echo -e "${GREEN}========================================${NC}"

# Detect platform
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    echo -e "${BLUE}ℹ Detected ARM architecture (Apple Silicon)${NC}"
    # For local ARM, build native for better performance
    PLATFORM=""
else
    echo -e "${BLUE}ℹ Detected x86_64 architecture${NC}"
    PLATFORM="--platform linux/amd64"
fi

# Build Docker image
IMAGE_NAME="taxi-benchmark"
IMAGE_TAG="${1:-latest}"

echo -e "${YELLOW}Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
if [ -n "$PLATFORM" ]; then
    echo -e "${YELLOW}Platform: linux/amd64${NC}"
fi

docker build \
    ${PLATFORM} \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo -e "${GREEN}Run with: ./run.sh --help${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi 