#!/bin/bash
set -e

IMAGE_NAME="my-paper-env:latest"
CONTAINER_NAME="yesong"

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Docker image ${IMAGE_NAME} not found."
    echo "Build it first with: docker build -t ${IMAGE_NAME} ."
    exit 1
fi

# Ensure container is running
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Starting Docker container..."
    docker start "${CONTAINER_NAME}" 2>/dev/null || \
    docker run -d --name "${CONTAINER_NAME}" --gpus all --ipc host \
        -v "$(pwd)":/workspace \
        "${IMAGE_NAME}" tail -f /dev/null
    sleep 2
fi

# Run the Python script
if [ -z "$1" ]; then
    echo "Usage: ./run_in_docker.sh <python_file>"
    echo "Example: ./run_in_docker.sh experiments/test.py"
    exit 1
fi

echo "Running $1 in Docker..."
docker exec -it "${CONTAINER_NAME}" python /workspace/$1
