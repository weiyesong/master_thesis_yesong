#!/bin/bash
# Quick script to run Python files in Docker

# Ensure container is running
if ! docker ps | grep -q yesong; then
    echo "Starting Docker container..."
    docker start yesong 2>/dev/null || \
    docker run -d --name yesong --gpus all \
        -v "$(pwd)":/workspace \
        my-paper-env:latest tail -f /dev/null
    sleep 2
fi

# Run the Python script
if [ -z "$1" ]; then
    echo "Usage: ./run_in_docker.sh <python_file>"
    echo "Example: ./run_in_docker.sh experiments/test.py"
    exit 1
fi

echo "Running $1 in Docker..."
docker exec -it yesong python /workspace/$1
