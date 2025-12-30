# Docker Development Setup

## What's Updated

- **Python**: Upgraded to 3.11 (from 3.8/3.10)
- **PyTorch**: 2.5.1 with CUDA 12.4 support
- **Packages**: All latest versions including:
  - torchgeo 
  - lightning-uq-box
  - pytorch-lightning
  - torchmetrics
  - wandb (for experiment tracking)
  - timm, kornia, seaborn

## How to Use

### Option 1: Using the helper script (Easiest)
```bash
./run_in_docker.sh experiments/test.py
```

### Option 2: Using VSCode Tasks
1. Press `Ctrl+Shift+P`
2. Type "Run Task"
3. Select "Run Python in Docker"
   - This will run the currently open Python file in Docker

### Option 3: Manual Docker commands
```bash
# Start the container (first time)
docker stop yesong && docker rm yesong
docker run -d --name yesong --gpus all \
  -v $(pwd):/workspace \
  my-paper-env:latest tail -f /dev/null

# Run a script
docker exec -it yesong python /workspace/experiments/test.py

# Interactive Python
docker exec -it yesong python

# Bash shell
docker exec -it yesong bash
```

## Rebuild After Dockerfile Changes

```bash
docker build -t my-paper-env:latest .
docker stop yesong && docker rm yesong
docker run -d --name yesong --gpus all \
  -v $(pwd):/workspace \
  my-paper-env:latest tail -f /dev/null
```

## Why Docker?

- ✅ **Consistent environment**: Same Python version, same packages everywhere
- ✅ **No conflicts**: Your system Python stays clean
- ✅ **Reproducible**: Share Dockerfile with collaborators
- ✅ **GPU support**: CUDA drivers managed by Docker
- ✅ **Easy cleanup**: Delete container, no traces left
