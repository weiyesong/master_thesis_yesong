# Docker Usage For EO UQ Experiments

This project now uses a Docker image based on:

- Python 3.11
- PyTorch 2.5.1
- CUDA 12.4
- torchvision, TorchGeo, torchmetrics
- Lightning and Lightning-UQ-Box for later UQ extensions

The Dockerfile is at `Dockerfile`. VS Code Dev Containers are configured in `.devcontainer/devcontainer.json`.

## 1. Recommended: VS Code Dev Container

Use this when you want the whole IDE, terminal, Python interpreter, and debugger inside Docker.

1. Open the repository root:

   ```bash
   code /home/yesong/master_thesis_yesong
   ```

2. In VS Code, press `Ctrl+Shift+P`.

3. Run:

   ```text
   Dev Containers: Rebuild and Reopen in Container
   ```

   Use **Rebuild and Reopen** the first time after Dockerfile changes. Later, plain **Reopen in Container** is fine.

4. Check the container terminal:

   ```bash
   python --version
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

5. Run the first-stage experiment:

   ```bash
   cd /workspace/eo_uq_experiments
   python main.py --config configs/eurosat_resnet18_rgb.yaml
   ```

Results will be written to:

```text
/workspace/eo_uq_experiments/outputs/results.csv
/workspace/eo_uq_experiments/outputs/logs/
/workspace/eo_uq_experiments/outputs/checkpoints/
```

Because `/workspace` is a bind mount of this repository, files created in the container appear on the host too.

## 2. Command-Line Docker Workflow

Use this when you do not want to reopen VS Code inside Docker.

Build the image from the repository root:

```bash
cd /home/yesong/master_thesis_yesong
docker build -t my-paper-env:latest .
```

Start a long-running container:

```bash
docker run -d --name yesong --gpus all --ipc host \
  -v $(pwd):/workspace \
  my-paper-env:latest tail -f /dev/null
```

If the container already exists:

```bash
docker start yesong
```

Run the EuroSAT RGB baseline:

```bash
docker exec -it yesong bash
cd /workspace/eo_uq_experiments
python main.py --config configs/eurosat_resnet18_rgb.yaml
```

Or run it in one command:

```bash
docker exec -it yesong bash -lc \
  "cd /workspace/eo_uq_experiments && python main.py --config configs/eurosat_resnet18_rgb.yaml"
```

## 3. Helper Script

After building the image, this helper starts the `yesong` container if needed and runs a Python file:

```bash
cd /home/yesong/master_thesis_yesong
./run_in_docker.sh eo_uq_experiments/main.py
```

For config-driven scripts, the explicit `docker exec` command is usually clearer because you can pass arguments directly:

```bash
docker exec -it yesong bash -lc \
  "cd /workspace/eo_uq_experiments && python main.py --config configs/eurosat_resnet18_rgb.yaml"
```

## 4. Changing Experiment Settings

Edit:

```text
eo_uq_experiments/configs/eurosat_resnet18_rgb.yaml
```

Common changes:

```yaml
training:
  epochs: 3          # smoke test
  batch_size: 64
```

For a longer thesis run:

```yaml
training:
  epochs: 20
```

For an even faster smoke test:

```yaml
training:
  epochs: 1
  limit_train_batches: 20
  limit_val_batches: 10
```

## 5. TensorBoard

The container forwards port `6006` in Dev Containers.

Inside the container:

```bash
tensorboard --logdir /workspace/eo_uq_experiments/outputs/logs --host 0.0.0.0 --port 6006
```

Then open:

```text
http://localhost:6006
```

## 6. Rebuild After Environment Changes

If `Dockerfile` changes:

```bash
docker build -t my-paper-env:latest .
docker stop yesong
docker rm yesong
docker run -d --name yesong --gpus all --ipc host \
  -v $(pwd):/workspace \
  my-paper-env:latest tail -f /dev/null
```

In VS Code, use:

```text
Dev Containers: Rebuild and Reopen in Container
```

## 7. Troubleshooting

Check whether the image exists:

```bash
docker image inspect my-paper-env:latest
```

Check whether the container is running:

```bash
docker ps
```

Check GPU access:

```bash
docker exec -it yesong python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If `Reopen in Container` says the image is missing, use:

```text
Dev Containers: Rebuild and Reopen in Container
```

If the `yesong` container name is already occupied by an old broken container:

```bash
docker stop yesong
docker rm yesong
```
