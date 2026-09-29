# C2 segmentation deterministic execution state

Last updated: 2026-08-20 14:15 UTC

## Current state

- Formal C2 runs completed: **0/24**.
- Batch-size-8 capacity smoke tests completed: **0/8**.
- No C2 training process is active in this container.
- `results/final_thesis/segmentation` contains no formal run artifacts.
- The only RTX 3090 is blocked by a compute process outside this container's PID namespace. At the last check it used 19,818 MiB, leaving 4,297 MiB, with active GPU utilization.
- The external process was not terminated or otherwise modified.

## Completed implementation gate

- Four final configs encode the frozen B5 protocol and seeds 42/43/44.
- Final segmentation training supports validation-mIoU checkpoint selection, exact early stopping, atomic best/last checkpoints, RNG and DataLoader-generator state capture, strict in-place resume, gradient audits, and input-artifact hashes.
- Test export includes labels, logits, probabilities, predictions, correctness, confidence, predictive entropy, valid masks, per-image metrics, and global-mean-pooled 768-dimensional dense backbone representations.
- The C2 auditor checks protocol identity, provenance, learning-history/early-stop semantics, checkpoints, gradient invariants, official test IDs, export schemas, representations, and optionally recomputes metrics from saved logits.
- All input dataset artifacts, manifests, pretrained weights, and the B5 report hash were reverified.
- Test suite: **62/62 passing** on 2026-08-20.

## Frozen pre-run code identity

- Aggregate code snapshot SHA256: `90623ef922184d9b2f77cc73d12e02674c3d2c80fb95de95bb16aee916991da8`
- `scripts/run_experiments.py`: `47221cd39d7f0b7b6cf85130ef9547a8de508ef6244116267ed42f98824e34a0`
- `scripts/segmentation_pipeline.py`: `870bb59f8c45e4749c1277bcc8895b747e3d8f463aa571c81e98dd8fb4c71a47`
- `scripts/audit_c2_segmentation_runs.py`: `c930933007678324b768fd711380462e45dd46d67bd4eb9db950c4cbd2b35ec2`
- B5 protocol: `cd569a938552f5cb85573d335d4f1a4c9b0815445a999853226b8052f099d711`

## Resume order after the GPU is exclusively available

Do not modify code/config files between starting a formal run and completing or exactly resuming it.

1. Confirm that total compute-process memory stays below 1 GiB for at least 30 seconds.
2. Run the four final configs with `--dry-run --seed 42`; each config contains its DOFA and Panopticon cell. This validates all eight cells at physical batch size 8.
3. If all smoke tests pass, run the same four configs with `--seed 42` and no `--dry-run`.
4. Run `python -m scripts.audit_c2_segmentation_runs --seeds 42 --deep --fail-on-nonpromotable`.
5. Review all eight learning curves and only proceed if the seed-42 audit is promotable and no protocol-level defect is found.
6. Run the four configs with explicit `--seed 43 --seed 44`.
7. Run the final auditor with `--seeds 42 43 44 --deep --fail-on-nonpromotable` and write the completion report.

Final config paths:

- `configs/cloudsen12_frozen_final.yaml`
- `configs/cloudsen12_full_finetune_final.yaml`
- `configs/spacenet7_frozen_final.yaml`
- `configs/spacenet7_full_finetune_final.yaml`

If a formal run is interrupted after a committed epoch, resume only with:

```bash
python -m scripts.run_experiments --resume-run-dir /absolute/path/to/the/run
```

The resume command deliberately refuses changed code/config snapshots or incomplete legacy checkpoints.
