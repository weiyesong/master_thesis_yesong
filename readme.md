# Uncertainty Quantification for Earth Observation Foundation Models

本仓库是硕士论文实验代码。当前主实验为 **DOFA + EuroSAT 图像分类**，已完成固定数据划分、统一配置与结果管理、逐样本预测导出，以及 frozen-backbone 和 full fine-tuning 两组 3-seed deterministic baselines。

当前状态更新于 **2026-08-09**。详细证据、代码位置和风险清单见 [代码审计报告](reports/code_audit.md)。

## 当前实验协议

| 项目 | 设置 |
|---|---|
| Model | DOFA ViT-Base，预训练权重 `DOFA/checkpoints/DOFA_ViT_base_e100.pth` |
| Dataset | EuroSAT，27,000 张 Sentinel-2 patch |
| Input | RGB `B04/B03/B02`，224 × 224 |
| Split | train 70% / validation 10% / calibration 10% / test 10% |
| Seeds | 42、43、44 |
| Head | `BatchNorm1d(768, affine=False, eps=1e-6) -> Linear(768, 10)` |
| Checkpoint selection | 最低 validation NLL |
| Test | 每个 seed 的 best checkpoint 做一次 clean deterministic evaluation |
| Metrics | accuracy、macro-F1、per-class precision/recall/F1、NLL、multiclass Brier、ECE-15 |
| Prediction export | Parquet 主表；随机采样预留 compressed NPZ；embedding 默认关闭 |

训练、验证、校准和测试职责严格分离：只有 train 用于梯度更新，validation 用于 early stopping 和 best checkpoint 选择，calibration 预留给后处理校准，test 只用于冻结协议后的最终评估。

## 已完成工作

- 统一 YAML 配置、唯一 `run_id`、resolved config、环境与运行时间记录。
- 统一固定随机种子：Python、NumPy、PyTorch CPU/CUDA、DataLoader generator/worker。
- 建立不可覆盖的 EuroSAT 固定 split CSV/JSON，并完成样本、文件哈希和 20 m 空间组检查。
- DataLoader 强制读取保存的 manifest；只有 train shuffle；四个 split 当前均使用确定性预处理。
- 支持 frozen 和 full fine-tuning；保存 best/last checkpoint、训练历史、参数和梯度审计。
- 逐样本导出 logits、probabilities、label、sample ID、置信度、margin、entropy 和实验元数据。
- 可从保存的预测文件重新计算全部 deterministic 指标，无需重新加载模型。
- 已完成 frozen 与 full fine-tuning 各 3 个 seed 的正式 baseline。
- 当前单元测试共 17 项，全部通过。

尚未执行 temperature scaling、MC Dropout 或 deep ensemble。代码中的 stochastic 三维数组格式只是为后续 UQ 实验预留，不能当作已经完成的 UQ 结果。

## 数据划分

固定 manifest 位于：

```text
splits/eurosat_70_10_10_10_spatial20m/
├── eurosat_splits.csv
├── eurosat_splits.json
├── class_distribution.csv
├── spatial_leakage_report.json
└── validation_report.json
```

| Split | 样本数 | 比例 |
|---|---:|---:|
| train | 18,866 | 69.87% |
| validation (`val`) | 2,707 | 10.03% |
| calibration | 2,713 | 10.05% |
| test | 2,714 | 10.05% |
| total | 27,000 | 100% |

划分使用 seed `20260803`。它在类别约束之外，将 20 m 内的空间相邻 patch 合并为同一 group 后再分配，从而避免已检测到的空间近邻跨 split。当前验证结果：无重复 sample ID、无重复路径、无缺失文件、split 两两无交集、无跨 split 相同内容哈希、无跨 split 空间组。CSV SHA-256 为：

```text
c5cadc7936394f0678307f7db25c55a2dc19f73ddb93891b772c16221e8abf22
```

重新验证现有 manifest，不会生成或覆盖 split：

```bash
python scripts/create_eurosat_splits.py \
  --data-root data \
  --output-dir splits/eurosat_70_10_10_10_spatial20m \
  --validate-only
```

## 正式 baseline 结果

以下均为 clean test deterministic evaluation，数值是 seeds 42/43/44 的 mean ± sample standard deviation。

| Adaptation | Accuracy | Macro-F1 | NLL ↓ | Brier ↓ | ECE-15 ↓ |
|---|---:|---:|---:|---:|---:|
| Frozen backbone | 0.98342 ± 0.00064 | 0.98234 ± 0.00069 | 0.05439 ± 0.00390 | 0.02600 ± 0.00152 | 0.00501 ± 0.00185 |
| Full fine-tuning | 0.96487 ± 0.00537 | 0.96375 ± 0.00549 | 0.10694 ± 0.02033 | 0.05327 ± 0.00818 | 0.01112 ± 0.00490 |

这是描述性汇总，不是最终统计结论。当前协议下 full fine-tuning 的平均 accuracy 比 frozen 低 1.855 个百分点，而且三个 full fine-tuning run 均出现明显的 post-best validation instability。差异主要集中在 `PermanentCrop` 与 `HerbaceousVegetation` 的混淆；不能据此用 test 结果反向调参。

汇总文件：

- Frozen：[aggregate_test_metrics.json](results/baselines/dofa_eurosat_frozen_bnlinear/aggregate_test_metrics.json)
- Full fine-tuning：[aggregate_test_metrics.json](results/baselines/dofa_eurosat_full_finetune/aggregate_test_metrics.json)
- 描述性对照：[comparison_frozen_vs_full.json](results/baselines/dofa_eurosat_full_finetune/comparison_frozen_vs_full.json)

## 运行命令

### 配置检查和 dry-run

`--dry-run` 会把训练改成 1 epoch、少量 batch，并写入独立 `dry_runs/` 目录。它仍会加载模型、数据和 checkpoint，因此可验证完整 pipeline，但不会启动正式训练。

```bash
python scripts/run_experiments.py \
  --config configs/eurosat_dofa_frozen_baseline.yaml \
  --dry-run

python scripts/run_experiments.py \
  --config configs/eurosat_dofa_full_finetune.yaml \
  --dry-run
```

### 正式训练模板

以下命令会依次执行配置中的 seeds 42、43、44，耗时较长。已有正式结果存在时，不要仅为了检查环境重复启动。

```bash
python scripts/run_experiments.py \
  --config configs/eurosat_dofa_frozen_baseline.yaml

python scripts/run_experiments.py \
  --config configs/eurosat_dofa_full_finetune.yaml
```

`main.py` 仍可作为兼容入口，但必须显式传配置：

```bash
python main.py --config configs/eurosat_dofa_frozen_baseline.yaml --dry-run
```

不带 `--config` 的 `python main.py` 当前默认走 RS3DBench depth 配置，不是 EuroSAT 主实验。

## 每个 run 的输出

```text
results/baselines/<experiment>/runs/<run_id>/
├── resolved_config.yaml
├── environment.json
├── model_audit.json
├── optimizer_group_audit.json       # full fine-tuning
├── gradient_audit.json
├── training_history.json
├── training_metrics.csv
├── batch_metrics.csv
├── validation_metrics.json
├── best.pt
├── last.pt
├── run_summary.json
├── results.json
├── training_dashboard.png
├── reliability.png
└── predictions/test/deterministic/
    ├── predictions.parquet
    ├── manifest.json
    ├── validation_report.json
    ├── metrics.json
    ├── per_class_metrics.json
    ├── confusion_matrix.npy
    └── confusion_matrix.csv
```

`environment.json` 记录 Python、PyTorch、CUDA、包版本、hostname 和 GPU。当前工作区的 Git 元数据不完整，`git rev-parse HEAD` 失败，因此已有 run 的 `git_commit` 为 `null`；这是待修复的 provenance 风险。

## 读取预测并重算指标

```python
from scripts.prediction_export import (
    load_prediction_export,
    recompute_metrics,
    validate_prediction_export,
)

prediction_dir = (
    "results/baselines/dofa_eurosat_frozen_bnlinear/runs/"
    "20260807T103233022656Z_eurosat_dofa_frozen_bnlinear_seed42_338e2700/"
    "predictions/test/deterministic"
)

bundle = load_prediction_export(prediction_dir)
report = validate_prediction_export(prediction_dir, expected_count=2714)
metrics = recompute_metrics(prediction_dir, n_bins=15)

print(bundle.table[["sample_id", "true_label", "predicted_label"]].head())
print(report["valid"], metrics)
```

Parquet 中保存每个样本的完整 logits 和 probabilities；`manifest.json` 保存 class-index mapping、数组 shape/dtype、checkpoint SHA-256 与数据来源。后续 MC Dropout/deep ensemble 的每次 pass/member 原始值应保存为 `[sample, pass_or_member, class]` 的 compressed NPZ，不能只保留均值。

## 关键代码与配置

```text
configs/
├── eurosat_dofa_frozen_baseline.yaml
└── eurosat_dofa_full_finetune.yaml

scripts/
├── experiment_manager.py           # config、run_id、seed、环境元数据
├── create_eurosat_splits.py        # 固定 split 生成和验证
├── run_experiments.py              # canonical 训练/评估入口
├── prediction_export.py            # 逐样本导出、读取、验证、重算指标
└── export_checkpoint_predictions.py

models/calibration.py               # NLL、Brier、ECE 和 reliability diagram
reports/code_audit.md               # 当前审计报告
tests/                              # 最小测试
```

## 已知限制与实验纪律

- test 已被用于当前两组 baseline 的描述性评估；后续不得依据这些 test 结果选择学习率、正则化或 UQ 超参数。
- calibration split 尚未用于任何正式后处理；temperature scaling 结果当前不存在。
- deterministic baseline 的 head dropout 为 0，因此它本身不能直接产生有意义的 MC Dropout 随机性。
- deep ensemble 尚未接入端到端评估；三个训练 seed 目前是独立重复，不应自动等同于 ensemble 结果。
- full fine-tuning 使用指定的高学习率协议并出现 validation instability；任何新协议都应只基于 train/validation 决定，并以新实验名保存。
- DOFA checkpoint 使用 `strict=False` 加载；当前正式 run 无 missing keys，unexpected keys 仅为预训练阶段的 `mask_token`、`projector.weight`、`projector.bias`。未来仍应保留允许列表审计。
- DOFA `pos_embed` 是固定 sin/cos positional embedding，151,296 个参数在 full fine-tuning 中仍为 `requires_grad=False`，不是误冻结。
- `results/first_stage_rgb/`、`thesis/experiment_1_dofa_rgb_record.md` 和旧配置属于历史协议，不应与当前 70/10/10/10 baseline 合并统计。

## 测试

```bash
python -m unittest discover -s tests -v
```

当前记录：17 tests passed。测试覆盖配置约束、seed/run 元数据、固定 split、manifest DataLoader、预测导出与无模型重算指标。

## 其他任务

仓库还保留 RS3DBench depth estimation 和若干早期/备用训练管线。它们不是当前 DOFA + EuroSAT baseline 的 canonical implementation。开始新实验前应确认使用上述两份正式 YAML，避免把不同 split、归一化或输出 schema 的旧结果混在一起。
