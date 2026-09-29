# Uncertainty Quantification for Earth Observation Foundation Models

本仓库是硕士论文的实验代码与结果文档。研究对象是 EO foundation model（**DOFA**、**Panopticon**）在下游适配后的概率校准与不确定性量化（UQ）。

**状态（更新于 2026-09-29）：全部计划实验已完成，论文进入写作阶段。** 16 个 dataset × model × adaptation 单元、72 次训练、Temperature Scaling / MC Dropout / Deep Ensemble 评估、审计与补充分析（A/E）均已完成；不需要再补训练或模型前向。

## 研究问题

- **RQ1**：不同 EO foundation model 在下游适配后，预测概率的校准程度如何？
- **RQ2**：不同 fine-tuning 方式（frozen vs. full fine-tuning）是否、以及怎样影响校准和任务性能？
- **RQ3**：Temperature Scaling、MC Dropout、Deep Ensemble 能否改善校准，是否以任务性能下降为代价？

审计标准见 [EO_FM_UQ_Core_RQ_Audit_Standard.md](EO_FM_UQ_Core_RQ_Audit_Standard.md)。

## 从哪里开始读

| 文档 | 内容 |
|---|---|
| [最终结果包](reports/core_rq_completion_20260921/published/final_thesis_results.md) | 全部正式表格、reliability diagram、性能–校准图、分割不确定性图 |
| [RQ1–RQ3 结果正文](reports/core_rq_completion_20260921/RESULTS_SECTION.md) | 每个 RQ 的回答、反例和结论边界 |
| [审计完成记录](reports/core_rq_completion_20260921/COMPLETION_REPORT.md) | 29 项检查：27 PASS、2 UNKNOWN（均为历史来源问题） |
| [A/E 补充研究报告](reports/thesis_followup_execution_20260926/FOLLOWUP_COMPLETION_REPORT.md) | 错误识别/拒识、置信度尺度、UQ 代理量、解析参照实验 |
| [论文结果草稿](reports/thesis_followup_execution_20260926/THESIS_RESULTS_DRAFT.md) | 可直接用于结果章的段落 |
| [论文准备与大纲](reports/thesis_readiness_20260926/THESIS_READINESS_AND_OUTLINE.md) | 章节结构与完成度评估 |
| [训练协议](reports/final_training_protocol.md) / [数据协议](reports/final_dataset_protocols.md) / [UQ 前协议冻结](reports/pre_uq_protocol_freeze.md) | 冻结的实验设置 |
| [证据矩阵](reports/thesis_evidence_matrix.md) | 每个结论对应的证据文件 |

## 实验矩阵

| 维度 | 设置 |
|---|---|
| Foundation models | DOFA ViT-Base、Panopticon ViT-B/14 |
| 数据集 | EuroSAT（10 类分类）、TreeSatAI（15 标签多标签分类）、CloudSEN12（4 类云分割）、SpaceNet7（建筑物分割） |
| 适配方式 | frozen backbone（只训练 head/decoder）、full fine-tuning |
| Seeds | 42、43、44（全部报告，不挑 seed） |
| UQ 方法 | Deterministic、Temperature Scaling（仅 EuroSAT，在专用 calibration split 上拟合）、MC Dropout（head-only，p=0.1，T=30）、Deep Ensemble（3 个 seed 成员的概率平均） |
| Checkpoint 选择 | 分类：最低 validation NLL；分割：最高 validation mIoU |
| 指标 | 分类：accuracy、Macro-F1、NLL、Brier、ECE-15；分割：mIoU、per-class IoU、pixel accuracy、NLL、Brier、ECE-15；SpaceNet7 另有 building/boundary ECE |

共 64 个方法单元：52 个有结果，12 个 Temperature Scaling 单元按协议记为 N/A（TreeSatAI 与分割）。另有 24 组同权重的 MC vs. dropout-off 对照。

数据划分：

- EuroSAT 使用项目自建的 70/10/10/10 空间分组划分（[splits/eurosat_70_10_10_10_spatial20m/](splits/eurosat_70_10_10_10_spatial20m/)）。
- TreeSatAI、CloudSEN12、SpaceNet7 使用 GEO-Bench-2 官方划分。

## 主要结果（摘要）

完整数值见[最终结果包](reports/core_rq_completion_20260921/published/final_thesis_results.md)。EuroSAT deterministic，mean ± SD（3 seeds）：

| Model | Adaptation | Accuracy | NLL ↓ | ECE-15 ↓ |
|---|---|---:|---:|---:|
| DOFA | frozen | 0.9834 ± 0.0006 | 0.0544 ± 0.0039 | 0.0050 ± 0.0019 |
| DOFA | full | 0.9649 ± 0.0054 | 0.1069 ± 0.0203 | 0.0111 ± 0.0049 |
| Panopticon | frozen | 0.9833 ± 0.0004 | 0.0505 ± 0.0036 | 0.0066 ± 0.0026 |
| Panopticon | full | 0.9627 ± 0.0073 | 0.1126 ± 0.0374 | 0.0095 ± 0.0050 |

核心结论（均限于已声明的配置范围）：

- **RQ1**：FM 的校准排序随任务、适配方式、指标和分箱而变化，没有跨全部条件一致的"校准冠军"。总体 ECE 低，不代表每个类别都校准良好。
- **RQ2**：frozen → full 的性能和校准变化因条件而异，方向不稳定本身就是结果。EuroSAT 上 full fine-tuning 更差；CloudSEN12 与 TreeSatAI 上 full 更好。两种方案的学习率和训练日程不同，不能把差异单独归因于"是否冻结"。
- **RQ3**：Temperature Scaling 不改变预测（argmax 不变），能降低部分 ECE/NLL。Deep Ensemble 在分割上提高 mIoU 并降低 NLL/Brier，但 EuroSAT full 的 ECE 变差。MC Dropout 相对 dropout-off 在分割上改善 ECE/NLL/Brier，在分类上收益不一致。
- **补充 A/E**：MSP 用于错误识别和拒识有效（EuroSAT 错误识别 AUROC 0.92–0.98）。但在 TreeSatAI 上 MI 的错误排序全面不如 MSP；校准改善也不保证错误排序改善。解析参照实验（E）表明，posterior expected entropy 不能直接等同于数据生成的条件熵。
- **已知边界**：DOFA–EuroSAT 历史归一化常数的统计来源无法追溯（2 项 UNKNOWN）；分割 Deep Ensemble 只有一组；MC Dropout 只在 head 上启用。

## 代码结构

```text
configs/                  # 16 个单元的正式 YAML（*_final.yaml、eurosat_*）与 c5_mc_dropout/
scripts/
├── run_experiments.py              # 分类训练/评估入口
├── segmentation_pipeline.py        # 分割训练/评估
├── geobench_datasets.py            # TreeSatAI/CloudSEN12/SpaceNet7 数据加载
├── create_eurosat_splits.py        # EuroSAT 固定 split 生成与验证
├── experiment_manager.py           # config、run_id、seed、环境元数据
├── prediction_export.py            # 逐样本预测导出、读取与指标重算
├── c3_calibration_ensembles.py     # Temperature Scaling 与 Deep Ensemble
├── c4_mc_dropout_pilots.py / c5_*  # MC Dropout 准备与推理
├── complete_segmentation_dropout_off.py
├── c6_build_final_results.py       # 生成最终结果表和图
└── build_*.py / audit_*.py         # 效应表、证据矩阵、审计
models/calibration.py     # NLL、Brier、ECE、reliability diagram
reports/                  # 协议、审计、结果与补充分析（见上表）
tests/                    # 单元测试
DOFA/                     # DOFA 上游代码（权重不入库）
```

## 运行

```bash
# 分类（例：EuroSAT Panopticon frozen，依次跑 seeds 42/43/44）
python scripts/run_experiments.py --config configs/eurosat_panopticon_frozen_baseline.yaml
# 加 --dry-run 仅验证 pipeline（1 epoch、少量 batch，写入 dry_runs/）

# 验证 EuroSAT split（不覆盖）
python scripts/create_eurosat_splits.py --data-root data \
  --output-dir splits/eurosat_70_10_10_10_spatial20m --validate-only

# 测试
python -m unittest discover -s tests -v
```

A/E 补充分析的复跑步骤见 [reports/thesis_followup_execution_20260926/README.md](reports/thesis_followup_execution_20260926/README.md)。这部分只需 CPU。

## 未入库的内容

数据集、checkpoint、原始预测和训练输出体积太大（数百 GB），没有放进 Git，仅保存在本地：`data/`、`datasets/`、`research_data/`、`results/`、`checkpoints/`、`RS3DBench/` 以及所有 `*.pt`/`*.pth`/`*.npy`。仓库中的报告以 SHA256 记录了这些产物的身份。

## 其他

`RS3DBench/` 深度估计、`eo_uq_experiments/`、`results/first_stage_rgb/` 和早期配置（如 `config.yaml`、`eurosat_dofa_rgb.yaml`）属于历史或备用管线，不属于论文的正式协议，不应与正式结果合并统计。
