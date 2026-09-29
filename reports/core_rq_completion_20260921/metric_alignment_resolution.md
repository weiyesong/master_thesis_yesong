# 新补评估的导出数值对齐

独立重算最初12个结果中11个通过。唯一失败为 SpaceNet7 / Panopticon / full / seed42 的 `confusion_matrix_exact`；所有标量指标差均小于容差2e-6，未放宽混淆矩阵相等要求。原结果见 [对齐前验证](segmentation_off_pre_alignment_verification.json)。

[完整CPU/CUDA重放探针](segmentation_off_softmax_alignment_probe.json) 直接读取保存logits，没有加载模型。1152幅图、57,802,752像素仅以下一个像素的argmax不同：`sample_5025`，index 873，row 79，col 123（均为零基索引），真值background=0。

| 对象 | background | building | argmax |
|---|---:|---:|---:|
| 保存logits | -0.013092570938169956 | -0.013092494569718838 | 1 |
| CPU导出float32 softmax | 0.4999999701976776 | 0.5 | 1 |
| CUDA float32 softmax | 0.5 | 0.5 | 0（平局取首类） |

logit差为7.6368451e-8。CPU重放与全部保存概率最大差为0；CUDA重放混淆矩阵精确等于原GPU累计矩阵。因此根因是近乎平局处的设备舍入及argmax平局规则，不是不同权重、标签错配或训练错误。

原矩阵 `[[52967931,807313],[3640568,386940]]`；与保存预测一致的矩阵 `[[52967930,807314],[3640568,386940]]`。相应mIoU和pixel accuracy变化分别约−1.70e-8、−1.73e-8，该像素也落在真值边界区域，boundary ECE变化约−5.74e-8。不改变结果量级；仍完成对齐，避免以“差很小”代替正确对应。

最小修复由 [align_export_metrics.py](align_export_metrics.py) 从保存logits按CPU导出路径重算全部主/前景/边界指标及逐图表，并要求重算softmax与全部保存概率逐元素相等。新文件为原run目录中的 `metrics_export_aligned.json`、`per_image_metrics_export_aligned.csv` 与 `metric_alignment.json`；最后一个文件记录原文件及新文件hash、来源代码和差值。原 `metrics.json`、`completion.json`、`per_image_metrics.csv` 及NPZ完全保留。三方表对该run采用新canonical指标，`metric_path`明确指出版本。

原run目录：[SpaceNet7/Panopticon/full/seed42](../../results/final_thesis/core_rq_completion_20260921/segmentation/spacenet7/panopticon/full_finetune/seed42/metric_alignment.json)。独立NumPy验证随后再次检查全部12个结果，并核验原completion hash与新对齐记录hash；验收见 [最终全量核验](segmentation_off_verification_summary.json)。这次修复只重算保存预测的指标，没有补推理、重新选择checkpoint或训练。
