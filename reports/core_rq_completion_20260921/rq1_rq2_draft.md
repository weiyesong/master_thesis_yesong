# RQ1 与 RQ2：可采用的结果段落（2026-09-21 UTC）

本文件直接依据逐预测重算指标、实际配置、checkpoint 和原始预测路径编写；不是对另一 agent 总结的转述。它是本次补充报告的可复核结果正文，不声称未提供的外部论文全文已经通过审阅。

## 证据、任务语义与重复层级

固定矩阵为两种 FM（DOFA、Panopticon）、四个 benchmark、两种适配方案，每格 seeds 42/43/44，共 48 个 deterministic 结果。EuroSAT 为 10 类分类；TreeSatAI 为 15 标签分类，主性能为 Macro-F1，accuracy 若报告则是 exact-match；CloudSEN12 与 SpaceNet7 为语义分割，主性能为 mIoU。NLL/Brier 描述概率质量，不能称为单独的校准误差。ECE-15 为 0–1 单位、15 个等宽右闭箱，0 落入首箱。EuroSAT/分割主 ECE 对应 top-1 confidence 对 correctness，TreeSatAI 主 ECE 对应展平 sample-label 二元决策的 confidence=max(p,1−p) 对 correctness；另列正类概率及逐标签诊断。分割忽略像素不进入指标，建筑概率校准使用全部有效像素上的 p(building) 与建筑指示变量。

三个 seed 的均值 ± sample SD 描述训练随机轨迹差异；没有把像素、标签或 MC pass 当作独立训练重复，没有据此构造显著性或等效性结论。CloudSEN12 acquisition product 的跨 split 情况、EuroSAT 原始 source-scene 不可得等边界见 [split_checks.json](../core_rq_audit_20260918/split_checks.json)。

本轮额外的直接核对：读取全部 24 个 deterministic 分类 parquet 的概率与真实标签，用独立 NumPy 公式重算 accuracy、Macro-F1、NLL、Brier 和右闭 ECE-15；相对下列重算表最大差分别为 `0, 0, 0, 0, 4.58e−16`。另读取全部 24 个 deterministic 分割 run 的 `test_metrics.json`，从原始 confusion_matrix 重算 mIoU/pixel accuracy 并核对 NLL/Brier/ECE，最大差 `2.31e−8`。分割概率的完整重算证据沿用上一轮 [segmentation_verification.json](../core_rq_audit_20260918/segmentation_verification.json)，不把本次 JSON 核对称作又一次全量像素重算。原始预测路径逐行列于 [actual_artifacts.csv](../core_rq_audit_20260918/actual_artifacts.csv)。 本轮逐 run 的计算值、参考值、差值、源文件 SHA256，以及 48 个 deterministic ECE-15 箱表加权重建的完整记录，保存于 [provenance_numeric_crosscheck.json](provenance_numeric_crosscheck.json)；[重跑脚本](provenance_numeric_crosscheck.py) 不调用项目指标函数。48 次逐 run 指标核对和 48 次箱表核对全部满足绝对容差 `1e−6`。

## RQ1：固定矩阵中的性能、经验校准与模型排序

48 个 deterministic 结果支持回答两套 FM 配置在各 benchmark/适配条件下的性能和经验校准程度，并表明校准排序随数据集、适配条件、指标与分箱发生变化。没有证据支持跨全部条件一致的 FM 校准冠军，也不能仅凭某个很小的总体 ECE 宣称所有类别均校准良好。以下均值/SD直接由 [分类逐 seed 重算指标](../core_rq_audit_20260918/classification_recomputed_metrics.csv) 与 [分割逐 seed 重算指标](../core_rq_audit_20260918/segmentation_recomputed_metrics.csv) 计算。

| 数据集 / 主性能 | 模型 | 适配 | 主性能 | ECE-15 | NLL | Brier |
|---|---|---|---|---|---|---|
| eurosat / accuracy | dofa | frozen | 0.98342 ± 0.00064 | 0.00501 ± 0.00185 | 0.05439 ± 0.00390 | 0.02600 ± 0.00152 |
| eurosat / accuracy | dofa | full_finetune | 0.96487 ± 0.00537 | 0.01112 ± 0.00490 | 0.10694 ± 0.02033 | 0.05327 ± 0.00818 |
| eurosat / accuracy | panopticon | frozen | 0.98330 ± 0.00043 | 0.00662 ± 0.00263 | 0.05046 ± 0.00363 | 0.02564 ± 0.00093 |
| eurosat / accuracy | panopticon | full_finetune | 0.96266 ± 0.00732 | 0.00945 ± 0.00502 | 0.11262 ± 0.03742 | 0.05576 ± 0.01362 |
| treesatai / macro_f1 | dofa | frozen | 0.22316 ± 0.00369 | 0.01081 ± 0.00088 | 0.25584 ± 0.00074 | 0.07398 ± 0.00020 |
| treesatai / macro_f1 | dofa | full_finetune | 0.24382 ± 0.01239 | 0.01255 ± 0.00390 | 0.24686 ± 0.00301 | 0.07169 ± 0.00097 |
| treesatai / macro_f1 | panopticon | frozen | 0.21085 ± 0.00548 | 0.00950 ± 0.00090 | 0.25700 ± 0.00209 | 0.07538 ± 0.00062 |
| treesatai / macro_f1 | panopticon | full_finetune | 0.21975 ± 0.02436 | 0.00763 ± 0.00356 | 0.25046 ± 0.00451 | 0.07319 ± 0.00161 |
| cloudsen12 / miou | dofa | frozen | 0.60444 ± 0.00371 | 0.04002 ± 0.01795 | 0.47326 ± 0.01123 | 0.24191 ± 0.00330 |
| cloudsen12 / miou | dofa | full_finetune | 0.65679 ± 0.00670 | 0.06368 ± 0.00445 | 0.47961 ± 0.01512 | 0.21216 ± 0.00482 |
| cloudsen12 / miou | panopticon | frozen | 0.65439 ± 0.00053 | 0.04162 ± 0.00729 | 0.41371 ± 0.02139 | 0.20612 ± 0.00399 |
| cloudsen12 / miou | panopticon | full_finetune | 0.67248 ± 0.00366 | 0.04216 ± 0.00379 | 0.38807 ± 0.01497 | 0.19142 ± 0.00364 |
| spacenet7 / miou | dofa | frozen | 0.48828 ± 0.00086 | 0.03913 ± 0.00304 | 0.31233 ± 0.02740 | 0.13075 ± 0.00317 |
| spacenet7 / miou | dofa | full_finetune | 0.49781 ± 0.00162 | 0.04965 ± 0.00637 | 0.47008 ± 0.07885 | 0.13408 ± 0.00392 |
| spacenet7 / miou | panopticon | frozen | 0.50133 ± 0.00152 | 0.04128 ± 0.00025 | 0.35486 ± 0.00088 | 0.13629 ± 0.00071 |
| spacenet7 / miou | panopticon | full_finetune | 0.50431 ± 0.00489 | 0.04541 ± 0.01022 | 0.43183 ± 0.08544 | 0.14117 ± 0.00466 |

EuroSAT frozen 的平均 ECE-15 为 DOFA 0.00501、Panopticon 0.00662，full 时分别为 0.01112、0.00945；该排序变化不能归因于纯 backbone 架构。DOFA 历史常数与 Panopticon 的 train-only 归一化不同，且 DOFA 六个历史 run 的训练代码绑定/统计推导来源仍 UNKNOWN，详见 [历史来源补查](provenance_followup.md)。Panopticon−DOFA 的逐 seed 差值及 ECE-10/15/30 变化均保留于 [paired_effect_summaries.csv](../core_rq_audit_20260918/paired_effect_summaries.csv)；细小差异应按该范围描述。

TreeSatAI 的低二元决策 ECE 不能覆盖逐标签校准。例如 Panopticon/full/seed42：decision-ECE=0.011267，汇总正类概率 ECE=0.018803，逐标签概率 ECE 宏均值=0.033700。直接读取该 run 标签后，87.42% 的 sample-label 项为负，91.88% 的预测判为负；这说明主 ECE 的汇总构成，不构成负标签导致某一校准数值的因果证明。须同时报告 [class_diagnostics.csv](../core_rq_audit_20260918/class_diagnostics.csv) 的每标签支持数、precision/recall/F1 和概率诊断。

CloudSEN12 的 12 个 deterministic 结果整体 confidence−accuracy 均为正，范围 0.01929–0.06802，支持整体平均过度自信的描述；这并不表示每个概率箱都过度自信。thin-cloud IoU 为 0.35289–0.47509，须和各类概率 ECE/支持数一起解释，不能用总体 ECE 代替关键类分析。逐箱差异、count、ECE-10/15/30 与各类结果见 [segmentation_reliability_bins.csv](../core_rq_audit_20260918/segmentation_reliability_bins.csv) 和 [segmentation_class_diagnostics.csv](../core_rq_audit_20260918/segmentation_class_diagnostics.csv)。

SpaceNet7 的 12 个 deterministic 结果 pixel accuracy 为 0.91330–0.92649，decision-ECE 为 0.03376–0.05691；建筑 IoU 仅为 0.04962–0.10564。总体正确率不能掩盖建筑分割表现。建筑概率 ECE 在全部有效像素上计算，为 0.04117–0.06441；它与只在真实建筑像素上的条件统计不同。上面范围均为全部 12 个 seed 结果的最小/最大值，不是四个 cell 均值的范围。SpaceNet7 foreground NLL 的原生 float32 上截断与 float64 直接计算的差异已由 [foreground_precision_verification.json](../core_rq_audit_20260918/foreground_precision_verification.json) 解释，不作为模型或测量失败。


### 逐条件过度/不足自信方向

令 signed gap=mean(confidence−correctness)，正值为整体平均过度自信，负值为整体平均不足自信。以下来自每个 seed 的 ECE-15 非空箱，按箱内数量加权重建，再与逐预测重算的 gap 核对；所有 48 个 run 一致。表中“混合箱”是同时出现正/负 gap 非空箱的 seed 数（每格总共 3），不把少数稀疏箱外推为总体结论。整体 gap 可以正负相消，因此它用于描述方向，不能替代 ECE 或逐箱诊断。

| 数据集 | 模型 | 适配 | 三 seed 平均 signed gap | 单 seed 范围 | 整体过/欠 seed 数 | 混合箱 seed 数 |
|---|---|---|---|---|---|---|
| eurosat | dofa | frozen | -0.00154 | [-0.00191, -0.00084] | 0/3 | 3 |
| eurosat | dofa | full_finetune | +0.00748 | [-0.00072, +0.01612] | 2/1 | 3 |
| eurosat | panopticon | frozen | +0.00453 | [+0.00304, +0.00575] | 3/0 | 3 |
| eurosat | panopticon | full_finetune | +0.00739 | [+0.00458, +0.01253] | 3/0 | 3 |
| treesatai | dofa | frozen | -0.00367 | [-0.00686, +0.00144] | 1/2 | 3 |
| treesatai | dofa | full_finetune | -0.00951 | [-0.01545, +0.00048] | 1/2 | 2 |
| treesatai | panopticon | frozen | -0.00302 | [-0.00625, -0.00011] | 0/3 | 3 |
| treesatai | panopticon | full_finetune | -0.00202 | [-0.00698, +0.00069] | 2/1 | 3 |
| cloudsen12 | dofa | frozen | +0.04001 | [+0.01929, +0.05067] | 3/0 | 1 |
| cloudsen12 | dofa | full_finetune | +0.06368 | [+0.05913, +0.06802] | 3/0 | 1 |
| cloudsen12 | panopticon | frozen | +0.04162 | [+0.03645, +0.04996] | 3/0 | 1 |
| cloudsen12 | panopticon | full_finetune | +0.04216 | [+0.03805, +0.04553] | 3/0 | 1 |
| spacenet7 | dofa | frozen | +0.03339 | [+0.03011, +0.03578] | 3/0 | 3 |
| spacenet7 | dofa | full_finetune | +0.04867 | [+0.04249, +0.05691] | 3/0 | 2 |
| spacenet7 | panopticon | frozen | +0.04015 | [+0.03898, +0.04114] | 3/0 | 3 |
| spacenet7 | panopticon | full_finetune | +0.04123 | [+0.02122, +0.05291] | 3/0 | 1 |

EuroSAT：DOFA/frozen 三个 seed 整体均轻微不足自信；DOFA/full 的整体方向有两个过度自信、一个不足自信。Panopticon 的 frozen/full 三个 seed 整体均过度自信。所有 EuroSAT run 同时存在正负方向的非空箱，因此这些“整体方向”不是全置信度区间的单一方向。

TreeSatAI：四格的三 seed 平均 signed gap 均为负，但只有 Panopticon/frozen 三个 seed 均整体不足自信。DOFA/frozen 和 DOFA/full 各有两个不足、一个过度自信；Panopticon/full 恰有两个过度、一个不足自信，负均值由幅度更大的不足自信 seed 主导。不能只用均值把这些格称为稳定欠自信。多数 run 的不同二元决策置信度箱也存在方向混合；这仍不等价于逐标签正类概率方向。

CloudSEN12：四格共 12 个 run 整体平均均过度自信，每格有一个 seed 同时存在过/欠自信箱。SpaceNet7：四格共 12 个 run 整体平均均过度自信，DOFA/frozen、DOFA/full、Panopticon/frozen、Panopticon/full 分别有 3、2、3、1 个 seed 的箱内方向混合。因此两数据集都应分别阅读整体方向和有 count 的逐箱曲线，不能将总体正 gap 描述为每个预测都过度自信。

所有可靠性图应与计数直方图、指标对象、分箱边界和 seed 标签一起发布；seed42 图仅用于可读展示，所有 seeds 保留在 CSV。箱内 confidence−outcome 的正/负分别表示过度/不足自信，不能用整体平均 gap 代替逐箱方向。

RQ1 的必要工作是把上述定义、条件性结论和类诊断纳入正式发布，修复原报告生成器的对象/分箱与 count 问题；已有预测足以支持这些分析，无已确认的 RQ1 必须新增推理或训练项。历史来源 UNKNOWN 作为已明确限制继续保留，而不将真实负结果或小效应视为未回答。

## RQ2：适配方案改变后的实际差值

24 个同 dataset、FM、seed 的 frozen→full 对照均存在。数据和任务内 checkpoint selector 相同，冻结/更新状态有权重与梯度证据；不同方案还改变 LR、weight decay、warm-up、epoch 上限和 patience，因此本题估计完整适配配方的差异。它不隔离“只打开 backbone 梯度”这一单因素因果效应。配置与选择依据见 [adaptation_config_pairs.json](../core_rq_audit_20260918/adaptation_config_pairs.json)、[final_training_protocol.md](../final_training_protocol.md) 及 [checkpoint_verification.json](../core_rq_audit_20260918/checkpoint_verification.json)。

下表 full−frozen，±为三个配对差值的 sample SD；表的每一行均保留三个 seed，未按结果选 seed。原始配对与 min/max 见 [paired_effects_from_predictions.csv](../core_rq_audit_20260918/paired_effects_from_predictions.csv) 与 [paired_effect_summaries.csv](../core_rq_audit_20260918/paired_effect_summaries.csv)。

| 数据集 / 主性能 | 模型 | Δ主性能 | ΔECE-15 | ΔNLL | ΔBrier |
|---|---|---|---|---|---|
| eurosat / accuracy | dofa | -0.01855 ± 0.00583 | +0.00611 ± 0.00611 | +0.05255 ± 0.02389 | +0.02727 ± 0.00925 |
| eurosat / accuracy | panopticon | -0.02063 ± 0.00706 | +0.00283 ± 0.00431 | +0.06215 ± 0.03444 | +0.03012 ± 0.01292 |
| treesatai / macro_f1 | dofa | +0.02066 ± 0.01539 | +0.00175 ± 0.00377 | -0.00899 ± 0.00228 | -0.00229 ± 0.00077 |
| treesatai / macro_f1 | panopticon | +0.00891 ± 0.02608 | -0.00187 ± 0.00268 | -0.00654 ± 0.00306 | -0.00219 ± 0.00120 |
| cloudsen12 / miou | dofa | +0.05236 ± 0.00642 | +0.02367 ± 0.02186 | +0.00635 ± 0.02629 | -0.02975 ± 0.00209 |
| cloudsen12 / miou | panopticon | +0.01809 ± 0.00420 | +0.00054 ± 0.00811 | -0.02565 ± 0.02324 | -0.01471 ± 0.00472 |
| spacenet7 / miou | dofa | +0.00953 ± 0.00076 | +0.01051 ± 0.00929 | +0.15775 ± 0.10397 | +0.00333 ± 0.00521 |
| spacenet7 / miou | panopticon | +0.00298 ± 0.00539 | +0.00413 ± 0.01047 | +0.07697 ± 0.08490 | +0.00489 ± 0.00395 |

full fine-tuning 对校准没有统一方向。EuroSAT 两 FM 的三个 accuracy 差值均为负，平均 NLL/Brier/ECE 均变差；但 Panopticon ECE 的逐 seed 差值跨零（−0.00184 至 +0.00665），故“每一个 seed 所有指标均变差”不成立。TreeSatAI 的平均 Macro-F1 上升、NLL/Brier 下降，DOFA 平均 ECE 上升，Panopticon 平均 ECE 下降；Panopticon 的 Macro-F1 差值为 −0.01263 至 +0.03790，适配收益的跨 seed 方向不稳定。

CloudSEN12 两 FM 的三个 mIoU 差值均为正，平均 Brier 下降；DOFA ECE 在三个 seed 都上升，而 Panopticon ECE 在 −0.00707 至 +0.00908 间变化，均值接近零不等于等效无影响。SpaceNet7 的平均 mIoU 略升，同时平均 NLL/Brier/ECE 变差；Panopticon mIoU 差值范围 −0.00161 至 +0.00891，再次说明均值不能替代重复间的不稳定性。

这些条件依赖的权衡和不稳定差异已经是 RQ2 的有效答案。对现有矩阵，无需为了制造同方向收益补训练、增 seed 或重选 checkpoint；也无需仅因没有显著改善将 RQ2 判为未充分回答。没有预定义等效界限或按独立组设计的区间，因此不声称“无影响”“等效”“普遍更优”或统计显著性。跨所有 EO 数据分布的普适结论、纯架构/纯冻结开关因果隔离属于后续探索，不是本轮的必要补救。

## 本次正文整理的验收边界

上述结果段落、明确任务语义、逐 seed 配对、类条件诊断和来源限制构成本次可审核文本。若采用它们，相关“本次整理报告的表述”可以按直接表格和产物验收；未提供的外部完整论文仍为未审范围。RQ1/RQ2 不需要新的训练；报告代码与图表的一致性由本次主任务实际重建结果验收，而不是由本文件预先宣称通过。
