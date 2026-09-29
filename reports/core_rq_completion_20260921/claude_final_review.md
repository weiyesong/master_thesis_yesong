核对全部完成：两格原始 NPZ 独立抽算、TreeSatAI 原始 parquet 抽算、表图分箱 CSV 与正文数字对照均已做完。以下是最终审查。

## 最终独立复核结论（reports/core_rq_completion_20260921）

**总体判定：接受。** C05/C09/R1.6/R3.1/R3.4 五项可以关闭，C01/C06 保留 UNKNOWN 是诚实且范围限定正确的。三问在"既定配置/测试集的描述性答案"范围内成立。没有发现必须修代码、重算、补评估或补训练的事项。只有一处报告链接缺陷需修正（见第 4 节）。

### 1. 独立抽算结果（不依赖 Codex 或验证脚本）

我用自写 NumPy 直接读取原始文件，逐块累积，未调用项目指标函数。

**SpaceNet7 / Panopticon / full / seed42**，Off 文件 `results/final_thesis/core_rq_completion_20260921/segmentation/spacenet7/panopticon/full_finetune/seed42/predictions.npz`，同格 D（run b6b452b4）与 MC（aggregate_uncertainty_maps.npz）：

| 项目 | 我的结果 | 正式值 |
|---|---|---|
| ID/label/valid_mask 三方逐元素一致 | 全部 True，1152 图，57,802,752 有效像素 | 一致 |
| Off 混淆矩阵 | [[52967930,807314],[3640568,386940]] | 与 metrics_export_aligned.json 完全相同 |
| Off mIoU / pixacc | 0.5012820042795728 / 0.9230506879672442 | 精确相同 |
| Off ECE15 / NLL / Brier | 0.0444545536 / 0.4324613203 / 0.1313567368 | 差 ≤ 8e-10 |
| Off 建筑概率 ECE（全有效像素） | 0.0543258592 | 差 1e-11 |
| MC mIoU / ECE15 | 0.5012782693569223 / 0.0442224791 | 精确 / 差 8e-10 |
| D mIoU / ECE15 / NLL | 0.5090204123518141 / 0.0529097867 / 0.4801957556 | 精确相同 |
| sample_5025 (idx 873, r79, c123) | logits −0.0130925709/−0.0130924946，probs 0.49999997/0.5，pred=1，label=0 | 与探针一致 |

三方表该行 `metric_path` 指向 `metrics_export_aligned.json`，`Off_miou` 等于对齐值而非原 GPU 值，确认正式表已用新版本。我从 logits 重做 CPU softmax 与保存概率最大差 1.19e-7（float32 精度），argmax 与保存 prediction 差 0。

**CloudSEN12 / Panopticon / frozen / seed43**（4 类、无 ignore 像素、48,921,600 有效像素）：Off 混淆矩阵与 metrics.json 精确相同，mIoU 0.6498242939590266、ECE15 0.0406012、NLL 0.4104202、Brier 0.2036064；MC 与 D 的 mIoU/ECE/NLL/Brier 与三方表差 ≤ 1e-9。

**TreeSatAI / Panopticon / full / seed42** 原始 parquet：负 sample-label 占比 0.8742、预测负占比 0.9188、flattened decision ECE15 0.0112668、pooled positive-label ECE15 0.0188031，与 RESULTS_SECTION 第 38 行三个数字完全一致；箱内计数 [429, 804, 893, 1104, 1583, 2443, 4442, 18302] 与 `classification_reliability_grid.csv` 逐箱相同。

### 2. 对照与指标是否支持三问

- **R3.1/R3.4 关闭成立。** 24 组三方对照的差值算术误差 ≤ 3e-16；12 个分割 Off 的 checkpoint hash 与 MC manifest 相同（本格 fda9fa2c…），completion.json 记录 strict load、BN buffer 不变、no_grad、dropout 关闭、`training_performed=false`。`segmentation_pipeline.py:959` 的 evaluate_segmentation 先 `model.eval()`，run_summary 的 checkpoint_selection 为 val mIoU max，因此"MC checkpoint 以 dropout-off 验证选择、MC 是选择后的推理改动"这一表述有代码依据。
- **正文的方向性陈述可复核。** 12 个分割 MC−Off 的 ECE/NLL/Brier 全为负；分类 12 个方向不一致；ECE10/15/30 三处符号翻转（EuroSAT DOFA frozen 44、DOFA full 42、TreeSat DOFA full 42）与 `mc_dropout_ece_sensitivity.csv` 一致；CloudSEN12 DOFA frozen 42 的 MC−D +0.01377、MC−Off −0.00479、Off−D +0.01856 与三方表相符。
- **未过度声称。** 正文明确 12/16 MC 格为 n=1，SD 为描述性而非区间，未用"无损""等效""显著"。SpaceNet7 MC−Off mIoU 差在 ±1.6e-5 量级，正文只写"观察到的小差值"，恰当。Ensemble 反例（EuroSAT full ECE 上升 +0.0162/+0.0125、SpaceNet7 mIoU 下降 0.0065–0.0110）我按发布表核算无误。RQ2 明确只估计"完整适配配方"差异，未声称冻结开关因果。

### 3. C05/C09 关闭与 UNKNOWN 处理

- 发布包 `package_validation.json`：64 格、568 个值与 master 误差 0；52 条主图曲线 ECE 与 seed 级指标误差 0；12 个 TS NA 全为空白（分类 4、分割 8，我逐行确认）。图 CSV 的 interval 列为 `[lower,upper]` 首箱、其余 `(lower,upper]`，SpaceNet7 Pan full42 D 曲线 8 个非空箱计数与我的独立分箱逐箱相同，由箱表重建 ECE = 0.0529097867 = seed 指标。
- 历史 TreeSat TS 12 行完整保留（11/12 seed 测试 ECE 变差未隐藏），8-24/8-25/8-26/8-27 时序在 `final_thesis_results.md` 与 `protocol_timeline_and_na.csv` 中一致，未推断动机。
- 数值对齐"不需训练"理由成立：根因是 7.64e-8 的 logit 近平局在 CUDA/CPU 舍入后 argmax 不同，指标变化 ≤ 1.7e-8；对齐前失败记录（passed 11/failed 1）原样保留，混淆矩阵精确性要求未放宽。
- C01/C06：`provenance_followup.md` 只找到常数的早期代码载体，无统计推导、无训练时 commit；报告未把事后快照 c99c1409 冒充训练版本，评估代码 hash 单列。UNKNOWN 影响范围限定为 DOFA-EuroSAT 六 run 的预处理来源，未漂白也未扩大为全局 INVALID。可接受。

### 4. 仍需处理的事项

1. **报告链接缺陷（必须修，表述类）：** `COMPLETION_REPORT.md` 第 37 行链接 `claude_final_review.md` 与 `claude_final_evidence_trace.jsonl`，两文件当前不存在，`claude_final_raw.json` 为 0 字节。应在本审查落盘后重跑 `write_completion_report.py`，或改为条件性措辞。
2. **前景 NLL 约定（仅需保留说明）：** 我用 float32 clip 1e-7 得到 0.4135，正式值 0.43245 与多类 NLL 几乎相等，说明该指标对截断约定敏感。这是 I06 已记录现象，正文未依赖该数字，维持现有说明即可。
3. 不需要补训练、补 seed、重选 checkpoint 或重算主指标。方法无改善、方向不稳定均已作为有效结论呈现；额外 seed、按 AOI 分组区间、因果隔离属于单列探索，不构成核心缺口。

**最终状态：** 29 项 27 PASS / 2 UNKNOWN 属实；RQ1/RQ2/RQ3 在声明范围内 ANSWERED；I02/I03/I04/I07 四项 FAIL 均有直接证据关闭；唯一待改为上述缺失链接。
