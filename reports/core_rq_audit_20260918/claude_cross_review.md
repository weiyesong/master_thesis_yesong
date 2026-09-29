核对完成。逐条给出接受/修正/保留意见及原始证据。

**1. C06 / C01 子范围：接受修正。** 原始证据确认 Codex 的表述：`reports/dofa_eurosat_final_manifest.json.source_snapshot.provenance_scope` 原文为 "Post-hoc snapshot of the preserved workspace source. Historical training-time Git commits remain unavailable and are not reconstructed."；六个 DOFA-EuroSAT run 目录的 `environment.json.git_commit` 均为 null，且目录内均无 `code_snapshot.json`（其余 42 个 C1/C2 run 及 24 个 MC run 都有）。归一化方面，六个 run 的 `resolved_config.yaml` 中 `data.normalization` 为空，实际常数来自 `scripts/run_experiments.py` L86-111 硬编码的 13 波段 `SENTINEL2_MEAN/STD` 表（RGB 取 1136.89/1120.77/1184.39），`final_training_protocol.md` L90 已声明来源未保存。修正后的判断：
- C06 整体保持 PASS（数值可从保存 logits 重算，我已重算 seed 42 与表值一致到 1e-16 级），但**DOFA-EuroSAT 六个 run 的"训练时代码版本"子范围改为 UNKNOWN**：只能证明预测与 checkpoint 一致，不能证明训练时代码等于事后快照。
- C01 整体保持 PASS，**旧 EuroSAT 归一化常数来源子范围改为 UNKNOWN**。不称泄漏、不要求重训；MC DOFA-EuroSAT 沿用同一硬编码常数，配对内部一致。

**2. R1.2：接受改判为 PASS，缺口拆分。** `thesis_master_results.csv` 全部 TreeSatAI 行的 `metric_semantics` 为 "multilabel: accuracy=strict exact match; NLL/Brier average sample-label decisions; ECE-15 flattened binary decisions"，`thesis_evidence_matrix.md` 亦写明。任务匹配指标（exact match、Macro-F1、逐 sample-label NLL/Brier）存在，不能凭旧 C6 列名判 FAIL。我的重算事实（负决策占 91.9%、正决策 ECE 0.050、逐类 ECE 均值 0.034、Quercus 0.123、5 类 F1=0）保留，但归口调整为：标签级/逐类校准不足归 **R1.4**（类别不均衡掩盖，TreeSatAI 与 SpaceNet7 并列，仍为需补表的缺口）；reliability 图（标签概率 vs 频率）与表内 ECE（决策置信度）定义不一致归 **C05/R1.3**。三处各记一次，不重复升级。

**3. TreeSatAI TS：接受。** 同意四点：C09 判定范围收窄为"最终产物间状态矛盾 + 决策时序未披露"，撤回任何暗示主观 cherry-picking 的措辞；排除理由不写成"数学无效"，标准 C02 允许披露后复用；无需新训练；不强制恢复为核心结果，最小补救即统一现行范围、显式保留历史诊断（含 ECE 变差 11/12、NLL 变差 10/12 的反例）并披露时序。撤回我关于分割 TS 补写"无独立 calibration split"理由的建议：协议未写该理由，审查不应代为发明；只需现行协议对分割 TS 的 N/A 有可引用的书面依据即可。

**4. checkpoint/聚合核验：不以总结为据，已自行抽查原始产物。** (a) EuroSAT/Panopticon/full_finetune 集成：直接读取三个成员 run 的原始 `predictions.parquet`（run_id 与 `member_predictions.npz` 一致），按 sample_id 对齐后概率算术平均，重算 acc 0.976419 / NLL 0.072574 / Brier 0.036389 / ECE 0.021987，与 `c3_classification_results.csv` 完全一致，与保存的 `ensemble_predictions.parquet` 概率最大差 0.0。(b) SpaceNet7/Panopticon/frozen/seed42 MC：`research_subset_stochastic_probabilities.npz` 形状 [32,30,2,224,224]，首图 30 pass 互不相同，其均值与 `aggregate_uncertainty_maps.npz` 对应图概率最大差 3.0e-8，argmax 一致率 1.0，熵差 3.0e-8。加上第一轮已重算的 SpaceNet7/DOFA/full MC 全测试集指标（与 manifest 完全一致），R3.5、R3.4 聚合部分、C06 数值部分维持 PASS。分割 44 份全量重算未完成前，其对应子范围我标"PASS（部分抽查）"，不阻塞结论。

**5. 三种差值的解释：确认。** 设 D = 独立训练的 p=0 确定性模型，Off = MC 模型同权重 dropout 关闭单次前向，MC = 同权重 30 次 dropout 概率均值。
- **MC − Off**：纯 MC 积分效应。同权重、同表征，唯一差别是随机遮蔽后的概率平均，这是对"MC Dropout 作为推理期 UQ 方法"的最干净估计。我的数字：EuroSAT DOFA frozen s42 ECE +0.0044（翻倍）、NLL −0.0007；TreeSatAI PAN full s42 各项 |Δ| ≤ 0.002。
- **Off − D**：训练配方效应（含 dropout 正则化与不同随机轨迹）。它不是 MC 方法的功劳；EuroSAT 例中 NLL −0.0077 几乎全部来自此项。因 D 与 Off 是不同 checkpoint，它与 RQ2 一样只是同 seed 设计配对。
- **MC − D**：现行报告值，等于前两项之和，是"带 dropout 重训并用 MC 推理"这一整套方案相对现有基线的总效应。它可以回答 RQ3 的实践问题，但不能归因给 MC 积分本身。
正文应三方并列，并明确 MC − Off 才支持对 Gal & Ghahramani 意义上 MC Dropout 的陈述；若 MC − Off 接近零而 MC − D 不为零，结论应写为"收益来自 dropout 训练而非贝叶斯近似推理"。

**需要更改的具体判断**

| 项 | 第一轮 | 修正后 |
|---|---|---|
| C06 | PASS | PASS；DOFA-EuroSAT 六 run 训练时代码版本子范围 UNKNOWN |
| C01 | PASS | PASS；旧 EuroSAT 归一化来源子范围 UNKNOWN |
| R1.2 | FAIL | PASS；缺口迁移至 R1.4（TreeSatAI 标签级/逐类）与 C05/R1.3（图表定义） |
| R1.4 | PASS 附补充 | 维持 PASS 但补充范围扩到 TreeSatAI；SpaceNet7 恒等列合并要求不变 |
| C09 | FAIL | FAIL，范围收窄为状态矛盾与时序未披露；删除"看似选择性删除"措辞；分割 TS 不补理由 |
| R3.4 | FAIL | FAIL 维持；分类 12 cell 由本轮新增证据补齐后可转 PASS，分割 12 cell 待单次 deterministic eval |

RQ 结论与 Codex 预计一致：RQ1 PARTIAL（图表类条件与正文，补图表后可给范围内草稿）；RQ2 ANSWERED（所采用方案差异，非纯因果）；RQ3 PARTIAL（分割同权重 MC 对照缺）。无新训练需求。