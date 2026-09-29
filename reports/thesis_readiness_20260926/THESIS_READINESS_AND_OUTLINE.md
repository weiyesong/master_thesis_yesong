# 硕士论文充分性判断与写作方案

日期：2026-09-26。基于用户确认的学位与定位、既有实验产物、Codex 直接核算、实际 Claude Code 独立复核和原始文献核查。

**判断：现有实验足以支撑一篇范围明确的遥感实证型硕士论文，可以现在进入完整写作；没有发现为这一定位必须新增训练、推理或重算主指标的缺口。** 目前尚不能支撑一篇以“证明 aleatoric / epistemic uncertainty 的真实耦合机制”或“推翻传统定义”为核心贡献的论文。论文验收要求尚未确认，研究材料足够不等于论文已经写完或必然通过答辩。

本次不重新执行 29 项全量审计，也不把另一 agent 的总结当通过依据。原 29 项结果仍为 27 PASS、0 FAIL、2 UNKNOWN；后两项是 C01/C06 中 DOFA–EuroSAT 历史来源未恢复。直接证据入口为 [上一轮完成报告](../core_rq_completion_20260921/COMPLETION_REPORT.md)、[逐项审计表](../core_rq_completion_20260921/checks.csv) 与 [本轮独立数值核算](independent_evidence_check.json)。本文件新增的是“这些证据如何构成论文”的判断，不将 UNKNOWN 改为 PASS。

## 1. 判断依据与边界

用户已确认：硕士、遥感方向；题目为 *Uncertainty for Foundation Models of Earth Observation*；目前没有已知的必须提出新方法或取得正向提升的要求；没有承诺新方法，但希望提出自己的思考；剩余时间有限。将“并为承诺”按上下文理解为“并未承诺”，将“两种 uncertainty”暂按 aleatoric 与 epistemic 理解。

| 判断对象 | 状态与含义 | 直接证据 | 最小后续处理 |
|---|---|---|---|
| 实验能否支撑限定范围的三个 RQ | PASS：可做固定配置、现有测试集上的描述性回答 | [主表](../core_rq_completion_20260921/published/tables/thesis_master_results.csv)、[配对效应](../core_rq_audit_20260918/paired_effect_summaries.csv)、[三方对照](../core_rq_completion_20260921/mc_dropout_three_way.csv) | 将结果与限定条件写进正文；不以改善为验收条件 |
| 覆盖是否透明 | PASS：64 个方法条件，52 个纳入、12 个 TS 条件为 NA；不是所有方法全矩阵三 seed | [覆盖矩阵](../core_rq_completion_20260921/coverage_matrix.csv) | 方法章明确适用范围、重复数及历史 TS 时序 |
| 历史训练代码与归一化来源是否完整 | UNKNOWN：DOFA–EuroSAT 六个历史 run；后续 MC 沿用其输入约定 | [来源补查](../core_rq_completion_20260921/provenance_followup.md) | 披露，限制架构归因和历史复现主张；保留原记录 |
| 真实 AU/EU 来源或耦合是否已识别 | UNKNOWN：目前没有真值或隔离来源的干预证据 | [MC 协议](../mc_dropout_protocol.md)、[MC 汇总及解释限制](../mc_dropout_summary.md) | 放入文献支持的讨论或未来研究；不宣称已证实 |
| 新算法是否是本论文必须交付 | NA：按用户当前定位，不是承诺的目标 | 本次用户确认 | 不为凑创新点启动新算法 |
| 学院/导师最终接受与全文质量 | UNKNOWN：尚无明确标准、完整论文稿或评阅意见 | 本次用户确认 | 向导师确认实证定位；完成写作和全文审阅 |

这不是“数量够了所以可以毕业”。充分性的依据是研究问题有可识别的比较对象、预测和指标可复核、反例与限制能构成连贯回答。正式要求未知不能简单等同于“导师只关心显著性检验”。

## 2. 三个研究问题各能回答什么

| RQ | 可以回答及现有实证 | 不支持的结论 / 缺少的证据 | 现定位下需要的工作 |
|---|---|---|---|
| RQ1：适配后的 EO FM 表现出什么预测性能与经验校准特征？ | 48 个 deterministic run 覆盖 2 FM × 4 数据集 × 2 适配 × 3 seed；可报告性能、ECE、NLL/Brier、置信偏差方向及类别诊断。SpaceNet7 像素准确率 0.9133–0.9265，但建筑 IoU 仅 0.0496–0.1056；总体分数不能替代目标类评价。 | 不能归纳所有 EO FM，也不能证明其优于非 FM；无统一传统模型对照，部分模型/预处理差异未隔离，历史来源仍有缺口。 | 写清固定配置与数据划分，显著展示关键类结果；无需修代码、重算或补训。若改为“FM 优势”问题才需匹配非 FM 对照。 |
| RQ2：两种完整适配配方如何改变性能与校准？ | 8 个模型–数据集块各有 3 个同 seed 配对。CloudSEN12–DOFA full 相对 frozen 的平均 mIoU +0.05236，同时 ECE +0.02367；TreeSatAI–Panopticon Macro-F1 配对差异跨零，不能声称稳定提升。 | 比较还改变学习率、训练日程等，不能解释成只改变冻结状态的因果效应；三个 seed 也不能证明广泛稳定性。 | 用均值、样本标准差和逐 seed 差值呈现条件性结果；无需为统一方向补训练。单因素归因需另设计控制实验。 |
| RQ3：TS、ensemble、MC 在什么条件下改善指标、有何代价？ | EuroSAT TS 12/12 保持准确率，ECE 8/12 下降；SpaceNet7 ensemble 四格 ECE/NLL/Brier 改善而建筑 IoU 下降 0.0184–0.0310。24 组 MC/Off 同权重对照中，分割 12/12 的 ECE/NLL/Brier 降低；分类 ECE 6/12、Brier 2/12 降低。 | 不能说某方法普遍更好、能识别真实 AU/EU、适用于部署/OOD。多数 MC 条件仅一个训练 seed；3 个 ensemble 成员不是 3 次独立 ensemble 实验。 | 以匹配对照写结果并报告计算代价与重复数；无需新增主实验。若声称误差筛查或 OOD 有效，必须补对应评估。 |

以上数值由 [verify_assessment_numbers.py](verify_assessment_numbers.py) 从主表、三方表与 MC 结果表直接核算；输入 SHA256 和完整数字保存在 [independent_evidence_check.json](independent_evidence_check.json)。已有原始预测级验证见 [分类验证](../core_rq_audit_20260918/classification_verification.json)、[分割验证](../core_rq_audit_20260918/segmentation_verification.json)、[新增 Off 验证](../core_rq_completion_20260921/segmentation_off_verification.json)。本轮没有声称重新跑过全量原始预测验证。

方法未改善、指标方向相反、适配差异不稳定，本身都是可报告的研究结果。只有无法建立有效对照、测量错误或证据不足，才应相应收缩回答。

## 3. 建议的论文主线与三项贡献

建议主线：**在所研究的 EO 基础模型与下游任务中，预测性能、经验校准和采样型不确定性度量并不总是给出一致评价；适配配方和 UQ 方法的作用，需要结合任务、类别、指标及对照方式解释。**

题目可以保留，在摘要第一段明确对象是“downstream predictive uncertainty and calibration”，而不是完整覆盖传感器误差、预训练知识缺口、OOD、因果来源分解等所有 uncertainty 问题。若允许副标题，可用 *An Empirical Study of Calibration, Adaptation, and Uncertainty Interpretation*。

可据实写成三项贡献：

1. **限定条件下的系统实证评价。** 联合评估两个 EO FM、四种遥感数据任务、两种适配配方的性能和概率质量，并保留类别诊断与 seed 差异。贡献是这组问题和配置上的独立证据，不声称创建数据集或首次研究 EO FM uncertainty。
2. **适配收益与可靠性指标的条件性发现。** 明确哪些任务上性能改善伴随校准变差，哪些差异随 seed 改变；给出可以复查的反例和限定结论。不能把“没有统一改善”写成“所有适配策略无效”。
3. **用匹配对照区分 MC 方案差异与随机推理差异。** D 是独立训练的 deterministic 模型，Off 是 dropout 训练模型关闭 dropout 的推理，MC 是同一模型进行随机概率平均。以三者避免把训练方案变化全部归给随机推理。CloudSEN12–DOFA frozen seed42 的 MC−D ECE 为 +0.013774，而 MC−Off 为 −0.004786，比较基准改变了结论方向。Off−D 仍含配方和训练轨迹差异，不是纯 dropout 正则化因果效应。

可复算产物和限制记录支撑上述贡献的可信度，但审计 PASS 数、agent 协作或文件数量本身不应被写成科学创新。

## 4. 如何安放 AU/EU 耦合的想法

建议讨论章提出：**对下游适配后的 EO FM，应该区分不确定性的来源概念、具体估计量，以及实验实际采样到的变化范围。** 这是能体现你判断力的观点；其一般思想有前人研究，应明确引用，而非包装为新理论。

三种不同问题要分别写：

- 概念层：相对于哪些观测变量、分辨率、标签定义和知识集合讨论“可约/不可约”？遥感中的混合像元、遮挡、有限波段和标注歧义是有价值的讨论例子，但本实验未隔离测量这些来源。
- 估计量层：expected entropy 与 MI-style disagreement 如何随采样分布和适配而改变？可以报告现有数据中的代理量变化；数学上可加不表示统计独立，也不自动赋予两个物理来源的解释。
- 来源层：观测噪声和知识不足是否实际耦合？若要把它升级成论文的核心实证结论，需要可控数据/噪声、标签重复或观测条件干预、模型知识变化等相匹配的设计。当前证据没有做到这一层。

对采样预测 \(p_t(y\mid x)\)，设 \(\bar p=T^{-1}\sum_t p_t\)，则

\[
H(\bar p)=\underbrace{T^{-1}\sum_tH(p_t)}_{\text{expected entropy}}+
\underbrace{\left[H(\bar p)-T^{-1}\sum_tH(p_t)\right]}_{\text{MI-style disagreement}}.
\]

这是选定经验采样分布下的恒等式。用下面的二分类教学例子即可说明“知道均值预测不够恢复分解”，不需训练：

| 采样方案 | 平均预测 | Predictive entropy（bit） | Expected entropy | Disagreement |
|---|---|---:|---:|---:|
| 每次都输出 (0.5, 0.5) | (0.5, 0.5) | 1 | 1 | 0 |
| 一半输出 (1, 0)，另一半 (0, 1) | (0.5, 0.5) | 1 | 0 | 1 |

对相同标签，两个方案的平均预测 ECE/NLL/Brier 相同，熵的分配却不同。这个例子是解释性数学事实，不是新的定理，不证明传统定义错误；实际完整采样张量当然能计算其自身的两项代理量。

项目中的 MC 只采样下游 head/decoder 的指定 dropout；ensemble 的三个成员共享预训练起点。因此，这些分歧量不覆盖全部预训练不确定性。没有真实 AU/EU 标签，不宜写“低估/高估真实 EU”或“已证实两来源耦合”。TreeSatAI 还要使用逐标签 Bernoulli 熵，不能把其 15 个标签当互斥类别套 softmax。

已有一个可用但必须限制的例证：EuroSAT seed42 中，DOFA frozen→full 的 MC disagreement 从 0.007572 降到 0.000869，而准确率从 0.98526 降到 0.95873；Panopticon 也出现同方向组合。可说“此处代理分歧和预测质量不同向”，不能说“因此 MI 与认知不确定性无关”，也不能归因于未经检验的特征尺度机制。full 两个 MC 条件都是 n=1。

Wimmer 等讨论过熵/互信息作为 AU/EU 度量的解释局限，Valdenegro-Toro 与 Saromo Mori 研究过估计中的相互影响；因此“首次发现耦合”不能作为贡献。[Wimmer et al., UAI 2023](https://proceedings.mlr.press/v216/wimmer23a.html)；[Valdenegro-Toro & Saromo Mori, CVPRW 2022](https://arxiv.org/abs/2204.09308)。

## 5. 七章组织建议

| 章 | 英文标题与写作任务 | 现有材料 / 建议核心图表 |
|---|---|---|
| 1 | **Introduction**：遥感应用中为何要同时考虑预测质量与可靠性；提出三个 RQ、范围和三项贡献 | 不承诺新算法；开篇用一个关键类或配对反例引出问题 |
| 2 | **Background and Related Work**：EO FM 与适配；calibration、proper scoring rules、UQ 代理量；TS/MC/ensemble；AU/EU 概念争议 | 按问题组织文献；校准不是全部 UQ。加入与最新相近工作的范围比较 |
| 3 | **Data and Experimental Design**：数据与划分、完整配方、模型、checkpoint 选择、方法适用性、seed、指标、D/Off/MC 对照 | 简化覆盖矩阵、三方对照图、指标语义表；明确不均衡、多标签、空间依赖和历史来源 |
| 4 | **Calibration and Adaptation of EO Foundation Models**：4.1 回答 RQ1；4.2 回答 RQ2 | deterministic 主表、带样本量的可靠性图、关键类诊断、逐 seed 配对差值；章节末分别回答 RQ1/RQ2 |
| 5 | **Evaluating Uncertainty Quantification Methods**：回答 RQ3；先 TS/ensemble，再 MC 三方对照，最后收益/代价 | 方法效果表、MC 三方差值、分箱敏感性、计算代价；一张图只回答一个问题 |
| 6 | **Discussion and Limitations**：跨 RQ 综合解释；AU/EU 解释边界；遥感含义；弱下游模型、单 seed、来源与外推限制 | 上述熵教学例子及现有代理量例证；未来探索单列，不能混成已验证结论 |
| 7 | **Conclusion**：逐 RQ 直接回答；对应三项贡献；未回答的范围 | 不在结论新增分析或把条件性结果改成普遍规律 |

附录放完整逐 seed 表、历史 TreeSatAI TS 结果与排除时序、详细协议和复算入口。主文展示每个结果所需的关键证据，不把整个审计日志塞入正文。

可直接取材的内容是 [RESULTS_SECTION.md](../core_rq_completion_20260921/RESULTS_SECTION.md)、[发布结果包](../core_rq_completion_20260921/published/final_thesis_results.md) 及其 [图目录](../core_rq_completion_20260921/published/figures)。它们是结果基础稿，不等于已完成引言、文献定位、方法叙述和综合讨论。

## 6. 时间有限时的完成路径

**当前必须做的是写作收束。** 建议按“第 3 章 → 第 4/5 章 → 第 6 章 → 第 2 章定位与第 1/7 章”的顺序形成完整初稿；文献阅读与引用整理并行贯穿，不等最后才补引用。先确保每个 RQ 都有“问题—设计—结果—回答—限制”，再润色英文。

写作时有四项不可省略：

1. TreeSatAI deterministic 条件均值 Macro-F1 仅约 0.21–0.24、SpaceNet7 建筑 IoU 约 0.05–0.11，应正面呈现。这限制对强模型、部署效用和普遍方法优劣的推论；不使已保存预测的校准描述自动无效，也不能当成 FM 在这些任务上能力不足的证明。
2. DOFA–EuroSAT 的 UNKNOWN 必须同时体现于方法与结论限制。同权重对照能固定输入条件，不能恢复训练期来源。
3. TS 的现行主范围是 EuroSAT；历史 TreeSatAI TS 已保留。不能写成“从未做过”，也不把结果产生后的范围冻结称作事先预注册。
4. 区分样本数、像素数、训练 seed 和 ensemble 成员数；当前均值/标准差是描述性统计，不写“显著改善”“等效无代价”或普适稳定性。

**可选的小补充，仅一项，非写作前置条件：** 若初稿完成后仍有余力，可从现存分类 MC 张量做 expected entropy、MI-style disagreement 与错分的描述性关联/区分分析，作为第 6 章小节。每个配置分别报告，保留类别和 seed 信息，并加入总熵或置信度作为参照。TreeSatAI 使用逐标签二元错分，注明汇总方式。高相关或相近 AUROC 不能证明无独立信息、真实来源耦合或分解无效；两项共享构造且受样本难度混杂。若无法在固定小范围内完成，直接移到未来工作，不延后主文。

当前主张不要求新增 FM、非 FM baseline、更多 seed、OOD、大规模噪声干预或重新训练强下游模型。它们可能回答有价值的新问题，但不因旧方法未改善而自动成为补救要求。若导师要求改变论文目标，再逐项判断需要补评估还是补训练，不预设“只能降主张”或“只能堆实验”。

建议向导师提交的一句话定位是：“本论文以两个 EO 基础模型在四个遥感任务上的校准、适配配方和 UQ 对照实验为主体，以可复核的条件性发现为贡献；AU/EU 的解释局限作为有文献依据的讨论，不承诺提出新算法。” 这项范围确认可以与写作并行。

## 7. 文献定位与协作记录

截至本次核查，已有与本项目明显相关的 EO FM uncertainty 工作，尤其是 2026-08-17 提交的 *Beyond Accuracy*。因此不能以“EO FM 尚无人研究校准”作为研究空白。本项目的可写差别在于所选模型/任务下的 frozen/full 完整配方比较及匹配 MC 对照；具体数字不能跨不同协议直接排优劣。[Lehmann et al., 2026](https://arxiv.org/abs/2608.16614)。更早的空间泛化与不确定性研究也应引用。[Ramos-Pollan et al., 2024](https://arxiv.org/abs/2409.08744)。阅读范围与区别见 [文献定位笔记](literature_positioning.md)。

Claude Code 的 [初评](claude_assessment.md) 与 [复核修订](claude_followup.md) 已保存；最终采用范围与未采用意见见 [REVIEW_RESOLUTION.md](REVIEW_RESOLUTION.md)。初评中的过强解释已明确修订，不能单独把初评复制进论文。可审阅的实际工具轨迹见 [evidence trace](claude_assessment_evidence_trace.jsonl)。
