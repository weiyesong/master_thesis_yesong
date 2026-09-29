我已独立读完准备稿、结果正文、覆盖矩阵、三方表、主表、来源补查、复核记录、MC 协议与摘要、数据归档说明，并用主表逐 seed 重算了 TS、ensemble 与 MC 差值。以下是结论。

## 一、能否进入写作：可以，且应立即开始

**实验有效性：足够。** 48 个 deterministic run 全部有原始预测与独立重算（分类差 0，分割差 2.3e-8），三方 MC 对照 24 组算术核对为 0，发布表与 master 精确一致。这是一个证据链闭合的固定矩阵，不是"数 run"。缺陷是明确的、已披露的、且不影响所有主结论：DOFA-EuroSAT 六 run 来源 UNKNOWN 只削弱 EuroSAT 上跨 FM 的 ECE 比较，不影响 RQ2 同模型配对与 RQ3 同权重对照。

**学术主线：成立但需换叙事。** 三个 RQ 目前是"基准问题"叙事。我认为更强的主线是：**在 EO 基础模型上，常用 UQ 方法给出的"改善"高度依赖任务、指标与训练随机性，其可靠性远低于文献默认。** 现有数据恰好在每个 RQ 都提供了反例，这是主线的证据而非弱点。

**贡献强度：中等偏描述性，硕士可接受。** 没有新方法，但有三点文献中少见的实证纪律：同权重 D/Off/MC 三方分解、逐 seed 保留的 24 组配对、关键类与决策级校准的分离。

**外部验收：无法判断。** 我不知道学院是否要求统计检验。唯一需要向导师确认的一项范围问题是：**接受"n=3 描述性结果 + 无显著性检验"作为硕士论文的实证标准吗？** 若接受，现有数据够；若不接受，也不该补训，而是把主张进一步降为描述。

## 二、两类不确定性的思考：如何写、哪些不能写

用户的想法（AE/EU 耦合、加法分解可疑）在文献中已有先行者（Wimmer 等 2023 批评条件熵/互信息；Valdenegro-Toro 与 Mori 2022 研究解耦），所以**不能写**"首次发现""推翻传统定义""证明 MI 不度量 epistemic"。

**可以写**的是 B 类批判性讨论，并用自己的数据做说明性而非证明性例证。我从 MC 摘要表里看到一个很好的例子：EuroSAT 两 FM 从 frozen 到 full，MI-style disagreement 从约 0.0076/0.0039 降到约 0.0009，降了 4 到 9 倍，而同时 accuracy 下降、NLL 与 Brier 翻倍。若把 MI 读作"模型知识不足"，这个方向是荒谬的。诚实的解释是：MI 项刻画的是**注入噪声的位置与特征尺度**，full 微调后头部特征分布改变，噪声敏感度降低，与"认知不确定性"无关。这正好支撑用户的直觉，且只需引用既有数值。

**必须写明**的边界：熵分解是代数恒等式，两项之和恒等于预测熵，这不构成物理来源被识别；本项目 MC 只在下游 head/decoder 加一层 p=0.1 dropout，ensemble 只有 3 个共享预训练起点的成员，两者对"函数空间"的采样极其有限，因此任何 EU 数值都是"该采样方案的分歧度"。这一节应命名为 "What the decomposition measures here"，放在讨论章，不放在贡献列表。

**不能写**的 C 类主张：任何"AE 真值/EU 真值"；"MC 与 ensemble 低估/高估 epistemic"；"分解失效的机制已被本文证实"；OOD 行为。

## 三、主论点与三个贡献

**主论点：** 对 DOFA 与 Panopticon 这两个 EO 基础模型，逐 seed、逐指标、同权重地审视后，主流后处理与采样型 UQ 方法的收益是条件性的、常常方向不一的、并伴随可测代价；论文的价值在于把这些条件清楚地界定出来。

1. **性能与经验校准的联合矩阵**（RQ1）。2 FM × 4 任务 × 2 适配 × 3 seed，含逐条件 signed gap 与逐箱可靠性。证据：分割 24 run 全部整体过度自信；EuroSAT 上过/欠方向随 seed 反转；SpaceNet7 pixel accuracy 0.91 以上而建筑 IoU 0.05 到 0.11，TreeSat 决策 ECE 0.01 而逐标签 ECE 0.034。结论是"总体 ECE 掩盖关键类"，这在 EO 场景有实际意义。

2. **适配配方对校准无统一方向**（RQ2）。24 组同 seed 配对。证据：EuroSAT full 三 seed accuracy 均降且 NLL 翻倍；CloudSEN12 DOFA full mIoU +0.052 但 ECE +0.024；TreeSat Panopticon 的 Macro-F1 差值跨零。已如实标注未隔离冻结单因素。

3. **后处理与采样 UQ 的条件性收益与代价**（RQ3）。我重算的关键证据：TS 在 EuroSAT 12 seed 上 argmax 零变化，但 ECE 仅 8/12 下降，T 在 0.955 到 1.409 之间，frozen 模型本已接近校准故 TS 几乎无事可做；ensemble 在 SpaceNet7 四格降低 ECE/NLL/Brier，却使建筑 IoU 下降 0.018 到 0.031，相对基线损失约 20% 到 40%，这是概率平均把稀有类推到阈值以下的决策级代价；MC 同权重对照在分割 12/12 改善概率指标而在分类无一致方向，且 CloudSEN12 DOFA frozen seed42 中总方案 MC−D 与推理项 MC−Off 方向相反，证明不做三方分解会误归因。

## 四、英文章节架构

1. **Introduction**：EO FM 的部署背景，为何 UQ 不能只看 ECE 均值，主论点与三个 RQ，贡献列表（上面三条）。
2. **Background and Related Work**（控制在 12 到 15 页）：EO 基础模型简述；校准指标与其陷阱（ECE 分箱、NLL/Brier 语义）；TS、MC dropout、deep ensemble 的原理；AE/EU 分解的标准定义与已有批评（Wimmer 等，Valdenegro-Toro 与 Mori）。不写贝叶斯推断综述。
3. **Experimental Design**：固定矩阵、协议冻结、checkpoint 选择、指标定义、seed 策略、三方对照设计、TS 范围排除及其时序、DOFA 来源 UNKNOWN 披露。图表：coverage_matrix 简化表、方法适用性表。
4. **RQ1 Results**：主表（RESULTS_SECTION 第一表）、signed gap 表、classification/segmentation reliability grid、accuracy-vs-ECE 与 mIoU-vs-ECE 散点、SpaceNet7 类概率可靠性图、TreeSat 正标签诊断图。
5. **RQ2 Results**：配对差值表、逐 seed 差值范围图。
6. **RQ3 Results**：TS 逐 seed 表、ensemble 相对成员均值表、MC 三方表与 16 格摘要、ECE 分箱敏感性、两张 MC 不确定性图。
7. **Discussion**：三个小节。7.1 低任务性能对结论的影响；7.2 What the decomposition measures here，用第二部分的 MI 例证；7.3 对 EO 从业者的条件性建议。
8. **Limitations and Future Work**：seed 数、单因素未隔离、采样覆盖、来源 UNKNOWN、TS 范围、无 OOD。
9. **Conclusion**。
附录：完整逐 seed 表、历史 TreeSat TS 表、重算脚本与哈希摘要。

## 五、时间分配

**写作前必须处理**：确定第二部分的叙事边界并写成一页提纲交导师；把 TS 排除时序与 DOFA UNKNOWN 的措辞定稿，避免后期返工。

**可选的一项小分析**：用 research_data 中 12 个分类 MC 的 [N,T,C] 原始张量，按样本计算 expected entropy 与 MI-style disagreement，报告两者的 Spearman 相关，以及各自对错分样本的 AUROC。NumPy 即可，无需模型。这直接服务用户关于耦合的讨论：若两项高度相关、且对错分区分力相近，就是"此处分解不提供独立信息"的描述性证据。只写成 7.2 的例证，不写成第四个贡献。

**明确不建议**：补 seed、bootstrap 置信区间、第三个 FM、传统 CNN 基线、OOD 实验、backbone 内 dropout、恢复 DOFA 来源。这些都不改变主论点，只会消耗写作时间。

## 六、正视低性能

TreeSat 宏 F1 约 0.20 到 0.25 与 SpaceNet7 建筑 IoU 约 0.05 到 0.11 是弱下游模型。对"描述固定模型输出概率的校准与方法差异"这一层主张，它们不构成无效：ECE、NLL、Brier 的语义不依赖模型有多强。对"可部署有用的 UQ""哪个 FM 更优"这一层，它们使结论不可推广，必须在讨论章明说：本文的 SpaceNet7 与 TreeSat 结果描述的是弱模型上的 UQ 行为，不能推断强模型。反过来，这两个任务恰好暴露了总体指标与关键类脱节的问题，这是它们的写作价值。TS 排除时序、MC 12 格 n=1、三 seed 只描述训练随机性而非普适稳定性，全部放在第三章与第八章两处，用同一措辞。

这套安排的核心判断是：现有证据足以支撑一篇"限定明确、反例齐全、边界诚实"的实证型硕士论文；唯一外部风险是学院对描述性统计的态度，这需向导师做一次确认，但不应阻止现在开始写第三、四章。
