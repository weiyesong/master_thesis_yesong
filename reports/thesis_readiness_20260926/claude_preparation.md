**状态：待用户确认，未作论文充分性判断。** 以下为独立证据备忘，数值均由我从原始表重算，不依赖审计 PASS 标签。

**1. 现有数据能直接支持的科学陈述**

我从 `reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv` 的 84 行逐 seed 记录重算了 16 个格的均值与样本 SD，与表内 MEAN_STD 行最大差 1.1e-16。另从 `mc_dropout_three_way.csv` 的绝对指标重算 MC−Off 差值，残差为 0。自行核对的关键数量如下：

| 陈述 | 我核对的数值 | 原始表 |
|---|---|---|
| RQ2：full 微调在 EuroSAT 两 FM 上 accuracy 均下降 | DOFA 三 seed 差值 −0.0236 至 −0.0122；Panopticon −0.0287 至 −0.0158 | thesis_master_results.csv 逐 seed 相减 |
| RQ2：CloudSEN12 DOFA full 的 mIoU 升、ECE 也升 | ΔmIoU +0.0524，ΔECE +0.0237，三 seed ECE 差均为正 | 同上 |
| RQ3：TS 保持 EuroSAT 分类决策 | 12 个 seed 的 TS accuracy 与 deterministic 逐位相等 | 同上 |
| RQ3：Deep Ensemble 在 SpaceNet7 四格构成权衡 | mIoU −0.0110 至 −0.0065，建筑 IoU −0.0310 至 −0.0184，ECE/NLL/Brier 均降 | ENSEMBLE 行减 MEAN_STD 行 |
| RQ3：同权重 MC−Off 在分割 12 个对照中概率指标全部下降 | ECE、NLL、Brier 各 12/12 为负；mIoU 变化 −0.00022 至 +0.00011 | mc_dropout_three_way.csv |
| RQ3：分类 MC−Off 无一致方向 | ECE 6/12 下降，Brier 仅 2/12 下降 | 同上 |

因此可以支持的陈述是条件性的：两 FM 在四个 benchmark、两种适配、三 seed 下的性能与经验校准；full 微调对校准无统一方向；TS 在 EuroSAT 上严格不改变决策且收益因 seed 而异；ensemble 与 MC 的收益与代价依任务不同。CloudSEN12 与 SpaceNet7 全部 24 个 deterministic run 的整体 signed gap 为正，支持"整体平均过度自信"的描述。

**2. 当前不能支持的主张**

- **样本/重复限制**：每格仅 3 个训练 seed，MC 三方对照 16 格中 12 格 n=1（`coverage_matrix.csv` 中 mc_dropout 的 actual_seeds 有 12 行为单 seed 42）。不能写显著性、等效性或"无实际损失"。
- **对照限制**：frozen 与 full 同时改变 LR、weight decay、epoch 上限等，只能陈述"完整适配配方"差异，不能声称冻结开关的单因素因果。Off−D 混入训练轨迹，不能归因为 dropout 正则本身。
- **来源限制**：DOFA–EuroSAT 六个历史 run 的归一化常数统计来源与训练时代码绑定仍为 UNKNOWN（`provenance_followup.md`；`checks.csv` 中 C01、C06 两项）。DOFA 与 Panopticon 在 EuroSAT 上的细小 ECE 差不能归因为纯架构效应。
- **范围限制**：TS 仅 EuroSAT 四格进入核心表，TreeSatAI 四格与分割八格为协议排除的 NA，历史 TreeSatAI TS 12 个结果保留在诊断表但其时序须如实披露。跨 EO 分布普适结论、OOD、AU/EU 分解均不在证据范围内。

这些是边界而非缺陷。"无稳定改善""方向不稳定"按 `EO_FM_UQ_Core_RQ_Audit_Standard.md` 第 1、6 节属于合格答案，但该文件是项目研究标准，不是学院评分规则。

**3. 判断充分性前需要用户回答的问题**

必要项，直接决定"数据是否够"：
- 开题或导师书面承诺是否包含三问之外的主张，如 OOD、第三个 FM、传统基线、AU/EU 分解。若有，现有证据存在明确缺口。
- 学院对硕士论文的验收形式：是否要求统计检验或置信区间，是否接受 n=3 描述性结果。
- 导师对 DOFA 历史来源 UNKNOWN 的接受程度，以及是否要求 TreeSatAI/分割的 TS 结果进入正文。
- 剩余期限与算力预算：这决定 UNKNOWN 与 n=1 格是只做披露，还是有条件补齐。

仅影响写作安排、不影响充分性：主性能指标的呈现顺序，是否把 signed gap 表与可靠性图放正文或附录，MC 三方分解的篇幅，历史 TS 时序放在方法还是限制章节。

**4. 可用的贡献候选与证据定位**

- 两个 EO FM 在四个任务、两种适配下的性能与校准联合矩阵，逐 seed 保留，主表位于 `thesis_master_results.csv`，方向诊断位于 `RESULTS_SECTION.md` 逐条件 signed gap 表。
- frozen 与 full 的 24 组同 seed 配对差值及其跨 seed 不稳定性，位于 `paired_effects_from_predictions.csv`。
- 同权重 D/Off/MC 三方分解，可把总方案差异与推理差异分开，CloudSEN12 DOFA frozen seed42 中 MC−D 为 +0.0138 而 MC−Off 为 −0.0048，位于 `mc_dropout_three_way.csv`。
- 关键类校准诊断：SpaceNet7 建筑 IoU 与全像素建筑概率 ECE 的分离，TreeSatAI 决策 ECE 与逐标签 ECE 的差异。
- 可复核工程链：48 次逐 run 指标与箱表独立重算全部在 1e-6 容差内，脚本 `provenance_numeric_crosscheck.py` 不调用项目指标函数。

以上均未与文献比对，不声称创新性已被证明，也不判断是否符合学位要求。
