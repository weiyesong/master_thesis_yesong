我已读完档案重点章节与全部直接产物，并核对了上一轮协调结论。下面是审阅报告。

# 一、总判断与对上一轮结论的修正

**总判断。** 上一轮"可直接写范围明确的实证论文"的判断成立，但要按三处收窄。直接证据支持的范围是：固定 2 FM × 4 数据集 × 2 适配的描述性回答，deterministic 每格 3 seed，MC 仅 4 格 3 seed，Deep Ensemble 每格 1 组，TS 仅 EuroSAT。所有 PASS 只覆盖"产物存在、数值可复算、结论已限定"，不保证学位验收。

| 项目 | 状态 | 范围与直接证据 |
|---|---|---|
| 48 个 deterministic run 与配对差值 | PASS | `reports/core_rq_completion_20260921/coverage_matrix.csv`、`RESULTS_SECTION.md`；主指标已独立重算 |
| RQ3 中 TS 覆盖 | PASS，但只有 4/16 格 | `published/tables/thesis_master_results.csv` 中 12 个 TS 条件为 N/A；TreeSatAI 历史验证集拟合 TS 12 行保留在 `published/tables/historical_treesatai_temperature_scaling.csv` |
| MC 同权重三方对照 | PASS | `mc_dropout_three_way.csv` 24 行含 D/Off/MC；注意 `coverage_matrix.csv` 的 validity_note 仍写 "MC lacks same-weight dropout-off comparison"，与三方表矛盾，属文档过期 |
| RQ3 的"计算代价"部分 | 部分 UNKNOWN | 主表中 MC 推理时间为 UNAVAILABLE_NOT_RECORDED，TS 拟合时间未记录；只有训练秒数和 deterministic 单次推理秒数 |
| DOFA–EuroSAT 六个历史 run 的归一化来源与训练时代码 | UNKNOWN | `provenance_followup.md`、`checks.csv` C01/C06 |
| Pre-UQ 冻结文件是否存在 | PASS | `reports/pre_uq_protocol_freeze.md` 日期 2026-08-25，档案 §1 的疑问已解决；但它是 deterministic 结果已知后的事后冻结，只对随后的 MC 训练是前瞻性的 |
| 档案所称"62 tests passed" | UNKNOWN | 我未复跑；只建议引用仓库中可复现的日志 |

**档案中被上一轮忽略或需加强的明确承诺。** 档案 §2 与 §3 是导师要求的锚点，其中三项在当前产物里覆盖不全：一是 RQ3 明确包含"计算代价"，MC 与 TS 的推理/拟合开销缺实测；二是 RQ3 问的是三种方法是否改善校准，TS 在分割与 TreeSatAI 均为协议排除，论文必须把"TS 结论只限 EuroSAT"写成范围限制，而不是把 N/A 当作已回答；三是 §3 RQ2 把"confidence on errors / reliability shape"列为关注对象，当前 `RESULTS_SECTION.md` 只有 signed gap 与逐箱表，没有错误样本上的置信度分布。档案 §18、§19、§21、§22 的 OOD、标签稀缺、CKA、AU/EU 交互都被 §43 与 §45 明确降为可选，不是承诺。§4.4 与 §5.5 的"超越 metrics-only"是用户本人的反复立场，档案把它定位为讨论与诊断层，不是新增核心实验的理由。

上一轮已撤回的过度解释不再重复：代理 MI 下降伴随错误增多不证明 AU/EU 定义错误；高相关不证明无独立信息；交互项非零不证明物理来源耦合；CKA 是关联不是因果；三方 Off−D 含训练轨迹差异。三方表本身就给出反例：SpaceNet7 DOFA full seed42 的 D 与 Off 的 NLL 差异极大，而两者的最佳 epoch 分别是 23 与 5，这是训练轨迹差异，不是 dropout 推理效应。

# 二、分层评估与每步设计

**层 1：核心 3 RQ 实证论文。** 已有：全部 deterministic、配对差值、EuroSAT TS、24 组三方 MC、16 组 DE、类诊断、逐箱表。未知：历史来源、计算代价实测。这一层应成为当前完成门槛。需要的补充全部是"无需计算"：把 TS 范围与 TreeSatAI 历史 TS 时序写进方法章；在计算代价表中把 MC 写成 T 倍单次推理的解析代理并注明未实测；修正 `coverage_matrix.csv` 的过期备注。

**层 2：行为效用与错误识别。** 类型：已有预测重分析。分类 parquet 已含 logits、probabilities、correct、margin、predictive_entropy；MC 原始 [N,T,C] 与 DE 成员 [N,M,C] 在 `research_data/manifest.parquet` 中登记；分割 npz 含 logits、correctness、valid_mask，MC 全测试图含 expected_predictive_entropy 与 mi_style_disagreement。设计如下。

| 任务 | 统计单位与错误定义 | 分数 | 指标与基线 | 失败边界与停止条件 |
|---|---|---|---|---|
| EuroSAT | 图像；top-1 错误 | 1−MSP、熵、负 margin；MC/DE 另加预测熵、期望熵、MI | AUROC 为主，因它不受错误率影响；AURC 必须同时给随机基线 AURC=错误率和 oracle AURC，报告 E-AURC；固定 coverage 80/90/95% 的风险 | 每 seed 约 40 到 50 个错误，AUROC 区间宽；方法间差异小于三 seed 范围或图像级 paired bootstrap 区间时写"无一致差异"，不加 seed。TS 后 AUROC 应几乎不变，若大幅变化先查代码 |
| TreeSatAI | 不用 exact-match 错误，其错误率约 75%，AURC 无信息；用 sample-label 二元决策错误，与主 ECE 定义一致；bootstrap 按样本不按决策 | 决策置信度 max(p,1−p)；样本级用 15 个 Bernoulli 熵之和对 Hamming 错误数做 Spearman | 逐标签 AUROC 仅限测试正例 ≥50 的标签；测试集正例数为 13、23、12 的三个标签只列支持数，不报指标 | 若逐标签结果随 seed 翻转，只报范围。不能把这套指标与 EuroSAT 的 top-1 指标混排 |
| CloudSEN12 / SpaceNet7 | 图像为单位：先算每图像内像素级 AUROC，再报跨图像中位数与四分位；池化像素 AUROC 只作次要指标并注明像素相关 | 像素熵、MC 期望熵、MI | SpaceNet7 必须按真实类别分层：背景像素内、建筑像素内、边界与内部分别报；否则"高 AUROC"只反映不确定性集中在建筑附近 | 无错误或无建筑的图像 AUROC 未定义，记 NaN 并计数。分组 bootstrap：SpaceNet7 按 12 个 AOI，CloudSEN12 按 ROI，用 `reports/dataset_manifests/*_actual_manifest.csv` 的 aoi/roi_id 映射 sample_id；同一组内同时抽取比较双方 |

允许主张："在本配置与测试集上，某分数对错误的排序能力为 X，方法间差异在/不在重复范围内"。不允许："某方法的不确定性更有意义"这类跨任务概括，或把 MI 与熵的 AUROC 差异解读为 EU/AU 含义。

关于"旧 test 已被查看"：以上全部是同一测试集上的事后诊断。运行前先写一页分析说明，固定分数定义、coverage 点、支持数阈值与分层规则，标注日期；论文中称"事后描述性诊断"，不称预注册验证。目前 run 目录只有 `predictions/test`，没有验证集预测导出，因此无法做"验证集探索、测试集报告"的安排；若想有，需要对 72 个 checkpoint 做验证集单次推理，属于新推理，只在时间允许时做。

**层 3：解释性机制分析。** 类型分两种。logit 几何是重分析：分类与分割 logits 均已保存，可比较 frozen 与 full 的 logit 范数、top1−top2 margin 在正确与错误样本上的分布，并与 ΔECE、ΔNLL 并列。CKA 是重分析但覆盖不全：deterministic 测试嵌入只有 TreeSatAI 两 FM 与 EuroSAT Panopticon 共 18 个 run，六个 DOFA–EuroSAT 历史 run 没有嵌入，只能用 MC 训练 run 的嵌入替代并注明是另一组训练；分割 24 个 run 有 [N,768] 全局表征。设计：同测试样本上 frozen 对 full 的 linear CKA，每个模型–数据集块 3 个值，与该块的 ΔECE/ΔNLL 一起列成 8 行表格。允许主张：描述性关联。不允许：回归或 p 值，因只有 8 个块；也不能说"表征漂移导致校准变化"，因为 frozen 与 full 的学习率、weight decay、预算均不同，见 `checks.csv` R2.2。干预阶梯、同配方对照、Hessian 均为新训练，不建议；若坚持只做同配方对照，也要 2 FM × 1 数据集 × 2 条件 × 3 seed 共 12 个 run，且它只服务"解冻本身导致 X"这一新增结论。

**层 4：AU/EU 干预交互。** 两个操纵因子不能真正隔离：输入退化同时抬高期望熵与 MI，减少训练数据或改变适配也同时改变两者。所以交互对比 C 只能证明"两种操作相互作用"，不能证明"两种来源耦合"。已有数据能支持的最强、也最诚实的版本是档案 §22.3 的干预稳定性：同一测试输入，MC 期望熵在 frozen 与 full 之间不同，说明该项依赖估计量。这是重分析，8 个块均有 seed42 的 frozen 与 full MC。写法：作为讨论章的一个图表，引用 Wimmer 等人的既有论证，不称新发现。若要加 corruption：类型为新推理，代码存在 corruption_type 接口但我未验证其实现，状态 UNKNOWN；限定 EuroSAT 与 CloudSEN12、高斯噪声与模糊、3 个严重度、seed42 的 D、MC 与 DE 成员；TS 保持干净数据拟合值。标签保持边界：CloudSEN12 不能用云遮挡类退化，因为云本身是目标类；RGB 三通道模型不能做波段丢弃；模糊会使分割边界标签失真，须只报内部像素。指标：性能、ECE、NLL、平均熵与 MI 随严重度的单调性。不做标签比例扫描与地理 OOD，前者是新训练，后者标签分布同时改变。

**层 5：新理论或方法。** 无产物，不承诺，不做。

# 三、止步点、必要性归类与立即处理

**建议止步点。** 完成层 1，加上层 2 的重分析，再加层 3 的 logit 几何与层 4 的干预稳定性图表，然后停止并写作。层 3 的 CKA 只在写作后仍有时间时做。不启动任何新训练。新推理只保留两项候选：MC 与 TS 的推理/拟合计时，以及分割 TS 用验证集拟合后套用已保存的测试 logits。后者只在导师认为 RQ3 的 TS 覆盖不足时做，并须写明验证集同时用于 checkpoint 选择，且是在测试结果已知后新增的条件。

**毕业必要与结论必要的区分。**

- 毕业必要：层 1 的三项无计算补充，即 TS 范围披露、计算代价代理表、过期备注修正。
- 为"不确定性有行为意义"这一结论必要：层 2 重分析。
- 为"适配改变置信度几何"这一结论必要：logit 几何重分析。
- 为"解冻本身导致校准变化"必要：同配方对照新训练，不建议。
- 为"DE 可靠改善"必要：每格再训 2 组 ensemble，不建议；当前 DE 只能作点估计。
- 为"跨训练 seed 稳定"必要：更多 seed；bootstrap 只表示固定模型下的测试抽样不确定性，不能替代。

**是否要求立刻修代码、重算或重训。** 否。当前证据没有指向测量错误：72 个 checkpoint 权重核对、100 份预测主指标重算、TS argmax 不变均已直接核验。需要立刻改的只有文档：`coverage_matrix.csv` 的 MC 备注，以及在方法章列出 MC 推理时间未实测。

**低性能任务的写法。** TreeSatAI Macro-F1 约 0.20 到 0.25，SpaceNet7 建筑 IoU 约 0.05 到 0.11。两者的校准与行为指标都要与任务性能并列，并明确低性能下的 ECE 含义：在 SpaceNet7 上模型近乎总预测背景，像素准确率高与建筑 ECE 低都不代表可用。不要为提高性能补训练，档案 §4.11 已把低建筑 IoU 定为真实结果。
