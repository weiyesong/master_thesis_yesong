# Claude Code 协作复核与采用范围

实际 Claude Code session：`2b6ecc7f-85f6-4a80-b847-2e48eaa4e6eb`。独立[初评](claude_review.md)与[交叉复核](claude_followup.md)均保留；[Codex 独立笔记](codex_independent_notes.md)在读取初评结论前保存。直接证据以实验表、归档文件与本轮核查为准，不以两位审阅者意见一致作为通过依据。

**共同结论：** 现有三个 RQ 能支持限定范围的完整实证论文；档案体现的更深目标尚未全部验证。推荐 S0 核心写作 → S1 已存预测的行为分析 → S2 输出尺度和不确定性代理分析，然后收束。三步都可以不新增训练或模型前向。CKA、扰动、模块归因与 AU/EU 来源研究是各自有证据门槛的扩展分支。

## 接受 Claude 发现并直接核实的事项

1. **覆盖表备注确实过期。** 原表 16 条 MC 备注仍写缺少同权重 Off；实际 [三方表](../core_rq_completion_20260921/mc_dropout_three_way.csv)有 24 组完整对照。本轮生成 [coverage_matrix_review_view.csv](coverage_matrix_review_view.csv)，原始列及更正依据一并保留；不修改封存原件，不要求重复评估。
2. **耗时证据不完整。** [成本证据表](compute_cost_evidence.csv)直接摘自主表：MC 的 24 条推理耗时缺失；TS 的 12 条拟合/应用耗时缺失；DE 的 16 条推理时间由成员求和，聚合开销未记录。只支持操作次数/已有时间口径，不能声称精确全流程效率；需要此强主张时才补统一计时。
3. **理论愿景不等于已完成的实验。** 档案明确将 OOD、CKA、模块干预、AU/EU 理论分支降为可选。当前结果不能据此冒充机制证据。

## 初评建议的修订

| 初评建议 | 复核与最终采用 |
|---|---|
| 没有验证预测，需要 72 checkpoint 新前向 | Claude 撤回总数。实际另有 TreeSatAI 12 validation 与 EuroSAT 12 calibration exports；只补具体新分析真正缺少的数据 |
| TS 的 AUROC 应基本不变，大变化先视为可疑 | 已撤回不变保证；多分类 TS 保持单样本 argmax，不保证跨样本置信度排序 |
| AUROC 不受错误率影响，e-AURC可作通用修正 | 限定固定类别条件分布下的比例不变性；更换模型会换错误集合，仍需报告错误率和任务效用，面积指标不独立证明优劣 |
| TreeSatAI 只报告正例≥50的标签 | 撤回硬门槛；全部标签保留支持数、有效性和不确定性，避免删掉稀有类 |
| 用 MC embedding 代替缺失的 deterministic embedding | 不用于冒充 D frozen/full 比较；单独报 MC 或对所需6个D checkpoint导出特征 |
| 直接比较 raw multiclass logit norm | 改为中心化 logits 的范数和 margin；TreeSat sigmoid 采用另行匹配的定义 |
| Hessian/Jacobian都需要训练 | 改为需要额外梯度计算，未必需要训练；不把它们列成当前必要项 |
| RGB不能遮蔽波段；blur只能看内部像素 | 撤回一概禁止；按干预目标、输入信息和标签保持语义制定协议 |
| DE 稳定性必须每格再训2组 | 撤回通用数量；额外重复取决于主张与精度，3组也不保证普适稳定 |
| 成本补充/文档纠错是“毕业必要” | 改为当前研究范围和具体主张的要求；正式学位验收未知 |

Claude 复核仍未定位 EuroSAT calibration export；协调者已直接读取 12 个实际文件的 footer/schema 与 split/dataset 值。例：[DOFA frozen seed42 calibration parquet](../../results/final_thesis/c3_temperature_scaling_and_ensembles/classification/calibration_exports/dofa/frozen/seed42/predictions.parquet)。核查记录为 [eurosat_calibration_export_check.json](eurosat_calibration_export_check.json)。因此该数据是否存在已解决，不保留为未找到；这不等于已具备所有模型的 validation 输出。

## 保留的解释差异

Claude 倾向将 TreeSatAI exact-match AURC 限于附录，认为高 exact-match 错误率使其解释价值较弱。协调者认为高错误率本身不能证明排序无价值，应由应用的损失/拒识单位决定是否有用。最终方案以逐标签二元错误和逐图 Hamming risk 为主，exact-match 如纳入则单独定义并报告错误率，不因其难看删除，也不与 EuroSAT top-1 混排。该差异不影响推荐工作范围。

双方都赞成运行 S1/S2 前写一份有日期的分析说明，固定分数、比较范围、工作点和分组规则。已有 test 被查看，说明书不是追溯性“预注册”，也不自动形成独立确认性证据。

以上修订落实于 [最终报告](DOSSIER_ASSESSMENT_AND_EXPERIMENT_ROADMAP.md)。没有执行新实验，没有发现当前需要修模型代码、重算主指标或重训的实质错误。
