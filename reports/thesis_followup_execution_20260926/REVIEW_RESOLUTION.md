# 独立复核与问题处理

本轮使用实际Claude Code CLI，同一session为`5afb33d9-4dac-4ed1-8d17-2e1ec2a65b13`。协议、代码和最终结果分别审查。Claude承担对照/单位/结论边界的独立检查，并以独立算式读取代表性原始预测；Codex实现全部矩阵及直接数值核验。各自记录先保存，再交叉核对；PASS依据是原数据、实现和实际数值，不是对方的意见。

公开原文：[协议复核](claude_protocol_review.md)、[代码与代表数据复核](claude_code_review.md)、[最终复核](claude_final_review.md)。会话标识、工具次数和原始响应状态见[会话元数据](claude_session_metadata.json)，实际读文件与计算命令/输出见[公开工具证据](claude_evidence_trace.jsonl)。只导出公开tool_use/tool_result，不导出内部思考。

| ID | 发现与处置 | 最终状态 | 直接证据、影响及最小动作 |
|---|---|---|---|
| AE01 | float32概率饱和/ties来源未固定；采用保存概率为主，增加统一logits64敏感性 | PASS，接受问题，未采用全部重建的建议 | ANALYSIS_PROTOCOL.md v2；metrics.csv的tie计数；precision_sensitivity.csv。保持同一保存预测语义；MC平均logits不能替代概率均值。无需改旧表 |
| AE02 | 非整数coverage需明确定义 | PASS，修订定义 | code/metrics.py；scientific_tests_final.log中fractional/tie穷举。qN质量按同分块共同fraction接纳，不按标签排序 |
| AE03 | Gini方差类均值与类和、Tree熵sum/mean混淆风险 | PASS，显式转换并核对 | code/analyze.py；raw_classification_MC_verification.csv；各MC分割done.json。Categorical方差类求和，Tree保留Bernoulli语义；无补推理 |
| AE04 | 边界radius草案2与现行1不同 | PASS，计算前改1 | ANALYSIS_PROTOCOL.md v2；parameters.json；code/analyze.py boundary_mask；test_boolean_correlation_and_boundary_contract。没有运行radius2结果 |
| AE05 | 缺类/常量错误目标/连续图像损失不可伪填指标 | PASS，NA保留 | analysis_applicability.csv；label_metrics.csv；pixel_summary.csv的valid_images；pixel_paired_effects.csv的common_valid_images。NA是定义边界，不是缺失实验 |
| AE06 | 来源分组和12 AOI限制 | PASS，按实际metadata；限制保留 | grouping_summary.json；sample_groups.csv；原metadata路径与hash。Cloud184连通组；Space12 AOI；Tree图像单位仍有未知邻近相关性。更强泛化主张需新地域评估 |
| AE07 | 新分析异构字段、bool相关性、object字符串、空CSV入口异常 | PASS，修复并验证 | 失败日志、code_versions/、scientific_tests_final.log、全部100对象done.json、alignment_verification.csv。未完成入口重启；已完成对象数值直接核验；无需改模型代码 |
| AE08 | 像素指标不同NA集合可能改变配对分母 | PASS，取共同有效图 | code/aggregate.py；pixel_paired_effects.csv。逐项先共同finite再作配对差与组bootstrap；不补像素/seed |
| AE09 | 分割DE相对全部3成员新增RC超出主seed矩阵 | NA，明确范围 | DE_versus_three_member_core_metrics.csv；paired_effects.csv。新增全测试RC仅DE−D42，三成员均值使用既有核心指标；无额外分割seed分析 |
| AE10 | E代理交互不能解释为真实来源耦合/理论反证 | PASS，限制结论 | toy/posterior_terms.csv；toy/condition_means.csv；toy/verification.json；THESIS_RESULTS_DRAFT.md。888项是精确加权枚举，两个X层镜像；固定机制下代理有限支持效应 |
| AE11 | 整体拒识风险下降不代表目标建筑物效用提高 | PASS，增加所要求的目标内容诊断 | objects/spacenet7*/per_image_class.csv；full_image_class_coverage.csv；源labels/mask。50%图像只保留11.77%–18.82%建筑像素；这是内容保留量，不是recall |
| AE12 | 未做统一推理计时，不得写严格效率结论 | NA，分支关闭 | parameters.json；resource_profile.csv；resource_summary.json。只报告本轮CPU分析成本及旧run-wall总和，不作GPU加速比 |
| AE13 | C01/C06 DOFA–EuroSAT历史来源缺口 | UNKNOWN，保留 | ../core_rq_completion_20260921/provenance_followup.md；checks_delta.csv中的原直接路径。新计算不恢复历史代码/归一化统计，需真实历史记录而非虚填PASS |
| AE14 | Git无HEAD且无tracked files，无法给历史commit diff | UNKNOWN，仅历史revision不可用 | workspace_git_status.txt；code_revision.json；本轮代码SHA和analysis_code.patch完整交付。不能将内容hash当历史训练代码revision；不阻断本轮数值复跑 |
| AE15 | SpaceNet7 MC与Off曲线重合使橙线难以辨认 | PASS，接受最终审查建议 | code/figures.py将Off改为点线，figure_manifest图注明确可能重合；所有图重绘，数值表未改变 |
| AE16 | 连续Hamming损失的AUROC差值NA不能计入“区间跨0” | PASS，保留定义限制 | within_predictor_score_effects.csv的valid_bootstraps=0和NA；上述MC−Off概括仅针对AURC，MI−MSP的Tree结论仅针对micro_label_decision |

实际样本复核覆盖数学/ties/权重、EuroSAT DE、TreeSat DE、Cloud DE、分类MC、分割MC、Off与Space建筑物内容；它是独立抽查，不代表Claude重复算完全部100对象。Codex对全部100对象执行了来源/标签/分组对齐与核心指标核验。原29项沿用27 PASS、2 UNKNOWN，明确标注哪些本轮重查；新增8项为7 PASS、1 NA，不把表行数当独立证据数量。

最终审查第一次CLI在写最终响应前返回143，公开工具结果已在session中保存；恢复同session完成剩余检查。中断状态单独保留在claude_final_attempt1_status.json，没有伪造为成功输出。Claude自身一次按不存在的model列筛选像素配对表失败，后续按object_id修正；这不是指标代码失败。

最终复核没有发现必须重算、修模型代码或补实验的阻断问题。详见 [Claude独立算式与数值摘录](claude_validation/independent_recomputation_summary.md) 与公开工具原记录。Claude的样本PASS和Codex全矩阵检查分开列示，不把抽查包装成两次完整复算。
