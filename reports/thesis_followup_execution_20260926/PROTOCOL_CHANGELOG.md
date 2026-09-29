# 协议与实现变更记录

本轮目录沿用启动日期2026-09-26，数值计算和整理完成于2026-09-27。冻结定义保存在 [ANALYSIS_PROTOCOL.md](ANALYSIS_PROTOCOL.md)；其v2 hash保存在 [source_snapshot.json](source_snapshot.json)，完成时再次核验。本文是变更索引，不覆盖冻结协议。

| 阶段 | 变更 | 时序与影响 | 直接记录 |
|---|---|---|---|
| 计算前v1 | 固定A100对象、E4条件、分数、单位、配对、bootstrap1000次与seed | 用户开启A/E，其余关闭；已有test上的探索性分析 | ANALYSIS_PROTOCOL.md；parameters.json；claude_protocol_request.md |
| 计算前v2 | 保存概率为主，logits64仅敏感性；fractional工作点；Gini类和与Tree标签均值；边界半径2改1 | Claude独立检查后、实际行为指标计算前写定；没有按test结果选择 | ANALYSIS_PROTOCOL.md版本2；claude_protocol_review.md；source_snapshot.json |
| 分类入口 | 异构NPZ缺class_names时由现行配置给出顺序，并核对可用归档字段 | 修复未完成对象的读取错误；不改标签/概率/模型 | classification_run.log；classification_run_2.log；code/analyze.py |
| 分类入口 | Spearman计算先把bool转float64 | 修复入口异常；此前完成EuroSAT DE结果未受影响并直接复算 | classification_run_3.log；classification_run_4.log；code_versions/；scientific_tests_final.log |
| 汇总读取 | 已知本地产物的object字符串组数组可读；以后保存Unicode；可选空CSV显式跳过 | 修复序列化兼容；分组值和指标不变 | aggregation_partial.log；code/common.py；code/aggregate.py |
| 像素配对 | 每一指标先取双方共同有效图，再计算配对均值与区间 | 防止无错误图的NA造成两个不同分母；最终表全部按此规则生成 | code/aggregate.py；pixel_paired_effects.csv |
| 文档与图表 | 区分图像等权风险与像素加权性能；补logit尺度与Shannon/Gini图；保留旧审计必要动作字段 | 使用既有结果，不新增分数、数据、训练或统计假设 | FOLLOWUP_COMPLETION_REPORT.md；code/figures.py；checks_delta.csv |

没有开启其他实验包，没有修改旧训练或推理实现、checkpoint、封存主表。原C01/C06的UNKNOWN没有因新分析而消失。

当前Git没有有效HEAD、没有已跟踪文件，不能提供有效的历史commit diff。交付使用逐文件SHA256作本轮代码revision，另保存所有本轮代码相对空文件的补丁、修前快照与最终代码的补丁，以及真实Git状态。它们证明所交付代码内容，不声称恢复了项目历史版本；见 `code_revision.json`、`analysis_code.patch`、`implementation_changes.patch` 和 `workspace_git_status.txt`。
