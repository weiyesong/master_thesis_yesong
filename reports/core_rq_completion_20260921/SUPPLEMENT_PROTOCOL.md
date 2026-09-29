# 审计后补充协议（2026-09-21）

目标是完成上一轮 I02/I03/I04 的必要补充并整理正文，沿用现行实验范围与全部既定种子；本轮没有新的训练或方法选择。原审计快照 sha256:af23e14b2d75b8e64a69f582c35734075b5316873fc12c1f6e32af5e9e5890df。完整模型/数据/适配/seed矩阵及已排除项见 ../core_rq_audit_20260918/coverage_matrix.csv、actual_artifacts.csv；仍为64方法cell，52已有核心证据、12 TS NA。补充Off是匹配对照，不把矩阵扩成新的研究方法。

## 预先固定的补评估

对 run_registry.json 中全部12个分割MC checkpoint做一次全test dropout-off预测；不根据结果排除。CloudSEN12每run 975图，SpaceNet7每run 1152图。保留原数据加载、预处理、ignore mask、类别顺序，严格加载全部参数，model.eval()、no_grad、全部随机层关闭、BN buffer不变。保存每图IDs、mask、logits、probabilities、性能/概率/类别/建筑/边界指标与完整来源hash。分类12个Off复用上轮已生成产物，不重复backbone计算。

对照：D为原p=0独立训练配方；Off为p=.1训练checkpoint单次确定性推理；MC为同checkpoint既有30次概率均值。MC-Off估计在固定权重上采用当前有限30次随机推理的观测效果；Off-D包含训练配方及随机轨迹差异，不能纯归因dropout正则；MC-D是总方案差异。对24行完整保留，另按seed42主矩阵与四个3seed鲁棒性block汇总。30 forwards不是30训练重复，单个三成员ensemble不是3独立ensemble。方法不改善仍是有效结果。

## 指标与验收

沿用ECE15右闭(lo,hi]、首箱含0；单标签/分割top-label，TreeSatAI flattened binary decision confidence与correctness；positive-label及类概率校准单独命名。fraction单位。主性能EuroSAT accuracy、TreeSatAI macro-F1、分割mIoU，SpaceNet7同时建筑IoU。NLL/Brier为概率质量，不能作校准误差同义词。重算ECE10/30用于敏感性描述，不选最有利分箱替代主指标。

独立NumPy脚本直接从新预测重算，主指标容差2e-6；float32前景clip遵照源数值语义，保留任何差异解释。直接核验D/Off/MC全样本IDs/标签/ignore masks，checkpoint与原MC hash必须相同。有效边界像素为四邻域标签变化的半径1扩张；无边界图的边界指标未定义而不是0。

## 报告范围和不可恢复来源

核心TS仍仅EuroSAT；TreeSat4和seg8 cells保留NA。历史TreeSat TS（val拟合、test评估）结果另列，不称方法无效，公开结果早于范围冻结的时序，不归因作者动机。报告修复代码输出新版本 published/，原产物保留。

DOFA EuroSAT旧常数的计算数据来源及六个历史run训练时代码若不能恢复，C01/C06继续UNKNOWN。事后补hash不升级为训练时证据。新写的结果正文只对本报告内容负责，不声称审阅未提供的完整论文。没有发现新的实现/数据缺陷时，不补训练。

## 执行记录

runner为 scripts/complete_segmentation_dropout_off.py，两个GPU按registry序号奇偶分片；首次启动停止以补入运行前同checkpoint hash断言和运行前后源码hash核验；停止时两个Cloud DOFA run已在压缩预测，留下不完整NPZ（无completion标记），已移入 results/final_thesis/core_rq_completion_20260921/interrupted_before_completion/，不纳入任何结果。最终版本从头补评估，最终日志以最终执行为准。最终runner执行与独立Claude工具轨迹另存。baseline_integrity.json 检查时报告协作者已修改C6，故唯一hash变化为本轮授权的生成器修改；其余2393旧文件无变化。
