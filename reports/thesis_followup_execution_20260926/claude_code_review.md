独立抽查完成。以下按“代码 bug / 科学定义选择 / 后续建议”区分，所有数字均由我用 numpy、scipy、sklearn 独立实现算得，未调用协调者的 `RankPlan`（仅在对拍时读取其输出）。

## 1. tie 期望 AURC、AUROC/AP、加权组 bootstrap、fractional 工作点

- 用随机整数分数（大量 ties）、随机 0/1 损失、随机非负整数权重做 5 组试验：`RankPlan` 加权结果与我把权重展开后的 sklearn `roc_auc_score`、`average_precision_score` 差异 ≤1e-16；AURC 与我独立写的“tie 块内按比例接纳”解析式差 ≤6e-16，与 4000 次随机破 tie 的蒙特卡洛均值差在噪声内（~5e-4）。
- 连续损失的 tie AURC 与穷举全部 tie 内排列的平均差 1e-16，说明调和数公式 `(prev_e − avg·prev_n)(H(cum_n) − H(prev_n)) + avg·count` 推导正确。
- 工作点：目标 qN 非整数时按块内同 fraction 接纳，四个 coverage 与我独立计算差 ≤6e-17；`acceptance(0.5)` 返回 [1, 2/3, 2/3, 2/3, 0, 0]，和为 3 = 0.5N，与标签无关。
- 组 bootstrap：多项式抽组后按组内全部图像计数权重，估计对象保持图像等权，与协议一致。

结论：无数学错误。

## 2. EuroSAT DOFA frozen DE

- 成员文件 seeds [42,43,44]，成员均值与保存 ensemble 概率差 0；成员 42 与 deterministic seed42 导出概率差 0；ID 集合与标签与 D42 完全一致；派生文件中按 ID 排序的 ID/标签/概率与我独立排序一致；786 组与 splits 表逐样本一致。
- 44 错误，30 个 p_max=1 且其中 0 错误。MSP 错误为正：sklearn AUROC 0.948945、AP 0.366098，我算 AURC 0.001079、oracle 0.000135、risk@0.9 0.003275，与 metrics.csv 一致到 1e-7 以内；unique_scores 2210 一致。
- 我用不同 RNG 的 1000 次组 bootstrap 得 AURC 区间 [0.00034, 0.00220]，协调者 [0.00036, 0.00225]，差异为抽样噪声。
- accuracy/Brier/NLL 与主表差 0。

## TreeSat DOFA frozen DE（已 done）

- 三个抽查标签（Abies 49 正例、Fagus 735、Tilia 12）的 MSP AUROC/AP/AURC 与 sklearn/我方差 ≤6e-15；逐图 Hamming 基率 0.09437、micro 30000 决策 2831 错误均一致；exact-match 0.2605、Brier、NLL（按原 float32-eps 截断）与主表一致到 1e-10。无 p∈{0,1} 饱和。

## 3. CloudSEN12 DOFA frozen DE

- 子集图 ROI_00044：由 raw 三成员概率重算均值与全测试保存均值差 3.0e-8（float32 舍入）；4463 错误、基率 0.08895 一致；MSP AUROC/AP/AURC 与 pixel_metrics 差 ≤5e-8（源于我用 float64 成员均值而协议用保存 float32 均值，符合协议 2 第 1 条）；工作点差 ≤7e-17。Gini 恒等式残差 3e-16，MI 非负。半径 1 边界支持 9720、边界错误率 0.35607 均一致。per_image 的 msp/entropy/NLL 均值一致到 1e-10。
- 全测试图像级：我从 per_image.csv 重算 MSP AURC 0.063262、oracle 0.051290、risk@0.9 0.127448、Spearman 0.8788，均一致；独立组 bootstrap 区间 [0.0581, 0.0687] 对 [0.0582, 0.0694]。说明：这一步只验证 per_image 表到 RC 的算术，未重新读取 975 图。
- 全局 pixel_accuracy/mIoU/四类 IoU 与主表差 0，NLL/Brier 差 ~1e-9。

## 4. E toy

- 我用 quad 数值积分而非 digamma 独立枚举 4 条件 × 两层，TU/EE/MI/Gini 三项与 condition_means 差 ≤5e-16。EE−oracle 在 a=0.1 为 +0.036/+0.004，a=0.4 为 −0.040/−0.005：前者是 Beta(1,1) 先验把后验均值拉向 0.5，后者是 Jensen 凹性。EE 的差分之差 0.0675 等于 ee_minus_oracle 的差分之差（oracle 项本身可加），即完全由有限样本偏差随 a 变号造成；MI 的 DiD −0.001 几乎为零。文件的“非可加性而非源耦合”解释正确，两层为镜像也已声明。

## 5. 快照、分组、CI/NA 传播

**代码 bug（已修，无需重算）**：`correlation()` 对 bool 数组调 `np.ptp` 崩溃，导致 run_3 在 TreeSat micro 单元中断；修复仅加了 float64 转换，`groups_for` 的 dtype=str 对已是字符串的列无影响。修前完成的 4 个 EuroSAT DE 对象我已独立复算一致，不受影响。code_versions 保留了修前版本。

**小缺陷**：D/TS 以外对象的 precision_sensitivity.csv 为无表头空文件（pandas 读取会抛 EmptyDataError），汇总脚本需跳过或写表头。

**科学定义选择（合理，需在报告写明）**：
- 固定子集 pixel_summary 的 AUROC/AP 均值只在 25/32 张有错误的图上取（7 张零错误图 NA），而 AURC/base_risk 在 32 张上取；同一表内分母不同，已用 valid_images 列标出。
- 32 图仅来自 30 个来源组，子集 bootstrap 组数少、区间宽属预期。
- 分割图像级 AUROC/AP 显式 NA（连续损失），Spearman 用作关联，符合协议。
- Off 与分割 Off 输入不在 research_data，hash 为本轮计算，来源为审计目录与三方表路径，已在 inputs_manifest 标注。

**结论**：抽查范围内未发现需要重算的输出；剩余 76 对象沿用相同代码路径，最终复核时建议再抽 1 个 MC 分类（核 raw 均值 vs 保存 mean 与 EE/MI）、1 个 MC 分割（核全图 variance 求和 vs raw）与 1 个 Off 对象的主表对拍。
