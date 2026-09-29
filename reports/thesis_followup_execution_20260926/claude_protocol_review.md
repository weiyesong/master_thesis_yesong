核对完成，下面是独立审查结论。

## 已直接验证、与协议一致的项

- **分组单位**：CloudSEN12 全 manifest 上按 roi/equi/product 并集连通，test 975 图、195 ROI 合并为 184 组（roi↔equi 一一对应，合并仅由 product 造成；2 个 test 组经 3 个跨 split product 与 train/val 相连）。EuroSAT test 786 个 spatial_group，无跨 split；组大小 1–93，最大组占 3.4%。TreeSatAI test 2000 图坐标全唯一。SpaceNet7 12 AOI，每 AOI 84–100 图。
- **MC 方差约定**：分类 `np.var(axis=1)` 默认 ddof=0，分割 `unbiased=False`，与 Gini EU 的总体方差定义一致；分类保存 `[N,C]`、分割保存 `[N,C,H,W]` 逐类方差，可按类求和恢复。
- **E 闭式**：Shannon EE、Gini EE/EU 与 Beta 数值积分吻合到 1e-12，`TU−EE−EU` 残差 ~1e-17；四条件 Binomial 权重和为 1，项数 888。
- **tie 期望 RC**：用小数组对随机破 tie 的蒙特卡洛平均验证，解析期望风险一致（差异在 MC 噪声内）；全并列时风险恒为均值，随机参照=平均损失成立。
- **边界定义**：现行 `_boundary_mask` 即四邻接 GT 变化后 `max_pool` 方形膨胀再与 valid 求交，协议描述准确。
- 子集 32 图的 MC raw 与 DE member 文件 ID 顺序与固定定义完全一致，DE 成员 seed 42/43/44。
- EuroSAT 12 个 D 的错误数按主表换算为 43–124（frozen 43–46），与规格一致。

## 需要在实现前修正或写明的实质问题

1. **float32 饱和造成的大 tie 块**。EuroSAT 保存概率是 float32：DOFA-frozen-42 有 33 个样本 max(p) 精确等于 1.0，Panopticon-full-44 有 302 个（11%），`1−max` 只有 ~2000 个唯一值；而由 float64 logits 重算 softmax 则 2714 个值全部不同。两种来源下 AUROC/AP/AURC 会不同，不只是 1e-6 数值差。协议只说"定位并另列复核"，未固定主分析用哪一来源。建议事前固定：每个预测器统一用 float64 logits 重算（D/TS/Off/MC 均有 logits，DE 成员有 float32 概率则例外说明），并报告 tie 块占比与其中错误数。
2. **coverage 点非整数**。`mean_{k=1..N} risk(k/N)` 定义清楚，但 0.9×975=877.5、组 bootstrap 后 N 每次变化，工作点与 .01 网格对应的 k 取整规则（floor/ceil/线性内插）未写明，会影响 1000 次重复的一致性。
3. **1e-6 阈值与历史容差冲突**。历史 TS 导出校验对"保存 float32 概率 vs float64 softmax(logits)"用的是 atol 2e-6，实测差异达 1.9–2.1e-7 属正常，但分类 NLL/熵在极小概率处放大误差，会超出 1e-6。应把该已知边界预先列为"精度差异"类别，而不是执行时才决定。
4. **Gini EU 与现有摘要的因子 C**。c5 的 `mean_predictive_variance`（逐图行和 audit 摘要）是对类维度取均值，即 Gini EU/C。若用它核对，须乘 C（Cloud 4、Space 2）；否则会误判恢复失败。
5. **TreeSat 图像分数的 sum/mean**。保存的 `predictive_entropy` 是 15 标签熵之和，协议图像分数取均值。排序不变，但与保存值核对须除以 15；`confidence` 是均值口径则不必。
6. **边界半径与主表不可比**。现行配置 `boundary_radius: 1`，主表 boundary ECE 用半径 1；协议用半径 2，属新定义，不能与主表边界指标并排比较，需标注。

## 值得写明的非错误事项

- SpaceNet7 test 标签只有 {0,1}，valid_mask 全 True；98/1152 图无建筑像素。CloudSEN12 975 图中 242/644/304 图分别缺 thick/thin/shadow 类。逐图逐类 IoU/recall 的 NA 数量会很大，需在协议中明确 NA 传播规则，避免逐图均值隐性删图。
- E 中两层为 k↔n−k 镜像，逐层量对 X 等权平均恰等于单层值，888 项实际只有 444 个不同项；两层不构成额外证据，应说明以免读者误认为独立复核。
- Off 输入不在 `research_data/manifest.parquet`：分类在 `reports/core_rq_audit_20260918/classification_dropout_off/`（列 sample_id/label/logits/probabilities，无 correct），分割在三方表 `output_path` 指向的 `results/final_thesis/core_rq_completion_20260921/...`。inputs_manifest 须对这 24 份另行 hash。
- MC raw 子集文件无 `valid_mask`，需由 label≠ignore_index 派生并与全图文件核对。
- 12 AOI 的 bootstrap 百分位很粗；EuroSAT 93 图的大组将主导重采样方差。均非错误，但结论要按此解释。

整体判断：协议的统计单位、对照与分解定义没有发现方向性错误；上述第 1、2 项会直接改变所报数值，应在运行新增指标前补入协议版本 2。
