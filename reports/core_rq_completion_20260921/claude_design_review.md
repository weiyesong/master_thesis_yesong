## 复核记录（Claude Code 独立审查，2026-09-22）

**结论先行**：补评估设计成立，当前运行的 runner 版本可用于 I04 关闭，不需要重跑或调整脚本。同权重对照足以把 MC−Off（固定权重上的推理效果）与 MC−D（总方案）分开，但 Off−D 只能作为"配方加轨迹"差异披露，单 seed 的 8 个 cell 不能归因于 dropout 正则。已独立抽核分类 1 个 cell、分割 2 个已完成 cell，数字全部与管线一致。

### 一、实际读取的证据

- 标准全文、上轮 issue_register.csv、mc_dropout_off_gaps.csv、classification_dropout_off_three_way.csv 及其生成脚本 recompute_classification_dropout_off.py。
- 本轮 SUPPLEMENT_PROTOCOL.md、baseline_integrity.json、两份 GPU 日志、build_core_rq_completion.py。
- runner scripts/complete_segmentation_dropout_off.py 读了两次。首次读到的是 23:42 之前的旧版，没有"同 MC checkpoint hash 断言"。当前磁盘版（sha256 前缀 af1fe73d，大小 6877 字节）在第 35 行加入了该断言和运行前后源码 hash 核验。字节码缓存记录的源码时间戳为 23:42:01，运行中的两个主进程启动于 23:42:37，两个已完成 cell 的 completion.json 记录的代码 hash 也是 af1fe73d。因此正在执行的就是当前版本。旧版产出的两个不完整 NPZ 已移入 interrupted_before_completion，未被引用。
- 旧训练/MC 代码：segmentation_pipeline.py 的解码器、指标累加器、导出与训练循环，run_experiments.py 分割训练与 checkpoint 选择，c5_mc_dropout_inference.py，mc_dropout.py，c5_prepare_mc_dropout.py，mc_dropout_protocol.md，run_registry.json。
- 配置对比：cloudsen12 与 spacenet7 的 D 配置与 C5 MC 配置 diff，数据段、训练预算、早停、选择规则完全一致，唯一实质差异是 decoder dropout 0.1 与 prediction_export 关闭。
- 两条训练轨迹 training_history.json、model_audit.json、code_snapshot.json（D 与 MC 各一）。
- 各 cell 的 D run_summary、MC manifest、MC 训练 run_summary。

### 二、独立检查与数值

分类抽核 EuroSAT / DOFA / frozen / seed 42。我从 MC 训练 checkpoint 直接取 BN 统计量与线性层权重，对 MC 推理保存的 backbone 表征做 eval BN 加线性层，重导出 Off 概率，与上轮 Off 预测逐元素最大差为 0。MC 均值概率与 30 次 pass 均值最大差 3.6e-7。三方 ID 与标签逐一相同。

| 指标 | D（原 p=0） | Off（同权重） | MC（30 次均值） | MC−Off | Off−D | MC−D |
|---|---|---|---|---|---|---|
| accuracy | 0.983051 | 0.984893 | 0.985262 | +0.00037 | +0.00184 | +0.00221 |
| NLL | 0.058109 | 0.050414 | 0.049743 | −0.00067 | −0.00770 | −0.00837 |
| Brier | 0.026906 | 0.024169 | 0.024561 | +0.00039 | −0.00274 | −0.00234 |
| ECE-15 | 0.004231 | 0.004339 | 0.008762 | +0.00442 | +0.00011 | +0.00453 |

所有数字与上轮 CSV 一致到小数点后 15 位。Off 对 MC 的 argmax 只有 1/2714 个样本不同，Off 对 D 有 23 个不同。

分割抽核两个已完成 cell。我用 NumPy 从 predictions.npz 的 float32 概率独立算 mIoU、像素精度、NLL、Brier、ECE-15，与 metrics.json 最大差 2.7e-9。softmax(logits) 与保存概率最大差 2.1e-7。三方 sample_id 顺序与 label 数组逐位相同。completion.json、MC manifest、上轮 gaps.csv 三处 checkpoint hash 相同。predictions.npz 的 sha256 与 completion.json 一致。

| CloudSEN12 / DOFA | MC−Off ECE | Off−D ECE | MC−D ECE | MC−Off NLL | Off−D NLL | MC−Off mIoU |
|---|---|---|---|---|---|---|
| frozen s42 | −0.0048 | +0.0186 | +0.0138 | −0.0068 | +0.0091 | −0.00007 |
| full_finetune s42 | −0.0025 | −0.0101 | −0.0126 | −0.0233 | −0.0157 | −0.00007 |

frozen 这一行说明为什么必须要这个对照：原报告 MC−D 的 ECE 上升 0.0138，拆开后推理侧其实降了 0.0048，全部上升来自训练侧变化。像素 argmax 上，Off 对 MC 只有 0.28% 不同，Off 对 D 有 8.96% 不同。

### 三、设计缺陷判定

**接受，需在报告中处理（不需重跑）**

- Off−D 含训练轨迹混杂。D 与 MC 用同一 seed 42，但 dropout 消耗随机流，轨迹分叉。CloudSEN12 DOFA frozen 的 D 最佳 epoch 16/26，MC 为 23/33。SpaceNet7 DOFA full 三个 seed 的 D 最佳 epoch 为 23、16、24，MC 为 5、10、14，MC 系统性更早停。8 个单 seed cell 没有噪声底。4 个鲁棒性 cell 有 D 三 seed 参照：CloudSEN12 Panopticon frozen 的 D ECE 跨 seed 范围 0.0365 到 0.0500，SpaceNet7 DOFA full 为 0.0450 到 0.0569，EuroSAT DOFA frozen 的 Off−D ECE 全部小于 D 的 seed 间波动。建议三方表加入两条轨迹的 best_epoch 与 last_epoch，并把 Off−D 标注为"配方加轨迹"差异，鲁棒性 cell 附 D seed 范围作参照。
- MC checkpoint 是用 dropout 关闭的 val mIoU 选出来的，即选择的是 Off 配置，MC 是选择后的推理改动。这一点协议里写了，正文应保留。
- build_core_rq_completion.py 的 point_estimate_category 只按符号分类。mIoU 差 −0.00007 会被记为"performance decreases"。表里有 delta 数值，但类别标签单独摘录会误导，建议措辞加上量级或只在正文引用 delta。
- 该汇总脚本引用 segmentation_off_verification.json，但 verify_segmentation_off.py 目前在磁盘上任何位置都不存在。这是待产物项。
- runner 的 `assert repeat_error==0` 是逐位相等。两个已完成 cell 通过。若某个 cell 因非确定性内核失败，会在写任何文件前中止，不会污染产物，只是需要重启。不是有效性问题。

**拒绝**

- 数据或 mask 不一致：数据段配置逐字段相同，ID、label 三方逐位相同，ignore、类别顺序、boundary_radius 均由同一 resolved_config 读取。
- 指标实现不一致：ECE 分箱在分类脚本用 searchsorted、在分割用 ceil，二者都是右闭区间且首箱含 0，我用 ceil 复算分类结果完全一致。
- 分类 Off 未用 backbone 前向所以"不算同权重"：full_finetune 的表征来自 MC 推理时该 checkpoint 的 backbone eval 前向，BN 与线性层直接取自同一 best.pt，我重导出差为 0。

**不变的项**：I01 历史 DOFA-EuroSAT 归一化来源继续 UNKNOWN。completion.json 里的 code_sha256 是评估时代码，不能改写成训练时代码。I05 无正文，不评。

### 四、待最终产物才能验收

- 12 个 completion.json 全部 passed，且每个 checkpoint_sha256 等于对应 MC manifest 与 gaps.csv。
- verify_segmentation_off.py 独立重算全部 12 个 cell，主指标容差 2e-6，直接核对 D、Off、MC 三方 ID、label、valid_mask。SpaceNet7 需覆盖 ignore 255、建筑 IoU、前景 ECE/NLL/Brier、边界指标及无边界图的 NaN 处理，并说明 float32 前景 clip 语义。
- mc_dropout_three_way.csv 24 行齐全；效应汇总按 seed42 主矩阵与 4 个三 seed block 分开。
- I02：正式生成器输出的图表带每箱计数、seed、分箱定义，decision 校准与 positive-label 校准分别命名。
- I03：最终表、图、正文与 applicability 对 TreeSatAI TS 状态一致，历史结果单列并给出时序。