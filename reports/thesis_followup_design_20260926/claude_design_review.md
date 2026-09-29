以下为独立评审报告（本轮只读，未训练、未推理、未新算研究指标）。

## 一、推荐默认止步点与对暂定设计的核查结论

**默认推荐执行到：A 包全量 + E 包的解析 toy。** 两者都是 0 新训练、0 新 FM 前向。B 包的"仅用已存 embedding"子集也是 0 前向，可作低优先附加。C、D、E 的 EO 版本都不建议默认执行，只在论文明确要新增对应主张时才开。

**核查依据（本轮直接读取）：**
- 主表 136 行，INDIVIDUAL_SEED 记录为 D 48、MC 24、TS 12，ENSEMBLE 16。训练秒数求和与协调者一致：D 45.949 h，MC 18.911 h，合计 64.860 累积 run-wall-hours。MC 的 24 条推理耗时全部未记录。
- 84 个 checkpoint 路径全部在盘。研究归档 208 项，分类/分割产物角色与 roadmap 一致。
- MC 推理实现是 backbone 特征算一次、head 或 decoder 跑 30 次（`scripts/c5_mc_dropout_inference.py:380`，分割在 623 行附近）。因此 T=30 的成本是 head 级，任何把它写成 30 倍 backbone 时间的表述都错误。
- 分类 Off 对照是"保存的确定性特征 + eval 模式 head"，分割 Off 是整模型一次前向。Off 文件不在 research_data 归档内，分类在 core_rq_audit_20260918 下，分割在 core_rq_completion_20260921 下，12+12 个都存在。
- 测试集与分组单位：EuroSAT 2714 张，另有 2713 张 calibration 导出含 logits；TreeSatAI 2000 张、15 标签；CloudSEN12 975 patch 属 195 个 ROI（每 ROI 5 个日期）；SpaceNet7 1152 patch 只属 12 个 AOI（每 AOI 84–100 patch），与训练 AOI 无重叠。

**对暂定设计的推翻与修正：**

1. **A 包分割部分不需要退到 32 张子集做 MI。** MC 全测试图已保存 `mi_style_disagreement` 和 `predictive_variance [N,C,H,W]`。Gini 分解的 EU 项恰等于各类概率方差之和，所以 Shannon 与 Gini 两套分解在全测试集上都可由已存 MC 图直接得到。DE 的成员级量由 3 个 D 全概率文件按 sample_id 对齐得到。32 张子集只用于 T 子采样敏感性与逐 draw 的其他评分规则，且 MC 与 DE 必须用同一 32 个 ID。
2. **B 包"9 对 frozen/full"存在伪配对。** 冻结 backbone 下三个 seed 的 backbone 表示应完全相同，等于预训练表示。所以分类只有 3 个唯一冻结参照对 9 个 full 运行，分割只有 4 个唯一冻结参照对 12 个 full 运行；"同适配跨 seed 差异"在冻结侧恒为 0，只能在 full 侧算。补 DOFA–EuroSAT 时冻结表示只需 1 次前向，加 3 个 full 共 4 次，而不是 6 次。执行前须数值验证冻结 embedding 是否真的跨 seed 相同，因为保存的 `embeddings` 与 `backbone_representation` 是两个数组。
3. **C 包计数要区分 backbone 前向与 head 评估。** 输入被扰动后特征缓存全部失效，冻结模型也必须重跑 backbone。但同一冻结 backbone 可被该 cell 的 D 三个 seed、MC、Off 共享一次前向，前提是执行代码支持特征共享，当前导出脚本按 checkpoint 逐个前向，不支持。
4. **D 包旧 run 不可复用，应按 24 个新训练位置计。** 现行冻结配方与 full 配方本身就不同（head LR 1e-3/WD 0.01/50 epoch/patience 10 对 backbone 4e-4/head 4e-3/WD 0/warm-up/100 epoch/patience 15），DOFA–EuroSAT 的 full 还是不可动的历史件且归一化常数不同。统一配方下四个条件都要新训。另外，"仅输入模块可训"仍需反向传播穿过整个 Transformer 才能到达 patch_embed，单步成本接近 full FT，不是冻结成本。
5. **E toy 的 20 次配对重复是多余的自由参数。** Beta–Binomial 下所有量都有闭式：后验 Beta(1+k, 1+n−k)，H(p̄)、E_post[H(θ)]（digamma 表达式）、Gini 各项都解析；对数据 k 的外层期望可直接对 Binomial(n,p) 的 n+1 个取值枚举，得到精确格均值，无需重复。重复只用于展示抽样变异，数量任意且几乎零成本，应写明。
6. **A 包的 EuroSAT 错误数很少。** 各 D 模型准确率约 0.983，2714 张里只有约 40–60 个错分正类。错误识别 AUROC 的区间会很宽，方法间差异多半在噪声内，验收必须允许"无差异"。

## 二、各包设计、严格去重计数与相对基准

基准：72 个唯一训练 run；48 个 D 单次测试评估有耗时记录（合计 1159.6 s，硬件历史值）。下表的"比例"按 run 数或测试评估数计，不是 GPU 小时。

| 包 | 新训练 run | 新 backbone 测试前向 | 对 72 run 比例 | 成本 proxy | 复用条件 / 未知 |
|---|---|---|---|---|---|
| A 分析 | 0 | 0 | 0% | CPU 排序、分组 bootstrap、npz 解压 | 需 ~42 GB 分割文件逐个解压（seed42 核心），另 21 GB 若读全部 24 个 D 做 DE 成员 |
| B0 已存 embedding CKA | 0 | 0 | 0% | 18+24 个 [N,768] 小矩阵 | 冻结侧跨 seed 恒等待验证 |
| B1 补 DOFA–EuroSAT | 0 | 4（去重后）或 6 | 前向数 = 现有 D 测试评估的 8–13% | 每次 2714 张 DOFA 前向，历史单次 7–9 s | 必须用历史归一化常数；不恢复来源 |
| C 仅 D | 0 | 8 | 相当于现有 EuroSAT D 测试评估（12 次）的 67% | 8×2714 = 21 712 张图前向 | 扰动定义与强度须在 calibration split 上事先定 |
| C 加 TS/MC/Off/DE | 0 | 24（无特征共享）或 20（共享） | 相当于现有 EuroSAT 全部测试 backbone 前向（18 次）的 111–133% | 另加 head 评估：MC 每状态 30 次、Off 1 次、TS 为 logits 变换 | seed42 D 是 DE 成员可复用；43/44 需扰动前向 |
| D 模块 2×2，2 FM | 24 | 24 个新 test 评估 | run 数 33% | 用 EuroSAT 现有 cell 均值外推约 21 h 累积 run-wall（冻结 6 个 2.5 h + full 级 18 个 18.9 h） | 统一配方会改变 epoch/patience，实际只会更高；3 seed 只是描述性 |
| D 单 FM | 12 | 12 | 17% | 上表一半左右 | 结论只覆盖该 FM |
| E toy | 0 | 0 | 0% | 解析式，秒级 | 无 |
| E EO 版 | 9（100% 复用）或 12 | 冻结 backbone 共享则每强度 1 次，否则 12×2 | run 数 12.5–16.7%；按 EuroSAT/DOFA/frozen 均值 0.30 h 计约 2.7–3.6 h | 复用 100% 位置要求配方与归一化匹配，需核对 c5 配置 |

**A 包精确对象数。** 分类 68 个平均预测对象：D 24、TS 12、MC 12、Off 12、DE 8。其中唯一 checkpoint 只有 36 个（24 D + 12 MC），TS 是 12 个 D 的 logits 变换，Off 与 MC 同 checkpoint，DE 成员就是 D。MC 的 43/44 seed 只在 EuroSAT/DOFA/frozen 与 TreeSatAI/Panopticon/full 两格存在。分割 seed42 八格：D 8、Off 8、MC 8、DE 8 共 32 个全测试对象，再加两格 MC/Off 的 43/44 共 8 个作为稳健性，合计 40。每对象像素数 CloudSEN12 约 48.9 M、SpaceNet7 约 57.8 M。

**A 包分析规则。** 固定同一预测器比较排序分数（1−max p、预测熵、margin；EE 与 MI 只对 MC/DE 定义，D/TS/Off 标 NA 不填 0）。跨预测器比较必须并列错误率与错误数，AUROC/AP/AURC 都不单独排名。覆盖点预先固定为 100/90/80/50%。bootstrap 单位：EuroSAT 按样本并声明缺场景信息；TreeSatAI 按图像；CloudSEN12 按 195 个 ROI；SpaceNet7 按 12 个 AOI，并明确 12 组的区间极粗，不得改回 patch 级去"变窄"。TreeSatAI 15 个标签全部保留并报正例数（测试集 Tilia 12、Populus 13、Prunus 23、Abies 50），主口径是逐标签二元错误与逐图 Hamming risk，exact-match 单列并报其约 75% 错误率。S2a 用 12 组已存 D/TS 同 checkpoint、原温度；中心化 logit 范数与 top1−top2 margin；TreeSatAI 改逐标签 log-odds。S2b 用 Shannon 与 Gini 两套分解，检查恒等式残差。

**B 包。** 只做 linear CKA 与类别可分性关联，报告 CKA 变化与校准变化的关联，不写"导致"。分割表示是最终 backbone 特征的全局均值，不能做层级或空间分析。

**C 包。** 限 EuroSAT、一类保标签退化（如高斯模糊，在 64×64 原生分辨率上施加再 resize）、两档强度。强度由物理尺度或 calibration split 上的标签无关技术检查确定，测试集不参与选择。接受结果为任何方向，不要求"不确定性随强度单调增加"。

**D 包模块边界核实。** DOFA 的 `patch_embed` 是 `Dynamic_MLP_OFA`（波长驱动的 `weight_generator` 与 `fclayer`），Transformer 主体是 `blocks` 与 `norm`，位置编码为固定正弦。Panopticon 的 `model.patch_embed` 是 `PanopticonPE`，内含 patch 卷积与通道注意力 `chnfus`（`ChnAttn`/`ChnEmb`），主体是 ViT `blocks`；SAR 专用的 `embed_transmit/receive/orbit` 在现行 full 配置中已结构性冻结。现行代码只有 `freeze_backbone` 布尔量加 `expected_frozen_backbone_parameters` 名单，优化器只分 backbone/head 两个 LR 组（`scripts/run_experiments.py:1484`）。理论上可以用名单把 Transformer 全部参数列为冻结实现"仅输入模块"，无需改代码，但 `adaptation_mode` 标签会写成 full_finetune 且审计不变量不为此设计，执行时需要一个小而受控的代码修改。避免混杂的必要控制：四条件统一 head LR、统一 backbone 组 LR、统一 WD、统一 epoch 上限与 patience、统一归一化常数、相同 head 初始化配对，并保存实际参数更新量。可训容量差异（输入模块几百万参数对 Transformer 约 8600 万）不可消除，只能承认。

**E 包 toy 检查。** 设计成立：p 是已知歧义的 oracle，H(p) 是参照；n 是训练支持；Beta(1,1) 后验积分是估计器，不是"true EU"。建议按"负对照"定位：两个来源在生成机制上独立，但熵基交互项 C 仍可非零，这只反映熵的非线性与有限 n 的后验收缩，不识别耦合。用共同随机数耦合标签使 n0 为 n1 的前缀、p0/p1 共用同一均匀数是正确的配对。EO 版去重后为 12 个唯一训练位置（3 个 100% + 9 个 20%），若选 EuroSAT/DOFA/frozen 且配方匹配，100% 的 D 与 MC 三 seed 均已存在，新增 9 个。EO 版研究的是"训练支持×观测退化"下代理量的响应，不是 AU×EU 识别，最小必要性低，默认不做。

## 三、执行 prompt 必备要素、停止条件与待用户决定事项

**必须写进 prompt 的参数。**
- 授权范围：A 包与 E toy；禁止训练、禁止任何模型前向、禁止修改 research_data 与已封存 reports、禁止读取凭据。
- 输入路径清单：研究归档 manifest、分类 Off 目录、分割 Off 目录、三方表、主表、分割 32 张子集 ID 文件、EuroSAT calibration 导出。
- 对象清单：分类 68 个对象与分割 40 个对象逐一列出（数据集、模型、适配、方法、seed、文件路径），并标明哪些量对哪些方法为 NA。
- 分数与定义：熵以 nats 计；AUROC 与 AP 的正类为"错分"，ties 处理规则；AURC 与 e-AURC 只作摘要；覆盖点 100/90/80/50；Brier 与 Gini 在 EuroSAT、TreeSatAI、分割三种任务的归一化定义各自写死。
- 统计单位与 bootstrap：分组列名（`roi_id`、`aoi`、图像 ID）、重采样次数固定一次、随机种子固定、报告区间不做显著性宣称。
- 数值验收：sample_id 与标签 100% 对齐；从原始张量重算的准确率、NLL 与主表逐对象一致，容差写明；MC 均值概率与保存均值一致；Shannon 与 Gini 恒等式残差在浮点容差内；Gini EU 等于类方差之和。
- 资源约束：分割文件逐个解压、可用 mmap、单进程内存上限；不得一次载入多文件。
- 交付：带日期的分析说明先于运行写好并记录哈希；结果表 csv、图、每对象 NA 原因表、事后改动日志；探索性标注写入每个表头。
- 停止条件：所有清单对象处理完或标 NA 即结束；结果为无改善、无差异或更差都算完成；任何缺失文件、对齐失败、恒等式残差超容差则停下报告，不自行补前向。

**prompt 中必须显式禁止的扩 scope 行为。** "有时间就扩到其他任务"；尝试新扰动类型或调 T；因 AUROC 低而换分数或换覆盖点；按测试结果挑 cell、挑 seed、挑分组方式；为 32 张子集之外的分割样本重跑 MC；为 DOFA–EuroSAT 补 embedding；重 hash 65 GB 归档；重算并"修正"主表；写论文正文；把 30 次 draw、3 个成员、15 个标签、像素或同 AOI patch 当独立样本；把 TS 后 AUROC 变化判为 bug；删稀有标签或把未定义 AUROC 写成 0.5；以"改善>0.01"之类门槛作为验收。

**是否需要用户新回答。** 准备 A 与 E toy 不需要新答案，可按默认继续起草 prompt。以下问题不阻塞准备，但决定是否开 B1/C/D/E-EO：

1. 论文是否要新增模块归因或退化鲁棒性的实证主张。没有，则 C、D 不做。
2. 若做 D，选 EuroSAT（便宜、有 calibration split、错误极少）还是 TreeSatAI（错误多、模型弱、多标签）。
3. 4×3090 是否仍可用，以及可接受的累积 run 时长上限。
4. E toy 是否进入论文讨论章作为负对照图，还是只作内部参照。

本轮未执行任何新分析；所有计数来自主表、manifest、三方表、数据集清单与代码只读检查。
