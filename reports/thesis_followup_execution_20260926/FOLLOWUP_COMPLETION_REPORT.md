# A + E 补充研究完成报告

启动协议目录日期2026-09-26；执行与整理完成日期2026-09-27。仅执行用户授权的A、E。**新增模型训练0、模型前向0**；B/C/D/F、额外分割seed、全测试DE分解及模型效率前向均未启动。

现有三RQ结论继续成立于已声明配置范围。本轮新增了错误识别/拒识、输出尺度、代理量及解析参照证据，能够把论文从“校准结果比较”推进为“校准、行为效用与代理解释的实证研究”。它没有证明真实EO AU/EU耦合，也没有提出已经验证的新算法或新理论。

## 1. 实际完成度与直接证据

| 内容 | 实际完成 | 直接证据 |
|---|---|---|
| 分类全测试分析 | 68预测对象：D24、TS12、MC12、Off12、DE8；40 EuroSAT、28 TreeSatAI | [输入清单](inputs_manifest.csv)、[全部指标](metrics.csv)、[逐标签表](label_metrics.csv) |
| 分割全测试图像级分析 | 32主对象；CloudSEN12每对象975图，SpaceNet7每对象1152图 | [逐对象数据目录](objects/)、[全图像RC曲线](curves.csv) |
| 分割像素诊断 | 每任务相同32图，8配置×4方法=1024图像–预测切片 | [像素摘要](pixel_summary.csv)、[共同有效图配对](pixel_paired_effects.csv)；逐图原表在各对象目录 |
| 核心指标重算 | 436个标量比较全部通过，最大绝对差5.77×10⁻⁹；另有108条配对/分解验证PASS、8条不适用 | [核心验证表](core_metric_verification.csv) |
| 对应关系 | 全100对象ID、标签、分组及分割mask一致；40 EuroSAT对象另对原split标签核验 | [对齐表](alignment_verification.csv)、[原标签验证](eurosat_original_label_verification.csv) |
| TS与MC直接验证 | 12组TS raw logits匹配D、argmax变化0；12分类MC×5字段=60项raw–summary核验通过；8分割MC固定子集核验全图方差/熵/均值 | [TS验证](TS_same_checkpoint_verification.csv)、[MC验证](raw_classification_MC_verification.csv)、各对象done.json |
| 统计 | 1000次配对组bootstrap；EuroSAT786空间组、TreeSat2000图、Cloud184来源连通组、Space12 AOI | [分组依据](grouping_summary.json)、[配对效应](paired_effects.csv)、[同预测器分数差异](within_predictor_score_effects.csv) |
| E解析参照 | 4条件、888加权后验项；恒等式及独立数值积分验证 | [条件均值](toy/condition_means.csv)、[解析验证](toy/verification.json)、[交互](toy/interactions.csv) |
| 科学测量测试和图表 | 10项小数组科学测试通过；11幅PDF＋PNG图 | [测试日志](scientific_tests_final.log)、[图表清单](figures/figure_manifest.json) |

这里的对象数、指标行数和bootstrap次数都不是独立训练重复数。分割主分析仍只有seed42以及一组三成员DE；没有把3成员算3组ensemble。

分割DE与三个单模型成员均值的对照使用[既有任务/概率质量指标](DE_versus_three_member_core_metrics.csv)；新增全测试图像级RC只与D42比较。分类的全部D成员已在A内，因此同时提供新增行为指标的DE−D42和DE−三个成员指标均值。未为了补齐分割成员行为矩阵开启额外seed分析。

## 2. 逐题回答

### A1：不确定性是否有错误筛查和拒识价值？

**ANSWERED，限这些保存预测及预定损失。** EuroSAT的MSP错误识别AUROC在40对象中为0.9221–0.9781；TreeSatAI的逐标签micro错误AUROC在28对象中为0.8120–0.8279。它们是不同任务和错误事件，不能混排冠军。TreeSat图像Hamming损失与标签决策的拒识曲线分别报告，不能拿exact-match准确率替代。

在16个Cloud配置/方法对象中，保留50%图像后的图像等权有效像素错误比例为0.0387–0.0678，而不拒识时为0.1202–0.1692；SpaceNet7对应为0.0177–0.0274与0.0693–0.0867。这说明预定图像分数能在当前测试集上筛出较低全像素损失的子集，**不构成未来部署风险保证**。[逐对象指标](metrics.csv)、[RC图](figures/risk_coverage_cloudsen12.pdf)。

**关键限制与EO解释：** SpaceNet7保留50%图像时，16对象实际仅保留11.77%–18.82%的真值建筑物像素。整体风险降低伴随大量目标内容被拒绝，不能概括为建筑物识别更可靠。真值只用于事后内容诊断，不用于构造拒识分数。[全测试类别保留量](full_image_class_coverage.csv)、[图](figures/spacenet_building_retention.pdf)。

缺少的更强证据：新地域/时间评估、独立选择后的部署拒识阈值验证、业务损失与容忍风险定义。本轮描述性问题不要求这些；若提出对应部署主张，需要补评估，而不是因本轮方法不够好就补训练。

### A2：MC的随机推理、MI等代理是否带来更好的筛查？

**ANSWERED，结论是条件性与反例，不是通用改进。** MC−Off的MSP AURC变化多数很小，方向随配置改变，多数条件区间跨0；这不能证明等效，也不支持跨任务一致收益。[同权重效应](paired_effects.csv)，过滤`MC_minus_Off`。

固定同一个预测器和同一错误集合时，TreeSatAI全部10个既有MC/DE对象的MI逐标签micro错误AUROC低于MSP，差值为−0.1520至−0.0190；相应图像组bootstrap区间均位于0以下。EuroSAT10个MC/DE对象中，MI−MSP为−0.0236至+0.0021，包含小幅正例和负例。这里的MI是特定head-only dropout（p=0.1、T=30）或每配置单个三成员集合上的disagreement proxy，不能据此否定所有epistemic estimators。[同预测器对照](within_predictor_score_effects.csv)、[图](figures/MI_vs_MSP_error_ranking.pdf)。

这些区间是探索性的逐项条件区间，没有把数千CSV行当作独立实验或作全家族显著性结论。缺少的更强证据包括更多独立ensemble组、其他posterior近似、预先定义的应用损失；仅在追求普遍方法主张时才需要新评估/训练。

### A3：校准、置信度尺度与错误识别是否是同一个问题？

**ANSWERED：不是同一测量对象。** 12组EuroSAT同checkpoint TS对照全部保持argmax不变，其中8组ECE下降。MSP错误AUROC差值仍可改变，范围为−0.002579至+0.000126；改变概率校准并不保证改善识错排序。[TS校准–行为表](TS_calibration_behavior.csv)、[同checkpoint验证](TS_same_checkpoint_verification.csv)、[图](figures/TS_calibration_and_error_ranking.pdf)。

多分类尺度分析使用中心化logits范数和top1−top2 logit margin；Tree使用逐标签log-odds，没有对15标签做softmax。[尺度表](scale.csv)、[logit尺度图](figures/TS_logit_scale.pdf)。这些是描述性关联，不识别logit尺度导致校准变化的因果机制。

保存float32概率可能饱和或并列。本轮主分析保留这些ties，另用logits64重建EuroSAT D/TS概率作为精度敏感性。该敏感性中AUROC绝对差最大约1.75×10⁻⁵，AP约4.85×10⁻⁶，AURC约3.23×10⁻⁶；原表不改写。[精度表](precision_sensitivity.csv)。不把“高精度重建”与主预测来源静默混用。

本题不缺必要训练或模型前向。若要提出新的温度/分数选择算法，需非test选择和新的独立评估；本轮没有开发或选择新方法。

### A4／E：传统AU/EU解释是否可以直接套到这些量上？

**代理依赖与有限支持偏差已回答；真实EO来源耦合仍未被识别。** A提供Shannon/Gini分解、模型/适配配方间的代理差异及代理与错误的关系。[代理均值](proxy_means.csv)、[配对差](proxy_paired_effects.csv)、[相关表](proxy_correlations.csv)、[Shannon/Gini图](figures/Shannon_Gini_classification.pdf)。相关性受共享分量和分解关系约束，不是来源独立性检验。

E固定真实生成分布后改变每层训练支持量，真实条件熵保持不变，posterior EE仍改变：

| 真实歧义a | 真实条件熵H(p) | n=20时平均EE | n=200时平均EE |
|---|---:|---:|---:|
| 0.1 | 0.325083 | 0.361315 | 0.328859 |
| 0.4 | 0.673012 | 0.633408 | 0.668487 |

单位nats；n是每X层标签数，总标签数2n。均值是对训练数据Binomial分布精确枚举后的量。EE在两种歧义下相对生成熵的偏差方向相反，支持量增加后偏差缩小。在同一已知机制中，EE差分之差为0.067536，而oracle条件熵的相应交互为0。[解析表](toy/condition_means.csv)、[图](figures/analytic_proxy_reference.pdf)。

这支持的结论是：**把posterior expected entropy直接当作数据生成条件熵，需要额外假设和估计误差论证；代理响应非加性不足以证明物理AU/EU耦合。** 它不反驳`TU=EE+MI`的数学恒等式，也不反驳在指定二阶预测分布下将其作为操作性定义；更不说明真实p(y|x)会因换模型而改变。两X层为镜像设计，不是两份独立实验。

若要回答真实EO AU/EU来源耦合，需要可验证的观测/标签生成机制、重复观测/标注或受控仿真，以及排除模型失配与优化随机性的设计。当前缺口无法通过简单重算指标、追加seed或普通blur自动解决。E是解释性参照，不能直接列为新的理论发现。

## 3. 对原三个RQ及论文完成的影响

| 问题 | 当前能回答什么 | 仍保留什么边界 | 后续动作 |
|---|---|---|---|
| RQ1 | 指定FM、任务、适配下的性能与经验校准；现在另有行为与目标类别诊断 | DOFA–EuroSAT历史来源UNKNOWN；不推广所有EO模型 | 无必要重训/新前向；保留来源限制 |
| RQ2 | 完整frozen/full配方的差异及seed变化，代理和行为差异可补充解释 | LR/训练日程等不同，不能归因纯冻结开关；无内部机制因果证据 | 不因差异不稳定补训；模块归因需另开D |
| RQ3 | 校准与性能代价，以及匹配Off下新增随机推理的行为变化 | 单组DE、head-only MC，未测全流程推理成本 | 本轮无必要重训；更广方法/效率主张另设计 |

建议结果章仍按RQ1–RQ3组织，后接“Error detection and selective prediction”；讨论章以“Calibration, operational utility and uncertainty proxies”串联A与E。E可放讨论中的小参照或附录，并在方法中明确它是独立解析实验。这样有明确的批判性观点及自己的EO证据，不必把未验证的新算法包装为贡献。[可用结果段落与落点](THESIS_RESULTS_DRAFT.md)。

本轮原模型训练/推理代码未修改，封存主表未重算覆盖；新增的是分析实现和派生指标。修复了新分析入口对异构归档字段的兼容、NumPy布尔相关性和输出序列化问题，完成对象结果经独立抽查不受未完成入口失败影响。现在无需为已完成A/E继续补训练或模型前向。

## 4. 成本、复核和限制

100对象的单对象处理墙钟时长之和约**16.51分钟**（分类3.02分钟、分割13.49分钟），进程峰值RSS约**3.67 GiB**；E解析求和和积分另见[计时](toy/verification.json)。该总和不等于整个协作任务历时，不包含实现、Claude审阅、聚合、制图及文稿时间，也不是GPU时间。对比原72次训练的64.86累计run-wall-hours，应理解为工作类型转向CPU分析，不能作为同硬件性能加速比。[分阶段成本](resource_profile.csv)。

实际Claude先独立审协议、再独立实现小范围重算，保留[协议复核](claude_protocol_review.md)、[代码/代表数据复核](claude_code_review.md)、[最终复核](claude_final_review.md)及公开工具记录。所有通过依据仍落在上述原始输入、代码、数字表与验证文件，详见[复核处理](REVIEW_RESOLUTION.md)。

原29项状态仍为27 PASS、2 UNKNOWN；新检查单独列FUP，不以统计行数增加PASS比例。C01/C06历史来源没有恢复。[检查delta](checks_delta.csv)。分割像素结论只覆盖固定32图；Cloud有来源关联、Space只有12 AOI；测试集的新分析是探索性，不能追称预注册。类别缺失/无错误等NA保留总数与有效数；分割像素配对只在双方共同有效的图上求差。

所有未选分支均作为后续探索保留，见上一轮[设计](../thesis_followup_design_20260926/DESIGN_AND_WORKLOAD.md)。复跑方法见[README](README.md)。

## 5. 后续探索：本轮未执行，也不是现有论文定位的必要前置

| 只有提出以下更强问题时 | 最小额外工作类型 | 已有设计的量级与边界 |
|---|---|---|
| 表示变化是否与校准/代理变化相关 | B：既有特征离线分析；需要补齐缺失表示时才补前向 | 已有分类9、分割12适配配对；DOFA–EuroSAT缺6份D表示，需独立开启补特征；相关仍非因果 |
| 对预设图像退化能否保持筛查效用 | C：补评估 | D-only为4配置×2新增blur状态＝8次checkpoint×全测试评估，无训练；仅支持该退化，不等于地域OOD |
| 输入机制与Transformer主体哪个可训练部分影响结果 | D：修训练入口并补统一配方训练 | 两FM×4条件×3seed＝24次新训练；统一优化预算，旧frozen/full不能直接替代端点 |
| 训练支持×输入退化的代理响应是否非加性 | F：补支持量设计、训练与多状态评估 | crossed默认12次新训练、36个模型–输入状态；仍不是物理AU/EU来源识别 |
| 真实EO AU/EU来源是否耦合 | 新的可识别生成/观测设计 | 需重复观测/标注或受控仿真等直接来源证据；不能用固定几次seed或普通blur保证回答 |

以上只整理缺口，不自动授权执行。当前以可追溯评价、行为诊断和代理解释为贡献的硕士论文，可以进入写作；无需为了出现正结果再增加训练。这里是研究证据充分性的判断，不能替代导师/学位项目尚未给出的具体验收要求。
