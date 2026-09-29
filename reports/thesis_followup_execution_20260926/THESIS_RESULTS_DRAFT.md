# 论文可用结果与讨论草稿

以下文字针对本次固定实验与探索性补充分析。数值来源在句后链接，写入论文时应换为正式表图编号。原RQ1–RQ3核心结果仍采用2026-09-21完成包，不能用这里的行为指标替代校准指标。

## 方法补充：Operational uncertainty diagnostics

在既定测试预测上，我们评估了不确定性分数与错误筛查、选择性预测之间的关系。EuroSAT使用top-1错误；TreeSatAI分别使用逐标签二元错误和逐图Hamming损失；分割主分析以每图有效像素错误比例为损失。分数包括最大决策概率的补数、预测熵、概率margin、Gini总量，以及MC/DE预测集合上的expected entropy和disagreement项。比较同一预测器内的分数时，错误集合固定；比较不同预测器时，并列其基础错误率和任务性能。

风险–覆盖曲线按不确定性从低到高接纳预测。并列分数组内采用标签无关随机接纳的期望风险；非整数覆盖质量允许分数接纳。AURC是k=1,…,N各离散覆盖点的期望风险均值；错误识别AP采用average precision而非梯形PR面积。我们保留随机排序和oracle排序参照，不将面积指标单独解释为跨模型的总体优劣。

区间通过1000次配对组bootstrap计算，仅反映固定训练模型下的评估样本不确定性。EuroSAT按786个已有空间组，CloudSEN12按184个ROI/位置/来源产品连通组，SpaceNet7按12个AOI，TreeSatAI按2000幅图像；其未知邻近空间相关性作为限制。训练seed、MC draws和ensemble成员分属不同层级。[协议](ANALYSIS_PROTOCOL.md)、[分组](grouping_summary.json)。

分割像素诊断限于每数据集已固定的32幅共同研究图像，汇总逐图指标并报告有效图数；不把像素当独立统计样本。全测试MC的熵和概率方差用于图像级代理分析，MC/DE分解的像素比较使用相同32图。分割DE的三成员平均性能对照复用已审核心指标，新全测试拒识仅与D42比较。[范围清单](analysis_applicability.csv)。

## 结果补充：Error detection is useful but task dependent

在已评估预测对象中，MSP对EuroSAT top-1错误的AUROC为0.9221–0.9781，对TreeSatAI逐标签micro错误的AUROC为0.8120–0.8279。后者不能解释为逐图exact-match识错率。图像级拒识降低了当前测试子集上的平均损失，但其收益依赖损失定义及接纳内容。[指标](metrics.csv)、四任务[曲线清单](figures/figure_manifest.json)。

SpaceNet7尤其说明总体损失与目标类效用的区别。16个主预测对象在50%图像覆盖时，仅保留原测试集11.77%–18.82%的真值建筑物像素。由此，较低全像素风险可能伴随大量目标信息被拒绝；整体风险下降本身不能保证建筑物任务的实用性。该诊断使用真值评价选择后的内容，而拒识排序本身不使用真值。[类别保留表](full_image_class_coverage.csv)、[图](figures/spacenet_building_retention.pdf)。

固定同一预测器时，MI-style disagreement并非普遍优于简单的决策置信度。TreeSatAI的10个MC/DE对象中，逐标签micro错误AUROC的MI结果均低于MSP，差值为−0.1520至−0.0190；EuroSAT出现小幅正例及更大的负例，差值范围−0.0236至+0.0021。该结果限定于本项目的head-only dropout（p=0.1、T=30）和每配置单组三成员ensemble，不能外推至所有epistemic uncertainty方法。连续Hamming损失的图像AUROC不适用，不能将其NA当作区间跨0。[固定预测器差值](within_predictor_score_effects.csv)、[图](figures/MI_vs_MSP_error_ranking.pdf)。

## 结果补充：Calibration and ranking respond differently to temperature

12组EuroSAT TS比较使用同一checkpoint及既有非test温度拟合。所有预测类别保持不变，8组ECE下降，而MSP错误AUROC变化仍介于−0.002579和+0.000126。概率尺度改变可以影响校准及跨样本排序，其方向不必一致。因此校准收益、保持分类决策和改善错误筛查是三个需要分别验证的主张。[同checkpoint验证](TS_same_checkpoint_verification.csv)、[校准–行为表](TS_calibration_behavior.csv)。

保存的float32概率包含饱和和并列分数。主分析保留保存概率，并对D/TS用float64 logits重建概率作精度敏感性分析。该替代计算没有改变分类决策，AUROC变化的绝对值最高约1.75×10⁻⁵；结果表仍分别保存两种来源，没有按效果选择一种作为主结论。[精度分析](precision_sensitivity.csv)。多分类尺度关联使用中心化logits范数，避免原始范数的公共平移任意性。[尺度表](scale.csv)。

## 解析参照：Expected entropy is not automatically the generating entropy

为区分预测集合的分解与生成来源的解释，我们构造两个等权输入层，真实正例概率分别为a与1−a，使总体类别先验固定0.5。对a∈{0.1,0.4}和每层n∈{20,200}，使用Beta(1,1)先验，并对Binomial计数精确枚举训练数据外层期望。Shannon/Gini分解使用同一后验，真实条件熵则由已知生成分布计算；二者不是同一对象。两个层互为镜像，不构成独立重复。

在a=0.1时，真实条件熵为0.325083 nats，平均posterior expected entropy从n=20的0.361315下降至n=200的0.328859；在a=0.4时，真实熵为0.673012，而对应EE从0.633408上升至0.668487。真实歧义不随训练支持改变，代理值却随有限样本后验变化。EE的差分之差为0.067536，oracle条件熵的相应交互为0。[解析均值](toy/condition_means.csv)、[交互](toy/interactions.csv)、[图](figures/analytic_proxy_reference.pdf)。

这个例子不反驳熵分解恒等式，而是表明从代理量到数据来源的解释需要额外识别假设。非加性响应可以出现在真实歧义与训练支持的受控设计中，因此仅凭交互、相关或排序反转无法证明真实AU/EU物理耦合。该解析示例与EO实证一起，支持对操作性指标含义的审慎区分；它不是新的普适理论证明。

## 讨论与论文落点

论文可以组织为：既有三RQ的校准/性能结果 → 不确定性的错误筛查效用 → 选择后目标类别内容 → 概率尺度与代理解释 → 解析参照和识别限制。由此形成三条相互关联的贡献：一套可追溯的EO FM评价；对操作性效用和目标类保留的诊断；用实际结果与解析参照划清代理分解与真实来源解释的边界。

不要写成“证明传统AU/EU定义错误”“提出新的AU/EU理论”“MI普遍无用”“TS没有改变识错能力”“拒识保证建筑物检测安全”或“没有显著差异即等效”。更稳妥的表述是：在所评估模型、任务和采样近似下，不同指标回答不同问题；一项校准或代理改善并不自动保证所需的决策效用。

保留DOFA–EuroSAT历史归一化/训练代码来源UNKNOWN；旧测试集上的新分析标为探索性。局部区间不作多重比较校正后的普遍显著性主张；少量训练seed与一个ensemble组不支持跨训练分布的稳定性保证。若未来开发新方法或拒识阈值，需非test选择及未用于开发的独立评估。
