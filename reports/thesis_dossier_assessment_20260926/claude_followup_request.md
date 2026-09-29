感谢独立检查。协调者已直接确认两个重要发现并采纳：coverage_matrix.csv有16条过期“MC lacks same-weight dropout-off comparison”备注；主表MC24条推理耗时未记录，TS12条拟合/应用耗时未记录，DE16条为成员推理时间求和且聚合耗时缺失。我们保留封存原件，另建更正阅读表，明确不是重开24组对照；成本只报实际口径/操作次数，不当成T倍实测wall time。

请简短复核以下修正是否接受，输出约800–1200中文字即可，不需要重新通读大档案，也不运行新实验：
1. 你称“目前run目录只有test，需72checkpoint新val前向”过宽。实际已有 results/final_thesis/c3_temperature_scaling_and_ensembles/classification/validation_exports/treesatai/.../predictions.parquet（12组）以及EuroSAT calibration exports。应按新增分析实际需用的数据/模型逐项盘点，不自动72个新前向。旧test探索不等于任何可用val数据都没有。
2. 多分类TS保持每样本argmax，但跨样本置信度排序/AUROC不保证保持，不能把大变化当作必然bug。AUROC在固定类别条件分布下不随正负混合比例变，并不表示换预测模型/错误集合后可不看错误率；e-AURC也非万用修正。TreeSat exact-match高错误率不使AURC数学上无信息；逐标签正例≥50不是定义/必要门槛，可保留小类并报支持数、区间及未定义量。
3. 多分类raw logit norm受公共平移影响，建议中心化norm+margin。缺失DOFA-EuroSAT deterministic embedding不能用MC embedding代替后仍称D frozen/full比较；可独立报MC或补6个feature前向。
4. Hessian/Jacobian类诊断需要新增梯度计算，未必需要训练。RGB通道遮蔽不是原则上不能做，而是要定义输入信息与扰动语义。Blur不必然改变潜在分割真值，不宜机械要求仅内部像素；按任务目标和严重度检验标签保持，云分割人工添云则更直接存在标签语义问题。
5. “DE可靠改善必须每格再训2组ensemble”没有通用数字门槛，独立重复需要按想做的结论/精度定义，3组也不保证普遍稳定。正式毕业必要性未知，建议称当前研究范围必要。
6. 你把计算代价实測缺失列为档案更强主张的缺口是有益的，但现行29项规范明确计算成本可补充、不因此新增核心实验。采纳条件化方案：若写实测效率比较则需新统一计时，否则披露缺失与操作次数足够。

协调者推荐：S0核心写作；S1现存预测行为分析；S2既有D/TS尺度诊断+熵/Gini代理稳定性（0新前向/训练）；到此收束。CKA、corruption是分支，模块训练和AU/EU已知生成参照为更强结论的条件要求，不是全部毕业前置。请给修订后共同点与仍有异议，不必同意没有依据的部分。
