请对最终设计与完整执行prompt做一次有界交叉复核，只读，不运行新研究指标/前向/训练。请读：
reports/thesis_followup_design_20260926/EXECUTION_PROMPT.md
reports/thesis_followup_design_20260926/DESIGN_AND_WORKLOAD.md
reports/thesis_followup_design_20260926/workload_estimates.json

你的初评已保留在claude_design_review.md。我们已采用全测试MC方差可恢复Gini、冻结参照去重、模块24新训、toy枚举解析及真实分组等重要发现。协调者直接核对master/manifest和MC实现，仍需澄清以下点：
1. C-all从零开始8cell-state*(3D+1MC)=32 encoder dataset-evals；你写24无cache，若表示在D-only8之后追加24是正确，总计仍32。20是严格验证跨frozen D/MC等价并共享缓存后。请明确总数，不能漏8个MC或误混追加数。
2. D不能说旧时长外推21h后新协议“实际只会更高”；固定steps和新选择可增加或降低，故不采纳下界，采用计数+分条件短计时。仅输入可训仍反传Transformer，已加入。
3. B的21seed配对本身不无效，问题是不能当21独立冻结参照。仅当权重/buffer/preproc/extractor/numerical一致才去重，不能无验证断言冻结侧恒0。
4. 默认A分割全test做image-level，以32共同子集做pixel-level是计算范围取舍，不是误认为full MI缺失；MC全图Shannon/Gini利用已存总体方差，DE全图分解可选。默认32对象，extraMC/Offseeds8另开而不自动把40设强制。请检查是否足以支撑明确限定结论。
5. E改成平衡X两层，使a变化不改总体classprior；四(a,n)条件，对每层n枚举k，2a*2strata*(21+201)=888后验加权项。20重复仅可选抽样变异。闭式检查和非物理耦合限制已写。
6. F默认选Pan-Euro-frozen MC，该cell只有seed42（master直接核查），故即使旧配方可复用也只能6-1=5、12-1=11；DOFA同cell3seed可3/9但来源边界保留；统一step控制按6/12全部新训。需train/valfeature前向，静态cache与augmentation兼容。请核实。
7. 最终默认[A]而非A+E，是用户时间有限下的范围选择，E完整设计已保留开关。结果段落草稿在交付中，符合用户整理论文目的；不是自动写整篇论文。

输出请限制约1500中文字符：列真正阻断性问题、仍需修正的科学错误及其文件段落；或确认这些修订合理但指出残余边界。不要重新写一份长报告，不重复读取历史大文件，不扩大任务。审查依据应有直接文件/数学而不是双方一致。
