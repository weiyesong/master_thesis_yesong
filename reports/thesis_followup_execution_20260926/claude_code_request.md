继续实际Claude独立复核，当前用户授权A/E已在运行。请有界审查新测量代码和已完成的代表对象/E，不等待或重算全矩阵，不训练/推理/改文件。允许用独立numpy/scipy/sklearn代码小范围抽算，不能仅调用协调者函数就宣称独立验证。输出中文约2000字。

读 code/metrics.py、code/analyze.py、code/toy.py、ANALYSIS_PROTOCOL.md版本2（都在reports/thesis_followup_execution_20260926）。原始输入在inputs_manifest.csv，dataset metadata group见sample_groups.csv。

协议2已固定保存概率为主（保留float32 tie），EuroD/TS logits64另作精度敏感性，不采用你建议的统一logits重建作为主分析，原因是忠实于保存预测且MC概率均值不得换平均logits。qN允许fractional接纳已明确；boundary改1；Gini sumvariance及Treesum/15明确。

请重点独立抽查：
1. tie-group离散AURC解析公式与二元AUROC/AP、weighted group bootstrap、workpoint fractional含义。代码8个小测试已通过，但你用独立方法核实。
2. 原始分类Euro DOFA frozen DE（对象eurosat__dofa__frozen__deep_ensemble）的mean probabilities、样本对齐、错误为正的MSP AUROC/AP/AURC。另抽Tree多标签对象若done.json已存在；未完成可只检查代码。
3. 原始分割Cloud DOFA frozen DE（cloudsen12__dofa__frozen__deep_ensemble）选一个固定子集图，自raw mean概率核验像素指标和分解；全测试图像损失从已保存per_image表验证RC抽样，同时注明这一步本身不是重新读全975图。
4. E toy条件均值、oracle偏差、交互解释，自己枚举或积分核对，不假称它证明真实源耦合。
5. 输入快照、分组和CI/NA传播是否有真正科学错误，需要重算哪些受影响输出。请区分代码bug、科学定义选择和后续研究建议；默认范围不扩展。

已有协议review原文保留。当前全量作业运行中，部分对象缺done不意味着承诺缺失。后续协调者会完成剩余表图/报告，再做最终完整性复核。不要多次阅读全文档案，不读凭据。
