# Codex 第一轮独立发现（交叉核对前记录）

快照：snapshot.json。记录时尚未读取 Claude 的独立审查输出。此文件保存初始判断，不是最终验收表；最终状态见 checks.csv 与 AUDIT_REPORT.md。

- 已按现行冻结协议建立 64 个方法比较格：52 格有产物，12 格 TS NA；共 100 份实际结果（48 deterministic、24 MC、12 TS、16 ensemble）。NA 不能计为缺失，也不能计成有效数值结果。标准要求的 MC 同权重 dropout-off 对照没有进入现有矩阵。
- 直接读取全部 72 个训练 checkpoint，对照原始预训练 tensor、训练 history 和 run summary；哈希、最早 validation 最优 epoch、frozen 未变/full 已变均核对通过（checkpoint_verification.json）。历史 DOFA–EuroSAT 6 runs 无 code_snapshot.json，且归一化常数来源不明，C06/C01 对该来源保留 UNKNOWN；不能声称已证实泄漏。
- 分类 56 份全测试预测用独立 NumPy 重算；全部主指标在 2e-6 容差内一致，最大误差约 2.3e-9。样本 ID 与官方 manifest 一致，各方法/模型/seed 的标签一致。12 TS 温度为正，calibration/test 分离且 argmax 不变。
- MC 24 个训练模型用 dropout p=.10、指定 head/decoder dropout 和 30 次概率平均。原表与 p=0 独立训练模型配对，不能从该对照分离训练正则化和采样效果。R3.1/R3.4 必需补同 checkpoint dropout-off test evaluation；现有 checkpoint 足够，不需要补训练。
- TreeSatAI 是 15 标签、T=1 静态任务，BCE/sigmoid/0.5 阈值与 exact-match accuracy 正确。表 ECE 是所有 label 的 binary-decision confidence；C6 图是 pooled positive probability，二者不是同一 ECE。独立 label-probability ECE 更大，必须区分这些对象并保留各标签证据，不能用低 pooled ECE 宣称所有标签校准。
- C6 reliability 图代码计算 count 但未展示，也没有说明只展示 seed42。应提供与表 ECE 匹配的图、箱数及逐 seed CSV。已补算分类 10/15/30 箱与类概率统计，分割正在全量重算；这属于已有预测重分析。
- 旧 final_thesis_results.md/C6 表仍展示 TreeSatAI TS，主证据表与冻结协议已排除。应修报告生成器与发布入口。冻结文档所谓 selection 与 calibration 共用 validation 必然无效的理由过强；标准明确不必如此。排除发生在历史 TS 之后，需要明确披露时间线，不能伪称预注册或为好结果删不利组合。
- EuroSAT 两模型归一化常数不同：仅能比较整个已采用配置，不是纯 FM 效应。其他三个数据集输入契约相同。RQ2 的冻结/全调使用不同优化配方与预算，可回答方案差异，不是仅冻结开关的因果效应；无需强制相同 LR。
- 四个数据集 split ID/组已直接重查。CloudSEN12 发现 1 个 train/validation 和 2 个 train/test acquisition product 重叠，ROI/equi ID 不重叠；SpaceNet7 AOI 完全分离。维持已声明 benchmark 泛化范围，不机械判泄漏。
- RQ2 成对差值已有完整证据，不稳定方向是结果。RQ3 Ensemble 在 EuroSAT full 可提升性能/NLL/Brier 而使 ECE 升高；SpaceNet7 可降低 ECE 同时降低 building IoU；应量化，不能概括为无代价普遍改善。
- 初步：RQ1 原报告诊断与叙述仍 PARTIAL，可用本次既有预测补充收窄；RQ2 在适配方案的描述性范围内 ANSWERED；RQ3 为 PARTIAL（TS/ensemble 与全方案 MC 结果可答，same-weight MC 问题缺对照）。最终按重算及双方直接证据交叉核对落定。
- 未发现足以要求新训练的证据；不把低建筑物 IoU、未改善、seed 波动判为科学缺陷。OOD、corruption、CKA、机制及传统模型比较不纳入本轮必需任务。
