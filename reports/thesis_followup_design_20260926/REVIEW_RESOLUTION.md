# 实际 Claude Code 独立评估与交叉修订

本次用户请求设计、相对实验量和完整prompt。未运行新研究指标、模型前向或训练；仅进行文档/代码读取、成本算术、产物路径/header/ID及字段别名验证。

实际Claude session：`5cf1f601-8904-4e8c-b643-7deba2098e15`。保留[独立初评](claude_design_review.md)、[完整prompt交叉复核](claude_followup.md)、两轮原始JSON及[公开工具调用记录](claude_design_tool_trace.json)。不以其总结或两者意见一致作为通过证据。

协调者的直接依据包括[72次训练成本核算](workload_estimates.json)、[100个计划对象清单](planned_A_prediction_objects.csv)、[实际头部与来源检查](design_input_checks.json)、[20份子集ID对齐](fixed_subset_id_checks.json)、[三个表示导出别名核验](embedding_semantics_check.json)，以及相关实现和现行协议。原始研究结果未被更改。

## 采用并落实的发现

- MC全测试已经保存MI和概率总体方差，Gini分解可以离线恢复。直接实现为 [classification variance](../../scripts/c5_mc_dropout_inference.py:260) 与 [segmentation unbiased=False](../../scripts/c5_mc_dropout_inference.py:496)；实际NPZ头部也有对应数组。默认32图像素分析是计算范围选择，不是全测试没有MI。
- frozen表示需按唯一编码器去重，但先验证权重、buffers、输入和提取层。21个配对不是21个独立冻结参照。
- 当前旧frozen/full配方不同；D模块干预按两FM24次/单FM12次新训练规划。只训练输入模块仍需梯度穿过Transformer。
- toy采用Beta–Binomial解析枚举求均值，20次重复改为可选抽样展示。两个平衡X层保持总体类别先验0.5。888项计数只针对逐层条件量与其均值；跨层混合概率的非线性量需联合枚举，不在默认设计中。
- 输入退化仍沿用各run实际归一化。F的静态特征缓存与[现行无随机增强协议](../final_training_protocol.md:68)兼容，但训练缓存入口需要实现，train/validation特征也不能假定已导出。

## 初评/交叉复核中的修正与未采纳部分

| 原建议或措辞 | 最终处理与依据 |
|---|---|
| C-all无缓存总计24次编码 | 从零开始是`8×(3D+1MC)=32`。Claude在复核中澄清24是D-only8次之后的追加数。共享frozen缓存严格验证后可降20；head总计272。 |
| D外推约21小时，实际“只会更高” | 撤回单向下界。新预算/选择/实现会改变时长，采用逐条件计时后估算；64.86累计run-wall-hours不是GPU时间。 |
| B的seed配对是伪配对，冻结侧恒0 | 配对比较可以有效；不能把共同参照当独立重复。冻结等价须验证，未预先断言。 |
| 默认A+E、分割40对象 | 默认A、分割32主对象；E和额外8个MC/Off对象分别有开关。属于时间/主张的范围取舍，不是否定这些输入存在。 |
| EuroSAT所有D只有约40–60个错分 | 直接按主表accuracy×2714换算为43–124；frozen43–46。最终prompt要求从原预测核实每对象错误数，不能把full漏掉。 |
| `embeddings`可能是head BN后的量而`backbone_representation`相同 | 不能由不同字段名推出该结论。当前[导出代码](../../scripts/prediction_export.py:250)将同一数组写入两字段；[提取器](../../scripts/prediction_export.py:721)调用模型的head之前features；三个代表实际文件两数组完全相等。仍要求执行B时逐文件核实具体版本。 |
| 泛化F最低新增3/9 | 必须指明配置。主表中Pan–Euro frozen MC只有42，沿用且匹配旧配方时最多复用1个，即新增5/11；DOFA同格有42/43/44才可能3/9。新统一优化步数设计仍预算6/12。 |
| subset全文件对齐的描述数量不明确 | 协调者实际读取12份MC与8份DE子集全部20份ID，逐序匹配固定定义；不借用初评总结替代此证据。 |

交叉复核未发现阻断性问题。对完整prompt的最终细化已根据直接证据落实；未运行未来研究来预先保证正结果。执行前仍需按prompt核实全部实际使用对象、归一化/mask/统计单位，复核新测量与旧主指标的一致性。

## 最终取舍

推荐完成A后收束；若希望给AU/EU讨论增加可控参照，再显式开启E。B、C、D、F分别支撑表示关联、特定退化响应、模块干预和支持量交互，不能相互替代，也不全是完成当前三个RQ的必要工作。尚未定义的新方法/新理论不能诚实定额承诺实验次数。
