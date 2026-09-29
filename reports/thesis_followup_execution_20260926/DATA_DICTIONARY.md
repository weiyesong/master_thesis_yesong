# 结果阅读口径

所有新增分析为已有测试集上的探索性分析。基础模型、训练seed、MC draws、DE成员、图像/地点、标签和像素不是可互换的重复单位。

| 字段或文件 | 含义与边界 |
|---|---|
| object_id | 输入清单中一个预测对象。D/MC/Off/TS带训练seed；DE是42/43/44三个D的一个组合，没有独立ensemble seed |
| image_top1 | EuroSAT每图top-1二元错误，2714图 |
| image_hamming | TreeSat每图15个阈值0.5二元决策错误比例，2000图；连续损失，所以图像AUROC/AP为NA |
| micro_label_decision | TreeSat共30000标签决策，但bootstrap仍以图像组为单位；与逐标签macro均值不同 |
| image_pixel_error_fraction | 每图有效像素错误比例，然后对图像等权；与全测试像素加权accuracy/mIoU分开 |
| pixel_summary.csv | 固定32图的逐图像素指标均值；不是汇集所有像素后算一次指标；有效图数逐指标列明 |
| pixel_paired_effects.csv | 双方该指标共同有效图上的差值及组bootstrap；没有把NA补0 |
| msp | 越大越不确定。单标签/像素为1−max(p)；多标签为1−max(p,1−p)逐标签，图像分数再均值 |
| negative_margin | 多类为负top1−top2概率差；Bernoulli为−abs(2p−1)；与logit margin不同 |
| entropy / expected_entropy / mutual_information | Shannon TU / EE / TU−EE，单位nats；Tree逐标签Bernoulli口径，图像均值。保存MC摘要中的标签熵sum需除15核对 |
| gini_total / gini_expected / gini_disagreement | 多类1−Σp²及对应期望、类方差和；Tree单Bernoulli p(1−p)及对应分解；方差ddof=0。Tree与二类categorical相差2 |
| auroc / average_precision | 错误为正类，分数越大越不确定。AP是average precision；常量目标显式NA，不使用梯形PR面积 |
| aurc | 按升序分数接纳，k=1,…,N各风险的均值。同分组内取随机接纳期望；不是依赖文件次序的stable-sort结果 |
| risk_at_q | 接纳qN质量；同分块按共同fraction接纳，允许非整数。不是用错误标签打破ties |
| oracle_aurc | 按真实损失排序的理想参考，不可部署；随机排序参考为base_risk |
| *_ci_low/high；valid_bootstraps | 固定训练模型下1000次配对组bootstrap的百分位区间与有效重复数；非跨训练seed区间、非部署风险保证、非多重比较校正结论 |
| full_image_class_coverage.csv | 真值类别像素被接纳图像包含的比例，即内容保留量；不是该类recall。retained_expected_confusion_iou是期望混淆计数之比，不是随机IoU的期望 |
| NA与空值 | 定义不适用见analysis_applicability.csv；逐标签/像素常量目标和无类支持见对应na_reason/valid_images列；没有作为0参与排名 |
| core_metric_verification.csv | 旧主表与保存预测同定义重算核对，不包含重新训练；status需按字符串读，避免pandas默认把字面NA转缺失 |
| inputs_manifest.csv / inputs_lineage.csv | 前者是实际对象、输入hash来源和范围，后者补充标签字段、类顺序、分组来源与checkpoint成员关系 |

Brier沿用旧主表：categorical为类平方误差和再按样本/有效像素平均；TreeSat为所有图像和15个标签上的Bernoulli平方误差均值，即也除以标签数15。概率分数排序使用保存概率，float64仅用于算术；EuroSAT D/TS的logits64重建另在precision_sensitivity.csv，不能混用来选更好结果。

资源计量中旧64.86小时与本轮16.51分钟都是相应处理单元wall time之和，不能直接叫GPU时间、任务历时或同硬件加速比。早期resource_profile.csv的uncompressed_bytes_read只计分割全图主文件已解压数组；辅助raw数组另由resource_summary.json列出，避免冒称包含全部字节。
