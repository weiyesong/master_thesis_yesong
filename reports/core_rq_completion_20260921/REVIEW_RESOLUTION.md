# 独立复核的接受与更正记录

实际Claude Code最终报告和逐工具证据原文保留于 [claude_final_review.md](claude_final_review.md)、[claude_final_evidence_trace.jsonl](claude_final_evidence_trace.jsonl)。下面对其发现逐项处理；PASS仍以源文件与独立计算为依据。

| 发现 | 处理 | 直接证据 |
|---|---|---|
| 同权重对照、24组三方差值、29项27PASS/2UNKNOWN及三问的限定结论可接受 | 接受。Claude独立读两格分割NPZ三方对照和TreeSat原始parquet；协调者另有全12复算与算术核对 | segmentation_off_verification.json、three_way_arithmetic_verification.json、raw NPZ/parquet路径列于mc_dropout_three_way.csv |
| 最终复核文件链接在Claude仍运行时不存在 | 已解决。CLI结束后将原文与40个工具结果的轨迹落盘，最终链接检查要求全部存在 | claude_final_review.md、claude_final_evidence_trace.jsonl、final_integrity.json |
| 前景NLL抽算约0.4135而正式值0.43245 | 不接受把它归类为同一公式的dtype误差。审查工具代码先选真实类别概率，再以1e-7作下界；项目协议先clip建筑概率，下界1e-12、上界1−1e-7，再计算二元NLL，两者是不同数值公式 | reviewer_foreground_clipping_probe.py/.json；Claude工具轨迹中的 `pbin=np.clip(pbin,np.float32(1e-7),...)`；scripts/segmentation_pipeline.py |
| 发布52曲线“误差0” | 审查文字过度四舍五入。568个正式表值与master精确相等；52曲线最大ECE差是2.737400972563364e-8，不能写成全部精确0 | published/tables/package_validation.json、reporting_hash_verification.json |
| 近平局修复的“指标变化≤1.7e-8” | 仅近似适用于mIoU。pixel accuracy变化约−1.73e-8、ECE约−1.74e-8，boundary ECE约−5.74e-8。修复与复核容差结论不受影响 | metric_alignment_resolution.md、原run的metric_alignment.json、segmentation_off_verification.json |

最小前景复现直接读取同一份NPZ，项目协议所得NLL **0.4324517296592825**，与正式CPU对齐指标 **0.43245172892568334** 相差 **7.34e-10**；审查替代公式重现 **0.41346555106397953**。因此保留原协议与指标，无需再改数据、重推理或训练。这与上一轮I06的float32/float64上截断表示差异是两个应区分的数值说明。

Claude的NumPy softmax抽查与本项目Torch CPU导出在末位可不同，而其argmax与导出一致。项目CPU原路径完整重放与保存概率差0；该表述不代表不同库的所有softmax实现都逐位相同。

设计复核的建议也已处理：三方表保留D/MC训练best/last epoch和D跨seed范围；Off−D明确为配方加轨迹；标签按实际delta量级呈现，不用符号标签宣称实践损失或等效。设计阶段所称“verifier尚不存在”是当时并行产物尚未就绪，最终已有12/12直接复算证据。

无需循环修改模型以获取正结果。独立审查的科学结论已接受；上面链接及数值措辞以直接证据完成更正，未更写Claude原文。
