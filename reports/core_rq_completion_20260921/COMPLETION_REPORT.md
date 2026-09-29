# 审计补充完成记录

29项：**27 PASS、0 FAIL、2 UNKNOWN、0 NA**。两个UNKNOWN均为历史来源问题，没有把它们自动升级为通过。实际方法覆盖仍为64个cell：52个有核心结果、12个TS NA；另完成24个MC同权重Off对照。NA属于矩阵cell，不与29条共用检查状态混淆。

本轮启动于2026-09-21、完成于2026-09-23。没有新增训练、修改checkpoint或重新挑选结果。新旧原始产物均保留。

## 阅读顺序

1. [三个RQ结果正文](RESULTS_SECTION.md)：当前能回答什么、反例及结论边界。
2. [重建正式图表](published/final_thesis_results.md)：统一TS范围与ECE语义，曲线下方附counts。
3. [24组三方绝对指标和差值](mc_dropout_three_way.csv)、[逐seed效应](mc_dropout_effects.csv)、[48块摘要](mc_dropout_effect_summaries.csv)。
4. [29项机器可读检查](checks.csv)、[唯一问题清单](issue_register.csv)、[核心覆盖矩阵](coverage_matrix.csv)、[24个对照覆盖](mc_dropout_control_coverage.csv)。

## 完成了什么

- 补齐12分割MC checkpoint全test dropout-off预测，CloudSEN12每模型975图、SpaceNet7每模型1152图，共12,762图次；加既有12分类Off形成完整24组三方对照。
- 从保存预测独立重算全部新结果，并直接匹配D/MC的IDs、标签、ignore mask和checkpoint hash。
- 修复C6 TreeSat decision-ECE、右闭分箱、counts/seed标注；64表项与52主图曲线逐项核对master，NA指标为空白。
- 保留历史TreeSat TS全部12结果和时间顺序；将其与现行核心NA明确区分。
- 对齐一个近平局像素引起的GPU累计/CPU导出混淆矩阵差，保留原始差异和修复记录。
- 完成新的结果正文，未声称审过缺失的外部论文全文。

## 三问的回答与后续动作

| RQ | 状态及当前答案 | 仍缺的必要来源/更强主张证据 | 修代码/重算/补评估/补训练 |
|---|---|---|---|
| RQ1 | ANSWERED（声明范围）—不同FM的经验校准排序依任务/适配/指标而变；低总体ECE不能覆盖类概率和建筑表现。 | DOFA-EuroSAT统计来源与六run训练代码绑定；外部论文全文未审；不支持全EO/纯因果/等效无损主张 | 已修报告生成器；已重算并对齐补评估指标；无必要新增；不需要；未启动 |
| RQ2 | ANSWERED（声明范围）—完整适配配方的性能与校准变化因条件而异，方向不稳定本身是有效结果。 | DOFA-EuroSAT统计来源与六run训练代码绑定；外部论文全文未审；不支持全EO/纯因果/等效无损主张 | 已修报告生成器；已重算并对齐补评估指标；无必要新增；不需要；未启动 |
| RQ3 | ANSWERED（声明范围）—TS保持决策，ensemble有任务依赖收益与代价；分割MC-Off改善ECE/NLL/Brier但性能小幅波动，分类收益不一致。 | DOFA-EuroSAT统计来源与六run训练代码绑定；外部论文全文未审；不支持全EO/纯因果/等效无损主张 | 已修报告生成器；已重算并对齐补评估指标；已完成12分割MC Off；不需要；未启动 |

ANSWERED指声明范围内的描述性研究答案已给出，不表示所有历史可追溯检查都PASS。DOFA-EuroSAT的train-only统计来源和训练代码绑定若是外部验收的强制要求，相关来源条款仍未满足；此处明确保留UNKNOWN，不能宣称完全排除历史预处理泄漏。其余无已确认必须追加的核心实验。

## 直接证据与独立复核

[独立全量补评估核验](segmentation_off_verification_summary.json)：12/12通过；原始标量指标与NumPy重算最大绝对差 7.68e-08。[37项原有相关测试](targeted_tests.log)通过；生成器专项测试见 reporting_fix_notes.md。原2394文件中仅本轮授权C6生成器变化，原104审计文件均未变（baseline_integrity.json / previous_audit_integrity.json）。

Claude Code由实际CLI调用，先独立检查runner/训练配置并抽算分类与分割，再复核最终固定产物。报告和逐工具证据见 [设计复核](claude_design_review.md)、[最终交叉复核](claude_final_review.md)、claude_design_evidence_trace.jsonl、claude_final_evidence_trace.jsonl。每个PASS由代码、原始预测或可复算表支持，未仅依赖另一agent的结论。评估代码hash与训练时代码分开保存。审查链接补齐及独立复核数值措辞的更正见 [复核处理记录](REVIEW_RESOLUTION.md)，Claude原文完整保留。

## 29项逐项状态

| ID | 状态 | 影响RQ | 发现/验收 | 证据（相对此目录；原始运行见对应CSV） | 最小补救 |
|---|---|---|---|---|---|
| C01 | UNKNOWN | RQ1;RQ2;RQ3 | 四数据集 test ID 与训练/选择集分离，72 checkpoint 均按最早 val 最优选择；旧 DOFA–EuroSAT 归一化常数的统计来源未保留（MC DOFA–EuroSAT沿用）。 更早backup代码载有常数，但无统计推导/训练run绑定，补查后仍UNKNOWN。 | provenance_followup.md | 保留历史来源UNKNOWN；更强主张需恢复证据 |
| C02 | PASS | RQ3 | EuroSAT 12 个 TS 的 calibration IDs 精确匹配专用 split，均与 test 分离；TreeSatAI TS/分割 TS 是现行协议 NA。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| C03 | PASS | RQ1;RQ2;RQ3 | EuroSAT CE/softmax/argmax；TreeSatAI 15标签 BCE/sigmoid/0.5阈值，accuracy是exact match；CloudSEN12四类，SpaceNet7标签1→0、2→1、0→ignore255。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| C04 | PASS | RQ1;RQ2;RQ3 | 直接读取72 checkpoint并与实际预训练tensor比较：frozen所有可匹配backbone tensor未变，full均有更新；权重哈希匹配；输入契约/结构冻结例外与代码和config一致。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| C05 | PASS | RQ1;RQ2;RQ3 | 报告主图与表采用同一decision-ECE及右闭分箱，带counts/seed；独立重算补评估并解决导出近平局像素，原生clip语义保留。 | published/tables/package_validation.json;segmentation_off_verification.json;metric_alignment_resolution.md | 已完成；无需训练 |
| C06 | UNKNOWN | RQ1;RQ2;RQ3 | 100份预测全量主指标重算、样本/标签对应及72checkpoint哈希均有直接证据；旧DOFA–EuroSAT六run只保存事后代码快照，training-time git_commit=null。 更早backup代码载有常数，但无统计推导/训练run绑定，补查后仍UNKNOWN。 | provenance_followup.md | 保留历史来源UNKNOWN；更强主张需恢复证据 |
| C07 | PASS | RQ1;RQ2;RQ3 | 48 deterministic seeds、24 MC seeds、12 TS seed结果和16单组ensemble分开；mean/std及成对差值均由独立重算核验。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| C08 | PASS | RQ1;RQ2;RQ3 | 直接manifest复核：EuroSAT ID/内容hash/spatial_group不跨split；TreeSatAI ID/source不跨；Cloud ROI/equi不跨但3个product跨；SpaceNet7 AOI/source/patch不跨，test12AOI。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| C09 | PASS | RQ1;RQ2;RQ3 | 现行TS仅EuroSAT，12 NA blank；历史TreeSat TS结果完整单列并公开8-24至8-27时序；异常中断与数值对齐均留痕。 | published/tables/method_applicability.csv;published/tables/historical_treesatai_temperature_scaling.csv;SUPPLEMENT_PROTOCOL.md | 已完成；无需训练 |
| R1.1 | PASS | RQ1 | 两个FM在相同test样本/标签、适配条件下比较。EuroSAT历史DOFA与新Panopticon归一化不同；其他三dataset共用输入契约。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R1.2 | PASS | RQ1 | 分类accuracy/Macro-F1/ECE/NLL/Brier、分割mIoU/per-class IoU/pixel accuracy和概率指标齐全；TreeSatAI准确率语义在权威master明确。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R1.3 | PASS | RQ1 | 本轮从全部保存预测补算10/15/30箱、每箱count/confidence/outcome和signed gap，并生成seed42图；全部seeds箱表保留。旧C6不显示count且未标seed。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R1.4 | PASS | RQ1 | SpaceNet7已有全有效像素p(building)对indicator校准和建筑IoU；本轮补齐Cloud各类、TreeSatAI15标签、SpaceNet7每类的概率箱表与类指标。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R1.5 | PASS | RQ1 | 保存全部既定seeds；本轮给Panopticon−DOFA逐seed效应、均值/sample SD/min/max，并用10/15/30箱检查细小ECE排序。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R1.6 | PASS | RQ1 | 本轮正式结果正文逐任务/适配描述经验校准、过欠自信与类诊断，保留不稳定排序及历史来源限制；外部完整论文不在已审范围。 | RESULTS_SECTION.md;provenance_numeric_crosscheck.json;published/final_thesis_results.md | 已完成；无需训练 |
| R2.1 | PASS | RQ2 | frozen只训练head/decoder、full更新backbone；72checkpoint与预训练tensor直接比较及梯度/optimizer记录支持。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R2.2 | PASS | RQ2 | 同FM/dataset/seed的两适配data配置完全一致、head概念可比、无augmentation、selector相同；LR/WD/warmup/预算/patience按方案不同。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R2.3 | PASS | RQ2 | 24个matched-seed frozen→full对照及8个三seed摘要完整，逐seed差值和差值SD直接重算。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R2.4 | PASS | RQ2 | 每个对照同时给ΔPerformance/ΔECE/ΔNLL/ΔBrier，分割另保留建筑IoU。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R2.5 | PASS | RQ2 | 两FM、四dataset分类/分割均有对照；方向随条件变化，反例完整。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R3.1 | PASS | RQ3 | 24个MC训练checkpoint均有同权重Off，与D及MC全ID/标签/mask配对；推理、配方轨迹、总方案三个差值分开。 | mc_dropout_three_way.csv;segmentation_off_verification.json;../core_rq_audit_20260918/classification_dropout_off_verification.json | 已完成；无需训练 |
| R3.2 | PASS | RQ3 | EuroSAT正温度只用专用calibration labels拟合NLL，在相同未拟合test上比较。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R3.3 | PASS | RQ3 | 12个EuroSAT TS标量T>0，scaled logits为raw/T，raw logits与原预测逐项一致，argmax变化均0。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R3.4 | PASS | RQ3 | 12分割补评估全部dropout-off、严格权重加载、BN不变、no_grad、全test；同MC checkpoint哈希相同，独立重算通过。分类12Off已有直接features+head复现。 | mc_dropout_control_coverage.csv;segmentation_off_verification.json;segmentation_off_verification_summary.json | 已完成；无需训练 |
| R3.5 | PASS | RQ3 | 16 ensembles来自独立seed42/43/44训练checkpoint，成员IDs/labels一致；直接成员概率平均：分类全测试、分割固定前8图，最大误差<3e-8。主指标对ensemble概率重算。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R3.6 | PASS | RQ3 | 24行三方绝对指标/72行差值完整，同时报告ECE、NLL/Brier、主性能及建筑IoU；保留训练epoch与D跨seed范围。 | mc_dropout_three_way.csv;mc_dropout_effects.csv;RESULTS_SECTION.md | 已完成；无需训练 |
| R3.7 | PASS | RQ3 | 16 MC格seed42，只有预定4格三seed；ensemble每格一组；30pass不是n=30。 | ../core_rq_audit_20260918/checks.csv | 已完成；无需训练 |
| R3.8 | PASS | RQ3 | 正文完整保留TS/ensemble/MC反例与指标冲突；分类MC推理不一致、分割微小代价与总方案不同方向均披露。 | RESULTS_SECTION.md;mc_dropout_effects.csv;published/tables/historical_treesatai_temperature_scaling.csv | 已完成；无需训练 |
| R3.9 | PASS | RQ3 | 按实际Δ性能与ΔECE描述收益/代价及数值量级；只有TSargmax不变性支持严格决策保持，其余不声称等效无损。 | mc_dropout_effect_summaries.csv;RESULTS_SECTION.md | 已完成；无需训练 |

## 重现本轮补充

使用现有环境，在/workspace运行：

```sh
python -m scripts.complete_segmentation_dropout_off --device cuda:0 --shard 0
python -m scripts.complete_segmentation_dropout_off --device cuda:1 --shard 1
python reports/core_rq_completion_20260921/align_export_metrics.py
python reports/core_rq_completion_20260921/verify_segmentation_off.py --require-complete
python -m scripts.build_core_rq_completion
python reports/core_rq_completion_20260921/compute_off_sensitivity.py
python reports/core_rq_completion_20260921/write_completion_report.py
```

两个GPU分片可同时启动；已完成run会核验hash后跳过；不完整目录会拒绝覆盖，首次中断产物保存在interrupted_before_completion。正式C6重建命令为 `python -m scripts.c6_build_final_results --output-root <新的空目录>`，当前发布位于published。现有文件的SHA256与来源清单见 completion_manifest.json。

## 单列后续探索

额外训练重复、按AOI/地点组配对区间与预定义性能容忍界限、纯架构/冻结开关/dropout正则因果控制、OOD/corruption与机制解释均需单独确定范围。本轮不以方法无改善、性能权衡或适配效应不稳定为由要求追加训练。
