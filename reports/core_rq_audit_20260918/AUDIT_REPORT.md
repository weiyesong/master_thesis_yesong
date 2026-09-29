# EO FM 三个核心 RQ 审查报告

快照基准：2026-09-18；交付完成：2026-09-19（UTC）。规范：[EO_FM_UQ_Core_RQ_Audit_Standard.md](../../EO_FM_UQ_Core_RQ_Audit_Standard.md)，29项全部逐项记录。

**当前结论：RQ1 PARTIAL，RQ2 ANSWERED（限定所采用适配方案的描述性比较），RQ3 PARTIAL。未发现要求新增训练的已确认问题。** 本轮完成了必要的已有预测重分析和12个分类MC同权重Off补算；尚缺12个分割MC同权重Off评估，正式报告生成器/范围声明需要修复。无改善、不稳定或性能下降本身均为有效研究结果。

## 0. 审查边界、独立性与证据

- Git分支没有提交，不能编造HEAD。固定代码/配置/协议/结果元数据快照为 `sha256:af23e14b2d75b8e64a69f582c35734075b5316873fc12c1f6e32af5e9e5890df`，2394个文件见 [snapshot.json](snapshot.json)。大型预测及checkpoint分别在验证JSON中记录实际SHA256。最终检查原快照文件未改变。
- Codex先保存 [独立发现](codex_independent_notes.md)，Claude Code由本机CLI实际启动并保存 [独立审查](claude_independent_review.md)，之后进行 [交叉核对](claude_cross_review.md)。唯一问题清单为 [issue_register.csv](issue_register.csv)，保留接受/驳回依据。Claude的完整原始返回、session及重算取证日志另存；**任何PASS都不以agent总结作为唯一依据**。
- 直接读取72训练checkpoint，哈希、最早validation最优epoch、frozen不变/full更新全部核对通过。独立NumPy重算全部100份已报告预测（56分类、44分割）；所有主指标与表一致，除foreground NLL外最大绝对差 `5.08e-08`。foreground NLL按原生dtype/clip独立复现最大差 `2.38e-09`，详见数值说明。
- 16 ensembles：分类直接读取全部真实成员预测；分割对固定前8图逐像素平均抽验。12个分割MC：使用预定research subset首图的全部30pass做聚合抽验。所有样本/标签对齐，最大概率差 `<3e-8`。这是聚合抽查的范围，不能称完整逐pass分割全量复现；全test聚合结果的指标已全量重算。
- 37个相关现有测试通过，见 [targeted_tests.log](targeted_tests.log)。未启动训练，未修改训练/指标源代码、旧checkpoint或历史结果。
- 本报告的 `status` 是截至审查结束（含本次低成本新增证据）的状态；`snapshot_status` 保留原快照的缺口。R1.3/R1.4证据已补齐，但正式报告仍需同步。C01/C06的UNKNOWN严格限历史来源，不能自动否定其他比较。

## A. 已声明与实际覆盖

依据现行 `final_training_protocol.md`、`final_dataset_protocols.md`、`pre_uq_protocol_freeze.md` 和逐run配置建立矩阵。**64个方法格：52格存在全部承诺结果，12格TS为NA；总计100份结果。** 这只是覆盖完成，不等于全部有效性检查PASS。

| 方法 | 已声明 | 实际 | 仍需区分 |
|---|---|---|---|
| Deterministic | 16格 × seeds42/43/44 | 48预测、48checkpoint | 旧DOFA-EuroSAT6run训练时代码来源UNKNOWN |
| TS | EuroSAT4格 × seeds42/43/44 | 12预测，专用calibration拟合 | TreeSatAI4格及分割8格NA；历史TS单独保留 |
| MC Dropout | 16格seed42 + 4预定格加43/44 | 24模型/预测 | 原MC−D是全方案差异；新增12分类Off，仍缺12分割Off |
| Deep Ensemble | 每格一组42/43/44成员 | 16组概率平均结果 | 每格只有一个ensemble，无ensemble训练重复SD |

完整64格逐项表：[coverage_matrix.csv](coverage_matrix.csv)。100份逐产物索引（模型权重版本、输入/训练/模块契约、checkpoint、配置hash、run代码hash、预测位置）：[actual_artifacts.csv](actual_artifacts.csv)。MC匹配对照缺口：[mc_dropout_off_gaps.csv](mc_dropout_off_gaps.csv)。

预训练权重为 DOFA `DOFA_ViT_base_e100.pth`（SHA256 `4720985e…165b1`）和 Panopticon `VIT_BASE14` teacher（SHA256 `55024f41…26e`，源revision `c8c2bb9555819e8b2bcedf5b3b00e3bf531554e7`）；本轮重算文件hash见 [pretrained_weight_hashes.txt](pretrained_weight_hashes.txt)。不要求第三个FM或非FM基线。

| 数据集 | 任务/输入 | train / selection / calibration / test | 泛化边界 |
|---|---|---|---|
| EuroSAT | 10类，RGB B04/B03/B02 | 18866 / 2707 / 2713 / 2714 | spatial_group分离；源景ID不可得；两FM归一化不同 |
| TreeSatAI | 15标签，12个S2波段，T=1静态 | 4000 / 1000 / 无独立split / 2000 | ID/source分离；不可宣称多时相 |
| CloudSEN12 | 4类分割，12个S2波段 | 4000 / 535 / 不拟合TS / 975 | ROI/equi分离，3个acquisition product跨split |
| SpaceNet7 | 2类建筑分割，RGB | 3500 / 652 / 不拟合TS / 1152 | AOI完全分离；test12AOI，像素非独立重复 |

实际索引全集及交集检查见 [split_checks.json](split_checks.json) 和它指向的原始CSV。分类按validation NLL最小、分割按validation mIoU最大，平局最早epoch；72个checkpoint与完整history逐项相符。校准定义：15等宽右闭箱、空箱不贡献、0–1单位；multiclass Brier逐类求和，多标签Brier逐sample-label平均；ignore mask统一。TreeSatAI主ECE是展平二元决策置信度，不是类概率处处校准的证明。

TreeSatAI TS的NA是现行范围决定。8-24历史val拟合结果已经存在，8-25才作排除；不能称为全部TS结果前预注册，也不能据此推断动机。标准允许披露后复用selection validation拟合TS，故“不独立命名calibration”不是必然无效。保持历史诊断和不利结果索引，详见 [protocol_timeline_and_na.csv](protocol_timeline_and_na.csv)。分割TS按已声明协议NA，不替协议发明其他理由。

## B. 29项检查

最终状态计数：{'UNKNOWN': 3, 'PASS': 22, 'FAIL': 4}。**不按通过率给论文打总分。** 完整机器可读字段（包括run/checkpoint ID、code revision、影响、最小补救、owner、acceptance evidence）见 [checks.csv](checks.csv) / [checks.json](checks.json)。下表是其精简视图；每行的证据文件继续指向真实代码/产物，不以另一agent意见替代。

| ID | RQ | 状态 | 发现/范围 | 直接证据与本轮验证 | 最小补救 |
|---|---|---|---|---|---|
| C01 | RQ1;RQ2;RQ3 | UNKNOWN | 四数据集 test ID 与训练/选择集分离，72 checkpoint 均按最早 val 最优选择；旧 DOFA–EuroSAT 归一化常数的统计来源未保留（MC DOFA–EuroSAT沿用）。 | splits/eurosat_70_10_10_10_spatial20m/eurosat_splits.csv;reports/dataset_manifests/*_actual_manifest.csv;scripts/run_experiments.py:2138;reports/final_training_protocol.md:90；split_checks.json;checkpoint_verification.json；旧 run environment/config 见 actual_artifacts.csv | 查回常数 derivation；无法恢复则声明该来源 UNKNOWN，保留本 benchmark 描述性结果。 |
| C02 | RQ3 | PASS | EuroSAT 12 个 TS 的 calibration IDs 精确匹配专用 split，均与 test 分离；TreeSatAI TS/分割 TS 是现行协议 NA。 | results/final_thesis/c3_temperature_scaling_and_ensembles/classification/temperature_scaling/eurosat/*/*/seed*/metrics_and_manifest.json;reports/pre_uq_protocol_freeze.md:72；classification_verification.json:calibration_ids_exact/calibration_test_disjoint | 保留现行范围；说明 TreeSatAI 历史 val 复用与排除时序。 |
| C03 | RQ1;RQ2;RQ3 | PASS | EuroSAT CE/softmax/argmax；TreeSatAI 15标签 BCE/sigmoid/0.5阈值，accuracy是exact match；CloudSEN12四类，SpaceNet7标签1→0、2→1、0→ignore255。 | scripts/run_experiments.py:1104;scripts/prediction_export.py:527;scripts/geobench_datasets.py:158;reports/dataset_manifests/treesatai_actual_manifest.csv；classification_verification.json;segmentation_verification.json;targeted_tests.log | 在论文指标定义中保留任务差异。 |
| C04 | RQ1;RQ2;RQ3 | PASS | 直接读取72 checkpoint并与实际预训练tensor比较：frozen所有可匹配backbone tensor未变，full均有更新；权重哈希匹配；输入契约/结构冻结例外与代码和config一致。 | scripts/run_experiments.py:872;scripts/geobench_datasets.py;results/**/model_audit.json;results/**/gradient_audit.json;DOFA/checkpoints/DOFA_ViT_base_e100.pth;models/pretrained_cache/hub/checkpoints/panopticon_vitb14_teacher.pth；checkpoint_verification.json;pretrained_weight_hashes.txt;actual_artifacts.csv;targeted_tests.log | 无需修训练代码或重训；保留DOFA固定pos_embed与Panopticon非光学嵌入结构例外。 |
| C05 | RQ1;RQ2;RQ3 | FAIL | 主指标数值可重现。报告层：C6 TreeSatAI图使用 pooled正类概率，表ECE使用binary-decision confidence；图用[lo,hi)而表用(lo,hi]，未清楚对照两个量。分割foreground NLL还需声明原生dtype与截断。 | scripts/c6_build_final_results.py:471;scripts/prediction_export.py:538;scripts/c3_calibration_ensembles.py:117;scripts/segmentation_pipeline.py:410；classification_verification.json;segmentation_verification.json;foreground_precision_verification.json；本次*_reliability_bins.csv明确kind和边界 | 报告生成器统一分箱，分别命名decision-ECE与label-probability ECE；公开Brier归一化、ignore、截断策略。已有预测足够。 |
| C06 | RQ1;RQ2;RQ3 | UNKNOWN | 100份预测全量主指标重算、样本/标签对应及72checkpoint哈希均有直接证据；旧DOFA–EuroSAT六run只保存事后代码快照，training-time git_commit=null。 | reports/dofa_eurosat_final_manifest.json:source_snapshot.provenance_scope;results/baselines/dofa_eurosat_*/runs/*/environment.json;results/**/code_snapshot.json；actual_artifacts.csv;classification_verification.json;segmentation_verification.json;checkpoint_verification.json | 尽量恢复历史source；否则保留UNKNOWN与受影响run列表，不能编造版本或默认重训。 |
| C07 | RQ1;RQ2;RQ3 | PASS | 48 deterministic seeds、24 MC seeds、12 TS seed结果和16单组ensemble分开；mean/std及成对差值均由独立重算核验。 | reports/thesis_master_results.csv;reports/rq2_adaptation_effects.csv;reports/rq3_uq_effects.csv;scripts/build_rq_effect_tables.py；summary_arithmetic_checks.json;paired_effects_from_predictions.csv | 只做n=3或n=1的描述性推断；若将来给评估区间，另注明固定模型/组抽样层级。 |
| C08 | RQ1;RQ2;RQ3 | PASS | 直接manifest复核：EuroSAT ID/内容hash/spatial_group不跨split；TreeSatAI ID/source不跨；Cloud ROI/equi不跨但3个product跨；SpaceNet7 AOI/source/patch不跨，test12AOI。 | splits/eurosat_70_10_10_10_spatial20m/spatial_leakage_report.json;reports/dataset_manifests/*_actual_manifest.csv;reports/final_dataset_protocols.md；split_checks.json | 保留Cloud同acquisition与Euro源景不可得限制；将来bootstrap须按合理组且比较组共同抽样。 |
| C09 | RQ1;RQ2;RQ3 | FAIL | checkpoint选择与异常运行排除有记录且直接复核通过；但C6最终包把TreeSatAI TS列COMPLETE，现行master/rq3表列NA。TS产物8-24已产生，排除冻结8-25。 | scripts/c6_build_final_results.py:258;reports/final_thesis_results.md:10;reports/final_thesis_tables/classification_results.csv;reports/pre_uq_protocol_freeze.md:72;reports/c3_treesatai_temperature_scaling_results.csv；protocol_timeline_and_na.csv;checkpoint_verification.json;issue_register.csv | 让报告生成器消费权威master，统一NA；说明决定时序并保留历史诊断及不利结果索引。不得把validation复用称为数学无效，也不擅自重新纳入核心范围。 |
| R1.1 | RQ1 | PASS | 两个FM在相同test样本/标签、适配条件下比较。EuroSAT历史DOFA与新Panopticon归一化不同；其他三dataset共用输入契约。 | reports/final_training_protocol.md:85;results/**/resolved_config.yaml;scripts/run_experiments.py:345；actual_artifacts.csv:data_contract/model_contract;classification_verification.json;segmentation_verification.json | 写明整套配置比较；纯架构隔离属于额外范围，当前不要求补训练。 |
| R1.2 | RQ1 | PASS | 分类accuracy/Macro-F1/ECE/NLL/Brier、分割mIoU/per-class IoU/pixel accuracy和概率指标齐全；TreeSatAI准确率语义在权威master明确。 | reports/thesis_master_results.csv:metric_semantics;results/**/predictions/test/deterministic/metrics.json;scripts/prediction_export.py:527；classification_recomputed_metrics.csv;segmentation_recomputed_metrics.csv | 发布时采用exact-match accuracy和明确的ECE名称。 |
| R1.3 | RQ1 | PASS | 本轮从全部保存预测补算10/15/30箱、每箱count/confidence/outcome和signed gap，并生成seed42图；全部seeds箱表保留。旧C6不显示count且未标seed。 | scripts/c6_build_final_results.py:487;actual_artifacts.csv:artifact_path；classification_reliability_bins.csv;segmentation_reliability_bins.csv;*_reliability_with_counts.png | 把审查新增图表纳入正式报告并标seed42，不用单个低计数箱外推。 |
| R1.4 | RQ1 | PASS | SpaceNet7已有全有效像素p(building)对indicator校准和建筑IoU；本轮补齐Cloud各类、TreeSatAI15标签、SpaceNet7每类的概率箱表与类指标。 | scripts/segmentation_pipeline.py:504;results/**/predictions/test/deterministic/per_class_metrics.json;actual_artifacts.csv:artifact_path；class_diagnostics.csv;segmentation_class_diagnostics.csv;*_class_probability_reliability.png;treesatai_*_labels_*.png | 正式表加入必要类条件诊断，解释类别支持数和二类冗余列；不要求提高建筑IoU。 |
| R1.5 | RQ1 | PASS | 保存全部既定seeds；本轮给Panopticon−DOFA逐seed效应、均值/sample SD/min/max，并用10/15/30箱检查细小ECE排序。 | actual_artifacts.csv:artifact_path;reports/thesis_master_results.csv；paired_effects_from_predictions.csv;paired_effect_summaries.csv;classification_recomputed_metrics.csv;segmentation_recomputed_metrics.csv | 保留逐seed与替代分箱结果，弱差异用描述性措辞。 |
| R1.6 | RQ1 | UNKNOWN | 仓库没有完整论文正文；已有证据表未充分陈述逐条件过/欠自信与类条件限制。本报告给出直接答案草稿。 | thesis/experiment_1_dofa_rgb_record.md;reports/thesis_evidence_matrix.md;reports/final_thesis_results.md；AUDIT_REPORT.md:RQ1草稿（不是未提供正文的替代证据） | 采用本报告RQ1草稿和相应表图后，对实际正文复审；无需训练。 |
| R2.1 | RQ2 | PASS | frozen只训练head/decoder、full更新backbone；72checkpoint与预训练tensor直接比较及梯度/optimizer记录支持。 | results/**/model_audit.json;results/**/optimizer_group_audit.json;results/**/gradient_audit.json;scripts/run_experiments.py:1759；checkpoint_verification.json;run_evidence.json;targeted_tests.log | 无需补训练；保留可训练模块/例外清单。 |
| R2.2 | RQ2 | PASS | 同FM/dataset/seed的两适配data配置完全一致、head概念可比、无augmentation、selector相同；LR/WD/warmup/预算/patience按方案不同。 | reports/final_training_protocol.md:24;results/**/resolved_config.yaml；adaptation_config_pairs.json | 正文列出配方差异并收窄因果措辞，不强制相同LR。 |
| R2.3 | RQ2 | PASS | 24个matched-seed frozen→full对照及8个三seed摘要完整，逐seed差值和差值SD直接重算。 | reports/rq2_adaptation_effects.csv;actual_artifacts.csv:artifact_path；paired_effects_from_predictions.csv;paired_effect_summaries.csv;summary_arithmetic_checks.json | 保留逐seed对照，不用独立两表替代配对。 |
| R2.4 | RQ2 | PASS | 每个对照同时给ΔPerformance/ΔECE/ΔNLL/ΔBrier，分割另保留建筑IoU。 | reports/rq2_adaptation_effects.csv;actual_artifacts.csv:artifact_path；paired_effect_summaries.csv | 采用本报告逐dataset结论及配对表。 |
| R2.5 | RQ2 | PASS | 两FM、四dataset分类/分割均有对照；方向随条件变化，反例完整。 | reports/rq_effect_summary.md;actual_artifacts.csv:artifact_path；paired_effect_summaries.csv;AUDIT_REPORT.md:RQ2 | 结论限定声明矩阵；不为制造一致方向继续训练。 |
| R3.1 | RQ3 | FAIL | TS同checkpoint，ensemble对其真实成员，原MC对同seed独立p=0模型。分类12个同权重Off对照本轮已补；分割12个仍缺。 | scripts/c5_mc_dropout_inference.py:351;scripts/build_rq_effect_tables.py:312;results/final_thesis/mc_dropout/run_registry.json；classification_dropout_off_three_way.csv;mc_dropout_off_gaps.csv | 复用12个分割MC checkpoint做dropout-off全test一次前向；保留D/Off/MC三方表，不重训。 |
| R3.2 | RQ3 | PASS | EuroSAT正温度只用专用calibration labels拟合NLL，在相同未拟合test上比较。 | results/final_thesis/c3_temperature_scaling_and_ensembles/classification/temperature_scaling/eurosat/*/*/seed*/metrics_and_manifest.json;scripts/c3_calibration_ensembles.py:194；classification_verification.json:temperature/calibration_ids_exact | 保持拟合/评估来源披露；不为NA补训练。 |
| R3.3 | RQ3 | PASS | 12个EuroSAT TS标量T>0，scaled logits为raw/T，raw logits与原预测逐项一致，argmax变化均0。 | scripts/c3_calibration_ensembles.py:194;results/final_thesis/c3_temperature_scaling_and_ensembles/classification/temperature_scaling/eurosat/*/*/seed*/test_predictions.parquet；classification_verification.json:argmax_changes/scaled_logits_max_error | 无需修改；保留温度和不变性检查。 |
| R3.4 | RQ3 | FAIL | p=.1在训练时启用，只推理head/decoder指定module，BN eval/无更新，T=30，概率均值；分类全pass及分割固定subset直接聚合抽验通过。分割同权重Off仍缺。 | scripts/mc_dropout.py:32;scripts/c5_mc_dropout_inference.py:237;results/final_thesis/mc_dropout/inference/**/manifest.json;results/final_thesis/mc_dropout/inference/**/c5_audit.json；aggregation_verification.json;classification_verification.json;classification_dropout_off_verification.json;mc_dropout_off_gaps.csv | 补12个分割Off eval；分类复用新增三方表即可，均不需训练。 |
| R3.5 | RQ3 | PASS | 16 ensembles来自独立seed42/43/44训练checkpoint，成员IDs/labels一致；直接成员概率平均：分类全测试、分割固定前8图，最大误差<3e-8。主指标对ensemble概率重算。 | scripts/c3_calibration_ensembles.py:636;results/final_thesis/**/deep_ensembles/**/manifest.json;results/final_thesis/dofa_eurosat_ensembles/*/manifest.json；aggregation_verification.json;classification_verification.json;segmentation_verification.json | 无需新增训练；保留成员表及聚合层级。 |
| R3.6 | RQ3 | PASS | 现行承诺的TS/ensemble/MC全方案比较均保留性能和三概率指标差值，负结果保留；分类新增MC−Off/Off−D/MC−D三种差值。 | reports/rq3_uq_effects.csv;actual_artifacts.csv:artifact_path；paired_effects_from_predictions.csv;classification_dropout_off_three_way.csv | 正式表采用三方对照并等待分割补评估结果，无论方向如何均保留。 |
| R3.7 | RQ3 | PASS | 16 MC格seed42，只有预定4格三seed；ensemble每格一组；30pass不是n=30。 | reports/pre_uq_protocol_freeze.md:146;reports/thesis_master_results.csv:record_type/replication_role；coverage_matrix.csv;paired_effect_summaries.csv | 保留replication_role和比较基准，不强制扩展成笛卡尔积。 |
| R3.8 | RQ3 | PASS | EuroSAT full ensemble性能/NLL/Brier改善但ECE变差；SpaceNet7 ensemble概率指标改善而建筑IoU下降；MC有改善也有反例。 | reports/rq3_uq_effects.csv;reports/c3_treesatai_temperature_scaling_results.csv;actual_artifacts.csv:artifact_path；paired_effect_summaries.csv;classification_dropout_off_three_way.csv;protocol_timeline_and_na.csv | 逐指标、逐条件陈述；对已排除历史结果写明范围与时间线。 |
| R3.9 | RQ3 | PASS | 本报告按实际ΔPerformance与ΔECE分类；EuroSAT TS预测严格不变，其余报告实际损益而不以不显著宣称无损。 | actual_artifacts.csv:artifact_path;reports/rq3_uq_effects.csv；effect_outcome_categories.csv;classification_dropout_off_three_way.csv;AUDIT_REPORT.md:RQ3 | 采用effect_outcome_categories.csv的点估计类别和重复限制；如果要证明小损失可忽略，再预定义容忍范围并按组评估。 |

## C. 逐研究问题：当前能回答什么

### RQ1 — PARTIAL；实证核心已能描述，正式报告与部分来源仍未闭合

当前能回答：在这两个FM、四个固定benchmark和两种适配配方下，deterministic预测的任务性能和经验校准程度是多少，以及模型排序是否随条件改变。48个训练seed的原始预测均已重算；本轮补齐逐箱方向、count、类概率与10/15/30箱敏感性。

直接答案草稿：EuroSAT frozen的decision-ECE约0.0050（DOFA）和0.0066（Panopticon），full约0.0111和0.0095；两模型的细小排序不能解释为稳定架构优势，因为seed波动、分箱和归一化差异均相关。TreeSatAI的低decision-ECE主要包含大量负标签判断；例如Panopticon/full/seed42 decision-ECE=0.011267，正类概率汇总ECE=0.018803，逐标签ECE宏均值=0.033700，不能把第一项解释为15标签均充分校准。CloudSEN12整体常见过度自信，但各类和各箱方向须分别看。SpaceNet7总体pixel accuracy约0.92、decision-ECE约0.04，建筑IoU仅约0.052–0.093，整体分数不能掩盖建筑表现；这本身是有效结果。

方向证据以逐箱confidence−outcome为准：正值过度自信、负值不足自信；允许同一模型不同箱同时出现两种方向。不能用平均gap的小值替代各箱诊断。下列图均明确为deterministic seed42，所有seed原始箱值另存CSV：

- [EuroSAT reliability与count](eurosat_reliability_with_counts.png)、[TreeSatAI decision reliability与count](treesatai_reliability_with_counts.png)。
- [CloudSEN12 reliability与count](cloudsen12_reliability_with_counts.png)、[Cloud逐类概率](cloudsen12_class_probability_reliability.png)。
- [SpaceNet7 reliability与count](spacenet7_reliability_with_counts.png)、[背景/建筑类概率](spacenet7_class_probability_reliability.png)。
- TreeSatAI15标签图为 `treesatai_{model}_{adaptation}_labels_*.png`；对应 [类诊断表](class_diagnostics.csv)。所有图源数据：[classification_reliability_bins.csv](classification_reliability_bins.csv)、[segmentation_reliability_bins.csv](segmentation_reliability_bins.csv)。

有依据的结论：没有跨所有条件一致的FM校准冠军；某些条件总体ECE低，同时关键类别性能/类概率校准仍弱。NLL/Brier是概率质量，不能称校准误差。未知/必要缺口：历史DOFA-EuroSAT来源不完整、旧正式图表定义/NA状态未统一，完整论文正文未提供。现有图表已能形成局部充分回答，但不能给未审正文或历史版本来源补一个PASS。**需要修报告代码与表述、纳入本轮重分析；不需要补训练，也无因RQ1本身新增模型推理的已确认需求。**

### RQ2 — ANSWERED；限于所采用适配方案，不是仅冻结开关的因果效应

24个同FM/dataset/seed的frozen→full对照完整，数据和评估样本一致，checkpoint选择在任务内一致。full和frozen的LR、WD、warmup、epoch上限、patience差异已公开，因此结论指完整适配配方。

直接答案：full fine-tuning对校准没有统一方向。EuroSAT两FM的性能与NLL/Brier/ECE一起变差；TreeSatAI的NLL/Brier改善，Macro-F1点估计上升，但ECE方向/跨seed稳定性不同；CloudSEN12 mIoU上升、Brier下降，ECE未随性能稳定改善；SpaceNet7建筑/mIoU小幅改善，概率质量/校准可变差。这些“不统一/存在权衡”的结果已经回答RQ2。

下表均为full−frozen，±为三个逐seed差值的sample SD，Performance在EuroSAT为accuracy、TreeSatAI为Macro-F1、分割为mIoU，不跨任务汇总：

| dataset | FM | Performance | ΔPerformance | ΔECE | ΔNLL | ΔBrier |
|---|---|---|---|---|---|---|
| eurosat | dofa | accuracy | -0.01855 ± 0.00583 | +0.00611 ± 0.00611 | +0.05255 ± 0.02389 | +0.02727 ± 0.00925 |
| eurosat | panopticon | accuracy | -0.02063 ± 0.00706 | +0.00283 ± 0.00431 | +0.06215 ± 0.03444 | +0.03012 ± 0.01292 |
| treesatai | dofa | macro_f1 | +0.02066 ± 0.01539 | +0.00175 ± 0.00377 | -0.00899 ± 0.00228 | -0.00229 ± 0.00077 |
| treesatai | panopticon | macro_f1 | +0.00891 ± 0.02608 | -0.00187 ± 0.00268 | -0.00654 ± 0.00306 | -0.00219 ± 0.00120 |
| cloudsen12 | dofa | miou | +0.05236 ± 0.00642 | +0.02367 ± 0.02186 | +0.00635 ± 0.02629 | -0.02975 ± 0.00209 |
| cloudsen12 | panopticon | miou | +0.01809 ± 0.00420 | +0.00054 ± 0.00811 | -0.02565 ± 0.02324 | -0.01471 ± 0.00472 |
| spacenet7 | dofa | miou | +0.00953 ± 0.00076 | +0.01051 ± 0.00929 | +0.15775 ± 0.10397 | +0.00333 ± 0.00521 |
| spacenet7 | panopticon | miou | +0.00298 ± 0.00539 | +0.00413 ± 0.01047 | +0.07697 ± 0.08490 | +0.00489 ± 0.00395 |

逐seed、模型差值及min/max见 [paired_effects_from_predictions.csv](paired_effects_from_predictions.csv) / [paired_effect_summaries.csv](paired_effect_summaries.csv)，与既有结果表的独立核验见 [summary_arithmetic_checks.json](summary_arithmetic_checks.json)。没有显著性标签或等效性声称；小样本不能证明“没有影响”。**无需为RQ2补训练或补评估；只需把配方差异、seed限制与反例写入正文。** 若另要声称仅冻结/解冻的因果作用，属于新范围，不是当前三问的默认补实验。

### RQ3 — PARTIAL；TS/ensemble可答，分类MC同权重对照已补，分割仍缺

**TS（仅EuroSAT）**：12个T均为正，同checkpoint原logits/T，专用calibration拟合，test argmax零变化，因此accuracy严格保持。ECE/NLL改善依模型/适配/seed而变，frozen的额外收益很小；不能把“未改善”当失败。TreeSatAI和分割保持现行NA，历史TreeSatAI诊断可说明val复用情境但不混入核心比较。

**Deep Ensemble**：每格一组三seed真实概率均值。CloudSEN12四格相对其精确成员均值的mIoU增加约0.013–0.018，同时ECE/NLL/Brier改善。SpaceNet7四格ECE/NLL/Brier改善，但mIoU下降约0.0065–0.0110，建筑IoU亦下降，是明确的性能—校准权衡。EuroSAT full的ensemble可提高accuracy/NLL/Brier却提高ECE（DOFA约+0.0162、Panopticon约+0.0125），所以不能说所有概率指标改善。只有一组ensemble，不声称跨ensemble训练稳定。

**MC Dropout**：设D为原独立p=0训练模型，Off为MC模型同权重关闭dropout，MC为同权重30次概率均值。

- MC−D：现行已报告的完整训练加推理方案差异；同seed只是设计配对。
- MC−Off：固定权重下采用指定dropout随机推理/有限30次平均的差异；不自动证明完整贝叶斯或机制解释。
- Off−D：训练方案及随机轨迹的差异，不能单独归因dropout正则化。

原矩阵只有MC−D。本轮直接用已存backbone features和同checkpoint的eval BatchNorm/Linear权重完成全部12个分类Off结果，保存新的logits/probabilities与三方delta：[classification_dropout_off_three_way.csv](classification_dropout_off_three_way.csv)。这是重算已有表征，不是重新训练或运行backbone。EuroSAT/DOFA/frozen/seed42：Off accuracy=0.984893、ECE=0.004339；MC accuracy=0.985262、ECE=0.008762；MC−Off的Δaccuracy=+0.000368、ΔECE=+0.004423、ΔNLL=−0.000671、ΔBrier=+0.000393。说明MC采样并未在此改善ECE，且不同指标存在冲突。其余逐seed均保留，不能只挑此例或正例。

**尚缺的必要证据只有12个分割MC checkpoint的Off测试预测及三方差值**，影响R3.1/R3.4。checkpoint已经存在；只需每个模型关闭指定dropout做一次完整test前向，保持相同IDs、mask、类序、后处理与指标。现有32图research subset和30pass预测不能精确还原非线性softmax之前的dropout-off结果，因此不能从概率均值代替Off，也无需训练。补评估若改善、无改善或变差，分别支持收益、无观察收益或不利结果，均可完成RQ3。

全部现有方法/比较格的点估计分类见 [effect_outcome_categories.csv](effect_outcome_categories.csv)。除TS的类别不变性外，没有预定义实际损失容忍度或按组CI，所以报告实际损益，不用“p不显著”或微小差值宣称无代价。MC稳健性只能限于预定四个三seed格。

## D. 必需后续任务与验收

| 类型 | 最小任务 | 检查ID / RQ | 现有产物能否解决、验收条件 |
|---|---|---|---|
| 修报告代码 | C6改为消费权威master；TS NA/图例/正文一致，统一分箱、显示counts/seed、明确ECE对象 | C05/C09/R1.3，RQ1/3 | 可以，无需模型推理；输出表/图与master范围一致 |
| 已有结果重分析 | 本轮已完成100预测重算、逐箱/类概率/替代分箱、72checkpoint核对、12分类Off | C04–C07/R1.3/R1.4/R3.4 | 成果在本目录；正式报告纳入而非覆盖不可变历史产物 |
| 补评估 | 12分割MC checkpoint关闭dropout，单次完整test推理 | R3.1/R3.4/R3.9，RQ3 | 保存hash/IDs/labels/mask/logits/probabilities和全部主、建筑、边界指标；与D/MC三方配对；不训练 |
| 恢复证据/限定表述 | 查回历史DOFA-EuroSAT训练版本/归一化来源，若找不到如实保留UNKNOWN | C01/C06，三RQ中的相关子范围 | 不能拿事后快照冒充训练版本；无需自动重训 |
| 修改正文 | 逐RQ给条件化答案，公开TreeSatAI TS时序、EuroSAT预处理差异、EO相关性和重复层级；用实际损益而非普遍无损 | R1.6/R2.2/R2.5/R3.8/R3.9 | 完整正文提供后复审，当前仅给答案草稿 |
| 新训练 | **当前没有必要项** | — | 未发现现有训练被具体实现/数据错误破坏，也未发现必须新训练才能补的承诺对照 |

数值说明：独立float64 raw-probability重算foreground NLL与原torch原生dtype会略异，例SpaceNet7/DOFA/frozen/seed44为0.335851540 vs报告0.335846698。float32中`1−1e−7`实际为0.9999998807907104；按该clip重算为0.335846696。全部22个SpaceNet7产物的原生路径复现见 [foreground_precision_verification.json](foreground_precision_verification.json)，保留raw重算差异而非删掉警告。既有CloudSEN12一像素ULP情况亦不触发重训；本轮核对的是保存预测与其指标，不承诺重新GPU推理逐bit一致。

## E. 单独列出的后续探索（本轮不要求）

OOD/corruption、label scarcity、CKA/表征探针、AU/EU或机制解释、边界干预、匹配归一化的纯架构因果实验、传统模型对照、扩增所有MC seeds或独立ensemble组，均不自动纳入本轮。只有将来的正文提出对应实质主张，才需另行范围决策。SpaceNet7边界校准已由现行freeze明确要求，因此本轮保留其全有效边界像素统计及无边界图undefined，不能把undefined当零。

## F. 复核路径

本目录脚本仅写本目录：`build_inventory.py`、`verify_predictions.py`、`verify_checkpoints.py`、`verify_aggregation.py`、`recompute_classification_dropout_off.py`、`verify_foreground_precision.py`、`summarize_evidence.py`、`write_report.py`。每个PASS对应原始路径、代码位置和实际run/checkpoint，详见checks及artifact索引。大二进制仅读取；旧报告/配置/模型未被静默修复。历史快照完整性复查与本目录最终文件hash见 `snapshot_integrity.json` / `audit_output_manifest.json`。
