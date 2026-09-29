"""Publish the completed, scoped RQ narrative and 29-item resolution register."""
from pathlib import Path
import csv,json,hashlib,collections,statistics
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent;OLD=ROOT/'reports/core_rq_audit_20260918'
def readcsv(p):return list(csv.DictReader(p.open()))
def writecsv(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def j(p):return json.loads(p.read_text())
def mdtable(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(x):return f'{float(x):+.6f}'
def rootpaths(value):return ';'.join(str((OUT/v).resolve().relative_to(ROOT)) for v in value.split(';'))
def main():
 verification=j(OUT/'segmentation_off_verification_summary.json');assert verification['complete'] and verification['failed']==0
 pub=j(OUT/'published/tables/package_validation.json');assert pub['all_passed'] and pub['na_cells']==12
 effects=readcsv(OUT/'mc_dropout_effects.csv');sums=readcsv(OUT/'mc_dropout_effect_summaries.csv');three=readcsv(OUT/'mc_dropout_three_way.csv')
 assert len(three)==24 and len(effects)==72 and len(sums)==48
 # Do not claim to review an absent external dissertation: this is the results
 # section written for this completion package itself.
 draft=(OUT/'rq1_rq2_draft.md').read_text()
 draft=draft.replace('# RQ1 与 RQ2：可采用的结果段落（2026-09-21 UTC）','# 三个研究问题：整理后的结果正文（补充于 2026-09-23 UTC）')
 draft=draft.replace('本文件直接依据逐预测重算指标、实际配置、checkpoint 和原始预测路径编写；不是对另一 agent 总结的转述。它是本次补充报告的可复核结果正文，不声称未提供的外部论文全文已经通过审阅。','本结果正文依据逐预测重算指标、实际配置、checkpoint 和原始预测文件。它是本次发布的可复核正文，不声称未提供的外部论文全文已经通过审阅。主报告见 [完成记录](COMPLETION_REPORT.md)。')
 draft=draft.replace('RQ1 的必要工作是把上述定义、条件性结论和类诊断纳入正式发布，修复原报告生成器的对象/分箱与 count 问题；已有预测足以支持这些分析，无已确认的 RQ1 必须新增推理或训练项。','上述定义和诊断现已纳入 [重建发布包](published/final_thesis_results.md)，生成器已修复对象/分箱和 count 问题，并由 package_validation.json 直接核对主表及图表ECE。已有预测支持这些分析，无已确认的 RQ1 必须新增推理或训练项。')
 draft=draft.replace('所有可靠性图应与计数直方图、指标对象、分箱边界和 seed 标签一起发布；','本次可靠性图与计数直方图、指标对象、分箱边界和 seed 标签一起发布；')
 draft=draft.replace('SpaceNet7 foreground NLL 的原生 float32 上截断与 float64 直接计算的差异已由', 'SpaceNet7 foreground NLL 先以原生 float32 将建筑概率截断到 [1e-12, 1−1e-7]，再在全部有效像素上计算二元负对数似然；不能替换为对真实类别概率采用1e-7下界。其原生 float32 上截断与 float64 直接计算的差异已由')
 draft=draft.split('## 本次正文整理的验收边界')[0]
 lines=[draft,'## RQ3：校准收益、性能代价与同权重对照','',
 '核心TS只含EuroSAT四个cell、每格三个seed；TreeSatAI四格与分割八格保留现行NA。历史TreeSatAI TS验证集拟合、测试集评估的12次结果完整保留在 [历史诊断表](published/tables/historical_treesatai_temperature_scaling.csv)，包括不利结果。结果在2026-08-24已可见，8月25日范围冻结排除，8月26日旧C6仍列COMPLETE，8月27日master列NA。新表图及正文统一遵守master，并公开此时序；不能称这一排除决定先于所有TS结果，也不据此推断作者动机。validation复用并不自动使held-out评估无效。','',
 'TS用正的单个温度缩放同一checkpoint logits，12个EuroSAT结果argmax变化为0，因此accuracy严格保持。ECE与NLL/Brier收益因模型/适配/seed不同；这是当前测试集上的实测结论，并不保证对任意分布都改善。逐seed及四格汇总见 [原始方法配对](paired_effects_from_predictions.csv) 与 [摘要](paired_effect_summaries.csv)。','',
 'Deep Ensemble每格只有一组seed42/43/44成员的概率均值，性能代价相对这三个成员的均值报告，不把成员个数当作独立ensemble重复。CloudSEN12四格mIoU上升约0.013–0.018，同时ECE/NLL/Brier下降；SpaceNet7四格ECE/NLL/Brier下降，但mIoU下降约0.0065–0.0110，建筑IoU也下降，构成明确的点估计权衡。EuroSAT full的ensemble相对成员均值改善accuracy、NLL/Brier，却使ECE上升（DOFA约+0.0162，Panopticon约+0.0125）。这些反例全部保留，不能概括为ensemble在全部条件下改善校准或没有性能代价。','',
 'MC的D、Off和MC三方现在覆盖全部24个已训练checkpoint。D是原p=0训练方案；Off使用p=.1训练checkpoint并关闭全部dropout；MC使用同checkpoint既有30次随机概率均值。MC−Off回答固定权重下当前有限30次推理的观测差异；Off−D包含训练配方、随机轨迹和所选epoch的变化；MC−D回答总方案差异。分类/分割MC checkpoint均以dropout-off验证指标选择，MC是checkpoint选择后的推理改动。即使seed编号相同，dropout也会改变随机流，因此不能把Off−D纯归因于dropout正则。','',
 '[24行三方表](mc_dropout_three_way.csv) 同时给出三组绝对指标、三组差值、两条训练轨迹best/last epoch、D的三个seed范围、checkpoint及预测路径。[72行效应表](mc_dropout_effects.csv) 保留所有既定seed；[48行摘要](mc_dropout_effect_summaries.csv) 报告逐比较块均值、sample SD及min/max。主矩阵仍逐格显示seed42；仅预定EuroSAT DOFA frozen、TreeSatAI Panopticon full、CloudSEN12 Panopticon frozen、SpaceNet7 DOFA full四格有3个训练重复。以下16格为MC−Off摘要，n=1只支持该模型实例，n=3的SD描述训练随机差异，不是置信区间。性能/ECE以比例值（非百分数）报告，NLL为nats，Brier保持各任务原归一化。','']
 table=[]
 for r in sums:
  if r['contrast']!='MC_minus_Off':continue
  n=int(r['n_training_seeds']);perf=fmt(r['delta_performance_mean']);ece=fmt(r['delta_ece_15_mean'])
  if n>1:perf+=' ± '+f"{float(r['delta_performance_sample_sd']):.6f}";ece+=' ± '+f"{float(r['delta_ece_15_sample_sd']):.6f}"
  table.append([r['dataset'],r['model'],r['adaptation'],n,r['primary_metric'],perf,ece,fmt(r['delta_nll_mean']),fmt(r['delta_brier_mean'])])
 lines += [mdtable(['数据集','FM','适配','n','性能指标','Δ性能','ΔECE15','ΔNLL','ΔBrier'],table),'',
 '分割12个实测MC−Off的ECE、NLL、Brier均下降。CloudSEN12的mIoU差约−0.000222至+0.000107；SpaceNet7约−0.0000163至+0.0000158，建筑IoU仍有小幅升降。这些是观察到的小差值，没有预定义等效界限及适当组抽样区间，不能把它们写成“已证明无实际损失”。更不能把MC的30次forward当30个独立训练重复。','',
 '分类的推理收益不一致。EuroSAT DOFA frozen的三个seed，MC−Off ECE分别约+0.004423、+0.004027、−0.000793，Brier都上升，accuracy有升、有降、有不变。TreeSatAI DOFA frozen的Macro-F1下降约0.001063、ECE上升约0.000683；Panopticon full三个seed的ECE都下降，但Brier都上升。NLL/Brier与ECE冲突应分别报告，不能挑选有利指标后称“概率质量全部改善”。','',
 'ECE分箱敏感性见 [24个MC−Off的10/15/30箱对照](mc_dropout_ece_sensitivity.csv)。三个分类对照的差值方向随分箱改变：EuroSAT DOFA frozen seed44、EuroSAT DOFA full seed42、TreeSatAI DOFA full seed42。主指标仍固定ECE15，不依据结果改选分箱；其余21个对照在这三种分箱下方向一致，但这不构成总体校准或跨训练稳定性的证明。','',
 'CloudSEN12 DOFA frozen seed42尤其说明对照的必要性：总方案MC−D的ECE约+0.0138，而固定权重MC−Off约−0.0048；Off−D约+0.0186。该分解表明总方案与推理改动可呈相反方向，不支持将总方案退化全部解释为随机推理本身。','',
 '本轮对SpaceNet7 Panopticon full seed42发现导出数值边界：GPU累积混淆矩阵与CPU softmax导出的预测在一个近平局像素上不同。已从保存logits以CPU路径重算正式指标并独立核验，原GPU指标与completion原样保留；具体像素与数值差见 [数值对齐记录](metric_alignment_resolution.md)。没有重新训练、重新选择checkpoint或改变标签。','',
 '因此，RQ3在已声明范围内得到条件性答案：TS有严格不变的分类决策；ensemble存在任务依赖的收益与代价；MC在分割固定权重比较中观察到校准/概率质量改善与微小性能变化，在分类中没有一致改善。无改善和方向不稳定都是有效结果；不需要为取得正结果增加训练。','',
 '## 仍然未知的证据与后续探索','',
 'DOFA-EuroSAT历史归一化统计的计算数据范围和六个训练run的训练时代码绑定仍无法恢复，C01/C06保留UNKNOWN；更早代码中出现同常数只证明历史载体，不能证明train-only统计或训练版本。当前结果可描述保存预测与整体配置差异，不能声称完全排除了历史预处理泄漏，也不能把差异隔离为纯架构因果效应。若要作这些更强主张，需恢复当时来源或建立预先固定的新对照；这不是为了改变当前方法效果而补训练。','',
 '外部完整论文尚未提供，本轮验收仅覆盖此结果正文和发布包。额外seed、按AOI/地点组配对bootstrap及预定义等效界限、纯冻结开关/纯dropout正则的因果实验、OOD/corruption、CKA或机制解释均属于后续探索，不作为本轮三问的必要补救。','',
 '## 关键诊断图','',
 '![分割类概率校准](../core_rq_audit_20260918/spacenet7_class_probability_reliability.png)','',
 'SpaceNet7图使用全部有效像素上的类概率与类别事件；counts、seed和bins在图/CSV中保留。CloudSEN12与TreeSatAI逐标签图及所有seed的类指标见 [原始类诊断](../core_rq_audit_20260918/segmentation_class_diagnostics.csv)、[分类类诊断](../core_rq_audit_20260918/class_diagnostics.csv) 和 [发布包](published/final_thesis_results.md)。']
 (OUT/'RESULTS_SECTION.md').write_text('\n'.join(lines)+'\n')
 checks=readcsv(OLD/'checks.csv');changes={
 'C05':('PASS','报告主图与表采用同一decision-ECE及右闭分箱，带counts/seed；独立重算补评估并解决导出近平局像素，原生clip语义保留。','published/tables/package_validation.json;segmentation_off_verification.json;metric_alignment_resolution.md'),
 'C09':('PASS','现行TS仅EuroSAT，12 NA blank；历史TreeSat TS结果完整单列并公开8-24至8-27时序；异常中断与数值对齐均留痕。','published/tables/method_applicability.csv;published/tables/historical_treesatai_temperature_scaling.csv;SUPPLEMENT_PROTOCOL.md'),
 'R1.6':('PASS','本轮正式结果正文逐任务/适配描述经验校准、过欠自信与类诊断，保留不稳定排序及历史来源限制；外部完整论文不在已审范围。','RESULTS_SECTION.md;provenance_numeric_crosscheck.json;published/final_thesis_results.md'),
 'R3.1':('PASS','24个MC训练checkpoint均有同权重Off，与D及MC全ID/标签/mask配对；推理、配方轨迹、总方案三个差值分开。','mc_dropout_three_way.csv;segmentation_off_verification.json;../core_rq_audit_20260918/classification_dropout_off_verification.json'),
 'R3.4':('PASS','12分割补评估全部dropout-off、严格权重加载、BN不变、no_grad、全test；同MC checkpoint哈希相同，独立重算通过。分类12Off已有直接features+head复现。','mc_dropout_control_coverage.csv;segmentation_off_verification.json;segmentation_off_verification_summary.json'),
 'R3.6':('PASS','24行三方绝对指标/72行差值完整，同时报告ECE、NLL/Brier、主性能及建筑IoU；保留训练epoch与D跨seed范围。','mc_dropout_three_way.csv;mc_dropout_effects.csv;RESULTS_SECTION.md'),
 'R3.8':('PASS','正文完整保留TS/ensemble/MC反例与指标冲突；分类MC推理不一致、分割微小代价与总方案不同方向均披露。','RESULTS_SECTION.md;mc_dropout_effects.csv;published/tables/historical_treesatai_temperature_scaling.csv'),
 'R3.9':('PASS','按实际Δ性能与ΔECE描述收益/代价及数值量级；只有TSargmax不变性支持严格决策保持，其余不声称等效无损。','mc_dropout_effect_summaries.csv;RESULTS_SECTION.md')}
 for r in checks:
  r['previous_status']=r['status'];r['review_scope']='本轮结果正文/发布包；不是未提供的外部论文全文'
  r['code_revision']+='; completion evaluation source hashes in supplemental_artifacts.csv / completion_manifest.json'
  if r['check_id'] in changes:
   status,finding,ev=changes[r['check_id']];r.update(status=status,finding=finding,evidence_path=r['evidence_path']+';'+rootpaths(ev),acceptance_evidence=rootpaths(ev),minimal_resolution='已完成；无需新增训练。',impact_on_answer='支持已声明配置与测试集范围的结论，重复/来源限制见正文。')
  if r['check_id'] in ('C01','C06'):
   r['finding']+=' 更早backup代码载有常数，但无统计推导/训练run绑定，补查后仍UNKNOWN。';r['evidence_path']+=';'+rootpaths('provenance_followup.md');r['minimal_resolution']='若需完全历史复现/train-only来源证明，恢复原始统计与训练代码；本轮明确保留UNKNOWN，不默认补训练。'
  if r['check_id'] in ('R1.3','R1.4'):
   r['evidence_path']+=';'+rootpaths('published/tables/package_validation.json;RESULTS_SECTION.md');r['minimal_resolution']='已发布有counts/seed的诊断；无必要补训练。'
  r['owner']='Codex实施/直接指标核验；Claude Code独立对照与范围复核'
 writecsv('checks.csv',checks);(OUT/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
 issues=readcsv(OLD/'issue_register.csv')
 for r in issues:
  r['previous_status']=r['status']
  if r['issue_id'] in ('I02','I03','I04','I05'):
   r['status']='PASS';r['resolution']={'I02':'已修复生成器并新发布；64表项及52图曲线逐项核对master，图含counts/seed/语义。','I03':'核心TS与master统一，历史12TreeSat结果完整单列并公开时序。','I04':'新增12分割全test Off并独立重算；与既有12分类组成24组三方对照。','I05':'新增RESULTS_SECTION.md已覆盖三问的条件、反例与限制；外部完整论文未提供，未声称已审全文。'}[r['issue_id']]
   r['acceptance']={'I02':'published/tables/package_validation.json','I03':'published/tables/method_applicability.csv;published/tables/historical_treesatai_temperature_scaling.csv','I04':'segmentation_off_verification_summary.json;mc_dropout_three_way.csv','I05':'RESULTS_SECTION.md;provenance_numeric_crosscheck.json'}[r['issue_id']]
  if r['issue_id']=='I06':
   r['resolution']+=' 本轮Claude替代clip公式的0.4135与正式0.43245差异已直接复现，系下界及clip位置不同；REVIEW_RESOLUTION.md明确更正，并非新的指标故障。';r['acceptance']+=';reviewer_foreground_clipping_probe.json'
  if r['issue_id']=='I01':r['resolution']='已完成有限来源恢复和明确范围限定；统计来源/训练代码绑定仍UNKNOWN，见provenance_followup.md。'
 issues.append({'issue_id':'I07','check_ids':'C05;C06','rq':'RQ1;RQ2;RQ3','status':'PASS','previous_status':'FAIL','finding':'新补评估SpacePanfull42 GPU累积指标与CPU导出概率在近平局像素的argmax不同。','resolution':'保留原始GPU记录；从保存logits重算export-aligned指标和逐图表，独立验证新confusion exact。','action_type':'已有预测重算/数值对齐','new_training':'否','acceptance':'metric_alignment_resolution.md;segmentation_off_pre_alignment_verification.json;segmentation_off_verification.json','decision':'最小数值修复，无需补推理/训练；不改标签/权重/选择规则。'})
 writecsv('issue_register.csv',issues)
 rqrows=[]
 for rq,answer in [('RQ1','不同FM的经验校准排序依任务/适配/指标而变；低总体ECE不能覆盖类概率和建筑表现。'),('RQ2','完整适配配方的性能与校准变化因条件而异，方向不稳定本身是有效结果。'),('RQ3','TS保持决策，ensemble有任务依赖收益与代价；分割MC-Off改善ECE/NLL/Brier但性能小幅波动，分类收益不一致。')]:
  rqrows.append(dict(rq=rq,status='ANSWERED',scope='既定FM/benchmark/适配配置及保存测试预测的描述性答案；历史来源UNKNOWN和重复层级限制保留',current_answer=answer,supported_evidence='RESULTS_SECTION.md;checks.csv;mc_dropout_three_way.csv;paired_effect_summaries.csv',missing_evidence='DOFA-EuroSAT统计来源与六run训练代码绑定；外部论文全文未审；不支持全EO/纯因果/等效无损主张',repair_code='已修报告生成器',recompute_metrics='已重算并对齐补评估指标',extra_evaluation='已完成12分割MC Off' if rq=='RQ3' else '无必要新增',extra_training='不需要；未启动',future_exploration='额外重复/组CI/因果隔离/OOD机制；非核心补救'))
 writecsv('rq_evidence.csv',rqrows)
 counts=collections.Counter(r['status'] for r in checks)
 compact=[]
 for r in checks:
  ev=changes.get(r['check_id'],('', '', '../core_rq_audit_20260918/checks.csv'))[2]
  if r['check_id'] in ('C01','C06'):ev='provenance_followup.md'
  compact.append([r['check_id'],r['status'],r['rq'],r['finding'],ev,'保留历史来源UNKNOWN；更强主张需恢复证据' if r['status']=='UNKNOWN' else '已完成；无需训练'])
 summary=['# 审计补充完成记录','',f"29项：**{counts['PASS']} PASS、{counts['FAIL']} FAIL、{counts['UNKNOWN']} UNKNOWN、{counts['NA']} NA**。两个UNKNOWN均为历史来源问题，没有把它们自动升级为通过。实际方法覆盖仍为64个cell：52个有核心结果、12个TS NA；另完成24个MC同权重Off对照。NA属于矩阵cell，不与29条共用检查状态混淆。",'',
 '本轮启动于2026-09-21、完成于2026-09-23。没有新增训练、修改checkpoint或重新挑选结果。新旧原始产物均保留。','',
 '## 阅读顺序','',
 '1. [三个RQ结果正文](RESULTS_SECTION.md)：当前能回答什么、反例及结论边界。\n2. [重建正式图表](published/final_thesis_results.md)：统一TS范围与ECE语义，曲线下方附counts。\n3. [24组三方绝对指标和差值](mc_dropout_three_way.csv)、[逐seed效应](mc_dropout_effects.csv)、[48块摘要](mc_dropout_effect_summaries.csv)。\n4. [29项机器可读检查](checks.csv)、[唯一问题清单](issue_register.csv)、[核心覆盖矩阵](coverage_matrix.csv)、[24个对照覆盖](mc_dropout_control_coverage.csv)。','',
 '## 完成了什么','',
 '- 补齐12分割MC checkpoint全test dropout-off预测，CloudSEN12每模型975图、SpaceNet7每模型1152图，共12,762图次；加既有12分类Off形成完整24组三方对照。\n- 从保存预测独立重算全部新结果，并直接匹配D/MC的IDs、标签、ignore mask和checkpoint hash。\n- 修复C6 TreeSat decision-ECE、右闭分箱、counts/seed标注；64表项与52主图曲线逐项核对master，NA指标为空白。\n- 保留历史TreeSat TS全部12结果和时间顺序；将其与现行核心NA明确区分。\n- 对齐一个近平局像素引起的GPU累计/CPU导出混淆矩阵差，保留原始差异和修复记录。\n- 完成新的结果正文，未声称审过缺失的外部论文全文。','',
 '## 三问的回答与后续动作','',mdtable(['RQ','状态及当前答案','仍缺的必要来源/更强主张证据','修代码/重算/补评估/补训练'],[[r['rq'],r['status']+'（声明范围）—'+r['current_answer'],r['missing_evidence'],r['repair_code']+'；'+r['recompute_metrics']+'；'+r['extra_evaluation']+'；'+r['extra_training']] for r in rqrows]),'',
 'ANSWERED指声明范围内的描述性研究答案已给出，不表示所有历史可追溯检查都PASS。DOFA-EuroSAT的train-only统计来源和训练代码绑定若是外部验收的强制要求，相关来源条款仍未满足；此处明确保留UNKNOWN，不能宣称完全排除历史预处理泄漏。其余无已确认必须追加的核心实验。','',
 '## 直接证据与独立复核','',
 f"[独立全量补评估核验](segmentation_off_verification_summary.json)：12/12通过；原始标量指标与NumPy重算最大绝对差 {verification['maximum_absolute_metric_difference']:.3g}。[37项原有相关测试](targeted_tests.log)通过；生成器专项测试见 reporting_fix_notes.md。原2394文件中仅本轮授权C6生成器变化，原104审计文件均未变（baseline_integrity.json / previous_audit_integrity.json）。",'',
 'Claude Code由实际CLI调用，先独立检查runner/训练配置并抽算分类与分割，再复核最终固定产物。报告和逐工具证据见 [设计复核](claude_design_review.md)、[最终交叉复核](claude_final_review.md)、claude_design_evidence_trace.jsonl、claude_final_evidence_trace.jsonl。每个PASS由代码、原始预测或可复算表支持，未仅依赖另一agent的结论。评估代码hash与训练时代码分开保存。审查链接补齐及独立复核数值措辞的更正见 [复核处理记录](REVIEW_RESOLUTION.md)，Claude原文完整保留。','',
 '## 29项逐项状态','',mdtable(['ID','状态','影响RQ','发现/验收','证据（相对此目录；原始运行见对应CSV）','最小补救'],compact),'',
 '## 重现本轮补充','',
 '使用现有环境，在/workspace运行：','',
 '```sh\npython -m scripts.complete_segmentation_dropout_off --device cuda:0 --shard 0\npython -m scripts.complete_segmentation_dropout_off --device cuda:1 --shard 1\npython reports/core_rq_completion_20260921/align_export_metrics.py\npython reports/core_rq_completion_20260921/verify_segmentation_off.py --require-complete\npython -m scripts.build_core_rq_completion\npython reports/core_rq_completion_20260921/compute_off_sensitivity.py\npython reports/core_rq_completion_20260921/write_completion_report.py\n```','',
 '两个GPU分片可同时启动；已完成run会核验hash后跳过；不完整目录会拒绝覆盖，首次中断产物保存在interrupted_before_completion。正式C6重建命令为 `python -m scripts.c6_build_final_results --output-root <新的空目录>`，当前发布位于published。现有文件的SHA256与来源清单见 completion_manifest.json。','',
 '## 单列后续探索','',
 '额外训练重复、按AOI/地点组配对区间与预定义性能容忍界限、纯架构/冻结开关/dropout正则因果控制、OOD/corruption与机制解释均需单独确定范围。本轮不以方法无改善、性能权衡或适配效应不稳定为由要求追加训练。']
 (OUT/'COMPLETION_REPORT.md').write_text('\n'.join(summary)+'\n')
 (ROOT/'reports/CORE_RQ_CURRENT.md').write_text('# 当前核心RQ结果入口\n\n当前版本为 [2026-09-21启动、2026-09-23完成的补充报告](core_rq_completion_20260921/COMPLETION_REPORT.md)。\n\n- [三问结果正文](core_rq_completion_20260921/RESULTS_SECTION.md)\n- [重建图表发布包](core_rq_completion_20260921/published/final_thesis_results.md)\n- [29项状态](core_rq_completion_20260921/checks.csv)\n- [MC同权重三方对照](core_rq_completion_20260921/mc_dropout_three_way.csv)\n\n旧 `reports/final_thesis_results.md` 与 `final_thesis_tables/` 为2026-08历史发布产物，保留以审计，不作为现行TS适用范围入口。权威核心矩阵仍为 `thesis_master_results.csv`；本次补充Off对照另表保留。\n')
 print('Wrote completion report, results narrative, 29 checks, issue register and current entry point.',dict(counts))
if __name__=='__main__':main()
