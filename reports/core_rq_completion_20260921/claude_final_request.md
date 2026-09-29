你是实际 Claude Code 独立研究审查者。请在 /workspace 对本轮已完成补充做最终独立复核。只读文件和数值脚本检查，不训练、不改文件、不联网、不读凭据。输出中文审查，控制在约2000字，附具体文件/数字。不是只认可Codex总结：至少直接读取一份新增SpaceNet7 Off NPZ及同MC/D数据，并直接核对正式表图分箱CSV/NA条目与正文。

标准为 EO_FM_UQ_Core_RQ_Audit_Standard.md，涉及 C05/C09/R1.6/R3.1/R3.4 关闭；历史 C01/C06继续UNKNOWN。上一轮清单 reports/core_rq_audit_20260918/issue_register.csv。你此前设计复核及直接工具证据在 reports/core_rq_completion_20260921/claude_design_review.md 和 claude_design_evidence_trace.jsonl，仅作线索，不能代替独立判断。

最终待审目录 reports/core_rq_completion_20260921/：
- COMPLETION_REPORT.md、RESULTS_SECTION.md：新的本轮正文，外部完整论文未提供。29项 checks.csv 当前应27PASS/2UNKNOWN，RQ均限定配置/测试集描述性ANSWERED，不能读成全历史来源通过。
- published/final_thesis_results.md、tables/package_validation.json、tables/{classification_results,segmentation_results,method_applicability,historical_treesatai_temperature_scaling}.csv、figures/*reliability*.csv/png：核心TS仅Euro，TreeSat/seg共12NA，历史TreeSat12条保留并公开8-24已有结果→8-25排除→8-26旧C6矛盾→8-27master的时序。Tree主ECE用flattened decision，positive-label另列；主图count/seed右闭bins与表一致。
- mc_dropout_three_way.csv 24行，mc_dropout_effects.csv 72行，mc_dropout_effect_summaries.csv 48块。D原p0训练、Off同MC权重关闭dropout、MC原30pass概率均值。MC-Off固定权重推理，Off-D含配方及轨迹，MC-D总方案。已附D/MC best/last epoch及D seed范围。主矩阵seed42；只有既定4格三seed，不得将MCpass当训练重复。
- segmentation_off_verification.json/summary：独立NumPy全12复算，D/Off/MC全ID/label/mask对应及hash。
- 新Off原始NPZ results/final_thesis/core_rq_completion_20260921/segmentation/{dataset}/{model}/{adaptation}/seed*/，含logits/probs/labels/mask，所有checkpoint皆现有MC权重。12全部已完成，无训练。
- 一项真实的数值边界已最小修复：SpaceNet7 Panopticon full42，GPU累积vsCPU导出在sample_5025 row79 col123的softmax近平局，一个像素argmax不同。原metrics/completion保留；metrics_export_aligned.json、per_image_metrics_export_aligned.csv为正式版本，metric_alignment.json含全部hash。完整CPU/CUDA探针 segmentation_off_softmax_alignment_probe.json，metric_alignment_resolution.md。独立验证没有放宽confusion exact。请检查表是否使用新版本及“不需训练”理由。
- provenance_followup.md、provenance_numeric_crosscheck.json保留历史UNKNOWN，24分类直接重算/24分割confusion核对/48箱表。完整源hash见completion各manifest，不把评估代码hash冒充训练时代码。

最终需判断：1实际对照和指标是否充分支持三問当前范围；2是否过度声称无损/稳定/因果；3所有4FAIL是否关闭、2UNKNOWN是否诚实；4是否还有真正必须修代码、重算、补评估或补训练事项；5你的独立抽算结果。方法无改善/适配差异不稳定均可为有效结论，后续探索不自动升级核心缺口。请给接受/拒绝具体证据，不因仍未知历史来源一概否定其他已验证结论，也不将UNKNOWN漂白。
