进入交叉核对阶段。你的第一轮已保存为 claude_independent_review.md。Codex 在读取前已保存 codex_independent_notes.md。请只针对以下实质争议核对原始证据并给接受/修正/保留争议及证据，不重写全29项。
1. C06 你的全局PASS是否过强：reports/dofa_eurosat_final_manifest.json.source_snapshot.provenance_scope 明确只是 post-hoc source；六个训练 environment.json.git_commit=null且无训练时 code_snapshot。Codex 倾向该子范围 UNKNOWN，数值可重算为PASS，绝不称已知泄漏或要求重训。C01 旧EuroSAT归一化统计来源不明也只能UNKNOWN该子范围。
2. R1.2 主证据入口 thesis_evidence_matrix.md 和 master.metric_semantics 已明确 strict exact match，不能因旧C6列名就断言任务指标缺失。Codex 倾向 R1.2 PASS，label probability/逐类校准不足归R1.4、图表定义不一致归C05/R1.3，避免重复升级。
3. TreeSatAI TS现行冻结明确排除，但发生在8-24历史TS结果之后；接受C09范围/时间线披露问题，不能凭时序证明主观cherry-picking。标准允许validation复用，故排除理由不能写成数学无效。Codex不会擅自重新纳入；最小补救是统一现行范围、显式保留历史诊断及反例，并披露决定时序。是否同意无需新训练、也不必强制恢复为核心结果？不要为分割TS发明协议未写的NA原因。
4. Codex已直接读取72个训练checkpoint逐tensor比对预训练状态/哈希/最早val选epoch，全部通过(checkpoint_verification.json)。56分类结果原始预测独立重算max误差约2.3e-9；分割44份全量重算仍在进行。aggregation_verification.json已直接核对16 ensemble（分类全数据/分割固定前8图）及12 MC分割预定subset首图30pass平均。不要仅凭这些总结PASS，必要时查相应原始artifact。
5. 你已经展示两个分类 MC dropout-off 重算数字；Codex正在对全部12个分类MC checkpoint复现CPU embeddings+head。分类缺口将由审查新增证据补齐；12分割仍需单次deterministic eval。不需要训练。请确认同权重MCvsOff、Offvs原始p0、MCvs原始p0三种差值的解释。
结论预计：原项目RQ1 PARTIAL（图表类条件/正文）；本轮补充图表后能提供范围内回答草稿；RQ2 ANSWERED（方案差异非纯因果）；RQ3 PARTIAL（分割同权重MC对照缺）。请指出需要更改的具体判断。
