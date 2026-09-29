请继续同一实际 Claude Code 独立审计会话，完成本轮 A/E 最终复核。用户只授权 A/E，禁止训练、模型前向、补 seed 或开启其他分支。目录 /workspace/reports/thesis_followup_execution_20260926。你只读项目与结果，可用 CPU 小范围直接重算；不修改协调者报告/代码。请先独立记录问题及直接证据，再给判断，不能因协调者宣称完成而 PASS。

数值计算现在 100/100 完成。请读 FOLLOWUP_COMPLETION_REPORT.md、THESIS_RESULTS_DRAFT.md、checks_delta.csv、analysis_applicability.csv，核对结论范围、统计单位、NA 和图表/数据一致性。之前抽查已验证 DE/数学/E；这轮重点补齐你上次提出的：

1. 从 inputs_manifest.csv 与 auxiliary_inputs_manifest.csv 找一个分类 MC（可用 TreeSat，验证 Bernoulli 求和/均值约定），直接读 raw probabilities 与保存 summary，独立核对平均概率、EE/MI、ddof=0 variance，及一项 MSP/MI 错误识别指标；不要只读协调者 verification 表。
2. 选一个分割 MC，在固定 32 图中的一图，直接 raw draw vs full-test summary 核对 variance 类求和、均值、EE/MI、有效 mask；必要时用自己的 zip 单数组解压读取控制内存。可以用协调者 LazyNPZ 做 I/O，但独立写测量算式，不调用其 decomposition/RankPlan 当独立算法。
3. 一个 Off 对象，直接保存概率/标签核对原审计三方表 task/probability metric，与本轮结果一致；核查 MC/Off 同 checkpoint 来源。
4. SpaceNet7 “MSP 接纳50%图像只保留11.77%–18.82%总真值建筑物像素”的至少一个对象，独立从逐图标签类支持/分数核对接纳质量与分子/分母，并确认它是内容保留量而非建筑物recall。若需要，直接源 labels/mask重算其中一图support。报告风险为图像等权的有效像素错误比例，不应写成全局像素加权风险。
5. 最终范围：分割新增DE RC只对D42；相对三D成员均值用既有核心指标。固定32像素配对改为双方共同有效图。逐项探索性区间不等于校正后全家族主张。C01/C06仍UNKNOWN。E是生成条件熵与有限后验代理差异，不是新理论/物理来源耦合证据。严格区分需要修代码/重算和仅加强表述。

已有输入对齐100、核心标量436通过；这些数字仍要抽查源数据和计数，不可直接当通过依据。独立脚本/临时文件若需落盘，请仅放本轮目录 claude_validation/；保留可复核算式与数字。输出中文最终复核原文，明确实际抽查范围、发现/状态/证据路径/影响/最小补救，指出是否存在必须重算或补实验的阻断问题。公开工具记录会另外从会话导出，不需要披露内部思考过程。
