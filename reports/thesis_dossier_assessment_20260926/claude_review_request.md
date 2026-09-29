请作为实际 Claude Code 独立审阅者，读取用户刚提供的研究思路档案，结合直接实验产物，判断目前可写到什么程度，以及每一层更强主张必须新增什么实验。当前任务只做评估与路线设计，不执行新训练/推理/评估，不修改原产物、不联网、不读取凭据。不要仅引用另一 agent 总结作为完成证明。

用户：遥感硕士，题目 Uncertainty for foundation models of earth observation，没有承诺新方法，目前无明确其他硬性要求，时间有限。但曾希望超越metrics-only，思考uncertainty/error、机制、AU/EU耦合等。最新请求：结合 EO_UQ_Thesis_Conversation_Dossier_for_Codex.md，与Claude协同报告现有实验能将论文完成到什么程度，后续每一步完善需哪些必要实验。

先独立阅读档案（2812行，可分段，不能把历史助手建议当导师现行要求）：
reports/core_rq_completion_20260921/EO_UQ_Thesis_Conversation_Dossier_for_Codex.md
尤其§2–5、18–24、29、35、40–48。档案是对话重建，不是完成证据，也不是学校正式规定。§18/19/21/22有历史愿景，§35/43/45约束核心范围。

直接证据：
reports/core_rq_completion_20260921/coverage_matrix.csv
reports/core_rq_completion_20260921/mc_dropout_three_way.csv
reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv
reports/core_rq_completion_20260921/RESULTS_SECTION.md
reports/core_rq_completion_20260921/checks.csv
reports/core_rq_completion_20260921/provenance_followup.md
reports/core_rq_completion_20260921/REVIEW_RESOLUTION.md
reports/mc_dropout_protocol.md
research_data/README.md
research_data/manifest.json（若实际文件名不同，先Glob查找；不要加载大张量）

先形成自己的结论，再可参看上一轮协调结果 reports/thesis_readiness_20260926/REVIEW_RESOLUTION.md。注意不要重复此前被撤回的过度解释：代理MI降低同时错误变多不证明AU/EU定义错误；高相关/相似AUROC不证明无独立信息；数学交互项非零不证明两个物理来源耦合；CKA相关不是因果；三方Off-D含训练轨迹变化。

请输出中文、带路径的审阅报告，回答：
1. 档案使上一轮“可直接写实证论文”的判断哪些需要加强/收窄，是否有被遗漏的明确承诺？不要凭档案写“所有A-C完成”。
2. 逐层区分：核心3RQ实证论文；加入行为效用/错误识别的更完整论文；解释性机制分析；AU/EU干预交互；新理论/方法。每层已有/未知、证据门槛，是否应成为当前完成门槛。
3. 每个推荐步骤明确“无需计算/已有预测重分析/新推理/新训练”的类型、最小控制与矩阵、指标和统计单位、允许主张、停止条件与不建议的范围扩张。不能只说做AURC、CKA、corruption；需要具体设计和可能失败的边界。无需估计未经测量的GPU小时。
4. 重点：行为指标以已有预测可做；不同预测错误率下要考虑比较基线；TreeSat多标签定义；seg空间单位；OOD标签保持问题；低SpaceNet/Tree任务性能；旧test已被查看不能把后续探索说成预注册验证；bootstrap不创造跨训练seed稳定性；AU/EU两个操纵因子是否真可隔离。
5. 给出你建议的最终止步点；哪些补充是“为某个新增结论必要”而非“毕业必要”；是否有当前证据要求立刻修代码/重算/重训。

可用状态PASS/FAIL/UNKNOWN/NA，但必须给范围，不能用PASS保证学位验收。简洁但设计可执行。将报告直接输出终端，由协调者保存；不要自行创建或修改文件。
