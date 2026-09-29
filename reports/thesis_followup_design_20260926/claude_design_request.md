请作为实际 Claude Code 独立评审者，系统审查后续实验设计、相对既有项目的工作量，以及可交给执行agent的完整prompt应包含什么。当前用户只请求设计/估量/prompt，不授权启动新实验；只读配置/表/metadata，不训练、不新推理、不新算研究指标、不读凭据。直接输出中文报告，协调者保存。

用户目标：遥感硕士，已完成现行3RQ，时间有限，想深化错误识别、尺度/代理量、模型机制和AU/EU耦合；现请求“要回答后续问题，如何设计后续实验，大概实验量级与前面比如何？和claude系统评估后给出完整prompt”。
已完成的上一轮独立复核及路线：
reports/thesis_dossier_assessment_20260926/DOSSIER_ASSESSMENT_AND_EXPERIMENT_ROADMAP.md
reports/thesis_dossier_assessment_20260926/REVIEW_RESOLUTION.md
reports/thesis_dossier_assessment_20260926/evidence_inventory.json
reports/thesis_dossier_assessment_20260926/coverage_matrix_review_view.csv
reports/core_rq_completion_20260921/EO_UQ_Thesis_Conversation_Dossier_for_Codex.md（只读与新增实验有关的§5/20–22/47，需要时定位，不必再通读）
直接表：reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv
research_data/manifest.parquet，reports/mc_dropout_protocol.md，reports/final_training_protocol.md。

请优先审查以下暂定设计，但独立推翻不合理之处：
A默认推荐包：分类8cells全部既有seed（D24/TS12/MC12/Off12/DE8=68个预测对象，不是68新run），完成error detection+RC+同权重TS/中心化logit及Shannon/Gini代理量；seg主seed42八cells可补D/Off/MC/DE全测试mean概率，MI成员比较先统一固定32subset或由原D成员全概率离线计算。新训练0、新FM前向0；cost主要CPU排序/分组bootstrap/磁盘解压，不能口头说几分钟。
B可选已存pooled CKA：18分类embeddings=9 frozen/full pairs，seg24global pooled=12pairs；补齐缺的D-DOFAEuro6份需额外feature前向，不能MC代替。
C可选corruption最小：EuroSAT两FM两adapt seed42，clean+单扰动两强度；D-only新增8个checkpoint×全test评估（不是8次单样本forward）。若加MC/DE：每强度D成员3+MC1及MC Off，MC T30；现有Dseed42属于DE成员可复用；TS直接变换已有D logits。请给严格去重计数和相对既有test评估粗量，注意冻结分类可cachefeature而full可能也在推理缓存（dropout只head），不能把T写成30×backbonewalltime。
D模块2x2训练：1任务，2FM×(输入模块开关×Transformer开关)×3seed=24训练位置。严格新统一配方下旧frozen/full未必可复用，计划12–24新增；1FM缩小版6–12新增。3seed是描述性起点不是普适检验力保证。请核实真实模块区分能否实现、如何避免学习率/预算混杂（无需修改代码）。
E AU/EU：推荐先独立廉价可解析toy：二分类Beta(1,1)先验，p真值代表歧义，n代表训练支持，明确oracle H(p)与posterior积分代理是不同对象；二水平p×二水平n，20配对数据重复=80解析位置而非80FM训练。建议p0=.1,p1=.4,n0=20,n1=200，标签U随机数耦合，nested n。可算H(meanBeta)、E_H、MI及Brier/Gini解析式，不偷换“true EU”。非零交互并不推翻恒等式/识别物理源。EO版一个冻结FM两数据支持水平×3seed=6headrun位置，训练抽样与优化seed仍混合；若要分开建议3subset draws×3optimization seeds，100%subset相同不可重复计，唯一训练位置需去重（3 full+9 low=12，而不是18）；每模型clean+2强度。请检查这个设计和最小必要性。

协调者直接主表算得：D48训练run-seconds求和45.9493h，MC24=18.9109h，共72唯一run=64.8603累积run-wall-hours；DE复用D不再重复加48，TS非神经训练。它不是GPU-hours、也不是项目历时，硬件/并行/训练预算不同不能用均值机械外推。你请独立轻量核算并给每个包对72run的比例、可复用条件、成本proxy与未知。

需要你指出：设计里的伪重复、double count、test选择、任意统计门槛、无法支撑的机制主张、可能导致执行agent无限扩scope的prompt陷阱。尤其：AUROC跨不同模型不脱离error rate解释；AURC/eAURC非万用修正；Tree多标签不删稀有类；seg AOI/ROI组/伪独立；T30/M3可识别内容有限；旧test的新分析标探索。要求明确的验收但不能强制改善。

输出：1推荐默认执行到哪；2各包实验设计+准确数量+相对基准；3完整执行prompt必须具备的参数/交付/停止条件；4是否需要用户回答新问题或可按默认继续准备（本轮不执行）。无需保证学校验收，无需给未测量的GPU小时/日历时间。
