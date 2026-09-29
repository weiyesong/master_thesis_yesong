# 可直接复制给 Codex 的执行 prompt

下文是完整任务指令。默认只执行 A；若要执行其他分支，先改“本次范围”中的开关。关闭的分支用于保留设计，不构成执行授权。本文件形成于 2026-09-26，设计阶段未运行这些新实验。

---

你是当前 EO foundation model 不确定性硕士论文项目的实现与证据负责人。请与**实际 Claude Code** 协作，执行下面选定的补充研究，直到交付可复核结果及论文可用的结果说明。不要只给计划，也不要把另一位 agent 的判断当作通过证据。

## 0. 本次范围、角色与完成标准

```yaml
enabled_packages: [A]
A_segmentation_extra_seeds: false
A_segmentation_full_test_DE_decomposition: false
B_fill_missing_embeddings: false
C_variant: D_only             # 仅 C 开启时有效；另可选 all_methods
D_models: [dofa, panopticon]   # 仅 D 开启时有效
D_dataset: eurosat
E_sampling_illustration: false
F_variant: crossed            # 仅 F 开启时有效；另可选 joint_pilot
F_model: panopticon
F_dataset: eurosat
F_adaptation: frozen
F_uq_training: mc_dropout
timing_benchmark: false
bootstrap_replicates: 1000
analysis_seed: 20260926
coverage_points: [1.0, 0.9, 0.8, 0.5]
new_outputs: reports/thesis_followup_execution_<UTC日期>/
```

如果用户在同一请求中明确修改这些参数，以用户指令为准；不要因为下文描述了某分支就自动开启。A、B 的既有特征部分、E 不需要新模型前向或训练。默认 A 做完即收束，不为获得改善而扩展指标、训练 seed、任务或扰动。

研究定位：遥感硕士，题目为 *Uncertainty for Foundation Models of Earth Observation*；没有已知的硬性新方法要求，时间有限。现行三个 RQ 是适配后校准、适配配方的差异、UQ 方法的校准与任务性能代价。后续分析旨在增加行为与解释证据。没有改善、差异不稳定、代理量不一致都可以是有效结果。

分工：你负责实现、实际覆盖矩阵、产物来源、指标、实验记录与计算成本。Claude Code 独立阅读必要原始配置/代码/产物，检查对照、统计单位、测量含义、证据充分性和结论范围。双方先分别保存发现，再交叉核对；有实质争议则做最小验证，或保留争议。没有新证据，不循环互相改写意见。使用真实 Claude CLI/已配置连接，保存其公开结果、session 标识与公开工具调用记录；不得将另一个 Codex agent 称为 Claude。若 Claude 不可用，完成可独立完成的已授权工作并如实标记独立复核未完成，不假称双审通过。

## 1. 先读权威材料，再建立本轮清单

工作目录 `/workspace`，优先读取：

1. `EO_FM_UQ_Core_RQ_Audit_Standard.md`（29项标准及结论边界）。
2. `reports/core_rq_completion_20260921/COMPLETION_REPORT.md`、`RESULTS_SECTION.md`、`checks.csv`、`mc_dropout_three_way.csv`、`provenance_followup.md`。
3. `reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv`。
4. `reports/thesis_dossier_assessment_20260926/DOSSIER_ASSESSMENT_AND_EXPERIMENT_ROADMAP.md`、`REVIEW_RESOLUTION.md`、`coverage_matrix_review_view.csv`、`evidence_inventory.json`、`eurosat_calibration_export_check.json`。
5. `reports/core_rq_completion_20260921/EO_UQ_Thesis_Conversation_Dossier_for_Codex.md` 中最终范围、后续问题及已接受决策；历史设想不覆盖最终协议。
6. `reports/final_dataset_protocols.md`、`final_training_protocol.md`、`pre_uq_protocol_freeze.md`、`mc_dropout_protocol.md` 及相关任务协议。
7. `research_data/manifest.parquet`，`reports/mc_dropout_segmentation_research_subset_ids.json`。
8. 本设计的 `reports/thesis_followup_design_20260926/DESIGN_AND_WORKLOAD.md`、`REVIEW_RESOLUTION.md`、`workload_estimates.json`。

优先级：用户当前指令 > 已确认研究范围和现行协议 > 对应 run 的实际记录 > 历史讨论。发现冲突要记录并用直接证据解决，不静默选一个方便的版本。

保留以下事实与限制：

- 原项目是 48 次 D + 24 次 MC = 72 次唯一神经训练。16 个 DE 组复用 D；每组 3 个成员共享预训练起点、下游独立训练。现行主表 TS 是 12 个 EuroSAT 结果；历史 TreeSat TS 不能静默混入当前矩阵。
- 64.86 小时是已存训练 run 时长之和，不是项目历时或 GPU-hours。MC 推理、TS 拟合等耗时不完整。
- MC 仅 head/decoder dropout，p=0.1、T=30，验证选择在 dropout-off 下进行。MC−Off 是同 checkpoint 推理比较；Off−D 还包含训练配方与轨迹差异。
- D 全部配置 3 seed；MC 只有部分配置 3 seed。不得把 30 draws、3 ensemble 成员、15 标签、像素或同一区域 patch 当独立训练重复。
- 原覆盖表有16条过期的“缺MC Off”备注，已有24个有效Off对照，以更正阅读表和三方表为依据，不重做这些前向。
- DOFA–EuroSAT 历史代码/归一化来源仍有 C01/C06 UNKNOWN 边界；保存预测的有效性核验不恢复历史来源，也不自动证明泄漏。保留已有范围限定。
- 2026-08-25 的 UQ 协议冻结只适用于其后相应实验；本轮新增分析用已查看过的 test，必须标为探索性，不能追称预注册或独立确认。

建立 `inputs_manifest.csv`：每个分析对象列 dataset/model/adaptation/method/seed/member seeds、checkpoint/run、预测/标签/分组/特征路径、split、hash来源、样本数、适用分析、缺失原因。路径从真实 manifest 和三方表解析，不仅依赖命名猜测。封存报告和 `research_data` 只读；新代码/报告放独立位置，不复制整套归档。已有文件不覆写，不破坏工作区其他改动。

分类 Off 在 `reports/core_rq_audit_20260918/classification_dropout_off/`；分割 Off 的准确位置从三方表 `output_path` 读取。EuroSAT 12 个 calibration 导出在 `results/final_thesis/c3_temperature_scaling_and_ensembles/classification/calibration_exports/`。这些是存在性线索，执行时仍验证对象ID与用途。

## 2. A：默认补充包，零新训练、零新模型前向

### A1. 实际覆盖与配对

分类两数据集×两FM×两适配，共8格，处理全部已存有效 seed：D24、TS12、MC12、Off12、DE8，共 **68个预测对象**；唯一训练 checkpoint 是24D+12MC，不能把68写成训练次数。EuroSAT 40对象，TreeSatAI 28对象。缺失或不适用要保留状态和原因，不能用另一训练轨迹补位。

分割 CloudSEN12/SpaceNet7 的两FM×两适配，共8格：主分析统一seed42的D、Off、MC及其对应三成员DE，共 **32个平均预测对象**。DE没有“独立ensemble seed42”这一重复含义。主seed42范围在看新增结果前固定。若显式开启额外seed开关，只加入已存在的两格MC/Off的43/44，共8额外对象；不补训练。

每个比较要求 sample_id、标签、类顺序、split、mask 与实际评估单位一致。D/TS同checkpoint；MC/Off同checkpoint；D/Off/MC同配置同seed；DE同时对照其成员的逐模型指标与概率平均预测，不能拿成员ECE均值冒充ensemble ECE。DE相对seed42 D和相对三个D成员的均值是两个不同估计对象，分别标注。

### A2. 错误识别与拒识行为

固定一个预测器时，比较多个不确定性分数排序同一错误集合；跨预测器时，同时比较基础错误率、任务性能和排序效用，不能只按AURC或AUROC选总冠军。

EuroSAT：softmax、top-1错误为正类。分数方向统一为越大越不确定：`1-max(p)`、Shannon预测熵、负的概率top1-top2 margin。D/TS另做下节的logit尺度诊断。EE、MI仅对具有明确MC/DE预测分布的对象定义；D/TS/Off记NA，不伪造为0并参与排名。

按现有主表accuracy换算，12个EuroSAT D的错误数为43–124，其中frozen为43–46；执行时从实际预测核实每对象的错误数。错误正例有限可能使排序差异区间较宽；接受无稳定差异，不为获得显著性删配置或追加seed。

TreeSatAI：15标签sigmoid/BCE、现行阈值0.5，绝不改成argmax。逐标签错误为`(p>=0.5) != y`，决策置信度`max(p,1-p)`；熵按Bernoulli定义。主分析包括：

- 15个标签分别的错误识别，报告每标签正/负例、预测错误/正确数及NA原因；不因标签稀有而删除。
- 逐图Hamming损失（15个二元错误的均值）及逐图分数（对应逐标签不确定性的均值），研究拒绝一幅图的行为；不能把Hamming风险写成exact-match风险。
- macro/micro含义和有效标签数明确，micro重采样仍以图像/可用地点为单位。Exact-match仅作定义独立的补充损失，报其错误基率，不因高错误率认定指标无效。

分类指标：错误为正类的AUROC；**average precision**（明确实现及定义，不与梯形PR-AUC混名）；risk–coverage曲线与预设coverage 1/.9/.8/.5处风险；AURC摘要、随机排序与oracle排序参照；覆盖和拒识的类别组成。e-AURC如报告，不能宣称消除了所有基模型影响。无正确或无错误时AUROC记NA；AP的常量目标情形须显式标记，不填一个看似可比的普通值。

tie处理：相同分数作为同一阈值组，使用标签无关的随机接纳期望/分数段内插约定，或明确且一致的整组接纳；记录实际coverage。不能按错误标签打破ties，不能用stable sort碰巧有利的文件顺序制造性能。oracle使用标签仅为参照，不作为可部署分数。

### A3. 分割采用两个层次，控制排序与重采样成本

全测试集**图像级主分析**：CloudSEN12 975图，SpaceNet7 1152图；以实际ID为准。每图计算有效像素错误比例，作为图像的损失；对`1-confidence`和预测熵等取有效像素均值作为图像不确定性。报告图像排序的risk–coverage、工作点、AURC及不确定性与连续错误比例的描述性关联。不将“图中任意一个像素错”当主二元目标，以免全部图像为正类。

主风险按图像等权，另列有效像素数和现行pixel-weighted性能口径；它不是全局像素拒识。并列mIoU、per-class IoU/recall，尤其SpaceNet7建筑物，避免背景掩盖失败。真值类别/边界分层只用于诊断，不能用真值构造实际拒识分数。

MC全测试文件已存predictive entropy、expected entropy、MI-style disagreement和`predictive_variance`。验证其估计约定后，可零前向得到全测试Shannon/Gini及逐图汇总；**不能误写为只有32图才有MI**。DE全测试分解可由三个D成员的全概率离线计算，但默认不开这个额外读取分支；跨MC/DE完整分解比较默认限于共同子集，避免范围错配。

像素诊断：每数据集使用已固定32张研究子集，所有方法同ID、同mask。八配置×四方法×32图，共1024个“图像–预测对象”切片；报告逐图像素AUROC/AP/RC、关键类/边界诊断、有效图数及组数。MC T=30与DE M=3的raw probabilities在此共同子集可用，按样本对齐后比较Shannon/Gini分解。逐图像素指标汇总与“合并全部像素后的指标”是不同估计对象，明确命名，默认前者。不把子集结果写成全测试像素结论。

这项子集选择是控制CPU排序成本的研究范围，不是数据不存在。全测试像素级拒识、逐draw T敏感性和更多分割seed均是关闭的扩展，不自动追加。

### A4. 尺度和代理量解释

EuroSAT使用12组D/TS同checkpoint、既有正温度，不重拟合test。检查相同样本的argmax与accuracy应保持（极少数数值平局按已有审计说明处理）。TS可能改变跨样本置信度、错误识别AUROC/AP与拒识排序，变化本身不是bug。

多分类logit尺度用中心化`z - mean(z)`的范数及top1-top2 logit margin，解释它们与置信度、错误/正确、校准的关系；原始logit范数有公共平移任意性，不能作主要证据。概率margin与logit margin分开标注。TreeSat的逐标签log-odds遵循Bernoulli语义，不对15标签做softmax或categorical centering。优先使用保存logits；若仅有概率，log-odds计算必须披露截断，不能冒称原始logit。

对同一MC/DE预测集合`p_t`，`p_bar=mean_t(p_t)`，熵单位nats：

```text
Shannon TU = H(p_bar)
Shannon EE = mean_t H(p_t)
Shannon MI = TU - EE
Categorical Gini TU = 1 - sum_c p_bar[c]^2
Categorical Gini EE = mean_t (1 - sum_c p_t[c]^2)
Categorical Gini EU = mean_t sum_c (p_t[c] - p_bar[c])^2
```

Gini EU等于总体方差（分母T或M，ddof=0）的类维度和。保存的MC variance正是该定义仍须核对实现/headers与子集实际值。可由TU−EU得到EE，不要求重跑draws。TreeSat以单Bernoulli方差口径`TU=p_bar(1-p_bar)`、`EE=mean[p_t(1-p_t)]`、`EU=Var(p_t)`逐标签计算；若与两类categorical表达比较，明确相差因子2。Shannon逐标签求和/均值的汇总约定固定。不同类别数、标签数、输出单位的原始量不直接混排。

Brier沿用现行主表的任务口径并验证：categorical为每样本/有效像素的类平方误差和后平均；multilabel为Bernoulli平方误差按实际协议聚合，明确是否再除标签数。若现行实现与上述概括不符，先从代码确认并分别命名，不改旧表归一化去迎合公式。

检查分解恒等式及残差，报告模型/适配改变时代理量如何变化、分数排序是否一致及与错误识别的联系。相关性受共享分量和`TU=EE+MI`约束；EE/MI相关、排序反转或交互非零均不能单独证明物理AU/EU耦合，也不能说同一生成分布的真实`H(Y|X)`因换模型而改变。MC30与DE3代表不同近似分布，不把差异都归因于样本数量或称已识别真实AU/EU。

### A5. 统计与资源方案

在看新增结果前写有日期的分析说明，固定矩阵、分数、损失、配对、单位、工作点、bootstrap seed/次数及NA规则。已有test分析标探索性，不进行基于test表现的选指标或调阈值。不拟合新的部署拒识器，也不声称有限样本风险保证。

分组从实际metadata检查：CloudSEN12优先ROI/相同来源组，SpaceNet7按AOI；若同源产品把多个ROI连接起来，记录更保守的独立单位及其理由。既有记录有195个Cloud ROI与12个Space AOI，需要实际核实。EuroSAT/TreeSat有可靠地点分组则用地点，否则按图像并声明空间相关信息不足；不要臆造scene ID或把样本前缀无验证当独立地点。

配对比较使用同一bootstrap重采样的图像/组，固定1000次、95%区间并列有效重复数。为避免改变估计对象，组重采样后仍按预定图像权重聚合，明确AOI等权和图像等权的区别。像素诊断按图像/组重采样摘要，不以像素为独立单位。12个AOI的区间可很粗，不改回patch级来变窄。稀有错误造成的无效bootstrap重复须统计，不伪造区间；不能以“不显著”证明等效。

训练随机性与评估样本不确定性分开：逐seed结果及描述性均值/范围；bootstrap区间只条件于既定训练模型，不能代替新独立训练。每格一个DE组不能给跨ensemble稳定性结论。多配置共享样本/模型也不独立。

成本：分类归档约0.59 GiB，其中MC raw约72.3 MiB；分割归档约64.71 GiB，其中MC全图约20.58 GiB，Off另存。这些不是A必须读取的精确总量，也不是RAM。先用一份分类、一份分割测metadata读取、解压、逐图聚合、排序、重采样各阶段，记录硬件、并发、峰值RSS、实际字节、wall time。按同结构产物和不同阶段估剩余量，给区间与假设，不机械线性外推排序/内存成本。

先单进程；检查可用RAM/临时盘。压缩NPZ不因`mmap_mode`就可随机读取内部数组，采用逐数组解压/流式提取到本轮scratch memmap，逐批处理，保留紧凑派生表。不得一次加载多个分割全文件或bootstrap时反复解压。输入hash复用既有可靠记录；对实际使用文件必要时一次流式校验，不反复重hash无关65GB。只清理本轮创建的scratch。

## 3. B：仅显式开启时做表示分析

默认B范围只使用已保存pooled表示：分类18份对应9个frozen/full seed配对，分割24份对应12配对。缺DOFA–EuroSAT的6份D表示。验证实际数组的层、提取过程、pooling、样本ID和输入一致；不能用MC表示替代缺失D。

分类优先按`backbone_representation`字段和导出实现定位语义。当前`prediction_export.py`将同一个`embedding_values`同时存为`embeddings`与`backbone_representation`，本轮三个代表文件已验证二者完全相等；不要仅凭两个字段名就断言一个是head BatchNorm后的表示。后续对所有实际使用文件验证别名/提取层，若真有不同导出版本，分别标注。

冻结backbone理论上跨head seed相同，但先核对权重、buffers、预处理和表示数值。若确认相同，分类为3个唯一冻结参照对9个full表示、分割为4个唯一冻结参照对12个full表示；21个配对不等于21个独立冻结参照。full跨seed变化用作表示随机性背景，冻结侧恒同是验证结果而非伪造重复。

linear CKA比较相同样本/层/表示，给与性能/校准变化的描述性关联；不拟合test分类器，不从相关宣称导致校准。分割现存`[N,768]`是最终特征的全局均值，不能声称逐层/空间证据。

只有`B_fill_missing_embeddings=true`才为6个D checkpoint补测试特征；保守6次checkpoint×全测试编码。验证三个frozen共享同一编码器/预处理后，1 frozen+3 full可降为4次编码，仍生成6个可追溯导出。保留历史来源UNKNOWN。逐层/空间探针另立范围，不自动追加；这些需要前向或梯度计算，但未必需要训练。

## 4. C：仅显式开启时做一种输入退化

EuroSAT两FM×两适配×seed42，clean复用有效原结果，新增两档Gaussian blur：**在原生64×64上、resize/normalization前**逐通道操作，sigma为0.5和1.0原生像素，clean=0，反射边界，kernel=`2*ceil(3*sigma)+1`。记录图像范围、库版本、顺序与输出hash。先用非test的calibration样本作标签无关技术/可视检查；若检查发现操作或保标签假设不成立，记录原因并修订协议后锁定，不按test效果挑强度。只支持该退化和强度，不能推广到地域OOD或把blur当作纯AU干预。

扰动之后仍走每个原run自己的预处理和归一化常数，特别保留DOFA–EuroSAT历史常数；不能为方便cache而统一改变旧模型输入。相同原生blur状态不代表所有模型共享同一归一化张量，cache key必须绑定完整输入处理。

`D_only`：4格×2新状态=**8个checkpoint×全测试集评估**，每次2714图；不是8个单样本forward。

`all_methods`：每格每状态评估D的3成员，以及1个MC模型（T30）和该模型Off；TS离线变换D logits，温度沿用clean calibration拟合，不在test重校准。D42已是DE成员，不重复运行。

- 总编码次数保守`4格×2状态×(3D+1MC)=32`；其中D成员24次、MC8次。**这是从无新扰动预测开始的总数**；若D_only的8次已完成且产物完全相同，则再增加24次，累计仍32。
- head/decoder数据集级计算次数`8×(3+30+1)=272`，非272次backbone编码。
- 现行MC实现每batch只编码一次再调用head30次，full适配也如此。若跨冻结D/MC的编码器state/buffers、预处理和特征实现一致，且实现受控cache，可降至`2FM×2状态×(1 frozen + 4 full)=20`次唯一编码。只凭“frozen”名字不能共享。

保存平均概率、必要draws/方差、输入状态和cache key，测任务性能、校准、A的识错/RC与代理响应，并报告差分及范围。任何方向均为有效结果；不要求随强度单调增不确定性。缓存/聚合/数据加载成本计入实测，T/M操作数不当wall-time倍率。

## 5. D：仅显式开启时做模块可训练性的2×2干预

一个任务、两FM、输入机制trainable与Transformer主体trainable两个开关，head始终训练。四条件：两者冻结、只输入机制、只主体、两者训练。三seed42/43/44，共**24次新训练**；单FM版12次。旧frozen/full的LR、WD、上限、patience等不同，当前不能直接当统一配方端点复用。未来若真有严格匹配端点，才按清单扣除；不要据此把现在预算写成12/6。

先输出实际参数分区清单并核实：DOFA输入`patch_embed/Dynamic_MLP_OFA`，主体`blocks/norm`；Panopticon输入`model.patch_embed/PanopticonPE`及其channel相关子模块，主体ViT blocks及实际norm。所有positional、special token、final norm、原本结构性冻结的SAR参数必须显式归属/保持规则；不能遗漏后笼统说“输入与主体互斥穷尽”。按真实实现确认名称，保存可训练参数数目和buffer处理。

最小代码修改支持四种adaptation标识、独立input/body/head参数组、冻结/更新审计与日志，不能继续把“仅输入模块”错误登记为full。对四条件统一输入/归一化、增强、初始权重、head结构与初始化、head LR、每个活跃backbone参数组LR、WD、optimizer、固定优化步数上限及validation频率/选择规则。采样器和初始化RNG分离，保持可配对数据序列。训练固定预算，checkpoint按同一非test标准选择；如使用不同预算/调参，明确估计对象随之变成配方效应。

EuroSAT候选新协议的具体起点：FP32、物理batch64、AdamW(beta=.9/.999, eps=1e-8)、WD=.01，活跃input/body组LR=1e-4、head LR=1e-3；令`R=ceil(N_train/64)`，总步数`50R`，前`5R`步从零线性warm-up到设定LR，其后constant，每`R`步完整validation，按最低val NLL保留checkpoint、同分保留较早者，无基于指标的early stopping。N_train=18866时R=295、总14750步。此为**待机械/资源smoke验证的新统一协议提案**，不是已验证最优超参或旧结果的精确复刻；四条件同步应用。若资源不容许此预算，在任何正式运行前统一修订步数/批量并记版本，不能只缩减不利条件。不同任务需在同样约束下另列明确数值，不能套用这个EuroSAT步数。

先做受控validation smoke及梯度/参数更新检查；记录这些试运行的实际训练成本并列入预算，不触发自动LR搜索。冻结的Transformer仍须传回到可训练输入模块的梯度，不能用`no_grad`包住整条路径；“仅输入模块”成本可能接近full，不能按冻结head时长估算。

执行前根据当前资源做每条件短计时和明确步数总预算，输出具体矩阵/命令/成本；若用户未提供必要资源上限，完成可审查配置后只询问缺少的上限，不无限开启训练。没有通用3seed检验力保证，先报效应量和逐seed范围。结果只能归因于指定训练设计下的模块可训练性，容量/优化可达性是效应的一部分；两FM预训练差异仍使架构普遍因果比较不成立。

D默认使用D模型；若主张涉及MI/EE的模块效应，必须为四条件一致定义MC/ensemble估计器及额外预算，不能拿D模型伪造MI。该升级不在默认24次范围内。

## 6. E：仅显式开启时做可解析参照，零FM训练

这是解释性小研究，不是EO来源识别实验，也不是自动新理论。用已知生成分布检验代理量能否被直接解释为真实来源。

设计：X有−1/+1两个等权层；歧义a∈{0.1,0.4}，真实`P(Y=1|X=-1)=a`、`P(Y=1|X=+1)=1-a`，总体类别先验固定0.5。每层n∈{20,200}个标签，总量40/400。每层独立Beta(1,1)先验，观察k个正例后的后验`α=1+k, β=1+n-k, s=α+β, m=α/s`。

```text
oracle_AU = h(p_true)
Shannon TU = h(m)
Shannon EE = psi(s+1) - [α psi(α+1) + β psi(β+1)] / s
Shannon MI = TU - EE
binary categorical Gini TU = 2 m(1-m)
Gini EE = 2 α β / [s(s+1)]
Gini EU = 2 α β / [s²(s+1)]
```

四个(a,n)条件，分别枚举`k=0..n`并按真实Binomial(n,p_true)权重求数据外层期望，无需用任意20重复近似均值；两个层直接计算共888个加权后验项。数值检查概率权重和、分解恒等式、少量积分参照；oracle与posterior proxy必须分列。按预设单位给差分和差分之差，说明非线性/有限支持能产生非加性代理响应；不称其识别了物理耦合或反证分解恒等式。

本设计只报告给定X的逐层量及这些量的等权平均，即先计算条件不确定性再对X平均。不要把两层概率先平均再计算熵；那是丢弃X后的另一估计对象。若另行研究跨层混合预测的非线性量，需要对`(k_minus,k_plus)`联合枚举，888项的单层计数不再适用，且不属于此默认toy。

只有`E_sampling_illustration=true`才加抽样变异示例：20个配对数据重复，两个X层各有独立U流，跨a共用U、n20为n200前缀，共80条件重复、160个后验实例。20仅用于展示，不是功效保证；解析枚举仍为主。无GPU，无FM训练，记录实际CPU耗时，不预报秒级完成。

## 7. F：仅显式开启时做EO训练支持×输入退化

建议固定一个来源较清晰的Panopticon–EuroSAT frozen模型，用相同head dropout p=.1、T30的MC体系研究代理量。将因素明确命名为**训练支持与观测退化**，不把两个开关预先命名AU和EU。

低支持取训练池20%，全支持100%；保持非test数据划分、分层类别/可靠地点规则，子集只从train选择。低支持子集嵌于相应全训练池，具体索引和hash先固定。两种输入退化沿用C定义，clean加两强度。模型训练只用clean；在三输入状态评估每个固定模型。

- `joint_pilot`：低/全支持×3个联合重复，共6个唯一训练位置；每个重复固定数据抽样seed和优化seed。二者变化混合，不能分解数据与优化方差。**建议新统一配方预算6次，约原72次的8.3%**；共18个模型–输入状态。
- `crossed`：低支持3个独立子集×3优化seed=9，完全相同的100%训练池×3优化seed=3，共12个唯一位置，**建议新增12次，约16.7%**；共36个模型–输入状态。不能把同一个全集当3份独立数据再凑出18次设计。

若估计“仅支持量”的影响，采用相同优化步数/选择日程；低支持会重复更多epoch，需披露。旧按epoch/早停的run通常不匹配新步数控制，默认不复用。若改为沿用旧训练配方并限制结论，经逐run验证后才扣除已有位置：Panopticon–EuroSAT frozen MC目前仅seed42，最多复用1个，最低新增5/11；DOFA对应MC有42/43/44，理论最低3/9，但相关历史来源边界和具体配方仍须检查。不能默认任意配置都有3个完整MC。

F的新统一配方起点为FP32、batch64、冻结编码器、现行BN-linear head加p=.1 dropout、AdamW(beta=.9/.999, eps=1e-8)、WD=.01、head LR=1e-3 constant。`R=ceil(N_full_train/64)`，两支持条件均训练`50R`步、每`R`步完整validation、minimum val NLL选择，无指标early stopping。低支持用同一固定子集按预定seed循环shuffle取batch；dropout/采样RNG分别记录。所有seed使用相同规则。当前full训练池18866时总14750步，低支持相当于更多遍历，这是固定优化预算设计的已声明代价。机械/资源smoke先行，正式运行前统一修订才允许改变这个新协议。

冻结head可缓存编码器特征降低新训练成本，但现有归档不等于已有全部train/validation特征。先列缺失的train/val/test/扰动输入编码；按split大小计量，不能称“head训练无需任何新FM前向”。共享编码器必须验证权重/预处理/buffers和特征语义，防止跨checkpoint错缓存。现行`final_training_protocol.md`第5节明确所有split无随机增强，因此静态特征缓存与该输入协议兼容；当前训练入口逐epoch调用模型，加入缓存是需要实现和验证的代码改动，不是现成功能。若未来改变在线增强，必须重新检查缓存是否改变训练数据分布。缓存保留在head之前，head BatchNorm、dropout和优化过程仍按训练协议执行。

报告support、corruption及其交互的任务性能、校准和代理效应，区分subset与optimization重复及其层级；有限3×3主要支持描述，不宣称充分估计所有随机效应。非零交互仍不是物理AU/EU来源证明。真实来源识别需要额外的可验证生成机制、重复观测/标注或受控模拟以及排除替代解释，不能承诺再训练固定几次就完成。

## 8. 可选统一成本计时

仅`timing_benchmark=true`时，对明确选择的同task/batch/hardware/dtype执行D、TS、MC、Off、DE一致计时。区分数据加载、backbone、head/draw、聚合、CPU传输、TS拟合与应用；报告同步、warm-up、重复、cache、峰值显存和wall time。TS温度复用已有非test拟合；不重新调test。可只做代表性预定小范围，结论限该范围。此项需要新的前向，不属于默认A。

## 9. 验证、交付与结束条件

先实现最小科学测量测试：完美/反序/tied分数的AUROC/AP/RC、multilabel决策、ignore mask/无有效类、ID错配会失败、logit公共平移不改中心化尺度、Shannon/Gini恒等式。浮点公式用float64验证，容差明确；float32产物检查使用与精度匹配的误差预算，不能为了通过放大容差。

对实际使用对象确认sample_id/标签/类顺序100%对应；均值概率与保存均值一致；重算必要accuracy/NLL或分割任务指标与已有审计主表一致。优先复用已验证实现，通常1e-6量级，已记录的数值边界按原证据解释；遇到不符先定位精度、clip/mask/聚合，不能未经说明覆盖旧结果。科学错误只停止受影响分支，其他独立工作继续；缺失不能通过虚填NA而宣称完整通过，必须在覆盖矩阵保留缺口。

默认A只新增分析代码和派生指标，不修改模型训练机制。发现需改旧模型代码/重算主表/补推理/补训练的问题，给最小复现和影响范围；按用户已选范围推进，超出范围的实验不自动运行。

至少交付：

1. 日期明确的`ANALYSIS_PROTOCOL.md`、参数文件、输入/对象清单、代码revision/工作区diff及后续协议变更日志。
2. 直接来源和ID对齐验证、必要数值复核、NA/缺失清单、分阶段成本profile与实测总量。
3. 每对象结果CSV、紧凑逐样本/逐图分数、配对效应及区间CSV、可复现命令；避免保存冗余数十GB复制。
4. 按任务分面的RC曲线、固定工作点、错误识别、TS/logit尺度与Shannon/Gini比较图，标明样本/seed/评估单位及探索性范围。
5. Claude独立审查原文、公开工具证据索引、唯一问题清单和接受/驳回的直接证据。只用他人的总结不能PASS。
6. `checks_delta.csv`沿用29项审查ID及PASS/FAIL/UNKNOWN/NA，附evidence_path、影响RQ/后续问题、minimal_resolution；未重新检查的项目引用既有有效直接证据或明确UNKNOWN，不伪称全部重审。新后续问题可增加FUP编号，不混改原标准。
7. `FOLLOWUP_COMPLETION_REPORT.md`逐题回答：当前能回答什么；证据支持哪些结论；必要证据缺口；需要修代码/重算指标/补评估/补训练中的哪一种；论文可用结果段落草稿与限制；后续探索单独列出。
8. 最终manifest、产物存在性检查、输入未修改核查；给出可复跑入口和完成/缺口列表。

验收按覆盖、实现有效性、对照和证据范围判定，不设“必须提高0.01”“必须显著”等门槛。任务性能差、方法没改善、跨适配差异不稳定可以完成研究。没有新增可验证主张需要时，A完成并经独立复核即停止，交付结果，不继续替论文寻找正结果。
