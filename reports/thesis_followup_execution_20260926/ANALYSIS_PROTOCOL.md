# A + E 探索性补充分析协议

2026-09-26，版本1。此文件在新增研究指标计算前写定。用户授权A、E；其他分支、额外分割seed、全测试DE分解、T敏感性、模型效率前向均关闭。全部计算只读取已存预测或解析分布，不加载模型进行前向/训练。

执行依据：`../thesis_followup_design_20260926/EXECUTION_PROMPT.md`，以当前用户的`enabled_packages: [A,E]`覆盖其默认值。现行三个RQ及DOFA–EuroSAT C01/C06来源UNKNOWN保持。已查看过的test上的新分析全部是探索性的。

## 对象、统计单位与对照

- 分类68对象：D24、EuroSAT TS12、MC12、MC Off12、DE8。EuroSAT40、TreeSatAI28；全部既有有效seed，不新增选择。
- 分割32对象：两任务×两FM×两适配，seed42 D/Off/MC与三成员DE。全test分析以图像为单位；逐图像素诊断仅用事前已固定的每任务32图，相同ID、mask、类顺序。
- 分类DE均值与其3个D成员匹配；MC-Off同权重、Off-D独立训练轨迹、MC-D总方案分别解释。对照中的seed与ensemble组不可互换。
- EuroSAT按已存spatial_group重采样（test 786组）。TreeSatAI实际test坐标2000组/2000图，按图像，未知邻近空间相关性保留限制。
- CloudSEN12将整个已有样本manifest中共享roi_id、equi_id或sentinel2_product_id的样本作连通分组，再限制到test；195 test ROI合并为184个来源连通组。保留既有跨split product重叠披露，不称全scene隔离。SpaceNet7按12 AOI。
- 图像等权为主估计对象。组bootstrap重采样组后保留组内全部图像，因此组大小不同导致每次总图数可能不同；不偷换成AOI等权风险。

## 预定分数与损失

分类和像素基分数：1-max决策概率、预测熵、负概率margin、Gini TU。MC/DE另加Shannon EE/MI及Gini EE/EU。D/TS/Off的EE/MI分解NA，不能填0。每个固定预测器内比较分数，跨预测器并列错误率、性能和错误数。

EuroSAT是top-1错误。TreeSatAI逐标签阈值0.5，逐标签二元错误、逐图Hamming损失、micro标签决策与macro标签指标分开；逐标签分数使用Bernoulli语义，图像分数取15标签均值。全部15标签保留支持数；exact-match仅列任务背景性能，不作为主拒识损失。

全测试分割每图有效像素错误率和各分数均值，评价图像级RC与Spearman关联；不将连续损失硬转为“任何像素错误”的AUROC。MC全图用已存EE/MI/总体variance恢复分解；DE全图只做基分数。共同32图子集MC/DE直接读raw概率分解，与各自全图均值/标签/mask核对。

子集逐图像素指标：全有效像素AUROC/AP/RC；真值类别及边界/非边界的支持、错误率、平均不确定性与预定coverage下保留/拒绝组成。边界采用现行四邻接类别变化、半径2方形膨胀、有效mask。GT只用于评价分层，不构造拒识器。无某类或无正确/错误时显式NA。

## 具体指标与区间

- AUROC：错误为正类，ties给0.5秩贡献。AP采用阈值组的average precision，不是梯形PR积分；全正确/全错误目标记NA并报基率。
- RC按不确定性升序接受；ties同组随机接纳的期望损失。coverage固定1/.9/.8/.5，另保存曲线网格.01至1。面积用离散定义`mean_{k=1..N} risk(k/N)`；ties内期望风险解析计算，oracle按实际损失排序，随机参照为平均损失。此约定对Hamming/图像连续损失同样适用。
- 工作点允许同分组内分数接纳；输出期望接纳/拒绝类别组成与coverage，不声称确定单个像素的取舍。
- 主区间：所有分类图像级对象/分数、Tree micro决策对象/分数、分割全test图像级对象/分数，1000次组bootstrap，seed20260926，同任务相同重采样用于配对差。95%百分位区间，保存有效重复数；不做显著性筛选。
- Tree逐标签、macro汇总及逐图像素诊断均报告点估计和有效性；像素诊断跨图/组的均值与配对差用1000次bootstrap作用于已存逐图指标，不重排像素。逐标签不作单独显著性断言。训练seed范围与上述固定模型条件区间严格分开。
- 配对主要覆盖TS−D、MC−Off、Off−D、MC−D、DE−D42、DE−D三成员指标均值，以及固定预测器的其他分数−MSP；适配配方full−frozen和FM配置Pan−DOFA按匹配对象报告。DE成员均值是平均单模型指标，不是ensemble指标。

## A4 尺度/代理与E

EuroSAT12对D/TS使用保存logits和既有温度，验证argmax及raw/scale关系；中心化logit范数、logit margin与错误/置信度关联，概率margin单列。Tree直接logits逐标签解释，不categorical centering。

Shannon单位nats。多类Gini类维度求和；Tree用Bernoulli p(1-p)，另注明与binary categorical相差2。方差ddof0，直接核对raw均值与保存summary；不把代理变化解释为真实生成分布条件熵变化。

E按执行prompt：a=.1/.4，n=20/200每X层，两个等权层真实p=a和1-a，Beta(1,1)先验。对每层k=0..n按Binomial真分布枚举，共888项、4条件。只报告条件量及其X等权均值；无随机20重复，无跨层混合概率熵。以数值积分抽查闭式EE和Gini矩，报告恒等式残差、oracle差及差分之差；非零交互不证明物理耦合或理论新颖性。

## 验证和资源

先测试小数组：完美/反序/并列排名、常量目标、多标签、mask和空类、对齐失败、logit平移及分解恒等式。主概率质量按现行实现定义，跨对象核对accuracy/NLL/Brier或分割pixel_accuracy/mIoU/NLL。默认1e-6量级差异阈值；保存的float32概率与旧logits/float32 clip若导致差异，定位并另列同定义复核，不修改旧表或放宽阈值掩盖。

先单进程测一份分类和一份分割的读取/解压/聚合/排序/重采样耗时。系统约504GiB物理内存、5.5TiB可用盘；仍逐文件/数组处理，分割压缩NPZ逐数组解压到本轮scratch并mmap，紧凑输出后清理。仅本轮代码和产物可写；输入hash/size/mtime及已有来源hash记录，必要校验不重复扫描整归档。长任务分阶段日志、可恢复产物，完整计算后再绘图/结果写作。

完成标准：授权对象完成或真实缺口明确，指标有效，对照和结论范围准确，实际Claude独立抽查，29项delta及后续检查有直接证据。无改善/反例/区间宽均可完成，不追加训练追求正结果。

## 版本2补充（2026-09-27，真实行为指标计算之前）

实际Claude session `5afb33d9-4dac-4ed1-8d17-2e1ec2a65b13`已独立检查分组、方差约定、E解析式和输入精度。

1. **主分析固定为保存的平均概率**，升为float64仅为稳定算术，不从logits偷偷改写概率或消除ties。MC/DE的raw预测仅用于分解，并核对与保存均值的浮点差；分解内部以raw均值保持数学恒等式，与主概率TU的极小差单列。保存的MSP浮点饱和可能影响排序，故所有对象记录唯一分数数、最大tie块、p_max=1及其错误数。EuroSAT D/TS另外统一报告float64 logits重建概率的**精度敏感性点估计**，不替换主分析、不按效果选择版本。MC绝不将平均logits的softmax当概率均值。
2. 工作点使用**分数tie块内的分数接纳期望**：目标接受质量为qN，允许非整数，在块内按同一fraction接纳；无需floor/ceil。AURC仍在整数k=1..N上计算期望风险。
3. probability consistency采用已有float32导出校验量级atol=2e-6；这不意味着所有NLL偏差也用同一容差。主NLL需要按原logits/log-prob/clip定义复核；保存概率饱和造成的语义/精度区别另列，不修改既有指标。
4. Gini EU恢复使用完整逐类variance求和；原`mean_predictive_variance`类均值是EU/C。Tree保存熵sum核对时除以15才是图像mean。边界改为**半径1**与现行配置一致，之前半径2仅为草拟选择，不曾运行结果。
5. 子集MC无valid_mask时由label!=255生成并与全图对应mask验证。逐类缺失时NA，汇总同时保留总图数/有效图数，不将NA补0或默默删去。
6. E两层镜像是维持总体类别先验的设计，不是两份独立证据；枚举888项包含镜像，444个不同后验计算位置。
