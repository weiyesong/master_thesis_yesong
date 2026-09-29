# 原始文献定位笔记

检索日期：2026-09-26。目的为检查本论文定位与主张边界，非系统综述或全球新颖性认证。以下为原始论文/作者预印本；网页摘要、正文阅读范围分别注明。

| 文献 | 本轮核查范围与最相关内容 | 对本论文的处理 |
|---|---|---|
| [Hüllermeier & Waegeman, 2021, Aleatoric and epistemic uncertainty in machine learning](https://arxiv.org/abs/1910.09457)，期刊 DOI [10.1007/s10994-021-05946-3](https://doi.org/10.1007/s10994-021-05946-3) | 核对作者预印本摘要、版本与期刊信息；概念和方法综述 | 用作基础概念来源；本轮未逐节审读其长篇全文，不据此宣称所有相关理论已核完 |
| [Valdenegro-Toro & Daniel Saromo Mori, CVPRW 2022, A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement](https://arxiv.org/html/2204.09308v1) | 阅读摘要、引言及分解方法相关段落；研究不同 UQ 方案中的估计相互影响 | “两类估计可能相互影响”已有先例；其 sampling-softmax 采样量建议不能直接用于否定本项目不同流程的 T=30 |
| [Wimmer et al., UAI 2023, Quantifying aleatoric and epistemic uncertainty in machine learning](https://proceedings.mlr.press/v216/wimmer23a.html)，[正文](https://proceedings.mlr.press/v216/wimmer23a/wimmer23a.pdf) | 阅读熵分解及批评解释的相关段落；其质疑对象是度量的适当性与解释，不是熵恒等式的数学正确性 | 讨论章区分概念、估计量、可观测证据；不能声称首次质疑加法分解或已经推翻定义 |
| [Ramos-Pollan, Kalaitzis & Panner Selvam, 2024, Uncertainty and Generalizability in Foundation Models for Earth Observation](https://arxiv.org/abs/2409.08744) | 核读作者摘要与提交信息；研究多 FM、AOI、下游任务的空间泛化与不确定性 | 不能说此前 EO FM 不确定性无人研究；本项目不把其空间泛化结论当成本项目的 OOD 证据 |
| [Lehmann et al., 2026, Beyond Accuracy: Assessing Calibration of Geospatial Foundation Models and Their Sensitivity to Distribution Shifts](https://arxiv.org/html/2608.16614v1) | 阅读摘要、相关工作、III-A 方法、数据表及附录 C；以 frozen encoders 研究校准、偏移及 UQ 方法，附录说明 SpaceNet7 指标区分问题 | 是直接相近工作，必须在相关工作比较；本项目增加的研究轴是其所读设定未覆盖的 full 配方对照及本项目的匹配 MC 分析，不宣称这是全球首次；保留自身 SpaceNet7 结果并解释局限 |

以上区别是本次读到的文本与本项目协议之间的比较，不是对相关作者实现和全部结论的复核。特别不要照搬任何论文中的广泛“此前无人做过”措辞。

写作时可采用一张简短比较表，列“研究对象、适配方式、任务、UQ 方法、评价维度、外推边界”。不要以模型数量或实验总数竞争创新，也不要将不同数据划分和配方的绝对指标直接对比。
