# A/E执行产物入口与复跑

先读 [完成报告](FOLLOWUP_COMPLETION_REPORT.md)，论文整理使用 [结果草稿](THESIS_RESULTS_DRAFT.md)。表字段和单位见 [数据字典](DATA_DICTIONARY.md)，输入成员/标签/分组关系见 [来源补表](inputs_lineage.csv)。科学定义见 [协议v2](ANALYSIS_PROTOCOL.md) 与 [参数](parameters.json)。所有新研究分析为探索性；没有训练或模型前向。

原始预测来自只读`research_data`及已有Off目录；[输入清单](inputs_manifest.csv)与[辅助raw输入](auxiliary_inputs_manifest.csv)记录来源。`objects/<object_id>/`保留逐对象紧凑派生数据、bootstrap数值、逐图指标和验证，CSV总表由它们汇总。`done.json`表示该对象处理完成，重跑分析入口会跳过已完成对象。

## 复跑数值流水线

在`/workspace`现有环境执行，CPU即可。全新计算请先将此目录的`code/`、`parameters.json`、`ANALYSIS_PROTOCOL.md`和`source_snapshot.json`复制到另一个新报告目录，再从新目录中的code路径启动；代码通过自身父目录确定输出位置。原数据仍在`/workspace`。复制范围不包括已计算的objects目录或整套研究归档。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/common.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s reports/thesis_followup_execution_20260926/code -p test_metrics.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/toy.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/analyze.py --task classification
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/analyze.py --task segmentation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/aggregate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/verify_and_extend.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/checks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reports/thesis_followup_execution_20260926/code/provenance.py
```

新目录复跑时将上述命令的代码路径替换为新目录；论文文字和Claude独立复核不由数值脚本冒充自动生成。`common.py`会重建输入清单/分组；不要把某个旧done对象与另一份输入清单混用。精确复现需验证source_snapshot和输入预期hash仍对应。

`LazyNPZ`将单个源文件的所需数组解压到本轮scratch，以memmap逐图处理，完成即清理；它没有假定压缩NPZ可直接mmap。分类中旧ensemble IDs及本轮早期group IDs为已知本地产物的object字符串数组，因此对应读取允许这些已知文件的pickle；不用于任意外部文件。

## 主要输出

- `metrics.csv`：664条对象×单位×分数结果；图像级和micro标签单位分开。
- `paired_effects.csv`、`within_predictor_score_effects.csv`：分别为预测器比较和固定预测器分数比较；不是数千独立实验。
- `pixel_summary.csv`、`pixel_paired_effects.csv`：32图逐图指标均值及共同有效图差值；有效数与总数均保留。
- `full_image_class_coverage.csv`：选择后目标内容与期望混淆计数比率，不是实际部署中使用GT的算法。
- `scale.csv`、`precision_sensitivity.csv`、`proxy_means.csv`、`proxy_paired_effects.csv`：尺度、浮点来源和代理诊断。
- `toy/`：解析后验项、条件均值、交互与积分验证。
- `figures/`：11幅PDF/PNG及图注/数据来源manifest。
- `checks_delta.csv`：29项既有审计状态及新增FUP检查；原历史UNKNOWN保留。

## 执行过程的实现修订

首次分类入口发现部分ensemble NPZ没有class_names，改为读取对应现行配置并核对有字段的归档。另修复NumPy布尔数组不能直接`ptp`的问题；尚未完成的对象重新处理。早期group字符串NPZ允许读取其已知object数组，后续写为Unicode；数值和分组不变。空的单对象可选CSV由汇总器显式跳过，全局适用性另在`analysis_applicability.csv`列明。首次结果未形成新的论文结论；失败日志和`code_versions/`保留以便追溯。

这些是新分析代码的实现修订，没有修改旧模型、旧checkpoint、原主表或封存审计。分数、矩阵、阈值、seed和负结果验收条件没有按结果调整。

最终完整交付校验用 `code/finalize.py`：它检查表、报告链接、科学测试日志、实际Claude成功响应及公开证据后生成 `final_verification.json` 和 `completion_manifest.json`。这是交付封装检查，不能替代新目录的数值复跑和真实独立复核。逐文件hash不包含manifest自身。代码revision、补丁和实际Git局限见 [变更记录](PROTOCOL_CHANGELOG.md)，独立复核处理见 [REVIEW_RESOLUTION.md](REVIEW_RESOLUTION.md)。
