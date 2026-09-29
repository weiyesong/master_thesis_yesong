# 历史来源最小补查（2026-09-21 UTC）

结论：I01 的 C01/C06 仍为 **UNKNOWN（限定 DOFA–EuroSAT 历史来源）**。找到了常数在较早项目源码中的记录，未找到六个历史 run 与训练时完整源码之间的可靠绑定，也未找到这些常数的统计推导或原始引用。该补查不构成测量错误或数据泄漏的证据，不要求替换不可变模型或补训练。

本文件只做本地有界取证；没有读取认证配置、密钥或账户凭据。没有修改旧 run、checkpoint、配置或既有审计文件。

## 查找范围与结果

| 检查对象 | 直接发现 | 能支持的范围 |
|---|---|---|
| `六 run 目录的 environment.json、resolved_config.yaml、目录文件列表` | git_commit 均为 null；各 run 无 code_snapshot.json；保留 normalize=true、RGB 波段与图像尺寸，但配置没有具体 mean/std 数值 | 保存配置与结果可检查；训练代码版本无法据此恢复 |
| `reports/dofa_eurosat_final_manifest.json 与 results/final_thesis/dofa_eurosat_freeze_20260810/code_snapshot.json` | manifest 明示 Post-hoc snapshot；代码摘要 c99c1409e6a01bd4c4e2aca5208e847548b487e3ab8b84a798c796d1e16baa01 | 2026-08-10 后验保留的代码证据，不是六 run 训练时版本 |
| `.git 当前全部 refs` | rev-list --all --count = 0 | 无可绑定六 run 的本仓库提交 |
| `.git.backup 的全部 refs、reflog、对象检查` | 仅提交 ea6686abbe282a4f30bd701f867f7aaebd0857ca；提交元数据时间为 2025-12-31T06:04:31+08:00；无 scripts/run_experiments.py | 历史项目提交存在，但不能作为 2026-08 训练提交 |
| `.git.backup:ea6686a:datamodule_fixed.py` | blob c910be56d734868d4a796cc767766c77ded00e9f，第 43–54 行已有同组 Sentinel-2 常数；仅有 from EuroSAT dataset 注释 | 常数在较早项目源码中出现过；没有计算脚本、计算样本 ID、split、像素数或原始统计出处，derivation 仍 UNKNOWN |
| `.git.backup unreachable objects` | 6 个小 blob（49–1091 字节）、1 个 tree；tree 只含 .gitignore 和 Dockerfile；六 blob 不含常数、run_experiments、目标 run 或 20260807 标记 | 未恢复相关历史训练源码 |
| `scripts / configs / experiments / thesis / reports / environment / 项目隐藏配置目录 / 六 run 与事后冻结目录` | 精确检索 1136.89、1184.39 等常数；命中当前代码、早期实验记录和明确声明来源缺失的协议 | 后续重复记录不是独立推导来源 |
| `/home/yesong/.vscode-server/data/User/History 的 entries.json` | 仅 1 份 metadata；没有 run_experiments、datamodule、EuroSAT、experiment_manager、normalization 相关 resource | 该有限本地历史索引未提供补充；未将目录不存在或未命中解释为全世界不存在证据 |

可重复的只读取证命令：

```sh
git --git-dir=.git.backup log --all --format="%H %aI %s"
git --git-dir=.git.backup ls-tree -r --name-only HEAD
git --git-dir=.git.backup show HEAD:datamodule_fixed.py
git --git-dir=.git.backup reflog --all
git --git-dir=.git.backup fsck --full --unreachable --no-reflogs
git --git-dir=.git rev-list --all --count
```

没有将提交日期当作外部可信时间戳；它只是本地 Git 对象中的元数据。没有从“from EuroSAT dataset”一句注释推断统计使用了哪个 split。

## 受影响 run 与源文件

| 适配 | seed | run ID / 原始 environment |
|---|---|---|
| frozen | 42 | [20260807T103233022656Z_eurosat_dofa_frozen_bnlinear_seed42_338e2700](../../results/baselines/dofa_eurosat_frozen_bnlinear/runs/20260807T103233022656Z_eurosat_dofa_frozen_bnlinear_seed42_338e2700/environment.json) |
| frozen | 43 | [20260807T104927933179Z_eurosat_dofa_frozen_bnlinear_seed43_3567033c](../../results/baselines/dofa_eurosat_frozen_bnlinear/runs/20260807T104927933179Z_eurosat_dofa_frozen_bnlinear_seed43_3567033c/environment.json) |
| frozen | 44 | [20260807T110954653187Z_eurosat_dofa_frozen_bnlinear_seed44_36c48b49](../../results/baselines/dofa_eurosat_frozen_bnlinear/runs/20260807T110954653187Z_eurosat_dofa_frozen_bnlinear_seed44_36c48b49/environment.json) |
| full_finetune | 42 | [20260808T211626550846Z_eurosat_dofa_full_finetune_seed42_466f9f00](../../results/baselines/dofa_eurosat_full_finetune/runs/20260808T211626550846Z_eurosat_dofa_full_finetune_seed42_466f9f00/environment.json) |
| full_finetune | 43 | [20260808T215707981044Z_eurosat_dofa_full_finetune_seed43_53c8b069](../../results/baselines/dofa_eurosat_full_finetune/runs/20260808T215707981044Z_eurosat_dofa_full_finetune_seed43_53c8b069/environment.json) |
| full_finetune | 44 | [20260808T225252226548Z_eurosat_dofa_full_finetune_seed44_a9291cb9](../../results/baselines/dofa_eurosat_full_finetune/runs/20260808T225252226548Z_eurosat_dofa_full_finetune_seed44_a9291cb9/environment.json) |

同一历史归一化输入约定也被后来的 DOFA–EuroSAT MC 方案沿用。对这些配置，RQ1 只能描述整套模型与预处理配置差异，不能隔离纯架构效应；RQ2 仍可描述同模型、同预处理下的两种完整适配配方；RQ3 的同权重 Off/MC 对照固定该输入约定，可以回答随机推理差异，但不能替历史统计来源背书。

当前可明确的常数为 RGB mean `[1136.89, 1120.77, 1184.39]`、std `[965.23, 712.12, 650.20]`。Panopticon EuroSAT 使用另有 train-only 统计记录的常数，详见 [final_dataset_protocols.md](../final_dataset_protocols.md) 与 [final_training_protocol.md](../final_training_protocol.md)。这些“当前代码和协议中的具体值”与“训练期绑定/统计推导已证明”是不同证据层级。

## 最小处理与验收

本轮已完成：补查本地备份、明确受影响 run、记录较早常数载体，并在 [RQ1/RQ2 结果段落](rq1_rq2_draft.md) 保留配置差异与来源限制。验收应是 UNKNOWN 的范围和影响被准确披露；不能把披露完成改写成来源已恢复。

若未来拿到当时的 source archive/执行包、带 run ID 的训练期 hash，或常数计算脚本及其输入样本/统计清单，可再核对后升级来源状态。当前无需重新训练；重训也不会恢复历史 run 的真实 provenance。不要为了提高审计 PASS 数而重命名后验 snapshot。

I05 的完整外部论文仍未提供，因此“未提供的全文是否恰当限定结论”仍无法审阅。新整理结果段落可作为本次交付中已审范围，其内容与证据可直接复核；它不声称已经审完外部全文。

## 关键证据文件 SHA256（本轮读到的版本）

| 文件 | SHA256 |
|---|---|
| [reports/dofa_eurosat_final_manifest.json](../../reports/dofa_eurosat_final_manifest.json) | `bdea21e9901219b6fa6d587ef41917ad781ee9eaf076f95a0ad2c09178c77cd1` |
| [results/final_thesis/dofa_eurosat_freeze_20260810/code_snapshot.json](../../results/final_thesis/dofa_eurosat_freeze_20260810/code_snapshot.json) | `4694b894c76d80659932fb2c404bff1a637bb5c7c8e8df8a760236e894895717` |
| [reports/final_dataset_protocols.md](../../reports/final_dataset_protocols.md) | `4bd00de7c7baa263e12d48fd4c3eba738103ac4c63cfd3d10a12aa5da16c0331` |
| [reports/final_training_protocol.md](../../reports/final_training_protocol.md) | `cd569a938552f5cb85573d335d4f1a4c9b0815445a999853226b8052f099d711` |


## 本轮数值与方向交叉核对的可复现记录（更新于 2026-09-22 UTC）

[provenance_numeric_crosscheck.json](provenance_numeric_crosscheck.json) 保存 24 个分类原始 parquet 的独立公式重算、24 个分割原始 `test_metrics.json`/confusion 算术核对、48 个 deterministic ECE-15 箱表的加权 gap/ECE 重建。每行含 run 定位字段、原始路径、相应源文件 hash、计算值/参考值/差值和容差判定；方向记录保留非空箱数、正负箱数及是否混合，16 个条件摘要继续保留三个 seed 的方向计数和范围。所有记录通过绝对容差 `1e−6`。

这是本轮的直接数值证据，既不恢复历史训练源码，也不替代上一轮全量分割概率重算。具体算法见 [provenance_numeric_crosscheck.py](provenance_numeric_crosscheck.py)，可从项目根目录重跑：

```sh
python reports/core_rq_completion_20260921/provenance_numeric_crosscheck.py
```

脚本只读取原始产物与既有审计表，并更新同目录的该 JSON；不会改动旧结果或训练模型。JSON 记录本次脚本与四张输入表的 SHA256。分类验证包含原始 parquet SHA256，分割核对包含原 run `test_metrics.json` SHA256；分割本轮没有声称再次重算全量概率。方向结论已写入 [RQ1/RQ2 结果段落](rq1_rq2_draft.md) 的逐条件方向小节。
