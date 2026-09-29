# 当前硕士论文实验完成度审计（最新且唯一版本）

审计日期：2026-08-09（UTC）  
审计范围：`/workspace` 全仓库，包括现行/历史代码、configs、运行目录、日志、checkpoint、CSV/JSON、Parquet/NumPy、图像、notebook、shell/作业脚本、备份 Git 元数据和数据目录。  
审计约束：只读检查；未训练、未重新评估、未执行推理、未下载数据或权重、未运行测试；唯一写入是覆盖本文件。用户原先要求的 `current_thesis_completion_audit.md`、`remaining_work.csv`、`current_blockers.md` 均未另建，其内容已全部合并到本文件。

## 审计口径与最重要结论

- 当前论文目标共有 16 个 deterministic model × dataset × adaptation 单元：8 个分类、8 个分割。只有 **EuroSAT–DOFA–frozen** 与 **EuroSAT–DOFA–full fine-tuning** 已完成，各有 3 个正式独立 seed；即 deterministic 单元完成 2/16。
- 这 6 个正式 DOFA checkpoint 已有完整 test logits/probabilities、Accuracy、Macro-F1、NLL、multiclass Brier、ECE-15、per-class 指标与 confusion matrix，**不应重训，也不需要为 deterministic baseline 重新推理**。
- 上述每组三个 seed 使用相同 split、class order、head 和协议，test sample ID/order 完全一致。因此两组 Deep Ensemble 均已有足够成员，**只需从现有 Parquet 做概率聚合，不需要训练或模型推理**。
- Temperature Scaling 没有正式结果。现有 test logits 可直接复用；只缺 calibration-split logits、正确的 calibration-only temperature fitting 以及应用/作图。**不需要重训**。
- MC Dropout 没有可用 checkpoint。所有 6 个正式 DOFA baseline 的 head dropout 为 0，DOFA backbone 构造时 `drop_rate=0.0`；Panopticon 当前 config/head 和官方默认 ViT dropout 也为 0。把这些 checkpoint 在推理时临时改成非零 dropout 不能视为训练时含 dropout 的标准 MC Dropout。因此当前设计下 MC Dropout **确实需要新的 dropout-enabled training**。
- Panopticon 的 EuroSAT 分类 wrapper、官方 TorchGeo factory、nm 波长 metadata、frozen/full configs 已实现，但没有 pretrained-weight 本地 artifact、smoke-run 或训练 checkpoint。它是 **implementation complete enough for verification, experiments missing**，不是“完全未实现”。
- So2Sat、CloudSen12、SpaceNet7 没有数据、loader、split、config 或运行产物。主入口只允许 classification+EuroSAT；segmentation 分支明确抛出 `NotImplementedError`。分割 decoder、指标、pixelwise export 和 uncertainty maps 均不存在。
- 没有证据支持对任何现有正式 DOFA baseline 使用 `INVALID`。Full fine-tuning 的 post-best instability 很严重，但 best checkpoint 是按 validation NLL 预先选择并在 test 前重新加载的，仍可复用；不要因稳定性差或缺 UQ 指标而重训。

## 1. 仓库与 artifact 全量盘点

### 1.1 搜索结果

| Artifact 类别 | 实际发现 | 审计结论 |
|---|---:|---|
| 下游 `.pt` checkpoint | 35 | 全部归入 19 个可识别 run/attempt：6 个正式 DOFA–EuroSAT run、10 个 completed dry run、1 个 aborted preflight、1 个 legacy DOFA run、1 个 auxiliary ResNet run。 |
| foundation `.pth` | 2 | `DOFA/checkpoints/DOFA_ViT_base_e100.pth` 与 `checkpoints/DOFA_ViT_base_e100.pth` 内容相同，SHA-256 `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`。 |
| Parquet prediction files | 19 | 6 个正式 test exports、9 个 canonical dry-run test exports、1 个 legacy-dry validation export、3 个 superseded schema/debug exports。 |
| stochastic output / embedding `.npz` | 0 | 没有 MC passes、ensemble-member tensor 或保存的 backbone embedding。 |
| Pickle | 0 | 无。 |
| `.npy` | 15 | 全是 deterministic confusion matrices，不是 stochastic predictions。 |
| TensorBoard event files | 0 | `train_uqbox.py` 有 logger 代码，但没有运行日志或 checkpoint 证据。 |
| figures | 30 PNG | 正式 run 的 validation reliability/training dashboard、dry-run 图、上游 DOFA assets 和 RS3DBench 文档图；无 segmentation qualitative/uncertainty map。 |
| notebooks | 1 | `DOFA/demo.ipynb` 是上游 DOFA demo，部分已执行；不是本论文下游实验。 |
| shell scripts | 6 | 当前/历史启动脚本及上游 DOFA pretraining script；无 SLURM/SBatch/PBS/job 文件。 |
| archived/legacy | `.git.backup`, `_superseded`, `main_old.py`, `eo_uq_experiments/`, `first_stage_rgb/` | 已逐项纳入下文；备份 Git 只有一个 2025-12-31 initial commit，没有隐藏实验结果。 |

数据方面实际存在两份 EuroSAT：`data/.../tif` 的 27,000 个 13-band GeoTIFF，以及 `eo_uq_experiments/data/eurosat/2750` 的 27,000 个 RGB JPEG。另有 102 GB RS3DBench 数据，但没有 `results/rs3dbench`、训练日志或 checkpoint。没有 So2Sat、CloudSen12 或 SpaceNet7 数据目录。

### 1.2 版本与 provenance 风险

当前 `/workspace/.git` 缺少 HEAD/config/index，普通 `git status` 和 `git rev-parse` 失败。`.git.backup` 可读，但只有历史 commit `ea6686abbe282a4f30bd701f867f7aaebd0857ca`（2025-12-31 initial commit），不能代表 2026 年运行时的大量未跟踪代码。六个正式 run 的 `environment.json` 因此均记录 `git_commit: null`。每个 run 有 resolved config、环境、checkpoint hash 和不可覆盖的唯一目录，可提供较强 artifact provenance，但缺精确源代码 commit 是必须在论文限制中说明的工程/复现风险。

## 2. 实验协议族：实际配置而非文件名推断

### 2.1 Canonical DOFA–EuroSAT frozen

| 字段 | 已核实值 |
|---|---|
| task / dataset / model | classification / EuroSAT / DOFA ViT-Base |
| pretrained weights | `DOFA/checkpoints/DOFA_ViT_base_e100.pth`；load missing keys `[]`，unexpected keys 仅 `mask_token`, `projector.weight`, `projector.bias` |
| input | Sentinel-2 RGB `[B04,B03,B02]`；训练时 fallback 波长 `[0.665,0.560,0.490]` µm；当前 YAML 已显式写成 `[665,560,490]` nm 并在 DOFA 前转换为 µm |
| resolution / normalization | 原始 64×64 GeoTIFF resize 到 224×224；Sentinel-2 RGB mean `[1136.89,1120.77,1184.39]`、std `[965.23,712.12,650.20]` |
| adaptation / head / dropout | backbone 全冻结且训练时保持 eval；`BatchNorm1d(768, affine=False, eps=1e-6) -> Linear(768,10)`；dropout 0/不存在 |
| optimizer | AdamW；head LR `1e-3`；weight decay `0.01`；无 scheduler/warm-up |
| epochs | max 50，early-stop patience 10；seeds 42/43/44 的 best epoch 7/11/9，stop epoch 17/21/19 |
| split | 固定 manifest：train 18,866 / val 2,707 / calibration 2,713 / test 2,714；split generation seed 20260803 |
| selection | validation NLL 最小；test 不参与选模；保存 best 和 last |
| UQ method actually run | deterministic only (`uq_methods: [{name: none}]`) |
| available metrics | Accuracy、Macro-F1、NLL、sum-over-class multiclass Brier、equal-width top-label ECE-15、per-class precision/recall/F1/support、confusion matrix、mean confidence、mean predictive entropy |
| predictions | 每 seed 完整 test Parquet，2,714 个唯一样本，完整 logits/probabilities；无 stochastic/member tensor、无 embeddings |

### 2.2 Canonical DOFA–EuroSAT full fine-tuning

除下列字段外与 frozen 协议相同：

| 字段 | 已核实值 |
|---|---|
| adaptation | 111,160,832 个 backbone 参数与 7,690 个 head 参数可训练；固定 `pos_embed` 151,296 参数是 DOFA 结构性 sin/cos embedding，不是误冻结 |
| optimizer | AdamW；backbone LR `4e-4`，head LR `4e-3`；weight decay 0；5-epoch linear warm-up，之后常数；无 clipping/layer-wise decay |
| epochs | max 100，early-stop patience 15；seeds 42/43/44 的 best epoch 1/7/10，stop epoch 16/22/25 |
| gradient evidence | 每个正式 run 的 first-batch audit 显示 171 个预期 parameter tensors 全部获得有限、非零 gradient |
| stability | 三个 seed 均有 post-best instability review；最大 validation NLL 分别 6.77、13.41、56.91；无 NaN/Inf。Best checkpoint 仍由 validation NLL 选出并在 test 前重新加载。 |

### 2.3 Legacy DOFA first-stage run

- `results/first_stage_rgb/eurosat_dofa_rgb/`：seed 42，frozen DOFA ViT-Base + simple Linear head，RGB/224/Sentinel normalization，AdamW LR `1e-3`、WD `1e-4`、batch 32、20 epochs，best epoch 13 by validation NLL。
- 使用旧 TorchGeo 60/20/20 files 中的 train 16,200 和 val 5,400；没有使用 test 5,400，没有 sample-level predictions，没有 current 70/10/10/10 manifest、calibration split、BNLinear head或 multiseed。
- 旧训练集与当前重新分配的 split 不同；14,564/27,000 样本的 current split 与 official split 名称不同。因此仅把旧 checkpoint 在 current test 上重评可能包含训练样本，不能使其成为当前协议的合法 final run。

### 2.4 Auxiliary ResNet run

- `eo_uq_experiments/outputs/checkpoints/eurosat_resnet18_rgb_best.pt`：ImageNet-pretrained frozen ResNet18 + new linear FC，seed 42，RGB JPEG、224×224、ImageNet normalization，随机 80/20 train/val split，AdamW LR `1e-3`、WD `1e-4`、batch 64、1 epoch。
- Best/only epoch 1；validation Accuracy `0.905185`、NLL `0.326750`、ECE `0.080506`、Brier `0.156683`；无 test、Macro-F1 或 predictions。ResNet 不在当前 foundation-model matrix 中。

## 3. 每个实际 run 的唯一 reuse status

Status 优先级说明：当一个正式 run 同时可做 final baseline、UQ base 和 ensemble member 时，只赋予最高的 `REUSE_FINAL`；额外用途在“UQ/ensemble capability”中注明，从而满足“每个实验 exactly one status”。`last.pt` 只用于 provenance/resume，不单独作为成员；final/UQ/ensemble 一律用 validation-selected `best.pt`。下表中的 `REEVALUATION_REQUIRED=YES` 对六个 formal checkpoint **只表示尚需一次 calibration-split inference 以完成 Temperature Scaling**；它们的 deterministic test evaluation 已完成，不应重跑。Dry/pilot/superseded checkpoint 均无需继续训练或重评，因为对应的 current deterministic cells 已由 formal runs 覆盖。

路径规则已由各目录实际内容核实：frozen formal runs 位于 `results/baselines/dofa_eurosat_frozen_bnlinear/runs/<完整 run ID>/`，其 dry runs 位于同一实验根目录的 `dry_runs/runs/<完整 run ID>/`；full formal runs 位于 `results/baselines/dofa_eurosat_full_finetune/runs/<完整 run ID>/`，其 preflight/dry runs 位于 `dry_runs/runs/<完整 run ID>/`。有 checkpoint 的 canonical run 将 `best.pt`、`last.pt`、metric/审计 JSON/CSV 和 plots 直接保存在 run 根目录，prediction output 位于 `predictions/test/deterministic/`；下表完整 run ID 与该路径规则组合即为准确 checkpoint/output path。

| Run | Evidence / output | Status | TRAINING_REQUIRED | REEVALUATION_REQUIRED | UQ / ensemble capability |
|---|---|---|---|---|---|
| `eurosat_resnet18_rgb` | `eo_uq_experiments/outputs/`; 1 epoch, val-only | PILOT_ONLY | NO | NO | 当前 thesis 不含 ResNet；只保留历史 sanity baseline。 |
| `20260803T174846621260Z_eurosat_dofa_rgb_seed42_c1b79464` | `results/first_stage_rgb/dry_runs/runs/<run ID>/`; 1 train/val batch、1 epoch；32-sample partial val export | PILOT_ONLY | NO | NO | 只能验证旧 pipeline/export；不要续训，current cell 已被 formal runs 覆盖。 |
| `eurosat_dofa_rgb` | `results/first_stage_rgb/eurosat_dofa_rgb/`; best/last、20-epoch history、val-only | PILOT_ONLY | NO | NO | 不可替代 current cell；无需再花 compute，canonical frozen 已完成。 |
| `20260807T103123130123Z_eurosat_dofa_frozen_bnlinear_seed42_387e7ada` | frozen canonical dry；64/2714 partial test | PILOT_ONLY | NO | NO | plumbing only；已被同 seed formal run 覆盖。 |
| `20260807T103137959054Z_eurosat_dofa_frozen_bnlinear_seed43_cd258395` | frozen canonical dry；64/2714 partial test | PILOT_ONLY | NO | NO | plumbing only；已被同 seed formal run 覆盖。 |
| `20260807T103145815391Z_eurosat_dofa_frozen_bnlinear_seed44_c126d8c4` | frozen canonical dry；64/2714 partial test | PILOT_ONLY | NO | NO | plumbing only；已被同 seed formal run 覆盖。 |
| `20260807T103233022656Z_eurosat_dofa_frozen_bnlinear_seed42_338e2700` | frozen canonical formal；`best.pt` SHA-256 `7cb285a3e45dcf6acb67c58e474d0b2bfed1aa78ddb6139ecd1dc261c84b54d0`；full test export | REUSE_FINAL | NO | YES | deterministic 无需重评；仅需 calibration inference；可作 ensemble member。 |
| `20260807T104927933179Z_eurosat_dofa_frozen_bnlinear_seed43_3567033c` | frozen canonical formal；`best.pt` SHA-256 `67b3eb2be1b5f7bd2fb280fd975ac0583fc83bc3d4b3bc0eed2fb9f64c0b7960`；full test export | REUSE_FINAL | NO | YES | 同上。 |
| `20260807T110954653187Z_eurosat_dofa_frozen_bnlinear_seed44_36c48b49` | frozen canonical formal；`best.pt` SHA-256 `22e7fa5dae5d6e50606fc2c0e0707db513962efc13d62d6d71b22e2fe506a0b7`；full test export | REUSE_FINAL | NO | YES | 同上。 |
| `20260808T211325587947Z_eurosat_dofa_full_finetune_seed42_b7fa26c9` | full-FT preflight；仅 config/environment，无 checkpoint/metrics | INVALID | NO | NO | 旧 structural-freeze audit false positive 导致中止；已被正式 seed42 supersede，不要重跑该 attempt。 |
| `20260808T211425420318Z_eurosat_dofa_full_finetune_seed42_0a0fae12` | full-FT dry；1 train/val batch、1 epoch、64/2714 partial test | PILOT_ONLY | NO | NO | plumbing/gradient audit only；已被同 seed formal run 覆盖。 |
| `20260808T211435364025Z_eurosat_dofa_full_finetune_seed43_e135c6c4` | full-FT dry；同上 | PILOT_ONLY | NO | NO | plumbing/gradient audit only；已被同 seed formal run 覆盖。 |
| `20260808T211442149575Z_eurosat_dofa_full_finetune_seed44_9daa25ff` | full-FT dry；同上 | PILOT_ONLY | NO | NO | plumbing/gradient audit only；已被同 seed formal run 覆盖。 |
| `20260808T211525501974Z_eurosat_dofa_full_finetune_seed42_1324a006` | repeated full-FT dry；同上 | PILOT_ONLY | NO | NO | plumbing only；已被同 seed formal run 覆盖。 |
| `20260808T211534875922Z_eurosat_dofa_full_finetune_seed43_35d4af7d` | repeated full-FT dry；同上 | PILOT_ONLY | NO | NO | plumbing only；已被同 seed formal run 覆盖。 |
| `20260808T211541588687Z_eurosat_dofa_full_finetune_seed44_b4163d25` | repeated full-FT dry；同上 | PILOT_ONLY | NO | NO | plumbing only；已被同 seed formal run 覆盖。 |
| `20260808T211626550846Z_eurosat_dofa_full_finetune_seed42_466f9f00` | full-FT canonical formal；`best.pt` SHA-256 `dc5cbeaaada0ae763de0b916fa9de099613c0ca495156b4984c09105e7010ae6`；full test export | REUSE_FINAL | NO | YES | deterministic 无需重评；仅需 calibration inference；可作 ensemble member。 |
| `20260808T215707981044Z_eurosat_dofa_full_finetune_seed43_53c8b069` | full-FT canonical formal；`best.pt` SHA-256 `87635a994bd56920a2f0eab0dd1cddde528b2e6a277d57476c7d1dd67eb3d719`；full test export | REUSE_FINAL | NO | YES | 同上。 |
| `20260808T225252226548Z_eurosat_dofa_full_finetune_seed44_a9291cb9` | full-FT canonical formal；`best.pt` SHA-256 `db55348289cad41a30a36365f96585b9dd1d98d877919cacdbc4b65b253fa730`；full test export | REUSE_FINAL | NO | YES | 同上。 |

补充 artifact classification：`results/prediction_exports/_superseded/` 的三个 32-sample export 是 schema/debug 产物，其中 `pre_partial_flag` 错把 partial export 标为 false；均为 `PILOT_ONLY`/superseded，不是实验。`results/prediction_exports/dry_run_checkpoint_val_32_final` 是修正后的 partial-export sanity artifact，仍为 `PILOT_ONLY`。配置中但没有任何运行证据的 `dofa_small_eurosat`、`dofa_base_eurosat`、`dofa_large_eurosat`、alternate-pipeline `eurosat_dofa_rgb`、`dofa_base_segmentation_template`、两份 Panopticon config 和 RS3DBench depth config 均为 `MISSING` run，而不是已完成实验。

## 4. 正式已有结果

以下均直接读取现有 `metrics.json`；mean/std 是对已有三个 seed 文件的审计汇总，不包含新推理。标准差为 sample standard deviation (`ddof=1`)。

| Adaptation | Seeds | Accuracy | Macro-F1 | NLL ↓ | Brier ↓ | ECE-15 ↓ |
|---|---|---:|---:|---:|---:|---:|
| DOFA frozen | 42,43,44 | `0.983419 ± 0.000638` | `0.982342 ± 0.000691` | `0.054390 ± 0.003900` | `0.025999 ± 0.001521` | `0.005008 ± 0.001855` |
| DOFA full fine-tuning | 42,43,44 | `0.964873 ± 0.005369` | `0.963748 ± 0.005491` | `0.106944 ± 0.020335` | `0.053269 ± 0.008176` | `0.011117 ± 0.004898` |

每份 formal test Parquet 都含 2,714 个唯一 sample ID、完整 10-class logits/probabilities、MSP、top1-top2 margin、predictive entropy 与 metadata，且 manifest `partial_export=false`、validation report valid。六个实际 checkpoint SHA-256 与各自 prediction manifest 一致。现有 `reliability.png` 是 raw **validation** reliability diagram；若论文需要 test reliability plot，可直接从 test Parquet 重画，`PLOT_ONLY`，不需要模型推理。

## 5. Classification target matrix

### 5.1 逐单元核查

| Dataset | Model | Adaptation | Valid runs / seeds | Deterministic eval | Temperature Scaling | MC Dropout | Enough ensemble members | Reuse status / next action |
|---|---|---|---|---|---|---|---|---|
| EuroSAT | DOFA | frozen | 3 / 42,43,44 | COMPLETE，完整 test metrics+probs | 无结果；现有 checkpoint/test logits 可复用，只需 calibration inference+fit | 无；正式模型 dropout=0，需新训练 | YES；3 members 且 sample order 一致 | REUSE_FINAL；先做 cheap Temp/ensemble，不重训 baseline |
| EuroSAT | DOFA | full | 3 / 42,43,44 | COMPLETE，完整 test metrics+probs | 同上 | 无；dropout=0，需新训练 | YES | REUSE_FINAL；同上；保留 instability caveat |
| EuroSAT | Panopticon | frozen | 0 | MISSING | 无 checkpoint，待 baseline 后做 | 无 | NO | REUSE_PARTIAL implementation；先验证 preprocessing/official weight load，再训练 |
| EuroSAT | Panopticon | full | 0 | MISSING | 同上 | 无 | NO | REUSE_PARTIAL implementation；同上 |
| So2Sat | DOFA | frozen | 0 | MISSING | 无 | 无 | NO | MISSING；先实现 dataset/split/input metadata |
| So2Sat | DOFA | full | 0 | MISSING | 无 | 无 | NO | MISSING；同上 |
| So2Sat | Panopticon | frozen | 0 | MISSING | 无 | 无 | NO | MISSING；dataset pipeline 缺失，Panopticon wrapper 可部分复用 |
| So2Sat | Panopticon | full | 0 | MISSING | 无 | 无 | NO | MISSING；同上 |

Panopticon 当前实现证据：TorchGeo 0.8.1 提供 `panopticon_vitb14` 与 `Panopticon_Weights.VIT_BASE14`；wrapper 给官方输入 `{"imgs": [B,C,H,W], "chn_ids": [B,C]}`，光学 channel IDs 使用 nm。官方 weight URL 在 TorchGeo enum 中，但仓库和已检查本地缓存没有 Panopticon `.pt/.pth`。当前 tests 只 mock backbone/factory；没有证据证明实际 pretrained forward、optimizer audit、显存和 current EuroSAT normalization 已通过。

### 5.2 Classification master completion table

| Task | Dataset | Model | Adaptation | Training | Deterministic Eval | Temp Scaling | MC Dropout | Ensemble | Status | Required Next Action |
|---|---|---|---|---|---|---|---|---|---|---|
| Classification | EuroSAT | DOFA | frozen | COMPLETE (3 seeds) | COMPLETE | PARTIAL implementation / no result | MISSING | PARTIAL: members ready | PARTIAL overall | calibration inference+temperature fit；aggregate saved probabilities |
| Classification | EuroSAT | DOFA | full | COMPLETE (3 seeds) | COMPLETE | PARTIAL implementation / no result | MISSING | PARTIAL: members ready | PARTIAL overall | 同上；不因 instability 重训 |
| Classification | EuroSAT | Panopticon | frozen | MISSING | MISSING | MISSING | MISSING | MISSING | PARTIAL | actual-backbone smoke/metadata verification，再训练 3 seeds |
| Classification | EuroSAT | Panopticon | full | MISSING | MISSING | MISSING | MISSING | MISSING | PARTIAL | 同上 |
| Classification | So2Sat | DOFA | frozen | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | dataset pipeline + training |
| Classification | So2Sat | DOFA | full | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | dataset pipeline + training |
| Classification | So2Sat | Panopticon | frozen | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | dataset pipeline + integration + training |
| Classification | So2Sat | Panopticon | full | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | dataset pipeline + integration + training |

## 6. Segmentation target matrix

`scripts/run_experiments.py` 对任何 `task == segmentation` 明确抛出 `NotImplementedError`。`configs/experiments.yaml` 只有一个 `status: pending_dataset` 的 DOFA segmentation template；`DOFA/downstream_tasks/README.md` 只列出任务名称，没有本地 decoder/训练代码。现有 DOFA/Panopticon classification wrappers 输出 pooled representations，不提供当前 segmentation pipeline 所需的 dense feature pyramid/token-to-map adapter。

| Dataset | Model | Adaptation | Loader/datamodule | Preprocess/split | Dense adapter | Decoder/head | Training | Deterministic eval + calibration metrics | MC Dropout | Ensemble | Qualitative UQ maps | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CloudSen12 | DOFA | frozen | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| CloudSen12 | DOFA | full | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| CloudSen12 | Panopticon | frozen | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| CloudSen12 | Panopticon | full | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| SpaceNet7 | DOFA | frozen | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| SpaceNet7 | DOFA | full | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| SpaceNet7 | Panopticon | frozen | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| SpaceNet7 | Panopticon | full | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

可复用但不等于 segmentation 已实现的组件：全局 seeding/run-ID/environment/resolved-config/checkpoint patterns；DOFA/Panopticon spectral metadata 转换；classification 的 NLL/Brier/ECE 数学实现可作为 pixelwise 版本的参考；stochastic `[sample,pass/member,class]` schema 可作为新 pixelwise schema 的设计参考。RS3DBench 有 ResNet18 depth decoder/upsampling code，但模型、标签、loss、数据格式和指标均不匹配这两个 segmentation task，只能作为工程参考，不能直接进入结果。

## 7. Dataset implementation audit

| Dataset | DATASET_STATUS | Split/reproducibility | Channels / resolution / labels | Leakage / validity |
|---|---|---|---|---|
| EuroSAT | EXPERIMENTS_EXIST | 固定 70/10/10/10 manifest；generation seed 20260803；27,000 样本 exhaustive、无 ID/path/hash overlap；训练 seeds 不改变 split | 源为 13-band GeoTIFF；正式实验选 B04/B03/B02，resize 224；10-class integer label；无 ignore index | 15,626 spatial groups；控制 byte duplicate、重叠 footprint 和 ≤20 m adjacency，cross-split 均为 0。`source_scene_id_available=false`，更广的同 acquisition/product scene leakage 无法排除，因此不标 `FULLY_VALIDATED`。 |
| So2Sat | NOT_IMPLEMENTED | 无 split/config/seed evidence | channels、normalization、resolution、label format、class count 均未定义 | 无法评估 spatial/site leakage |
| CloudSen12 | NOT_IMPLEMENTED | 无 split/config/seed evidence | channels、resolution、mask class mapping、ignore index 均未定义 | 无法评估 scene/tile leakage |
| SpaceNet7 | NOT_IMPLEMENTED | 无 split/config/seed evidence | channels、temporal handling、resolution、mask mapping、ignore index 均未定义 | 无法评估 city/tile/temporal leakage |

EuroSAT 不标 `FULLY_VALIDATED` 不是发现了 concrete leakage，也不使六个正式 run `INVALID`；它表示项目自身已有的 leakage report 明确承认 source-scene provenance 缺失。当前 spatial split 比仓库中的 official 60/20/20 files 更严格，因为后者存在 414 条 cross-split overlapping-footprint edges 和 10,812 条 cross-split ≤20 m adjacency edges。

## 8. Model implementation audit

| Capability | DOFA | Panopticon |
|---|---|---|
| EuroSAT classification | YES，6 个 formal run 验证 | IMPLEMENTED in code/config；无 actual pretrained run |
| So2Sat classification | NO end-to-end；backbone wavelength list可泛化，但 loader/gate/config 缺失 | NO end-to-end；channel-ID wrapper可复用，但 loader/gate/config 缺失 |
| CloudSen12 segmentation | NO | NO |
| SpaceNet7 segmentation | NO | NO |
| frozen adaptation | YES，artifact-validated | YES in wrapper/config，mock-tested only |
| full fine-tuning | YES，gradient/optimizer artifact-validated | YES in wrapper/config，mock-tested only |
| MC Dropout | Generic collection code exists；formal checkpoints dropout=0 | Generic collection code可调用；current config/official default dropout=0；无 checkpoint |
| logits/probabilities | YES | YES by wrapper contract，未实际运行验证 |
| backbone representations | `extract_features()` + optional exporter；未保存 embeddings | `extract_features()` + optional exporter；未实际运行验证 |
| spectral metadata | 当前 YAML nm→DOFA µm 转换正确；formal checkpoint config未显式记录 wavelength，但训练代码 fallback 常量正确 | wrapper传 nm，符合 TorchGeo Panopticon API；当前 normalization/dynamic-range compatibility 尚未有证据 |

## 9. UQ 与 calibration 全仓库审计

| Method / quantity | Implementation | Existing formal result | Reuse judgment |
|---|---|---|---|
| NLL | classification logits implementation + artifact recomputation | YES，6 formal test runs | REUSE_FINAL |
| multiclass Brier | 每样本对所有 classes squared error 求和再平均；未除 class count | YES | REUSE_FINAL；论文必须写明 convention |
| ECE | equal-width top-label ECE-15 | YES | REUSE_FINAL；不是 adaptive/classwise ECE |
| predictive entropy | deterministic per-sample export + mean | YES | REUSE_FINAL |
| Temperature Scaling | `TemperatureScaling`/`TemperatureScaler` 类存在 | NO | REUSE_PARTIAL implementation；现有路径把 temperature 在 val 上 fit 后又在同一 val 上评分，不能作 final claim；temperature 也未正值参数化 |
| calibration split workflow | loader 有独立 calibration split；checkpoint exporter可导出该 split | NO calibration predictions/results | REUSE_WITH_REEVALUATION 对 6 个 formal checkpoints；只需 inference+fit，不需 training |
| MC Dropout passes | 两套 collector；`prediction_export.py` 可保存全部 passes | NO | REUSE_PARTIAL code；`run_experiments` 的旧 UQ path只保留 mean probability，且 formal dropout=0 |
| expected entropy | 无 | NO | MISSING |
| mutual information | 无 | NO | MISSING |
| probability variance | 无 | NO | MISSING |
| disagreement | 无 | NO | MISSING |
| Deep Ensemble | model collector + stochastic exporter helper存在 | NO aggregate/result | EuroSAT–DOFA 两组为 `REUSE_AS_ENSEMBLE_MEMBER` capability；直接聚合已有 test probabilities 即可 |
| reliability diagrams | raw validation plot exists | YES for validation deterministic；NO for test/UQ | test raw plot为 PLOT_ONLY；UQ plot待对应 probabilities |
| segmentation UQ/metrics | 无 mIoU/per-class IoU/pixel accuracy/pixelwise NLL/Brier/ECE | NO | MISSING |

`train_uqbox.py` 的 Lightning-UQ-Box 名称容易误导：它只包装 `DeterministicClassification`，仓库中没有对应 TensorBoard/checkpoint/output。不能把该脚本或 `configs/experiments.yaml` 中列出的 method 名称当作 UQ 已执行证据。

## 10. HISTORICAL EXPERIMENTS THAT CAN BE SALVAGED

1. **六个 canonical DOFA–EuroSAT formal runs**：原本用于 deterministic frozen/full baselines；当前仍完全有效。它们已经替代任何重训需求，并可进一步替代 Deep Ensemble member training。Temperature Scaling 仅需对其做 calibration inference；ensemble test 甚至可直接从现有 Parquet 聚合。MC Dropout 除外，因为 dropout=0。
2. **Legacy `eurosat_dofa_rgb`**：原本是 frozen DOFA RGB 线性探针。可保留其 20-epoch learning-curve、checkpoint-loading与“DOFA representation transfers well”的 pilot 证据，也证明旧代码能完成训练。它不能替代当前任何 final cell，因为 split/head/provenance不兼容，且 current test 可能与其 old train overlap；追加 inference/metrics无法修复。
3. **Canonical dry runs**：原本用于 frozen/full parameter、gradient、prediction-export preflight。它们可继续作为 plumbing regression evidence，避免未来重复做同类大规模调试；不能作为 ensemble member或最终定量结果。
4. **Auxiliary ResNet18 run**：原本是 1-epoch RGB calibration baseline。可复用其旧 pipeline 和 metric sanity evidence，但模型不在 current matrix、训练预算/split不匹配，不能替代 DOFA/Panopticon cell。
5. **Prediction exporter debug artifacts**：证明 sample ID、Parquet list dtype、partial-export validation 的演进；只可用于 schema regression，不能进入论文数值。

结论：没有任何历史 So2Sat/CloudSen12/SpaceNet7/Panopticon checkpoint 可挽救，因为没有发现这类训练 artifact。RS3DBench 只有数据和未执行的 depth pipeline，不能替代任何当前 thesis experiment。

## 11. Compute-oriented remaining work（合并原 `remaining_work.csv`）

`ACTION_REQUIRED` 和 `COMPUTE_LEVEL` 严格使用用户指定枚举。Priority：P0 立即冻结/复用，P1 低成本高价值，P2 缺失基础设施，P3 新训练或依赖前序工作。

| task | dataset | model | adaptation | missing component | reusable checkpoint if available | ACTION_REQUIRED | COMPUTE_LEVEL | priority | notes |
|---|---|---|---|---|---|---|---|---|---|
| classification | EuroSAT | DOFA | frozen | deterministic baseline | F42/F43/F44 formal best | NO_ACTION | NEGLIGIBLE | P0 | 已完整；禁止重复训练 |
| classification | EuroSAT | DOFA | full | deterministic baseline | T42/T43/T44 formal best | NO_ACTION | NEGLIGIBLE | P0 | 已完整；保留 instability caveat，不重训 |
| classification | EuroSAT | DOFA | frozen | calibration-split logits | F42/F43/F44 | INFERENCE_ONLY | LOW | P1 | test logits 已有；只跑 calibration split |
| classification | EuroSAT | DOFA | full | calibration-split logits | T42/T43/T44 | INFERENCE_ONLY | LOW | P1 | 同上 |
| classification | EuroSAT | DOFA | frozen | Temperature Scaling | calibration logits + existing test logits | TEMPERATURE_FITTING_ONLY | NEGLIGIBLE | P1 | fit 只用 calibration；apply 到保存的 test logits |
| classification | EuroSAT | DOFA | full | Temperature Scaling | calibration logits + existing test logits | TEMPERATURE_FITTING_ONLY | NEGLIGIBLE | P1 | 同上 |
| classification | EuroSAT | DOFA | frozen | 3-member Deep Ensemble | 3 formal test Parquets | ENSEMBLE_AGGREGATION_ONLY | NEGLIGIBLE | P1 | sample order 已验证一致，无模型 inference |
| classification | EuroSAT | DOFA | full | 3-member Deep Ensemble | 3 formal test Parquets | ENSEMBLE_AGGREGATION_ONLY | NEGLIGIBLE | P1 | 同上 |
| classification | EuroSAT | DOFA | frozen/full | raw test reliability plots | 6 formal test Parquets | PLOT_ONLY | NEGLIGIBLE | P2 | 当前 PNG 是 validation plot |
| classification | EuroSAT | DOFA | frozen | MC Dropout model/results | none with active dropout | TRAINING_REQUIRED | MEDIUM | P3 | 需预先固定 dropout placement/p/passes；旧 checkpoint 不可伪装成 MC model |
| classification | EuroSAT | DOFA | full | MC Dropout model/results | none with active dropout | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| classification | EuroSAT | Panopticon | frozen/full | actual pretrained forward、normalization和model audit | official factory/config only | SHORT_SMOKE_TEST | LOW | P1 | 先验证再投 GPU 正式训练 |
| classification | EuroSAT | Panopticon | frozen | deterministic 3 seeds | none | TRAINING_REQUIRED | MEDIUM | P2 | 三个 seeds 同时可作 ensemble members |
| classification | EuroSAT | Panopticon | full | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P2 | 同上 |
| classification | EuroSAT | Panopticon | frozen | MC Dropout model/results | none | TRAINING_REQUIRED | MEDIUM | P3 | current dropout=0 |
| classification | EuroSAT | Panopticon | full | MC Dropout model/results | none | TRAINING_REQUIRED | HIGH | P3 | current dropout=0 |
| classification | EuroSAT | Panopticon | frozen/full | Temp Scaling after baselines | future deterministic checkpoints | TEMPERATURE_FITTING_ONLY | LOW | P3 | 还需 cheap calibration inference |
| classification | EuroSAT | Panopticon | frozen/full | ensemble after 3 seeds | future deterministic checkpoints | ENSEMBLE_AGGREGATION_ONLY | LOW | P3 | 不应另训“ensemble-only”成员 |
| classification | So2Sat | DOFA/Panopticon | frozen/full | loader、split、preprocess、class mapping、spectral metadata、leakage checks | generic classification infrastructure | IMPLEMENTATION_ONLY | MEDIUM | P2 | 当前完全无 dataset-specific implementation |
| classification | So2Sat | DOFA | frozen | deterministic 3 seeds | none | TRAINING_REQUIRED | MEDIUM | P3 | 在 pipeline 验证后训练 |
| classification | So2Sat | DOFA | full | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| classification | So2Sat | Panopticon | frozen | deterministic 3 seeds | none | TRAINING_REQUIRED | MEDIUM | P3 | 同上 |
| classification | So2Sat | Panopticon | full | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| classification | So2Sat | DOFA | frozen/full | MC Dropout models | none | TRAINING_REQUIRED | HIGH | P3 | 独立于 deterministic checkpoint；可按 adaptation 分批 |
| classification | So2Sat | Panopticon | frozen/full | MC Dropout models | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| classification | So2Sat | DOFA/Panopticon | frozen/full | Temperature Scaling | future deterministic checkpoints | TEMPERATURE_FITTING_ONLY | LOW | P3 | 无额外模型训练；需 calibration inference |
| classification | So2Sat | DOFA/Panopticon | frozen/full | Deep Ensemble | future 3-seed deterministic outputs | ENSEMBLE_AGGREGATION_ONLY | LOW | P3 | deterministic seeds 应直接复用 |
| segmentation | CloudSen12 | DOFA/Panopticon | frozen/full | loader、split、preprocess、mask mapping、ignore index、leakage checks | generic run manager only | IMPLEMENTATION_ONLY | MEDIUM | P2 | dataset-specific work不存在 |
| segmentation | SpaceNet7 | DOFA/Panopticon | frozen/full | loader、split、temporal/preprocess、mask mapping、ignore index、leakage checks | generic run manager only | IMPLEMENTATION_ONLY | MEDIUM | P2 | dataset-specific work不存在 |
| segmentation | CloudSen12/SpaceNet7 | DOFA | frozen/full | dense feature adapter + decoder contract | DOFA pretrained backbone only | IMPLEMENTATION_ONLY | MEDIUM | P2 | current wrapper只输出 pooled features |
| segmentation | CloudSen12/SpaceNet7 | Panopticon | frozen/full | dense feature adapter + decoder contract | Panopticon factory only | IMPLEMENTATION_ONLY | MEDIUM | P2 | current wrapper只输出 pooled features |
| segmentation | CloudSen12/SpaceNet7 | both | both | mIoU/per-class IoU/pixel accuracy/NLL/Brier/ECE、pixelwise export | classification metric/export code as reference | IMPLEMENTATION_ONLY | MEDIUM | P2 | 需处理 ignore index、memory和aggregation |
| segmentation | CloudSen12 | DOFA | frozen | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | pipeline/decoder 验证后 |
| segmentation | CloudSen12 | DOFA | full | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| segmentation | CloudSen12 | Panopticon | frozen | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| segmentation | CloudSen12 | Panopticon | full | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| segmentation | SpaceNet7 | DOFA | frozen | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| segmentation | SpaceNet7 | DOFA | full | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| segmentation | SpaceNet7 | Panopticon | frozen | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| segmentation | SpaceNet7 | Panopticon | full | deterministic 3 seeds | none | TRAINING_REQUIRED | HIGH | P3 | 同上 |
| segmentation | both | both | frozen/full | MC Dropout models and pass export | none | TRAINING_REQUIRED | HIGH | P3 | 8 个 adaptation cells 均无 dropout-enabled checkpoint |
| segmentation | both | both | frozen/full | Deep Ensemble aggregation | future 3-seed deterministic outputs | ENSEMBLE_AGGREGATION_ONLY | LOW | P3 | 不增加独立训练；复用 deterministic seeds |
| segmentation | both | both | frozen/full | expected entropy/MI/variance + qualitative maps | future stochastic outputs | METRICS_ONLY | LOW | P3 | maps 本身不是 blocker；先有正确 pixel outputs |

缩写：F42/F43/F44 分别是三条 frozen formal `best.pt`；T42/T43/T44 分别是三条 full-fine-tune formal `best.pt`，精确路径见第 3 节。表中没有任何 `RETRAINING_REQUIRED`：没有现有 formal run 被证明 scientifically invalid；所有真正缺失的 cell 是首次 `TRAINING_REQUIRED`。Dry/pilot artifacts 不值得重训，因为它们对应的 DOFA–EuroSAT deterministic cells已有更好的 formal replacements。

## 12. True blockers（合并原 `current_blockers.md`）

### Scientific blockers

1. **So2Sat/CloudSen12/SpaceNet7 protocol 未定义**：split unit、channels、normalization、resolution、class mapping、segmentation ignore index 和 leakage controls 均缺；这些必须在训练前固定。
2. **Panopticon input normalization/dynamic range 未验证**：nm channel IDs 已正确实现，但 current transform直接复用 DOFA 的 Sentinel statistics；TorchGeo weight enum 的 transform 为 Identity，仓库没有实际 pretrained run证明该选择与上游预训练约定一致。
3. **MC Dropout protocol 未定义且无 active-dropout checkpoint**：dropout placement、rate、训练时启用方式、passes和 uncertainty decomposition 必须预先固定。现有 checkpoint 不能只靠 inference 时改 `p` 变成科学上等价的 MC Dropout。
4. **EuroSAT broader same-source-scene leakage 无法最终排除**：当前 split 已控制重叠/20 m adjacency，但原 acquisition/product ID 不存在。是需披露/决定的 verification risk，不是已有 concrete leakage 或 `INVALID` 证据。
5. **DOFA full-FT post-best instability**：必须透明报告，并始终使用 validation-selected best。它不是重训 blocker，除非监督方事先要求采用一套新的稳定训练协议。

### Engineering blockers

1. So2Sat loader/datamodule 和 end-to-end routing 缺失。
2. CloudSen12/SpaceNet7 loaders、preprocessing、mask mapping 与 split manifests 缺失。
3. DOFA/Panopticon dense segmentation adapters 和统一 decoder/head 缺失。
4. Segmentation metrics、pixelwise probability export、stochastic aggregation和 qualitative uncertainty maps 缺失。
5. Proper Temperature Scaling workflow 缺失：当前函数在 validation 同时 fit/evaluate，且 temperature 无正值约束；需要 calibration-only fit + frozen test apply。
6. Deep Ensemble 有 collector/schema但无 checkpoint/Parquet aggregator、compatibility manifest和正式 result writer。
7. MC Dropout 的 raw-pass collector/exporter未接入 checkpoint CLI；expected entropy、MI、variance、disagreement 未实现。
8. Active Git metadata 损坏，正式 run 无 commit hash；影响 exact code provenance。

### Compute blockers

1. Panopticon–EuroSAT frozen/full 没有任何 downstream checkpoint，正式 deterministic cell 需要首次训练。
2. So2Sat 的四个 model×adaptation deterministic cells 均没有 checkpoint，pipeline 完成后需要首次训练。
3. 八个 segmentation deterministic cells 均没有 checkpoint，implementation 完成后需要首次训练。
4. 全部 16 个 MC Dropout cells 均没有 dropout-enabled checkpoint，需要相应训练；EuroSAT–DOFA 的现有 deterministic checkpoint不能替代。

明确不是 compute blocker：缺 test reliability plot；缺 NLL/Brier/ECE（正式 DOFA 已有）；DOFA–EuroSAT Temperature Scaling；DOFA–EuroSAT Deep Ensemble。它们只需要 plot、cheap inference/fitting或已有 predictions 的 aggregation。

## 13. Executive summary

### 1. ALREADY COMPLETE

- EuroSAT–DOFA–frozen deterministic：seeds 42/43/44，validation-NLL best selection、完整 test predictions 和全部主指标。
- EuroSAT–DOFA–full fine-tuning deterministic：seeds 42/43/44，同样完整；best checkpoints 可用，post-best instability 已有 artifact 记录。
- EuroSAT 固定 70/10/10/10 manifest、calibration split、class balance、hash/overlap/≤20 m spatial-group checks和可复现 seed。
- 通用 classification run management、seeding、resolved config、environment、checkpoint、per-sample deterministic export和 metric recomputation。

### 2. COMPLETE BUT NEEDS CHEAP RE-EVALUATION

- 六个 DOFA–EuroSAT formal best checkpoints：deterministic 本身无需重评；为了 Temperature Scaling，只需各自导出 calibration logits并拟合 temperature，然后应用到已有 test logits。
- 两组三成员 Deep Ensemble：成员训练已完成；现有 test Parquet 可直接聚合，无需模型 inference。
- Raw test reliability diagrams：直接从现有 probabilities 重画即可。

### 3. PARTIALLY COMPLETE

- Panopticon–EuroSAT classification：wrapper、factory、nm metadata、frozen/full configs和 mock tests 已有；actual pretrained smoke与训练缺失。
- Temperature Scaling：数学类存在，但 final split workflow和正值约束不合格，正式结果缺失。
- MC Dropout：collector/stochastic schema存在，但未接 end-to-end，且没有 active-dropout checkpoint。
- Deep Ensemble：collector/export schema存在；DOFA–EuroSAT members已齐，只缺 aggregation/result artifact。
- Backbone embedding extraction：DOFA/Panopticon hook和 exporter option存在，但没有保存的 embedding artifact。
- Segmentation 可复用通用 run infrastructure，但所有 task-specific data/model/metric components 缺失。

### 4. GENUINELY MISSING

- So2Sat 全部 dataset pipeline、四个 deterministic cells及全部 UQ results。
- CloudSen12 和 SpaceNet7 loaders/splits/preprocessing/mask definitions。
- DOFA/Panopticon segmentation dense adapters、decoders、8 个 deterministic cells。
- 全部分割 calibration metrics、MC Dropout、Deep Ensemble和 qualitative uncertainty maps。
- Expected entropy、mutual information、probability variance和disagreement metrics。
- 任何正式 Temperature Scaling、MC Dropout 或 Deep Ensemble result artifact。

### 5. NEW TRAINING ACTUALLY REQUIRED

- **14 个尚无 deterministic checkpoint 的 cells**：EuroSAT–Panopticon 2 个、So2Sat classification 4 个、segmentation 8 个。原因都是没有现有可复用 downstream checkpoint；在相应 implementation/verification 完成后首次训练。每个 cell 的多 seed runs应同时作为 Deep Ensemble members，避免另训一套 ensemble。
- **16 个 MC Dropout cells**：当前没有任何训练时启用 non-zero dropout 的 checkpoint。EuroSAT–DOFA 也在内；其 deterministic weights不能通过只在 inference 时改 dropout rate获得标准 MC Dropout validity。
- 不需要新训练：EuroSAT–DOFA 两个 deterministic cells、它们的 Temperature Scaling、它们的 Deep Ensembles、以及任何可由现有 Parquet得到的 metrics/plots。
- 当前没有 `RETRAINING_REQUIRED` 项；没有 concrete evidence 使六个 formal checkpoints失效。

### 6. ESTIMATED CURRENT THESIS COMPLETION

- **experiment infrastructure：约 55–60%**。Classification 的复现、checkpoint和export较成熟，Panopticon分类接入已写；proper UQ orchestration、第二分类数据集和完整 segmentation stack 尚缺。
- **classification deterministic experiments：25% cells complete（2/8）**，但已完成的两个 cell质量高且各有 3 seeds。
- **classification UQ experiments：0% formal results；约 30% enabling code**。没有任何 Temp/MC/ensemble final artifact；DOFA ensemble和Temp可快速补齐。
- **segmentation deterministic experiments：0% results；task-specific infrastructure接近 0%**。
- **segmentation UQ experiments：0%**。
- **overall supervisor-required experimental work：约 8–12%**。依据是 56 个方法单元（8 classification×4 methods + 8 segmentation×3 methods）中仅两个 deterministic 单元完成；现有高质量 infrastructure和可零训练补齐的两个 DOFA ensemble/Temp路径使工程完成度高于 raw cell ratio。

### 7. RECOMMENDED NEXT 5 ACTIONS

1. 将六个 formal DOFA–EuroSAT `best.pt`、test Parquet和 hashes登记为冻结的 final baseline/ensemble members；不要重训或使用 `last.pt`。
2. 直接从两组三份现有 test Parquet 做 probability-mean Deep Ensemble aggregation并生成指标；不加载模型。
3. 对六个 best checkpoint只运行 calibration-split inference，修正为 calibration-only positive-temperature fitting，再应用到已有 test logits并作 reliability plots。
4. 在任何新训练前冻结三项 scientific protocol：Panopticon normalization/dynamic range、MC Dropout placement/rate/passes、三个新数据集的 split/class/ignore-index/leakage rules。
5. 实现并做短 smoke verification：So2Sat pipeline、CloudSen12/SpaceNet7 pipelines、DOFA/Panopticon dense adapters和 segmentation metrics；全部通过后才排队执行真正缺失的 GPU training cells。
