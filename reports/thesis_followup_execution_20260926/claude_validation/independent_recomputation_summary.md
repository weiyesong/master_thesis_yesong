# Claude 独立抽算记录（会话 5afb33d9-4dac-4ed1-8d17-2e1ec2a65b13，2026-09-27）

独立实现：numpy/scipy/sklearn；未调用协调者 RankPlan/decomposition 作为算法。临时解压数组已删除。

## 分类 MC：treesatai__dofa__frozen__mc_dropout__42
- raw [2000,30,15] 与 summary 同序；mean vs mean_probabilities 2.98e-8
- 逐标签 TU/EE/MI vs saved：2.98e-8 / 2.98e-8 / 3.5e-9；标签和 TU/EE/MI：2.38e-7 / 2.38e-7 / 1.4e-8
- variance ddof0 vs saved 1.79e-9；ddof1 差 2.0e-3（确认总体方差）
- 对象 derived score_expected_entropy/MI/gini_disagreement 与我方差 0；saved 熵和/15 = 图像均值分数（3.3e-8）
- micro 2888 错误：MSP AUROC 0.81875486 (sklearn) vs 0.81875488；MI AUROC 0.79019844 一致；AP 一致到 1e-8

## 分割 MC：cloudsen12__dofa__frozen__mc_dropout__42，子集图 ROI_05830__20200611T085559_20200611T091323_T34NBM
- raw 30 draw 均值 vs 全图保存概率 2.98e-8；TU/EE/MI vs 保存 5.96e-8/5.96e-8/1.8e-9
- 逐类 variance ddof0 vs 保存 2.3e-10（ddof1 差 2.7e-4）；Gini TU−Σvar = raw EE（4.4e-16）
- per_image：EE 0.2369028、MI 0.0080501、GiniEU 0.00118657、GiniEE 0.1210146、err 0.0544085 均一致
- 像素 MSP AUROC 0.94111307 vs 0.94111308；MI AUROC 0.65540134 一致；2730 错误

## Off：eurosat__dofa__frozen__mc_dropout_off__42
- 重算 acc 0.9848931466 / NLL 0.0504143581 / Brier 0.0241687674 / ECE15 0.0043391945 = 三方表 Off_* 与 done.json
- softmax(logits) vs 保存概率 1.96e-7；metrics_and_provenance.json：same_checkpoint_as_mc=True，checkpoint sha ec70e798…，与三方表一致

## SpaceNet7 建筑物保留：spacenet7__dofa__frozen__deterministic__42
- 我方 fractional 接纳 576=0.5×1152 图，无切点 tie；保留建筑物像素 580530 / 4027508 = 0.144141，与表一致
- 对照量：接纳集内建筑物 recall 0.0345，全集 0.0551 —— 保留量是内容量而非 recall
- 源标签核对两图 support：sample_4152 10131、sample_4752 1022，与 per_image_class 一致
- risk@0.5 图像等权 0.0218396（本数据集有效像素均为 50176，等权=像素加权）
- 16 对象保留范围 0.117697–0.188230，与报告 11.77%–18.82% 一致

## 配对范围
- pixel_paired MC−Off cloud dofa frozen msp auroc：共同有效 26/32，均差 0.000836，我方 CI [-1.3e-5, 2.14e-3] vs [-2.8e-5, 2.23e-3]
- paired_effects 分割 contrasts 无 DE_minus_mean_D_members；DE 三成员对照仅用主表指标（DE_versus_three_member_core_metrics.csv）
- 资源：done.json total_seconds 之和 16.51 min，峰值 RSS 3.67 GiB，与 resource_summary.json 一致
