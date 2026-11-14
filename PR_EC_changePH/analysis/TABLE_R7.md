# W6 / Q1：Table R7 只读统计分析

## 已运行的结果

最终结果目录：`PR_EC_changePH/outputs/rebuttal_w6/table_r7_results/`。
该目录中的 `summary.json` 保存精确数值、输入来源、配置、软件版本和运行前后 SHA-256。
没有训练 Initial EC、MPR、QSR 或分类器，没有覆盖任何输入或旧实验结果。

| EC representation | ABIDE-I mean absolute g | ABIDE-II mean absolute g | Replicated edges | ASD-enhanced | ASD-reduced |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial EC | 0.09157544 | 0.09927051 | 202 | 101 | 101 |
| Full PR-EC (Original PH) | 0.09158654 | 0.10220628 | 213 | 111 | 102 |
| Full PR-EC (Age/Sex/Site) | 0.09158292 | 0.10211608 | 210 | 109 | 101 |

Original PH 实际复算得到 213/111/102，与论文计数一致。与项目已存的论文复现连接 CSV
比较，213 条连接的索引集合及双队列方向也完全一致。历史 CSV 的 ABIDE-I 个别 g 与本次
seed42 EC 复算存在小差异（重叠连接上最大绝对差 0.0003295132）；ABIDE-II 为零。
本表没有直接套用历史 g、q 或论文计数，统一使用当前所选 EC 的实际统计值。

Age/Sex/Site 保留 Original PH 的 210/213 条复现连接（98.59%），210 条均保持原方向。
没有新增的跨队列复现连接。若不要求显著性，原 213 条在两个队列均保持原方向。
未保留的 3 条不是方向反转，而是 ABIDE-II 的 q 略高于 0.05：

| Stored source -> target (0-based) | Direction | ABIDE-II q, Age/Sex/Site |
| --- | --- | ---: |
| 54 -> 70 | ASD-enhanced | 0.05192969 |
| 64 -> 27 | ASD-reduced | 0.05024990 |
| 70 -> 84 | ASD-enhanced | 0.05217740 |

相对于 Initial EC，Original PH 的 mean absolute g 在 ABIDE-I 增加 0.00001110
（0.0121%），ABIDE-II 增加 0.00293577（2.9573%）；Age/Sex/Site 分别增加
0.00000749（0.0082%）和 0.00284557（2.8665%）。未对这些描述性变化另作显著性检验。

## 输入与身份审计

- Initial 和 Original PH：`PR_EC/outputs/seed42/<dataset>/`。
- Age/Sex/Site：`PR_EC_changePH/outputs/rebuttal_w6/<dataset>/age_sex_site/`。
- W7 `PR_EC_E2E/outputs/rebuttal_w7` 不属于 PH 消融，本次不使用。
- ABIDE-I：871 人，ASD 403，TC 468；ABIDE-II：975 人，ASD 462，TC 513。
- 每队列三种表示的 subject IDs、labels、sites 完全相同，未取交集或删人；labels 逐人核对
  项目清洗后的 phenotype CSV，沿用本项目 DX_GROUP=2 -> ASD=1、DX_GROUP=1 -> TC=0。
- 两种 refined archive 内嵌的 original_ec 与选定 Initial EC 逐值相同，测试 OOF fold IDs
  逐人相同。仅使用测试 OOF QSR EC，不混用 PGR 或训练 EC。
- Initial archive 没有显式 ROI 名称；refined archive 使用 ROI_001..ROI_090 占位名。
  本次通过共同的前 90 ROI 预处理约定、内嵌 original_ec 逐值比对及占位顺序核验连接索引，
  不将占位名冒充独立解剖学标注或独立的物理 ROI 顺序验证。
- 使用保存矩阵的 `[source_index,target_index]`，与原论文统计脚本一致；不再转置或对称化。
  Initial EC 提取代码在保存前已将窗口平均 attention 转置。
- 历史 Original summary 的路径仍指向迁入 seed42 前的目录；本次明确使用 seed42 文件，
  不转向读取这些旧路径的现有文件。配置与此缓存迁移情况保存在审计中。

## 方法

复用 `PR_EC_changePH/analysis/cross_cohort_replication/run_cross_cohort_replication.py`
的 `compute_effects_and_tests`、`edge_table`、`merge_tables`。每个队列、每种表示分别对
全部 8,010 条非对角有向连接作双侧 Welch t 检验；Hedges g 为 ASD-minus-TC 均值差除以
合并组内 SD，再乘原实现的精确 gamma 小样本校正。BH-FDR 分别作用于各自的 8,010 条边。
复现要求两个队列均 q < 0.05 且非零同号；g > 0 为 ASD-enhanced，g < 0 为 ASD-reduced。
mean absolute g 使用全部非对角有向边，不只显著边。本次不增加协变量调整、Top-K 排名筛选、
置换或其他阈值。另用 SciPy `false_discovery_control` 独立核对原实现的 BH q。

## 输出

- `table_r7.csv`：最终六列结果。
- `summary.json`：完整审计、精确统计、213 条核验和重叠统计。
- `cohort_summary.csv`：逐队列样本数、mean absolute g 及显著连接数。
- `subject_alignment.csv`：逐人诊断及两种 refined EC 的测试折号。
- `original_edges_age_sex_site_retention.csv`：原 213 条逐边保留、方向和 q。
- `initial/`、`original_ph/`、`age_sex_site/`：各有两队列逐边 CSV（g、p、q、Welch t、组均值）、
  `all_edges_cross_cohort.csv`、`replicated_edges.csv`。

统计脚本只写入指定的新空目录，拒绝覆盖已有分析。需要再次复算时换一个输出目录：

```bash
PYTHONDONTWRITEBYTECODE=1 /storage/users/liulin/miniconda3/envs/default/bin/python -B \
  PR_EC_changePH/analysis/analyze_table_r7.py \
  --output-dir PR_EC_changePH/outputs/rebuttal_w6/table_r7_recheck
```

轻量测试：

```bash
PYTHONDONTWRITEBYTECODE=1 /storage/users/liulin/miniconda3/envs/default/bin/python -B \
  -m unittest discover -s PR_EC_changePH/tests -p test_table_r7.py -v
```

## 可用于 rebuttal 的客观表述

在相同被试及统一的 Welch 检验、Hedges g 和队列内 BH-FDR 标准下，Original PH 的 Full
PR-EC 复现了论文报告的 213 条跨队列连接（111 条 ASD-enhanced、102 条 ASD-reduced）。
将 PH 替换为 Age/Sex/Site 后得到 210 条复现连接，其中 210 条均属于原始集合且保持相同
效应方向，保留率为 98.59%；其余 3 条方向未变，仅 ABIDE-II 的 q 超过阈值。
相较 Initial EC，两种精炼配置的全边 mean absolute g 在 ABIDE-I 基本不变、在 ABIDE-II
小幅增加。上述结果描述了组间统计差异及跨队列复现的稳定性，不证明等效性，也不表示
真实因果连接估计更准确；本次仍为未调整协变量的原论文统计流程。
