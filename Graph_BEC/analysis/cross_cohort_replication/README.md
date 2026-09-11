# Cross-cohort replication analysis

这个实验不再用 ABIDE-I/ABIDE-II 内部的效应量排名定义核心连接。Top-k
排名只作为结果展示信息；核心集合由两个独立队列中的统计复现决定。

## 核心定义

对每一条相同的有向连接 `i -> j`，在 ABIDE-I 和 ABIDE-II 中分别完成 ASD vs HC
分析，并在每个队列内部对全部非对角有向边进行 Benjamini–Hochberg FDR 校正。
当且仅当满足以下三个条件时，该连接进入 replicated ASD-related BEC set：

```text
q_I  < 0.05
q_II < 0.05
sign(g_I) == sign(g_II)
```

其中 `g` 是 ASD − HC 的 Hedges' g：

- `g_I > 0` 且 `g_II > 0`：`replicated ASD-enhanced BEC`；
- `g_I < 0` 且 `g_II < 0`：`replicated ASD-reduced BEC`。

因此，一条连接即使在 ABIDE-I 排名第 17、在 ABIDE-II 排名第 35，只要两个队列都
显著且方向一致，仍然进入 replicated set。相反，单个队列 Rank 1 但另一个队列
不显著或方向相反，不能进入核心集合。

## 分析流程

```text
ABIDE-I: ASD vs HC -> g_I, p_I -> q_I
ABIDE-II: ASD vs HC -> g_II, p_II -> q_II
              ↓
      按相同 (source -> target) 合并
              ↓
      q_I < .05 且 q_II < .05
              ↓
       sign(g_I) == sign(g_II)
              ↓
   replicated enhanced / replicated reduced
```

`ABIDE-I Rank` 和 `ABIDE-II Rank` 保存在结果中，仅用于描述排名受队列影响的情况，
不参与筛选。复现集合内部另外提供 `ReplicatedRank`，按两队列平均绝对 Hedges' g
排序，用于选择图中展示的前若干条边；论文结论应基于完整 replicated set。

## 运行

在仓库根目录执行：

```bash
python3 Graph_BEC/analysis/cross_cohort_replication/run_cross_cohort_replication.py
```

可切换效应量数组：

```bash
python3 Graph_BEC/analysis/cross_cohort_replication/run_cross_cohort_replication.py \
  --bec-key original_bec --alpha 0.05
```

## 输出

- `outputs/all_edges_cross_cohort.csv`：全部相同有向边的合并统计表；
- `outputs/replicated_edges.csv`：满足双队列显著且方向一致的完整复现集合；
- `outputs/replication_summary.csv`：样本数、单队列显著边数和复现边总数。

## RQ1：Whole-brain reproducibility

若问题是“ABIDE-I 与 ABIDE-II 的全部 8010 条 BEC disease effects 是否整体一致”，
可运行：

```bash
python3 Graph_BEC/analysis/cross_cohort_replication/summarize_rq1.py
```

脚本从 `all_edges_cross_cohort.csv` 计算：

- 两队列 signed Hedges' g 的 Pearson/Spearman 相关；
- 8010 条边的方向一致率；
- 两队列同时 `q < 0.05` 的边数及其中方向一致的边数；
- ABIDE-I/II 内部 rank 的相关性，作为描述性指标。

输出：

- `outputs/rq1_whole_brain_summary.json`；
- `outputs/rq1_whole_brain_report.md`。

当前 `bec` 结果显示 Pearson `r = 0.373`、Spearman `rho = 0.351`，全边方向一致率
为 `61.0%`；168/8010 条边同时满足双队列 FDR 显著且方向一致。因此结论应表述为：
两个队列存在**阳性但中等程度的整体效应模式一致性**，但不是所有边的普遍复现；
可复现性是选择性的。两队列 rank 相关仅为 `rho = 0.099`，支持“不用 rank 定义核心异常”。

建议后续画图时从 `replicated_edges.csv` 中按 `ReplicatedRank` 选前 10 条，或按
方向分别选取；但图中展示的 Top-10 不能替代论文对完整 replicated set 的报告。

## Methods wording

> To identify robust ASD-related BEC alterations, we prioritized cross-cohort
> reproducibility rather than within-cohort effect-size ranking. A directed
> connection was considered cross-cohort replicated when it showed a
> statistically significant ASD–HC difference in both ABIDE-I and ABIDE-II and
> exhibited the same direction of effect across the two cohorts. Connections
> showing ASD > HC in both cohorts were classified as replicated ASD-enhanced
> BECs, whereas those showing ASD < HC in both cohorts were classified as
> replicated ASD-reduced BECs.
