# PR_EC 群体差异分析

当前 `analysis` 只保留两个核心描述性分析脚本。两者都使用同一套标签和差值定义：

```text
label 0 = TC
label 1 = ASD
Difference = ASD_mean - TC_mean
```

因此：

- `Difference > 0`：ASD 中该连接增强；
- `Difference < 0`：TC 中该连接增强，也可表述为 ASD 中相对减弱。

边差异排序同时报告 Welch 组间检验及全边 Benjamini–Hochberg FDR q 值；ROI 和弦图仍是
基于差异分数的描述性 candidate ROIs/edges，不应单独解读为显著性结论。

## 1. Top-10 有向连接边

脚本：

```bash
python PR_EC/analysis/group_edge_difference.py
```

默认输入是 ABIDE II 模型阶段生成的
`PR_EC/outputs/abide-ii/abide_ii_qsr_refined_subject_ec.npz`，默认分析
`ec`，输出为：

```text
PR_EC/analysis/outputs/group_edge_difference/top_edges_abide_ii.csv
```

默认输出 20 行：10 条 `ASD_enhanced` 和 10 条 `TC_enhanced`。表格包括
`Source`、`Target`、`ASD_mean`、`TC_mean`、`Difference`、`PValue` 和 `FDR_q`
等字段，并保留有向边方向。`PValue` 使用 Welch 两独立样本 t 检验，`FDR_q`
使用 Benjamini–Hochberg 方法在全部 90×89 条非对角有向边上校正。

若要生成截图风格的 Top-5 ABIDE II 表：

```bash
python3 PR_EC/analysis/group_edge_difference.py --top-k 5
```

ABIDE I 仍可显式指定输入文件运行：

```bash
python3 PR_EC/analysis/group_edge_difference.py \
  --ec-path PR_EC/outputs/abide-i/abide_refined_subject_ec.npz \
  --output-dir PR_EC/analysis/outputs/group_edge_difference/abide_i \
  --output-name top_edges_abide_i.csv
```

如需分析 QC 弱监督结果：

```bash
python PR_EC/analysis/group_edge_difference.py --ec-key qc_refined_ec
```

## 2. Top-10 差异 ROI

脚本：

```bash
python PR_EC/analysis/group_roi_difference.py
```

输出为：

```text
PR_EC/analysis/outputs/group_roi_difference/top_rois_asd_vs_tc.csv
```

每个 ROI 的排名指标是其所有入边和出边的平均绝对差异：

```text
TotalDifferenceScore(ROI) = mean(abs(Difference))
```

表格同时报告 ASD 增强分数、TC 增强分数、净方向、以及该 ROI 相关的最大绝对差异有向边。因此一个 ROI 可以是 mixed：它可能同时包含两个方向的差异。

QC 弱监督 ROI 分析：

```bash
python PR_EC/analysis/group_roi_difference.py --ec-key qc_refined_ec
```

## 3. Top-K ROI 有向弦图

脚本：

```bash
python PR_EC/analysis/chord_diagram.py
```

输出为：

```text
PR_EC/analysis/outputs/group_chord/chord_top10_rois_asd_tc.png
```

一张图内并排两个面板，共用同一套 top-K ROI（与 `group_roi_difference.py` 相同的
`TotalDifferenceScore` 排名，默认 K=10，可用 `--top-k` 调整）：

- **左：增强面板（ASD-enhanced）**。每个 ROI 一条红色有向弦连向 ASD 组带，
  弦宽 ∝ 该 ROI 的 `ASDEnhancedScore`；
- **右：减弱面板（TC-enhanced）**。每个 ROI 一条蓝色有向弦连向 TC 组带，
  弦宽 ∝ 该 ROI 的 `TCEnhancedScore`。

两个面板共用同一个弦宽比例尺（按两面板所有得分的最大值归一化），因此
跨面板可以直接比较强弱。箭头方向统一为 ROI → 组带，表示该 ROI 的连接差异
偏向哪个组。每个 ROI 的两个得分即
`group_roi_difference/top_rois_asd_vs_tc.csv` 中的 `ASDEnhancedScore` /
`TCEnhancedScore` 列。

默认输入为 `PR_EC/outputs/pgr_ec_refined_subject_ec.npz`（键 `ec`），
可用 `--ec-key refined_ec` 或 `--ec-key original_ec` 切换（`refined_ec`
与 `ec` 数值相同）。

## 4. 解释限制

Top-10 结果是描述性候选边/候选 ROI，不是统计显著结果。后续若需要“显著增强/显著减弱”，还要另行加入组间检验、置换检验和多重比较校正。

## 5. ABIDE-I/II 跨队列复现分析

固定 Top-k 不用于定义稳定异常。对 ABIDE-I 和 ABIDE-II 中相同的有向边分别完成
ASD vs HC 的 Welch 检验和全边 Benjamini–Hochberg FDR 校正，再按以下规则定义
replicated ASD-related EC：

```text
q_ABIDE-I < 0.05
q_ABIDE-II < 0.05
sign(g_ABIDE-I) == sign(g_ABIDE-II)
```

运行：

```bash
python PR_EC/analysis/cross_cohort_replication/run_cross_cohort_replication.py
```

详细方法和结果见 `PR_EC/analysis/cross_cohort_replication/README.md`。
其中单队列 Rank 只保留作描述，复现集合按平均绝对 Hedges' g 提供展示排序，
不参与筛选。

## 6. Multivariate residual QC association

`multivariate_qc_r2.py` measures how much of each directed EC edge is explained by the
three QC variables jointly:

```text
A[i, e] = beta0 + beta1*FD[i] + beta2*DVARS[i] + beta3*Quality[i] + epsilon[i]
```

For each edge, the script fits ordinary least squares and computes its edge-wise
`R^2`. The reported connectome-level metric is the mean over off-diagonal directed
edges:

```text
Q_R2(eta) = mean_e R2_e(eta)
```

The script expects one NPZ archive per `eta`. Since the current archive writer does
not store `eta`, pass the mapping explicitly using repeated `--archive ETA=PATH`
arguments. All archives must contain the same subjects in the same order.

Example for ABIDE-I:

```bash
python PR_EC/analysis/multivariate_qc_r2.py \
  --archive 0.00=PR_EC/outputs/abide-i/qsr_eta_0.00.npz \
  --archive 0.05=PR_EC/outputs/abide-i/qsr_eta_0.05.npz \
  --archive 0.10=PR_EC/outputs/abide-i/qsr_eta_0.10.npz \
  --archive 0.15=PR_EC/outputs/abide-i/qsr_eta_0.15.npz \
  --archive 0.25=PR_EC/outputs/abide-i/qsr_eta_0.25.npz \
  --archive 0.50=PR_EC/outputs/abide-i/qsr_eta_0.50.npz \
  --archive 0.75=PR_EC/outputs/abide-i/qsr_eta_0.75.npz \
  --archive 1.00=PR_EC/outputs/abide-i/qsr_eta_1.00.npz \
  --phenotype-csv dataset/ABIDE-I/Phenotypic_Processing.csv \
  --subject-id-column FILE_ID \
  --ec-key qc_refined_ec \
  --output-dir PR_EC/analysis/outputs/multivariate_qc_r2/abide_i \
  --plot
```

For ADHD-200, use its phenotype identifier and the tabular columns expected by the
loader:

```bash
python PR_EC/analysis/multivariate_qc_r2.py \
  --archive 0.00=PR_EC/outputs/adhd200/qsr_eta_0.00.npz \
  --archive 0.15=PR_EC/outputs/adhd200/qsr_eta_0.15.npz \
  --phenotype-csv dataset/ADHD200/Phenotypic_Processing.csv \
  --subject-id-column "ScanDir ID" \
  --qc-columns func_mean_fd func_dvars func_quality \
  --output-dir PR_EC/analysis/outputs/multivariate_qc_r2/adhd200 \
  --plot
```

The script writes:

- `multivariate_qc_r2_summary.csv`: one row per `eta`, including `mean_r2`;
- `multivariate_qc_r2_edges.csv`: one row per eta and directed edge;
- `multivariate_qc_r2_vs_eta.png`: optional curve with the requested y-axis label.

Self-edges are excluded by default. Missing QC rows are excluded consistently from all
eta values, and constant edges are omitted from the mean because their `R^2` is
undefined. This analysis is descriptive; it does not by itself provide uncertainty
intervals or a significance test for differences between eta values.
