# Graph_BEC 群体差异分析

当前 `analysis` 只保留两个核心描述性分析脚本。两者都使用同一套标签和差值定义：

```text
label 0 = TC
label 1 = ASD
Difference = ASD_mean - TC_mean
```

因此：

- `Difference > 0`：ASD 中该连接增强；
- `Difference < 0`：TC 中该连接增强，也可表述为 ASD 中相对减弱。

当前分析不做统计显著性检验，因此结果应称为 descriptive group-level differences 或 candidate edges/ROIs。

## 1. Top-10 有向连接边

脚本：

```bash
python Graph_BEC/analysis/group_edge_difference.py
```

默认输入是模型阶段生成的 `Graph_BEC/outputs/refined_subject_bec.npz`，默认分析
`pgr_bec`，输出为：

```text
Graph_BEC/analysis/outputs/group_edge_difference/top_edges_asd_vs_tc.csv
```

默认输出 20 行：10 条 `ASD_enhanced` 和 10 条 `TC_enhanced`。表格包括 `source`、`target`、`asd_mean`、`tc_mean`、`difference_asd_minus_tc` 等字段，并保留有向边方向。

如需分析 QC 弱监督结果：

```bash
python Graph_BEC/analysis/group_edge_difference.py --bec-key qc_refined_bec
```

## 2. Top-10 差异 ROI

脚本：

```bash
python Graph_BEC/analysis/group_roi_difference.py
```

输出为：

```text
Graph_BEC/analysis/outputs/group_roi_difference/top_rois_asd_vs_tc.csv
```

每个 ROI 的排名指标是其所有入边和出边的平均绝对差异：

```text
TotalDifferenceScore(ROI) = mean(abs(Difference))
```

表格同时报告 ASD 增强分数、TC 增强分数、净方向、以及该 ROI 相关的最大绝对差异有向边。因此一个 ROI 可以是 mixed：它可能同时包含两个方向的差异。

QC 弱监督 ROI 分析：

```bash
python Graph_BEC/analysis/group_roi_difference.py --bec-key qc_refined_bec
```

## 3. Top-K ROI 有向弦图

脚本：

```bash
python Graph_BEC/analysis/chord_diagram.py
```

输出为：

```text
Graph_BEC/analysis/outputs/group_chord/chord_top10_rois_asd_tc.png
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

默认输入为 `Graph_BEC/outputs/pgr_bec_refined_subject_bec.npz`（键 `bec`），
可用 `--bec-key refined_bec` 或 `--bec-key original_bec` 切换（`refined_bec`
与 `bec` 数值相同）。

## 4. 解释限制

Top-10 结果是描述性候选边/候选 ROI，不是统计显著结果。后续若需要“显著增强/显著减弱”，还要另行加入组间检验、置换检验和多重比较校正。
