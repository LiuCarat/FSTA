# Top-k cutoff 检查

这是一个引子实验，用于检查 ABIDE-I 和 ABIDE-II 中固定 Top-10 / Top-20
是否对应明显的效应量断点，而不是定义“稳定异常”。

## 方法

每个数据集分别进行 ASD vs HC（仓库标签约定为 `0 = HC/TC`、`1 = ASD`）：

1. 对每个受试者的 90×90 BEC 矩阵计算组间效应量；
2. 排除 90 条自连接，仅保留 90×89 条有向边；
3. 使用 ASD − HC/TC 的 Hedges' g，并按 `abs(Hedges' g)` 从大到小排序；
4. 检查 Rank 10 vs Rank 11、Rank 20 vs Rank 21，并计算
   `Delta = AbsG(rank) - AbsG(rank + 1)`。

Hedges' g 使用合并组内标准差计算 Cohen's d，再乘以小样本校正因子：

```text
J(df) = Gamma(df / 2) / (sqrt(df / 2) * Gamma((df - 1) / 2))
df = n_ASD + n_HC - 2
```

排序只使用绝对值；`G` 和 `NextG` 保留方向，便于核对边是否是 ASD 增强或减弱。

## 运行

在仓库根目录执行：

```bash
python3 Graph_BEC/analysis/topk_cutoff_check/run_topk_cutoff_check.py
```

默认读取：

```text
Graph_BEC/outputs/abide-i/abide_qsr_refined_subject_bec.npz
Graph_BEC/outputs/abide-ii/abide_ii_qsr_refined_subject_bec.npz
```

也可以切换 BEC 数组，例如：

```bash
python3 Graph_BEC/analysis/topk_cutoff_check/run_topk_cutoff_check.py \
  --bec-key original_bec \
  --ranks 10 20 50
```

## 输出

- `outputs/topk_cutoff_summary.csv`：每个数据集和每个 cutoff 的摘要表，包含
  `AbsG`、`NextAbsG`、`Delta` 以及两条边的信息；
- `outputs/all_edges_ranked_by_absolute_hedges_g.csv`：两套数据集全部有向边的完整排序。

如果 `Delta` 相对于边界附近的效应量很小，说明该处没有明显的自然统计分界。
因此固定 Top-10 / Top-20 更适合作为结果展示，而不宜单独作为“稳定异常”的定义。
该判断建议结合完整排序表和不同数据集/重采样下的稳定性分析，而不要只依据单个 Delta。
