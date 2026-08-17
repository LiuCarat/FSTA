# ASD/TC 群体级 Original BEC 描述性分析

这个实验对应第二个大类：不训练新的模型，只使用已经得到的 individual Original BEC，描述 ASD 与 TC 两组在群体平均层面的有向 BEC 差异。当前版本不进行 p 值、置换检验、校正或其他统计显著性检验。

## 1. 分析定义

对每名受试者保留一个原始个体 BEC，按受试者等权计算：

```text
MeanBEC_ASD = mean(BEC_i | i 属于 ASD)
MeanBEC_TC  = mean(BEC_i | i 属于 TC)
Difference   = MeanBEC_ASD - MeanBEC_TC
```

矩阵方向沿用现有代码约定：

```text
BEC[source, target] = source ROI -> target ROI
```

因此 `Difference[source, target] > 0` 表示该有向边在 ASD 平均 BEC 中高于 TC，负值表示在 TC 中更高。这里不进行额外 z-score、阈值化、对称化或按时间窗口数加权；每名受试者对组均值贡献相同。

当前 ABIDE-I Original BEC 归档的标签约定为 `0=TC(HC)`、`1=ASD`。脚本把两个标签作为显式参数，避免把标签含义隐藏在代码里。

## 2. 默认运行

在仓库根目录执行：

```bash
python Graph_BEC/analysis/group_bec_descriptive.py
```

默认输入是：

```text
downstream_abide_i/outputs/original/loss_alpha_0.8/seed_42/epochs_101/subject_bec.npz
```

如需更换 Original BEC 版本：

```bash
python Graph_BEC/analysis/group_bec_descriptive.py \
  --bec-path downstream_abide_i/outputs/original/loss_alpha_0.7/seed_42/epochs_101/subject_bec.npz \
  --output-dir Graph_BEC/analysis/outputs/original_group_bec_alpha07
```

如果某个归档使用了相反的标签编码，可显式指定：

```bash
python Graph_BEC/analysis/group_bec_descriptive.py \
  --bec-path PATH_TO_BEC.npz --asd-label 0 --tc-label 1
```

## 3. 输出文件

默认输出目录为 `Graph_BEC/analysis/outputs/original_group_bec/`：

- `group_bec_table.csv`：全部 `90×89` 条非对角有向边，列严格为 `Source, Target, ASD_mean, TC_mean, Difference`，按 `|Difference|` 降序排列；
- `group_bec_difference_heatmap.png`：唯一热力图，颜色表示 `MeanBEC_ASD - MeanBEC_TC`，蓝色表示 TC 更高，红色表示 ASD 更高。

热力图由脚本直接写出 PNG，不依赖 matplotlib；因此在当前默认环境无法联网安装 matplotlib 时也可以生成。

## 4. 建议的解读顺序

1. 先报告两组样本数以及 `MeanBEC_ASD`、`MeanBEC_TC` 的整体范围；
2. 再展示差值矩阵，区分 ASD 增强的边和 TC 增强的边；
3. 最后按 `absolute_difference` 排名前若干条有向边，结合 ROI 解剖名称讨论候选网络；
4. 所有结果都表述为“描述性组间差异/候选边”，不能写成“显著差异”，因为本实验没有统计检验。

这个结果适合作为后续统计检验、置换检验或网络层面分析的输入，而不是这些检验的替代品。
