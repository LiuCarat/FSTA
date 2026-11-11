# ABIDE-II 极端 Mean FD 核查与敏感性分析

## 已确认的数据来源

`covariate_adjusted_outputs/abide-ii_subject_audit.csv` 中的数值来自
`dataset/ABIDE-II/Phenotypic_Processing.csv` 的 `func_mean_fd`，不是回归生成的数值，
也不是按错误行号拼接的其他被试。精确匹配 `FILE_ID`，并核对 `SUB_ID`、站点和标签。

| 被试 | 站点 | 项目 ASD 标签 | 原 CSV Mean FD | 原 CSV DVARS | 原 CSV func_quality |
| --- | --- | ---: | ---: | ---: | ---: |
| sub-29887 | EMC_1 | 1 | 69.489473 | 3.3622742e+10 | 0.075 |
| sub-29888 | EMC_1 | 1 | 44.670752 | 142.65377 | 0.28125 |

当前 `dataset/ABIDE-II/scripts/extract_aal90.py` 中 `qc_values()` 的逻辑为：
读取 fMRIPrep confounds TSV 中 `framewise_displacement`，缺失帧填 0，再计算所有帧
的算术平均；没有乘 100/1000、除以 TR、取百分比或计算 FD 总和。
`func_quality` 是 FD <= 0.2 的帧比例。该代码的逻辑不能代替逐帧原始文件核查。

压缩包外层 `fmriprep/dataset_description.json` 标记 fMRIPrep 版本为 **25.2.5**，
与项目运行脚本默认版本一致。外层描述快照和 EMC_1 的源数据描述保存于 `raw_confounds/metadata/`。

## 原始逐帧 FD 与单位：已完成核查

实际压缩结构为 `fmriprep.tar.zst -> fmriprep/EMC_1.tar.gz -> EMC_1/sub-xxxxx/ses-1/func/`。
从内层站点压缩包提取 29880、29887、29888 的原始 confounds TSV 和 JSON，
未解压或修改 BOLD、EC 或表型 CSV。每个被试均为 160 帧，首帧 FD 为缺失值。

原始 JSON 明确标注 `framewise_displacement` 和 `trans_x/y/z` 的单位为 **mm**，
`rot_x/y/z` 的单位为 **rad**。独立从六个运动参数重算：

```text
FD[t] = sum(abs(trans_xyz[t] - trans_xyz[t-1]))
      + 50 * sum(abs(rot_xyz[t] - rot_xyz[t-1]))
```

半径为 50 mm，重建 FD 与原始 FD 列的最大绝对误差均不超过 `1.14e-13 mm`。
调用项目现有 `qc_values()` 对逐帧数据重算，得到：

| 被试 | 重算 Mean FD，首帧置零，mm | 原始单帧最大 FD，mm | FD > 100 mm 帧数 | 最大相邻平移跳变，三个轴，mm |
| --- | ---: | ---: | ---: | --- |
| sub-29887 | 69.4894730294 | 831.9341971 | 16 | 269.819 / 364.698 / 299.550 |
| sub-29888 | 44.6707515170 | 986.9338605 | 17 | 426.112 / 413.368 / 293.361 |
| sub-29880 | 6.2748616049 | 35.7028373 | 0 | 3.348 / 14.320 / 17.297 |

重算均值与 CSV 一致至保存精度，故不是审计表匹配错误、均值/总和混淆、百分比、TR 归一化或
本次回归导致的小数点问题。若只平均 159 个有限帧，前两者均值分别为 69.9265137403 和
44.9516996397 mm，首帧填零约定无法解释异常。

**异常源于原始 fMRIPrep 运动估计/FD 输出。** 如此大的配准参数跳变及 DVARS 异常，
强烈提示运动估计/配准失败或严重图像质量问题；不能将它直接解释为被试真实移动近一米。
最终成因仍需原始 BOLD、头动校正图像和预处理报告的视觉核查，不能凭本次参数核查区分。
原始 TSV/JSON 保存在 `raw_confounds/sub-*/`；摘要、SHA-256、运动参数重建误差和帧索引
见 `raw_confounds_summary.json`。该核查比仅依据字段命名推断单位更直接。

## 高杠杆点诊断

在 ABIDE-II 指定的完整模型设计矩阵中：

| 被试 | 设计矩阵杠杆值 h_ii | FD partial leverage |
| --- | ---: | ---: |
| sub-29887 | 0.7004912324 | 0.6806913329 |
| sub-29888 | 0.2881794427 | 0.2683710587 |

这里 FD partial leverage 定义为：先将 Mean FD 对截距、ASD、Age、Sex 和站点
虚拟变量回归，得到残差 r，再计算 r_i² / sum(r²)。两名被试合计为 **94.9062%**。
这是针对 FD 设计列的影响诊断，不是“这两人解释了 94.9% 的 ASD 效应”，也不是边级 Cook 距离。

边级 OLS 诊断进一步显示：排除这两人前后，8,010 条边的 FD 系数相关性为
**0.691633**，ASD 系数相关性为 **0.981412**。按完整模型 FD 系数标准误缩放，
两人联合删除造成的 FD 系数绝对变化中位数为 **12.4774**；该量不是单被试 DFBETAS。
采用仅作为筛查的 Cook 距离阈值 `4/975`，sub-29887 在 6,201 条边上超阈值，
sub-29888 在 3,351 条边上超阈值。Cook 距离和阈值筛查不自动等同于数据错误或排除依据。
详见 `sensitivity/edge_influence_summary.json` 和 `sensitivity/abide_ii_edge_influence.csv`。

## 敏感性结果

各方案均直接复用原最终 EC、原标签和 ROI 顺序；每队列重新拟合回归，
对各自全部 8,010 条非对角有向边重新 BH-FDR，重复标准仍为双队列 q < 0.05 且 ASD 系数同号。

| 方案 | ABIDE-I / II 样本量 | 跨队列重复 | 原 213 条保留 | 原 213 条队列间方向一致 |
| --- | --- | ---: | ---: | ---: |
| 原始 Mean FD | 871 / 975 | 90 | 59 | 212 |
| 仅排除 sub-29887、sub-29888 | 871 / 973 | 50 | 46 | 213 |
| 排除 Mean FD > 2 的被试 | 871 / 969 | 31 | 31 | 211 |
| 全部被试保留，FD 改为 log1p(FD) | 871 / 975 | 46 | 43 | 213 |

各方案“两个队列均保留原始效应方向”的原 213 条数量，与表格最后一列相同。
后三个方案的重复连接均为原始 Mean FD 模型 90 条重复连接的子集。

Mean FD > 2 方案在 ABIDE-I 无排除；ABIDE-II 排除 6 人：
`sub-28832; sub-29102; sub-29514; sub-29880; sub-29887; sub-29888`，按项目标签均为 ASD。
原始数据缺失排除仍为 0；此处是质量敏感性排除，不能写成缺失排除。
`log1p_fd` 在两个队列均应用，不只改 ABIDE-II。

## 解释限制

- 原始 90/59/212 是按原 CSV 和指定线性公式得到的可复现结果，但显著性结论对极端 FD 敏感。
- 目前不能据此把 69.49 当作 0.6949，或自行更改单位/小数点。
- 上述排除和变换均为发现异常后的探索性敏感性分析，不应按连接数选择“最好”的方案。
  Mean FD > 2 是用于展示影响的阈值，不是本项目预注册或已有的质量阈值。
- 即使逐帧 FD 数值能被正确重建，也只能证实数值传递和计算一致，不能证明扫描质量合格。
- 下一步应据预处理报告及原始影像/配准诊断制定独立于连接显著性的统一 QC 规则。
  本次不修改表型、EC 或原统计输出，不重新训练。

## 文件与复现

- `sensitivity/sensitivity_summary.csv`：上表数字。
- `sensitivity/cohort_samples.csv`：各方案样本量、排除 ID 和单队列显著边数。
- `sensitivity/subject_leverage.csv`：全部被试的杠杆值与 FD partial leverage。
- `sensitivity/*_edges.csv`、`sensitivity/*_cross_cohort.csv`：各方案全边系数、p、q 和跨队列结果。
- `../audit_fd_extremes.py`：原始小型 confounds 文件提取、数值核查和敏感性分析脚本。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python PR_EC/analysis/cross_cohort_replication/audit_fd_extremes.py \
  --output-dir PR_EC/analysis/cross_cohort_replication/fd_audit/sensitivity_rerun
```

原始 FD 方案与前次保存的全部 ASD 系数及 q 值通过数值一致性检查。
