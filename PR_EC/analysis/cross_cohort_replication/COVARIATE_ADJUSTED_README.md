# 协变量调整后的跨队列 EC 重复发现

## 主要结果

本实验直接使用已有最终 QSR-refined EC 的 `ec` 字段，不训练或修改任何模型。
复算原 Welch 检验流程后，得到的重复连接集合与原始 `outputs/replicated_edges.csv`
中的 **213 条连接完全相同**。原始脚本和 `outputs/` 中的结果未被覆盖。

| 问题 | 结果 |
| --- | ---: |
| 协变量调整后的跨队列重复连接 | **90 条** |
| 原 213 条中仍满足调整后重复发现标准 | **59 条（27.7%）** |
| 原 213 条中调整后两个队列的效应方向仍一致，不要求显著 | **212 条（99.5%）** |
| 原 213 条中调整后在两个队列均保留原效应方向 | **212 条（99.5%）** |

调整后的 90 条包括 59 条原连接和 31 条新增连接。原 213 条中 154 条不再满足
“两个队列均 q < 0.05 且方向一致”。方向一致不等同于调整后仍显著。
唯一队列间方向不一致的原连接为 `ROI_022 -> ROI_039`，其调整后 ASD 系数为
ABIDE-I：`-0.0002059331953006624`，ABIDE-II：`0.0000011218574617990587`。

## 方法

每个队列独立拟合：

```text
EC_edge ~ ASD + Age + Sex + C(Site) + Mean_FD
```

- `Age`、`Sex`、`Site`、`Mean_FD` 分别来自原始表型 CSV 中的
  `AGE_AT_SCAN`、`SEX`、`SITE_ID`、`func_mean_fd`，不插补或变换。
- ASD 使用 EC 文件中既有的 `labels`。严格保留项目既有编码约定：
  `DX_GROUP=2 -> labels=1`，`DX_GROUP=1 -> labels=0`；本实验不重新解释或更改诊断标签。
- 包含截距；站点使用 treatment-coded 固定效应，排序后的第一个站点作为参照。
  `Sex` 按公式作数值项；其取值为 1/2，含截距时与二分类虚拟变量编码等价。
- 采用普通最小二乘、常规残差方差标准误及残差自由度的双侧 t 检验，提取 ASD 系数与 p 值。
  通过共享设计矩阵的 QR 分解同时计算各边，与逐边拟合相同模型等价。
- 每个队列分别对全部 **90 × 89 = 8,010 条非对角有向边**的 ASD p 值做 BH-FDR。
  不对称化、不限制在原 213 条或 top-k 内进行校正。
- 跨队列重复标准为同一有向边在两个队列均 `q < 0.05`，且 ASD 系数同号且非零。
- 除替换统计检验和其效应方向指标外，EC 文件、字段、标签、ROI 顺序及有向边定义均与原流程一致。

## 被试匹配和缺失

| 队列 | EC 被试 | 唯一匹配 | 缺失排除 | 纳入 | ASD / HC（项目标签） | 单队列显著边 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABIDE-I | 871 | 871 | **0** | 871 | 403 / 468 | 602 |
| ABIDE-II | 975 | 975 | **0** | 975 | 462 / 513 | 705 |

1. 去除 CSV 表头和值的首尾空白，处理 ABIDE-II 表头尾随空格。
2. 按 EC 文件顺序，以 `subject_ids == FILE_ID` **精确匹配**，不依赖 CSV 行顺序，禁止重复 ID。
3. 独立核对数值 `SUB_ID`：ABIDE-I 被试文件名末尾数字与 `SUB_ID` 数字一致；
   ABIDE-II 的 `sub-xxxxx` 与 `SUB_ID` 一致。
4. 对每个匹配被试核对表型 `SITE_ID` 与 EC `site_ids`、诊断编码与 EC `labels`。
   未发现错配、重复或未匹配 EC 被试。
5. 所需协变量均完整。脚本对缺失值、非有限值和常见负数缺失编码进行完整案例排除，
   并在逐被试审计表记录排除原因；不填补缺失值。

ABIDE-I 有 20 个站点，设计矩阵秩 24，残差自由度 847；ABIDE-II 有 16 个站点，
设计矩阵秩 20，残差自由度 955。两个设计矩阵均满秩。

## 保存的结果

所有新增输出位于 `covariate_adjusted_outputs/`：

- `summary.json`：核心结果、样本量、匹配/排除统计、设计矩阵信息、输入文件 SHA-256 与数值校验。
- `abide-i_edge_statistics.csv`、`abide-ii_edge_statistics.csv`：每队列所有非对角边的 ASD 系数、标准误、t、p、q。
- `all_edges_cross_cohort.csv`：8,010 条边的跨队列结果与原始效应比较。
- `replicated_edges.csv`：调整后 90 条重复连接。
- `original_213_comparison.csv`：原 213 条的调整后结果、显著保留标记、队列间方向一致性和原方向保持标记。
- `abide-i_subject_audit.csv`、`abide-ii_subject_audit.csv`：EC 行号、被试 ID、表型文件行号、协变量及纳入/排除原因。

## 运行和校验

在仓库根目录运行（只需 NumPy 和 SciPy）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python PR_EC/analysis/cross_cohort_replication/run_covariate_adjusted_replication.py \
  --output-dir PR_EC/analysis/cross_cohort_replication/covariate_adjusted_rerun
```

默认写入 `covariate_adjusted_outputs/`；为避免覆盖，脚本拒绝写入非空输出目录，
也拒绝写入原始 `outputs/` 目录及其子目录。复跑须指定新的空目录。

已完成的校验：

- 每队列 9 条分布于边索引空间的边，用独立逐边 `numpy.linalg.lstsq` 核对系数及 p 值，
  最大系数绝对误差小于 `6e-18`，最大 p 值绝对误差小于 `5e-14`。
- 每队列另选最小/最大 p 值、最接近 q=0.05 及首尾边，用 Frisch–Waugh–Lovell
  残差化独立核对 OLS 系数和 p 值，通过。
- 用 `scipy.stats.false_discovery_control(method="bh")` 核对全部 8,010 个 q 值，通过。
- 反转表型文件行序后重新匹配，协变量及纳入集合不变。
- 临时构造缺失 FD、重复 ID 和站点错配样本，分别确认完整案例排除、重复 ID 拒绝和站点错配拒绝。
