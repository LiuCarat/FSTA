# Graph-QSR-EC

## Rebuttal：仅评估 Initial EC（严格五折 OOF）

新增独立入口 `main_simulation.py`。此入口只使用本目录的初始 STF-EC
编码器，不导入或运行 MPR、QSR、患者图和疾病分类器，也不使用 Ground Truth
训练、选择 checkpoint 或调整阈值。原有临床数据入口不受此实验入口影响。

在仓库根目录运行：

```bash
python "PR_EC rebuttal/main_simulation.py"

# 选择本次运行的数据集（默认只运行 Network1）
python "PR_EC rebuttal/main_simulation.py" --network 2 --gpu-id 0
python "PR_EC rebuttal/main_simulation.py" --gpu-id cpu

# 仅在显式指定多个数据集时才依次运行多个
python "PR_EC rebuttal/main_simulation.py" --networks 1 2 3 4

# 按实际被试数进行 10 折 OOF，约 90% 训练、10% 测试
python "PR_EC rebuttal/main_simulation.py" --network 3 --n-splits 10
```

从 `MemoryEC/data/processed/` 读取所选的 `Network{编号}_amp.npz`，
支持编号 1、2、3、4，默认只选 1。字段为 `bold`（被试 × 时间 × ROI）
和 `adj`（真实有向图）。被试数不再限制为 60，直接使用文件中的实际数量。
当前四组均有 60 名被试；前三组为 5 个 ROI，Network4 为 10 个 ROI；
编码器和熵正则中的节点数自动适配。
每名被试的每个 ROI 独立做时间维 z-score，不拟合跨被试的标准化统计量。

实验协议：

- 默认 `KFold(n_splits=5, shuffle=True, random_state=42)`，按实际被试数划分，
  约 80% 训练、20% 测试。例如 60 人时为 48/12，100 人时为 80/20；
  数量不能整除折数时，各测试折相差最多 1 人，仍保证每人恰好测试一次。
  可用 `--n-splits K` 调整为约 `(K-1)/K` 训练、`1/K` 测试，要求 `2 <= K <= 被试数`。
  各网络独立训练，不共享模型或预训练权重。
- 每折从头初始化编码器，只把当前折的训练被试传给原有 `train_stf_ec()`；模型结构、
  重建损失、熵正则和当前入口的训练超参数不变（epochs 见入口配置，默认 final checkpoint）。
  `--stf-checkpoint best` 如启用，也仅按训练损失选择。
- 仅对测试被试调用原有 `extract_subject_ec()`，然后按原始索引回填 OOF；
  每名被试恰好预测一次。不对所有人的 EC 先求均值。
- **方向约定（修正）**：按照作者提供的论文定义，初始 EC 为窗口平均注意力的转置，
  即 `extract_subject_ec()` 的输出，入口不再二次转置。注意力的行是 query（目标），
  列是 key（源），转置后作为 source × target 的 EC；原始网络文本与预处理脚本
  明确 `adj[i,j]` 表示 `i → j`。FSTA 的 `train_epoch()` 同样返回 `adj_mean.T`。
  解码器另有 `features @ attention` 的右乘操作，其混合方向与此 query/key 解释相反；
  先前仅据右乘就对 EC 再转置的依据不足。这里以作者给出的论文 EC 定义统一方向，
  不以两种方向的分数高低决定。工作区没有论文正文可供独立核对，正式提交时仍应对照
  原文明确该定义，并说明解码器混合方向与报告 EC 方向的区别。
- 只评价非对角线有向边，分别保留 `i → j`、`j → i`。主要指标 Directed AUPRC
  使用 `average_precision_score`（非插值 AP，不是梯形积分 PR-AUC），无阈值。
- Precision、Recall、F1、Accuracy、SHD 使用 **FSTA-style relative threshold**：
  对每名被试的非对角预测值单独计算 `min + 0.5 * (max - min)`，以 `>=` 二值化。
  对角线不参与最小值、最大值或评分。系数固定 0.5，不提供阈值系数搜索选项。
  原 FSTA 是在群体平均预测矩阵上应用该规则，本实验是同一规则的个体级扩展。
  这属于对初始固定阈值 0.1 的**事后修正**，不是实验前已预注册的规则；没有根据
  Ground Truth 搜索其他系数或挑选最优阈值。所有非对角预测相同时，遵循 `>=`
  规则将全部非对角边预测为存在。Precision/Recall/F1 的未定义情况按 0 处理。
- SHD 定义为非对角线有向邻接矩阵的不一致条目数：增边或删边各计 1，反向计 2。
  所有指标先逐被试计算，再汇总所有被试的均值和样本标准差（`ddof=1`），
  不是对各 fold 均值计算标准差。

默认输出 `PR_EC rebuttal/outputs/initial_ec_oof_paper_relative/`：

```text
experiment_summary.csv             # 各网络的六项指标 mean/std
results.md                         # 带阈值、方向说明的结果表格
Network1_amp/                      # 仅生成本次选中的网络目录
├── oof_ec.npz                     # oof_ec、ground_truth、fold_ids、个体指标数组
├── subject_metrics.csv            # 原始顺序的逐人指标
├── folds.json                     # 每折训练/测试原始索引与训练损失
└── summary.json                   # 指标汇总、完整参数与评分约定
```

`oof_ec.npz` 中保存 `auprc_scores`、`precision_scores`、`recall_scores`、
`f1_scores`、`accuracy_scores`、`shd_scores`，均为 `(S,)`，S 为实际被试数。
`oof_ec` 为 `(S, N, N)`，`subject_indices`/fold 索引从 0 开始，
`subject_ids` 从 1 开始，对应原始 BOLD 文件编号。
`subject_thresholds` 保存每个被试的实际相对阈值，逐人 CSV 也包含 `threshold`。

### 已有 OOF 结果的事后重算（无需重新训练）

```bash
python "PR_EC rebuttal/rescore_simulation.py" --networks 1 2 3 4
```

原始 `outputs/initial_ec_oof/` 保持不变，另生成两组有明确标注的结果：

- `outputs/initial_ec_oof_relative/`：只修改阈值，保留历史二次转置方向，
  AUPRC 数组和汇总值严格保留；其他五项指标及逐人阈值重新计算。
  **这张表是历史阈值修订审计表，不是按论文定义修正方向后的正式提交表。**
- `outputs/initial_ec_oof_paper_relative/`：按作者提供的论文定义撤销历史二次转置，
  使用同一个固定 0.5 相对阈值规则。因为有向边的位置变化，AUPRC 也必须重新计算；
  不能把历史方向的 AUPRC 与新方向的 F1/SHD 拼成一张表。

两张表均提供 `results.md`、`experiment_summary.csv` 和完整逐人结果。
方向修正是依定义作出的代码纠正，不是按分数选择方向；两表并列仅用于追溯变化。

快速检查代码流程时可使用 `--epochs 1 --output-dir /tmp/initial_ec_smoke`；
该命令仅用于冒烟验证，不应作为论文的正式实验结果。

本目录实现一个从 ROI 时间序列生成个体化有向 EC，并使用患者相似图进行无诊断标签修正的流程。最终通过 Directed BrainNetCNN 比较不同 EC 表示的 ASD/TC 分类性能。

## 主要流程

```text
ROI 时间序列
    ↓ STF-EC Encoder（无监督重建）
个体化有向 EC
    ↓ phenotype / fusion 患者图
邻居参考 EC
    ↓ PGR-EC / QSR-EC（仅使用训练 fold）
Original、Refined、QC-refined EC
    ↓ Directed BrainNetCNN
交叉验证分类与结果汇总
```

- EC 生成、患者图构建和 EC 修正不使用诊断标签。
- 诊断标签仅用于下游分类和群体差异分析。
- 默认使用 `fusion` 图，同时考虑表型和 fMRI 功能连接；也可以使用 `phenotype` 图。
- 现在每个数据集都有独立主入口，参数和默认配置直接写在对应的 `main_*.py` 中。

## 数据准备

默认路径由三个数据集入口中的配置管理，目录结构如下：

```text
dataset/
├── ABIDE-I/
│   ├── cpac/filt_noglobal/rois_aal/   # AAL ROI 时间序列
│   └── Phenotypic_Processing_filled.csv
├── ABIDE-II/
│   ├── cpac/filt_noglobal/             # AAL ROI 时间序列
│   └── Phenotypic_Processing.csv
└── ADHD200/
    ├── cleaned/AAL_TCs_filtfix/       # 推荐目录；也支持 AAL_TCs_filtfix/
    └── adhd200_preprocessed_phenotypics.tsv
```

三个数据集都会使用 116 个输入 ROI 的前 90 个 ROI，并在加载时进行时间序列标准化。

## 运行不同数据集

请在仓库根目录运行命令：

### ABIDE-I

使用已有 EC（默认模式）：

```bash
python PR_EC/main_abide_i.py
```

从 ROI 时间序列重新生成 EC：

```bash
python PR_EC/main_abide_i.py --input-mode raw
```

### ADHD200

使用已有 EC（默认模式）：

```bash
python PR_EC/main_adhd200.py
```

从 ROI 时间序列重新生成 EC：

```bash
python PR_EC/main_adhd200.py --input-mode raw
```

### ABIDE-II

```bash
python PR_EC/main_abide_ii.py
python PR_EC/main_abide_ii.py --input-mode raw
```

调参时直接修改对应 `main_*.py` 文件中的 `parser.add_argument(..., default=...)`，或在命令行覆盖参数；固定入口不需要再传 `--dataset`。

### 常用选项

```bash
# 仅运行指定图模式
python PR_EC/main_abide_i.py --graph-mode phenotype
python PR_EC/main_adhd200.py --graph-mode fusion

# 仅比较指定 EC 表示
python PR_EC/main_abide_i.py \
  --representations original refined qc_refined

# 指定 GPU；使用 CPU 时填写 cpu
python PR_EC/main_adhd200.py --gpu-id 0
python PR_EC/main_adhd200.py --gpu-id cpu
```

`--input-mode ec`（默认）要求对应 `main_*.py` 中配置的 EC 文件已经存在；`--input-mode raw` 会读取原始 ROI 时间序列、训练 STF-EC 编码器，并覆盖保存对应的 EC 文件。数据根目录或表型文件位置不采用默认值时，可使用 `--data-root` 和 `--phenotype-csv` 覆盖。

不同数据集需要的连续表型权重数量不同：ABIDE-I 为 2 个，ADHD200 为 3 个。例如：

```bash
python PR_EC/main_abide_i.py \
  --continuous-weights 1.0 0.3
python PR_EC/main_adhd200.py \
  --continuous-weights 1.0 0.3 0.3
```

## 输出文件

Baseline 方法及其默认参数的 LaTeX 汇总表见
`PR_EC/baseline/parameter_settings.tex`。

输出目录默认为：

```text
PR_EC/outputs/abide-i/
PR_EC/outputs/abide-ii/
PR_EC/outputs/adhd200/
```

主要文件包括：

- `*_subject_ec.npz`：原始 EC；
- `*_refined_subject_ec.npz`：PGR-EC 修正结果；
- `*_qsr_refined_subject_ec.npz`：QC/QSR 修正结果；
- `experiment_summary.csv`：各表示的交叉验证指标；
- `summary.json`：运行配置和汇总结果。

运行结束后，终端会打印每个表示的分类指标均值和标准差。详细的群体边/ROI 分析脚本位于 `PR_EC/analysis/`，说明见 `PR_EC/analysis/README.md`。

## 关键模块

```text
PR_EC/
├── main_abide_i.py          # ABIDE-I 配置和固定入口
├── main_abide_ii.py         # ABIDE-II 配置和固定入口
├── main_adhd200.py          # ADHD200 配置和固定入口
├── runner.py                # 三个入口共享的完整运行流程
├── dataset_configs.py       # 非主流程工具的兼容配置注册
├── data/                    # 数据、表型和 QC 加载
├── model/individual_ec/        # STF-EC 编码器与 EC 生成
├── model/mpr/                 # 患者相似图与参考 EC
├── model/ablation/mpr_refinement.py  # 去除 QC 的 PGR 消融结构
├── model/qsr/                   # 最终 QSR-EC 修正模型
├── downstream/              # BrainNetCNN、分类器和指标
└── analysis/                # 结果分析与可视化
```

## 注意事项

- 运行前请安装项目所需的 Python 依赖，并确保 PyTorch 可以正常使用目标设备。
- 三个数据集的表型文件格式、字段名、默认路径和默认超参数已分别写入对应 `main_*.py`；更换数据位置时优先使用命令行参数覆盖。
- 结果解释应同时报告 Original、Refined 和 QC-refined EC，避免只比较单一表示。
