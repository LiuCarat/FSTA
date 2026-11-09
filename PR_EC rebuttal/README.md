# Graph-QSR-EC

## Sim21 最终恢复对照：Initial、参考 R、实际 MPR 输出

`main_sim21_recovery.py` 不是从原始 MAT 开始的一键训练入口。默认要求已经存在
`outputs/sim21/summary.json`、`oof_predictions.npz`、每折参考与 Encoder 缓存。
缺少这些文件时会退出并给出路径提示，不会自动训练；已有结果在其他目录时请传
`--source-dir`，缓存分开放置时再传 `--cache-dir`。如果只有完整 Encoder 缓存，
可用 `main_sim21.py --reuse-cache --cache-dir ...` 并保持原训练参数重新导出结果；
如果缓存和 OOF 都没有，则必须先单独运行 `main_sim21.py`（会训练 Encoder）。
Sanchez 输出、原始 `sim21.mat` 和 `/tmp` 中的单 epoch 冒烟结果不能冒充正式结果。

```bash
# 只读已有 Sim21 结果与折内缓存；绝不训练 Encoder 或修正器
python "PR_EC rebuttal/main_sim21_recovery.py" --source-dir "PR_EC rebuttal/outputs/sim21"

# 若确实有五折 Sim21 MPR 模型及来源记录，加载模型进行推理
python "PR_EC rebuttal/main_sim21_recovery.py" --mpr-dir /path/to/sim21_mpr

# 已有 Sim21 Encoder 缓存时，明确启用“只训练 MPR”，不重训 Encoder
python "PR_EC rebuttal/main_sim21_recovery.py" --train-mpr

# 之后只加载刚才的五折 MPR 模型，不重复训练
python "PR_EC rebuttal/main_sim21_recovery.py" --mpr-dir "PR_EC rebuttal/outputs/sim21_mpr"
```

控制台和 `outputs/sim21_recovery/table_R2_recovery.md` 显示三行：
Initial EC、MPR Group Reference、Initial EC + MPR，列为 Directed AUPRC、F1、SHD、
真实强度相关性、配对强弱判断准确率、相对 Initial 的矩阵平均绝对改变量。
普通数值保留两位小数，改变量使用科学计数法，避免非零变化被显示成 0.00。
CSV/NPZ/JSON 保留完整精度，逐人拓扑指标和变化另存 CSV。

每名被试使用自己的 `net`，非对角 `net != 0` 为真实边（不是按权重正负判断）；
拓扑评分完全复用 R1 的个体级 AP/F1/SHD 与固定 FSTA-style 0.5 相对阈值。
真实强度相关性保持原 R2 的定义：目标边在 A、B 组内分别计算 Pearson r，
再取 Fisher-z 平均，不悄悄改成跨所有人/所有边的相关性。

**现有 `main_sim21.py` 只训练 Encoder、构建参考，不训练/保存 MPR 模型。**
因此仅有该入口的输出，不能计算实际精炼 EC。未指定 `--mpr-dir` 或 `--train-mpr` 时，最后一行
明确为 Pending（待验证），不将 R 或 A 冒充 MPR 输出；提供的模型不完整或来源
不匹配则报错，不会补训、调参或使用 Sanchez 模型替代。

`--train-mpr` 是显式授权的新增分支：复用每折 40 人训练 EC，在训练折重建排除
自身的 FC 邻居参考，仅训练现有 `PGRECStatic` 无 QC 修正器。不会加载或训练
Encoder，更不会将测试邻居混入 Initial EC。默认 MPR 缓存写到 `outputs/sim21_mpr/`；
可用 `--mpr-dir` 指定另一处新目录，已有 checkpoint 则拒绝覆盖。
固定复用既有设置：80 epochs、lr=0.03、gate_max=0.4、anchor=0.8、gate L1=0.38、
variance weight=1.0、retention=0.85；不根据 GT 调参或挑选结果。
原损失可能仍收缩到零门控；即使最终指标不变，也会打印真实结果和恒等输出警告。

**Initial EC 基线永远不参考邻居**：它逐元素取自冻结 Encoder 的原始测试 EC，
已核对每折缓存并记录 SHA256，MPR 前后检查基线未被修改。只有 Reference 和
Initial + MPR 两行使用群体信息。Initial + MPR 是实际修正器输出，不是固定混合。

可选的 MPR 文件协议为每个 `fold_01/` 至 `fold_05/` 包含 `mpr.pt`（现有
`PGRECStatic` 的 state_dict）和 `metadata.json`。JSON 必须记录：

- `dataset: "NetSim Sim21"`、`prediction_orientation: "paper_initial_ec_attention_transpose"`。
- `train_indices`、`test_indices`：与配对五折完全一致的原始被试索引。
- `encoder_identity`：对应 Encoder 缓存 `metadata.json` 的完整 `identity` 对象。
- `fc_weights_sha256`：当时测试 FC 权重数组 `np.ascontiguousarray(fc_weights).tobytes()`
  的 SHA256；用于拒绝把不同 K 的参考套到已有模型上。
- `gate_max`、`hidden_channels`：模型实际结构参数，不能猜测。

这些记录应在真正的折内 MPR 训练时保存，不能事后给不明来源的模型补写假来源。
评估器校验数据指纹、配对划分、缓存测试 EC、参考重建和 checkpoint 来源，然后
仅执行 CPU 推理。会打印每折门控强度、平均/最大改变量；若最终 50 张矩阵与
Initial 完全相同，会明确警告恒等输出，不能宣称最终 MPR 提升。

## NetSim Sim21：真实个体与组间连接强度差异保留（Table R2）

```bash
# 正式实验：独立读取 NetSim 数据，不使用 Sanchez 的 OOF EC
python "PR_EC rebuttal/main_sim21.py" --gpu-id 0

# 单 epoch 冒烟验证，不作为正式实验结果
python "PR_EC rebuttal/main_sim21.py" --epochs 1 --gpu-id cpu \
  --output-dir /tmp/sim21_smoke

# 复用同参数的已存 Sim21 折内 EC，离线重算参考与指标，不再训练 Encoder
python "PR_EC rebuttal/main_sim21.py" --gpu-id cpu --reuse-cache

# 自行指定邻居数；更换 K 可复用同一批 Encoder 缓存，不必重训
python "PR_EC rebuttal/main_sim21.py" --gpu-id cpu --reuse-cache --reference-k 10
```

默认输入 `dataset/NetSim/sim21.mat`，输出 `PR_EC rebuttal/outputs/sim21/`；
支持 `--mat-path`、`--output-dir`、`--cache-dir`。训练参数复用当前
`main_simulation.py` 的默认值，可在命令行预先指定。复用缓存时必须传回相同的
epochs 等编码器参数，数据指纹/参数/划分/方向不匹配或缺少缓存会报错，绝不补训。
缓存已存在但不传 `--reuse-cache` 也会拒绝覆盖。此入口没有 MPR gate 或 QSR 训练。

数据验证：50 人、200 时间点、5 节点；`ts.reshape(50,200,5)`，保留真实
加权图 `net` 而非二值化。配对为 `(0,25),...,(24,49)`，检查每对仅
`net[:,1,2]` 不同且 A 组更大。真实目标均值从文件计算，不硬编码 0.408/0.204。
按 25 对做 `KFold(5, shuffle=True, random_state=42)`，每折 40 人训练、10 人测试，
每对必须同进同出。BOLD 按被试/ROI 时间维 z-score；编码器结构、重建和熵损失不变。
训练中不传 GT 或组别，直接使用 `extract_subject_ec()` 的 `attention.T` 输出，
不根据 GT 选择转置。目标索引遵循附件约定 `net[:,1,2]`；源—目标解释沿用论文定义。

四种比较：

- Initial EC：该折编码器提取的测试 EC。
- FC-neighbor Reference：Pearson FC 上三角，训练折拟合均值/标准差，
  邻居数通过 `--reference-k` 设置，默认 5，允许 1–40；随机参考同步匹配
  调整后的非零邻居数和权重集合。K 不属于 Encoder 参数，改变 K 不影响缓存复用。
  不应根据测试 GT 得分挑选最有利的 K 用于正式报告。
  cosine Top-K 并裁剪负相似度，行归一化后加权训练 EC；复用现有 FC 构图规则。
  非正相似度权重为零，全零行等权回退，实际非零邻居可能少于 K。
- Random-neighbor Reference：固定 seeds 0–19，从当前训练折不放回随机选邻居，
  匹配每个测试被试 FC 参考的非零邻居数及权重集合，仅随机替换身份和权重分配。
  20 次分别生成 OOF、计算指标，再汇总跨种子均值/样本标准差；不选最佳种子。
- Global Reference：当前折全部 40 人训练 EC 的平均，每个测试被试共享，不跨折聚合。

Table R2 不报告 AUPRC/F1/SHD，专门评价强度异质性：

- **Group Hedges' g**：目标边 A-B 的带符号组间效应，pooled sample SD，
  使用自由度 48 的精确 gamma 小样本修正；不是配对标准化效应。
- **Paired Sign Accuracy**：25 对中预测 A 边权严格大于 B 的比例，平分算错。
  这是真实强度效应的正负，不是 A→B 神经连接方向准确率。
- **Within-group r**：分别在 A、B 组内算预测与 GT 的 Pearson r，
  表格报告两组 r 的 Fisher-z 平均；两组原始 r 另存 CSV。常数预测的相关性记 N/A。
- **Spurious Group Difference**：其余 19 条非对角有向边预测 A-B 均值差的
  绝对值再平均，同时导出总和及各边明细。这是预测尺度的组效应，不是原始因果参数 MAE。

控制台和 Markdown 表格保留两位小数，CSV、JSON、NPZ 保留完整精度；
很小的非零数可能显示为 `0.00`，需查看 CSV 才能区分。控制台直接打印四行 R2。
输出 `table_R2.md/.csv`、`subject_target_weights.csv`、
`random_seed_metrics.csv`、`spurious_edge_effects.csv`、`oof_predictions.npz`、
每折图权重和 `summary.json`。OOF 中 Initial/FC/Global 均为 `(50,5,5)`，
Random 为 `(20,50,5,5)`；`encoder_cache/` 保存每折模型、训练/测试 EC 和身份配置。

解释限制：Random 行的 ± 是跨 20 种子的 SD，其余行是一套 50 人 OOF 的整体指标，
不能误称被试均值 ± SD。不同折由不同编码器生成，尺度与折构成可能影响 pooled r。
Global 在同一折测试对内完全相同，因此严格配对一致率必为 0；其 pooled r 即使非零
也不代表个体化信息。结果仅评价群体参考，不能代替训练后 MPR/QSR 输出的表现，
不按结果搜索 K、种子或混合比例，也不据此作未经检验的显著性结论。

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
