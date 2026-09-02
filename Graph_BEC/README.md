# FSTA-Graph-BEC

本目录实现一个从 ROI 时间序列生成个体化有向 BEC，并使用患者相似图进行无诊断标签修正的流程。最终通过 Directed BrainNetCNN 比较不同 BEC 表示的 ASD/TC 分类性能。

## 主要流程

```text
ROI 时间序列
    ↓ FSTA-EC（无监督重建）
个体化有向 BEC
    ↓ phenotype / fusion 患者图
邻居参考 BEC
    ↓ PGR-BEC / QSR-BEC（仅使用训练 fold）
Original、Refined、QC-refined BEC
    ↓ Directed BrainNetCNN
交叉验证分类与结果汇总
```

- BEC 生成、患者图构建和 BEC 修正不使用诊断标签。
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

使用已有 BEC（默认模式）：

```bash
python Graph_BEC/main_abide_i.py
```

从 ROI 时间序列重新生成 BEC：

```bash
python Graph_BEC/main_abide_i.py --input-mode raw
```

### ADHD200

使用已有 BEC（默认模式）：

```bash
python Graph_BEC/main_adhd200.py
```

从 ROI 时间序列重新生成 BEC：

```bash
python Graph_BEC/main_adhd200.py --input-mode raw
```

### ABIDE-II

```bash
python Graph_BEC/main_abide_ii.py
python Graph_BEC/main_abide_ii.py --input-mode raw
```

调参时直接修改对应 `main_*.py` 文件中的 `parser.add_argument(..., default=...)`，或在命令行覆盖参数；固定入口不需要再传 `--dataset`。

### 常用选项

```bash
# 仅运行指定图模式
python Graph_BEC/main_abide_i.py --graph-mode phenotype
python Graph_BEC/main_adhd200.py --graph-mode fusion

# 仅比较指定 BEC 表示
python Graph_BEC/main_abide_i.py \
  --representations original refined qc_refined

# 指定 GPU；使用 CPU 时填写 cpu
python Graph_BEC/main_adhd200.py --gpu-id 0
python Graph_BEC/main_adhd200.py --gpu-id cpu
```

`--input-mode bec`（默认）要求对应 `main_*.py` 中配置的 BEC 文件已经存在；`--input-mode raw` 会读取原始 ROI 时间序列、训练 FSTA-EC，并覆盖保存对应的 BEC 文件。数据根目录或表型文件位置不采用默认值时，可使用 `--data-root` 和 `--phenotype-csv` 覆盖。

不同数据集需要的连续表型权重数量不同：ABIDE-I 为 2 个，ADHD200 为 3 个。例如：

```bash
python Graph_BEC/main_abide_i.py \
  --continuous-weights 1.0 0.3
python Graph_BEC/main_adhd200.py \
  --continuous-weights 1.0 0.3 0.3
```

## 输出文件

输出目录默认为：

```text
Graph_BEC/outputs/abide-i/
Graph_BEC/outputs/abide-ii/
Graph_BEC/outputs/adhd200/
```

主要文件包括：

- `*_subject_bec.npz`：原始 BEC；
- `*_refined_subject_bec.npz`：PGR-BEC 修正结果；
- `*_qsr_refined_subject_bec.npz`：QC/QSR 修正结果；
- `experiment_summary.csv`：各表示的交叉验证指标；
- `summary.json`：运行配置和汇总结果。

运行结束后，终端会打印每个表示的分类指标均值和标准差。详细的群体边/ROI 分析脚本位于 `Graph_BEC/analysis/`，说明见 `Graph_BEC/analysis/README.md`。

## 关键模块

```text
Graph_BEC/
├── main_abide_i.py          # ABIDE-I 配置和固定入口
├── main_abide_ii.py         # ABIDE-II 配置和固定入口
├── main_adhd200.py          # ADHD200 配置和固定入口
├── runner.py                # 三个入口共享的完整运行流程
├── dataset_configs.py       # 非主流程工具的兼容配置注册
├── data/                    # 数据、表型和 QC 加载
├── model/fsta_ec/           # FSTA-EC 与 BEC 生成
├── model/fusion_graph/      # 患者相似图
├── model/pgr/               # PGR-BEC 修正
├── model/qsr/               # QSR/QC-BEC 修正
├── downstream/              # BrainNetCNN、分类器和指标
└── analysis/                # 结果分析与可视化
```

## 注意事项

- 运行前请安装项目所需的 Python 依赖，并确保 PyTorch 可以正常使用目标设备。
- 三个数据集的表型文件格式、字段名、默认路径和默认超参数已分别写入对应 `main_*.py`；更换数据位置时优先使用命令行参数覆盖。
- 结果解释应同时报告 Original、Refined 和 QC-refined BEC，避免只比较单一表示。
