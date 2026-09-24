# ICLR 实验复现设置清单

> 本文档面向论文的 **Experiments / Experimental Setup / Reproducibility** 部分，按当前仓库实际实现整理。表中标记为“代码已确定”的内容可以直接写入论文；标记为“需补充”的内容需要根据数据下载版本、运行日志和目标 ICLR 论文原文核对后再定稿。
>
> 参考链接：OpenReview `N51nP3TBwR`。截至 **2026-09-13**，该链接在当前环境中未返回可读取的论文正文，因此本文不臆测参考论文中未出现在仓库的作者、数据划分或超参数。

## 0. 定稿前总检查

- [ ] 明确论文实验目标：分类性能、有效连接估计质量、跨队列泛化，或三者兼有。
- [ ] 明确主方法名称及缩写，并保证正文、图表、代码中的命名一致。
- [ ] 记录代码版本：Git commit、Python 版本、PyTorch 版本、CUDA/cuDNN 版本。
- [ ] 固定随机种子；当前主入口默认 `seed=42`，且每个交叉验证 fold 使用 `seed + fold * 1000`。
- [ ] 记录实际运行日期、GPU 型号、显存、CPU 核数和运行时长。
- [ ] 把所有最终结果与对应配置文件、日志和输出文件绑定，避免只保留手工整理后的均值。
- [ ] 核对参考论文原文的实验设置；若原文与当前实现不同，正文应明确写“复现设置”而不是声称完全一致。

## 1. 实验任务与总体流程

### 1.1 任务定义

当前代码将每名受试者的 ROI 时间序列转换为一个有向脑有效连接（brain effective connectivity, BEC）矩阵，并使用冻结的 BEC 表示进行二分类。默认标签编码为：

| 数据集 | 患者标签 | 对照标签 | 任务 |
|---|---:|---:|---|
| ABIDE-I | ASD = 1 | TC = 0 | ASD vs. typical control |
| ABIDE-II | ASD = 1 | TC = 0 | ASD vs. typical control |
| ADHD200 | ADHD = 1 | HC = 0 | ADHD vs. healthy control |

论文中应补充：最终纳入的受试者数、患者/对照人数、站点数、每名受试者时间点数范围，以及因缺失表型、时间序列或 QC 信息而排除的样本数。

### 1.2 当前实现流程

```text
ROI 时间序列
  -> 每名受试者、每个 ROI 沿时间维 z-score
  -> 保留前 90 个 AAL ROI
  -> STF-BEC 无监督训练/推理
  -> 得到原始有向 BEC
  -> 在每个训练 fold 内构建患者相似图
  -> 计算训练集邻居参考 BEC
  -> PGR-BEC 修正（refined）
  -> QC/QSR-BEC 修正（qc_refined）
  -> 冻结 BEC 表示
  -> Directed BrainNetCNN 下游分类
  -> 10-fold stratified test metrics
```

重要原则：BEC 生成、患者图构建、参考 BEC 计算、PGR/QSR 修正和下游模型选择均不得使用测试 fold 标签。诊断标签只用于下游分类损失和最终测试评估。

## 2. 数据集与输入

### 2.1 数据来源与预处理

| 项目 | 当前实现 | 论文中应写明 |
|---|---|---|
| 输入模态 | 静息态 fMRI ROI 时间序列 | 数据集官方网站/论文、下载版本和预处理管线 |
| 预处理管线 | CPAC `filt_noglobal/rois_aal` | 是否沿用发布的 CPAC 结果，是否重新预处理 |
| 原始 ROI | 116 个 | AAL 图谱版本与 ROI 顺序 |
| 实际建模 ROI | 前 90 个 ROI | 为什么截取 90 个、具体 ROI 列表 |
| 受试者标准化 | 每名受试者、每个 ROI 做时间维 z-score | 公式、零方差处理方式 |
| 多 run 处理 | 当前 ADHD200 loader 使用每名受试者一个选定 run；不拼接多个 run | 选定 run 的规则和数据文件清单 |
| 缺失值 | ADHD200 fold 内用训练集拟合的中位数/众数填补；类别未知值单独编码 | 缺失比例、填补策略和是否只在训练 fold 拟合 |
| QC 变量 | `func_mean_fd`, `func_dvars`, `func_quality` | 指标定义、来源和单位 |

### 2.2 数据路径与字段

| 数据集 | ROI 路径约定 | 表型文件 | ID 字段 | 标签字段 | 站点字段 | 连续表型 |
|---|---|---|---|---|---|---|
| ABIDE-I | `cpac/filt_noglobal/rois_aal/` | `Phenotypic_Processing_filled.csv` | `FILE_ID` | `DX_GROUP` | `SITE_ID` | `FIQ`, `PIQ` |
| ABIDE-II | `cpac/filt_noglobal/` | `Phenotypic_Processing.csv` | `FILE_ID` | `DX_GROUP` | `SITE_ID` | `FIQ`, `PIQ` |
| ADHD200 | `cpac/filt_noglobal/*_rois_aal.1D` | `Phenotypic_Processing.csv` | `ScanDir ID` | `DX` | `Site` | `Age`, `Full4 IQ`, `Handedness` |

当前配置中的混杂变量为：ABIDE-I/II 使用 `AGE_AT_SCAN`, `SEX`, `FIQ`, `PIQ`；ADHD200 使用 `Age`, `Gender`, `Full4 IQ`, `Handedness`。需在论文中说明这些变量用于何处：相似图、QC 修正、统计分析，还是额外的混杂控制；不要笼统写成“已控制所有混杂因素”。

## 3. 数据划分与防止信息泄漏

### 3.1 主评估协议

- [ ] 使用 `10-fold stratified cross-validation`。
- [ ] 每个 outer fold 的测试集只出现一次。
- [ ] 从 outer-train pool 中按 `validation_size=0.2`、保持类别比例划分 validation set。
- [ ] `StratifiedKFold(shuffle=True, random_state=42)`；validation split 的随机种子为 `42 + fold`。
- [ ] BEC 标准化的均值/标准差只由当前训练 fold 拟合。
- [ ] 连续表型标准化只由当前训练 fold 拟合。
- [ ] 类别编码、ADHD200 缺失值填补只由当前训练 fold 拟合。
- [ ] 患者图的训练节点、邻居参考 BEC 和融合图均按 fold 单独构建。
- [ ] PGR/QSR 只在当前 fold 的训练数据上拟合。
- [ ] 分类器用 validation loss 早停和选最优 checkpoint，test labels 不参与模型选择。
- [ ] 分类阈值由 validation set 的 Youden index 选择，再应用到 test set；AUC 使用连续概率。

### 3.2 结果汇报

主表至少报告 `ACC`, `AUC`, `Precision`, `Recall`, `F1` 的 fold mean ± standard deviation。若论文报告 specificity、敏感度或置信区间，应说明计算公式和聚合方式；当前 `classification_metrics` 默认没有在主结果字典中输出 specificity。

建议同时保存：

- 每个 fold 的 train/validation/test subject IDs；
- 每个 fold 的 best epoch、validation loss 和 test metrics；
- OOF（out-of-fold）预测概率与阈值；
- 每个表示的 BEC archive、随机种子和完整命令行。

## 4. STF-BEC 表征学习设置

### 4.1 输入与窗口

| 参数 | ABIDE-I | ABIDE-II | ADHD200 |
|---|---:|---:|---:|
| window length | 78 | 80 | 76 |
| stride | 39 | 40 | 25 |
| batch size | 32 | 32 | 32 |
| epochs | 81 | 81 | 141 |
| checkpoint | `final` | `final` | `final` |
| loss mode | `entropy` | `entropy` | `entropy` |
| loss alpha | 0.01 | 0.01 | 0.01 |

窗口采样器每个 epoch 为每名受试者确定性地抽取一个窗口；随机数由全局 seed、epoch 和 subject index 共同决定。论文应明确这是“每 epoch 每名受试者一个随机窗口”，而不是把所有滑动窗口当作相互独立的样本。

### 4.2 网络和优化器

| 参数 | 值 |
|---|---:|
| `d_model` | 16 |
| `d_inner_hid` | 64 |
| `d_k`, `d_v` | 8, 8 |
| `n_head` | 2 |
| embedding dropout | 0.2 |
| attention layers | 1 |
| attention heads | 2 |
| hidden activation | GELU |
| attention probability dropout | 0.5 |
| hidden dropout | 0.5 |
| initializer range | 0.02 |
| Adam betas | (0.9, 0.98) |
| weight decay | 0 |
| warmup steps | 4000 |
| learning-rate multiplier | 1.2 |
| scheduler | Transformer-style scheduled learning rate |

需补充：实际训练时是否使用 `--no-filters`、PyTorch/CUDA 版本、单卡还是多卡，以及 STF-BEC 是否在全体数据上预训练后再做下游 CV，或在每个 fold 内重新训练。当前 `input-mode=bec` 使用已有 BEC；`input-mode=raw` 会重新生成并覆盖对应 BEC 文件，论文必须写清楚采用哪一种。

## 5. 患者相似图与参考 BEC

### 5.1 图构建

| 参数 | 默认值 |
|---|---:|
| graph mode | `fusion` |
| number of neighbors `k` | 20 |
| fusion coefficient `beta` | 0.6 |
| phenotype bandwidth | 2.0 |
| categorical penalty | 4.0 |
| phenotype permutation | disabled |
| ABIDE continuous weights | `[1.0, 0.3]` |
| ADHD200 continuous weights | `[1.0, 0.3, 0.3]` |

`phenotype` 图使用类别表型和连续表型；`fusion` 图进一步融合训练 fold 内标准化后的 fMRI 特征相似图。fMRI 图和表型图分别取 top-k，再通过 `fusion_beta` 融合并重新保留 top-k。邻居参考 BEC 由训练集 BEC 按图权重做 normative aggregation 得到。

### 5.2 论文必须交代的消融

- [ ] `phenotype` graph vs. `fusion` graph。
- [ ] 不使用患者图，仅使用 original BEC。
- [ ] 随机/置换表型图（`--permute-phenotype`）。
- [ ] 不使用 QC 修正的 PGR-BEC。
- [ ] 仅 QC/QSR 修正或完整方法。
- [ ] `k` 和 `fusion_beta` 的敏感性分析。

## 6. PGR-BEC 与 QC/QSR-BEC 设置

### 6.1 PGR-BEC

| 参数 | ABIDE-I | ABIDE-II | ADHD200 |
|---|---:|---:|---:|
| refiner epochs | 80 | 100 | 80 |
| learning rate | 0.03 | 0.03 | 0.01 |
| gate max | 0.40 | 0.30 | 0.40 |
| gate L1 weight | 0.38 | 0.36 | 0.01 |
| anchor weight | 0.80 | 1.20 | 1.00 |
| variance weight | 1.00 | 0.80 | 1.00 |
| variance retention | 0.85 | 0.80 | 0.65 |

论文应定义原始 BEC、邻居参考 BEC、门控修正量和最终 refined BEC 的数学形式，并说明对角线是否置零。当前实现会将 BEC 对角线置零。

### 6.2 QC/QSR-BEC

所有数据集默认使用 QC 特征：`func_mean_fd`, `func_dvars`, `func_quality`。

| 参数 | ABIDE-I | ABIDE-II | ADHD200 |
|---|---:|---:|---:|
| epochs | 80 | 100 | 100 |
| learning rate | 0.003 | 0.03 | 0.001 |
| hidden channels | 8 | 8 | 8 |
| eta | 0.15 | 0.30 | 0.20 |
| r max | 0.03 | 0.18 | 0.25 |
| corruption scale | 0.5 | 0.5 | 0.5 |
| gate max | 0.50 | 0.45 | 0.38 |
| gate weight | 0.001 | 0.001 | 0.01 |
| variance weight | 0.10 | 0.25 | 0.25 |
| variance retention | 0.85 | 0.85 | 0.95 |
| basis ridge | 0.001 | 0.001 | 0.01 |

需补充：QC 指标的方向变换、标准化方法、合成 corruption 的采样方式，以及 `eta`、`r_max` 和 gate 的物理含义。建议正文给出一条完整的训练目标函数，而不是只列超参数。

## 7. 下游分类器

### 7.1 Directed BrainNetCNN

- 输入：每名受试者的 `R x R` 有向 BEC 矩阵，`R=90`。
- 输入通道：由有向矩阵转换得到 outgoing/incoming 两个通道。
- E2E channels：`(4, 8)`。
- E2N channels：16。
- N2G channels：16。
- 全连接层：8 -> 1。
- 激活：leaky ReLU，negative slope `0.1`。
- classifier dropout：`0.3`。
- 损失：`BCEWithLogitsLoss`。
- 优化器：Adam，learning rate `1e-3`，weight decay `1e-4`。
- batch size：32。

### 7.2 训练控制

| 参数 | ABIDE-I | ABIDE-II | ADHD200 |
|---|---:|---:|---:|
| max epochs | 100 | 100 | 60 |
| early-stopping patience | 20 | 20 | 12 |
| classifier repeats | 1 | 1 | 1 |
| threshold | validation Youden threshold | validation Youden threshold | validation Youden threshold |

每个表示分别训练一个分类器，默认比较：`original`, `refined`, `qc_refined`。论文中应避免把同一分类器在多个表示上的结果称为“同一模型直接对比”，应描述为“相同下游架构和训练协议下的表示比较”。

## 8. Baseline 复现登记表

当前仓库已有 baseline 参数汇总于 `Graph_BEC/baseline/parameter_settings.tex`。正式论文中建议为每个 baseline 增加以下信息：

| 记录项 | 必填内容 |
|---|---|
| 输入 | 相同 ROI、相同时间序列和相同受试者筛选 |
| 方法 | 原论文/官方实现/仓库实现 |
| 参数 | 全部关键超参数、默认值和修改项 |
| 训练 | epoch、batch size、optimizer、learning rate、早停 |
| 图/连接 | 对角线处理、是否对称、是否稀疏化、阈值 |
| 随机性 | seed、重复次数、模型选择规则 |
| 输出 | 每 fold 指标和 mean ± std |
| 公平性 | 是否使用相同 downstream classifier；若不同需解释 |

仓库当前登记的 baseline 包括 Pearson-FC、Partial-Correlation-FC、Sparse-VAR、GVAR、NAVAR、CR-VAE、NPI-MLP、VarCoNet 和 FSTA-EC。不要直接把仓库中的方法名、年份或参数视为论文最终版本；特别是未绑定唯一出版物的实现，需补充引用和版本号。

## 9. 推荐运行命令

### 9.1 使用已有 BEC

```bash
python Graph_BEC/main_abide_i.py --input-mode bec
python Graph_BEC/main_abide_ii.py --input-mode bec
python Graph_BEC/main_adhd200.py --input-mode bec
```

### 9.2 从原始 ROI 时间序列重新生成 BEC

```bash
python Graph_BEC/main_abide_i.py --input-mode raw
python Graph_BEC/main_abide_ii.py --input-mode raw
python Graph_BEC/main_adhd200.py --input-mode raw
```

### 9.3 指定表示和图模式

```bash
python Graph_BEC/main_abide_i.py \
  --graph-mode fusion \
  --representations original refined qc_refined \
  --seed 42 \
  --gpu-id 0
```

只运行 phenotype 图或只评估某个表示时：

```bash
python Graph_BEC/main_abide_i.py --graph-mode phenotype
python Graph_BEC/main_abide_i.py --representations original
python Graph_BEC/main_adhd200.py --gpu-id cpu
```

## 10. 结果表与图的最低要求

- [ ] 主结果表：每个数据集 × 每个表示/方法的 `ACC`, `AUC`, `Precision`, `Recall`, `F1` mean ± std。
- [ ] 结果表脚注：10-fold stratified CV、validation ratio、seed、分类器和阈值策略。
- [ ] 消融表：original / phenotype-refined / fusion-refined / QC-refined。
- [ ] 敏感性图：`k`、`fusion_beta`、窗口长度/步长、QC 修正强度。
- [ ] 训练曲线：STF-BEC reconstruction loss、PGR/QSR loss、分类器 validation loss。
- [ ] 数据划分图或表：每个数据集的样本数、类别数、站点数和各 fold 分布。
- [ ] 若声称跨站点泛化，新增 leave-one-site-out 或 train/test site-disjoint 实验；当前默认 10-fold stratification 并不等价于跨站点泛化。
- [ ] 若声称有效连接具有神经科学解释，报告边/ROI 选择规则、统计检验、多重比较校正和稳定性分析。

## 11. 可直接改写进论文的 Experimental Setup 模板

> We evaluated the proposed method on [datasets]. For each subject, the preprocessed resting-state fMRI signal was represented by 116 AAL ROI time series, of which the first 90 ROIs were retained to match the implementation. Each subject’s ROI time series was standardized independently along the temporal dimension. The STF-BEC encoder was trained without diagnostic labels using subject-level randomly sampled temporal windows. The resulting directed BEC matrices were used to construct fold-local patient similarity graphs and neighbor-reference BECs. All normalization, imputation, categorical encoding, graph construction, refinement, and model selection operations were fitted using the training partition of each fold only.
>
> We used 10-fold stratified cross-validation. Within each outer training partition, 20% of the subjects were held out as a validation set for early stopping and threshold selection. The downstream classifier was a Directed BrainNetCNN trained independently on the original, refined, and QC-refined BEC representations. We report the mean and standard deviation of accuracy, ROC-AUC, precision, recall, and F1 score across folds. The random seed was fixed to 42, and the test labels were used only once for final evaluation.

中文对应版本：

> 我们在[数据集名称]上评估所提出的方法。对于每名受试者，将预处理后的静息态 fMRI 信号表示为 116 个 AAL 脑区的 ROI 时间序列，并保留前 90 个 ROI 以与实现保持一致。随后对每名受试者的每个 ROI 沿时间维度独立进行标准化。STF-BEC 编码器在不使用诊断标签的条件下，以受试者级随机时间窗口进行无监督训练。得到的有向 BEC 矩阵用于在每个交叉验证训练 fold 内构建患者相似图和邻居参考 BEC。所有标准化、缺失值填补、类别编码、图构建、连接修正和模型选择操作均仅使用对应 fold 的训练划分拟合。
>
> 我们采用 10 折分层交叉验证，并从每个外层训练划分中进一步划出 20% 的受试者作为验证集，用于早停和分类阈值选择。下游分类器为 Directed BrainNetCNN，分别在原始、修正后和 QC 修正后的 BEC 表示上独立训练。我们报告各 fold 上 accuracy、ROC-AUC、precision、recall 和 F1 score 的均值与标准差。随机种子固定为 42，测试标签仅用于最终评估。

## 12. 最终提交前核对文件

- [ ] `Graph_BEC/main_abide_i.py`
- [ ] `Graph_BEC/main_abide_ii.py`
- [ ] `Graph_BEC/main_adhd200.py`
- [ ] `Graph_BEC/workflow.py`
- [ ] `Graph_BEC/downstream/classifier.py`
- [ ] `Graph_BEC/baseline/parameter_settings.tex`
- [ ] 每个数据集的 `summary.json`、fold metrics 和运行日志
- [ ] 参考论文原文中的数据集、划分、指标和超参数

若以上清单中仍有“需补充”项目，不建议在论文中使用“exactly reproduce”或“完全复现”等表述；更稳妥的表述是“按照公开实现重建实验流程，并在下述设置下进行复现实验”。
