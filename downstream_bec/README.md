# 个体水平有向 BEC 下游分类

本目录用于评估 FSTA 生成的个体脑有效连接（BEC）是否包含 BD/HC 判别信息，与原有的组间因果分析流程完全独立。

以下原始文件不会被修改：

```text
model/FSTA.py
model/STMultiHeadAtt.py
train_FSTA_BDCore20_multi.py
```

## 整体流程

```text
全部170名 BD + HC 受试者时序
              ↓
共享 SubjectFSTA 无标签训练
              ↓
生成170张连续有向个体 BEC
              ↓
固定 BEC，不再更新 SubjectFSTA
              ↓
Directed BrainNetCNN 分层五折交叉验证
              ↓
ACC、SEN、SPE、AUC
```

具体步骤如下：

1. 使用全部170名 BD+HC 受试者训练一个共享的无标签 `SubjectFSTA`。
2. 空间注意力只平均时间维度和注意力头维度，保留受试者维度。
3. 为每名受试者导出一张连续有向 BEC，不进行阈值化或二值化。
4. 固定生成的170张 BEC，只对 `DirectedBrainNetCNN` 进行分层五折交叉验证。
5. 使用多个 FSTA 随机种子重复实验，并可构建不依赖诊断标签的共识 BEC。

## 方向约定

模型直接输出的注意力采用：

```text
raw_attention[target, source]
```

为了与原有组级 FSTA 输出的方向约定一致，保存的 BEC 定义为：

```text
bec[source, target]
```

两者关系为：

```python
bec = raw_attention.transpose(-1, -2)
```

`bec[i, j]` 表示脑区 `i → j` 的有向连接。BEC 对角线设置为0，但不会对非对角连接进行对称化、阈值化或二值化。

## 数据情况

当前 BDCore20 数据包括：

```text
HC：121名
BD：49名
总计：170名
脑区数量：20
实际时序长度：152
```

原始文本共有153行，其中第一行为表头；按照原项目的 `skiprows=1` 读取后，每名受试者实际得到 `[152,20]` 的时序矩阵。

## 训练共享 SubjectFSTA

建议在包含 PyTorch 和 scikit-learn 的 `fs2g` Conda 环境中运行：

```bash
conda activate fs2g
```

也可以直接使用解释器：

```text
/data/users/liulin/miniconda3/envs/fs2g/bin/python
```

### 原始损失复现版

原始损失由时序重建误差和 `sum(adj)` 正则项组成：

```bash
python -m downstream_bec.train_shared_fsta \
  --loss_mode original_sum \
  --loss_alpha 0.8 \
  --seeds 2026 \
  --gpu_id auto
```

### 熵约束改进版

熵约束通过最小化注意力分布的归一化熵，鼓励每个脑区将连接权重集中到更少的目标脑区：

```bash
python -m downstream_bec.train_shared_fsta \
  --loss_mode entropy \
  --loss_alpha 0.1 \
  --seeds 2026 \
  --gpu_id auto
```

建议先使用一个种子完成调试，然后再扩展为多个种子：

```bash
python -m downstream_bec.train_shared_fsta \
  --loss_mode entropy \
  --loss_alpha 0.1 \
  --seeds 2026,2027,2028,2029,2030 \
  --gpu_id auto
```

正式实验可以进一步扩展到10个预先指定的随机种子。不要根据下游分类 AUC 选择表现最好的 FSTA 随机种子。

## SubjectFSTA 输出

每个随机种子的输出目录为：

```text
downstream_bec/outputs/shared_fsta/
└── entropy/
    └── seed_2026/
        ├── model.pt
        ├── training_history.csv
        ├── bec_summary.json
        ├── subject_bec.npz
        └── individual/
            ├── sub-10159_HC.npy
            ├── ...
            └── sub-60001_BD.npy
```

`subject_bec.npz` 包含：

```text
raw_attention [170,20,20]  模型直接输出的注意力矩阵
bec           [170,20,20]  转置并清除对角线后的连续有向 BEC
labels        [170]        HC=0，BD=1
subject_ids   [170]        受试者编号
roi_names     [20]         脑区名称及矩阵顺序
```

可以使用以下代码检查文件：

```python
import numpy as np

data = np.load(
    "downstream_bec/outputs/shared_fsta/entropy/seed_2026/subject_bec.npz"
)

print(data.files)
print(data["raw_attention"].shape)
print(data["bec"].shape)
print(data["labels"].shape)
print(data["subject_ids"][:5])
print(data["roi_names"])
```

## BrainNetCNN 五折分类

BrainNetCNN 只读取已经生成并固定的 BEC，不会反向修改 SubjectFSTA 或 BEC。

运行原始损失复现版 BEC 的五折分类：

```bash
python -m downstream_bec.train_brainnetcnn_cv \
  --bec_path downstream_bec/outputs/shared_fsta/original_sum/seed_2026/subject_bec.npz \
  --output_dir downstream_bec/outputs/brainnetcnn_cv/original_sum/seed_2026 \
  --gpu_id auto
```

运行熵约束版 BEC 的五折分类：

```bash
python -m downstream_bec.train_brainnetcnn_cv \
  --bec_path downstream_bec/outputs/shared_fsta/entropy/seed_2026/subject_bec.npz \
  --output_dir downstream_bec/outputs/brainnetcnn_cv/entropy/seed_2026 \
  --gpu_id auto
```

分类阶段采用：

```text
外层：StratifiedKFold，5折
内层：从当前训练折中划分20%作为验证集
阳性类别：BD=1
分类阈值：0.5
模型选择：验证集 AUC
类别不平衡：BCEWithLogitsLoss 的 pos_weight
```

所有 FSTA 随机种子共用同一份划分文件：

```text
downstream_bec/splits/brainnetcnn_5fold_seed42.json
```

这样可以将 FSTA 初始化差异与 BrainNetCNN 数据划分差异分开。

## 分类评价指标

只计算以下四个指标：

```text
ACC：准确率
SEN：敏感性，BD 识别率
SPE：特异性，HC 识别率
AUC：使用 BD 连续预测概率计算的 ROC-AUC
```

每次五折分类会输出：

```text
fold_01/ ... fold_05/
fold_metrics.csv
oof_predictions.csv
oof_metrics.json
```

`fold_metrics.csv` 保存每折的 ACC、SEN、SPE、AUC 及五折均值和标准差；`oof_metrics.json` 使用全部170名受试者的折外预测计算总体指标。

## 构建多种子共识 BEC

完成多个 FSTA 随机种子的 BEC 生成后，可以对每名受试者的同一条有向边求平均：

```bash
python -m downstream_bec.build_consensus \
  --inputs downstream_bec/outputs/shared_fsta/entropy/seed_*/subject_bec.npz \
  --output_dir downstream_bec/outputs/consensus/entropy
```

输出包括：

```text
consensus_bec.npz
bec_stability.csv
```

共识计算不会把 `BEC[i,j]` 与 `BEC[j,i]` 合并，因此不会破坏方向性，也不会进行二值化。

`bec_stability.csv` 报告不同 FSTA 随机种子之间的：

```text
个体 BEC 平均相关性
个体 BEC 相关性标准差
组平均 BEC 相关性
```

## 推荐实验顺序

```text
1. 使用 seed=2026 跑通 original_sum 复现版
2. 使用 seed=2026 跑通 entropy 改进版
3. 分别完成两种损失对应的 BrainNetCNN 五折分类
4. 扩展到5个 FSTA 随机种子检查稳定性
5. 正式实验扩展到10个预先指定的随机种子
6. 报告各随机种子的 ACC、SEN、SPE、AUC 均值和标准差
7. 构建共识 BEC 并完成补充分类实验
```

## 当前推荐的改进实验

当前默认 BrainNetCNN 已调整为更小的网络：

```text
E2E：2 → 4
E2E：4 → 8
E2N：8 → 16
N2G：16 → 16
FC：16 → 8
Dropout：0.5
学习率：3e-4
Weight decay：1e-3
```

每个外层 Fold 会使用验证集 ROC 的 Youden Index 选择分类阈值，再将该阈值固定应用到测试集。测试集标签不会参与阈值选择。

### 第一步：重新运行小型 BrainNetCNN

不要覆盖之前的结果，使用新的输出目录：

```bash
python -m downstream_bec.train_brainnetcnn_cv \
  --bec_path downstream_bec/outputs/shared_fsta/original_sum/seed_2026/subject_bec.npz \
  --output_dir downstream_bec/outputs/brainnetcnn_cv_small/original_sum/seed_2026 \
  --gpu_id auto
```

### 第二步：运行传统分类基线

以下命令使用完全相同的 Train/Validation/Test 受试者划分，同时运行逻辑回归和随机森林：

```bash
python -m downstream_bec.train_classical_baselines_cv \
  --bec_path downstream_bec/outputs/shared_fsta/original_sum/seed_2026/subject_bec.npz \
  --output_dir downstream_bec/outputs/classical_cv/original_sum/seed_2026
```

判断方法：

```text
传统模型和 BrainNetCNN 都较差：BEC 中的 BD/HC 信号较弱
传统模型明显更好：BrainNetCNN 仍然过拟合或结构不合适
BrainNetCNN 更好：矩阵拓扑结构提供了额外判别信息
```

### 第三步：重新测试熵系数

当前不建议继续使用 `loss_alpha=0.1`。先测试：

```text
0.0001
0.001
0.01
```

建议优先运行：

```bash
python -m downstream_bec.train_shared_fsta \
  --loss_mode entropy \
  --loss_alpha 0.001 \
  --seeds 2026 \
  --output_root downstream_bec/outputs/shared_fsta_entropy_0001 \
  --gpu_id auto
```

然后运行小型 BrainNetCNN：

```bash
python -m downstream_bec.train_brainnetcnn_cv \
  --bec_path downstream_bec/outputs/shared_fsta_entropy_0001/entropy/seed_2026/subject_bec.npz \
  --output_dir downstream_bec/outputs/brainnetcnn_cv_small/entropy_0001/seed_2026 \
  --gpu_id auto
```

不要根据单次测试集 AUC 反复选择参数。先使用 `seed=2026` 比较少量预先指定的配置，确定合理范围后，再固定配置运行5个或10个 FSTA 随机种子。
