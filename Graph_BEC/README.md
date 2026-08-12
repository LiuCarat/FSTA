# FSTA-Graph-BEC 方案说明

> 本文档根据当前 `Graph_BEC` 目录中的实际代码整理，适合用于向老师或合作人员介绍模型框架、训练过程和实验评价方式。

## 1. 研究目标

本项目的核心目标是：

> 从每名受试者的 ROI 时间序列中生成个体化、具有方向性的 BEC 矩阵，并利用不包含诊断标签的患者相似图进行轻量修正，最后通过 ASD/TC 下游分类验证 BEC 是否包含可泛化的群体差异信息。

需要区分两个概念：

- **BEC 的生成与修正**：不使用 `DX_GROUP`，避免模型直接制造 ASD/TC 的组间差异。
- **BEC 的验证**：使用 `DX_GROUP` 进行最终的组间统计和分类，因为分类是检验 BEC 是否具有诊断判别价值的下游工具。

当前实现是**静态 PGR-BEC 修正方案**，动态 refiner 已经不属于当前主流程。

## 2. 总体流程

```text
ABIDE-I ROI 时间序列
        │
        ├── raw 模式：训练 FSTA
        │                 │
        │                 └── 提取个体初始 BEC
        │
        └── bec 模式：读取已有 subject_bec.npz

个体初始 BEC A_i ∈ R^(90×90)
        │
        ├── phenotype graph：SEX + FIQ + PIQ
        │
        └── fusion graph：phenotype graph + fMRI FC graph
                         │
                         └── 得到邻居参考 BEC N_i

[A_i, N_i, |N_i - A_i|]
        │
        └── PGRBECStatic 边级门控网络
                         │
                         └── 得到 Refined BEC A_i^refined

Original BEC / Refined BEC
        │
        ├── BEC 群体统计与边效应量分析
        │
        └── 冻结 BEC 后训练 Directed BrainNetCNN
                         │
                         └── ASD/TC 下游分类
```

主入口为：

```text
Graph_BEC/FSTA_Graph_BEC.py
```

## 3. 数据输入

### 3.1 raw 模式

运行参数：

```bash
--input-mode raw
```

流程为：

```text
ABIDE-I 数据目录
→ 读取每名受试者的 AAL90 ROI 时间序列 [T, 90]
→ FSTA 无监督训练
→ 对每名受试者的多个时间窗口提取空间注意力
→ 多窗口平均
→ 得到个体 BEC [90, 90]
```

当前默认数据配置：

```text
数据目录：dataset/ABIDE-I
pipeline：cpac
derivative：rois_aal
strategy：filt_noglobal
ROI 数量：90
```

raw 模式下，FSTA 训练完成后会打印每名受试者或定期打印其 BEC 提取重建误差。若指定的 `--bec-path` 中包含同一批受试者，程序还会比较在线生成的 BEC 与归档 BEC：

```text
raw-vs-archive BEC: max_abs=... mean_abs=...
```

只有两者近似一致时，raw 与 bec 模式才具有严格可比性。

### 3.2 bec 模式

运行参数：

```bash
--input-mode bec
```

程序直接加载：

```text
subject_bec.npz
```

归档至少需要包含：

```text
bec
labels
subject_ids
site_ids
```

当前默认 BEC 文件为：

```text
downstream_abide_i/outputs/entropy/loss_alpha_0.01/seed_42/epochs_101/subject_bec.npz
```

bec 模式不重新训练 FSTA，只在已有 BEC 上执行后续患者图构建、BEC 修正和分类验证，因此适合快速比较不同图参数或 refiner 参数。

## 4. FSTA 模块：从时间序列生成 BEC

### 4.1 FSTA 的输入和输出

每个训练样本是一个时间窗口：

```text
X ∈ R^(window_length × 90)
```

当前默认参数：

```text
window_length = 78
stride = 39
epochs = 101
batch_size = 32
d_model = 16
d_inner_hid = 64
n_head = 2
d_k = 8
d_v = 8
dropout = 0.2
loss_mode = entropy
loss_alpha = 0.01
```

### 4.2 FSTA 内部结构

当前 `Graph_BEC/model/fsta_components/fsta.py` 中，FSTA 主要包含：

```text
输入时间序列
→ 1×1卷积映射到 d_model
→ 位置编码 + LayerNorm
→ Fourier Attention
→ 时序多头注意力
→ Positionwise Feed-Forward Network
→ 空间注意力融合
→ 1×1卷积解码
→ 重建时间序列 + 空间注意力矩阵
```

FSTA 已经包含 LayerNorm、残差式注意力结构和 FFN，因此当前项目并不是简单的线性模型。

### 4.3 FSTA 训练目标

FSTA 不使用 ASD/TC 分类损失。其窗口级损失为：

```text
L_FSTA = L_reconstruction + α · L_regularization
```

其中：

- `L_reconstruction`：重建时间窗口与原始时间窗口之间的均方误差；
- `L_regularization`：
  - `original` 模式使用注意力和作为正则项；
  - `entropy` 模式使用归一化注意力熵约束。

因此，FSTA 学习的是能够重建输入时间序列的时空关系，而不是直接学习 ASD/TC 分类边界。

### 4.4 个体 BEC 提取

对每名受试者的全部固定时间窗口分别运行 FSTA：

```text
每个窗口 → 空间注意力矩阵
```

然后对所有窗口的注意力矩阵求平均，并转置为当前代码定义的有向 BEC：

```python
bec = np.mean(np.stack(attentions), axis=0).T
```

BEC 的方向约定为：

```text
BEC[source, target] = source ROI → target ROI
```

当前实现保持方向性，不进行强制对称化，并将对角线置零。

## 5. 患者图模块

患者图的作用不是直接分类，而是为每名受试者提供一个弱的邻域参考：

```text
相似患者的 BEC
→ 加权平均
→ 邻居参考 BEC N_i
```

当前有两种可选图模式：

```bash
--graph-mode phenotype
--graph-mode fusion
```

### 5.1 phenotype graph

默认模式为：

```text
SEX + FIQ + PIQ
```

字段含义：

- `SEX`：类别变量；
- `FIQ`：连续变量；
- `PIQ`：连续变量。

表型数据来自：

```text
dataset/ABIDE-I/Phenotypic_Processing_filled.csv
```

其中 `DX_GROUP` 只保存用于最终分类和统计，不参与患者图构建。

### 5.2 表型距离

对连续变量先进行 fold-safe 标准化，即只用训练折拟合均值和标准差，再应用到验证折和测试折。

患者之间的距离大致为：

```text
连续特征加权平方距离
+ SEX 不一致惩罚
```

公式可以写为：

\[
d_{ij}
= \sum_m w_m (x_{im}-x_{jm})^2
+ \lambda_{sex} \cdot \mathbb{I}(sex_i \neq sex_j)
\]

当前默认参数：

```text
reference_k = 20
reference_bandwidth = 2.0
categorical_penalty = 4.0
continuous_weights = [1.0, 0.5]
```

在当前代码中，`continuous_weights` 的两个位置依次对应：

```text
FIQ、PIQ
```

`SEX` 不是连续权重，而是由 `categorical_penalty` 控制。

### 5.3 top-k 邻域和归一化

对每个查询受试者只保留距离最近的 `k` 个训练参考受试者，然后使用 RBF 形式的相似度：

\[
w_{ij} = \exp(-d_{ij}/\sigma)
\]

最后对每一行归一化，使邻居权重和为1：

\[
\sum_j W_{ij}=1
\]

重要的数据划分原则：

```text
训练受试者：只能参考训练受试者
验证受试者：只能连接训练参考受试者
测试受试者：只能连接训练参考受试者
```

这避免了测试受试者之间相互参考，也避免测试 BEC 参与邻域均值计算。

### 5.4 fusion graph

fusion 模式融合两种无诊断标签的相似性：

```text
phenotype graph
+
fMRI graph
```

fMRI 图的构建步骤为：

```text
ROI 时间序列
→ Pearson 相关矩阵 [90×90]
→ 取上三角 4005维 FC 特征
→ 训练折标准化
→ cosine 相似度
→ top-k 图
```

然后按固定系数融合：

\[
W_{fusion}
= \beta W_{fMRI} + (1-\beta)W_{phenotype}
\]

当前默认：

```text
fusion_beta = 0.60
```

之后再次保留 top-k 并进行行归一化。

fusion 模式本质上是固定权重的多视图患者图，不是 DeepASD 那种使用分类损失学习的监督图。

## 6. 邻居参考 BEC

设训练参考受试者的个体 BEC 为：

\[
A_j \in \mathbb{R}^{90 \times 90}
\]

患者图权重为：

\[
W_{ij}
\]

则第 `i` 名受试者的邻居参考 BEC 为：

\[
N_i = \sum_j W_{ij} A_j
\]

代码中通过 `normative_reference()` 执行这个加权平均。

注意：

- `N_i` 不是一个新的真实 BEC；
- 它是根据相似患者得到的参考矩阵；
- 它保留了参考邻域的群体结构；
- 它不直接使用 ASD/TC 标签。

## 7. PGR-BEC Static：静态 BEC 修正

当前修正模块为：

```text
Graph_BEC/model/pgr_bec_static.py
```

### 7.1 输入

对每名受试者构造三类边级输入：

```text
A_i              原始个体 BEC
N_i              邻居参考 BEC
|N_i - A_i|      两者绝对差异
```

组合为：

\[
[A_i, N_i, |N_i-A_i|]
\]

输入形状为：

```text
[batch, 3, 90, 90]
```

### 7.2 边级门控网络

当前网络为：

```text
3通道输入
→ 1×1 Conv，hidden_channels=16
→ LeakyReLU
→ 1×1 Conv，hidden_channels=16
→ LeakyReLU
→ 1×1 Conv，输出1通道
→ Sigmoid
→ 乘以 gate_max
```

`1×1 Conv` 对每个有向 BEC 边使用共享参数，因此可以理解为一个边级 MLP。它根据当前边的：

- 原始值；
- 邻居参考值；
- 原始值和参考值的差异；

预测该边应该使用多少邻域信息。

### 7.3 Refined BEC 公式

令门控矩阵为 `G_i`：

\[
G_i = g_{max} \cdot \sigma(\text{Gate}(A_i,N_i,|N_i-A_i|))
\]

修正结果为：

\[
A_i^{refined}
= A_i + G_i \odot (N_i-A_i)
\]

也可以写成：

\[
A_i^{refined}
= (1-G_i)\odot A_i + G_i\odot N_i
\]

因此 Refined BEC 不是完全替换 Original BEC，而是在每条有向边上进行小幅、可学习的插值。

当前实现还保证：

```text
对角线为0
保持source→target方向
不强制对称化
```

`gate_max` 控制修正强度上限。当前默认值为 `0.5`，表示理论上每条边最多向邻居参考移动50%。实际门控通常会更小。

## 8. 静态修正损失

修正器不使用 ASD/TC 分类损失，当前损失为：

\[
L_{total}
= \lambda_{anchor}L_{anchor}
+ \lambda_{gate}L_{gate}
+ \lambda_{variance}L_{variance}
\]

### 8.1 Anchor loss

\[
L_{anchor}
= SmoothL1(A^{refined}, stopgrad(A))
\]

作用是让修正后的 BEC 不要偏离原始 BEC 太远。

### 8.2 Gate sparsity loss

\[
L_{gate}=mean(|G|)
\]

作用是鼓励门控稀疏、避免所有边都大量使用邻居信息。

### 8.3 Variance retention loss

先计算原始和修正后 BEC 在受试者维度上的平均边方差：

\[
V_{original}=Var(A), \quad V_{refined}=Var(A^{refined})
\]

如果修正后方差低于原始方差的指定比例，则惩罚：

\[
L_{variance}
= max(0, rV_{original}-V_{refined})
\]

当前默认：

```text
variance_retention = 0.85
```

这用于防止患者图造成过度平滑，使所有受试者的 BEC 变得过于相似。

## 9. 每个交叉验证 fold 的训练过程

当前主程序使用分层交叉验证，并在每个 fold 中进一步划分验证集：

```text
全部受试者
→ stratified train/validation/test
```

每个 fold 的逻辑为：

```text
1. 根据 train/validation/test 索引切分 BEC、表型和标签
2. 只用 train 表型拟合连续变量标准化器
3. 构建 train-only phenotype/fusion graph
4. 从训练参考受试者 BEC 生成 train/val/test neighbor BEC
5. 只用 train BEC 和 train neighbor BEC 训练 PGRBECStatic
6. 用训练好的 refiner 处理 validation/test BEC
7. 分别对 Original BEC 和 Refined BEC 训练独立分类器
8. 使用 validation labels 选择 Youden threshold
9. 只在 test labels 上报告最终指标
```

### 9.1 防止数据泄漏

当前代码遵守以下原则：

- 患者图构建不使用 `DX_GROUP`；
- 测试受试者只连接训练参考受试者；
- BEC 标准化只使用训练折统计量；
- fMRI FC 特征标准化只使用训练折统计量；
- 分类器早停使用验证集损失；
- 分类阈值使用验证集 ROC 曲线确定；
- 测试集只用于最后一次评估；
- Original 与 Refined 分类器使用相同的 fold-local seed，保证配对比较。

## 10. Original BEC 与 Refined BEC 的区别

### Original BEC

```text
FSTA 输出的个体 BEC
→ 直接送入 Directed BrainNetCNN
```

它表示不经过患者图修正的原始个体方向性 BEC，是当前实验的基线。

### Refined BEC

```text
FSTA 输出 Original BEC
→ 根据 phenotype/fusion graph 找邻居
→ 计算邻居参考 BEC
→ PGRBECStatic 预测边级 gate
→ 得到 Refined BEC
→ 送入相同的 Directed BrainNetCNN
```

Original 和 Refined 使用相同的分类器结构、数据划分和随机种子，区别只在于输入 BEC 是否经过图引导修正。

因此，二者的差异可以解释为：

> 患者图参考和无标签边级门控是否提高了 BEC 的下游 ASD/TC 判别价值。

## 11. 下游 Directed BrainNetCNN

文件：

```text
Graph_BEC/downstream/brainnetcnn.py
```

由于 BEC 是方向性的，分类器不是把矩阵简单当作对称功能连接，而是显式使用两个方向通道：

```text
通道1：BEC[source, target]
通道2：BEC[target, source]
```

输入形状为：

```text
[batch, 2, 90, 90]
```

网络主要结构：

```text
Directed E2E
→ Directed E2E
→ Directed E2N
→ N2G
→ 全连接层
→ 二分类 logit
```

其中：

- `E2E`：提取边到边的空间关系；
- `E2N`：将边信息聚合到 ROI 节点；
- `N2G`：将节点信息聚合到全局；
- 全连接层输出 ASD/TC 分类结果。

BrainNetCNN 在当前方案中是**下游验证器**，不参与患者图构建，也不参与 Refined BEC 的训练。

## 12. 评价指标

程序会分别报告 Original 和 Refined 两种表示的：

```text
ACC
SPE
AUC
Precision
Recall
F1
```

最终格式为：

```text
65.89±6.41
```

含义是：

```text
10个fold上的平均百分比 ± fold间标准差百分比
```

此外还计算 BEC 群体差异指标：

### 12.1 Fisher ratio

将每名受试者的 `90×90` BEC 展平为向量，计算 ASD 和 TC 两组中心之间的距离，并除以组内离散程度：

\[
Fisher
= \frac{\|\mu_{ASD}-\mu_{TC}\|^2}
{D_{within}^2}
\]

Fisher ratio 越大，说明两组中心相对组内波动越分离。

当前代码还计算：

- `bec_centroid_distance`：两组 BEC 均值中心距离；
- `bec_within_dispersion`：组内离散程度；
- `mean_abs_d`：所有有向边平均绝对效应量；
- `max_abs_d`：最大有向边效应量；
- `edges_abs_d_gt_0p5`：绝对效应量大于0.5的边数；
- `variance_retention`：修正后跨受试者方差相对 Original BEC 的保留比例；
- `paired_auc_delta_mean`：Refined 相对 Original 的配对 AUC 改变量。

## 13. 输出文件

默认输出目录：

```text
Graph_BEC/outputs/
```

当前主要保留：

```text
experiment_summary.csv
summary.json
```

### experiment_summary.csv

每一行对应一个 fold，保存 Original 和 Refined 的分类指标及 BEC 统计指标。

### summary.json

保存：

- 本次实验配置；
- FSTA 训练信息；
- 各 fold 结果；
- Original/Refined 指标的均值和标准差；
- `mean±std` 展示字符串；
- BEC 群体差异和方差保留结果。

主程序在保存前会清理输出目录中除 `experiment_summary.csv` 和 `summary.json` 以外的普通文件。

## 14. 常用运行方式

### 14.1 使用已有 BEC，运行表型图修正

```bash
python Graph_BEC/FSTA_Graph_BEC.py \
  --input-mode bec \
  --graph-mode phenotype \
  --seed 42
```

### 14.2 使用已有 BEC，运行 fusion 图修正

```bash
python Graph_BEC/FSTA_Graph_BEC.py \
  --input-mode bec \
  --graph-mode fusion \
  --fusion-beta 0.60 \
  --seed 42
```

fusion 模式即使使用已有 BEC，也需要重新读取 ROI 时间序列来计算 fMRI FC 图。

### 14.3 从 ROI 时间序列端到端生成 BEC

```bash
python Graph_BEC/FSTA_Graph_BEC.py \
  --input-mode raw \
  --graph-mode phenotype \
  --seed 42
```

### 14.4 快速调试

```bash
python Graph_BEC/FSTA_Graph_BEC.py \
  --input-mode bec \
  --graph-mode phenotype \
  --max-subjects 100 \
  --n-splits 3 \
  --classifier-epochs 20 \
  --refiner-epochs 10
```

快速调试只用于检查代码和流程是否可以运行，不能作为最终实验结果。

## 15. 当前方案的科学解释

当前模型不是通过标签监督直接让 ASD 聚在一起、TC 聚在一起，而是：

```text
根据无诊断标签的表型或多模态相似性
→ 找到参考邻居
→ 用邻居 BEC 提供弱先验
→ 只对每条边进行受限修正
→ 用独立分类器验证修正后的 BEC
```

如果 Refined BEC 在严格交叉验证中获得更好的 AUC、F1、Precision 或 Recall，同时满足：

- 测试集未参与图权重拟合；
- 测试集未参与 refiner 训练；
- 分类器和随机种子公平配对；
- 方差没有严重坍缩；
- 性能提升在重复实验中稳定；

可以说明：

> 患者图引导的无标签 BEC 修正提高了 BEC 中具有泛化能力的 ASD/TC 判别信息。

但不能仅凭分类提升断言每一条 BEC 边都具有真实生物学因果意义。分类结果证明的是**诊断判别价值**，边效应量和统计分析则用于进一步定位可能存在群体差异的方向性连接。

## 16. 当前方案的限制

1. **表型图维度较低**：当前只有 `SEX + FIQ + PIQ`，相似性表达能力有限。
2. **fusion 权重固定**：`fusion-beta` 是手工设置，不是通过训练学习。
3. **refiner 的无标签信号较弱**：当前 anchor、gate sparsity 和 variance retention 主要约束修正幅度，不能直接保证邻居参考一定更接近真实 BEC。
4. **站点仍可能影响 BEC**：当前没有将 `SITE_ID` 作为患者相似性特征，但 fMRI 数据及 QC 仍可能含有站点差异。
5. **FSTA 使用全体受试者训练**：如果研究重点是严格的端到端泛化评估，需要额外设计 fold 内 FSTA 训练；当前流程更适合作为固定 BEC 表示的比较实验。
6. **分类结果需要重复验证**：单次10-fold结果可能受分类器初始化和数据划分影响，建议使用多个预先固定的重复种子进行配对比较。

## 17. 一句话汇报版本

> 本研究首先利用无监督 FSTA 从每名受试者的 AAL90 ROI 时间序列中提取具有方向性的个体 BEC；随后根据不包含诊断标签的 SEX、FIQ、PIQ 表型相似性，或表型与 fMRI 功能连接的融合相似性，构建患者参考图并生成邻居参考 BEC；最后通过一个无诊断标签的边级门控网络对原始 BEC 进行小幅修正，并将 Original BEC 与 Refined BEC 分别输入冻结流程下的 Directed BrainNetCNN，通过严格交叉验证比较 ASD/TC 分类性能及 BEC 边水平群体差异，从而验证患者图引导的 BEC 修正是否具有可泛化的诊断判别价值。
