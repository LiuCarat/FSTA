# QSR-BEC 新架构实现与汇报说明

> **QSR-BEC：QC-guided Self-supervised Refinement of BEC**  
> 更严格地说，该方法包含“QC 弱监督 + QC-like synthetic restoration”的训练机制。  
> 核心目标不是利用 QC 直接提高 ASD/TC 分类，而是利用训练集 QC 信息学习 **BEC 中可能与成像质量相关的变化模式**，对个体 BEC 做受限、方向可学习的弱修正，再用独立的 ASD/TC 分类验证修正后的 BEC 是否更有判别价值。

---

## 1. 研究动机

当前 FSTA-Graph-BEC 流程已经能够：

1. 从 ROI 时间序列中通过 FSTA 得到每名受试者的有向 Original BEC；
2. 根据 phenotype / fusion patient graph 得到 Neighbor BEC；
3. 通过 PGR-BEC 对 Original BEC 做边级弱修正；
4. 使用 Directed BrainNetCNN 对 Original / Refined BEC 做 ASD/TC 下游验证。

当前 PGR-BEC 的核心公式为：

\[
A_i^{refined}
=
A_i + G_i\odot(N_i-A_i)
\]

其中 \(A_i\) 是 Original BEC，\(N_i\) 是 Neighbor BEC，\(G_i\) 是边级 gate。

这个设计能够限制修改幅度，但训练信号主要来自：

- anchor loss；
- gate sparsity；
- variance retention。

因此它更擅长学习“**不要改太多**”，但缺少一个明确的正向监督告诉模型：

> **BEC 应该往什么方向修，什么样的变化更可能是 QC-related artifact。**

QSR-BEC 的目的就是补上这一点。

---

## 2. 新架构的核心思想

QSR-BEC 不把 QC 当作分类特征，也不使用 QC 构建新的 patient graph，更不根据 QC 删除受试者。

QC 只在 **training fold** 中承担“训练监督”的角色：

\[
\boxed{
QC
\rightarrow
学习 QC-sensitive BEC pattern
\rightarrow
构造弱修正 pseudo-target
\rightarrow
训练 BEC refiner
}
\]

测试阶段则完全不输入测试受试者自己的 QC：

\[
\boxed{
A_{test}
\rightarrow
QSR
\rightarrow
A_{test}^{refined}
}
\]

因此整个方法可以概括为：

> **QC teaches the model what QC-sensitive BEC patterns look like during training, while QC is not required during inference.**

---

## 3. 新旧架构关系

### 保持不变的模块

- FSTA：继续负责从 ROI 时间序列生成 Original BEC；
- phenotype / fusion patient graph：继续负责找到与当前受试者相关的训练参考邻居；
- Neighbor BEC：继续作为个体修正的上下文参考；
- Directed BrainNetCNN：继续只作为下游 ASD/TC 验证器；
- 交叉验证的数据泄漏控制原则保持不变。

### 主要替换的模块

原来的：

```text
Original BEC
     ↓
Neighbor BEC
     ↓
PGR-BEC Static
     ↓
Refined BEC
```

替换为：

```text
Original BEC
     │
     ├──────────── Patient Graph ───────────→ Neighbor BEC
     │
Training-fold QC
     ↓
QC artifact basis
     ↓
Pseudo-target + Synthetic QC corruption
     │
     └──────────────────────────────┐
                                    ↓
                           QSR-BEC Refiner
                                    ↓
                              Refined BEC
```

原 PGR-BEC 建议继续保留，作为新架构最重要的 baseline / ablation。

---

# 4. 总体数据流

## 4.1 训练阶段

```text
                         TRAIN FOLD ONLY

ROI Time Series
      ↓
     FSTA
      ↓
Original BEC A_i
      │
      ├──────────────────────────────┐
      │                              │
      │                     Phenotype + fMRI
      │                              ↓
      │                         Patient Graph
      │                              ↓
      │                       Neighbor BEC N_i
      │
Train QC + Confounds
FD / DVARS / Quality
SITE / AGE / SEX / FIQ / PIQ
      │
      └───────┬──────────────────────┘
              ↓
   Confound-controlled QC-BEC regression
              ↓
        QC artifact basis B_k
              ↓
       QC-sensitive map M
              │
       ┌──────┴────────┐
       ↓               ↓
Estimate subject-   Sample realistic
specific QC-like    QC change Δq
component           from train fold
       ↓               ↓
A_i^pseudo       Synthetic corruption
                       ↓
                     Ã_i
       └───────────────┬────────────────┐
                       ↓                ↓
              Path A: A_i        Path B: Ã_i
                       │                │
                       └───────┬────────┘
                               ↓
                [X_i, N_i, |N_i-X_i|, M]
                               ↓
                  Directed Context Encoder
                               ↓
                       Shared features
                         /           \
                        ↓             ↓
                     Gate G       Direction D
                         \           /
                          \         /
                           ↓       ↓
             A_ref = X + G⊙D⊙(N-X)
                               ↓
                         Refined BEC
```

训练阶段包含两个任务：

\[
\boxed{
A_i\rightarrow A_i^{pseudo}
}
\]

以及：

\[
\boxed{
\widetilde A_i\rightarrow A_i^{pseudo}
}
\]

第一条负责让网络学习对 **Original BEC 本身进行保守弱修正**；  
第二条负责让网络学习识别和恢复更明显的 **QC-like synthetic corruption**。

---

## 4.2 测试阶段

```text
Test Original BEC A_test
          │
          ├────→ Patient Graph
          │         ↓
          │   Train-reference Neighbor BEC N_test
          │
          ↓
       QSR-BEC
          ↓
   Refined BEC
          ↓
 Directed BrainNetCNN
          ↓
     ASD / TC
```

测试阶段不输入：

```text
test meanFD
test DVARS
test func_quality
DX_GROUP
```

其中 `DX_GROUP` 只用于最后的分类标签和统计验证。

---

# 5. Step 1：训练集 QC 预处理

第一版建议只使用三个互补的功能像质量指标：

```text
func_mean_fd
func_dvars
func_quality
```

原因：

- `meanFD`：反映头动程度；
- `DVARS`：反映相邻时间点全脑信号变化；
- `func_quality`：反映整体功能像质量；
- 三者信息相对互补；
- 第一版避免加入过多高度相关 QC 指标，保证模型简单、易解释。

所有 QC 统计量必须 **fold-safe**。

对于第 \(k\) 个 QC 指标：

\[
z_{ik}
=
\frac{
\log(1+x_{ik})-\mathrm{Median}_{train,k}
}{
IQR_{train,k}
}
\]

只关注比训练集典型水平更差的一侧：

\[
z_{ik}^{bad}
=
\max(z_{ik},0)
\]

训练折的 Median / IQR 保存后，再应用到 validation / test 的**分析评价**中；但 validation / test QC 不作为 QSR 输入。

---

# 6. Step 2：学习 QC artifact basis

QSR-BEC 不直接把“BEC 与 QC 的相关性”当作 artifact，因为 QC 还可能与站点、年龄、性别、IQ 等因素相关。

因此，在 **每个 training fold 内**，对每一条有向 BEC 边 \(e\) 拟合一个控制非诊断混杂因素的模型：

\[
A_{i,e}
=
\beta_{0,e}
+
\beta_{FD,e}Q_{i,FD}
+
\beta_{DVARS,e}Q_{i,DVARS}
+
\beta_{Quality,e}Q_{i,Quality}
+
\gamma_e^\top C_i
+
\epsilon_{i,e}
\]

其中：

\[
C_i=
[
Site,\ Age,\ Sex,\ FIQ,\ PIQ,\ldots
]
\]

具体控制变量以当前 phenotype 文件中实际可用字段为准。

**不加入 `DX_GROUP`。**

最终只保留 QC 对应的系数：

\[
B_{FD}=\beta_{FD}
\]

\[
B_{DVARS}=\beta_{DVARS}
\]

\[
B_{Quality}=\beta_{Quality}
\]

每个 \(B_k\) 都是：

\[
90\times90
\]

的有向矩阵。

它们不表示“真实 artifact”，更严谨的定义是：

> **在 training fold 中控制非诊断混杂因素后，与 QC 独立相关的 BEC 变化方向。**

---

# 7. Step 3：构造 QC-sensitive map

三个 QC basis 可以合成为一个 fold-level QC-sensitive map：

\[
M_e
=
Normalize
\left(
\frac{1}{3}
\sum_{k=1}^{3}
|\beta_{k,e}|
\right)
\]

因此：

\[
M\in[0,1]^{90\times90}
\]

解释为：

```text
M_e 小
→ 该 BEC 边在训练集中与 QC 的关系较弱

M_e 大
→ 该 BEC 边更容易表现出 QC-related variation
```

同一个 fold 中：

- ASD 和 TC 使用同一个 \(M\)；
- \(M\) 不包含诊断标签；
- \(M\) 是训练折学到的“QC sensitivity prior”，不是某个受试者自己的 QC 特征。

---

# 8. Step 4：为 Original BEC 构造弱 pseudo-target

这是 QSR-BEC 与最早 synthetic-denoising 版本相比最关键的改进。

如果只训练：

\[
\widetilde A_i\rightarrow A_i
\]

网络只能学会去掉“人为后来加进去的 corruption”，却没有理由修改原始 \(A_i\) 中已经存在的真实 QC-related component。

所以必须为 Original BEC 本身构造一个非常保守的弱目标。

对受试者 \(i\)，估计其 QC-like component：

\[
R_i^{QC}
=
\sum_k
z_{ik}^{bad}B_k
\]

然后只减去其中很小的一部分：

\[
\boxed{
A_i^{pseudo}
=
A_i-\eta R_i^{QC}
}
\]

其中 \(\eta\) 是较小的弱修正系数。

另外设置全矩阵修改比例上限：

\[
\frac{
\|A_i^{pseudo}-A_i\|_F
}{
\|A_i\|_F
}
\le r_{max}
\]

例如可以将初始实验限制在几个百分点的变化范围内。

### 重要解释

\[
A_i^{pseudo}
\neq
A_i^{clean}
\]

我们不知道真实 clean BEC。

所以论文和汇报中应将 \(A_i^{pseudo}\) 描述为：

> **QC-guided conservative pseudo-target**

即：

> 根据训练集 QC–BEC 关系得到的、受修改幅度约束的弱修正目标。

---

# 9. Step 5：生成真实分布风格的 QC-like synthetic corruption

第二条训练路径需要人为制造可控的 QC-like perturbation。

不建议分别独立随机采样：

```text
δFD
δDVARS
δQuality
```

因为这些指标在真实数据中往往相关，独立采样容易产生不真实的 QC 组合。

更合适的方法是在 training fold 中从真实 QC 联合分布采样变化：

\[
\Delta q=q_a-q_b
\]

其中 \(q_a,q_b\) 来自两个真实训练受试者，或者使用训练 QC covariance 进行联合采样。

再将这个 QC 变化映射到 BEC 空间：

\[
\Delta_i^{QC}
=
s
\sum_k
\Delta q_k B_k
\]

并基于 pseudo-target 生成：

\[
\boxed{
\widetilde A_i
=
A_i^{pseudo}
+
\Delta_i^{QC}
}
\]

这样就得到一个明确的训练对：

\[
\boxed{
\widetilde A_i
\rightarrow
A_i^{pseudo}
}
\]

网络知道 synthetic corruption 是如何产生的，因此得到比单纯 anchor loss 更明确的正向学习信号。

---

# 10. Step 6：Neighbor BEC 仍然保留，但只作为 context

patient graph 不需要因为 QC 改写。

继续使用当前：

```text
phenotype graph:
SEX + FIQ + PIQ
```

或：

```text
fusion graph:
phenotype similarity + fMRI FC similarity
```

并继续遵守：

```text
train → 只参考 train
validation → 只参考 train
test → 只参考 train
```

得到：

\[
N_i=\sum_jW_{ij}A_j
\]

QSR-BEC 中，Neighbor BEC 的定位是：

> **个体修正的上下文参考，而不是 ground truth。**

因此模型不会被强制：

\[
A_i\rightarrow N_i
\]

而是只允许利用 \(N_i-A_i\) 提供一个受限的局部修正尺度和方向参考。

---

# 11. Step 7：QSR-BEC Refiner

## 11.1 输入

对于当前输入矩阵 \(X_i\)：

训练 Path A：

\[
X_i=A_i
\]

训练 Path B：

\[
X_i=\widetilde A_i
\]

测试：

\[
X_i=A_i
\]

Refiner 输入：

\[
\boxed{
[X_i,\ N_i,\ |N_i-X_i|,\ M]
}
\]

即 4 个 \(90\times90\) 通道：

1. 当前 BEC；
2. Neighbor BEC；
3. 当前 BEC 与邻居参考的绝对差异；
4. training-fold QC-sensitive map。

---

## 11.2 Directed Context Encoder

原 PGR-BEC 使用多层 `1×1 Conv`，基本上让每条边相对独立判断。

QSR-BEC 建议加入一个轻量的 **Directed Context Encoder**，使一条边的修正可以参考相关 ROI 的入边 / 出边上下文。

建议结构保持轻量：

```text
4-channel input
      ↓
1×1 Conv / edge embedding
      ↓
Directed E2E block
      ↓
Directed E2E block
      ↓
shared directed edge features
```

不需要 Transformer、GAT 或复杂大模型。

这样既能利用当前 Directed BrainNetCNN 已经采用的方向性图结构思想，又不会让 refiner 过度复杂。

---

## 11.3 两个输出头：Gate + Direction

网络输出：

### Gate head

\[
G_i
=
g_{max}\cdot\sigma(H_g)
\]

其中：

\[
G_i\in[0,g_{max}]
\]

第一版可以保持：

\[
g_{max}=0.5
\]

表示每条边仍然只能做受限修改。

### Direction head

\[
D_i=\tanh(H_d)
\]

因此：

\[
D_i\in[-1,1]
\]

含义：

```text
D > 0
→ 向 Neighbor 方向修正

D ≈ 0
→ 基本保持自身

D < 0
→ 允许轻微远离 Neighbor
```

---

# 12. Step 8：最终弱修正公式

QSR-BEC 的最终修正为：

\[
\boxed{
A_i^{refined}
=
X_i
+
G_i
\odot
D_i
\odot
(N_i-X_i)
}
\]

这个公式相比原 PGR：

\[
A_i^{refined}
=
A_i
+
G_i\odot(N_i-A_i)
\]

多了一个可学习的方向项 \(D_i\)。

因此 QSR 可以学习：

- 要不要改；
- 改多少；
- 是否应该沿 Neighbor 方向；
- 某些情况下是否应该轻微远离 Neighbor。

同时仍然满足：

\[
|\Delta A|
\le
g_{max}|N-X|
\]

所以它依然是 **weak refinement**，不是无限制重写 BEC。

---

# 13. 一个具体数值例子

假设受试者 S01 某条有向边：

\[
ROI_{12}\rightarrow ROI_{35}
\]

Original BEC：

\[
A=0.420
\]

训练集 QC 回归得到：

\[
B_{FD}=0.040
\]

\[
B_{DVARS}=0.020
\]

\[
B_{Quality}=0.010
\]

S01 的异常侧 QC z-score：

\[
z_{FD}^{bad}=1.2
\]

\[
z_{DVARS}^{bad}=0.6
\]

\[
z_{Quality}^{bad}=0
\]

则估计的 QC-like component：

\[
R^{QC}
=
1.2\times0.040
+
0.6\times0.020
=
0.060
\]

若：

\[
\eta=0.2
\]

则弱 pseudo-target：

\[
A^{pseudo}
=
0.420-0.2\times0.060
=
0.408
\]

假设 patient graph 得到该边的 Neighbor BEC：

\[
N=0.350
\]

QSR 对 Original BEC 推理时输出：

\[
G=0.20,\qquad D=0.80
\]

则：

\[
A^{refined}
=
0.420
+
0.20\times0.80\times(0.350-0.420)
\]

\[
A^{refined}
\approx0.409
\]

最后：

```text
Original BEC     = 0.420
Pseudo target    = 0.408
QSR output       = 0.409
Neighbor BEC     = 0.350
```

可以看到：

- 模型没有直接复制 Neighbor；
- QC 只是提供“可能应轻微下降”的训练信号；
- 网络最终学习到参考 Neighbor 的方向，但只做非常小的修正。

这就是 QSR-BEC 想实现的效果。

---

# 14. Step 9：训练目标

第一版建议不加入 EMA Teacher，先把核心问题验证清楚。

总损失：

\[
\boxed{
L
=
L_{pseudo}
+
L_{restore}
+
\lambda_gL_{gate}
+
\lambda_vL_{variance}
}
\]

## 14.1 Original weak-correction loss

\[
L_{pseudo}
=
SmoothL1
\left(
f_\theta(A_i,N_i,M),
A_i^{pseudo}
\right)
\]

它负责：

\[
A_i
\rightarrow
A_i^{pseudo}
\]

即真正教模型对 Original BEC 做保守弱修正。

---

## 14.2 Synthetic restoration loss

\[
L_{restore}
=
SmoothL1
\left(
f_\theta(\widetilde A_i,N_i,M),
A_i^{pseudo}
\right)
\]

它负责：

\[
\widetilde A_i
\rightarrow
A_i^{pseudo}
\]

即增强模型识别 QC-like perturbation 的能力。

---

## 14.3 Gate regularization

\[
L_{gate}=mean(|G|)
\]

用于防止所有边都进行大幅修改。

---

## 14.4 Variance retention

继续沿用当前 PGR 中防止跨受试者方差坍缩的思想：

\[
L_{variance}
=
\max
\left(
0,
rV_{original}-V_{refined}
\right)
\]

避免所有受试者被 patient graph 修得过于相似。

第一版可以从：

```text
λ_gate = 0.1
λ_variance = 0.1
```

附近开始做训练折内调节，而不是使用 test performance 选择参数。

---

# 15. Fold 内严格训练流程

每个交叉验证 fold 建议按下面顺序执行：

```text
1. 划分 train / validation / test

2. FSTA BEC 使用当前既定策略读取或生成

3. 仅用 train phenotype / fMRI
   拟合 patient graph 所需 scaler

4. 构建：
   W_train
   W_val
   W_test
   并保证 val/test 只参考 train

5. 仅用 train QC
   拟合 QC robust scaler

6. 仅用 train BEC + train QC + train confounds
   拟合 confound-controlled QC artifact basis

7. 由 train basis 得到 QC-sensitive map M

8. 为 train subject 计算 A_pseudo

9. 从 train QC 联合分布采样 Δq
   构造 train synthetic corrupted BEC Ã

10. 计算 train Neighbor BEC N_train

11. 用两条训练路径训练 QSR：
    A       → A_pseudo
    Ã       → A_pseudo

12. 使用固定的 train-fold QSR：
    validation Original BEC → Refined BEC
    test Original BEC       → Refined BEC

13. validation/test 推理时：
    不输入它们自己的 QC

14. Original / PGR / QSR Refined BEC
    分别送入相同 Directed BrainNetCNN

15. validation labels 选择分类 threshold

16. test labels 只用于最终性能报告
```

### 最重要的数据泄漏原则

以下信息都不能跨 fold 拟合：

- phenotype scaler；
- fMRI FC scaler；
- QC median / IQR；
- QC artifact basis；
- QC-sensitive map；
- synthetic QC 分布；
- QSR 参数；
- BrainNetCNN 参数。

所有这些均只能由当前 training fold 得到。

---

# 16. 建议代码实现方式

在尽量不破坏当前代码结构的前提下，可以采用“新增 QSR，保留 PGR”的方式。

## 16.1 建议新增 QC supervision 模块

例如：

```text
Graph_BEC/qc/qc_supervision.py
```

主要函数：

```python
fit_qc_scaler(train_qc)
transform_qc(qc, scaler)

fit_qc_artifact_basis(
    train_bec,
    train_qc,
    train_confounds
)

build_qc_sensitive_map(qc_basis)

build_pseudo_target(
    bec,
    qc_badness,
    qc_basis,
    eta,
    r_max
)

sample_joint_qc_delta(train_qc)

qc_corrupt(
    pseudo_bec,
    qc_basis,
    delta_q,
    scale
)
```

---

## 16.2 建议新增 QSR Refiner

例如：

```text
Graph_BEC/model/qsr_bec.py
```

主要组件：

```text
QSRBECRefiner
├── input projection
├── DirectedContextEncoder
├── gate_head
└── direction_head
```

forward 输入：

```python
x
neighbor
qc_sensitive_map
```

内部构造：

```python
[x, neighbor, abs(neighbor - x), qc_sensitive_map]
```

输出：

```python
refined_bec
gate
direction
```

---

## 16.3 修改主流程

当前主入口：

```text
Graph_BEC/FSTA_Graph_BEC.py
```

建议新增参数，例如：

```bash
--refiner-mode pgr
--refiner-mode qsr
```

这样可以在完全相同的数据划分和分类器条件下直接比较：

```text
PGR
vs
QSR
```

并增加 QSR 相关参数：

```text
--qsr-eta
--qsr-r-max
--qsr-gate-max
--qsr-lambda-gate
--qsr-lambda-variance
--qsr-corruption-scale
```

---

# 17. 推荐实验设计

主实验不要一次堆很多模型。

建议核心比较：

| Model | Patient Graph | QC supervision | Direction Head | Test QC input |
|---|---:|---:|---:|---:|
| Original BEC | × | × | × | × |
| PGR-BEC | ✓ | × | × | × |
| QSR-BEC | ✓ | ✓ | ✓ | × |

如果需要更完整的 ablation，可增加：

| Ablation | 目的 |
|---|---|
| QSR w/o Direction | 验证 Direction Head 是否必要 |
| QSR w/o QC-sensitive map | 验证 \(M\) 是否提供有效先验 |
| QSR w/o synthetic restoration | 验证 synthetic QC corruption 的作用 |
| QSR shuffled QC | 验证提升是否真正依赖 QC 结构 |
| Analytic \(A^{pseudo}\) | 判断网络是否只是机械执行 QC 回归 |

其中 **shuffled QC** 非常关键：

\[
QSR_{trueQC}
>
QSR_{shuffledQC}
\]

才更有力地说明方法利用的是 QC 中有意义的结构，而不是额外训练步骤本身。

`A^{pseudo}` 直接分类只建议作为 **QC-dependent analytic comparator**，用于判断 QSR 是否超过机械线性修正；它不是主方法，也不能与“测试阶段不需要 QC”的 QSR 等同解释。

---

# 18. 评价指标

不能只看分类 AUC 是否提高。

建议同时验证四个方面。

## 18.1 下游分类

Original / PGR / QSR 使用相同：

- fold；
- seed；
- BrainNetCNN；
- optimizer；
- epoch / early stopping；
- validation threshold 逻辑。

报告：

```text
ACC
SEN / Recall
SPE
Precision
F1
AUC
```

最重要的是配对比较：

\[
\Delta AUC
=
AUC_{QSR}-AUC_{Original/PGR}
\]

---

## 18.2 QC–BEC association

对每条 BEC 边计算：

\[
\rho_e
=
Spearman(BEC_e,QC)
\]

比较：

```text
Original BEC
vs
PGR BEC
vs
QSR BEC
```

理想情况：

\[
|Corr(QSR\ BEC,QC)|
<
|Corr(Original\ BEC,QC)|
\]

同时不能出现组间信息和个体差异完全被抹掉。

---

## 18.3 弱修改程度

报告：

\[
\frac{
\|A^{refined}-A\|_F
}{
\|A\|_F
}
\]

证明 QSR 做的是弱修正，而不是重构整个 BEC。

还可以报告：

```text
mean gate
95% gate quantile
modified-edge ratio
```

---

## 18.4 Individuality / variance retention

继续报告：

\[
\frac{
V_{refined}
}{
V_{original}
}
\]

避免模型为了降低 QC association 而把不同受试者全部拉向相似均值。

---

# 19. 什么结果才能真正支持 QSR-BEC？

理想结果不是单独：

\[
AUC\uparrow
\]

而是同时满足：

\[
\boxed{
\begin{aligned}
&Classification:\quad AUC/F1\uparrow\\
&QC\ association:\quad |Corr(BEC,QC)|\downarrow\\
&Correction:\quad \|\Delta A\|/\|A\|\ \text{较小}\\
&Individuality:\quad Variance\ retention\ \text{较高}\\
&Specificity:\quad True\ QC>Shuffled\ QC
\end{aligned}
}
\]

如果进一步观察到：

\[
QSR>A^{pseudo}
\]

则说明：

> QSR 并不是简单照抄线性 QC regression，而是利用 Original BEC、Neighbor context 和 directed connectivity pattern 学到了更有价值的个体化弱修正。

---

# 20. 与旧 Soft QC-Filter 的区别

早期方案是：

\[
W_{ij}
\rightarrow
W_{ij}q_j
\rightarrow
N_i^{QC}
\]

也就是根据邻居 QC 给 patient-graph message 降权。

它的优势是简单，但本质上仍然是手工设计 reliability weight。

QSR-BEC 则变成：

\[
\boxed{
QC
\rightarrow
训练监督
\rightarrow
学习 QC-sensitive BEC pattern
\rightarrow
个体化 weak refinement
}
\]

因此两者解决的问题不同：

```text
Soft QC-Filter:
“这个邻居值得信多少？”

QSR-BEC:
“当前 BEC 是否表现出训练阶段学到的 QC-sensitive pattern，
如果有，应该在邻居 context 的约束下如何小幅修正？”
```

所以 QSR-BEC 更接近一个真正可学习的新 BEC refinement 方法。

---

# 21. 需要向老师主动说明的限制

## 21.1 QC basis 不是因果 artifact map

QC–BEC regression 得到的是关联方向，不应声称：

> 这些边一定是由 motion / QC 导致的。

更准确的表述是：

> 在控制已建模混杂因素后，与 QC 独立相关的 BEC variation。

---

## 21.2 Pseudo-target 不是 clean ground truth

\[
A^{pseudo}
\]

只是一个弱监督目标。

因此方法应描述为：

```text
QC-guided weak refinement
```

而不是：

```text
recover the true clean BEC
```

---

## 21.3 QC 与疾病可能相关

如果 ASD 与 TC 的运动水平本身存在差异，过强 QC correction 可能误删一部分与疾病共同变化的信息。

所以必须通过：

- 小 \(\eta\)；
- \(r_{max}\)；
- gate limit；
- variance retention；
- shuffled QC；
- group-effect / classification retention；

共同证明模型没有把疾病差异一起消除。

---

## 21.4 Synthetic corruption 的真实性很关键

QSR 的效果很大程度取决于 synthetic QC perturbation 是否接近真实数据中的 QC 变化模式。

因此第一版优先使用：

```text
train-fold empirical joint QC difference
```

而不是独立 Gaussian noise。

---

# 22. 论文 / 汇报中的方法定位

建议把 QSR-BEC 定位为：

> **一种 training-only QC-guided 的方向性 BEC 弱修正框架。模型首先在训练折内学习控制非诊断混杂因素后的 QC-sensitive BEC variation，并据此生成保守 pseudo-target 和现实分布风格的 QC-like synthetic perturbation；随后通过包含 directed context、gate 与 direction head 的 refiner 学习对个体 BEC 进行受限修正。推理阶段不需要 QC 或诊断标签，最终仅通过独立 Directed BrainNetCNN 和 QC–BEC association 分析验证修正后的 BEC 是否具有更好的诊断判别价值和更低的质量混杂。**

---

# 23. 一句话版本

> **QSR-BEC 不把 QC 当成分类特征，而是在训练阶段把 QC 转化为 BEC 修正监督：先学习 QC-sensitive BEC 变化，生成保守 pseudo-target 和 QC-like synthetic corruption，再让方向感知 refiner 学习“哪里需要改、改多少、往哪改”；测试时完全不输入 QC，只根据受试者自己的 BEC 与训练参考邻域生成 Refined BEC，最后用独立 ASD/TC 分类和 QC–BEC 关联下降来验证方法是否有效。**

---

# 24. 给老师汇报时建议重点讲的 5 句话

1. **原 PGR 最大的问题不是不能训练，而是缺少“正确修正方向”的正向监督。**
2. **QSR 不使用 ASD/TC 标签训练，也不把 QC 直接输入测试模型。**
3. **QC 只在 training fold 中学习 QC-sensitive BEC variation，并构造保守 pseudo-target。**
4. **QSR 同时学习 Original → pseudo-target 和 synthetic corrupted → pseudo-target，两条路径分别解决真实弱修正与 QC-like artifact 识别。**
5. **最终有效性不能只看 AUC，还要同时看到 QC–BEC association 下降、修改幅度小、方差保留，以及 true QC 优于 shuffled QC。**

---

## 最终架构名称

\[
\boxed{
\textbf{QSR-BEC:
QC-guided Self-supervised Refinement of Brain Effective Connectivity}
}
\]

更严谨的中文描述：

> **QC 引导的弱监督 / 自监督有向脑有效连接弱修正网络。**
