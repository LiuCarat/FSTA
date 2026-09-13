# 方法部分写作参考：基于时空特征与患者相似图的个体化有向脑有效连接建模

> **使用说明**：本文档是一份可直接改写为论文“方法（Methods）”章节的参考稿。方括号中的内容需要根据最终实验设置、数据集名称和统计结果替换。为保证论文表述准确，建议在定稿前将所有默认参数与实际运行配置逐项核对。

## 1. 方法概述

本文提出一个由**个体化有向脑有效连接估计、患者相似图参考建模以及质量控制引导的连接修正**组成的分析框架。给定每名受试者的 ROI 时间序列，首先通过无监督时空特征编码器学习时间动态和脑区间交互，并利用空间自注意力矩阵构造个体化有向脑有效连接（brain effective connectivity, BEC）。随后，根据受试者的表型信息和静息态 fMRI 功能连接特征建立患者相似图，将训练集中的相似受试者连接模式聚合为当前受试者的参考 BEC。最后，将原始 BEC、邻居参考 BEC 以及由质量控制（quality control, QC）指标估计的敏感性先验输入质量控制引导的修正网络，得到更加稳健的个体化连接表示，并在冻结的 BEC 表示上完成下游分类。

整个流程不使用诊断标签来生成 BEC、构建患者图或训练连接修正器；诊断标签仅用于下游分类器训练和最终性能评估。

```text
ROI 时间序列
      │
      ├─ 每名受试者独立标准化
      ├─ 滑动窗口采样
      ▼
时空特征编码器（频域建模 + 时间注意力 + 空间注意力）
      │
      ├─ 窗口级空间注意力
      └─ 跨窗口平均与转置
      ▼
个体化原始有向 BEC
      │
      ├─ 表型相似图
      ├─ fMRI 功能连接相似图（可选）
      └─ 多视图图融合
      ▼
训练集邻居参考 BEC
      │
      ├─ QC 指标变换与混杂控制
      ├─ QC 敏感性先验
      └─ 门控式连接修正
      ▼
质量控制引导的修正 BEC
      │
      ▼
有向 BrainNetCNN 分类与交叉验证评估
```

---

## 2. 符号约定

| 符号 | 含义 |
|---|---|
| $S$ | 受试者数量 |
| $T$ | 单名受试者的时间点数 |
| $R$ | ROI 数量，本文实现中为 $90$ |
| $X^{(s)}\in\mathbb{R}^{T\times R}$ | 第 $s$ 名受试者的 ROI 时间序列 |
| $L$ | 滑动窗口长度 |
| $H$ | 滑动步长 |
| $D$ | 隐藏特征维度 |
| $B^{(s)}\in\mathbb{R}^{R\times R}$ | 第 $s$ 名受试者的有向 BEC 矩阵 |
| $G\in\mathbb{R}^{S\times S}$ | 受试者相似图权重矩阵 |
| $\widetilde{B}^{(s)}$ | 受患者相似图参考后的邻居 BEC |
| $Q^{(s)}$ | 第 $s$ 名受试者的 QC 指标向量 |
| $M\in[0,1]^{R\times R}$ | QC 敏感性先验图 |
| $\widehat{B}^{(s)}$ | 最终修正后的 BEC |

除特别说明外，矩阵的主对角线均设置为零，以排除脑区自身连接对有效连接建模的影响。

---

## 3. 数据预处理

### 3.1 ROI 时间序列整理

对每名受试者读取 ROI 时间序列，并统一保留前 $R=90$ 个 ROI。对于第 $s$ 名受试者，原始时间序列记为 $X^{(s)}_{\mathrm{raw}}\in\mathbb{R}^{T_s\times R}$。为减弱不同 ROI 的量纲差异，对每个受试者、每个 ROI 沿时间维度进行 z-score 标准化：

$$
X^{(s)}_{t,r}=
\frac{X^{(s)}_{\mathrm{raw},t,r}-\mu^{(s)}_r}
{\sigma^{(s)}_r},
\qquad
\mu^{(s)}_r=\frac{1}{T_s}\sum_{t=1}^{T_s}X^{(s)}_{\mathrm{raw},t,r},
$$

$$
\sigma^{(s)}_r=
\sqrt{\frac{1}{T_s}\sum_{t=1}^{T_s}
\left(X^{(s)}_{\mathrm{raw},t,r}-\mu^{(s)}_r\right)^2}.
$$

当某一 ROI 的标准差接近零时，将其标准差置为 $1$，避免数值不稳定。表型信息、站点信息和 QC 信息按照受试者 ID 与时间序列严格对齐。

### 3.2 滑动窗口

为学习局部时空动态，将标准化后的时间序列划分为长度为 $L$、步长为 $H$ 的重叠窗口：

$$
X^{(s,w)}=X^{(s)}_{a_w:a_w+L-1,:},
$$

其中窗口起点为

$$
a_w=wH,\qquad
w=0,1,\ldots,W_s-1.
$$

若受试者包含多个有效扫描区间，则仅在各有效区间内生成窗口，以避免跨区间拼接引入人工时间依赖。

---

## 4. 时空特征编码器与原始 BEC 生成

### 4.1 输入嵌入与位置编码

对一个批次的窗口输入 $X\in\mathbb{R}^{B\times L\times R}$，首先通过 $1\times1$ 卷积将每个时间点和 ROI 映射到 $D$ 维隐空间：

$$
E=\mathrm{Conv}_{1\times1}(X)\in\mathbb{R}^{B\times L\times R\times D}.
$$

为了保留时间顺序，在时间维加入固定的正弦位置编码 $P$：

$$
P_{t,2i}=\sin\left(\frac{t}{10000^{2i/D}}\right),\qquad
P_{t,2i+1}=\cos\left(\frac{t}{10000^{2i/D}}\right).
$$

经 dropout 和层归一化后得到：

$$
Z_0=\mathrm{LayerNorm}\big(\mathrm{Dropout}(E+P)\big).
$$

### 4.2 频域特征建模

为捕获 ROI 信号中的周期性和跨时间尺度变化，对时间维执行实数快速傅里叶变换：

$$
\mathcal{F}(Z_0)=\mathrm{rFFT}(Z_0,\mathrm{dim}=t).
$$

在频域对每个频率、ROI 和隐藏通道学习复数权重 $W_f$，再通过逆傅里叶变换恢复到时域：

$$
Z_f=\mathrm{iFFT}\left(\mathcal{F}(Z_0)\odot W_f\right).
$$

其中 $\odot$ 表示逐元素复数乘法。频域模块采用残差连接和层归一化：

$$
Z_f\leftarrow\mathrm{LayerNorm}\big(Z_0+\mathrm{Dropout}(Z_f)\big).
$$

这一模块能够在不显式计算传统频谱统计量的情况下学习时间序列的频率响应。

### 4.3 时间维自注意力

将特征转置为 $[B,R,L,D]$，对每个 ROI 独立建模其时间依赖。对于第 $h$ 个注意力头：

$$
Q_h=Z_fW_h^Q,\qquad K_h=Z_fW_h^K,\qquad V_h=Z_fW_h^V,
$$

$$
A_h^{\mathrm{time}}=
\mathrm{Softmax}\left(\frac{Q_hK_h^{\mathsf T}}{\sqrt{d_k}}\right),
\qquad
Z_t=\mathrm{Concat}_h(A_h^{\mathrm{time}}V_h)W^O.
$$

经过位置前馈网络、残差连接和层归一化后，得到时间增强特征 $Z_t$。位置前馈网络可写为：

$$
\mathrm{FFN}(z)=W_2\,\mathrm{ReLU}(W_1z+b_1)+b_2.
$$

### 4.4 ROI 空间自注意力与有向连接

在 $[B,L,R,D]$ 排列下，对 ROI 维进行多头自注意力，得到空间注意力矩阵：

$$
A_h^{\mathrm{space}}=
\mathrm{Softmax}\left(\frac{Q_h^{\mathrm{space}}
(K_h^{\mathrm{space}})^{\mathsf T}}{\sqrt{d_k}}\right).
$$

实现中对批次、时间位置和注意力头进行平均，得到窗口级 ROI 关系矩阵：

$$
A^{(w)}=\frac{1}{BLH}\sum_{b=1}^{B}\sum_{t=1}^{L}\sum_{h=1}^{H_a}
A_{b,t,h}^{\mathrm{space}}\in\mathbb{R}^{R\times R},
$$

其中 $H_a$ 表示注意力头数。矩阵元素 $A^{(w)}_{ij}$ 表示第 $i$ 个 ROI 对第 $j$ 个 ROI 的注意权重。为使矩阵行列方向与有向边定义保持一致，将窗口级注意力矩阵转置后作为有向 BEC：

$$
B^{(s,w)}=\left(A^{(s,w)}\right)^{\mathsf T},
\qquad
B^{(s,w)}_{ij}=A^{(s,w)}_{ji}.
$$

因此，$B_{ij}$ 可解释为从源 ROI $i$ 指向目标 ROI $j$ 的有向连接强度。对所有窗口进行平均，得到受试者级原始 BEC：

$$
B^{(s)}=\frac{1}{W_s}\sum_{w=1}^{W_s}B^{(s,w)},
\qquad
B^{(s)}_{ii}=0.
$$

### 4.5 无监督训练目标

编码器通过重建窗口输入进行无监督训练。设重建结果为 $\widehat{X}$，重建损失为均方误差：

$$
\mathcal{L}_{\mathrm{rec}}=
\frac{1}{BLR}\sum_{b=1}^{B}\sum_{t=1}^{L}\sum_{r=1}^{R}
\left(\widehat{X}_{b,t,r}-X_{b,t,r}\right)^2.
$$

为避免空间注意力过度集中于少数 ROI，进一步对注意力分布施加归一化熵正则项。令 $p_i$ 为某一注意力行归一化后的概率，则：

$$
\mathcal{L}_{\mathrm{ent}}=
-\frac{1}{\log R}\sum_{i=1}^{R}p_i\log(p_i+\epsilon).
$$

最终训练目标为：

$$
\mathcal{L}_{\mathrm{STF}}=
\mathcal{L}_{\mathrm{rec}}+\alpha\mathcal{L}_{\mathrm{ent}},
$$

其中 $\alpha$ 为正则项权重，$\epsilon$ 为防止 $\log 0$ 的小常数。该训练过程不需要诊断标签，因此生成的 BEC 表示可用于无监督表征学习。

---

## 5. 患者相似图与邻居参考 BEC

### 5.1 表型特征距离

对于每名受试者，使用连续表型特征和离散表型特征构造患者相似图。设连续特征为 $c_s$，离散特征为 $u_s$，则受试者 $s$ 与 $q$ 的加权距离定义为：

$$
d_{sq}=\sum_{m=1}^{M_c}\omega_m(c_{s,m}-c_{q,m})^2
 +\lambda_c\sum_{n=1}^{M_d}\mathbb{I}(u_{s,n}\neq u_{q,n}),
$$

其中 $\omega_m$ 是连续表型权重，$\lambda_c$ 是离散特征不一致惩罚。对每个查询受试者仅保留距离最小的 $K$ 个训练集邻居，并采用径向基函数转换为亲和度：

$$
\widetilde{g}^{\mathrm{ph}}_{sq}=
\exp\left(-\frac{d_{sq}}{\tau}\right),
$$

$$
g^{\mathrm{ph}}_{sq}=
\frac{\widetilde{g}^{\mathrm{ph}}_{sq}}
{\sum_{q'\in\mathcal{N}_K(s)}\widetilde{g}^{\mathrm{ph}}_{sq'}}
\quad q\in\mathcal{N}_K(s),
$$

其余边权置为零。$\tau$ 为带宽参数。

### 5.2 fMRI 功能连接相似图

首先由每名受试者的 ROI 时间序列计算 Pearson 相关矩阵，并提取其上三角元素形成功能连接特征向量 $f_s$。对特征进行训练集统计量标准化后，使用余弦相似度构造 fMRI 图：

$$
\mathrm{sim}_{sq}^{\mathrm{fmri}}=
\max\left(0,\frac{f_s^{\mathsf T}f_q}
{\|f_s\|_2\|f_q\|_2}\right).
$$

同样保留每个查询受试者的 top-$K$ 邻居，并进行行归一化，得到 $G^{\mathrm{fmri}}$。对训练集内部建图时排除自身连接，避免受试者将自身作为最相似邻居。

### 5.3 多视图图融合

当采用融合图模式时，将表型图和 fMRI 图进行加权融合：

$$
G^{\mathrm{fuse}}=
\beta G^{\mathrm{fmri}}+(1-\beta)G^{\mathrm{ph}},
$$

其中 $\beta\in[0,1]$ 控制功能连接视图的贡献。融合后再次保留 top-$K$ 边并行归一化，得到最终患者相似图 $G$。若仅使用表型图，则令 $G=G^{\mathrm{ph}}$。

### 5.4 邻居参考 BEC

为避免数据泄漏，在每个交叉验证折中，参考库仅由训练集受试者构成。设训练集 BEC 为 $\{B^{(j)}\}_{j\in\mathcal{T}}$，查询受试者 $s$ 到训练受试者 $j$ 的图权重为 $g_{sj}$，则邻居参考 BEC 为：

$$
\widetilde{B}^{(s)}=
\sum_{j\in\mathcal{T}}g_{sj}B^{(j)},
\qquad
\sum_{j\in\mathcal{T}}g_{sj}=1.
$$

该参考矩阵可理解为与受试者 $s$ 在表型和/或功能连接空间相似的训练受试者的规范性连接模式。它不直接替换原始 BEC，而是作为后续门控修正的上下文信息。

---

## 6. QC 引导的 BEC 修正

### 6.1 QC 指标标准化

设原始 QC 指标为 $Q^{(s)}\in\mathbb{R}^{M_q}$。首先使用训练折统计量填补缺失值，并进行非负截断的对数变换：

$$
z^{(s)}_m=\log\left(1+\max(Q^{(s)}_m,0)\right).
$$

然后基于训练折的中位数和四分位距进行稳健标准化，并仅保留高于训练中位数的“坏度”部分：

$$
q^{(s)}_m=\max\left(\frac{z^{(s)}_m-\mathrm{median}(z_m)}
{\mathrm{IQR}(z_m)},0\right).
$$

这样得到的 $q^{(s)}$ 可同时表达头动、时间序列波动和整体质量等 QC 维度的异常程度。

### 6.2 混杂控制的 QC 相关连接基

为了区分 QC 相关变化和非诊断混杂因素的影响，构造设计矩阵：

$$
Z=\left[\mathbf{1},Q_{\mathrm{bad}},C\right],
$$

其中 $C$ 包含站点、年龄、性别及其他预先指定的非诊断协变量。对每个有向边 $(i,j)$，使用带岭正则的多元线性回归估计系数：

$$
\Theta=\arg\min_{\Theta}
\left\|B-Z\Theta\right\|_F^2
 +\lambda_r\left\|\Theta_{-0}\right\|_F^2.
$$

取对应于 QC 特征的系数并恢复为矩阵形式，得到 QC 相关连接基：

$$
\mathcal{A}=\{A_m\}_{m=1}^{M_q},
\qquad A_m\in\mathbb{R}^{R\times R}.
$$

进一步利用各 QC 维度的绝对回归系数平均值构造敏感性先验：

$$
M_{ij}=\frac{1}{M_q}\sum_{m=1}^{M_q}|A_{m,ij}|,
$$

并将 $M$ 线性归一化到 $[0,1]$，同时将对角线置零。$M_{ij}$ 越大，表示边 $(i,j)$ 与 QC 变化的统计关联越强。

### 6.3 QC 伪目标

利用 QC 相关连接基对原始 BEC 进行保守校正，构造训练时的弱监督伪目标：

$$
B_{\mathrm{prop}}^{(s)}=
B^{(s)}-\eta\sum_{m=1}^{M_q}q^{(s)}_mA_m,
$$

其中 $\eta$ 控制 QC 校正幅度。为防止伪目标偏离原始 BEC 过大，对每名受试者施加相对 Frobenius 范数约束：

$$
\frac{\|B_{\mathrm{pseudo}}^{(s)}-B^{(s)}\|_F}
{\|B^{(s)}\|_F+\epsilon}\le r_{\max}.
$$

这里 $B_{\mathrm{pseudo}}^{(s)}$ 表示截断后的伪目标。

### 6.4 门控式修正网络

对当前 BEC、邻居参考 BEC、二者的绝对差异以及 QC 敏感性先验进行通道拼接：

$$
U^{(s)}=\mathrm{Concat}\left(
B^{(s)},\widetilde{B}^{(s)},
\left|\widetilde{B}^{(s)}-B^{(s)}\right|,M
\right).
$$

经过 $1\times1$ 输入投影和两个有向边到边（edge-to-edge）上下文层后，分别预测门控图 $G^{(s)}$ 和方向图 $D^{(s)}$：

$$
G^{(s)}=g_{\max}\cdot\sigma\left(f_g(U^{(s)})\right),
$$

$$
D^{(s)}=\tanh\left(f_d(U^{(s)})\right).
$$

最终修正结果为：

$$
\widehat{B}^{(s)}=
B^{(s)}+G^{(s)}\odot D^{(s)}
\odot\left(\widetilde{B}^{(s)}-B^{(s)}\right),
\qquad \widehat{B}^{(s)}_{ii}=0.
$$

其中 $G^{(s)}$ 控制每条边的修正幅度，$D^{(s)}$ 控制修正方向，$g_{\max}$ 限制最大门控强度。因此，该模型能够在保留个体差异的同时，选择性地将受试者连接模式向其相似邻居的规范性模式靠拢。

### 6.5 修正网络训练目标

修正网络采用弱监督自监督训练。第一项约束原始输入经过修正后接近 QC 伪目标：

$$
\mathcal{L}_{\mathrm{pseudo}}=
\mathrm{SmoothL1}\left(\widehat{B}_{\mathrm{orig}},B_{\mathrm{pseudo}}\right).
$$

为提高模型对 QC 相关扰动的恢复能力，从训练折中两个受试者的 QC 坏度向量差异中采样联合扰动 $\Delta q$，并构造：

$$
B_{\mathrm{corr}}^{(s)}=B_{\mathrm{pseudo}}^{(s)}
 +\rho\sum_{m=1}^{M_q}\Delta q_mA_m.
$$

将该扰动输入修正网络，并要求其输出仍接近伪目标：

$$
\mathcal{L}_{\mathrm{restore}}=
\mathrm{SmoothL1}\left(\widehat{B}_{\mathrm{corr}},B_{\mathrm{pseudo}}\right).
$$

同时，为鼓励稀疏、保守的修正，引入门控正则项：

$$
\mathcal{L}_{\mathrm{gate}}=
\frac{1}{2}\left(
\|G_{\mathrm{orig}}\|_1+\|G_{\mathrm{corr}}\|_1
\right).
$$

为避免过度平滑导致个体间连接差异消失，引入方差保持项：

$$
\mathcal{L}_{\mathrm{var}}=
\max\left(\gamma\,V_{\mathrm{orig}}-V_{\mathrm{refined}},0\right),
$$

其中 $V$ 表示跨受试者、跨边的平均方差，$\gamma$ 为最低方差保持比例。最终损失为：

$$
\mathcal{L}_{\mathrm{QSR}}=
\mathcal{L}_{\mathrm{pseudo}}
 +\mathcal{L}_{\mathrm{restore}}
 +\lambda_g\mathcal{L}_{\mathrm{gate}}
 +\lambda_v\mathcal{L}_{\mathrm{var}}.
$$

上述训练仅使用当前训练折的 BEC、QC 和非诊断混杂信息，不使用验证折或测试折的 QC 数值来拟合修正器。

---

## 7. 下游有向脑网络分类

为了检验不同 BEC 表示的判别能力，将每个有向 BEC 分解为两个输入通道：

$$
\Phi(B)=\left[B, B^{\mathsf T}\right]
\in\mathbb{R}^{2\times R\times R}.
$$

两个通道分别表示正向和反向连接信息。随后使用有向 BrainNetCNN，包括：

1. 两层有向 edge-to-edge 卷积，用于提取边之间的全局拓扑关系；
2. 一层有向 edge-to-node 卷积，将边特征聚合到 ROI；
3. node-to-graph 卷积和全连接层，输出二分类 logit。

分类器使用二元交叉熵损失：

$$
\mathcal{L}_{\mathrm{cls}}=
-\frac{1}{N}\sum_{s=1}^{N}
\left[y_s\log p_s+(1-y_s)\log(1-p_s)\right].
$$

模型选择仅依据验证集损失，并采用 early stopping。分类阈值由验证集 ROC 曲线上的 Youden 指数确定：

$$
t^*=\arg\max_t\left[\mathrm{TPR}(t)-\mathrm{FPR}(t)\right].
$$

最终在测试集上报告 Accuracy、AUC、Precision、Recall 和 F1-score。

---

## 8. 交叉验证与防止数据泄漏

采用分层 $K$ 折交叉验证，并在每个外层折中划分训练集、验证集和测试集。具体原则如下：

1. **训练折**：拟合表型图、功能连接图的标准化统计量、QC 标准化参数、QC 连接基、敏感性先验、修正网络和分类器。
2. **验证折**：仅使用训练折拟合得到的图权重、参考 BEC、QC 先验和修正网络；验证标签仅用于分类器早停和阈值选择。
3. **测试折**：保持完全冻结，仅用于生成最终预测和计算测试指标。
4. **参考库限制**：验证和测试受试者的邻居参考 BEC 只能由训练折 BEC 加权得到。
5. **诊断标签隔离**：诊断标签不参与 BEC 生成、患者图构建、QC 先验估计或 BEC 修正器训练。

第 $k$ 折的测试输出可记为：

$$
\mathcal{Y}^{(k)}=\mathrm{Classifier}\left(
\mathrm{Refiner}\left(B_{\mathrm{test}}^{(k)},
\widetilde{B}_{\mathrm{test}}^{(k)},M^{(k)}\right)\right).
$$

所有折的测试结果取均值和标准差，作为最终性能报告。若需要保存全体受试者的修正 BEC，应仅回填每名受试者在其作为测试样本时得到的 out-of-fold 结果，避免将同一受试者的训练内预测误作为泛化结果。

---

## 9. 可直接放入论文的简短版本

下面的段落可作为方法章节的压缩版，再根据篇幅要求补充小节和参数表：

> 对每名受试者的 ROI 时间序列进行逐 ROI z-score 标准化，并采用重叠滑动窗口获得局部时间片段。随后，构建一个无监督时空特征编码器：首先通过线性嵌入和正弦位置编码获得时空隐表示；然后在时间维执行可学习频域滤波，以捕获多时间尺度动态；进一步分别沿时间维和 ROI 维施加多头自注意力，从而建模时间依赖和脑区间交互。编码器通过窗口重建损失联合注意力熵正则进行训练。对每个窗口，将 ROI 空间自注意力矩阵转置为有向连接矩阵，并在窗口维度求平均，得到个体化原始 BEC。
>
> 为利用个体间结构相似性，在每个交叉验证折内仅使用训练受试者构建患者相似图。相似图由表型距离和 fMRI 功能连接余弦相似度组成，并通过 top-$K$ 稀疏化和行归一化获得邻接权重。对每个查询受试者，将训练集 BEC 按相似图权重加权平均，得到邻居参考 BEC。与此同时，基于训练折 QC 指标和非诊断混杂变量估计 QC 相关连接基，并由其幅值构造边级 QC 敏感性先验。修正网络以原始 BEC、邻居参考 BEC、二者差异和 QC 敏感性图为输入，通过门控和方向分支学习保守的边级修正。训练目标由伪目标拟合损失、扰动恢复损失、门控稀疏正则和跨受试者方差保持项组成。最后，将原始或修正后的有向 BEC 转换为正向/反向双通道，并输入有向 BrainNetCNN 完成分类。所有图构建、QC 先验估计和修正器训练均在训练折内完成，以避免数据泄漏。

---

## 10. 参数表模板

| 类别 | 参数 | 符号 | 实际值 |
|---|---|---:|---:|
| 输入 | ROI 数量 | $R$ | 90 |
| 输入 | 窗口长度 | $L$ | [填写] |
| 输入 | 滑动步长 | $H$ | [填写] |
| 编码器 | 隐藏维度 | $D$ | [填写] |
| 编码器 | 注意力头数 | $H_a$ | [填写] |
| 编码器 | 熵正则权重 | $\alpha$ | [填写] |
| 患者图 | 邻居数 | $K$ | [填写] |
| 患者图 | 融合权重 | $\beta$ | [填写] |
| 患者图 | RBF 带宽 | $\tau$ | [填写] |
| QC 修正 | 伪目标幅度 | $\eta$ | [填写] |
| QC 修正 | 相对变化上限 | $r_{\max}$ | [填写] |
| QC 修正 | 门控权重 | $\lambda_g$ | [填写] |
| QC 修正 | 方差权重 | $\lambda_v$ | [填写] |
| 评估 | 交叉验证折数 | $K_{\mathrm{cv}}$ | [填写] |

## 11. 写作时建议明确的事项

- 明确 ROI 模板、ROI 数量、时间序列长度处理方式和窗口参数。
- 明确患者图使用哪些表型变量、哪些 QC 变量，以及连续变量权重的设置依据。
- 明确融合图中 $\beta$ 的定义，避免将“fMRI 权重”与“表型权重”写反。
- 明确所有标准化、插补和回归参数均只由训练折估计。
- 明确原始 BEC、邻居参考 BEC 和最终修正 BEC 的方向定义。
- 报告最终使用的随机种子、交叉验证折数、early stopping 策略和分类阈值选择方式。
- 将消融实验作为独立实验设置描述，不与主方法流程混写。
