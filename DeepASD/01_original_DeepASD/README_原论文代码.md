# 原论文 DeepASD 代码说明

## 1. 这个文件夹是什么

本文件夹保存上传压缩包中的 DeepASD 原论文核心实现。以下源代码保持原样，没有改成 BEC：

```text
layers.py
main.py
model.py
train.py
utils.py
run.sh
LICENSE
```

压缩包中与模型无关的 VS Code 缓存、日志和 `__pycache__` 没有保留。

## 2. 原代码完成的任务

原 DeepASD 的主要任务是 ASD/TC 分类：

```text
多个受试者级模态特征
        ↓
不同维度模态投影到共同空间
        ↓
对抗正则化
        ↓
建立受试者—受试者群体图
        ↓
SSGC
        ↓
ASD / TC 分类
```

原代码中的图是受试者群体图，形状大致为：

```text
受试者数 × 受试者数
```

它不是每名受试者的脑区连接图，所以原代码本身不会输出：

```text
90 × 90 个体 BEC
```

## 3. 主要文件

### `model.py`

包含：

- `VariDim_Projection`：不同维度模态投影；
- `Discriminator`：对抗判别器；
- `VariModal_GraphLearn`：受试者群体图学习；
- `SSGC`：最终 ASD/TC 分类模型。

### `train.py`

包含 DeepASD、MLP 和对抗模型的训练、验证与测试过程。

### `main.py`

读取作者预先处理好的 ABIDE_A 或 ABIDE_B 特征数据，并进行交叉验证分类。

### `layers.py`

图卷积与 SSGConv 等基础层。

## 4. 原代码的数据要求

原代码不是直接读取：

```text
*_rois_aal.1D
Phenotypic_V1_0b_preprocessed1.csv
```

它要求作者事先生成的受试者级多模态特征，例如：

```text
ABIDE_A/processed_standard_data.csv
ABIDE_A/modal_feat_dict.npy
```

或：

```text
ABIDE_B/RFE_512_processed_standard_data.npz
```

因此，不建议用这个文件夹直接生成 BEC。它主要用于：

1. 对照原论文实现；
2. 查看多模态共同空间和对抗正则化方式；
3. 将原始分类结果作为实验基线。

## 5. 原代码运行方式

作者的默认入口是：

```bash
python main.py --dataset ABIDE_B --model DeepASD
```

但只有在对应的预处理特征文件已经放入原代码预期的目录后才能运行。

请先查看 `main.py` 中：

```text
--datadir
--dataset
```

以及 ABIDE_A、ABIDE_B 两个数据读取分支。

## 6. 与 BEC 修改版的关系

本文件夹没有被修改。

真正用于 ABIDE-I、AAL90 和个体 BEC 生成的代码位于相邻文件夹：

```text
02_individual_BEC_AAL90
```

BEC 修改版只借鉴 DeepASD 的以下思想：

- 不同表型组映射到共同维度；
- 对抗约束减少不同表型组尺度差异；
- 学习个体化表示。

它不沿用原论文最后的 ASD/TC 分类头，也不把受试者群体图误当作脑区 BEC。
