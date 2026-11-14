# W7 / Q3：Subject-specific Feature Fusion 消融

开发阶段仅检查代码和历史配置元数据，并执行 CPU 合成张量 / 模拟流程测试。
**没有运行真实队列预训练、EC 提取、十折分类或结果分析。所有实验命令由用户手动执行。**

## 1. 严格控制的计算差异

原始空间注意力张量为 `[B, T, H, N, N]`。

- `batch_mean`（默认）：保留原运算顺序 `mean(B) → mean(T) → mean(H)`，
  得到 `[N, N]`，对整个 minibatch 的空间特征使用同一个融合矩阵。
- `subject_specific`：`mean(T) → mean(H)` 得到 `[B, N, N]`；在解码融合时
  扩展成 `[B, 1, N, N]`，与 `[B, T, d_model, N]` 做 batched matmul。
  每个样本只使用自己的矩阵，没有 detach 或阻断注意力梯度的操作。
- 两种模式都单独保留**原运算顺序**计算的 `[N, N]` batch-mean attention，
  entropy loss 始终作用于它，而不是对逐样本 entropy 求平均。
- 时间注意力仍使用原默认归约；Q/K/V、Fourier、解码器参数与参数量不变。
- `forward()` 默认仍返回二元组 `(reconstruction, fusion_attention)`；训练内部
  显式请求第三项 batch-mean attention 供正则化使用，避免将三维个人矩阵误传 loss。

推理时两组都逐被试、逐窗口、`batch_size=1` 调用模型，按时间和 heads 平均，
再平均该被试所有窗口，保留转置、对角线置零和 `[90,90]` 格式。
在相同权重下，B=1 时两模式的重建和提取 EC 一致；真正训练后的权重不同，
因此**不预设真实新 EC 或新分类结果与原结果一致**。

## 2. 已核查的历史配置与公平性

检查了 `PR_EC/outputs/seed42/<dataset>/summary.json` 的配置及记录的训练 epoch，
没有计算其中的分类结果：

| 数据集 | epochs / 记录完成 epoch | window / stride | batch size | checkpoint |
| --- | --- | --- | --- | --- |
| ABIDE-I | 81 / 81 | 78 / 39 | 32 | final |
| ABIDE-II | 81 / 81 | 80 / 40 | 32 | final |
| ADHD200 | 141 / 141 | 76 / 25 | 32 | final |

三者均为 seed42、十折、验证比例 0.2、PH+FC、K20、fusion beta 0.6。
当前 E2E 入口除路径、输入方式及选择报告的表示外，已有模型超参数与这些历史元数据相符。
正式运行前代码仍会重新校验，发现超参数偏离就报错，而不是自动重新调参。

需要特别防止的已有差异：

1. 原 E2E `runner.py` 实际导入 `PR_EC.data`，其 raw 分支又调用 `PR_EC` 编码器，
   仅改本地 encoder 不会生效。本次将入口接到本地 data / Initial EC 模块。
2. 原 E2E 还包含 restoration / target-correction 等其他消融，以及
   `run_e2e.py` 的另一套比较流程。W7 **不使用这些替代下游流程**，
   而是直接调用未修改的 `PR_EC.workflow.run_cross_validation`，复用原论文
   的 MPR、QSR、BrainNetCNN、损失、优化器、分类阈值和验证集模型选择代码。
3. E2E 不同入口原来选择的表示不完全相同。W7 固定使用论文 seed42 已记录的
   `original/refined/qc_refined`，同时得到 Initial、MPR-only 和 Full 分类。
4. 预训练始终是全队列无标签时间序列的 transductive 协议；`train_stf_ec`
   接收时间序列，不接收被试诊断标签。标签只用于来源一致性检查、固定折划分和下游监督。
   不调用 `PR_EC_Inductive`，不引入五折或每折重训初始编码器。
5. 新 Initial EC 必须和 Original 参考档案具有同序 subject IDs、labels、site IDs；
   从原档案继承 ROI 名称顺序。预训练前就核对固定十折的测试 fold IDs 与原 QSR OOF 一致。
   不一致会停止，不会悄悄更改样本或重新搜索划分。
6. 新缓存记录 fusion mode、完整 STF 配置、entropy 定义和 transductive 协议。
   `subject_specific` 不接受缺少新来源记录的旧缓存；两模式互换路径也会报错。
7. 代码核查和历史 JSON 元数据不能单独证明历史文件内容从未被改动；正式运行时的
   身份 / fold 校验会再检查实际参考档案。原始 ROI 文件及软件环境应保持论文版本。

## 3. 修改文件

- `model/individual_ec/temporal_attention.py`：显式选择 batch / subject 归约，保留旧 batch mean。
- `model/individual_ec/encoder.py`：只在空间分支切换模式；正确扩展 batch 维度；兼容旧返回接口。
- `model/individual_ec/training.py`：重建使用选定融合方式，entropy 显式使用旧 batch mean。
- `model/individual_ec/utils.py`：loss 拒绝非 batch-mean 矩阵；提取兼容二维 / 三维注意力。
- `model/individual_ec/ec_estimator.py`：保存 Initial EC 的融合配置、训练来源和 ROI 元数据。
- 三个 `main_*.py`：增加统一参数和 W7 分阶段入口。
- `data/__init__.py`：接通本地编码器；保存前后核验来源；pretrain-only 不进入 FC/QSR/分类。
- `runner.py`：仅显式 W7 或 subject-specific 请求转入独立 W7 runner；默认旧入口仍保留。
- `w7.py`：路径隔离、历史配置校验、缓存防混用、被试和折检查、指标导出。
- `w7_runner.py`：预训练 / 下游 / 一次完成三种阶段，直接复用原 Full PR-EC 下游函数。
- `utils/output.py`：删除目录级清理循环，只覆盖自己的命名结果文件，保留配置等产物。
- `analysis/summarize_table_r8.py`：实际十折指标汇总及来源核验，提供 Table R8。
- `tests/test_w7_fusion.py`：17 项 CPU 合成或模拟流程检查，不读取真实队列。
- `REBUTTAL_W7.md`：本说明与完整运行命令。

未修改 `PR_EC`、W6、Inductive 或任何其他目录中的代码。

## 4. 输出与安全约束

新实验固定输出在本目录内：

```text
PR_EC_E2E/outputs/rebuttal_w7/<dataset>/subject_specific/
```

初始缓存文件名分别为：

```text
abide-i/subject_specific/abide_subject_specific_subject_ec.npz
abide-ii/subject_specific/abide_ii_subject_specific_subject_ec.npz
adhd200/subject_specific/adhd200_subject_specific_subject_ec.npz
```

其余主要文件：

- `<dataset>_refined_subject_ec.npz`：原格式的测试 OOF PGR/QSR 合并档案。
- `<dataset>_qsr_refined_subject_ec.npz`：独立测试 OOF QSR EC；保留 IDs、labels、site、fold 和 ROI 顺序。
- `classification_folds.csv`：三种表示各十折 ACC、AUC、SEN、PRE、F1，单位为百分比。
- `classification_summary.csv`：每种表示的 mean、std、mean ± std；沿用原实现 `ddof=0`。
- 原有 `experiment_summary.csv`、`summary.json`。
- `protocol_config.json`：完整锁定配置，包含所有模型超参数和原始参考路径。
- `pretraining_config.json`、`downstream_config.json`：对应阶段的完整配置快照。
- `original_config_audit.json`：与历史 seed42 元数据逐项比较结果。
- `pretraining_metrics.json`、阶段完成标记、`split_membership.csv`。

已存在 Initial EC 时 raw 阶段会拒绝覆盖，应改用 downstream 阶段复用。
输出目录含有另一模式或另一套参数时会报错。Original 参考文件只读。
下述命令不重训 Original；Table R8 默认直接使用请求中给出的论文基线数字。

## 5. 完整运行命令：仅供用户手动执行

在仓库根目录、同一个 Bash 会话中执行。这里的解释器已通过轻量测试；
正式实验应使用与原论文相同的软件环境，可按实际情况替换 `PYTHON`。

```bash
PYTHON=/storage/users/liulin/miniconda3/envs/default/bin/python
ORIGINAL=PR_EC/outputs/seed42
COMMON=(--stf-fusion-mode subject_specific --seed 42 --n-splits 10 --validation-size 0.2 --batch-size 32 --stf-checkpoint final --graph-mode fusion --reference-k 20 --fusion-beta 0.6 --representations original refined qc_refined --gpu-id auto)
```

### A. Subject-specific Initial EC 全队列预训练 + 逐被试提取

此阶段不执行十折下游分类。

```bash
"$PYTHON" PR_EC_E2E/main_abide_i.py "${COMMON[@]}" \
  --input-mode raw --w7-stage pretrain --epochs 81 --window-length 78 --stride 39 \
  --reference-ec-path "$ORIGINAL/abide-i/abide_subject_ec.npz"

"$PYTHON" PR_EC_E2E/main_abide_ii.py "${COMMON[@]}" \
  --input-mode raw --w7-stage pretrain --epochs 81 --window-length 80 --stride 40 \
  --reference-ec-path "$ORIGINAL/abide-ii/abide_ii_subject_ec.npz"

"$PYTHON" PR_EC_E2E/main_adhd200.py "${COMMON[@]}" \
  --input-mode raw --w7-stage pretrain --epochs 141 --window-length 76 --stride 25 \
  --reference-ec-path "$ORIGINAL/adhd200/adhd200_subject_ec.npz"
```

### B. 使用新 Initial EC：Initial + MPR-only + Full PR-EC 十折分类

自动读取 A 阶段相应数据集的新缓存；不会再次预训练初始编码器。

```bash
"$PYTHON" PR_EC_E2E/main_abide_i.py "${COMMON[@]}" \
  --input-mode ec --w7-stage downstream --epochs 81 --window-length 78 --stride 39 \
  --reference-ec-path "$ORIGINAL/abide-i/abide_subject_ec.npz"

"$PYTHON" PR_EC_E2E/main_abide_ii.py "${COMMON[@]}" \
  --input-mode ec --w7-stage downstream --epochs 81 --window-length 80 --stride 40 \
  --reference-ec-path "$ORIGINAL/abide-ii/abide_ii_subject_ec.npz"

"$PYTHON" PR_EC_E2E/main_adhd200.py "${COMMON[@]}" \
  --input-mode ec --w7-stage downstream --epochs 141 --window-length 76 --stride 25 \
  --reference-ec-path "$ORIGINAL/adhd200/adhd200_subject_ec.npz"
```

如需一次完成，可将对应 A 命令的 `--w7-stage pretrain` 换成 `--w7-stage all`；
不要在已经完成 A 后再次运行 raw/all，应使用 B。

### C. 生成 Table R8

默认论文基线就是本次请求提供的三行数字，来源会明确记录；新结果必须来自实际完成的
`classification_folds.csv`，重新计算十折 mean ± std，并与 summary 核对。

```bash
"$PYTHON" PR_EC_E2E/analysis/summarize_table_r8.py \
  --root PR_EC_E2E/outputs/rebuttal_w7 \
  --original-source paper \
  --output PR_EC_E2E/outputs/rebuttal_w7/table_r8.csv
```

如希望基线也从已有 Original seed42 折级文件重新汇总，而不是使用论文表中的数字：

```bash
"$PYTHON" PR_EC_E2E/analysis/summarize_table_r8.py \
  --root PR_EC_E2E/outputs/rebuttal_w7 \
  --original-source saved --original-results-root "$ORIGINAL" \
  --output PR_EC_E2E/outputs/rebuttal_w7/table_r8_from_saved.csv
```

两种汇总方式都只读 Original 结果，不训练模型；缺失新实验结果会报错，
不生成伪造数值，也不自动启动实验。表的五列与请求一致，另存 `.sources.json` 记录来源。

### D. 轻量检查

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -W error::ResourceWarning \
  -m unittest discover -s PR_EC_E2E/tests -v
```

检查包含：旧 batch-mean 前向及梯度逐值一致；个人 attention 归约；时间分支不变；
B=1 等价；梯度到达每个个人矩阵和 Q/K；两种 entropy 值及梯度一致；实际编码器
在合成 90-ROI 序列上提取 EC；窗口均值/转置/零对角；预训练与缓存阶段隔离；
身份及 fold 一致性；配置防混写；汇总脚本只读取完成的模拟结果；输出不会删除配置。
