# W6 / Q1：Age/Sex/Site 消融和最终融合 Top-K 诊断一致率

所有命令在仓库根目录手动执行。开发阶段只做代码检查和合成数据单元测试，
没有运行真实数据诊断、Initial EC 训练、MPR/QSR 训练或分类实验。

## 修改范围与不变量

- 三个 `main_*.py`：增加 `--phenotype-feature-set original|age_sex_site`
  （默认 `original`）、`--diagnostics-only`，转入本目录独立 runner。
- `rebuttal.py`：按配置选择连续权重维度；新配置为 `[1.0]`，原 ABIDE
  为 `[1.0, 0.3]`，原 ADHD200 为 `[1.0, 0.3, 0.3]`。
  默认输出移至 `PR_EC_changePH/outputs/rebuttal_w6/<dataset>/<setting>/`。
  禁止向本目录外写实验产物、覆盖输入 EC 或复用冲突的输出配置。
- `data/__init__.py`、`model/mpr/population_similarity.py`：接入字段配置；
  Original 的数值/类别加载分支完全保留。已仅检查三个真实 CSV 的表头：

  | 数据集 | Original 连续字段 | Original 类别字段 | 新连续字段 | 新类别字段 |
  | --- | --- | --- | --- | --- |
  | ABIDE-I / II | FIQ, PIQ | SEX | AGE_AT_SCAN | SEX, SITE_ID |
  | ADHD200 | Age, Full4 IQ, Handedness | Gender | Age | Gender, Site |

  PH 字段不包含诊断、评分、问卷。新 Site 来自对齐后的 CSV，不推断自诊断。
  年龄沿用训练折中位数填补及标准化，类别沿用训练折字典及未知类别码。
  ADHD200 额外保留其原来的训练折类别众数/数值中位数填补步骤。
- `workflow.py`：沿用原 `PR_EC/workflow.py` 的 train/val/test 划分、FC
  标准化、融合、MPR/QSR 及分类流程，只在最终 `fused_graph` 的同一次
  Top-K 选择中额外返回索引。未更改稀疏图、权重、argpartition 的并列处理。
  诊断只在图和邻居已确定后读取标签。每个邻居索引都相对于该折实际 train，
  不包括 val/test。训练/验证/测试被试互斥且完整覆盖的检查在每折执行。
- `runner.py`：diagnostics-only 在任何模型拟合前返回；不加载 QSR QC/混杂变量。
  Full 模式保留原始各表示/各折输出，另写 Full PR-EC 分类表和测试 OOF QSR EC。
- `diagnostics.py`：以最终 Top-K 的**人数**而不是相似权重计数，
  `K=min(reference_k, 实际参考训练人数)`；Base rate 分母是排除验证集后的实际 train。
  即使稀疏融合图存在零分并列，也记录原算法实际选中的 K 个索引，不另行检索。
  总均值按所有 OOF 测试被试汇总，不平均不同大小折的均值。
- `utils/output.py`：复制原 archive/summary writer，仅移除目录级删除逻辑，
  不会清理新诊断 CSV、split manifest 或其它已有文件。原 `PR_EC` 未修改。
- `analysis/summarize_table_r6.py`：核验两配置的非 PH 设置、被试、标签和
  所有折的 train/val/test 完全一致；复核邻居均为训练被试、人数比例和 OOF 完整性。
  Original ACC/AUC 优先读本目录 classification_summary，否则只读原 seed42 的
  `experiment_summary.csv`，并要求原 `summary.json` 记录 seed42 / 十折。
- `tests/test_rebuttal_w6.py`：仅内存合成数组/临时 CSV、模拟模型函数；
  核验原模式数值等价、未知类别、无泄漏、输出隔离、OOF 元数据和统计公式。
- `utils/__init__.py`：本地输出模块命名空间；不替换原训练折预处理实现。

模型、分类器、原数据加载器、折划分函数直接复用 `PR_EC` 的现有实现，
不在其它目录打补丁。原始计算默认值不变，**输出路径改为独立目录是刻意的安全隔离**。
仅保留原 `raw` 输入接口供兼容；下述 W6 命令全部强制 `ec`，不会训练初始编码器。
Age/Sex/Site 或 diagnostics-only 会拒绝 raw、非融合图、置乱 PH、非 seed42、
非十折、非 K20。其他模型超参数按各数据集原默认值执行，不另调参。

## 输出

每个 `<dataset>/<setting>/` 包含：

- `neighbor_consistency_subjects.csv`：dataset、phenotype_feature_set、seed、fold、
  subject_id、true_label、diagnosis、same_diagnosis_count、K、same_diagnosis_pct、
  reference_train_n、reference_same_diagnosis_n、base_rate_pct、neighbor_subject_ids。
  最后一列是最终融合 Top-K 的 ID JSON 数组，便于逐被试复核。
- `neighbor_consistency_summary.csv`：全部 OOF 测试被试的均值。
- `neighbor_consistency_folds.csv`：每折人数及均值。
- `split_membership.csv`：每折每被试的 archive_index / ID / train、val、test 角色。
- `run_manifest.json`：非执行性参数及被试/标签摘要；阻止不同设置混写。
- Full 额外输出 `classification_folds.csv`、`classification_summary.csv`：
  仅 `qc_refined` 的 ACC、AUC、SEN、PRE、F1，单位为百分比。
  汇总为十折 `mean ± std`，std 沿用原代码的 `ddof=0`。
- Full 额外保留原 `experiment_summary.csv` / `summary.json` 及
  `<dataset>_qsr_refined_subject_ec.npz`；默认还保留 PGR/QSR 合并 NPZ。
  QSR archive 的 `ec` / `qc_refined_ec` 均为测试 OOF EC，并保留输入 archive
  的 subject_ids、labels（ABIDE-I 仍按原流程重新从表型对齐二分类标签）、site_ids、
  fold_ids 和 roi_names 顺序；若源文件无 roi_names，仍沿用原来的顺序占位名。

标签沿用现有代码，不根据 PH 配置变更：ABIDE-I 仍调用原来的 phenotype→label
函数；ABIDE-II / ADHD200 仍使用既有 EC archive 的二分类 labels。
常规标签为 1=ASD/ADHD、0=TC/Control。ADHD 的 DX 1/2/3 vs 0 定义不变。

## 运行命令（仅提供，不自动执行）

先在 Bash 中设置公共参数；以下 seed42 Initial EC 路径已做文件存在性检查，未加载分析：

```bash
PYTHON=/storage/users/liulin/miniconda3/envs/default/bin/python
COMMON=(--input-mode ec --seed 42 --n-splits 10 --validation-size 0.2 --graph-mode fusion --reference-k 20 --fusion-beta 0.6)
EC_I=PR_EC/outputs/seed42/abide-i/abide_subject_ec.npz
EC_II=PR_EC/outputs/seed42/abide-ii/abide_ii_subject_ec.npz
EC_ADHD=PR_EC/outputs/seed42/adhd200/adhd200_subject_ec.npz
```

### Original PH：三个数据集 diagnostics-only

```bash
"$PYTHON" PR_EC_changePH/main_abide_i.py "${COMMON[@]}" --ec-path "$EC_I" --phenotype-feature-set original --diagnostics-only
"$PYTHON" PR_EC_changePH/main_abide_ii.py "${COMMON[@]}" --ec-path "$EC_II" --phenotype-feature-set original --diagnostics-only
"$PYTHON" PR_EC_changePH/main_adhd200.py "${COMMON[@]}" --ec-path "$EC_ADHD" --phenotype-feature-set original --diagnostics-only
```

### Age/Sex/Site PH：三个数据集 diagnostics-only

```bash
"$PYTHON" PR_EC_changePH/main_abide_i.py "${COMMON[@]}" --ec-path "$EC_I" --phenotype-feature-set age_sex_site --diagnostics-only
"$PYTHON" PR_EC_changePH/main_abide_ii.py "${COMMON[@]}" --ec-path "$EC_II" --phenotype-feature-set age_sex_site --diagnostics-only
"$PYTHON" PR_EC_changePH/main_adhd200.py "${COMMON[@]}" --ec-path "$EC_ADHD" --phenotype-feature-set age_sex_site --diagnostics-only
```

### Age/Sex/Site PH：三个数据集 Full PR-EC 十折

保留默认 original/refined/qc_refined 表示，不额外改 QSR、分类器超参数或选择规则。
`--gpu-id auto` 与默认相同；需要时可手动换成指定 GPU ID。

```bash
"$PYTHON" PR_EC_changePH/main_abide_i.py "${COMMON[@]}" --ec-path "$EC_I" --phenotype-feature-set age_sex_site --gpu-id auto
"$PYTHON" PR_EC_changePH/main_abide_ii.py "${COMMON[@]}" --ec-path "$EC_II" --phenotype-feature-set age_sex_site --gpu-id auto
"$PYTHON" PR_EC_changePH/main_adhd200.py "${COMMON[@]}" --ec-path "$EC_ADHD" --phenotype-feature-set age_sex_site --gpu-id auto
```

Full 会重新生成相同的诊断 CSV，因此可先只跑 Original diagnostics，再直接跑新配置
Full；新配置单独 diagnostics 命令用于无需模型训练时独立核验。

### 完成后生成 Table R6

```bash
"$PYTHON" PR_EC_changePH/analysis/summarize_table_r6.py \
  --root PR_EC_changePH/outputs/rebuttal_w6 \
  --original-results-root PR_EC/outputs/seed42 \
  --output PR_EC_changePH/outputs/rebuttal_w6/table_r6.csv
```

找不到任一必需结果或一致性检查失败会报错，**不填充虚构值，不自动启动缺失实验**。
`classification_source` 记录实际复用的分类结果路径。

### 仅合成数据的轻量测试

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m unittest discover -s PR_EC_changePH/tests -v
```
