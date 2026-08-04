# ABIDE-A AAL90 个体 BEC（三模式）

## 文件结构

```text
src/
    data.py
    model.py
    train.py

scripts/
    run_experiment.py
    inspect_data.py
    compare_modes.py
    evaluate_qc_leakage.py
    classify_bec.py

examples/
    mode1_fmri_only.sh
    mode2_fmri_pheno.sh
    mode3_fmri_pheno_qcadv.sh
```

## 三种模式

```text
1  fmri_only
2  fmri_pheno
3  fmri_pheno_qcadv
```

模式 1 只用 AAL90 ROI 时间序列。

模式 2 加入：

```text
AGE_AT_SCAN, SEX, FIQ
```

模式 3 再增加 QC 对抗目标：

```text
func_mean_fd
func_dvars
func_outlier
func_perc_fd
```

QC 不直接输入 BEC 生成器。

## 输出方向

保存的矩阵统一为：

```text
BEC[source, target] = source ROI → target ROI
```

例如：

```python
weight = bec[5, 10]
```

表示：

```text
ROI 5 → ROI 10
```

模型内部为了计算：

```text
x(t+1) = A @ x(t)
```

使用 `[target, source]`，导出时在 `src/train.py` 中转置：

```python
saved_bec = internal_bec.T.copy()
```

因此所有 `.npz` 输出都是 `[source, target]`。

## 安装

```bash
pip install -r requirements.txt
```

## 检查数据

```bash
python scripts/inspect_data.py \
  --roi_root "/path/to/filt_noglobal" \
  --phenotypic_csv "/path/to/Phenotypic_V1_0b_preprocessed1.csv"
```

## 运行

直接参考 `examples/` 中的三个脚本。

快速运行模式 1，并在生成 BEC 后自动进行 ASD/HC 分类：

```bash
cd DeepASD/02_individual_BEC_AAL90
PYTHON_BIN=/data/users/liulin/miniconda3/envs/default/bin/python \
  bash examples/mode1_fmri_only.sh
```

该快速配置使用全部受试者、2折交叉验证、10个 epoch、78帧窗口。结果保存在：

```text
results/01_fmri_only_quick/
```

只验证代码流程时，可以直接运行更小的数据：

```bash
python scripts/run_experiment.py \
  --experiment_mode 1 \
  --max_subjects 40 \
  --folds 2 \
  --epochs 1 \
  --window_length 78 \
  --eval_windows 1 \
  --batch_size 8 \
  --device cpu \
  --result_dir /tmp/deepasd_mode1_check
```

三次实验请保持相同：

```text
seed
folds
window_length
eval_windows
epochs
batch_size
```

每种模式输出：

```text
oof_individual_bec_AAL90.npz
fold_01/
fold_02/
...
fold_metrics.csv
classification_metrics.csv
```

`fold_metrics.csv` 是 BEC 重建误差与结构统计，`classification_metrics.csv` 才是 ASD/HC 分类结果。默认不生成871个单独文件；需要时增加 `--save_individual_files`。

不计算组平均 BEC，也不计算 DEC。
