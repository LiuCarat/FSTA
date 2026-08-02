# ABIDE-I 共享窗口 FSTA

本目录使用 ABIDE-I AAL90 时间序列生成个体 BEC，再使用10折随机森林完成 ASD/HC 分类。

## 文件

```text
data.py                    读取表型和 AAL90 时间序列
modified_fsta.py           训练共享窗口 FSTA 并生成个体 BEC
train_rf_10fold.py         使用 BEC 进行10折随机森林分类
```

## 输出目录命名

不同配置必须使用不同的 `--output_dir`，建议名称至少包含：

```text
损失类型 + alpha + epoch + 是否标准化 + FSTA seed
```

例如：

```text
downstream_abide_i/outputs/modified_fsta/
├── entropy_alpha_0p01_e101_zscore_seed2026/
├── entropy_alpha_0p02_e101_zscore_seed2026/
├── entropy_alpha_0p01_e50_zscore_seed2026/
├── entropy_alpha_0p01_e101_raw_seed2026/
└── original_alpha_0p8_e301_zscore_seed2026/
```

这样不同配置不会互相覆盖。

## 1. 当前推荐配置

配置：

```text
loss_mode=entropy
loss_alpha=0.01
epochs=101
window_length=78
stride=39
batch_size=32
standardize=true
FSTA seed=2026
```

生成 BEC：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/modified_fsta.py \
  --loss_mode entropy \
  --loss_alpha 0.01 \
  --epochs 101 \
  --window_length 78 \
  --stride 39 \
  --batch_size 32 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/modified_fsta/entropy_alpha_0p01_e101_zscore_seed2026
```

运行 weighted 十折随机森林：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/train_rf_10fold.py \
  --bec_path downstream_abide_i/outputs/modified_fsta/entropy_alpha_0p01_e101_zscore_seed2026/subject_bec.npz \
  --output_dir downstream_abide_i/outputs/random_forest_10fold/entropy_alpha_0p01_e101_zscore_seed2026 \
  --n_splits 10 \
  --n_estimators 1000 \
  --seed 42
```

该配置之前得到：

```text
Weighted Precision = 65.69 ± 4.58
Weighted Recall    = 65.33 ± 3.95
Weighted F1        = 64.51 ± 4.06
```

## 2. 修改 entropy 权重

只修改 `loss_alpha=0.02`：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/modified_fsta.py \
  --loss_mode entropy \
  --loss_alpha 0.02 \
  --epochs 101 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/modified_fsta/entropy_alpha_0p02_e101_zscore_seed2026
```

对应分类命令：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/train_rf_10fold.py \
  --bec_path downstream_abide_i/outputs/modified_fsta/entropy_alpha_0p02_e101_zscore_seed2026/subject_bec.npz \
  --output_dir downstream_abide_i/outputs/random_forest_10fold/entropy_alpha_0p02_e101_zscore_seed2026 \
  --seed 42
```

## 3. 修改 epoch

只修改为50个 epoch：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/modified_fsta.py \
  --loss_mode entropy \
  --loss_alpha 0.01 \
  --epochs 50 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/modified_fsta/entropy_alpha_0p01_e50_zscore_seed2026
```

## 4. 关闭标准化

使用 `--no-standardize`：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/modified_fsta.py \
  --loss_mode entropy \
  --loss_alpha 0.01 \
  --epochs 101 \
  --no-standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/modified_fsta/entropy_alpha_0p01_e101_raw_seed2026
```

该配置之前得到：

```text
Weighted Precision = 62.46 ± 3.54
Weighted Recall    = 62.23 ± 3.13
Weighted F1        = 61.11 ± 3.11
```

当前数据上，保留 z-score 的 weighted F1 高约3.40个百分点。

## 5. 原始损失配置

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/modified_fsta.py \
  --loss_mode original \
  --loss_alpha 0.8 \
  --epochs 301 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/modified_fsta/original_alpha_0p8_e301_zscore_seed2026
```

## 查看结果

生成的全部受试者 BEC：

```text
<output_dir>/subject_bec.npz
```

训练后的共享 FSTA 模型：

```text
<output_dir>/model.pt
```

实验参数、最终训练损失和 BEC 汇总统计统一保存在：

```text
<output_dir>/summary.json
```

脚本不再生成 `training_history.csv`、`config.json` 和 `individual/`。下游随机森林只需要 `subject_bec.npz`。

随机森林最终结果：

```text
<rf_output_dir>/summary.json
```

每一折结果：

```text
<rf_output_dir>/fold_metrics.csv
```

比较配置时，每次只修改一个主要参数，并确保 FSTA 与随机森林目录使用相同的配置名称。
