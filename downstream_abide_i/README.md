# ABIDE-I 共享窗口 FSTA

本目录使用 ABIDE-I AAL90 时间序列生成个体 BEC，再使用10折随机森林完成 ASD/HC 分类。

## 文件

```text
data.py                    读取表型和 AAL90 时间序列
FSTA_BEC.py           训练共享窗口 FSTA 并生成个体 BEC
train_rf_10fold.py         使用 BEC 进行10折随机森林分类
```

## 输出目录命名

不同配置必须使用不同的 `--output_dir`，建议名称至少包含：

```text
损失类型 + alpha + epoch + 是否标准化 + FSTA seed
```

例如：

```text
downstream_abide_i/outputs/FSTA_BEC/
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
  downstream_abide_i/FSTA_BEC.py \
  --loss_mode entropy \
  --loss_alpha 0.01 \
  --epochs 101 \
  --window_length 78 \
  --stride 39 \
  --batch_size 32 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/FSTA_BEC/entropy_alpha_0p01_e101_zscore_seed2026
```

运行 weighted 十折随机森林：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/train_rf_10fold.py \
  --bec_path downstream_abide_i/outputs/FSTA_BEC/entropy_alpha_0p01_e101_zscore_seed2026/subject_bec.npz \
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
  downstream_abide_i/FSTA_BEC.py \
  --loss_mode entropy \
  --loss_alpha 0.02 \
  --epochs 101 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/FSTA_BEC/entropy_alpha_0p02_e101_zscore_seed2026
```

对应分类命令：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/train_rf_10fold.py \
  --bec_path downstream_abide_i/outputs/FSTA_BEC/entropy_alpha_0p02_e101_zscore_seed2026/subject_bec.npz \
  --output_dir downstream_abide_i/outputs/random_forest_10fold/entropy_alpha_0p02_e101_zscore_seed2026 \
  --seed 42
```

## 3. 修改 epoch

只修改为50个 epoch：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/FSTA_BEC.py \
  --loss_mode entropy \
  --loss_alpha 0.01 \
  --epochs 50 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/FSTA_BEC/entropy_alpha_0p01_e50_zscore_seed2026
```

## 4. 关闭标准化

使用 `--no-standardize`：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/FSTA_BEC.py \
  --loss_mode entropy \
  --loss_alpha 0.01 \
  --epochs 101 \
  --no-standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/FSTA_BEC/entropy_alpha_0p01_e101_raw_seed2026
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
  downstream_abide_i/FSTA_BEC.py \
  --loss_mode original \
  --loss_alpha 0.8 \
  --epochs 301 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/FSTA_BEC/original_alpha_0p8_e301_zscore_seed2026
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
# 表型条件化 FSTA

该模式让 AGE_AT_SCAN、SEX、FIQ 在 FSTA 内部参与空间注意力计算，因此最终生成的
90×90 BEC 已经受到表型调节。这属于模型级融合，不是随机森林阶段的早期或晚期融合。

## 实际接入位置

当前 FSTA 没有独立的 `encode_features()` 和 `generate_bec()` 接口。BEC 直接来自
`model/FSTA.py` 中的空间自注意力，因此实际流程为：

```text
fMRI窗口
  → 卷积嵌入、位置编码、Fourier模块
  → [B,T,N,d_model]
  → 表型门控条件化
  → 空间自注意力
  → 90×90 BEC
```

表型分支为：

```text
AGE_AT_SCAN、SEX、FIQ
  → 3维到隐藏维度
  → d_model维表型表示
  → 重建回3维表型
```

门控初始值为0，使初始前向结果等价于原始 FSTA。条件映射层不使用全零初始化，保证
门控在第一次反向传播时具有非零梯度，不会出现门控和映射层互相阻断梯度的问题。

## 缺失表型处理

当前版本不填补缺失值。表型条件模式默认使用：

```text
--phenotype_missing_policy drop
```

只要受试者在 AGE_AT_SCAN、SEX、FIQ 任一所选列存在缺失，就会同时从受试者记录、
fMRI 时间序列和表型矩阵中排除，保证三者顺序一致。排除详情保存在：

```text
<output_dir>/phenotype_exclusions.json
```

如需发现缺失时直接停止，可使用：

```text
--phenotype_missing_policy error
```

## 运行命令

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/Phenotype_FSTA_BEC.py \
  --phenotypic_csv dataset/ABIDE-I/Phenotypic_V1_0b_preprocessed1.csv \
  --phenotype_missing_policy drop \
  --phenotype_columns AGE_AT_SCAN,SEX,FIQ \
  --phenotype_hidden_dim 32 \
  --phenotype_loss_weight 0.01 \
  --loss_mode entropy \
  --loss_alpha 0.01 \
  --epochs 101 \
  --window_length 78 \
  --stride 39 \
  --batch_size 32 \
  --standardize \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/Phenotype_FSTA_BEC/entropy_alpha_0p01_e101_seed2026
```

纯 fMRI 的原始 FSTA BEC 使用独立入口：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/FSTA_BEC.py \
  --loss_mode entropy \
  --loss_alpha 0.01 \
  --epochs 101 \
  --seed 2026 \
  --gpu_id auto \
  --output_dir downstream_abide_i/outputs/FSTA_BEC/entropy_alpha_0p01_e101_seed2026
```

模型类型由入口文件名固定，不需要额外的模型切换参数。

## 新增输出

条件模式的 `subject_bec.npz` 额外包含：

```text
phenotype_standardized
phenotype_reconstructed
phenotype_columns
phenotype_gate
```

`summary.json` 的训练指标额外记录：

```text
phenotype_reconstruction_loss
phenotype_gate
```

## 下游比较

因为条件模式生成的 BEC 已经包含表型调节，下游应继续只使用 BEC 随机森林：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  downstream_abide_i/train_rf_10fold.py \
  --bec_path downstream_abide_i/outputs/Phenotype_FSTA_BEC/entropy_alpha_0p01_e101_seed2026/subject_bec.npz \
  --output_dir downstream_abide_i/outputs/random_forest_10fold/phenotype_conditioned_entropy \
  --n_splits 10 \
  --n_estimators 1000 \
  --seed 42
```

应与使用相同 FSTA 参数生成的原始 BEC 进行比较，不要再叠加 `early_fusion` 或
`late_fusion`，否则无法单独评价表型条件化 BEC 生成模块的贡献。
