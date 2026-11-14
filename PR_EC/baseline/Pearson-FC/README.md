# Pearson FC + Linear SVM

此目录实现非神经网络基础对照：每位被试的 ROI 时间序列 → Pearson 相关矩阵 → 上三角非对角特征 → `StandardScaler` → `SVC(kernel="linear")`。不训练 STF-EC、BrainNetCNN 或其他神经网络，不做 Fisher z、特征选择或概率校准。

## 与 PR-EC 对齐的评估协议

- **数据与预处理**：直接调用 `PR_EC.data.load_subject_dataset` 和 `FIXED_DATA_CONFIG`，使用 `cpac/filt_noglobal/rois_aal`、相同表型筛选、被试内逐 ROI 时间标准化及 ROI 顺序。116 个源 ROI 保留前 90 个；ADHD200 沿用单个选定 run 和公共加载器。90 ROI 得到 `90 × 89 / 2 = 4005` 个特征。常量 ROI 的边置零，FC 对角线置零且不作为特征。
- **相同被试与顺序**：若数据集默认 PR-EC EC 归档存在，自动按其中 `subject_ids` 对齐；也可用 `--reference-ec-path` 指定实际比较运行的归档（可为含 `fold_ids` 的 refined/QSR 归档）。缺失被试、站点或标签不匹配会报错。ABIDE-I 与 PR-EC 一样以当前表型重建标签，避免旧 EC 缓存反向标签影响划分。没有参考归档时，使用公共原始数据加载器的顺序；若实际 PR-EC 运行使用了不同子集或顺序，必须传入对应参考归档。
- **相同十折与种子**：直接复用 `PR_EC.utils.folds.make_stratified_splits`，默认 `--n-splits 10 --seed 42 --validation-size 0.2`。外层分层十折；每折剩余 90% 再按 `seed + fold` 划出 20% 验证集，因此约 72%/18%/10% 分别为 train/val/test，与 PR-EC 一致。SVM 使用 PR-EC 首次分类器重复对应的 `seed + fold * 1000 + 1`。参考归档含 `fold_ids` 时，在训练前核验测试折完全一致。非默认 PR-EC 运行应传入其实际 seed、折数和验证比例。
- **标准化与 C 选择不泄漏**：`StandardScaler` 放在 sklearn `Pipeline` 内，只在该折 `train_index` 上拟合。默认固定 `C=1`，不搜索。传入 `--c-grid` 时，`GridSearchCV` 使用 `PredefinedSplit` 精确复用 PR-EC 的折内 train/val，以验证 AUC 选择 C；整个搜索只接收 train/val，完全不接收外层测试折。这是外层十折内的单次 holdout 调参，**不是另加一层 k 折交叉验证**。
- **最终拟合**：选定 C 后仍只在原 `train_index` 重新拟合 scaler 和 SVM；不将验证集并入训练，以保持与 PR-EC 一致的训练被试数。固定 C 时验证集保留但不使用。测试折仅用于最终预测和报告，不用于 C、阈值或模型选择。
- **指标**：逐折输出 ACC、Recall、AUC、Precision、F1（正类为患者 1）。AUC 使用未校准的 `decision_function` 分数；其他指标使用 SVM 的原生预测、默认零边界，不在测试集优化阈值。汇总为十折均值及总体标准差（`ddof=0`）。

## 运行

使用现有 PR-EC 环境（NumPy、scikit-learn 及公共数据模块的依赖；公共模块会导入 PyTorch，但此基线不构造或训练神经网络，不使用 GPU）。在仓库根目录执行：

```bash
# 固定 C=1：默认 seed=42、十折
python PR_EC/baseline/Pearson-FC/run_pearson_fc.py --dataset abide
python PR_EC/baseline/Pearson-FC/run_pearson_fc.py --dataset abide_ii
python PR_EC/baseline/Pearson-FC/run_pearson_fc.py --dataset adhd200

# 可选：C 仅用每个外层训练池中的固定 train/val 划分选择
python PR_EC/baseline/Pearson-FC/run_pearson_fc.py \
  --dataset abide --seed 42 --n-splits 10 --validation-size 0.2 \
  --c-grid 0.001 0.01 0.1 1 10 100 \
  --reference-ec-path PR_EC/outputs/abide-i/abide_qsr_refined_subject_ec.npz \
  --output-dir PR_EC/baseline/Pearson-FC/outputs/linear_svm/abide_tuned
```

`--phenotype-csv` 实际用于公共加载器，而非仅作记录。自定义数据时需与比较的 PR-EC 运行保持一致。

可用 `--generation-only` 仅生成 FC；后续加 `--classification-only` 复用 FC。即使仅分类，也会读取原 ROI/表型，核对被试、标签、ROI 和预处理配置。旧神经网络基线 FC 缓存若缺少新配置或标签不一致，会拒绝复用；用 `--regenerate-fc` 重建。`--max-subjects` 仅用于 smoke test，不可作为完整样本的公平对照。

## 输出与复核

默认输出到 `outputs/linear_svm/<dataset>/`，不会覆盖此前 BrainNetCNN 基线结果。`--output-dir` 同时决定默认 FC 缓存位置；可用 `--fc-path` 单独指定。

- `subject_fc_<dataset>.npz`：被试级 FC、标签、ID、站点、预处理配置；保留 `ec` 键以兼容仓库归档约定。
- `metrics_<dataset>.csv/json`：每折 C 及五个指标，指标为 0–1 数值。
- `summary_<dataset>.json`：只汇总五个指标，不将 C 混入指标均值。
- `protocol_<dataset>.json`：输入配置、参考归档、版本、seed、被试顺序/标签，以及每折 train/val/test 索引、分类器种子、候选 C 的验证 AUC 和最终 C。
- `oof_predictions_<dataset>.npz`：每位被试一次外层测试预测、decision score、真实标签和 `fold_ids`，用于与 PR-EC 逐折配对比较。

本次将原来的 Pearson FC + BrainNetCNN 流程替换为 Pearson FC + Linear SVM；旧的 `--gpu-id`、`--classifier-epochs`、`--classifier-patience`、`--classifier-lr`、`--classifier-repeats` 参数不再适用。
