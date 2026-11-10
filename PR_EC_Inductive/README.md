# PR-EC 严格归纳式分类

使用方式与 `PR_EC` 相同：直接运行数据集入口，无需 tune/evaluate 阶段或 run name。

```bash
/data/users/liulin/miniconda3/envs/default/bin/python /data/users/liulin/PythonCode/ST-MRI/FSTA/PR_EC_Inductive/main_abide_i.py
```

ABIDE-II、ADHD200 分别运行 `main_abide_ii.py`、`main_adhd200.py`。

控制台逐折打印三个方法的 ACC、Recall、AUC、Precision、F1，最后打印各指标的折间均值±标准差（百分数）。

## 严格归纳式流程

- 沿用原来的十折划分、seed 和 train/validation/test 划分。
- 每折仅使用 train 被试的无标签 fMRI 训练一个 STF-EC encoder。
- 冻结 encoder，提取该折 train/validation/test EC，三个方法共享同一 encoder 输出。
- MPR 参考库、QSR 参数及跨被试预处理仅在训练被试上拟合。
- 分类器使用 validation 早停；test 不参与训练和模型选择。
- 不读取原 `PR_EC` 的全体被试 EC 缓存。

## 参数与缓存

和原项目一样，可直接修改各 `main_*.py` 的 `default=`，也可以命令行覆盖，例如：

```bash
/data/users/liulin/miniconda3/envs/default/bin/python PR_EC_Inductive/main_abide_i.py \
  --gpu-id 0 --reference-k 15 --refiner-lr 0.01 --qsr-lr 0.001 \
  --output-dir PR_EC_Inductive/outputs/abide-i-custom
```

查看全部参数：`python PR_EC_Inductive/main_abide_i.py --help`。

默认结果位于 `PR_EC_Inductive/outputs/<dataset>/`。相同配置的 fold encoder 自动恢复；`--no-resume-fold-encoders` 强制重训。`--encoder-cache-dir` 可指定共享的 fold encoder 缓存目录。

单折流程检查可使用 `--folds 1`；完整评估不传该参数。不要根据 test 指标选择参数，调参后的结果应与原始固定参数 R4 分开报告。

每折指标保存为 `experiment_summary.csv`，OOF 标签和预测保存为 `oof_predictions.csv` / `oof_predictions.npz`。已有分析脚本与结果保留，但不是运行分类入口的依赖。
