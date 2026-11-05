# MemoryEC

唯一实验入口为 `main.py`。`train.py` 保留为底层训练实现，`test.py` 保留旧
checkpoint/预测文件的兼容评价工具。原 `cross_network_cv.py` 和
`sparse_pair_experiment.py` 已合并删除。模型文件、处理后数据与既有输出不变。

## 快速运行

以下命令从项目根目录运行；请使用安装了 PyTorch 的 Python 环境。

```bash
# 默认：Pair Memory，九折，每折独立训练一次（seed=42）
python -u MemoryEC/main.py --device cuda --run-name pair_cv

# 只运行第9折：Test=Network9，Validation=Network1/2
python -u MemoryEC/main.py --fold 9 --device cuda --run-name pair_cv

# 比较 No Memory / Global Memory / Pair Memory，九折各一次
python -u MemoryEC/main.py --mode ablation --device cuda --run-name ablation_cv

# 三模型九折，每折20次独立初始化（共9×3×20=540次训练）
python -u MemoryEC/main.py --mode ablation --repeats 20 --device cuda --run-name ablation_cv20

# 仅Pair Memory，第9折20次独立训练
python -u MemoryEC/main.py --fold 9 --repeats 20 --device cuda --run-name pair_cv20

# Pair Memory固定稀疏系数，仍以验证集选择阈值
python -u MemoryEC/main.py --sparse_lambda 0.005 --device cuda --run-name pair_sparse005

# Pair Memory对比固定λ网格（0、0.005、0.02），每折用验证集选择λ
python -u MemoryEC/main.py --mode sparse --device cuda --run-name sparse_cv

# 只打印计划：不训练、不申请GPU、不写文件
python MemoryEC/main.py --mode ablation --repeats 20 --dry-run

# 重新打印已完成的折与重复实验结果：不训练、不推理、不需要GPU
python MemoryEC/main.py --mode evaluate --run-name ablation_cv

# 读取历史cross_network_cv结果，重建并验证FP/FN和边数，不训练
python MemoryEC/main.py --mode diagnose
```

`--device cuda` 在 GPU 不可用时明确报错；`--device auto` 优先 GPU，允许回退 CPU。
`--mode train` 为默认，使用 `--model-type st_pair_memory`（默认）；也支持另外两种模型。
`--mode ablation` 忽略单个 model-type，始终训练全部三种。
`--mode sparse` 只支持 Pair Memory，固定上述三种 λ，不能同时传非零 sparse_lambda。

## 参数与训练次数

```bash
python -u MemoryEC/main.py --mode train --model-type st_pair_memory \
  --fold 1 --repeats 1 --hidden-dim 32 --memory-slots 8 \
  --attention-heads 2 --dropout 0.2 --learning-rate 0.0003 \
  --weight-decay 0.001 --batch-size 8 --epochs 150 --patience 15 \
  --sparse_lambda 0 --seed 42 --device cuda --run-name pair_custom
```

`--fold k` 是测试基础网络编号，不是选择某个仿真变体。省略则运行全部九折：
Test=k，Validation=k+1/k+2（超过9循环），其余六家族 Train。同家族所有变体
整体分配；数据直接读取 `data/processed/`，不会重新预处理或改写 `splits.json`。

`--repeats` 只允许 1 或 20。重复编号r从1开始，seed=`base_seed+r-1`，默认
20次对应42..61。每次都重新构造模型和优化器，从头训练；不会接着上一次权重
继续，也不会只对同一权重重复测试。同一重复、不同模型和 λ 使用同一个 seed
便于配对比较。九折划分不随 seed 改变。

FSTA官方参考代码 `train_FSTA_sanch_multi.py` 在20次循环内重建模型与优化器。
本实现借用这一独立重训练原则，但显式记录seed序列，且维持本项目的跨家族
Train/Validation/Test协议，并非照搬FSTA的训练集或测试阈值选择。

## 评价和稀疏项

- 主 checkpoint 固定为最小未加惩罚的验证 BCE（`best.pt`），early stopping 也
  使用验证 BCE；不根据测试结果切换到 `best_val_auprc.pt`。
- 先对每场景全部被试的 sigmoid 概率求平均，再在非对角线评价有向边。
- 主阈值在验证集0.05..0.95、步长0.01搜索：家族内先平均各场景F1，再对两家族
  等权平均。并列取较低阈值，验证阈值保存后才预测测试数据。
- 固定0.5和FSTA自适应阈值只作对比；FSTA比例固定0.5，从场景非对角线概率的
  min/max中点计算。AUPRC/AUROC始终从连续概率计算。
- 稀疏项仅在训练阶段加入：`BCE + λ * sigmoid(logits_offdiag).mean()`。
  λ默认0，零分支保持原BCE计算；`pos_weight=None`，等效正类权重1。
  不直接惩罚Memory Key/Value或Spatial Attention权重。
- `sparse`模式所有候选训练和验证完成后，用验证家族宏平均F1选择λ；同分取
  较小λ。锁定λ和阈值后才测试。表中`validation_selected_lambda`是这一选择
  程序的结果，各`lambda=...`列只作探索性对比，不能根据Test F1事后选主λ。
  因为既有测试结果已被观察过，稀疏实验仍需标为探索性后续研究，而非全新确认性测试。

## 保存与恢复

```text
outputs/<run-name>/
  config.json
  repeat_01/fold_01/st_pair_memory/
    ...训练权重、日志、validation_selection.json、results.json
```

稀疏网格的三个候选权重放在模型目录下的 `lambda_0/`、`lambda_0.005/`、
`lambda_0.02/`。只保存checkpoint和指标，不再逐被试生成大量预测NPZ。
`config.json`记录九折场景、seed序列、模型参数和评价协议；`split.json`记录
每次每折的家族分配。兼容且完整的模型结果自动跳过，完整训练但尚未评价的
checkpoint可以继续评价。不完整训练不静默覆盖：会报错并要求更换run-name。
修改参数、次数、seed或模式也必须换run-name。不同折可用同一run-name分次运行，
不会用旧目录中不同设置的权重冒充本次结果。

## 控制台与结果表

训练进度仅每10个epoch输出一行（10、20、30……）；早停和训练结束各输出简短信息，
不打印包含checkpoint路径的大字典。完整逐epoch记录和训练汇总仍保存为JSON。

运行完成自动汇总所有已完成的折；部分九折完成时明确显示已完成家族集合。
只有某折的全部选定模型和重复均完成才纳入比较，避免模型间不等样本数。

- `scene_results.csv`：每次/每折/每模型或λ/每方法的场景级指标、阈值、最佳Epoch。
- `family_results.csv`：每次训练的测试家族场景宏平均。
- `network_results.csv`：每个网络/方法/模型的重复均值和重复标准差。
- `network_table.csv`：每网络一行的横向对比宽表。
- `summary.json`：家族宏平均、跨家族SD和重复宏平均SD及完成状态。

控制台只打印 Precision、Recall、F1、Accuracy、SHD 五项指标；20次重复时以
`均值 +/- 标准差` 显示，均保留两位小数。概率指标保持0..1量纲，不自动转百分比。
完整的 AUPRC、AUROC、FP、FN、预测边数、真实边数仍写入CSV/JSON，不截断精度。
每家族的边数是场景均值，不是被试总边数。

- Network行：重复实验均值 ± 重复标准差；单次则仅一个数。
- MacroMean：先平均家族内场景，再对测试家族等权平均，不混用场景数量权重；
  20次时同时显示各次家族宏平均之间的重复标准差。
- FamilyStd：各测试家族重复均值之间的标准差，不是随机seed标准差。
- MacroRepeatStd：每次重复的家族宏平均在重复之间的标准差。

标准差均使用总体SD（ddof=0）。1次重复时MacroRepeatStd为0，不代表没有不确定性。
