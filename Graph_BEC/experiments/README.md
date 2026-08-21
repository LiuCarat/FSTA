# Graph-BEC experiments

每个数据集是一套独立实验配置，不把 ABIDE 参数作为 ADHD200 的隐式基准。

## 运行

```bash
python -m Graph_BEC.main --dataset abide
python -m Graph_BEC.main --dataset adhd200
```

也可以在命令行覆盖 profile 中的单个参数：

```bash
python -m Graph_BEC.main --dataset adhd200 --reference-k 10
```

## 配置

- `experiments/abide_i_config.py`：ABIDE-I 的完整路径、标签、FSTA、患者图、refiner、QSR 和分类器参数。
- `experiments/adhd200_config.py`：ADHD200 的对应完整参数，包含 ADHD200 的 phenotype 列、QC 列和排除对象。
- `experiments/configuration.py`：仅负责注册和选择 profile。

`Graph_BEC/config.py` 只负责解析命令行，并根据 `--dataset` 注入对应 profile 的默认值。运行时可使用：

```bash
--patient-label 1 --control-label 0
```

内部暂时保留 `asd_label`/`tc_label` 兼容字段，避免一次性改动现有 workflow。

## 输出

ABIDE 和 ADHD200 不共享 NPZ：

```text
Graph_BEC/outputs/abide/abide_subject_bec.npz
Graph_BEC/outputs/abide/abide_refined_subject_bec.npz
Graph_BEC/outputs/adhd200/adhd200_subject_bec.npz
Graph_BEC/outputs/adhd200/adhd200_refined_subject_bec.npz
```
