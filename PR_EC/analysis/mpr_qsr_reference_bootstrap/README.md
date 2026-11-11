# W3：MPR/QSR 训练参考重采样敏感性

实验入口为上一级 `mpr_qsr_reference_bootstrap.py`，所有训练和统计均复用现有项目函数；
不改动 `workflow.py`、原始 EC、编码器、疾病分类器或旧实验结果。

## 运行

在仓库根目录，使用已安装 PyTorch、NumPy、SciPy、scikit-learn 的环境运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  /data/users/liulin/miniconda3/envs/default/bin/python -u \
  PR_EC/analysis/mpr_qsr_reference_bootstrap.py \
  --repeats 5 --seed 20261111 --gpu-id 0 \
  --output-dir PR_EC/analysis/mpr_qsr_reference_bootstrap/pilot_5_20261111

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/users/liulin/miniconda3/envs/default/bin/python \
  PR_EC/analysis/mpr_qsr_reference_bootstrap/validate_and_report.py \
  PR_EC/analysis/mpr_qsr_reference_bootstrap/pilot_5_20261111
```

已有非空输出目录将被拒绝；重跑必须指定新目录。代码默认做 5 次检查，可用
`--repeats` 扩展描述性敏感性实验，但增加重复次数本身不会使其成为正式 Bootstrap CI。
本次实际结果与独立核验见 `pilot_5_20261111/REPORT.md`、`validation.json`。

## 设计

- 从两个队列原 `outputs/*/summary.json` 读取保存的 MPR/QSR、seed、10 折和验证比例参数。
  在开始训练前断言初始 EC、被试顺序、标签、站点及测试折逐被试等于原 QSR 存档。
- 用 `workflow.py` 同一 `make_stratified_splits`、`prepare_fold_arrays`、
  `build_fold_reference`、`train_qsr_refiner` / `apply_qsr_refiner` 实现测试独立 OOF。
- 固定原外层测试折和内层验证折；仅在原验证划分后的训练子集内按被试 iid 有放回抽样，
  样本数不变。不在训练/验证/测试拆分前抽样，不抽样测试被试，不使用诊断标签分层 Bootstrap。
  原训练子集约占全队列的 72%，不是将验证被试也纳入外层 90% 参考池。
- 同一原始被试的所有重复副本均来自对应训练子集，不可能出现在该折验证或测试部分。
  多重副本保留用于 MPR、QC 模式与 QSR，沿用原 MPR 对训练矩阵对角自身行的排除规则；
  不额外将 Bootstrap 多重副本去重，否则会改变训练抽样分布。
- 每次每折重新拟合 EC/表型/fMRI 训练标准化、融合 MPR 图和群体参考、QC 标准化、
  QC–EC 敏感模式和伪目标、QSR 网络。保持原超参数和每折 QSR 随机种子，
  不重新训练初始 EC 编码器、MPR-only 消融 refiner 或疾病分类器。
- 新 OOF 反标准化到原 EC 尺度，所有被试每次恰输出一次测试预测，非对角有向边不对称化。
- 从**原始存档**复算 Welch 重复连接并核对固定 213 条；随后复用
  `cross_cohort_replication` 的 `align_covariates` 和 `fit_edges`，
  对新 OOF 分别拟合 `EC_edge ~ ASD + Age + Sex + C(Site) + Mean_FD`。
  常规 OLS 残差自由度双侧 t 检验，每队列全部 8,010 边独立 BH-FDR，
  双队列 q < 0.05 且同号才视为跨队列重复。

## 指标与限制

逐次报告跨队列重复连接数、原 213 条保留数、原 213 条两个队列均保持原 Hedges-g
符号的比例，以及两个队列当前 beta 同号的比例；方向一致不要求显著。
保留数遵循既有重复标准，另行报告是否保持原方向，避免将方向一致误当作显著。

MTG–丘脑 8 条包括左右 MTG 与左右丘脑之间的双向连接。使用项目现有 Panel A
的 AAL90 顺序：0-based THA.L/R=76/77、MTG.L/R=84/85，并断言 8 条均在原 213 条内。
保存每条每次 beta/p/q、逐队列原方向保持率、逐队列 FDR 通过频率及双队列重复频率。

**这只是固定原初始 EC、固定被试与 OOF 分区条件下，训练参考选择和 QSR 重新拟合的
敏感性检查；不是充分考虑全部被试抽样依赖性的正式 Bootstrap 置信区间。**
不同折的训练样本重叠，原始 EC 的训练过程没有重采样，5 次也仅是流程 pilot。
mean/SD/min/max 只能描述这几次训练参考抽样，不能用作总体 CI、标准误或正式稳定性证明。

严格沿用既有项目标签编码，不重新解释临床诊断。协变量模型也沿用未裁剪的 Mean FD，
包含已在 `cross_cohort_replication/fd_audit/` 中报告的两名 ABIDE-II 极端 FD 被试。
因此本结果不能替代 FD 异常值敏感性分析；与原未裁剪协变量基线的对比才具有相同统计口径。

## 文件

- `provenance.json`：完整参数、原始输入哈希、设备、Python、原始 213 条复算校验。
- `*_fixed_partitions.npz`：被试 ID 与固定 train/val/test 索引。
- `baseline_adjusted/`：原始 QSR EC 在相同协变量模型下的基线。
- `repeat_XX/`：新双队列 OOF NPZ、逐折训练抽样被试 ID、多重副本、QSR 训练指标、
  全边协变量统计、重复连接、原 213 条比较和逐次汇总。
- `replicate_summary.csv` / `summary.json`：逐次指标和描述性 mean/SD。
- `mtg_thalamus_per_repeat.csv`、`mtg_thalamus*_frequencies.csv`：8 条边的逐次/频率结果。
- `fold_sampling_audit.csv` / `validation.json`：独立无重叠、完整 OOF、原始文件不变、
  全边 OLS/FDR 复算核验。
- `REPORT.md`：中文方法、逐次结果、8 条边频率和解释边界。

NPZ 是实际保存在磁盘上的大文件；项目原 `.gitignore` 默认忽略 `*.npz`，
这不影响本地实验结果保存，但复制或归档时需连同这些文件一起保存。
