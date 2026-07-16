# UCLA CNP 数据集 (ds000030)

> **UCLA Consortium for Neuropsychiatric Phenomics LA5c Study**
>
> OpenNeuro 编号: [ds000030](https://openneuro.org/datasets/ds000030) · BIDS 版本: 1.0.2 · 许可: CC0

---

## 1. 数据集概况

| 项目 | 数值 |
|------|------|
| 受试者总数 (TSV) | 272 |
| 已下载目录 | 270 |
| func + anat 完整可用 | **261** |
| MRI 序列 | T1w MPRAGE + 静息态 fMRI (EPI) |
| 扫描仪 | Siemens TrioTim 3T |
| fMRI 参数 | TR = 2.0 s, TE = 0.03 s, 64×64 矩阵, GRAPPA ×2 |
| 静息态时长 | ~5 分钟 (152 个时间点) |
| 原始数据占用 | ~9.2 GB |
| 预处理后占用 | ~23 GB (含 fMRIPrep derivatives) |

## 2. 受试者分布

### 诊断分类

| 诊断 | 人数 | 亚组文件 |
|------|------|----------|
| CONTROL (健康对照) | 122 | `subject_lists/hc_subjects.txt` |
| SCHZ (精神分裂症) | 50 | `subject_lists/schz_subjects.txt` |
| BIPOLAR (双相障碍) | 49 | `subject_lists/bd_subjects.txt` |
| ADHD (注意缺陷多动障碍) | 40 | `subject_lists/adhd_subjects.txt` |
| 其他 (缺模态) | 11 | — |
| **合计** | **272** | |

### 人口学信息

| 项目 | 数值 |
|------|------|
| 男性 / 女性 | 155 / 117 |
| 年龄范围 | 21-50 岁 |
| 平均年龄 | 33.2 岁 |

## 3. 预处理流水线

### 3.1 概览

```
原始 BIDS 数据
  │
  ├─ fMRIPrep 25.2.5  (Docker, MNI152NLin6Asym, 跳过 FreeSurfer)
  │    ├─ 解剖预处理: N4 偏场校正 → 脑提取 → ANTs SyN 配准到 MNI
  │    └─ 功能预处理: 头动校正 → STC → BOLD→T1 配准 → MNI 重采样 → confounds
  │
  ├─ 后处理: preprocess_ucla_cnp.py
  │    ├─ 图谱: Harvard-Oxford 62 ROI (皮层 48 双侧合并 + 皮层下 14 左右分开)
  │    ├─ 去噪: 26 参数回归 (24 头动 + WM + CSF)
  │    ├─ 滤波: 0.01–0.1 Hz 带通
  │    ├─ 标准化: z-score (逐时间序列)
  │    └─ 验证: 体素覆盖 + 头动 FD 质控
  │
  └─ 输出: roi_timeseries/HO62/sub-XXXXX.txt  (152 × 62, tab 分隔)
```

### 3.2 fMRIPrep 命令

```bash
docker run --rm \
  -v $(pwd):/data:ro \
  -v $(pwd)/derivatives/fmriprep:/out \
  -v $(pwd)/derivatives/work:/work \
  -v $(pwd)/license.txt:/opt/freesurfer/license.txt:ro \
  nipreps/fmriprep:25.2.5 \
  /data /out participant \
  --participant-label XXXXX \
  --output-spaces MNI152NLin6Asym \
  --fs-no-reconall \
  --skip-bids-validation \
  --clean-workdir \
  --nprocs 42 \
  --omp-nthreads 2 \
  --mem-mb 26666 \
  -w /work/sub-XXXXX
```

**参数说明：**
- `--output-spaces MNI152NLin6Asym` — 仅输出 MNI152NLin6Asym 标准空间（2009cAsym 已删除以节省空间）
- `--fs-no-reconall` — 跳过 FreeSurfer 皮质重建（静息态 FC 分析不需要）
- `--clean-workdir` — 完成后自动清理工作目录，避免磁盘膨胀
- `--nprocs/omp-nthreads` — 由 `run_bd_pipeline.sh` 根据并行数自动计算

### 3.3 批量并行执行

```bash
# 3 人并行（默认），全部待处理被试
bash run_bd_pipeline.sh

# 5 人并行，只跑 10 个
bash run_bd_pipeline.sh 5 10
```

流水线逻辑：fMRIPrep 完成一个被试 → 立即调用 `preprocess_ucla_cnp.py` 提取 ROI → 验证 shape/NaN/flat → 清理 work 目录 → 写入 `pipeline_done.txt`。

### 3.4 完成进度

| 指标 | 数值 |
|------|------|
| 目标被试 (BD 亚组) | 49 |
| 已完成 fMRIPrep + ROI | **30** |
| fMRIPrep 完成 | 33 |
| 平均单被试耗时 | ~8-10 小时 |
| 并行数 | 3 人（可调整） |

### 3.5 硬件环境

| 项目 | 规格 |
|------|------|
| CPU | AMD EPYC 9554, 128 核, 251 GB RAM |
| GPU | NVIDIA RTX 5090 ×2 (32 GB) — fMRIPrep **不使用** GPU |
| 存储 | /storage 7.3 TB (可用 ~175 GB) |

## 4. 目录结构

```
ucla_cnp/
├── README.md                         ← 本文件
├── dataset_description.json           # 数据集元信息
├── participants.tsv                   # 受试者表（人口学 + 模态标记）
├── license.txt                        # FreeSurfer license（fmriprep 必需）
├── download.sh                        # 首次下载脚本
│
├── run_bd_pipeline.sh                 # 批量并行流水线 (fMRIPrep → ROI → 验证)
├── pipeline_done.txt                  # 已完成被试列表 (每行一个 ID)
├── docker_commands_bd.txt             # 单被试 fMRIPrep 命令参考
│
├── preprocess_ucla_cnp.py             # ← 项目根目录的同名脚本
│
├── subject_lists/                     # 按诊断分组的被试列表
│   ├── hc_subjects.txt                #  122 健康对照
│   ├── schz_subjects.txt              #   50 精神分裂症
│   ├── bd_subjects.txt                #   49 双相障碍
│   └── adhd_subjects.txt              #   40 ADHD
│
├── derivatives/
│   ├── fmriprep/                      # fMRIPrep 预处理输出
│   │   └── sub-XXXXX/
│   │       ├── anat/                  # T1w 预处理 + MNI 配准
│   │       ├── func/                  # BOLD 预处理 + confounds
│   │       │   ├── *_desc-preproc_bold.nii.gz    ← 核心输出
│   │       │   └── *_desc-confounds_timeseries.tsv
│   │       └── figures/               # QA 报告 (SVG/HTML)
│   └── work/                          # fMRIPrep 临时工作目录 (自动清理)
│
├── roi_timeseries/
│   └── HO62/                          # Harvard-Oxford 62 ROI 时间序列
│       ├── roi_labels.tsv             # 列名 X1..X62 ↔ ROI 名称映射
│       └── sub-XXXXX.txt              # 152 × 62, tab 分隔, z-score 标准化
│
└── sub-XXXXX/                         # 原始 BIDS 数据 (270 个目录)
    ├── anat/sub-XXXXX_T1w.nii.gz
    └── func/sub-XXXXX_task-rest_bold.nii.gz
```

## 5. ROI 图谱定义 (HO62)

| 组成部分 | 数量 | 说明 |
|----------|------|------|
| 皮层 (cortical) | 48 | Harvard-Oxford cort-maxprob-thr25-2mm，双侧合并 (symmetric_split=False) |
| 皮层下 (subcortical) | 14 | Harvard-Oxford sub-maxprob-thr25-2mm，筛选 7 类灰质结构 ×2 半球 |
| **总计** | **62** | |

皮层下保留的 14 个 ROI：Thalamus, Caudate, Putamen, Pallidum, Hippocampus, Amygdala, Accumbens（均为左右分开）。

> 注意：皮层 48 个节点未按半球拆分，而皮层下 14 个节点按左右拆分，整套 62 节点存在粒度不对称。

## 6. fMRIPrep 输出精简说明

每个被试 fMRIPrep 输出约 280 MB，其中 **FSTA 实际只用 2 个文件**（~200 MB）：

| 文件 | 必需? | 说明 |
|------|-------|------|
| `func/*_desc-preproc_bold.nii.gz` | ✅ | BOLD 预处理时间序列，MNI 空间 |
| `func/*_desc-confounds_timeseries.tsv` | ✅ | 头动/组织信号回归量 (26 参数模型) |
| `anat/` 下所有文件 | ❌ | T1w/分割/变换矩阵 — 只用于 fMRIPrep 内部配准 |
| `func/*_boldref.nii.gz` | ❌ | BOLD 参考像 — QA 用的中间产物 |
| `figures/` | ❌ | HTML/SVG QA 报告 |
| `log/` | ❌ | fmriprep.toml 运行日志 |

已删除 `*2009cAsym*` 空间文件，节省约 1.3 GB。

## 7. 引用

- **原始数据**: Poldrack, R. et al. *A phenome-wide examination of neural and cognitive function*. Scientific Data 3, 160110 (2016). https://www.nature.com/articles/sdata2016110
- **fMRIPrep**: Esteban, O. et al. *fMRIPrep: a robust preprocessing pipeline for fMRI data*. Nature Methods 16, 111–116 (2019). https://doi.org/10.1038/s41592-018-0235-4
- **OpenNeuro**: ds000030, DOI: [10.18112/openneuro.ds000030.v1.0.0](https://doi.org/10.18112/openneuro.ds000030.v1.0.0)

## 8. 更新记录

| 日期 | 操作 |
|------|------|
| 2026-07-10 | 首次下载 T1w + rest fMRI (BIDS格式, aria2c) |
| 2026-07-13 | 补全缺失的 func 数据 (58 个受试者)；fMRIPrep 单被试测试 |
| 2026-07-13 | 搭建并行流水线 `run_bd_pipeline.sh`，开始 BD 亚组批量处理 |
| 2026-07-14 | 删除 2009cAsym 空间文件，统一使用 MNI152NLin6Asym |
| 2026-07-14 | 完成 30 被试 ROI 提取 (HO62)，3 人并行持续处理中 |
