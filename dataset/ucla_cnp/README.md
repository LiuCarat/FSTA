# UCLA CNP 数据集 (ds000030)

> **UCLA Consortium for Neuropsychiatric Phenomics LA5c Study**
>
> OpenNeuro 编号: [ds000030](https://openneuro.org/datasets/ds000030) · BIDS 版本: 1.0.2 · 许可: CC0

---

## 1. 数据集概况

| 项目 | 数值 |
|------|------|
| 受试者总数 (TSV) | 272 |
| 已下载目录 | 269 |
| func + anat 完整可用 | **260** |
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
| CONTROL (健康对照) | 121 | `subject_lists/hc_subjects.txt` |
| SCHZ (精神分裂症) | 50 | `subject_lists/schz_subjects.txt` |
| BIPOLAR (双相障碍) | 49 | `subject_lists/bd_subjects.txt` |
| ADHD (注意缺陷多动障碍) | 40 | `subject_lists/adhd_subjects.txt` |
| 其他 (缺模态) | 12 | — |
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
  │    ├─ 图谱: Harvard-Oxford (cort + sub, thr25-2mm)
  │    │    ├─ HO110 (110 ROI): 皮层 96 (左右拆分) + 皮层下 14 (左右拆分)
  │    │    └─ HO55  (55 ROI):  皮层 48 (双侧合并) + 皮层下 7  (双侧合并)
  │    ├─ 去噪: 24 头动参数 + WM + CSF + 异常帧 (motion_outlier / non_steady_state_outlier)
  │    ├─ 滤波: 0.01–0.1 Hz 带通 (nilearn, 不另加 cosine 回归量)
  │    ├─ 标准化: z-score (逐时间序列)
  │    └─ 验证: 体素覆盖 + 头动 FD 质控 → subject_qc.tsv
  │
  └─ 输出: dataset/ucla_roi/{HO55,HO110}/{BD,HC}/sub-XXXXX.txt
           (152 × N, tab 分隔) + roi_labels.tsv + subject_qc.tsv
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
# HO110 图谱, BD 组, 5 人并行
python preprocess_ucla_cnp.py --pipeline bd --atlas HO110 --jobs 5

# HO55 图谱, 全部四组
python preprocess_ucla_cnp.py --pipeline bd   --atlas HO55 --jobs 5
python preprocess_ucla_cnp.py --pipeline hc   --atlas HO55 --jobs 5
```

流水线逻辑：fMRIPrep 完成一个被试 → 立即调用 ROI 提取 → 验证 → 清理 work 目录 → 写入 `pipeline_done.txt`。

### 3.4 完成进度

| 指标 | 数值 |
|------|------|
| BD 组 fMRIPrep + HO55/HO110 ROI | **49** / 49 ✅ |
| HC 组 fMRIPrep + HO55/HO110 ROI | **121** / 122 (sub-10524 仅128时间点，已排除) |
| **可用于分析的受试者** | **170** (49 BD + 121 HC) |
| 平均单被试耗时 (fMRIPrep) | ~8-10 小时 |
| ROI 提取耗时 (每被试) | ~2-3 秒 |

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
├── preprocess_ucla_cnp.py               # ← utils/ 下的同名脚本
│
├── subject_lists/                       # 按诊断分组的被试列表
│   ├── hc_subjects.txt                #  121 健康对照
│   ├── schz_subjects.txt              #   50 精神分裂症
│   ├── bd_subjects.txt                #   49 双相障碍
│   └── adhd_subjects.txt              #   40 ADHD
│
├── phenotype/                         # 表型数据（量表/认知测试/人口学）
│   ├── demographics.tsv/json          # 教育、SES
│   ├── scid.tsv/json                  # 金标准诊断
│   ├── medication.tsv/json            # 用药信息
│   ├── bprs/sans/saps.tsv/json        # SZ 症状量表
│   ├── hamilton/ymrs.tsv/json         # BD 症状量表
│   ├── asrs/adhd.tsv/json             # ADHD 量表
│   └── ... (共 16 组, 见 download.sh)  # Chapman / WAIS / TCI 等
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
└── sub-XXXXX/                           # 原始 BIDS 数据 (269 个目录)
    ├── anat/sub-XXXXX_T1w.nii.gz
    └── func/sub-XXXXX_task-rest_bold.nii.gz
```

## 5. ROI 图谱定义

ROI 时间序列输出于 `dataset/ucla_roi/`，提供两种 Harvard-Oxford (cort + sub, thr25-2mm) 变体：

### HO110（110 ROI，默认）

| 组成部分 | 数量 | 说明 |
|----------|------|------|
| 皮层 (cortical) | 96 | symmetric_split=True，48 区 × 2 半球，如 Left/Right Frontal Pole |
| 皮层下 (subcortical) | 14 | Thalamus, Caudate, Putamen, Pallidum, Hippocampus, Amygdala, Accumbens × 2 半球 |
| **总计** | **110** | 皮层和皮层下均为左右拆分，粒度统一 |

### HO55（55 ROI）

| 组成部分 | 数量 | 说明 |
|----------|------|------|
| 皮层 (cortical) | 48 | symmetric_split=False，双侧合并，如 Frontal Pole |
| 皮层下 (subcortical) | 7 | 左右配对合并，如 Thalamus, Caudate, … Accumbens |
| **总计** | **55** | 全部双侧合并，节点粒度完全均匀 |

### 已完成提取

| 图谱 | BD | HC | 合计 | 路径 |
|------|-----|-----|------|------|
| HO110 | 49 | 121 | 170 | `dataset/ucla_roi/HO110/` |
| HO55 | 49 | 121 | 170 | `dataset/ucla_roi/HO55/` |

每个受试者输出：`sub-XXXXX.txt` (152 × N, tab 分隔) + `roi_labels.tsv` + `subject_qc.tsv`

## 6. fMRIPrep 输出精简说明

每个被试 fMRIPrep 输出约 280 MB，其中 **FSTA 实际只用 2 个文件**（~200 MB）：

| 文件 | 必需? | 说明 |
|------|-------|------|
| `func/*_desc-preproc_bold.nii.gz` | ✅ | BOLD 预处理时间序列，MNI 空间 |
| `func/*_desc-confounds_timeseries.tsv` | ✅ | 头动/组织信号回归量 (24 头动 + WM + CSF + spike regressors) |
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
| 2026-07-13 | 搭建并行流水线，开始 BD 亚组批量处理 |
| 2026-07-14 | 删除 2009cAsym 空间文件，统一使用 MNI152NLin6Asym |
| 2026-07-16 | 移除 sub-10524 (仅128时间点)；新增 phenotype 表型数据下载 |
| 2026-07-16 | 新增 HO110/HO55 图谱支持，弃用 HO62 (粒度不对称) |
| 2026-07-16 | 去噪增加 motion_outlier / non_steady_state_outlier spike regressors |
| 2026-07-16 | 自动生成 subject_qc.tsv (mean_fd, roi_voxels 等) |
| 2026-07-16 | 49 BD + 121 HC = 170 受试者 HO55/HO110 ROI 全部提取完成 |
| 2026-07-16 | 新增 compare_BD_HC.py: 有向网络 FSTA 5折CV + HC3 GLM + FDR + 敏感性分析 |
