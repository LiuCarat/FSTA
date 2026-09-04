# ADHD200 数据集处理说明

本目录保存 ADHD200 数据集的下载结果、fMRIPrep 预处理结果、AAL ROI 时间序列和 Graph-BEC 输入文件。

## 当前筛选结果

用于下载和后续 fMRIPrep 的表型文件为：

```text
dataset/ADHD200/Phenotypic_Processing.csv
```

当前文件包含 **798 个受试者**。筛选条件为：

- `Age`、`Gender`、`Full4 IQ` 和 `Handedness` 均有值；
- 至少有一个 T1w 文件；
- 至少有一个 BOLD 文件；
- 排除 `DX=pending`；
- 排除 `Handedness=-999` 和非数值编码（当前排除 `L`）；
- 一个被试有多个 BOLD run 时，下载器默认只选择一个 run。

各站点数量如下：

| 站点 | 被试数 |
|---|---:|
| `KKI` | 83 |
| `NYU` | 209 |
| `OHSU` | 113 |
| `Peking_1` | 135 |
| `Peking_2` | 67 |
| `Peking_3` | 42 |
| `Pittsburgh` | 89 |
| `WashU` | 60 |
| **合计** | **798** |

`DX` 编码在 Graph-BEC 中按以下方式解释：

- `DX=0`：对照组；
- `DX=1/2/3`：患者组。

## 表型字段

`Phenotypic_Processing.csv` 的主要字段包括：

```text
ScanDir ID
Site
Gender
Age
Handedness
DX
Full4 IQ
```

其中：

- `Gender` 是二分类编码 `0/1`；
- `Age` 使用岁为单位，可以是小数，例如 `12.36`；
- `Handedness` 是利手评分，部分站点使用连续值，不应强行转换为整数；
- `DX` 是 ADHD200 的诊断编码；
- `Full4 IQ` 用作 Graph-BEC 的连续表型协变量。

fMRIPrep 本身不依赖 `DX`，但 Graph-BEC 训练需要有效的患者/对照标签。

## 下载数据

下载脚本会读取表型文件中的站点和被试 ID。默认每个被试、每个 task 只下载一个 BOLD run，并优先选择 `run-1`。

先查看下载清单：

```bash
python dataset/ADHD200/scripts/download_ADHD200.py \
    --subject-audit dataset/ADHD200/Phenotypic_Processing.csv \
    --dry-run
```

正式下载：

```bash
python dataset/ADHD200/scripts/download_ADHD200.py \
    --subject-audit dataset/ADHD200/Phenotypic_Processing.csv \
    --workers 1
```

网络稳定时可以增加线程数：

```bash
python dataset/ADHD200/scripts/download_ADHD200.py \
    --subject-audit dataset/ADHD200/Phenotypic_Processing.csv \
    --workers 4
```

下载脚本支持断点续传。下载过程中出现的 `.part` 文件表示文件尚未完成，不能送入 fMRIPrep。

## 原始 BIDS 目录

下载完成后，目录结构应类似：

```text
dataset/ADHD200/BIDS/<SITE>/sub-<ID>/ses-1/
├── anat/
│   └── *_T1w.nii.gz
└── func/
    └── *_task-rest*_bold.nii.gz
```

每个目标被试至少需要一个 T1w 和一个 BOLD。

## fMRIPrep 预处理

脚本位置：

```text
dataset/ADHD200/scripts/run_fmriprep_onebyone.sh
```

需要 Docker 和 FreeSurfer license：

```bash
export FS_LICENSE=dataset/ADHD200/fmriprep/license.txt
```

先处理一个被试进行测试：

```bash
bash dataset/ADHD200/scripts/run_fmriprep_onebyone.sh sub-1018959 KKI
```

处理全部已下载且满足文件条件的被试：

```bash
bash dataset/ADHD200/scripts/run_fmriprep_onebyone.sh
```

处理指定站点：

```bash
bash dataset/ADHD200/scripts/run_fmriprep_onebyone.sh '' NYU
```

输出目录为：

```text
dataset/ADHD200/fmriprep/<SITE>/sub-<ID>/
```

脚本只保留后续分析需要的文件：

- `space-MNI152NLin6Asym_desc-preproc_bold.nii.gz`；
- `desc-confounds_timeseries.tsv/json`；
- `space-MNI152NLin6Asym_desc-brain_mask.nii.gz`；
- BOLD JSON sidecar。

成功复制输出后，对应的 `.fmriprep.log` 会自动删除；失败或未完成的被试会保留日志。

### 站点特殊处理

NYU 和 OHSU 的原始 BOLD 缺少 JSON sidecar。脚本不会修改原始数据，而是在临时 BIDS 副本中补充：

- NYU：`RepetitionTime=2.0` 秒；
- OHSU：`RepetitionTime=2.5` 秒。

这两个值来自对应 BOLD NIfTI 头部。若需要覆盖默认值，可以设置：

```bash
NYU_BOLD_TR=2.0 \
  bash dataset/ADHD200/scripts/run_fmriprep_onebyone.sh '' NYU
```

```bash
OHSU_BOLD_TR=2.5 \
  bash dataset/ADHD200/scripts/run_fmriprep_onebyone.sh '' OHSU
```

## 生成 AAL ROI 时间序列和 QC

Graph-BEC 不直接读取 fMRIPrep 的 BOLD 文件，而是读取 AAL ROI 时间序列。因此 fMRIPrep 完成后，还需要运行：

```bash
python dataset/ADHD200/scripts/extract_aal90.py \
    --fmriprep-root dataset/ADHD200/fmriprep \
    --bids-root dataset/ADHD200/BIDS \
    --atlas dataset/ABIDE-II/atlas/AAL116_MNI152NLin6Asym.nii.gz \
    --output-root dataset/ADHD200/cpac/filt_noglobal \
    --phenotype-output dataset/ADHD200/Phenotypic_Processing.csv
```

该步骤会：

- 将 fMRIPrep BOLD 提取为 AAL116 时间序列；
- 保存前 116 个 ROI，Graph-BEC 实际使用前 90 个 ROI；
- 生成 `dataset/ADHD200/cpac/filt_noglobal/sub-<ID>_rois_aal.1D`；
- 从 fMRIPrep confounds 计算 QC 字段：
  - `func_mean_fd`；
  - `func_fd_gt_0_2`；
  - `func_fd_gt_0_5`；
  - `func_dvars`；
  - `func_quality`。

其中 `func_quality` 是 FD 不超过 `0.2` 的帧比例。这个步骤会重新生成带 QC 的 `Phenotypic_Processing.csv`，因此最终训练前不要手工删除 QC 列。

## Graph-BEC 训练

完成 ROI 和 QC 生成后运行：

```bash
python Graph_BEC/main_adhd200.py --input-mode raw
```

Graph-BEC 默认读取：

```text
dataset/ADHD200/Phenotypic_Processing.csv
dataset/ADHD200/cpac/filt_noglobal/
```

默认使用以下表型协变量：

```text
Age
Gender
Full4 IQ
Handedness
```

默认 QC 列为：

```text
func_mean_fd
func_dvars
func_quality
```

如果只使用已有 BEC archive 而不重新从 ROI 时间序列生成 BEC，可以使用：

```bash
python Graph_BEC/main_adhd200.py --input-mode bec
```

## 数据一致性检查

运行 Graph-BEC 前，至少确认：

```bash
find dataset/ADHD200/cpac/filt_noglobal \
    -name '*_rois_aal.1D' | wc -l
```

并检查最终表型文件包含 QC 列：

```bash
python - <<'PY'
import csv

path = "dataset/ADHD200/Phenotypic_Processing.csv"
with open(path, newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    fields = {field.strip() for field in reader.fieldnames or []}
    rows = list(reader)

required = {
    "ScanDir ID", "Site", "Gender", "Age", "Handedness", "DX", "Full4 IQ",
    "func_mean_fd", "func_dvars", "func_quality",
}
print("subjects:", len(rows))
print("missing columns:", sorted(required - fields))
PY
```

只有当 ROI 文件数量、表型行数和 QC 字段都与实际成功完成预处理的被试一致时，才建议开始 Graph-BEC 训练。
