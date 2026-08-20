import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
# Canonical diagnosis labels shared with Graph_BEC:
# 0 = TC/control, 1 = ASD.
TC_LABEL = 0
ASD_LABEL = 1
DX_TO_LABEL = {1: TC_LABEL, 2: ASD_LABEL}
LABEL_TO_GROUP = {TC_LABEL: "TC", ASD_LABEL: "ASD"}
SOURCE_ROI_COUNT = 116 # 原始脑区数量
ROI_COUNT = 90 # 目标脑区数量
ROI_INDICES = np.arange(ROI_COUNT, dtype=np.int64)


@dataclass(frozen=True)
class ABIDERecord:
    subject_id: str
    site_id: str
    label: int
    diagnosis: str
    time_series_path: Path

# 匹配表型文件和时间序列
def load_abide_records(
    data_root=REPO_ROOT / "dataset/ABIDE-I",
    pipeline="cpac",
    strategy="filt_noglobal",
    derivative="rois_aal",
):
    data_root = Path(data_root)
    phenotype_path = data_root / "Phenotypic_V1_0b_preprocessed1.csv"
    time_series_dir = data_root / pipeline / strategy
    suffix = f"_{derivative}.1D"
    # 读取表型文件并创建一个字典，键为FILE_ID，值为对应的行数据
    with phenotype_path.open(newline="", encoding="utf-8-sig") as phenotype_file:
        phenotype_rows = {
            row["FILE_ID"].strip(): row
            for row in csv.DictReader(phenotype_file)
            if row["FILE_ID"].strip() != "no_filename"
        }

    records = []
    unmatched = []
    # 遍历时间序列文件并匹配表型信息
    for time_series_path in sorted(time_series_dir.glob(f"*{suffix}")):
        subject_id = time_series_path.name[: -len(suffix)]
        row = phenotype_rows.get(subject_id)
        # 如果没有匹配的表型信息，则将该subject_id添加到unmatched列表中
        if row is None:
            unmatched.append(subject_id)
            continue
        # 如果匹配成功，则提取诊断信息并创建ABIDERecord对象
        diagnosis_code = int(float(row["DX_GROUP"]))
        label = DX_TO_LABEL[diagnosis_code]
        records.append(
            ABIDERecord(
                subject_id=subject_id,
                site_id=row["SITE_ID"].strip(),
                label=label,
                diagnosis=LABEL_TO_GROUP[label],
                time_series_path=time_series_path,
            )
        )
    return records

# 读取116列 AAL 数据并选择前90个脑区
def load_time_series(record, standardize=True):
    time_series = np.loadtxt(record.time_series_path, dtype=np.float32)
    # 检查数据维度是否正确
    if time_series.ndim != 2:
        raise ValueError(
            f"Expected a 2D time series for {record.subject_id}, got {time_series.shape}"
        )
    # 检查列数是否为116
    if time_series.shape[1] != SOURCE_ROI_COUNT:
        raise ValueError(
            f"Expected {SOURCE_ROI_COUNT} source columns for {record.subject_id}, "
            f"got {time_series.shape[1]}"
        )
    # 检查是否存在非有限值
    if not np.isfinite(time_series).all():
        raise ValueError(f"Non-finite values found for {record.subject_id}")

    time_series = time_series[:, ROI_INDICES]
    # 标准化时间序列数据
    if standardize:
        mean = time_series.mean(axis=0, keepdims=True)
        std = time_series.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        time_series = (time_series - mean) / std
    return time_series.astype(np.float32, copy=False)

# 生成数据清单
def write_manifest(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "subject_id",
                "site_id",
                "diagnosis",
                "label",
                "time_points",
                "roi_count",
                "time_series_path",
            ]
        )
        # 遍历记录并写入清单文件
        for record in records:
            time_series = load_time_series(record, standardize=False)
            writer.writerow(
                [
                    record.subject_id,
                    record.site_id,
                    record.diagnosis,
                    record.label,
                    time_series.shape[0],
                    time_series.shape[1],
                    str(record.time_series_path),
                ]
            )
