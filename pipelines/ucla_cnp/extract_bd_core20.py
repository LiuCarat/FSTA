#!/usr/bin/env python3
"""Extract BD-Core20 time series using the frozen UCLA CNP denoising settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .paths import ATLAS_LABELS_NAME, BDCORE20_DIR, FMRIPREP_DIR, SPACE
    from .preprocess import extract_roi_timeseries, load_bdcore20_atlas, save_timeseries_txt
except ImportError:
    from paths import ATLAS_LABELS_NAME, BDCORE20_DIR, FMRIPREP_DIR, SPACE
    from preprocess import extract_roi_timeseries, load_bdcore20_atlas, save_timeseries_txt


ROI_LABEL_FILE = ATLAS_LABELS_NAME



def find_func_dir(fmriprep_dir: Path, subject_id: str, group: str | None) -> tuple[Path, str]:
    subject = f"sub-{subject_id}"
    groups = [group] if group else [p.name for p in sorted(fmriprep_dir.iterdir()) if p.is_dir()]
    for candidate_group in groups:
        func = fmriprep_dir / candidate_group / subject / "func"
        if func.exists():
            return func, candidate_group
    raise RuntimeError(f"{subject}: 未找到 fMRIPrep func 目录")


def subject_ids(fmriprep_dir: Path, group: str | None, requested: str | None) -> list[str]:
    if requested:
        return [x.strip().replace("sub-", "") for x in requested.split(",") if x.strip()]
    groups = [group] if group else [p.name for p in sorted(fmriprep_dir.iterdir()) if p.is_dir()]
    found = set()
    for name in groups:
        root = fmriprep_dir / name
        if root.exists():
            found.update(p.name.replace("sub-", "") for p in root.glob("sub-*") if p.is_dir())
    return sorted(found)


def process_one(subject_id: str, args: argparse.Namespace, atlas_img, roi_names: list[str], outdir: Path) -> dict:
    func_dir, group = find_func_dir(args.fmriprep_dir, subject_id, args.group)
    bold_files = sorted(func_dir.glob(f"*_space-{SPACE}*_desc-preproc_bold.nii.gz"))
    confounds_files = sorted(func_dir.glob("*_desc-confounds_timeseries.tsv"))
    if not bold_files or not confounds_files:
        raise RuntimeError(f"sub-{subject_id}: 缺少 MNI BOLD 或 confounds")
    bold_path = bold_files[0]
    ts, voxel_counts, motion_qc = extract_roi_timeseries(
        str(bold_path), str(confounds_files[0]), atlas_img, roi_names,
        t_r=args.tr, smooth_fwhm=None, detrend=True, low_pass=0.1,
        high_pass=0.01, standardize="zscore_sample",
    )
    if ts.ndim != 2 or ts.shape[1] != 20:
        raise RuntimeError(f"sub-{subject_id}: 输出形状错误 {ts.shape}，应为 T×20")
    if not np.isfinite(ts).all():
        raise RuntimeError(f"sub-{subject_id}: 时间序列包含 NaN/Inf")
    group_dir = outdir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    save_timeseries_txt(ts, str(group_dir / f"sub-{subject_id}.txt"))

    atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int32)
    raw_counts = np.array([np.sum(atlas_data == i) for i in range(1, 21)], dtype=float)
    qc = {
        "subject_id": subject_id, "group": group, "n_timepoints": ts.shape[0],
        "mean_fd": motion_qc.get("mean_fd"), "max_fd": motion_qc.get("max_fd"),
    }
    for i, (count, raw_count) in enumerate(zip(voxel_counts, raw_counts), 1):
        qc[f"roi_{i:02d}_voxel_count"] = int(count)
        qc[f"roi_{i:02d}_coverage"] = float(count / raw_count) if raw_count else np.nan
        qc[f"roi_{i:02d}_temporal_sd"] = float(np.std(ts[:, i - 1], ddof=1))
        qc[f"roi_{i:02d}_nan_fraction"] = float(np.mean(~np.isfinite(ts[:, i - 1])))
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=BDCORE20_DIR)
    parser.add_argument("--fmriprep-dir", type=Path, default=FMRIPREP_DIR)
    parser.add_argument("--outdir", type=Path, default=BDCORE20_DIR)
    parser.add_argument("--subject", default=None, help="逗号分隔的 subject ID；省略则批量处理")
    parser.add_argument("--group", choices=["BD", "HC", "SCHZ", "ADHD"], default=None)
    parser.add_argument("--tr", type=float, default=None)
    args = parser.parse_args()
    args.atlas_dir = args.atlas_dir.resolve()
    args.fmriprep_dir = args.fmriprep_dir.resolve()
    args.outdir = args.outdir.resolve()
    atlas_img, roi_names = load_bdcore20_atlas(args.atlas_dir)
    ids = subject_ids(args.fmriprep_dir, args.group, args.subject)
    if not ids:
        raise RuntimeError("没有找到可处理的受试者")
    records = []
    for sid in ids:
        print(f"\n--- sub-{sid} ---")
        try:
            records.append(process_one(sid, args, atlas_img, roi_names, args.outdir))
        except Exception as exc:
            print(f"[Error] sub-{sid}: {exc}", file=sys.stderr)
            raise
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.outdir / "subject_qc.tsv", sep="\t", index=False, float_format="%.8f")
    labels = pd.read_csv(args.atlas_dir / ROI_LABEL_FILE, sep="\t")
    labels[["index", "name"]].rename(columns={"index": "label_id", "name": "roi_name"}).assign(column=lambda x: "X" + x["label_id"].astype(str)).to_csv(args.outdir / "roi_labels.tsv", sep="\t", index=False, columns=["column", "label_id", "roi_name"])
    print(f"\n完成: {len(records)}/{len(ids)}；输出 {args.outdir}")


if __name__ == "__main__":
    main()
