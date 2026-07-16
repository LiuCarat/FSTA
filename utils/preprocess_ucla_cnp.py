#!/usr/bin/env python3
"""
ucla_cnp 数据预处理：fMRIPrep → ROI 提取 → QC → 清理

────────────────────────────────────────────────────────
图谱变体（Harvard-Oxford cort + sub, thr25-2mm）
────────────────────────────────────────────────────────

  HO110 (110 ROI) — 皮层 96 (symmetric_split) + 皮层下 14 (左右分开)
      ▸ 皮层：48 区 × 2 半球，如 "Left Frontal Pole" / "Right Frontal Pole"
      ▸ 皮层下：7 类 × 2 半球，如 "Left Amygdala" / "Right Amygdala"
      ▸ 适用于需要皮层偏侧化信息的 BD vs HC 机制分析

  HO55  (55 ROI)  — 皮层 48 (双侧合并) + 皮层下 7 (左右合并)
      ▸ 皮层：48 区双侧合并，如 "Frontal Pole"
      ▸ 皮层下：7 类双侧合并，如 "Amygdala"
      ▸ 适用于全对称分析，粒度均匀

────────────────────────────────────────────────────────
预处理流水线（冻结配置）
────────────────────────────────────────────────────────
  标准空间    MNI152NLin6Asym (res-native)
  平滑        0
  去噪        24 参数头动 + WM + CSF + 异常帧回归变量
              (motion_outlier_XX, non_steady_state_outlier_XX)
  滤波        0.01–0.1 Hz 带通 (nilearn 内建, 不额外加 cosine 回归量)
  TR          从 BOLD JSON 自动读取
  标准化      z-score sample
  输出        T × N (tab 分隔, 首行 X1..XN)
  QC          每名受试者自动生成 subject_qc.tsv

────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────

  # ROI 提取（需要先有 fMRIPrep 输出，默认 HO110）
  python preprocess_ucla_cnp.py --subject 10273
  python preprocess_ucla_cnp.py --all --atlas HO110
  python preprocess_ucla_cnp.py --all --atlas HO55
  python preprocess_ucla_cnp.py --list

  # 全自动流水线（fMRIPrep + ROI + QC + 清理）
  python preprocess_ucla_cnp.py --pipeline bd --atlas HO110
  python preprocess_ucla_cnp.py --pipeline bd --atlas HO110 --jobs 5 --limit 6
  python preprocess_ucla_cnp.py --pipeline hc --atlas HO110 --jobs 5

依赖:
    - nilearn >= 0.10, nibabel, numpy, pandas
    - docker (fMRIPrep 25.2.5)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / 'dataset' / 'ucla_cnp'
FMRIPREP_BASE = DATASET_DIR / 'derivatives' / 'fmriprep'
WORK_DIR = DATASET_DIR / 'derivatives' / 'work'
LICENSE_FILE = DATASET_DIR / 'license.txt'
SUBJECT_LISTS = DATASET_DIR / 'subject_lists'
OUTPUT_DIR_HO110 = PROJECT_ROOT / 'dataset' / 'ucla_roi' / 'HO110'
OUTPUT_DIR_HO55  = PROJECT_ROOT / 'dataset' / 'ucla_roi' / 'HO55'

# 默认输出目录
OUTPUT_DIR = OUTPUT_DIR_HO110

SPACE = 'MNI152NLin6Asym'
DOCKER_IMAGE = 'nipreps/fmriprep:25.2.5'


def _fmriprep_dir(group=None):
    """返回 fMRIPrep 输出目录。group 为 BD/HC/SCHZ/ADHD 时指向分组子目录。"""
    if group:
        return FMRIPREP_BASE / group
    return FMRIPREP_BASE


def _detect_group(subject_id):
    """在分组目录中查找受试者所属的组别 (BD/HC/SCHZ/ADHD)。"""
    subject_label = f'sub-{subject_id}'
    for gd in sorted(FMRIPREP_BASE.iterdir()):
        if not gd.is_dir():
            continue
        if (gd / subject_label).exists():
            return gd.name
    return None


SUBCORTICAL_KEEP = [
    'Left Thalamus', 'Left Caudate', 'Left Putamen', 'Left Pallidum',
    'Left Hippocampus', 'Left Amygdala', 'Left Accumbens',
    'Right Thalamus', 'Right Caudate', 'Right Putamen', 'Right Pallidum',
    'Right Hippocampus', 'Right Amygdala', 'Right Accumbens',
]
MOTION_BASE = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
FD_THRESHOLD = 0.5


def _build_motion_24():
    names = []
    for b in MOTION_BASE:
        names.extend([b, f'{b}_derivative1', f'{b}_power2', f'{b}_derivative1_power2'])
    return names


CONFOUNDS_BASE = _build_motion_24() + ['csf', 'white_matter']

# ============================================================
# 图谱（懒加载，全局复用）
# ============================================================
_atlas_cache = None


def _load_atlas(atlas_type='HO110', data_dir=None):
    global _atlas_cache
    cache_key = (atlas_type, data_dir)
    if _atlas_cache is not None and _atlas_cache[0] == cache_key:
        return _atlas_cache[1], _atlas_cache[2]

    from nilearn import datasets
    from nilearn.image import new_img_like

    # ── HO110: 皮层左右拆分 (96) + 皮层下左右拆分 (14) ──
    if atlas_type == 'HO110':
        print('[Atlas] 加载 Harvard-Oxford 皮层图谱 (symmetric_split, ~96 ROI) ...')
        cort = datasets.fetch_atlas_harvard_oxford(
            'cort-maxprob-thr25-2mm', symmetric_split=True, data_dir=data_dir)
        cort_img = nib.load(cort.maps) if isinstance(cort.maps, str) else cort.maps
        cort_labels = [str(l) for l in cort.labels[1:]]

        print('[Atlas] 加载 Harvard-Oxford 皮层下图谱 (14 左右分开 ROI) ...')
        sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm', data_dir=data_dir)
        sub_img = nib.load(sub.maps) if isinstance(sub.maps, str) else sub.maps
        sub_labels_full = [str(l) for l in sub.labels[1:]]
        sub_labels, sub_indices = [], []
        for idx, name in enumerate(sub_labels_full, start=1):
            if name in SUBCORTICAL_KEEP:
                sub_labels.append(name)
                sub_indices.append(idx)

        cort_data = cort_img.get_fdata().astype(np.int32)
        sub_data = sub_img.get_fdata().astype(np.int32)
        combined_data = cort_data.copy()
        for new_id, old_id in enumerate(sub_indices, start=len(cort_labels) + 1):
            combined_data[sub_data == old_id] = new_id
        combined_img = new_img_like(cort_img, combined_data.astype(np.float32))
        all_labels = cort_labels + sub_labels
        print(f'[Atlas] HO110: {len(all_labels)} ROI (皮层 {len(cort_labels)} + 皮层下 {len(sub_labels)})')
        _atlas_cache = (cache_key, combined_img, all_labels)
        return combined_img, all_labels

    # ── HO55: 皮层双侧合并 (48) + 皮层下左右合并 (7) ──
    if atlas_type == 'HO55':
        print('[Atlas] 加载 Harvard-Oxford 皮层图谱 (48 双侧 ROI) ...')
        cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', data_dir=data_dir)
        cort_img = nib.load(cort.maps) if isinstance(cort.maps, str) else cort.maps
        cort_labels = [str(l) for l in cort.labels[1:]]

        print('[Atlas] 加载 Harvard-Oxford 皮层下图谱 (合并左右 → 7 ROI) ...')
        sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm', data_dir=data_dir)
        sub_img = nib.load(sub.maps) if isinstance(sub.maps, str) else sub.maps
        sub_labels_full = [str(l) for l in sub.labels[1:]]

        # 将左右配对合并为双侧 ROI
        pair_groups = {}
        for idx, name in enumerate(sub_labels_full, start=1):
            if name in SUBCORTICAL_KEEP:
                base = name.replace('Left ', '').replace('Right ', '')
                if base not in pair_groups:
                    pair_groups[base] = []
                pair_groups[base].append(idx)

        sub_data = sub_img.get_fdata().astype(np.int32)
        merged_sub_data = np.zeros_like(sub_data)
        merged_sub_labels = []
        for new_id, (base, indices) in enumerate(pair_groups.items(), start=1):
            merged_sub_labels.append(base)
            for old_idx in indices:
                merged_sub_data[sub_data == old_idx] = new_id

        cort_data = cort_img.get_fdata().astype(np.int32)
        combined_data = cort_data.copy()
        for new_id in range(1, len(merged_sub_labels) + 1):
            combined_data[merged_sub_data == new_id] = len(cort_labels) + new_id
        combined_img = new_img_like(cort_img, combined_data.astype(np.float32))
        all_labels = cort_labels + merged_sub_labels
        print(f'[Atlas] HO55: {len(all_labels)} ROI (皮层 {len(cort_labels)} + 皮层下 {len(merged_sub_labels)})')
        _atlas_cache = (cache_key, combined_img, all_labels)
        return combined_img, all_labels

    raise ValueError(f'未知图谱类型: {atlas_type!r}，可选: HO110, HO55')


# ============================================================
# ROI 提取
# ============================================================
def extract_roi_timeseries(bold_path, confounds_path, atlas_img, roi_names,
                           t_r=None, smooth_fwhm=None, detrend=True,
                           low_pass=0.1, high_pass=0.01, standardize='zscore_sample'):
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.image import index_img, resample_to_img

    if t_r is None:
        bold_json = Path(str(bold_path).replace('.nii.gz', '.json'))
        with open(bold_json) as f:
            t_r = float(json.load(f)['RepetitionTime'])

    nyquist = 1.0 / (2.0 * t_r)
    if low_pass is not None and low_pass >= nyquist:
        raise ValueError(f'low_pass={low_pass} >= Nyquist={nyquist:.4f} (TR={t_r}s)')

    confounds_arr = None
    motion_qc = None

    if confounds_path is not None:
        confounds_df = pd.read_csv(confounds_path, sep='\t')
        confounds_df.replace('n/a', np.nan, inplace=True)
        fd_col = 'framewise_displacement'
        if fd_col in confounds_df.columns:
            fd = pd.to_numeric(confounds_df[fd_col], errors='coerce').dropna().values
            motion_qc = {
                'mean_fd': round(float(np.mean(fd)), 4),
                'max_fd': round(float(np.max(fd)), 4),
                f'fd_gt_{FD_THRESHOLD}_frac': round(float((fd > FD_THRESHOLD).sum() / len(fd)), 4),
                'n_volumes': len(confounds_df),
            }
        spike_cols = [
            c for c in confounds_df.columns
            if c.startswith('motion_outlier_') or c.startswith('non_steady_state_outlier_')
        ]
        available = [c for c in CONFOUNDS_BASE if c in confounds_df.columns] + spike_cols
        if available:
            confounds_arr = confounds_df[available].apply(pd.to_numeric, errors='coerce').fillna(0)
        if spike_cols:
            print(f'[Confounds] 24 头动 + WM/CSF + {len(spike_cols)} 异常帧回归变量')

    if motion_qc is None:
        motion_qc = {'mean_fd': None, 'max_fd': None, f'fd_gt_{FD_THRESHOLD}_frac': None, 'n_volumes': None}

    bold_img = nib.load(bold_path)
    atlas_resampled = resample_to_img(atlas_img, index_img(bold_img, 0), interpolation='nearest')
    atlas_data = np.asarray(atlas_resampled.dataobj, dtype=np.int32)

    n_rois = len(roi_names)
    voxel_counts = np.array([int(np.sum(atlas_data == i)) for i in range(1, n_rois + 1)])
    zero_voxel = np.where(voxel_counts == 0)[0]
    if len(zero_voxel) > 0:
        bad_names = [roi_names[i] for i in zero_voxel]
        raise RuntimeError(f'{len(zero_voxel)} 个 ROI 无体素: {bad_names}')

    print(f'[Voxel] {n_rois}/{n_rois} ROI 覆盖 (min={voxel_counts.min()}, median={int(np.median(voxel_counts))}, max={voxel_counts.max()})')
    print(f'[Motion] mean FD={motion_qc["mean_fd"]}, max FD={motion_qc["max_fd"]}')

    masker = NiftiLabelsMasker(
        labels_img=atlas_resampled,
        labels=['Background'] + roi_names,
        resampling_target=None,
        t_r=t_r,
        standardize=standardize,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        smoothing_fwhm=smooth_fwhm,
        memory=None, memory_level=0, verbose=0, reports=False,
    )

    print(f'[Extract] {os.path.basename(bold_path)} ...')
    time_series = masker.fit_transform(bold_img, confounds=confounds_arr)
    print(f'[Extract] shape={time_series.shape}, NaN={np.isnan(time_series).sum()}, range=[{time_series.min():.4f}, {time_series.max():.4f}]')

    return time_series, voxel_counts, motion_qc


# ============================================================
# 保存
# ============================================================
def save_timeseries_txt(time_series, output_path):
    T, N = time_series.shape
    header = '\t'.join([f'X{i+1}' for i in range(N)])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, time_series, delimiter='\t', header=header, fmt='%.6f', comments='')
    print(f'[Save] {output_path} ({T} × {N})')


def save_roi_labels(labels, output_dir):
    path = os.path.join(output_dir, 'roi_labels.tsv')
    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w') as f:
        f.write('column\tlabel_id\troi_name\n')
        for i, name in enumerate(labels, start=1):
            f.write(f'X{i}\t{i}\t{name}\n')


def save_subject_qc(qc_records, output_dir):
    """将全部受试者的 QC 记录保存为 subject_qc.tsv。"""
    if not qc_records:
        return
    path = os.path.join(output_dir, 'subject_qc.tsv')
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(qc_records)
    columns = [
        'subject', 'group',
        'mean_fd', 'max_fd', 'fd_gt_0.5_frac', 'n_volumes',
        'n_rois', 'min_roi_voxels', 'median_roi_voxels', 'max_roi_voxels',
        'status',
    ]
    df = df[[c for c in columns if c in df.columns]]
    df.to_csv(path, sep='\t', index=False, float_format='%.6f')
    n_ok = sum(1 for r in qc_records if r.get('status') == 'ok')
    print(f'[QC] 受试者质控已保存至 {path} ({len(df)} 条记录, {n_ok} 通过)')


# ============================================================
# 单个受试者 ROI 提取
# ============================================================
def process_subject(subject_id, atlas_img, roi_names, output_dir,
                    group=None, t_r=None, smooth_fwhm=None, **masker_kwargs):
    subject_label = f'sub-{subject_id}'
    if group:
        fmri_dir = _fmriprep_dir(group)
        func_dir = fmri_dir / subject_label / 'func'
    else:
        # 自动在所有分组目录中查找
        func_dir = None
        for gd in sorted(FMRIPREP_BASE.iterdir()):
            if not gd.is_dir():
                continue
            candidate = gd / subject_label / 'func'
            if candidate.exists():
                func_dir = candidate
                group = gd.name
                break
        if func_dir is None:
            raise RuntimeError(f'{subject_label}: 在任何分组目录中未找到 fMRIPrep 输出')
    if not func_dir.exists():
        raise RuntimeError(f'{subject_label}: fMRIPrep func 目录不存在')

    bold_files = sorted(func_dir.glob(f'*_space-{SPACE}*_desc-preproc_bold.nii.gz'))
    if not bold_files:
        raise RuntimeError(f'{subject_label}: 未找到 {SPACE} BOLD')

    bold_path = str(bold_files[0])
    confounds_files = sorted(func_dir.glob('*_desc-confounds_timeseries.tsv'))
    if not confounds_files:
        raise RuntimeError(f'{subject_label}: 缺少 confounds')

    time_series, voxel_counts, motion_qc = extract_roi_timeseries(
        bold_path, str(confounds_files[0]), atlas_img, roi_names,
        t_r=t_r, smooth_fwhm=smooth_fwhm, **masker_kwargs
    )

    out_path = os.path.join(output_dir, f'{subject_label}.txt')
    save_timeseries_txt(time_series, out_path)

    # 构建 QC 记录
    qc = {
        'subject': subject_id,
        'group': group or '',
        'mean_fd': motion_qc.get('mean_fd'),
        'max_fd': motion_qc.get('max_fd'),
        'fd_gt_0.5_frac': motion_qc.get(f'fd_gt_{FD_THRESHOLD}_frac'),
        'n_volumes': motion_qc.get('n_volumes'),
        'n_rois': len(voxel_counts),
        'min_roi_voxels': int(voxel_counts.min()),
        'median_roi_voxels': int(np.median(voxel_counts)),
        'max_roi_voxels': int(voxel_counts.max()),
        'status': 'ok',
    }
    return qc


def find_preprocessed_subjects():
    subjects = []
    for group_dir in sorted(FMRIPREP_BASE.iterdir()):
        if not group_dir.is_dir():
            continue
        for sub_dir in sorted(group_dir.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith('sub-'):
                continue
            if list(sub_dir.glob(f'func/*_space-{SPACE}*_desc-preproc_bold.nii.gz')):
                subjects.append(sub_dir.name.replace('sub-', ''))
    return subjects


# ============================================================
# Pipeline 模式：fMRIPrep → ROI → 验证 → 清理
# ============================================================
def run_fmriprep(subject_id, group, nprocs=32, mem_mb=50000):
    label = f'sub-{subject_id}'
    fmri_dir = _fmriprep_dir(group)
    bold_file = fmri_dir / label / 'func' / f'{label}_task-rest_space-{SPACE}_desc-preproc_bold.nii.gz'
    conf_file = fmri_dir / label / 'func' / f'{label}_task-rest_desc-confounds_timeseries.tsv'

    if bold_file.exists() and conf_file.exists():
        return 'skip'

    log_file = DATASET_DIR / 'pipeline_logs' / f'{label}_fmriprep.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'docker', 'run', '--rm',
        '--user', f'{os.getuid()}:{os.getgid()}',
        '-v', f'{DATASET_DIR}:/data:ro',
        '-v', f'{fmri_dir}:/out',
        '-v', f'{WORK_DIR}:/work',
        '-v', f'{LICENSE_FILE}:/opt/freesurfer/license.txt:ro',
        DOCKER_IMAGE,
        '/data', '/out', 'participant',
        '--participant-label', str(subject_id),
        '--output-spaces', SPACE,
        '--fs-no-reconall',
        '--skip-bids-validation',
        '--clean-workdir',
        '--nprocs', str(nprocs),
        '--omp-nthreads', '2',
        '--mem-mb', str(mem_mb),
        '-w', f'/work/sub-{subject_id}',
    ]

    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        raise RuntimeError(f'fMRIPrep 失败, 日志: {log_file}')
    if not bold_file.exists():
        raise RuntimeError(f'fMRIPrep 完成但缺少 BOLD 文件')
    return 'done'


def cleanup_subject(subject_id, group):
    w = WORK_DIR / f'sub-{subject_id}'
    if w.exists():
        shutil.rmtree(w)
    fmri_dir = _fmriprep_dir(group)
    log = fmri_dir / f'sub-{subject_id}' / 'log'
    if log.exists():
        shutil.rmtree(log)


class Progress:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.start = time.time()

    def update(self, msg=''):
        pct = self.current * 100 // self.total
        w = 30
        f = pct * w // 100
        bar = '#' * f + ' ' * (w - f)
        elapsed = int(time.time() - self.start)
        eta = ''
        if self.current > 0:
            eta_sec = elapsed * (self.total - self.current) // self.current
            eta = f' ETA: {eta_sec//60:02d}:{eta_sec%60:02d}'
        print(f'\r\033[K[{bar}] {self.current:3d}/{self.total} ({pct}%){eta}  {msg}', end='', flush=True)

    def step(self, msg=''):
        self.current += 1
        self.update(msg)

    def done(self):
        elapsed = int(time.time() - self.start)
        print(f'\r\033[K[{"#" * 30}] {self.total}/{self.total} (100%)  耗时 {elapsed//60:02d}:{elapsed%60:02d} ({elapsed}s)')
        print()


def run_pipeline(group, jobs, limit, nprocs, mem_mb, atlas_type='HO110', output_dir=None):
    """全自动流水线：N 人并行 fMRIPrep，完成一个立即 ROI+验证+清理"""
    subject_file = SUBJECT_LISTS / f'{group.lower()}_subjects.txt'
    if not subject_file.exists():
        print(f'[Error] 找不到受试者名单: {subject_file}')
        sys.exit(1)

    ids = []
    with open(subject_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(line.split('\t')[0].replace('sub-', ''))
    total = len(ids)

    done_file = DATASET_DIR / f'{group.lower()}_pipeline_done.txt'
    done_set = set()
    if done_file.exists():
        with open(done_file) as f:
            done_set = {l.strip() for l in f if l.strip()}

    todo = [sid for sid in ids if sid not in done_set]
    batch = todo[:limit]
    n = len(batch)

    njobs = jobs
    np_per = nprocs or max(8, 128 // njobs)
    mb_per = mem_mb or max(16000, 80000 // njobs)

    print('=' * 60)
    print(f'  {group} 全自动流水线: {njobs} 人并行, {np_per}核/人, {mb_per}MB/人')
    print(f'  总计 {total} | 已完成 {len(done_set)} | 本次 {n}')
    print('=' * 60)
    print()

    if n == 0:
        print('全部完成！')
        return

    # 加载图谱
    atlas_img, roi_names = _load_atlas(atlas_type)
    group_dir = (output_dir or OUTPUT_DIR) / group
    os.makedirs(group_dir, exist_ok=True)
    save_roi_labels(roi_names, output_dir or OUTPUT_DIR)

    prog = Progress(n)

    def process_one(sid):
        try:
            result = run_fmriprep(sid, group, np_per, mb_per)
            qc = process_subject(sid, atlas_img, roi_names, group_dir, group=group)
            cleanup_subject(sid, group)
            with open(done_file, 'a') as f:
                f.write(f'{sid}\n')
            # 验证
            ts = np.loadtxt(os.path.join(group_dir, f'sub-{sid}.txt'), skiprows=1, delimiter='\t')
            assert ts.shape[0] == 152 and ts.shape[1] == len(roi_names), f'Shape={ts.shape}'
            assert np.isfinite(ts).all(), f'Non-finite values'
            return (sid, True, f'fMRIPrep:{result}', qc)
        except Exception as e:
            return (sid, False, str(e), {
                'subject': sid, 'group': group,
                'mean_fd': None, 'max_fd': None, 'fd_gt_0.5_frac': None,
                'n_volumes': None, 'n_rois': len(roi_names),
                'min_roi_voxels': None, 'median_roi_voxels': None,
                'max_roi_voxels': None, 'status': 'fail',
            })

    qc_records = []

    with ThreadPoolExecutor(max_workers=njobs) as pool:
        futures = {pool.submit(process_one, sid): sid for sid in batch}
        for fut in as_completed(futures):
            sid, ok, msg, qc = fut.result()
            qc_records.append(qc)
            prog.step(f'sub-{sid}: {msg}')
            if not ok:
                print()
                print(f'✗ sub-{sid} 失败: {msg}')
                save_subject_qc(qc_records, output_dir or OUTPUT_DIR)
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

    prog.done()
    save_subject_qc(qc_records, output_dir or OUTPUT_DIR)


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='ucla_cnp 预处理 — 提取 HO110/HO55 ROI 时间序列 / 全自动流水线'
    )

    # 模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=str, default=None, help='受试者 ID，逗号分隔')
    group.add_argument('--all', action='store_true', help='处理所有已 fMRIPrep 的受试者')
    group.add_argument('--list', action='store_true', help='列出已 fMRIPrep 的受试者')
    group.add_argument('--pipeline', type=str, default=None,
                       choices=['bd', 'hc', 'schz', 'adhd'],
                       help='全自动流水线 (fMRIPrep + ROI + 验证 + 清理)')

    # 图谱
    parser.add_argument('--atlas', type=str, default='HO110',
                        choices=['HO110', 'HO55'],
                        help='图谱 (默认 HO110=皮层拆分+皮层下拆分, HO55=全双侧合并)')
    parser.add_argument('--atlas-dir', type=str, default=None)

    # ROI 提取参数
    parser.add_argument('--tr', type=float, default=None)
    parser.add_argument('--smooth', type=float, default=0)
    parser.add_argument('--no-detrend', action='store_true')
    parser.add_argument('--low-pass', type=float, default=0.1)
    parser.add_argument('--high-pass', type=float, default=0.01)
    parser.add_argument('--standardize', type=str, default='zscore_sample',
                        choices=['zscore_sample', 'psc', 'false'])

    # 输出
    parser.add_argument('--output-dir', type=str, default=None)

    # Pipeline 参数
    parser.add_argument('--jobs', type=int, default=3, help='并行人数 (默认 3)')
    parser.add_argument('--limit', type=int, default=999, help='本次最多处理人数')
    parser.add_argument('--nprocs', type=int, default=None, help='每人 nprocs')
    parser.add_argument('--mem-mb', type=int, default=None, help='每人内存 MB')

    args = parser.parse_args()

    # ── Pipeline 模式 ──
    if args.pipeline:
        _atlas_to_dir = {'HO110': OUTPUT_DIR_HO110, 'HO55': OUTPUT_DIR_HO55}
        _out = _atlas_to_dir.get(args.atlas, OUTPUT_DIR_HO110)
        run_pipeline(args.pipeline.upper(), args.jobs, args.limit, args.nprocs, args.mem_mb,
                     atlas_type=args.atlas, output_dir=_out)
        return

    # ── ROI 提取模式 ──
    if args.list:
        subs = find_preprocessed_subjects()
        print(f'已预处理受试者 ({len(subs)} 人, {SPACE}):')
        for s in subs:
            print(f'  sub-{s}')
        return

    atlas_tag = args.atlas  # 'HO110' or 'HO55'
    _atlas_to_dir = {'HO110': OUTPUT_DIR_HO110, 'HO55': OUTPUT_DIR_HO55}
    output_dir = args.output_dir or str(_atlas_to_dir.get(args.atlas, OUTPUT_DIR_HO110))
    output_dir = Path(output_dir)

    smooth_fwhm = None if args.smooth == 0 else args.smooth
    standardize = False if args.standardize == 'false' else args.standardize

    print('=' * 60)
    print(f'加载 Harvard-Oxford 图谱 ({args.atlas}) ...')
    atlas_img, roi_names = _load_atlas(args.atlas, args.atlas_dir)
    print(f'图谱 ROI: {len(roi_names)}')
    print(f'输出目录: {output_dir}')
    print('=' * 60)

    os.makedirs(output_dir, exist_ok=True)
    save_roi_labels(roi_names, output_dir)

    if args.all:
        subject_ids = find_preprocessed_subjects()
        if not subject_ids:
            print(f'[Error] 未找到已预处理的受试者 ({SPACE})')
            sys.exit(1)
        print(f'将处理 {len(subject_ids)} 个受试者')
    else:
        subject_ids = [s.strip() for s in args.subject.split(',')]
        print(f'将处理 {len(subject_ids)} 个受试者: {subject_ids}')

    qc_records = []
    success = 0
    for sub_id in subject_ids:
        print(f'\n{"-" * 60}')
        print(f'处理受试者: sub-{sub_id}')
        print(f'{"-" * 60}')
        try:
            qc = process_subject(sub_id, atlas_img, roi_names, output_dir,
                                 t_r=args.tr, smooth_fwhm=smooth_fwhm,
                                 detrend=not args.no_detrend,
                                 low_pass=args.low_pass, high_pass=args.high_pass,
                                 standardize=standardize)
            qc_records.append(qc)
            success += 1
        except Exception as e:
            print(f'[Error] sub-{sub_id}: {e}')
            traceback.print_exc()
            qc_records.append({
                'subject': sub_id, 'group': _detect_group(sub_id) or '',
                'mean_fd': None, 'max_fd': None, 'fd_gt_0.5_frac': None,
                'n_volumes': None, 'n_rois': len(roi_names),
                'min_roi_voxels': None, 'median_roi_voxels': None,
                'max_roi_voxels': None, 'status': 'fail',
            })

    save_subject_qc(qc_records, output_dir)

    print(f'\n{"=" * 60}')
    print(f'完成: {success}/{len(subject_ids)} 受试者处理成功')
    print(f'输出目录: {output_dir}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
