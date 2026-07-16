#!/usr/bin/env python3
"""
BD vs HC 脑功能连接网络组间比较
────────────────────────────────────────────────────────
流水线:
  1. 加载 HO55 ROI 时间序列 + 人口学 + 头动质控
  2. 5折交叉验证训练 FSTA → 每人独立 N×N 邻接矩阵
  3. 逐边 GLM: edge ~ is_BD + age + sex + scanner + mean_fd
  4. Benjamini-Hochberg FDR 校正
  5. 敏感性分析: 随机等量 HC 重复 N 次 → 一致性评估

用法:
  # 完整主分析
  python compare_BD_HC.py --atlas HO55

  # 主分析 + 敏感性分析
  python compare_BD_HC.py --atlas HO55 --sensitivity 10

  # 只做组间比较 (用已有邻接矩阵, 跳过训练)
  python compare_BD_HC.py --atlas HO55 --adj_dir out/ucla_cnp/HO55/adj --skip_train

依赖:
    - torch, numpy, pandas, scipy, nilearn
    - 已提取的 ROI 时间序列: dataset/ucla_roi/{atlas}/{BD,HC}/
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedKFold

import torch
import torch.optim as optim

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
ROI_BASE = PROJECT_ROOT / 'dataset' / 'ucla_roi'
PARTICIPANTS_PATH = PROJECT_ROOT / 'dataset' / 'ucla_cnp' / 'participants.tsv'
PHENOTYPE_DIR = PROJECT_ROOT / 'dataset' / 'ucla_cnp' / 'phenotype'
OUT_BASE = PROJECT_ROOT / 'out' / 'ucla_cnp'

# FSTA 相关 import
sys.path.insert(0, str(PROJECT_ROOT))
from model.FSTA import FSTA
from model.Optim import ScheduledOptim
from utils.utils import get_free_device, loss_func, softThres, change01


# ============================================================
# 1. 数据加载
# ============================================================
def load_timeseries_and_phenotype(atlas='HO55', groups=('BD', 'HC')):
    """
    加载 ROI 时间序列、人口学数据和头动质控。

    Returns:
        data:       np.ndarray [S, T, N]  全部受试者的时间序列
        subject_ids: list of str           受试者 ID
        pheno:      pd.DataFrame           人口学 + 协变量
    """
    all_data = []
    all_subjects = []
    all_groups = []
    expected_shape = None
    skipped = []

    for grp in groups:
        data_dir = ROI_BASE / atlas / grp
        if not data_dir.exists():
            raise FileNotFoundError(f'数据目录不存在: {data_dir}')
        txt_files = sorted(data_dir.glob('sub-*.txt'))
        if not txt_files:
            raise ValueError(f'{data_dir} 中没有找到 .txt 文件')

        for f in txt_files:
            subject_id = f.stem.replace('sub-', '')
            # 读取时间序列, 跳过首行列名
            ts = np.loadtxt(f, skiprows=1, delimiter='\t')
            if expected_shape is None:
                expected_shape = ts.shape
            elif ts.shape != expected_shape:
                skipped.append(f'{grp}/{f.name} shape={ts.shape} expected={expected_shape}')
                continue
            all_data.append(ts)
            all_subjects.append(subject_id)
            all_groups.append(grp)

    if skipped:
        print(f'[警告] 跳过 {len(skipped)} 个形状不匹配的文件:')
        for s in skipped:
            print(f'  {s}')

    data = np.stack(all_data, axis=0)  # [S, T, N]
    print(f'[数据] 加载 {data.shape[0]} 人 × {data.shape[1]} 时间点 × {data.shape[2]} ROI')

    # 加载人口学
    if not PARTICIPANTS_PATH.exists():
        raise FileNotFoundError(f'人口学文件不存在: {PARTICIPANTS_PATH}')
    participants = pd.read_csv(PARTICIPANTS_PATH, sep='\t')
    participants['participant_id'] = participants['participant_id'].str.replace('sub-', '', regex=False)

    # 加载质控
    qc_path = ROI_BASE / atlas / 'subject_qc.tsv'
    if not qc_path.exists():
        raise FileNotFoundError(f'QC 文件不存在: {qc_path}')
    qc = pd.read_csv(qc_path, sep='\t')
    qc['subject'] = qc['subject'].astype(str)

    # 合并: subject_id → group + age + sex + scanner + mean_fd
    # 同时加载额外的 phenotype 数据
    extra = _load_extra_phenotype(all_subjects)

    records = []
    for sid, grp in zip(all_subjects, all_groups):
        pheno_row = participants[participants['participant_id'] == sid]
        qc_row = qc[qc['subject'] == sid]

        if pheno_row.empty:
            raise ValueError(f'受试者 {sid} 不在 participants.tsv 中')
        if qc_row.empty:
            raise ValueError(f'受试者 {sid} 不在 subject_qc.tsv 中')

        # 性别: M→0, F→1
        sex_str = str(pheno_row['gender'].values[0]).upper()
        sex = 1 if sex_str == 'F' else 0

        # 扫描仪: 编码为数值
        scanner = float(pheno_row['ScannerSerialNumber'].values[0])

        # 额外 phenotype
        ex = extra.get(sid, {})

        records.append({
            'subject_id': sid,
            'group': grp,
            'is_BD': 1 if grp == 'BD' else 0,
            'age': float(pheno_row['age'].values[0]),
            'sex': sex,
            'scanner': scanner,
            'mean_fd': float(qc_row['mean_fd'].values[0]),
            'school_yrs': ex.get('school_yrs', np.nan),
            'n_medications': ex.get('n_medications', 0),
            'ymrs_score': ex.get('ymrs_score', np.nan),
            'hamd_17': ex.get('hamd_17', np.nan),
        })

    pheno = pd.DataFrame(records)
    n_bd = (pheno['is_BD'] == 1).sum()
    n_hc = (pheno['is_BD'] == 0).sum()

    # 填充缺失的教育年限为中位数
    if pheno['school_yrs'].isna().any():
        median_edu = pheno['school_yrs'].median()
        pheno['school_yrs'].fillna(median_edu, inplace=True)

    print(f'[人口学] BD={n_bd}, HC={n_hc}, '
          f'年龄 {pheno["age"].mean():.1f}±{pheno["age"].std():.1f}, '
          f'女 {pheno["sex"].sum()}/{len(pheno)}, '
          f'教育 {pheno["school_yrs"].mean():.1f}±{pheno["school_yrs"].std():.1f} 年')
    print(f'[人口学] BD 用药: {(pheno[pheno["is_BD"]==1]["n_medications"]>0).sum()}/{n_bd} '
          f'(mean={pheno[pheno["is_BD"]==1]["n_medications"].mean():.1f}种)')
    print(f'[人口学] BD ymrs: {pheno[pheno["is_BD"]==1]["ymrs_score"].mean():.1f}±{pheno[pheno["is_BD"]==1]["ymrs_score"].std():.1f}, '
          f'hamd_17: {pheno[pheno["is_BD"]==1]["hamd_17"].mean():.1f}±{pheno[pheno["is_BD"]==1]["hamd_17"].std():.1f}')

    # 组间差异检验
    bd = pheno[pheno['is_BD'] == 1]
    hc = pheno[pheno['is_BD'] == 0]
    from scipy import stats
    for var, label in [('age', '年龄'), ('sex', '性别(F)'), ('school_yrs', '教育年限'),
                         ('mean_fd', 'mean FD'), ('n_medications', '用药种数')]:
        t, p = stats.ttest_ind(bd[var], hc[var], equal_var=False)
        sig = '*' if p < 0.05 else 'ns'
        print(f'[组间] {label}: BD={bd[var].mean():.2f}±{bd[var].std():.2f}, '
              f'HC={hc[var].mean():.2f}±{hc[var].std():.2f}, p={p:.4f} {sig}')

    return data, all_subjects, pheno


def _load_extra_phenotype(subject_ids):
    """
    加载额外 phenotype 数据: 教育年限、用药、YMRS、Hamilton。

    Returns:
        dict: {subject_id: {school_yrs, n_medications, ymrs_score, hamd_17}}
    """
    extra = {sid: {} for sid in subject_ids}

    # --- demographics.tsv: school_yrs ---
    demo_path = PHENOTYPE_DIR / 'demographics.tsv'
    if demo_path.exists():
        demo = pd.read_csv(demo_path, sep='\t')
        demo['participant_id'] = demo['participant_id'].str.replace('sub-', '', regex=False)
        for sid in subject_ids:
            row = demo[demo['participant_id'] == sid]
            if not row.empty:
                extra[sid]['school_yrs'] = pd.to_numeric(row['school_yrs'].values[0], errors='coerce')

    # --- medication.tsv: 统计用药种数 ---
    med_path = PHENOTYPE_DIR / 'medication.tsv'
    if med_path.exists():
        med = pd.read_csv(med_path, sep='\t')
        med['participant_id'] = med['participant_id'].str.replace('sub-', '', regex=False)
        med_name_cols = [c for c in med.columns if c.startswith('med_name')]
        for sid in subject_ids:
            row = med[med['participant_id'] == sid]
            if not row.empty:
                n = sum(1 for c in med_name_cols
                        if pd.notna(row[c].values[0])
                        and str(row[c].values[0]).strip() not in ('', 'n/a'))
                extra[sid]['n_medications'] = n
            else:
                extra[sid]['n_medications'] = 0

    # --- ymrs.tsv: ymrs_score (仅BD有数据) ---
    ymrs_path = PHENOTYPE_DIR / 'ymrs.tsv'
    if ymrs_path.exists():
        ymrs = pd.read_csv(ymrs_path, sep='\t')
        ymrs['participant_id'] = ymrs['participant_id'].str.replace('sub-', '', regex=False)
        for sid in subject_ids:
            row = ymrs[ymrs['participant_id'] == sid]
            if not row.empty and 'ymrs_score' in ymrs.columns:
                extra[sid]['ymrs_score'] = pd.to_numeric(row['ymrs_score'].values[0], errors='coerce')

    # --- hamilton.tsv: hamd_17 (仅BD有数据) ---
    ham_path = PHENOTYPE_DIR / 'hamilton.tsv'
    if ham_path.exists():
        ham = pd.read_csv(ham_path, sep='\t')
        ham['participant_id'] = ham['participant_id'].str.replace('sub-', '', regex=False)
        for sid in subject_ids:
            row = ham[ham['participant_id'] == sid]
            if not row.empty and 'hamd_17' in ham.columns:
                extra[sid]['hamd_17'] = pd.to_numeric(row['hamd_17'].values[0], errors='coerce')

    return extra


# ============================================================
# 2. FSTA 训练 & 推理
# ============================================================
def train_fsta_one_fold(train_data, test_data, opt, device):
    """
    在一个训练折上训练 FSTA, 并对测试集逐人推理。

    Args:
        train_data: torch.Tensor [n_train, T, N]
        test_data:  torch.Tensor [n_test,  T, N]
        opt:        模型超参数 argparse.Namespace
        device:     torch.device

    Returns:
        test_adjs:  np.ndarray [n_test, N, N]  测试集每人独立的邻接矩阵
    """
    S_train, T, N = train_data.shape
    opt.nodes_num = N
    opt.time_num = T

    # 构建模型
    model = FSTA(
        opt=opt,
        time_num=T,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    ).to(device)

    optimizer = ScheduledOptim(
        optim.Adam([{'params': model.parameters()}], betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps,
    )
    criterion = loss_func(alpha_sp=opt.alpha_sp).to(device)

    # 自监督: label 即数据本身
    train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    # 训练
    model.train()
    for epoch_i in range(opt.epoch):
        epoch_loss = []
        for data_batch, label_batch in train_loader:
            optimizer.zero_grad()
            output, adj = model(data_batch)
            loss = criterion(output, adj, label_batch)
            loss.backward()
            optimizer.step_and_update_lr()
            epoch_loss.append(loss.item())

        if (epoch_i + 1) % 50 == 0:
            print(f'    Epoch {epoch_i + 1}/{opt.epoch}, loss={np.mean(epoch_loss):.4f}')

        torch.cuda.empty_cache()

    # 推理: 逐人 forward, batch_size=1 保证每人独立 spa_attn
    model.eval()
    test_adjs = []
    with torch.no_grad():
        for i in range(len(test_data)):
            single = test_data[i:i+1]  # [1, T, N]
            _, spa_attn = model(single)
            # spa_attn: [N, N] — 该受试者的空间注意力矩阵 (即功能连接)
            adj_np = spa_attn.cpu().numpy()
            # 对称化 + 对角线置零
            adj_sym = (adj_np + adj_np.T) / 2.0
            np.fill_diagonal(adj_sym, 0)
            test_adjs.append(adj_sym)

    return np.stack(test_adjs, axis=0)  # [n_test, N, N]


def cross_validate_fsta(data, subject_ids, pheno, opt, device):
    """
    5折分层交叉验证: 每折训练 FSTA → 推理 → 收集所有人独立邻接矩阵。

    Returns:
        all_adjs:    np.ndarray [S, N, N]  所有受试者的邻接矩阵 (按 subject_ids 顺序)
        subject_ids: list
        pheno:       pd.DataFrame
    """
    S, T, N = data.shape
    labels = pheno['is_BD'].values  # 用于分层

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.seed)
    all_adjs = np.zeros((S, N, N), dtype=np.float32)

    data_tensor = torch.FloatTensor(data).to(device)

    fold_idx = 1
    for train_idx, test_idx in skf.split(np.arange(S), labels):
        n_train = len(train_idx)
        n_test = len(test_idx)
        print(f'\n[折 {fold_idx}/5] 训练 {n_train} 人, 测试 {n_test} 人')
        t0 = time.time()

        train_tensor = data_tensor[train_idx]
        test_tensor = data_tensor[test_idx]

        test_adjs = train_fsta_one_fold(train_tensor, test_tensor, opt, device)
        all_adjs[test_idx] = test_adjs

        elapsed = time.time() - t0
        print(f'  耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)')
        fold_idx += 1

    return all_adjs, subject_ids, pheno


# ============================================================
# 3. 组间比较: 逐边 GLM + FDR 校正
# ============================================================
def glm_per_edge(all_adjs, pheno):
    """
    对每条边 (i,j, i<j) 拟合 OLS:
        edge ~ is_BD + age + sex + scanner + mean_fd
    提取 is_BD 的系数和 p 值, 然后 FDR 校正。

    Returns:
        results: pd.DataFrame, 列:
            roi_i, roi_j, coef, t_stat, p_raw, p_fdr, significant
    """
    S, N, _ = all_adjs.shape
    print(f'\n[GLM] {S} 人 × {N} 节点, 共 {N*(N-1)//2} 条边')

    # 构建设计矩阵
    # edge ~ is_BD + age + sex + school_yrs + scanner + mean_fd
    # 注意: n_medications 与 is_BD 高度共线 (BD→用药, HC→不用药),
    # 不纳入主模型, 仅做描述统计; 可另做 BD 组内 medicated vs unmedicated 比较
    X = np.column_stack([
        np.ones(S),                      # 截距
        pheno['is_BD'].values,           # 核心预测变量 (BD vs HC)
        pheno['age'].values,             # 年龄
        pheno['sex'].values,             # 性别 (0=M, 1=F)
        pheno['school_yrs'].values,      # 教育年限
        pheno['scanner'].values,         # 扫描仪
        pheno['mean_fd'].values,         # 头动
    ])
    n_predictors = X.shape[1]

    # 预计算 (X'X)^(-1) 共用于所有边
    XtX_inv = np.linalg.inv(X.T @ X)
    df_resid = S - n_predictors

    results = []
    n_edges = N * (N - 1) // 2
    progress_interval = max(1, n_edges // 10)

    for idx, (i, j) in enumerate(zip(*np.triu_indices(N, k=1))):
        y = all_adjs[:, i, j]  # 边 (i,j) 在全部受试者上的权重

        # OLS
        beta = XtX_inv @ (X.T @ y)
        y_pred = X @ beta
        residuals = y - y_pred
        sigma2 = np.sum(residuals ** 2) / df_resid
        var_beta = sigma2 * XtX_inv
        se_beta = np.sqrt(np.diag(var_beta))

        # is_BD 的系数 (第1个预测变量, index=1)
        coef_bd = beta[1]
        t_stat_bd = coef_bd / se_beta[1]
        p_raw = 2.0 * scipy_stats.t.sf(abs(t_stat_bd), df_resid)

        results.append({
            'roi_i': i, 'roi_j': j,
            'coef': coef_bd,
            't_stat': t_stat_bd,
            'p_raw': p_raw,
        })

        if (idx + 1) % progress_interval == 0:
            print(f'  GLM 进度: {idx+1}/{n_edges} 边')

    df_results = pd.DataFrame(results)

    # ── Benjamini-Hochberg FDR 校正 ──
    p_values = df_results['p_raw'].values
    n_tests = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH 校正后的 p 值: p_fdr[k] = min(1, p[k] * m / k)
    # 然后强制单调递增: p_fdr[i] = min(p_fdr[i], p_fdr[i+1])
    p_fdr = np.ones(n_tests)
    for i in range(n_tests - 1, -1, -1):
        rank = i + 1  # 1-based rank
        p_fdr[sorted_idx[i]] = min(1.0, sorted_p[i] * n_tests / rank)
        if i < n_tests - 1:
            p_fdr[sorted_idx[i]] = min(p_fdr[sorted_idx[i]], p_fdr[sorted_idx[i + 1]])

    df_results['p_fdr'] = p_fdr

    # 用 p_fdr < 0.05 判定显著性
    df_results['significant'] = df_results['p_fdr'] < FDR_ALPHA

    # 找到 BH 阈值 (原始 p 值尺度)
    below_alpha = sorted_p <= (np.arange(1, n_tests + 1) / n_tests) * FDR_ALPHA
    if np.any(below_alpha):
        k_max = np.max(np.where(below_alpha)[0])
        p_raw_threshold = sorted_p[k_max]
    else:
        p_raw_threshold = 0.0

    n_sig = df_results['significant'].sum()
    n_bd_gt_hc = int((df_results.loc[df_results['significant'], 'coef'] > 0).sum())
    n_bd_lt_hc = int((df_results.loc[df_results['significant'], 'coef'] < 0).sum())
    print(f'\n[FDR] alpha={FDR_ALPHA}, 原始p阈值 {p_raw_threshold:.6f}, 显著边 {n_sig}/{n_tests}')
    print(f'[FDR] BD>HC: {n_bd_gt_hc}, BD<HC: {n_bd_lt_hc}')

    return df_results, p_raw_threshold


# ============================================================
# 4. 敏感性分析
# ============================================================
def sensitivity_analysis(all_adjs, pheno, main_df_results,
                          n_repeats=10, random_seed=42):
    """
    重复随机选择与 BD 等量的 HC, 在已有邻接矩阵上重跑 GLM + FDR,
    比较显著边与主分析的重叠率和效应方向一致性。

    复用主分析中已通过 FSTA 推理得到的每人独立邻接矩阵,
    仅重跑 GLM 部分, 速度快 (每次 ~10秒)。

    Args:
        all_adjs:          np.ndarray [S, N, N]  主分析得到的全部邻接矩阵
        pheno:             pd.DataFrame
        main_df_results:   主分析 GLM 结果
        n_repeats:         重复次数
        random_seed:       随机种子

    Returns:
        sensitivity_stats: list of dict
    """
    bd_mask = pheno['is_BD'].values == 1
    hc_mask = pheno['is_BD'].values == 0
    bd_idx = np.where(bd_mask)[0]
    hc_idx = np.where(hc_mask)[0]
    n_bd = len(bd_idx)
    print(f'\n[敏感性] 重复 {n_repeats} 次: 随机选 {n_bd}/{len(hc_idx)} HC + 全部 {n_bd} BD')
    print('  (复用主分析邻接矩阵, 仅重跑 GLM)')

    main_sig_edges = set()
    for _, row in main_df_results[main_df_results['significant']].iterrows():
        main_sig_edges.add((int(row['roi_i']), int(row['roi_j'])))

    sensitivity_stats = []
    rng = np.random.RandomState(random_seed)

    for rep in range(1, n_repeats + 1):
        t0 = time.time()

        # 随机抽取等量 HC
        selected_hc = rng.choice(hc_idx, size=n_bd, replace=False)
        selected_idx = np.sort(np.concatenate([bd_idx, selected_hc]))

        sub_adjs = all_adjs[selected_idx]
        sub_pheno = pheno.iloc[selected_idx].reset_index(drop=True)

        # GLM + FDR (复用邻接矩阵, 不重训)
        sub_results, sub_threshold = glm_per_edge(sub_adjs, sub_pheno)

        # 与主分析比较
        sub_sig = set()
        for _, row in sub_results[sub_results['significant']].iterrows():
            sub_sig.add((int(row['roi_i']), int(row['roi_j'])))

        overlap = main_sig_edges & sub_sig if main_sig_edges else set()
        jaccard = len(overlap) / len(main_sig_edges | sub_sig) if (main_sig_edges | sub_sig) else 0
        overlap_pct = len(overlap) / len(main_sig_edges) * 100 if main_sig_edges else 0

        # 效应方向一致性 (仅在重叠边中检查)
        coef_consistent = 0
        if overlap:
            for e in overlap:
                main_coef = main_df_results[
                    (main_df_results['roi_i'] == e[0]) & (main_df_results['roi_j'] == e[1])
                ]['coef'].values[0]
                sub_coef = sub_results[
                    (sub_results['roi_i'] == e[0]) & (sub_results['roi_j'] == e[1])
                ]['coef'].values[0]
                if main_coef * sub_coef > 0:
                    coef_consistent += 1

        stat = {
            'repeat': rep,
            'n_sig_main': len(main_sig_edges),
            'n_sig_sub': len(sub_sig),
            'n_overlap': len(overlap),
            'overlap_pct_of_main': round(overlap_pct, 2),
            'jaccard': round(jaccard, 4),
            'coef_consistent': coef_consistent,
            'p_raw_threshold': round(sub_threshold, 6),
        }
        sensitivity_stats.append(stat)

        elapsed = time.time() - t0
        print(f'  重复 {rep}/{n_repeats}: '
              f'主显著={len(main_sig_edges)}, 子显著={len(sub_sig)}, '
              f'重叠={len(overlap)}({overlap_pct:.1f}%), '
              f'方向一致={coef_consistent}/{len(overlap)}, '
              f'耗时 {elapsed:.1f}s')

    return sensitivity_stats


# ============================================================
# 辅助函数
# ============================================================
# FDR 显著性水平
FDR_ALPHA = 0.05


def save_results(df_results, p_threshold, output_dir, atlas, suffix=''):
    """保存 GLM 结果和显著边列表。"""
    os.makedirs(output_dir, exist_ok=True)

    # 完整结果
    csv_path = os.path.join(output_dir, f'glm_results{suffix}.csv')
    df_results.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'[保存] GLM 结果 → {csv_path}')

    # 显著边
    sig = df_results[df_results['significant']]
    sig_path = os.path.join(output_dir, f'significant_edges{suffix}.csv')
    sig.to_csv(sig_path, index=False, float_format='%.6f')
    print(f'[保存] {len(sig)} 条显著边 → {sig_path}')

    # 读取 ROI 标签
    labels_path = ROI_BASE / atlas / 'roi_labels.tsv'
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path, sep='\t')
        label_map = dict(zip(labels_df['label_id'], labels_df['roi_name']))

        sig_named = sig.copy()
        sig_named['roi_i_name'] = sig_named['roi_i'].apply(lambda x: label_map.get(x + 1, '?'))
        sig_named['roi_j_name'] = sig_named['roi_j'].apply(lambda x: label_map.get(x + 1, '?'))
        named_path = os.path.join(output_dir, f'significant_edges_named{suffix}.csv')
        sig_named.to_csv(named_path, index=False, float_format='%.6f')
        print(f'[保存] 显著边 (含名称) → {named_path}')


def save_adj_matrices(all_adjs, subject_ids, output_dir):
    """保存每人独立的邻接矩阵。"""
    adj_dir = os.path.join(output_dir, 'adj')
    os.makedirs(adj_dir, exist_ok=True)
    for i, sid in enumerate(subject_ids):
        np.savetxt(os.path.join(adj_dir, f'sub-{sid}.txt'),
                   all_adjs[i], fmt='%.6f', delimiter='\t')
    print(f'[保存] {len(subject_ids)} 个邻接矩阵 → {adj_dir}/')


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='BD vs HC 脑功能连接网络组间比较 (HO55/HO110)'
    )

    # 数据和模式
    parser.add_argument('--atlas', type=str, default='HO55',
                        choices=['HO55', 'HO110'],
                        help='图谱 (默认 HO55)')
    parser.add_argument('--skip_train', action='store_true',
                        help='跳过训练, 直接使用已有邻接矩阵做 GLM')
    parser.add_argument('--adj_dir', type=str, default=None,
                        help='已有邻接矩阵目录 (配合 --skip_train)')

    # 敏感性分析
    parser.add_argument('--sensitivity', type=int, default=0,
                        help='敏感性分析重复次数 (默认 0=不做)')

    # FSTA 模型参数
    parser.add_argument('--epoch', type=int, default=50,
                        help='每折训练 epoch 数 (默认 50)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_inner_hid', type=int, default=64)
    parser.add_argument('--d_k', type=int, default=8)
    parser.add_argument('--d_v', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--soft_threshold', type=float, default=0.5)
    parser.add_argument('--alpha_sp', type=float, default=0.8)
    parser.add_argument('--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--lr_mul', type=float, default=1.2)

    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default='auto')
    parser.add_argument('--out_dir', type=str, default=None)

    # 兼容参数
    parser.add_argument('--variance', type=float, default=5)
    parser.add_argument('--no_filters', action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--model_name', default='FMLPRec', type=str)
    parser.add_argument('--num_hidden_layers', default=1, type=int)
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default='gelu', type=str)
    parser.add_argument('--attention_probs_dropout_prob', default=0.5, type=float)
    parser.add_argument('--hidden_dropout_prob', default=0.5, type=float)
    parser.add_argument('--initializer_range', default=0.02, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--adam_beta1', default=0.9, type=float)
    parser.add_argument('--adam_beta2', default=0.999, type=float)

    opt = parser.parse_args()
    opt.d_word_vec = opt.d_model
    return opt


# ============================================================
# 主入口
# ============================================================
def main():
    opt = parse_args()
    device = get_free_device(opt.gpu_id)

    output_dir = opt.out_dir or str(OUT_BASE / opt.atlas)
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print(f'  BD vs HC 组间比较 — {opt.atlas} 图谱')
    print(f'  输出目录: {output_dir}')
    if opt.sensitivity > 0:
        print(f'  敏感性分析: {opt.sensitivity} 次重复')
    print('=' * 60)

    # ── 加载数据 ──
    data, subject_ids, pheno = load_timeseries_and_phenotype(atlas=opt.atlas)
    S, T, N = data.shape
    print(f'[数据] shape: {data.shape}')

    # ── FSTA 交叉验证 → 每人独立邻接矩阵 ──
    if opt.skip_train and opt.adj_dir:
        print(f'\n[跳过训练] 从 {opt.adj_dir} 加载已有邻接矩阵')
        adj_dir_path = Path(opt.adj_dir)
        all_adjs = np.zeros((S, N, N), dtype=np.float32)
        for i, sid in enumerate(subject_ids):
            adj_file = adj_dir_path / f'sub-{sid}.txt'
            all_adjs[i] = np.loadtxt(adj_file, delimiter='\t')
    else:
        print(f'\n[FSTA 交叉验证] 5折 × {opt.epoch} epochs, d_model={opt.d_model}')
        t0 = time.time()
        all_adjs, subject_ids, pheno = cross_validate_fsta(data, subject_ids, pheno, opt, device)
        elapsed = time.time() - t0
        print(f'\n[FSTA 交叉验证完成] 总耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)')

        # 保存邻接矩阵
        save_adj_matrices(all_adjs, subject_ids, output_dir)

    # ── 逐边 GLM + FDR ──
    df_results, p_threshold = glm_per_edge(all_adjs, pheno)
    save_results(df_results, p_threshold, output_dir, opt.atlas)

    # ── 敏感性分析 ──
    if opt.sensitivity > 0:
        sens_stats = sensitivity_analysis(
            all_adjs, pheno, df_results,
            n_repeats=opt.sensitivity, random_seed=opt.seed,
        )

        sens_df = pd.DataFrame(sens_stats)
        sens_path = os.path.join(output_dir, 'sensitivity_stats.csv')
        sens_df.to_csv(sens_path, index=False, float_format='%.4f')
        print(f'\n[保存] 敏感性统计 → {sens_path}')

        # 汇总
        print(f'\n{"=" * 60}')
        print('敏感性分析汇总')
        print(f'{"=" * 60}')
        print(f'平均重叠率: {sens_df["overlap_pct_of_main"].mean():.1f}% ± {sens_df["overlap_pct_of_main"].std():.1f}%')
        print(f'平均 Jaccard: {sens_df["jaccard"].mean():.3f} ± {sens_df["jaccard"].std():.3f}')
        print(f'平均方向一致性: {sens_df["coef_consistent"].mean():.1f} ± {sens_df["coef_consistent"].std():.1f}')

    print(f'\n{"=" * 60}')
    print('完成!')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
