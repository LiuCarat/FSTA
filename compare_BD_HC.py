#!/usr/bin/env python3
"""
BD vs HC 有向脑功能连接网络组间比较
────────────────────────────────────────────────────────
流水线:
  1. 数据加载 + 严格QC (形状/NaN/重复ID/ROI变化)
  2. 多随机种子 5折交叉验证训练 FSTA → 每人独立有向 N×N 邻接矩阵
  3. 逐边 GLM + HC3稳健标准误 + FDR校正 (2970条有向边)
  4. 两种敏感性分析: 统计子抽样 + 完整重训

GLM 模型:
  edge ~ is_BD + age + sex + school_yrs + scanner(分类) + mean_fd + fold(分类)

关键改动 (v2.0):
  - 邻接矩阵保留有向 (无需对称化)
  - 全部 2970 条有向边 (55×54, 非上三角)
  - scanner/fold 作为分类变量
  - HC3 稳健标准误
  - 多随机种子平均
  - 区分统计子抽样 vs 完整重训敏感性分析

用法:
  python compare_BD_HC.py --atlas HO55                    # 主分析 (3种子)
  python compare_BD_HC.py --atlas HO55 --n_seeds 5        # 5种子
  python compare_BD_HC.py --atlas HO55 --sensitivity_glm 50   # GLM子抽样
  python compare_BD_HC.py --atlas HO55 --sensitivity_full 10  # 完整重训

依赖:
    - torch, numpy, pandas, scipy, statsmodels, sklearn
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
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

sys.path.insert(0, str(PROJECT_ROOT))
from model.FSTA import FSTA
from model.Optim import ScheduledOptim
from utils.utils import get_free_device, loss_func

# 全局常量
FDR_ALPHA = 0.05
N_FOLDS = 5
EXPECTED_T = 152
EXPECTED_N = 55


# ============================================================
# 随机种子设置
# ============================================================
def set_all_seeds(seed):
    """同时设置 Python random, NumPy, PyTorch (CPU + CUDA) 种子。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 1. 数据加载 + QC
# ============================================================
def load_and_qc(atlas='HO55', groups=('BD', 'HC')):
    """加载 ROI 时间序列 + 人口学 + QC, 执行严格的输入检查。"""
    # ── 加载时间序列 ──
    all_data, all_subjects, all_groups = [], [], []

    for grp in groups:
        data_dir = ROI_BASE / atlas / grp
        if not data_dir.exists():
            raise FileNotFoundError(f'数据目录不存在: {data_dir}')
        txt_files = sorted(data_dir.glob('sub-*.txt'))
        for f in txt_files:
            sid = f.stem.replace('sub-', '')
            ts = np.loadtxt(f, skiprows=1, delimiter='\t')
            # 硬编码预期形状 (检查清单 #3)
            if ts.shape[0] != EXPECTED_T or ts.shape[1] != EXPECTED_N:
                raise ValueError(f'{grp}/{f.name} shape={ts.shape}, 预期 ({EXPECTED_T}, {EXPECTED_N})')
            # 检查 NaN/Inf (检查清单 #4)
            if not np.isfinite(ts).all():
                raise ValueError(f'{grp}/{f.name} 包含 NaN 或 Inf')
            # 检查每 ROI 有无变化 (检查清单 #5)
            stds = ts.std(axis=0)
            if (stds < 1e-8).any():
                bad_cols = np.where(stds < 1e-8)[0].tolist()
                raise ValueError(f'{grp}/{f.name} ROI {bad_cols} std≈0')
            all_data.append(ts)
            all_subjects.append(sid)
            all_groups.append(grp)

    # 检查重复 ID (检查清单 #6)
    if len(all_subjects) != len(set(all_subjects)):
        from collections import Counter
        dupes = [k for k, v in Counter(all_subjects).items() if v > 1]
        raise ValueError(f'重复受试者 ID: {dupes}')

    data = np.stack(all_data, axis=0)
    S, T, N = data.shape
    print(f'[QC] ✓ 加载 {S} 人 × {T} 时间点 × {N} ROI, 无NaN/Inf, 形状一致')

    # ── 加载人口学 ──
    participants = pd.read_csv(PARTICIPANTS_PATH, sep='\t')
    participants['participant_id'] = participants['participant_id'].str.replace('sub-', '', regex=False)

    # ── 加载 QC ──
    qc_df = pd.read_csv(ROI_BASE / atlas / 'subject_qc.tsv', sep='\t')
    qc_df['subject'] = qc_df['subject'].astype(str)

    # ── 加载额外 phenotype ──
    extra = _load_extra_phenotype(all_subjects)

    # ── 组装 pheno ──
    records = []
    for sid, grp in zip(all_subjects, all_groups):
        prow = participants[participants['participant_id'] == sid]
        qrow = qc_df[qc_df['subject'] == sid]
        if prow.empty:
            raise ValueError(f'sub-{sid} 不在 participants.tsv')
        if qrow.empty:
            raise ValueError(f'sub-{sid} 不在 subject_qc.tsv')

        sex_str = str(prow['gender'].values[0]).upper()
        scanner_val = str(int(prow['ScannerSerialNumber'].values[0]))
        ex = extra.get(sid, {})

        records.append({
            'subject_id': sid,
            'group': grp,
            'is_BD': 1 if grp == 'BD' else 0,
            'age': float(prow['age'].values[0]),
            'sex': 1 if sex_str == 'F' else 0,
            'scanner': scanner_val,               # 分类变量, 字符串
            'mean_fd': float(qrow['mean_fd'].values[0]),
            'school_yrs': ex.get('school_yrs', np.nan),
            'n_medications': ex.get('n_medications', 0),
            'ymrs_score': ex.get('ymrs_score', np.nan),
            'hamd_17': ex.get('hamd_17', np.nan),
        })

    pheno = pd.DataFrame(records)

    # 缺失教育年限: 填充中位数 + 记录 (检查清单 #74)
    edu_missing = pheno['school_yrs'].isna().sum()
    if edu_missing > 0:
        median_edu = pheno['school_yrs'].median()
        print(f'[QC] 教育年限缺失 {edu_missing}人, 填充中位数 {median_edu:.0f}')
        pheno['school_yrs'].fillna(median_edu, inplace=True)

    # ── 描述统计 ──
    n_bd = (pheno['is_BD'] == 1).sum()
    n_hc = (pheno['is_BD'] == 0).sum()
    bd = pheno[pheno['is_BD'] == 1]
    hc = pheno[pheno['is_BD'] == 0]

    print(f'\n[人口学] BD={n_bd}, HC={n_hc}')
    # 年龄: t检验
    t_a, p_a = scipy_stats.ttest_ind(bd['age'], hc['age'], equal_var=False)
    print(f'  年龄: BD={bd["age"].mean():.1f}±{bd["age"].std():.1f}, '
          f'HC={hc["age"].mean():.1f}±{hc["age"].std():.1f}, t={t_a:.2f}, p={p_a:.4f}')
    # 性别: 卡方检验 (检查清单 #84)
    sex_table = pd.crosstab(pheno['is_BD'], pheno['sex'])
    chi2_s, p_s, _, _ = scipy_stats.chi2_contingency(sex_table)
    print(f'  性别(F): BD={(bd["sex"]==1).sum()}/{n_bd}, HC={(hc["sex"]==1).sum()}/{n_hc}, '
          f'χ²={chi2_s:.2f}, p={p_s:.4f}')
    # 教育: t检验
    t_e, p_e = scipy_stats.ttest_ind(bd['school_yrs'], hc['school_yrs'], equal_var=False)
    print(f'  教育: BD={bd["school_yrs"].mean():.1f}±{bd["school_yrs"].std():.1f}, '
          f'HC={hc["school_yrs"].mean():.1f}±{hc["school_yrs"].std():.1f}, p={p_e:.4f}')
    # scanner: 卡方检验 (检查清单 #85)
    scanner_table = pd.crosstab(pheno['is_BD'], pheno['scanner'])
    chi2_sc, p_sc, _, _ = scipy_stats.chi2_contingency(scanner_table)
    print(f'  scanner分布: χ²={chi2_sc:.2f}, p={p_sc:.4f}')
    print(f'  {scanner_table.to_string()}')
    # mean_fd
    t_f, p_f = scipy_stats.ttest_ind(bd['mean_fd'], hc['mean_fd'], equal_var=False)
    print(f'  mean FD: BD={bd["mean_fd"].mean():.3f}±{bd["mean_fd"].std():.3f}, '
          f'HC={hc["mean_fd"].mean():.3f}±{hc["mean_fd"].std():.3f}, p={p_f:.4f}')
    # 用药 (仅描述, 检查清单 #88-89)
    print(f'  BD用药: {(bd["n_medications"]>0).sum()}/{n_bd} 人, 平均 {bd["n_medications"].mean():.1f} 种')
    if 'ymrs_score' in pheno.columns:
        print(f'  BD YMRS: {bd["ymrs_score"].mean():.1f}±{bd["ymrs_score"].std():.1f}')
    if 'hamd_17' in pheno.columns:
        print(f'  BD HAMD-17: {bd["hamd_17"].mean():.1f}±{bd["hamd_17"].std():.1f}')

    return data, all_subjects, pheno


def _load_extra_phenotype(subject_ids):
    """加载 phenotype/ 下的额外数据。"""
    extra = {sid: {} for sid in subject_ids}
    id_set = set(subject_ids)

    # demographics → school_yrs
    fp = PHENOTYPE_DIR / 'demographics.tsv'
    if fp.exists():
        df = pd.read_csv(fp, sep='\t')
        df['participant_id'] = df['participant_id'].str.replace('sub-', '', regex=False)
        for _, r in df.iterrows():
            sid = r['participant_id']
            if sid in id_set:
                extra[sid]['school_yrs'] = pd.to_numeric(r['school_yrs'], errors='coerce')

    # medication → n_medications
    fp = PHENOTYPE_DIR / 'medication.tsv'
    if fp.exists():
        df = pd.read_csv(fp, sep='\t')
        df['participant_id'] = df['participant_id'].str.replace('sub-', '', regex=False)
        med_cols = [c for c in df.columns if c.startswith('med_name')]
        for _, r in df.iterrows():
            sid = r['participant_id']
            if sid in id_set:
                n = sum(1 for c in med_cols
                        if pd.notna(r[c]) and str(r[c]).strip() not in ('', 'n/a'))
                extra[sid]['n_medications'] = n

    # ymrs
    fp = PHENOTYPE_DIR / 'ymrs.tsv'
    if fp.exists():
        df = pd.read_csv(fp, sep='\t')
        df['participant_id'] = df['participant_id'].str.replace('sub-', '', regex=False)
        for _, r in df.iterrows():
            sid = r['participant_id']
            if sid in id_set and 'ymrs_score' in df.columns:
                extra[sid]['ymrs_score'] = pd.to_numeric(r['ymrs_score'], errors='coerce')

    # hamilton
    fp = PHENOTYPE_DIR / 'hamilton.tsv'
    if fp.exists():
        df = pd.read_csv(fp, sep='\t')
        df['participant_id'] = df['participant_id'].str.replace('sub-', '', regex=False)
        for _, r in df.iterrows():
            sid = r['participant_id']
            if sid in id_set and 'hamd_17' in df.columns:
                extra[sid]['hamd_17'] = pd.to_numeric(r['hamd_17'], errors='coerce')

    return extra


# ============================================================
# 2. FSTA 训练 & 推理 (单种子 × 单折)
# ============================================================
def train_fsta_one_fold(train_data, test_data, opt, device):
    """训练一折 → batch_size=1推理 → 返回有向邻接矩阵 [n_test, N, N]"""
    _, T, N = train_data.shape
    opt.nodes_num = N
    opt.time_num = T

    model = FSTA(
        opt=opt, time_num=T,
        d_model=opt.d_model, d_inner=opt.d_inner_hid,
        n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
        dropout=opt.dropout,
    ).to(device)

    optimizer = ScheduledOptim(
        optim.Adam([{'params': model.parameters()}], betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps,
    )
    criterion = loss_func(alpha_sp=opt.alpha_sp).to(device)

    train_ds = torch.utils.data.TensorDataset(train_data, train_data)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True)

    # 训练
    model.train()
    losses = []
    for epoch_i in range(opt.epoch):
        epoch_loss = []
        for db, lb in train_loader:
            optimizer.zero_grad()
            outp, adj = model(db)
            loss = criterion(outp, adj, lb)
            loss.backward()
            optimizer.step_and_update_lr()
            epoch_loss.append(loss.item())
        avg_loss = np.mean(epoch_loss)
        losses.append(avg_loss)
        if (epoch_i + 1) % max(10, opt.epoch // 5) == 0:
            print(f'    Epoch {epoch_i+1}/{opt.epoch}, loss={avg_loss:.4f}')
        # 检查清单 #41: 不需要每个epoch empty_cache

    # 检查损失是否收敛 (检查清单 #37)
    loss_first = np.mean(losses[:5])
    loss_last = np.mean(losses[-5:])
    if loss_last > loss_first * 0.95 and loss_first > 0:
        print(f'    [警告] loss 未明显下降: {loss_first:.4f} → {loss_last:.4f}')

    # 推理: batch_size=1, 保留有向 (检查清单 #46-48)
    model.eval()
    test_adjs = []
    with torch.no_grad():
        for i in range(len(test_data)):
            single = test_data[i:i+1]
            _, spa_attn = model(single)
            # spa_attn[i,j] = 有向连接 ROI i → ROI j (检查清单 #46-47)
            adj_np = spa_attn.cpu().numpy()
            # 对角线置零 (检查清单 #50)
            # 不做对称化! (检查清单 #48)
            np.fill_diagonal(adj_np, 0)
            test_adjs.append(adj_np)

    return np.stack(test_adjs, axis=0)  # [n_test, N, N]


# ============================================================
# 3. 单种子 5折交叉验证
# ============================================================
def cross_validate_one_seed(data, pheno, opt, device, seed):
    """单种子 5折CV → {subject_id: adj(N,N), fold_idx}"""
    set_all_seeds(seed)
    S, T, N = data.shape
    labels = pheno['is_BD'].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_assignments = np.full(S, -1, dtype=int)

    data_tensor = torch.FloatTensor(data).to(device)

    all_adjs = {}     # subject_id → adj
    fold_of = {}      # subject_id → fold_idx

    fid = 1
    for train_idx, test_idx in skf.split(np.arange(S), labels):
        fold_assignments[test_idx] = fid
        n_tr, n_te = len(train_idx), len(test_idx)
        fold_seed = seed + fid  # 检查清单 #28
        set_all_seeds(fold_seed)

        print(f'\n  [折 {fid}/5] 训练 {n_tr}人, 测试 {n_te}人  (seed={seed}, fold_seed={fold_seed})')
        t0 = time.time()

        train_t = data_tensor[train_idx]
        test_t = data_tensor[test_idx]
        test_adjs = train_fsta_one_fold(train_t, test_t, opt, device)

        sub_ids_test = pheno.iloc[test_idx]['subject_id'].values
        for j, sid in enumerate(sub_ids_test):
            all_adjs[sid] = test_adjs[j]
            fold_of[sid] = fid

        elapsed = time.time() - t0
        print(f'    耗时 {elapsed:.0f}s')

        # 检查: 每折 BD/HC 比例 (检查清单 #23)
        y_tr = labels[train_idx]
        y_te = labels[test_idx]
        print(f'    BD比例 — 训练: {y_tr.mean():.2f}, 测试: {y_te.mean():.2f}')

        fid += 1

    # 检查清单 #19: 每个受试者只在一个测试折中出现
    assert len(all_adjs) == S, f'邻接矩阵数 {len(all_adjs)} ≠ 受试者数 {S}'

    return all_adjs, fold_of, fold_assignments


# ============================================================
# 4. 多种子交叉验证 + 平均
# ============================================================
def multi_seed_cv(data, pheno, opt, device, base_seed, n_seeds):
    """多个初始化种子独立进行5折CV, 对每人取平均邻接矩阵。"""
    all_seed_adjs = {}  # sid → [adj1, adj2, ...]
    all_fold_assignments = []

    for si in range(n_seeds):
        seed = base_seed + si * 100
        print(f'\n{"="*50}')
        print(f'  Seed {si+1}/{n_seeds} (seed={seed})')
        print(f'{"="*50}')

        adjs, fold_of, fold_arr = cross_validate_one_seed(data, pheno, opt, device, seed)
        all_fold_assignments.append(fold_arr)

        for sid, adj in adjs.items():
            if sid not in all_seed_adjs:
                all_seed_adjs[sid] = []
            all_seed_adjs[sid].append(adj)

    # 对每人, 多个种子的邻接矩阵取平均 (检查清单 #31)
    S, N = data.shape[0], data.shape[2]
    avg_adjs = np.zeros((S, N, N), dtype=np.float32)
    pheno_sorted = pheno.set_index('subject_id')

    for i, sid in enumerate(pheno['subject_id'].values):
        avg_adjs[i] = np.mean(all_seed_adjs[sid], axis=0)
        # 检查 NaN/Inf
        assert np.isfinite(avg_adjs[i]).all(), f'sub-{sid} 平均邻接矩阵含NaN/Inf'
        # 检查不同受试者不是完全相同的
        if i > 0:
            diff = np.max(np.abs(avg_adjs[i] - avg_adjs[i-1]))
            if diff < 1e-10:
                print(f'[警告] sub-{sid} 与前一受试者邻接矩阵完全相同!')

    # fold 分配: 用第一次运行的
    fold_assignments = all_fold_assignments[0]

    return avg_adjs, fold_assignments, all_seed_adjs


# ============================================================
# 5. 逐边 GLM + HC3 稳健标准误 + FDR
# ============================================================
def build_design_matrix(pheno, fold_arr):
    """构建设计矩阵 (检查清单 #63-68)

    模型: edge ~ is_BD + age + sex + school_yrs + scanner(分类) + mean_fd + fold(分类)

    Returns:
        X:       np.ndarray [S, k]  设计矩阵 (含截距)
        X_cols:  list of str        列名
        df_resid: int              残差自由度
    """
    S = len(pheno)

    # 手动创建虚拟变量 (检查清单 #65-67)
    scanner_cats = sorted(pheno['scanner'].unique())
    # scanner: 用较大的一类作为参考
    scanner_counts = pheno['scanner'].value_counts()
    scanner_ref = scanner_counts.index[0]

    fold_dummies = pd.get_dummies(
        [f'f{f}' for f in fold_arr], prefix='fold', drop_first=False
    )
    # fold f5 作为参考
    fold_cols = [c for c in sorted(fold_dummies.columns) if c != 'fold_f5']

    # 组装
    X_dict = {
        'intercept': np.ones(S),
        'is_BD': pheno['is_BD'].values.astype(float),
        'age': pheno['age'].values.astype(float),
        'sex': pheno['sex'].values.astype(float),
        'school_yrs': pheno['school_yrs'].values.astype(float),
        'mean_fd': pheno['mean_fd'].values.astype(float),
    }

    # scanner dummy (检查清单 #65)
    for cat in scanner_cats:
        if cat != scanner_ref:
            X_dict[f'scanner_{cat}'] = (pheno['scanner'].values == cat).astype(float)

    # fold dummy (检查清单 #66)
    for col in fold_cols:
        X_dict[col] = fold_dummies[col].values.astype(float)

    col_order = list(X_dict.keys())
    X = np.column_stack([X_dict[c] for c in col_order])

    df_resid = S - X.shape[1]

    return X, col_order, df_resid


def glm_per_edge(all_adjs, pheno, fold_arr):
    """逐边 GLM + HC3 SE + BH-FDR (2970条有向边, 检查清单 #58-82)"""
    import statsmodels.api as sm

    S, N, _ = all_adjs.shape
    n_edges = N * (N - 1)  # 全部有向非对角边
    print(f'\n[GLM] {S}人 × {N}节点, 共 {n_edges} 条有向边')

    X, X_cols, df_resid = build_design_matrix(pheno, fold_arr)
    is_bd_idx = X_cols.index('is_BD')
    n_predictors = X.shape[1]
    print(f'  设计矩阵: {n_predictors}个预测变量, df_resid={df_resid}')
    print(f'  列: {X_cols}')

    # 预计算 X'X 逆矩阵和帽子矩阵对角线 (HC3用)
    XtX_inv = np.linalg.inv(X.T @ X)
    H_diag = np.sum(X * (X @ XtX_inv), axis=1)  # 帽子矩阵对角线 (leverage)

    results = []
    progress_interval = max(1, n_edges // 10)

    for idx in range(n_edges):
        src = idx // (N - 1)
        dst = idx % (N - 1)
        if dst >= src:
            dst += 1  # 跳过对角线

        y = all_adjs[:, src, dst]

        try:
            # OLS
            beta = XtX_inv @ (X.T @ y)
            y_pred = X @ beta
            residuals = y - y_pred

            # HC3 robust SE (检查清单 #69)
            # HC3 = (X'X)^(-1) X' diag(e_i^2 / (1-h_ii)^2) X (X'X)^(-1)
            hc3_weights = (residuals ** 2) / ((1.0 - H_diag) ** 2)
            hc3_weights = np.clip(hc3_weights, 0, 1e10)  # 防止极端值
            X_weighted = X * hc3_weights[:, np.newaxis]
            vcov_hc3 = XtX_inv @ (X.T @ X_weighted) @ XtX_inv

            coef_bd = beta[is_bd_idx]
            se_hc3 = np.sqrt(max(vcov_hc3[is_bd_idx, is_bd_idx], 1e-20))
            t_stat = coef_bd / se_hc3 if se_hc3 > 1e-12 else 0.0
            p_raw = 2.0 * scipy_stats.t.sf(abs(t_stat), df_resid)

            # 置信区间 (检查清单 #70)
            t_crit = scipy_stats.t.ppf(0.975, df_resid)
            ci_low = coef_bd - t_crit * se_hc3
            ci_high = coef_bd + t_crit * se_hc3

            results.append({
                'source_roi': src,
                'target_roi': dst,
                'coef': coef_bd,
                'se_hc3': se_hc3,
                't_stat': t_stat,
                'p_raw': p_raw,
                'ci_low': ci_low,
                'ci_high': ci_high,
            })
        except Exception:
            results.append({
                'source_roi': src, 'target_roi': dst,
                'coef': np.nan, 'se_hc3': np.nan, 't_stat': np.nan,
                'p_raw': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
            })

        if (idx + 1) % progress_interval == 0:
            print(f'  GLM+HC3 进度: {idx+1}/{n_edges} 边')

    df = pd.DataFrame(results)

    # ── BH-FDR (检查清单 #76-77: 对全部2970统一校正) ──
    p = df['p_raw'].values.copy()
    nan_mask = np.isnan(p)
    p_valid = p[~nan_mask]
    n_valid = len(p_valid)
    sorted_idx = np.argsort(p_valid)
    sorted_p = p_valid[sorted_idx]

    p_fdr = np.ones(n_valid)
    for i in range(n_valid - 1, -1, -1):
        rank = i + 1
        p_fdr[sorted_idx[i]] = min(1.0, sorted_p[i] * n_valid / rank)
        if i < n_valid - 1:
            p_fdr[sorted_idx[i]] = min(p_fdr[sorted_idx[i]], p_fdr[sorted_idx[i+1]])

    df['p_fdr'] = np.nan
    df.loc[~nan_mask, 'p_fdr'] = p_fdr

    below = sorted_p <= (np.arange(1, n_valid+1) / n_valid) * FDR_ALPHA
    p_threshold = sorted_p[np.max(np.where(below)[0])] if np.any(below) else 0.0
    df['significant'] = df['p_fdr'] < FDR_ALPHA

    n_sig = df['significant'].sum()
    n_pos = int((df.loc[df['significant'], 'coef'] > 0).sum())
    n_neg = int((df.loc[df['significant'], 'coef'] < 0).sum())
    print(f'\n[FDR] α={FDR_ALPHA}, p阈值={p_threshold:.6f}, 显著边 {n_sig}/{n_valid} (BD>HC: {n_pos}, BD<HC: {n_neg})')

    return df, p_threshold


# ============================================================
# 6. 敏感性分析
# ============================================================
def sensitivity_glm_only(all_adjs, pheno, fold_arr, main_results, n_repeats, seed):
    """A. 统计阶段子抽样: 复用邻接矩阵, 随机等量 HC, 仅重跑 GLM (检查清单 #91-98)"""
    bd_idx = np.where(pheno['is_BD'].values == 1)[0]
    hc_idx = np.where(pheno['is_BD'].values == 0)[0]
    n_bd = len(bd_idx)
    print(f'\n[敏感性-GLM] {n_repeats}次子抽样: {n_bd}BD + {n_bd}/{len(hc_idx)}HC')
    print('  (复用主分析邻接矩阵, 不重训 FSTA)')

    main_sig = set()
    for _, r in main_results[main_results['significant']].iterrows():
        main_sig.add((int(r['source_roi']), int(r['target_roi'])))

    stats = []
    rng = np.random.RandomState(seed)
    for rep in range(1, n_repeats+1):
        t0 = time.time()
        selected_hc = rng.choice(hc_idx, size=n_bd, replace=False)
        selected_idx = np.sort(np.concatenate([bd_idx, selected_hc]))

        sub_adjs = all_adjs[selected_idx]
        sub_pheno = pheno.iloc[selected_idx].reset_index(drop=True)
        sub_fold = np.array([f'f{f}' for f in fold_arr[selected_idx]])

        # 重新5折分层 → 新的 fold 分配 (因为样本变了)
        sub_labels = sub_pheno['is_BD'].values
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed+rep)
        new_fold = np.zeros(len(sub_pheno), dtype=int)
        fid = 1
        for _, te_idx in skf.split(np.arange(len(sub_pheno)), sub_labels):
            new_fold[te_idx] = fid
            fid += 1

        sub_results, sub_thr = glm_per_edge(sub_adjs, sub_pheno, new_fold)

        sub_sig = set()
        for _, r in sub_results[sub_results['significant']].iterrows():
            sub_sig.add((int(r['source_roi']), int(r['target_roi'])))

        overlap = main_sig & sub_sig
        jaccard = len(overlap)/len(main_sig|sub_sig) if (main_sig|sub_sig) else 0
        overlap_pct = len(overlap)/len(main_sig)*100 if main_sig else 0

        coef_consistent = 0
        for e in overlap:
            mc = main_results[(main_results['source_roi']==e[0])&(main_results['target_roi']==e[1])]['coef'].values[0]
            sc = sub_results[(sub_results['source_roi']==e[0])&(sub_results['target_roi']==e[1])]['coef'].values[0]
            if mc * sc > 0:
                coef_consistent += 1

        stats.append({
            'repeat': rep, 'n_sig_main': len(main_sig), 'n_sig_sub': len(sub_sig),
            'n_overlap': len(overlap), 'overlap_pct': round(overlap_pct,1),
            'jaccard': round(jaccard,4), 'coef_consistent': coef_consistent,
            'p_threshold': round(sub_thr,6),
        })
        print(f'  {rep}/{n_repeats}: 重叠={len(overlap)}({overlap_pct:.0f}%), 方向一致={coef_consistent}/{len(overlap)}, {time.time()-t0:.0f}s')

    return stats


def sensitivity_full_retrain(data, pheno, opt, device, main_results, n_repeats, base_seed):
    """B. 完整重训: 重做 5折CV FSTA → GLM (检查清单 #99-108)"""
    bd_idx = np.where(pheno['is_BD'].values == 1)[0]
    hc_idx = np.where(pheno['is_BD'].values == 0)[0]
    n_bd = len(bd_idx)
    n_hc = len(hc_idx)
    print(f'\n[敏感性-FULL] {n_repeats}次完整重训: {n_bd}BD + {n_bd}/{n_hc}HC')

    main_sig = set()
    for _, r in main_results[main_results['significant']].iterrows():
        main_sig.add((int(r['source_roi']), int(r['target_roi'])))

    stats = []
    rng = np.random.RandomState(base_seed+1000)
    all_hc_lists = {}  # 保存每次的HC名单 (检查清单 #106)

    for rep in range(1, n_repeats+1):
        print(f'\n  --- 完整重训 {rep}/{n_repeats} ---')
        t0 = time.time()

        selected_hc = rng.choice(hc_idx, size=n_bd, replace=False)
        selected_idx = np.sort(np.concatenate([bd_idx, selected_hc]))
        all_hc_lists[rep] = selected_hc.tolist()

        sub_data = data[selected_idx]
        sub_pheno = pheno.iloc[selected_idx].reset_index(drop=True)

        rep_seed = base_seed + rep * 10
        sub_adjs, fold_arr, _ = multi_seed_cv(sub_data, sub_pheno, opt, device, rep_seed, opt.n_seeds)

        sub_results, sub_thr = glm_per_edge(sub_adjs, sub_pheno, fold_arr)

        sub_sig = set()
        for _, r in sub_results[sub_results['significant']].iterrows():
            sub_sig.add((int(r['source_roi']), int(r['target_roi'])))

        overlap = main_sig & sub_sig
        jaccard = len(overlap)/len(main_sig|sub_sig) if (main_sig|sub_sig) else 0
        overlap_pct = len(overlap)/len(main_sig)*100 if main_sig else 0

        coef_consistent = 0
        for e in overlap:
            mc = main_results[(main_results['source_roi']==e[0])&(main_results['target_roi']==e[1])]['coef'].values[0]
            sc = sub_results[(sub_results['source_roi']==e[0])&(sub_results['target_roi']==e[1])]['coef'].values[0]
            if mc * sc > 0:
                coef_consistent += 1

        stats.append({
            'repeat': rep, 'n_sig_main': len(main_sig), 'n_sig_sub': len(sub_sig),
            'n_overlap': len(overlap), 'overlap_pct': round(overlap_pct,1),
            'jaccard': round(jaccard,4), 'coef_consistent': coef_consistent,
            'p_threshold': round(sub_thr,6),
        })
        elapsed = time.time()-t0
        print(f'  重叠={len(overlap)}({overlap_pct:.0f}%), 方向一致={coef_consistent}/{len(overlap)}, 耗时 {elapsed/60:.1f}min')

        # 保存每次的单独结果 (检查清单 #107)
        sub_results.to_csv(os.path.join(opt.out_dir, f'sensitivity_full_{rep}_glm.csv'), index=False)

    # 保存 HC 名单
    pd.DataFrame(all_hc_lists).to_csv(os.path.join(opt.out_dir, 'sensitivity_full_hc_lists.csv'), index=False)
    return stats


# ============================================================
# 7. 保存
# ============================================================
def save_all_results(avg_adjs, pheno, fold_arr, glm_results, output_dir, atlas, subject_ids):
    """保存全部输出文件 (检查清单 #117-127)。"""
    os.makedirs(output_dir, exist_ok=True)

    # --- 每人的有向邻接矩阵 ---
    adj_dir = os.path.join(output_dir, 'adj')
    os.makedirs(adj_dir, exist_ok=True)
    for i, sid in enumerate(subject_ids):
        np.savetxt(os.path.join(adj_dir, f'sub-{sid}.txt'), avg_adjs[i], fmt='%.6f', delimiter='\t')
    print(f'\n[保存] {len(subject_ids)} 个有向邻接矩阵 → {adj_dir}/')

    # --- 元数据表 (检查清单 #118) ---
    meta = pheno[['subject_id', 'group', 'is_BD', 'scanner', 'mean_fd']].copy()
    meta['fold'] = [f'f{f}' for f in fold_arr]
    meta.to_csv(os.path.join(output_dir, 'subject_metadata.csv'), index=False)
    print(f'[保存] 元数据 → {output_dir}/subject_metadata.csv')

    # --- 完整 GLM 结果 ---
    glm_path = os.path.join(output_dir, 'glm_results.csv')
    glm_results.to_csv(glm_path, index=False, float_format='%.6f')
    print(f'[保存] GLM 结果 ({len(glm_results)}条边) → {glm_path}')

    # --- 显著边 ---
    sig = glm_results[glm_results['significant']]
    sig_path = os.path.join(output_dir, 'significant_edges.csv')
    sig.to_csv(sig_path, index=False, float_format='%.6f')
    print(f'[保存] {len(sig)}条显著边 → {sig_path}')

    # --- 显著边 (含 ROI 名称) ---
    labels_path = ROI_BASE / atlas / 'roi_labels.tsv'
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path, sep='\t')
        label_map = dict(zip(labels_df['label_id'], labels_df['roi_name']))
        sig_named = sig.copy()
        sig_named['source_name'] = sig_named['source_roi'].apply(lambda x: label_map.get(x+1, '?'))
        sig_named['target_name'] = sig_named['target_roi'].apply(lambda x: label_map.get(x+1, '?'))
        sig_named.to_csv(os.path.join(output_dir, 'significant_edges_named.csv'), index=False, float_format='%.6f')
        print(f'[保存] 显著边(含名称) → {output_dir}/significant_edges_named.csv')

    return sig


# ============================================================
# 8. 参数解析
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='BD vs HC 有向脑网络组间比较 (HO55/HO110)')
    # 数据和模式
    p.add_argument('--atlas', default='HO55', choices=['HO55', 'HO110'])
    p.add_argument('--skip_train', action='store_true')
    p.add_argument('--adj_dir', default=None)
    # 敏感性分析
    p.add_argument('--sensitivity_glm', type=int, default=0, help='统计子抽样次数 (默认0)')
    p.add_argument('--sensitivity_full', type=int, default=0, help='完整重训次数 (默认0)')
    # 种子
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_seeds', type=int, default=3, help='多随机种子数 (默认3, 检查清单 #30)')
    # FSTA
    p.add_argument('--epoch', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--d_model', type=int, default=16)
    p.add_argument('--d_inner_hid', type=int, default=64)
    p.add_argument('--d_k', type=int, default=8)
    p.add_argument('--d_v', type=int, default=8)
    p.add_argument('--n_head', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--soft_threshold', type=float, default=0.5)
    p.add_argument('--alpha_sp', type=float, default=0.8)
    p.add_argument('--n_warmup_steps', type=int, default=4000)
    p.add_argument('--lr_mul', type=float, default=1.2)
    # 其他
    p.add_argument('--gpu_id', default='auto')
    p.add_argument('--out_dir', default=None)
    # 兼容
    p.add_argument('--variance', type=float, default=5)
    p.add_argument('--no_filters', action='store_true')
    p.add_argument('--label_smoothing', action='store_true')
    p.add_argument('--model_name', default='FMLPRec')
    p.add_argument('--num_hidden_layers', type=int, default=1)
    p.add_argument('--num_attention_heads', type=int, default=2)
    p.add_argument('--hidden_act', default='gelu')
    p.add_argument('--attention_probs_dropout_prob', type=float, default=0.5)
    p.add_argument('--hidden_dropout_prob', type=float, default=0.5)
    p.add_argument('--initializer_range', type=float, default=0.02)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--adam_beta1', type=float, default=0.9)
    p.add_argument('--adam_beta2', type=float, default=0.999)

    opt = p.parse_args()
    opt.d_word_vec = opt.d_model
    opt.out_dir = opt.out_dir or str(OUT_BASE / opt.atlas)
    return opt


# ============================================================
# 主入口
# ============================================================
def main():
    opt = parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)

    print('='*60)
    print(f'  BD vs HC 有向脑网络组间比较 — {opt.atlas}')
    print(f'  多种子: {opt.n_seeds}, FDR α={FDR_ALPHA}')
    print(f'  输出: {opt.out_dir}')
    print(f'  时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)

    # 保存运行配置 (检查清单 #32)
    with open(os.path.join(opt.out_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=2, default=str)
    print('[配置] 已保存 → run_config.json')

    # ── 加载数据 + QC ──
    device = get_free_device(opt.gpu_id)
    data, subject_ids, pheno = load_and_qc(atlas=opt.atlas)
    S, T, N = data.shape
    print(f'\n[数据] {S}人 × {T}tp × {N}ROI, BD={(pheno["is_BD"]==1).sum()}, HC={(pheno["is_BD"]==0).sum()}')

    # ── 多种子 FSTA 交叉验证 ──
    if opt.skip_train and opt.adj_dir:
        print(f'\n[跳过训练] 从 {opt.adj_dir} 加载')
        adj_path = Path(opt.adj_dir)
        avg_adjs = np.zeros((S, N, N), dtype=np.float32)
        for i, sid in enumerate(subject_ids):
            avg_adjs[i] = np.loadtxt(adj_path / f'sub-{sid}.txt', delimiter='\t')
        # 从元数据读取 fold (如果有)
        meta_path = Path(opt.adj_dir).parent / 'subject_metadata.csv'
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            fold_arr = np.array([int(f.replace('f','')) for f in meta['fold']])
            print(f'[加载] fold 分配已从 {meta_path} 读取')
        else:
            fold_arr = np.ones(S, dtype=int)  # placeholder
    else:
        print(f'\n[FSTA] 多种子交叉验证: {opt.n_seeds} seeds × {N_FOLDS} folds × {opt.epoch} epochs')
        t0 = time.time()
        avg_adjs, fold_arr, all_seed_adjs = multi_seed_cv(data, pheno, opt, device, opt.seed, opt.n_seeds)
        print(f'\n[FSTA 完成] 总耗时 {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)')

        # 保存折分配 (检查清单 #18)
        fold_df = pd.DataFrame({'subject_id': pheno['subject_id'], 'fold': [f'f{f}' for f in fold_arr]})
        fold_df.to_csv(os.path.join(opt.out_dir, 'fold_assignments.csv'), index=False)
        print(f'[保存] 折分配 → fold_assignments.csv')

    # ── 逐边 GLM + HC3 + FDR ──
    glm_results, p_threshold = glm_per_edge(avg_adjs, pheno, fold_arr)
    sig = save_all_results(avg_adjs, pheno, fold_arr, glm_results, opt.out_dir, opt.atlas, subject_ids)

    # ── 敏感性 A: GLM 子抽样 ──
    if opt.sensitivity_glm > 0:
        sens_glm = sensitivity_glm_only(avg_adjs, pheno, fold_arr, glm_results, opt.sensitivity_glm, opt.seed)
        sdf = pd.DataFrame(sens_glm)
        sdf.to_csv(os.path.join(opt.out_dir, 'sensitivity_glm_stats.csv'), index=False, float_format='%.4f')
        print(f'\n[敏感性-GLM 汇总] 重叠率: {sdf["overlap_pct"].mean():.1f}±{sdf["overlap_pct"].std():.1f}%')
        print(f'  Jaccard: {sdf["jaccard"].mean():.3f}±{sdf["jaccard"].std():.3f}')
        print(f'  方向一致: {sdf["coef_consistent"].mean():.1f}±{sdf["coef_consistent"].std():.1f}')

    # ── 敏感性 B: 完整重训 ──
    if opt.sensitivity_full > 0:
        sens_full = sensitivity_full_retrain(data, pheno, opt, device, glm_results, opt.sensitivity_full, opt.seed)
        sdf = pd.DataFrame(sens_full)
        sdf.to_csv(os.path.join(opt.out_dir, 'sensitivity_full_stats.csv'), index=False, float_format='%.4f')
        print(f'\n[敏感性-FULL 汇总] 重叠率: {sdf["overlap_pct"].mean():.1f}±{sdf["overlap_pct"].std():.1f}%')
        print(f'  Jaccard: {sdf["jaccard"].mean():.3f}±{sdf["jaccard"].std():.3f}')

    # ── 最终汇总 ──
    print(f'\n{"="*60}')
    print(f'  完成!')
    print(f'  显著边: {len(sig)}/{len(glm_results)}')
    print(f'  BD>HC: {(sig["coef"]>0).sum()}, BD<HC: {(sig["coef"]<0).sum()}')
    print(f'  输出: {opt.out_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
