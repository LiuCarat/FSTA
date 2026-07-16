#!/usr/bin/env python3
"""
FSTA 训练脚本 - ucla_cnp 数据集版本

基于 train_FSTA_real_multi.py，适配 ucla_cnp ROI 时间序列数据。
与 real/sanch 不同，ucla_cnp 没有 ground truth 连接矩阵，
因此训练为纯自监督模式，输出学习到的邻接矩阵。

用法:
    # 基础训练（使用所有已提取的受试者）
    python train_FSTA_ucla_multi.py

    # 指定参数
    python train_FSTA_ucla_multi.py --epoch 500 --batch_size 16 --d_model 32

    # 限制受试者数量
    python train_FSTA_ucla_multi.py --max_subjects 50
"""

import argparse
import numpy as np
import random
import os
import copy
import torch
import gc
import torch.optim as optim
from pathlib import Path

from model.FSTA import FSTA
from model.Optim import ScheduledOptim
from utils.utils import *
from utils.FourierAttUtils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_rating_matrix

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'dataset' / 'roi_timeseries' / 'HO62' / 'BD'
DEFAULT_OUT_DIR = PROJECT_ROOT / 'out' / 'ucla_cnp'


def prepare_dataloaders(opt, device):
    """
    从 ucla_cnp ROI 时间序列 .txt 文件加载数据。

    格式与 real/sanch 完全一致: tab 分隔，第一行为列名，每行一个时间点。
    """
    path = opt.data_path
    if not os.path.isdir(path):
        raise FileNotFoundError(f'数据目录不存在: {path}')

    all_path = sorted(os.listdir(path))
    # 过滤非 .txt 文件
    all_path = [f for f in all_path if f.endswith('.txt')]

    if opt.max_subjects and opt.max_subjects < len(all_path):
        all_path = all_path[:opt.max_subjects]
        print(f'[Data] 限制受试者数量: {opt.max_subjects}/{len(os.listdir(path))}')

    subjects = len(all_path)
    if subjects == 0:
        raise ValueError(f'数据目录为空: {path}')

    print(f'[Data] 受试者数量: {subjects}')

    data = np.empty((subjects, 0, 0))
    for i, sub_path in enumerate(all_path):
        position = os.path.join(path, sub_path)
        data_tmp = np.loadtxt(position, skiprows=opt.skiprows, delimiter='\t')
        if i == 0:
            data = np.expand_dims(data_tmp, axis=0)
        else:
            data = np.concatenate((data, np.expand_dims(data_tmp, axis=0)), axis=0)

    print(f'[Data] 数据 shape: {data.shape}  (S={subjects}, T={data.shape[1]}, N={data.shape[2]})')

    # data: [S, T, N]
    data = torch.FloatTensor(data).to(device)
    opt.nodes_num = data.shape[2]
    opt.time_num = data.shape[1]
    label = data  # 自监督: label 即 data

    dataset = torch.utils.data.TensorDataset(data, label)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size)

    return data_loader


def train_epoch(model, data_loader, optimizer, criterion, opt, device, smoothing):
    """单训练 epoch"""
    model.train()
    train_loss = []
    batch_adj = []

    for data_tmp, label_tmp in data_loader:
        optimizer.zero_grad()
        output, adj = model(data_tmp)
        loss = criterion(output, adj, label_tmp)
        loss.backward()
        optimizer.step_and_update_lr()

        train_loss.append(loss.item())
        batch_adj.append(adj.cpu().detach().numpy())

    train_loss = np.average(train_loss)
    adj_mean = np.mean(batch_adj, axis=0)  # [c, N, N] -> [N, N]
    adj_mean = adj_mean.T

    return train_loss, adj_mean


def train(data_loader, device, opt):
    """训练主循环"""
    model = FSTA(
        opt=opt,
        time_num=opt.time_num,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    ).to(device)

    print(f'[Model] d_model={opt.d_model}, d_k/d_v={opt.d_k}, n_head={opt.n_head}, '
          f'alpha_sp={opt.alpha_sp}, soft_threshold={opt.soft_threshold}, dropout={opt.dropout}')
    print(f'[Model] nodes={opt.nodes_num}, time={opt.time_num}')

    optimizer = ScheduledOptim(
        optim.Adam([{'params': model.parameters()}], betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps,
    )

    criterion = loss_func(alpha_sp=opt.alpha_sp).to(device)

    for epoch_i in range(opt.epoch):
        train_loss, adj = train_epoch(
            model, data_loader, optimizer, criterion, opt, device,
            smoothing=opt.label_smoothing,
        )
        adj_init = copy.deepcopy(adj)
        adj[np.arange(opt.nodes_num), np.arange(opt.nodes_num)] = 0
        opt.threshold = softThres(adj, opt.soft_threshold)
        adj_binary = change01(adj, threshold=opt.threshold)

        # 计算邻接矩阵统计（无 ground truth，不做 SHD/F1 比较）
        total_edges = int(np.sum(adj_binary))
        density = total_edges / (opt.nodes_num * (opt.nodes_num - 1))
        n_edges_per_node = total_edges / opt.nodes_num

        if epoch_i % 100 == 0:
            print(f'[Epoch {epoch_i}] loss={train_loss:.4f}, '
                  f'edges={total_edges}, density={density:.4f}, '
                  f'edges/node={n_edges_per_node:.1f}, threshold={opt.threshold:.4f}')

        gc.collect()
        torch.cuda.empty_cache()

    return adj, adj_binary, total_edges, density


def main():
    parser = argparse.ArgumentParser(description='FSTA 训练 - ucla_cnp 数据集')

    # 数据
    parser.add_argument('--data_path', type=str, default=str(DEFAULT_DATA_DIR),
                        help='ROI 时间序列目录')
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='限制使用的受试者数量')
    parser.add_argument('--skiprows', type=int, default=1,
                        help='np.loadtxt 跳过的行数（表头）')

    # 训练
    parser.add_argument('--epoch', type=int, default=301)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--runs', type=int, default=20,
                        help='重复运行次数')

    # 模型
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_inner_hid', type=int, default=64)
    parser.add_argument('--d_k', type=int, default=8)
    parser.add_argument('--d_v', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    # 正则化
    parser.add_argument('--soft_threshold', type=float, default=0.5)
    parser.add_argument('--alpha_sp', type=float, default=0.8)
    parser.add_argument('--label_smoothing', action='store_true')

    # 优化器
    parser.add_argument('--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--lr_mul', type=float, default=1.2)

    # 输出
    parser.add_argument('--out_dir', type=str, default=str(DEFAULT_OUT_DIR),
                        help='输出目录')

    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default='auto')
    parser.add_argument('--variance', type=float, default=5)
    parser.add_argument('--no_filters', action='store_true')

    # FourierAtt / 序列模型参数（保持兼容）
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
    set_seed(opt.seed)
    opt.d_word_vec = opt.d_model

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    device = get_free_device(opt.gpu_id)

    # ========= Loading Dataset ========= #
    data_loader = prepare_dataloaders(opt, device)

    # ========= Training ========= #
    os.makedirs(opt.out_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'开始训练: {opt.runs} 次独立运行')
    print(f'数据: {opt.data_path}')
    print(f'受试者: {opt.max_subjects or "全部"}, 节点: {opt.nodes_num}, 时间点: {opt.time_num}')
    print(f'{"="*60}\n')

    all_adjs = []  # 存储每次运行的连续值邻接矩阵
    stats = []     # 存储每次运行的统计信息

    for run_i in range(1, opt.runs + 1):
        print(f'\n--- Run {run_i}/{opt.runs} ---')
        adj, adj_binary, total_edges, density = train(data_loader, device, opt)

        # 保存连续值邻接矩阵
        adj_path = os.path.join(opt.out_dir, f'{run_i}.txt')
        np.savetxt(adj_path, adj, fmt='%.04f', delimiter='\t')

        all_adjs.append(adj)
        stats.append([total_edges, density])

        print(f'[Run {run_i}] edges={total_edges}, density={density:.4f}')

    # ========= 汇总 ========= #
    print(f'\n{"="*60}')
    print('汇总统计')
    print(f'{"="*60}')

    arr = np.array(stats)
    mu = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    print(f'边数:        {mu[0]:.0f} ± {std[0]:.0f}')
    print(f'密度:        {mu[1]:.4f} ± {std[1]:.4f}')

    # 保存均值邻接矩阵
    mean_adj = np.mean(all_adjs, axis=0)
    np.savetxt(os.path.join(opt.out_dir, 'mean_adj.txt'), mean_adj, fmt='%.04f', delimiter='\t')
    print(f'\n均值邻接矩阵已保存至: {opt.out_dir}/mean_adj.txt')

    # 保存统计 CSV
    save_metrics_csv(
        os.path.join(opt.out_dir, 'stats.csv'),
        stats,
        headers=['run', 'total_edges', 'density'],
    )


if __name__ == '__main__':
    main()
