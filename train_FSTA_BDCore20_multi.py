import argparse
import copy
import gc
import os
import random

import numpy as np
import torch
import torch.optim as optim

from model.FSTA import FSTA
from model.Optim import ScheduledOptim
from utils.utils import *
from utils.FourierAttUtils import set_seed


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def prepare_dataloader(opt, device, group):
    data_path = os.path.join(opt.data_root, group)
    subject_files = sorted(
        file_name for file_name in os.listdir(data_path)
        if file_name.endswith('.txt')
    )
    if not subject_files:
        raise ValueError(f'No subject files found in {data_path}')

    subject_data = []
    expected_shape = None
    for file_name in subject_files:
        position = os.path.join(data_path, file_name)
        data_tmp = np.loadtxt(
            position,
            skiprows=opt.skiprows,
            delimiter='\t',
        )
        if data_tmp.ndim != 2:
            raise ValueError(f'Expected 2D data in {position}, got {data_tmp.shape}')
        if expected_shape is None:
            expected_shape = data_tmp.shape
        elif data_tmp.shape != expected_shape:
            raise ValueError(
                f'Inconsistent subject shape in {position}: '
                f'{data_tmp.shape} != {expected_shape}'
            )
        subject_data.append(data_tmp)

    data = np.stack(subject_data, axis=0)
    reference_adj = bdcore20_data_label(group)
    if data.shape[2] != reference_adj.shape[0]:
        raise ValueError(
            f'{group} has {data.shape[2]} nodes, but reference adjacency is '
            f'{reference_adj.shape}'
        )

    print(f'{group} subjects: {len(subject_files)}')
    print(f'{group} data: {data.shape}')

    data = torch.FloatTensor(data).to(device)
    opt.nodes_num = data.shape[2]
    opt.time_num = data.shape[1]
    dataset = torch.utils.data.TensorDataset(data, data)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
    )
    return data_loader, reference_adj


def train_epoch(model, data_loader, optimizer, criterion, opt, device, smoothing):
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
    adj_mean = np.mean(batch_adj, axis=0)
    adj_mean = adj_mean.T

    return train_loss, adj_mean


def train(data_loader, reference_adj, device, opt, group):
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
    print(
        f'train params: d_model:{opt.d_model}, d_k/d_v:{opt.d_k}, '
        f'n_head:{opt.n_head}, alpha_sp:{opt.alpha_sp}, '
        f'soft_threshold:{opt.soft_threshold}, dropout:{opt.dropout}'
    )
    optimizer = ScheduledOptim(
        optim.Adam(
            [{'params': model.parameters()}],
            betas=(0.9, 0.98),
            eps=1e-09,
        ),
        opt.lr_mul,
        opt.d_model,
        opt.n_warmup_steps,
    )
    criterion = loss_func(alpha_sp=opt.alpha_sp).to(device)

    for epoch_i in range(opt.epoch):
        train_loss, adj = train_epoch(
            model,
            data_loader,
            optimizer,
            criterion,
            opt,
            device,
            smoothing=opt.label_smoothing,
        )
        adj_init = copy.deepcopy(adj)
        adj[np.arange(opt.nodes_num), np.arange(opt.nodes_num)] = 0
        opt.threshold = softThres(adj, opt.soft_threshold)
        adj_binary = change01(adj, threshold=opt.threshold)
        precision, recall, F1, accuracy, SHD, TP, FP, FN, total_pred = (
            cal_metrics_detailed(adj_binary, reference_adj)
        )
        if epoch_i % 100 == 0:
            print(
                f'group:{group}, epoch:{epoch_i}, loss:{train_loss: .3f}, '
                f'TP:{TP}, FP:{FP}, FN:{FN}, '
                f'pred_edges:{total_pred}, SHD:{SHD}'
            )
            print('threshold:', opt.threshold)
            print(adj_init)
            print(adj_binary)
        gc.collect()
        torch.cuda.empty_cache()
    return adj, TP, FP, FN, total_pred, SHD


def train_group(opt, device, group):
    print(f'========== Training {group} ==========')
    data_loader, reference_adj = prepare_dataloader(opt, device, group)
    output_path = os.path.join(opt.output_root, group)
    os.makedirs(output_path, exist_ok=True)

    metrics = []
    for run_index in range(1, opt.runs + 1):
        print(f'{group} run: {run_index}')
        adj, TP, FP, FN, total_pred, SHD = train(
            data_loader,
            reference_adj,
            device,
            opt,
            group,
        )
        np.savetxt(
            os.path.join(output_path, f'{run_index}.txt'),
            adj,
            fmt='%.04f',
            delimiter='\t',
        )
        metrics.append([TP, FP, FN, total_pred, SHD])

    mu = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    print(f'{group} mu:{mu}, std:{std}')

    headers = ['run', 'TP', 'FP', 'FN', 'pred_edges', 'SHD']
    save_metrics_csv(
        os.path.join(output_path, 'metrics.csv'),
        metrics,
        headers=headers,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-group',
        '--group',
        choices=['BD', 'HC', 'both'],
        default='both',
        help='BDCore20 group to train; default trains BD and HC separately',
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='./dataset/BDCore20',
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='./out/BDCore20',
    )
    parser.add_argument('-skiprows', type=int, default=1)
    parser.add_argument('-runs', type=int, default=20)

    parser.add_argument('-epoch', type=int, default=301)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-d_model', type=int, default=16)
    parser.add_argument('-d_inner_hid', type=int, default=64)
    parser.add_argument('-d_k', type=int, default=8)
    parser.add_argument('-d_v', type=int, default=8)
    parser.add_argument('-n_head', type=int, default=2)

    parser.add_argument('-soft_threshold', type=float, default=0.5)
    parser.add_argument('-alpha_sp', type=float, default=0.8)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=1.2)
    parser.add_argument('-dropout', type=float, default=0.2)
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-time_num', type=int, default=None)
    parser.add_argument('--nodes_num', type=int, default=None)

    parser.add_argument('--model_name', default='FMLPRec', type=str)
    parser.add_argument('--num_hidden_layers', default=1, type=int)
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default='gelu', type=str)
    parser.add_argument('--attention_probs_dropout_prob', default=0.5, type=float)
    parser.add_argument('--hidden_dropout_prob', default=0.5, type=float)
    parser.add_argument('--initializer_range', default=0.02, type=float)
    parser.add_argument('--no_filters', action='store_true')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--adam_beta1', default=0.9, type=float)
    parser.add_argument('--adam_beta2', default=0.999, type=float)
    parser.add_argument(
        '--gpu_id',
        default='auto',
        type=str,
        help="GPU device ID: 'auto', '0'/'1', or 'cpu'",
    )
    parser.add_argument('--variance', default=5, type=float)

    opt = parser.parse_args()
    set_seed(opt.seed)
    opt.d_word_vec = opt.d_model

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    device = get_free_device(opt.gpu_id)

    opt.d_model = 16
    opt.n_head = 2
    opt.d_k = 8
    opt.d_v = 8
    opt.alpha_sp = 0.8
    opt.soft_threshold = 0.5
    opt.dropout = 0.2
    opt.epoch = 301

    groups = ['BD', 'HC'] if opt.group == 'both' else [opt.group]
    for group in groups:
        train_group(opt, device, group)


if __name__ == '__main__':
    main()
