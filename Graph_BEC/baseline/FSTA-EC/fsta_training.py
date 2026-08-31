"""Unsupervised shared FSTA training."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import FSTA, ScheduledOptim
from utils.utils import loss_func
from Graph_BEC.utils import RandomSubjectWindowDataset, set_seed


def build_fsta(args, device):
    options = argparse.Namespace(**vars(args), nodes_num=90, time_num=args.window_length)
    return FSTA(options, args.window_length, args.d_model, args.d_inner_hid, args.n_head, args.d_k, args.d_v, args.dropout).to(device)


def train_fsta(args, time_series, device):
    set_seed(args.seed)
    dataset = RandomSubjectWindowDataset(time_series, args.window_length, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    model = build_fsta(args, device)
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), betas=(args.adam_beta1, args.adam_beta2), eps=1e-9, weight_decay=args.weight_decay), args.lr_mul, args.d_model, args.n_warmup_steps)
    criterion = loss_func(args.alpha_sp).to(device)
    metrics = {}
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch); model.train(); values = []
        for windows in loader:
            windows = windows.to(device); optimizer.zero_grad()
            reconstruction, attention = model(windows)
            total = criterion(reconstruction, attention, windows)
            total.backward(); optimizer.step_and_update_lr()
            values.append((total.item(), float(torch.nn.functional.mse_loss(reconstruction, windows).item()), float(attention.sum().item())))
        mean_values = np.mean(values, axis=0)
        metrics = {"epoch": epoch, "loss": float(mean_values[0]), "reconstruction_loss": float(mean_values[1]), "regularizer": float(mean_values[2])}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"FSTA epoch={epoch}/{args.epochs} loss={metrics['loss']:.6f} reconstruction={metrics['reconstruction_loss']:.6f}")
    return model, metrics
