import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 获取项目根目录，保证直接运行本文件时也能导入项目模块。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_abide_i.data import ROI_COUNT, load_abide_records, load_time_series
from model.FSTA import FSTA
from model.Optim import ScheduledOptim


def set_deterministic_seed(seed):
    """固定 Python、NumPy 和 PyTorch 随机种子，便于复现实验。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(gpu_id):
    """选择 CPU、指定 GPU，或自动选择空闲显存最多的 GPU。"""
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id == "auto":
        free_memory = [
            torch.cuda.mem_get_info(index)[0]
            for index in range(torch.cuda.device_count())
        ]
        gpu_id = int(np.argmax(free_memory))
    return torch.device(f"cuda:{int(gpu_id)}")


class RandomSubjectWindowDataset(Dataset):
    """每个 epoch 从每名受试者中确定性随机抽取一个固定长度窗口。"""
    def __init__(self, time_series, window_length, seed):
        self.time_series = time_series
        self.window_length = window_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, index):
        subject_time_series = self.time_series[index]
        maximum_start = subject_time_series.shape[0] - self.window_length
        generator = np.random.default_rng(
            self.seed + self.epoch * len(self.time_series) + index
        )
        start = int(generator.integers(maximum_start + 1))
        window = subject_time_series[start : start + self.window_length]
        return torch.from_numpy(window)


class SharedWindowLoss(nn.Module):
    """计算时间序列重建损失，并可选原始或 entropy 正则项。"""
    MODES = ("original", "entropy")

    def __init__(self, mode, alpha, node_count, eps=1e-8):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.node_count = node_count
        self.eps = eps
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, reconstruction, attention, target):
        prediction_loss = self.reconstruction_loss(reconstruction, target)
        if self.mode == "original":
            # 保留原论文代码中的 attention.sum() 正则形式。
            regularizer = attention.sum()
        else:
            # 使用归一化熵约束注意力分布，熵越小表示连接越集中。
            probabilities = attention.clamp_min(self.eps)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1)
            regularizer = entropy.mean() / math.log(self.node_count)
        total_loss = prediction_loss + self.alpha * regularizer
        return total_loss, prediction_loss, regularizer


def fixed_window_starts(time_points, window_length, stride):
    """生成固定步长窗口起点，并确保最后一个窗口覆盖序列末尾。"""
    starts = list(range(0, time_points - window_length + 1, stride))
    final_start = time_points - window_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def build_model(args, device):
    """使用当前参数构建原始 FSTA 模型，不修改 model/FSTA.py。"""
    options = argparse.Namespace(
        **vars(args),
        nodes_num=ROI_COUNT,
        time_num=args.window_length,
    )
    return FSTA(
        opt=options,
        time_num=args.window_length,
        d_model=args.d_model,
        d_inner=args.d_inner_hid,
        n_head=args.n_head,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout,
    ).to(device)


def train_shared_model(args, dataset, device):
    """使用所有受试者窗口训练一套共享 FSTA 参数。"""
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    # 创建数据加载器，每个 epoch 从每名受试者中随机抽取一个固定长度窗口。
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = build_model(args, device)
    base_optimizer = optim.Adam(
        model.parameters(),
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )
    optimizer = ScheduledOptim(
        base_optimizer,
        args.lr_mul,
        args.d_model,
        args.n_warmup_steps,
    )
    criterion = SharedWindowLoss(
        mode=args.loss_mode,
        alpha=args.loss_alpha,
        node_count=ROI_COUNT,
    ).to(device)

    final_metrics = None
    # 每个 epoch 中，每名受试者只贡献一个随机窗口。
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        epoch_values = []
        for windows in loader:
            windows = windows.to(device)
            optimizer.zero_grad()
            reconstruction, attention = model(windows)
            loss, prediction_loss, regularizer = criterion(
                reconstruction,
                attention,
                windows,
            )
            loss.backward()
            optimizer.step_and_update_lr()
            epoch_values.append(
                (loss.item(), prediction_loss.item(), regularizer.item())
            )
        averages = np.mean(epoch_values, axis=0)
        final_metrics = {
            "epoch": epoch,
            "loss": float(averages[0]),
            "reconstruction_loss": float(averages[1]),
            "regularizer": float(averages[2]),
        }
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch}/{args.epochs} loss={averages[0]:.6f} "
                f"reconstruction={averages[1]:.6f} "
                f"regularizer={averages[2]:.6f}"
            )
    return model, final_metrics


@torch.no_grad()
def extract_subject_bec(model, records, time_series, args, device, output_dir):
    """对每名受试者的全部固定窗口推理，并平均得到个体 BEC。"""
    model.eval()
    raw_attention_all = []
    bec_all = []
    labels = []
    subject_ids = []
    site_ids = []
    time_points = []
    window_counts = []
    reconstruction_mse = []
    mean_window_std = []

    # 对每名受试者的全部固定窗口推理，并平均得到个体 BEC。
    for index, (record, subject_time_series) in enumerate(
        zip(records, time_series),
        start=1,
    ):
        starts = fixed_window_starts(
            subject_time_series.shape[0],
            args.window_length,
            args.stride,
        )
        window_attention = []
        window_mse = []
        # 对每个窗口进行推理，计算注意力和重建误差。
        for start in starts:
            window = subject_time_series[start : start + args.window_length]
            window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            reconstruction, raw_attention = model(window_tensor)
            window_attention.append(raw_attention.cpu().numpy().astype(np.float32))
            window_mse.append(
                float(torch.mean((reconstruction - window_tensor) ** 2).item())
            )
        # 将所有窗口的注意力堆叠为一个数组，形状为 (窗口数, ROI数, ROI数)。
        window_attention = np.stack(window_attention)
        # 对同一受试者的全部窗口注意力取平均。
        raw_attention = window_attention.mean(axis=0)
        # 转置后 bec[i, j] 表示 ROI i 指向 ROI j。
        bec = raw_attention.T.copy()
        # 自连接不作为下游分类特征，因此将对角线设为0。
        np.fill_diagonal(bec, 0.0)
        window_std = window_attention.std(axis=0)

        raw_attention_all.append(raw_attention)
        bec_all.append(bec)
        labels.append(record.label)
        subject_ids.append(record.subject_id)
        site_ids.append(record.site_id)
        time_points.append(subject_time_series.shape[0])
        window_counts.append(len(starts))
        reconstruction_mse.append(np.mean(window_mse))
        mean_window_std.append(float(window_std.mean()))

        if index == 1 or index % args.subject_log_every == 0 or index == len(records):
            print(
                f"[{index}/{len(records)}] {record.subject_id} {record.diagnosis} "
                f"site={record.site_id} T={subject_time_series.shape[0]} "
                f"windows={len(starts)} mse={np.mean(window_mse):.6f}"
            )

    # 汇总保存全部受试者的 BEC、标签、站点和窗口信息。
    np.savez_compressed(
        output_dir / "subject_bec.npz",
        raw_attention=np.stack(raw_attention_all),
        bec=np.stack(bec_all),
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        site_ids=np.asarray(site_ids),
        time_points=np.asarray(time_points, dtype=np.int64),
        window_counts=np.asarray(window_counts, dtype=np.int64),
        reconstruction_mse=np.asarray(reconstruction_mse, dtype=np.float32),
        mean_window_attention_std=np.asarray(mean_window_std, dtype=np.float32),
        roi_names=np.asarray([f"AAL_{index:03d}" for index in range(1, 91)]),
    )
    return {
        "total_windows": int(np.sum(window_counts)),
        "mean_reconstruction_mse": float(np.mean(reconstruction_mse)),
        "mean_window_attention_std": float(np.mean(mean_window_std)),
    }


def parse_args():
    """读取数据、训练、模型和输出目录等命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=str(REPO_ROOT / "dataset/ABIDE-I"),
    )
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "downstream_abide_i/outputs"),
    )
    parser.add_argument("--output_dir")
    parser.add_argument("--loss_mode", choices=SharedWindowLoss.MODES, default="original")
    parser.add_argument("--loss_alpha", type=float, default=0.8)
    parser.add_argument("--window_length", type=int, default=78)
    parser.add_argument("--stride", type=int, default=39)
    parser.add_argument("--epochs", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--subject_log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", default="auto")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_inner_hid", type=int, default=64)
    parser.add_argument("--d_k", type=int, default=8)
    parser.add_argument("--d_v", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_warmup_steps", type=int, default=4000)
    parser.add_argument("--lr_mul", type=float, default=1.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--hidden_act", default="gelu")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--no_filters", action="store_true")
    return parser.parse_args()


def main():
    """加载数据、训练共享模型，并保存模型、BEC 和实验汇总。"""
    args = parse_args()
    if args.window_length != 78:
        raise ValueError("The quick ABIDE-I shared version is fixed to window_length=78")
    if args.stride <= 0 or args.stride > args.window_length:
        raise ValueError("stride must be in [1, window_length]")
    set_deterministic_seed(args.seed) # 固定随机种子，保证实验可复现
    # 加载 ABIDE-I 受试者记录和时间序列数据
    records = load_abide_records(
        data_root=args.data_root,
        pipeline=args.pipeline,
        strategy=args.strategy,
        derivative=args.derivative,
    )
    # 如果指定了 --max_subjects，则只使用前 N 名受试者。
    if args.max_subjects is not None:
        records = records[: args.max_subjects]
    # 加载每名受试者的时间序列数据，并可选标准化。
    print(f"Loading time series for {len(records)} subjects...")
    time_series = [
        load_time_series(record, standardize=args.standardize) for record in records
    ]
    # 检查是否有受试者的时间序列长度小于 window_length。
    if any(values.shape[0] < args.window_length for values in time_series):
        raise ValueError("At least one subject is shorter than window_length")
    # --output_dir 优先用于手动区分不同实验配置。
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.output_root)
        / args.loss_mode
        / f"loss_alpha_{args.loss_alpha}"
        / f"seed_{args.seed}"
        / f"epochs_{args.epochs}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # 选择设备（CPU、指定 GPU 或自动选择空闲显存最多的 GPU）。
    device = select_device(args.gpu_id)
    # 创建数据集，每个 epoch 从每名受试者中随机抽取一个固定长度窗口。
    dataset = RandomSubjectWindowDataset(
        time_series,
        args.window_length,
        args.seed,
    )
    print(
        f"Loaded {len(records)} subjects; one random [{args.window_length},90] "
        f"window per subject per epoch; device={device}"
    )
    # 训练共享 FSTA 模型，并保存模型参数。
    model, final_training_metrics = train_shared_model(args, dataset, device)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "nodes_num": ROI_COUNT,
            "time_num": args.window_length,
        },
        output_dir / "model.pt",
    )
    # 对每名受试者的全部固定窗口推理，并平均得到个体 BEC。
    bec_statistics = extract_subject_bec(
        model,
        records,
        time_series,
        args,
        device,
        output_dir,
    )
    # 保存实验汇总信息，包括配置、受试者数量、输出目录、模型路径、BEC 路径、训练指标和 BEC 统计信息。
    summary = {
        "config": vars(args),
        "result": {
            "subjects": len(records),
            "atlas": "AAL90",
            "bec_path": str(output_dir / "subject_bec.npz"),
            "transductive_shared_training": True,
            "final_training_metrics": final_training_metrics,
            **bec_statistics,
        },
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)


if __name__ == "__main__":
    main()
