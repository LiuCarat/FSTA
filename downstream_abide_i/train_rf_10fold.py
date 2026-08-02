import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


# 获取项目根目录
REPO_ROOT = Path(__file__).resolve().parents[1]
# 固定使用 weighted Precision、Recall 和 F1，与论文表格口径保持一致。
PRIMARY_METRICS = ("precision", "recall", "f1")


def parse_args():
    """读取实验配置、交叉验证和随机森林参数。"""
    parser = argparse.ArgumentParser(description="10-fold random forest classification using subject BEC matrices")
    
    # ================= 实验配置参数（用于自动拼接路径） =================
    parser.add_argument("--mood", type=str, default="original", help="数据子集名称，如 original、 entropy 等")
    parser.add_argument("--loss_alpha", type=float, default=0.8, help="loss_alpha 值")
    parser.add_argument("--epoch", type=int, default=21, help="训练的 epoch 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # ================= 路径参数（可选，不传则根据上述参数自动拼接） =================
    parser.add_argument("--bec_path", default=None, help="自定义 subject_bec.npz 路径（不传则自动拼接）")
    parser.add_argument("--output_dir", default=None, help="自定义结果输出目录（不传则自动拼接）")
    
    # ================= 模型与验证参数 =================
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--n_jobs", type=int, default=-1)
    
    return parser.parse_args()


def calculate_metrics(labels, predictions):
    """计算 weighted Precision、Recall 和 F1。"""
    return {
        "precision": precision_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def summarize(rows, names):
    """计算10折指标的均值和总体标准差。"""
    values = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=0)


def format_mean_std(mean, std):
    """将0到1的小数指标格式化为百分数形式，例如63.13±5.51。"""
    return f"{(mean*100):.2f}±{(std*100):.2f}"


def format_percent(value):
    """将0到1的小数指标格式化为百分数并保留两位小数。"""
    return f"{value * 100:.2f}"


def main():
    """读取个体 BEC，运行分层10折随机森林，并保存全部结果。"""
    args = parse_args()
    
    # ================= 自动拼接路径逻辑 =================
    base_dir = REPO_ROOT / "downstream_abide_i" / "outputs"
    exp_subdir = f"loss_alpha_{args.loss_alpha}/seed_{args.seed}/epochs_{args.epoch}"
    
    # 如果用户手动指定了路径，则使用手动指定的；否则自动拼接
    bec_path = Path(args.bec_path) if args.bec_path else (base_dir / args.mood / exp_subdir / "subject_bec.npz")
    output_dir = Path(args.output_dir) if args.output_dir else (base_dir / "random_forest_10fold" / args.mood / exp_subdir)
    # ==================================================

    if not bec_path.is_file():
        raise FileNotFoundError(
            f"BEC file not found: {bec_path}. Run train_shared_window_fsta.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # subject_bec.npz 中只需要 BEC 和分类标签。
    data = np.load(bec_path, allow_pickle=False)
    required_keys = {"bec", "labels"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise ValueError(f"Missing arrays in {bec_path}: {sorted(missing_keys)}")

    # BEC 形状为 [受试者数量, ROI数量, ROI数量]。
    bec = data["bec"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC shape [subjects, nodes, nodes], got {bec.shape}")
    if len(bec) != len(labels):
        raise ValueError("BEC and labels have different lengths")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1, got {np.unique(labels)}")

    # 去除90个自连接，将每张 BEC 展开为90×89=8010条有向边。
    directed_mask = ~np.eye(bec.shape[-1], dtype=bool)
    features = bec[:, directed_mask]
    print(
        f"Loaded BEC from {bec_path}: \n subjects={len(labels)}, "
        f"nodes={bec.shape[-1]}, directed_features={features.shape[1]}"
    )
    print("Metric average=weighted")

    # 分层划分保证每一折中的 ASD/HC 比例尽量一致。
    splitter = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    # 保存每一折的分类指标。
    fold_rows = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(features, labels),
        start=1,
    ):
        # 每一折重新训练一套随机森林，默认使用论文描述的1000棵树。
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.seed + fold,
        )
        classifier.fit(features[train_indices], labels[train_indices])
        predictions = classifier.predict(features[test_indices])

        # 固定计算 weighted 指标。
        primary = calculate_metrics(labels[test_indices], predictions)
        row = {
            "fold": fold,
            **primary,
        }
        fold_rows.append(row)

        print(
            f"fold={fold:02d} "
            f"weighted_precision={primary['precision'] * 100:.2f} "
            f"weighted_recall={primary['recall'] * 100:.2f} "
            f"weighted_F1={primary['f1'] * 100:.2f}"
        )

    # 汇总10折均值和标准差。
    primary_means, primary_stds = summarize(fold_rows, PRIMARY_METRICS)
    fold_fields = ["fold", *PRIMARY_METRICS]

    # fold_metrics.csv 保存每折结果以及最后的 mean、std。
    with (output_dir / "fold_metrics.csv").open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(
            {
                "fold": row["fold"],
                **{name: format_percent(row[name]) for name in PRIMARY_METRICS},
            }
            for row in fold_rows
        )
        writer.writerow(
            {
                "fold": "mean±std",
                **{
                    name: format_mean_std(mean, std)
                    for name, mean, std in zip(
                        PRIMARY_METRICS, primary_means, primary_stds
                    )
                },
            }
        )

    # summary.json 保存最终均值、标准差和完整分类配置。
    summary = {
        "metric_average": "weighted",
        "subjects": len(labels),
        "features": features.shape[1],
        "n_splits": args.n_splits,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "loss_alpha": args.loss_alpha,
        "epoch": args.epoch,
        "mood": args.mood,
        "precision_mean_percent": f"{primary_means[0] * 100:.2f}",
        "precision_std_percent": f"{primary_stds[0] * 100:.2f}",
        "recall_mean_percent": f"{primary_means[1] * 100:.2f}",
        "recall_std_percent": f"{primary_stds[1] * 100:.2f}",
        "f1_mean_percent": f"{primary_means[2] * 100:.2f}",
        "f1_std_percent": f"{primary_stds[2] * 100:.2f}",
    }
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(
        f"Precision={format_mean_std(primary_means[0], primary_stds[0])} "
        f"Recall={format_mean_std(primary_means[1], primary_stds[1])} "
        f"F1={format_mean_std(primary_means[2], primary_stds[2])}"
    )


if __name__ == "__main__":
    main()