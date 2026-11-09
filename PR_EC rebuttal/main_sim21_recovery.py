"""Sim21 recovery comparison; optionally train MPR only, never the encoder."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(ROOT.parent))
    __package__ = ROOT.name

from .simulation.sim21_recovery import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/sim21")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--mpr-dir", type=Path, default=None, help="Optional five Sim21 fold MPR checkpoints with provenance metadata")
    parser.add_argument("--train-mpr", action="store_true", help="Explicitly train fold-local MPR from cached EC with fixed existing settings; never retrain encoder")
    parser.add_argument("--mat-path", type=Path, default=ROOT.parent / "dataset/NetSim/sim21.mat")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sim21_recovery")
    args = parser.parse_args()
    if args.train_mpr and args.mpr_dir is None:
        args.mpr_dir = ROOT / "outputs/sim21_mpr"
    if not (args.source_dir / "summary.json").is_file():
        candidates = []
        outputs = ROOT / "outputs"
        if outputs.is_dir():
            for path in sorted(outputs.rglob("summary.json")):
                try:
                    summary = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if summary.get("dataset") == "NetSim Sim21" and (path.parent / "oof_predictions.npz").is_file():
                    candidates.append(path.parent)
        hints = "\n".join(f'  --source-dir "{path}"' for path in candidates)
        parser.error(
            f"未找到已保存的 Sim21 实验：{args.source_dir / 'summary.json'}\n"
            "此脚本需要已有 OOF 预测，不会训练 Encoder，也不能用原始 sim21.mat 代替 OOF 预测。\n"
            + (f"发现以下 Sim21 结果目录，请显式选择：\n{hints}\n" if candidates else
               "当前 outputs 下未发现其他 Sim21 OOF 结果。\n")
            + "如果结果在别处，请用 --source-dir 指定包含 summary.json 和 oof_predictions.npz 的目录。\n"
            "如果只有折内 Encoder 缓存，可先运行 main_sim21.py --reuse-cache --cache-dir 缓存路径，"
            "并传入生成缓存时相同的训练参数，重建参考与结果文件。\n"
            "如果从未生成或已丢失 Sim21 OOF 和缓存，则需另行运行 main_sim21.py 生成；"
            "这一步会训练 Encoder，本脚本不会自动执行。"
        )
    if args.cache_dir is None:
        args.cache_dir = args.source_dir / "encoder_cache"
    required = [args.source_dir / "oof_predictions.npz"]
    for fold in range(1, 6):
        required.extend([
            args.source_dir / f"fold_{fold:02d}_references.npz",
            args.cache_dir / f"fold_{fold:02d}/metadata.json",
            args.cache_dir / f"fold_{fold:02d}/ec.npz",
        ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Sim21 保存结果或折内缓存不完整，未开始评价或训练：\n  "
                     + "\n  ".join(missing)
                     + "\n请检查 --source-dir 和 --cache-dir，不能用其他数据集的 OOF EC 替代。")
    destination = args.output_dir.resolve()
    for source in (args.source_dir, args.cache_dir, args.mpr_dir):
        if source is not None:
            source = source.resolve()
            if destination == source or source in destination.parents or destination in source.parents:
                parser.error("Output must be separate from source, cache and MPR directories")
    return args


if __name__ == "__main__":
    run_evaluation(parse_args())
