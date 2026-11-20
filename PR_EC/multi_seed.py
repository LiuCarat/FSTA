"""Multi-seed orchestration and hierarchical metric summaries."""
from __future__ import annotations

import csv
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np

METRIC_SOURCES = {
    "ACC": "ACC",
    "AUC": "AUC",
    "SEN": "Recall",
    "PRE": "Precision",
    "F1": "F1",
}
MODES = ("reuse_ec", "end_to_end")
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORIES = {"abide": "abide-i", "abide_ii": "abide-ii", "adhd200": "adhd200"}
CONFIG_EXCLUSIONS = {
    "seed", "seeds", "input_mode", "ec_path", "refined_ec_path",
    "qsr_refined_ec_path", "output_dir", "gpu_id", "log_every", "profile",
    "multi_seed_mode", "seed42_results_dir",
}


def add_multi_seed_arguments(parser):
    parser.add_argument(
        "--multi-seed-mode",
        choices=MODES,
        default="reuse_ec",
        help=(
            "reuse_ec reuses one cached Initial EC and varies only the ten-fold "
            "pipeline; end_to_end retrains and re-extracts Initial EC per seed"
        ),
    )
    parser.add_argument(
        "--seed42-results-dir",
        type=Path,
        help=(
            "existing completed seed42 result directory; defaults to "
            "PR_EC/outputs/seed42/<dataset>"
        ),
    )


def validate_multi_seed_arguments(parser, args):
    seeds = list(args.seeds or [])
    if not seeds:
        return args
    if len(seeds) != len(set(seeds)):
        parser.error("--seeds must contain unique values")
    if seeds == [42]:
        args.seed = seeds[0]
        return args
    if 42 not in seeds:
        seeds.insert(0, 42)
    args.seeds = seeds
    expected_input = "ec" if args.multi_seed_mode == "reuse_ec" else "raw"
    if args.input_mode != expected_input:
        parser.error(
            f"--multi-seed-mode {args.multi_seed_mode} requires "
            f"--input-mode {expected_input}"
        )
    if len(seeds) < 2:
        parser.error("Multi-seed aggregation requires at least two seeds")
    return args


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_directory(args):
    seed_text = "_".join(str(seed) for seed in args.seeds)
    return Path(args.output_dir) / "multi_seed" / args.multi_seed_mode / f"seeds_{seed_text}"


def seed42_results_directory(args):
    if args.seed42_results_dir is not None:
        return Path(args.seed42_results_dir)
    try:
        directory = DATASET_DIRECTORIES[args.dataset]
    except KeyError as error:
        raise ValueError(f"No default seed42 result directory for dataset {args.dataset}") from error
    return REPO_ROOT / "PR_EC/outputs/seed42" / directory


def _normalise(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def load_seed42_summary(args):
    directory = seed42_results_directory(args)
    path = directory / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required completed seed42 summary not found: {path}. "
            "Seed 42 is mandatory and is reused rather than rerun."
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    if config.get("seed") != 42 or config.get("n_splits") != args.n_splits:
        raise ValueError("Existing seed42 result is not the requested seed42/fold protocol")
    if len(summary.get("folds", [])) != args.n_splits:
        raise ValueError("Existing seed42 result does not contain all folds")
    if list(summary.get("representations", [])) != list(args.representations):
        raise ValueError("Existing seed42 representations differ from this multi-seed run")
    requested = vars(args)
    mismatches = {}
    for key in sorted(set(config) & set(requested) - CONFIG_EXCLUSIONS):
        if _normalise(config[key]) != _normalise(requested[key]):
            mismatches[key] = {"seed42": config[key], "requested": requested[key]}
    if mismatches:
        raise ValueError(f"Existing seed42 configuration differs: {mismatches}")
    if args.multi_seed_mode == "reuse_ec":
        selected = Path(args.ec_path)
        expected = directory / selected.name
        if not selected.is_file():
            raise FileNotFoundError(f"Cached Initial EC not found: {selected}")
        if not expected.is_file():
            raise FileNotFoundError(f"Seed42 Initial EC provenance file not found: {expected}")
        if selected.resolve() != expected.resolve():
            with np.load(selected, allow_pickle=False) as left, np.load(expected, allow_pickle=False) as right:
                for key in ("ec", "labels", "subject_ids", "site_ids"):
                    if key not in left.files or key not in right.files or not np.array_equal(left[key], right[key]):
                        raise ValueError("Selected reusable Initial EC differs from the seed42 Initial EC")
    return summary, directory


def configure_seed_run(args, seed, root):
    """Create one isolated seed configuration without mutating the CLI args."""
    run_args = Namespace(**deepcopy(vars(args)))
    run_args.seed = int(seed)
    run_args.seeds = None
    run_args.output_dir = root / f"seed_{seed}"
    run_args.refined_ec_path = run_args.output_dir / Path(args.refined_ec_path).name
    run_args.qsr_refined_ec_path = run_args.output_dir / Path(args.qsr_refined_ec_path).name
    if args.multi_seed_mode == "reuse_ec":
        run_args.input_mode = "ec"
        run_args.ec_path = Path(args.ec_path)
    else:
        run_args.input_mode = "raw"
        run_args.ec_path = run_args.output_dir / Path(args.ec_path).name
    return run_args


def summarize_seed(seed, summary):
    rows = []
    for representation in summary["representations"]:
        row = {"seed": int(seed), "representation": representation, "n_folds": len(summary["folds"]), "fold_std_ddof": 0}
        for metric, source in METRIC_SOURCES.items():
            mean = float(summary[f"{representation}_{source}_mean"])
            std = float(summary[f"{representation}_{source}_std"])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_display_pct"] = f"{100 * mean:.2f} ± {100 * std:.2f}"
        rows.append(row)
    return rows


def aggregate_seed_means(seed_rows):
    representations = list(dict.fromkeys(row["representation"] for row in seed_rows))
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    if len(seeds) < 2:
        raise ValueError("Sample standard deviation requires at least two seeds")
    rows = []
    for representation in representations:
        selected = [row for row in seed_rows if row["representation"] == representation]
        if sorted(int(row["seed"]) for row in selected) != seeds:
            raise ValueError(f"Incomplete seed summaries for {representation}")
        row = {"representation": representation, "n_seeds": len(seeds), "seed_std_ddof": 1}
        for metric in METRIC_SOURCES:
            values = np.asarray([float(item[f"{metric}_mean"]) for item in selected])
            mean = float(np.mean(values))
            sample_std = float(np.std(values, ddof=1))
            row[f"{metric}_mean_of_seed_means"] = mean
            row[f"{metric}_sample_std"] = sample_std
            row[f"{metric}_display_pct"] = f"{100 * mean:.2f} ± {100 * sample_std:.2f}"
        rows.append(row)
    return rows


def print_seed_metrics(seed_rows):
    seed = seed_rows[0]["seed"]
    print(f"\nseed={seed}: ten-fold mean ± std (%)")
    print("representation       | ACC | AUC | SEN | PRE | F1")
    for row in seed_rows:
        print(
            f"{row['representation']:20s} | "
            + " | ".join(row[f"{metric}_display_pct"] for metric in METRIC_SOURCES)
        )


def print_cross_seed_metrics(rows):
    print("\nAcross-seed mean ± sample std of seed-level ten-fold means (%)")
    print("representation       | ACC | AUC | SEN | PRE | F1")
    for row in rows:
        print(
            f"{row['representation']:20s} | "
            + " | ".join(row[f"{metric}_display_pct"] for metric in METRIC_SOURCES)
        )


def run_multi_seed(args, run_single):
    root = _seed_directory(args)
    root.mkdir(parents=True, exist_ok=True)
    seed42_summary, seed42_directory = load_seed42_summary(args)
    seed_rows = summarize_seed(42, seed42_summary)
    print(
        f"\n{'=' * 18} seed=42 reused {'=' * 18}\n"
        f"source={seed42_directory.resolve()}\n"
        "No Initial EC training or ten-fold pipeline was rerun for seed 42."
    )
    print_seed_metrics(seed_rows)
    executed_seeds = []
    for seed in args.seeds:
        if seed == 42:
            continue
        run_args = configure_seed_run(args, seed, root)
        print(
            f"\n{'=' * 18} seed={seed} mode={args.multi_seed_mode} "
            f"{'=' * 18}\noutput={run_args.output_dir}\nInitial EC={run_args.ec_path}"
        )
        summary = run_single(run_args)
        executed_seeds.append(seed)
        current = summarize_seed(seed, summary)
        seed_rows.extend(current)
        print_seed_metrics(current)
    aggregate_rows = aggregate_seed_means(seed_rows)
    _write_csv(root / "seed_summaries.csv", seed_rows)
    _write_csv(root / "cross_seed_summary.csv", aggregate_rows)
    payload = {
        "dataset": args.dataset,
        "mode": args.multi_seed_mode,
        "seeds": list(args.seeds),
        "reused_seeds": [42],
        "executed_seeds": executed_seeds,
        "seed42_results_dir": str(seed42_directory.resolve()),
        "folds_per_seed": args.n_splits,
        "fold_std_ddof": 0,
        "cross_seed_std_ddof": 1,
        "shared_initial_ec": str(Path(args.ec_path).resolve()) if args.multi_seed_mode == "reuse_ec" else None,
        "seed_summaries": seed_rows,
        "cross_seed_summary": aggregate_rows,
    }
    (root / "multi_seed_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print_cross_seed_metrics(aggregate_rows)
    print(f"\nSaved multi-seed summary: {root.resolve()}")
    return payload
