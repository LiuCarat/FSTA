#!/usr/bin/env python3
"""Measure multivariate QC variance explained across an eta sweep.

For every directed off-diagonal edge e, fit

    A[i, e] = beta0 + beta1 * FD[i] + beta2 * DVARS[i]
              + beta3 * Quality[i] + epsilon[i]

and report the mean edge-wise R^2 across the connectome.

Archives are supplied as repeated ``--archive ETA=PATH`` arguments because the
current NPZ writer does not store eta in the archive metadata.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

DEFAULT_QC_COLUMNS = ("func_mean_fd", "func_dvars", "func_quality")
DEFAULT_EC_KEY = "qc_refined_ec"


def parse_archive_spec(value: str) -> tuple[float, Path]:
    try:
        eta_text, path_text = value.split("=", 1)
        eta = float(eta_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "archive must use the format ETA=PATH, for example 0.10=result.npz"
        ) from error
    if not np.isfinite(eta) or eta < 0:
        raise argparse.ArgumentTypeError("eta must be a finite non-negative number")
    path = Path(path_text)
    return eta, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        action="append",
        required=True,
        type=parse_archive_spec,
        metavar="ETA=PATH",
        help="EC archive for one eta value; repeat for every eta",
    )
    parser.add_argument("--phenotype-csv", type=Path, required=True)
    parser.add_argument("--subject-id-column", default="FILE_ID")
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Phenotype delimiter; use '\\t' for TSV files",
    )
    parser.add_argument(
        "--qc-columns",
        nargs=3,
        default=list(DEFAULT_QC_COLUMNS),
        metavar=("FD", "DVARS", "QUALITY"),
    )
    parser.add_argument("--ec-key", default=DEFAULT_EC_KEY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="Also save an eta-R2 PNG")
    parser.add_argument(
        "--include-diagonal",
        action="store_true",
        help="Include self-edges; omitted by default",
    )
    return parser.parse_args()


def _parse_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return np.nan
    return parsed if np.isfinite(parsed) else np.nan


def _find_row(rows: dict[str, dict[str, str]], subject_id: object) -> dict[str, str] | None:
    subject = str(subject_id).strip()
    return rows.get(subject) or rows.get(subject.rsplit("/", 1)[-1])


def load_qc(
    path: Path,
    subject_ids: np.ndarray,
    identifier: str,
    columns: tuple[str, ...],
    delimiter: str = ",",
) -> np.ndarray:
    if delimiter == r"\t":
        delimiter = "\t"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = [field.strip() for field in reader.fieldnames or []]
        missing = {identifier, *columns} - set(fieldnames)
        if missing:
            raise ValueError(f"QC file is missing columns: {sorted(missing)}")
        rows = {
            str(row.get(identifier, "")).strip(): {
                str(key).strip(): str(value or "").strip()
                for key, value in row.items()
            }
            for row in reader
            if str(row.get(identifier, "")).strip()
        }

    values = np.full((len(subject_ids), len(columns)), np.nan, dtype=np.float64)
    for index, subject_id in enumerate(subject_ids):
        row = _find_row(rows, subject_id)
        if row is None:
            raise ValueError(f"Could not match subject {subject_id!r} in {path}")
        for column_index, column in enumerate(columns):
            values[index, column_index] = _parse_float(row.get(column))
    return values


def load_archive(path: Path, ec_key: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"EC archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {ec_key, "subject_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        ec = np.asarray(archive[ec_key], dtype=np.float64)
        subject_ids = np.asarray(archive["subject_ids"]).astype(str).reshape(-1)
    if ec.ndim != 3 or ec.shape[1] != ec.shape[2]:
        raise ValueError(f"Expected EC shape [subjects, nodes, nodes], got {ec.shape}")
    if len(subject_ids) != len(ec):
        raise ValueError("subject_ids and EC subject dimension do not match")
    if not np.isfinite(ec).all():
        raise ValueError(f"EC archive contains NaN or infinite values: {path}")
    return ec, subject_ids


def build_design(qc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(qc).all(axis=1)
    if valid.sum() < 5:
        raise ValueError("At least five subjects with complete QC values are required")
    complete = qc[valid]
    # R^2 is invariant to feature scaling; median-center only improves numerical stability.
    centered = complete - np.mean(complete, axis=0, keepdims=True)
    design = np.column_stack((np.ones(len(centered)), centered))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("QC design is rank deficient; FD/DVARS/Quality are not independent")
    return design, valid


def edgewise_r2(ec: np.ndarray, design: np.ndarray, valid: np.ndarray, include_diagonal: bool) -> np.ndarray:
    values = ec[valid].reshape(valid.sum(), -1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    centered = values - values.mean(axis=0, keepdims=True)
    sse = np.sum(residual * residual, axis=0)
    sst = np.sum(centered * centered, axis=0)
    r2 = np.full(values.shape[1], np.nan, dtype=np.float64)
    nonconstant = sst > np.finfo(np.float64).eps
    r2[nonconstant] = 1.0 - sse[nonconstant] / sst[nonconstant]
    r2 = np.clip(r2, 0.0, 1.0)
    matrix = r2.reshape(ec.shape[1], ec.shape[2])
    if not include_diagonal:
        matrix[np.eye(matrix.shape[0], dtype=bool)] = np.nan
    return matrix


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["eta", "n_subjects", "n_edges", "mean_r2", "median_r2", "std_r2", "min_r2", "max_r2"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_edge_table(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["eta", "source_index", "target_index", "r2"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, summary: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib to be installed") from error
    eta = [float(row["eta"]) for row in summary]
    mean_r2 = [float(row["mean_r2"]) for row in summary]
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(eta, mean_r2, marker="o", linewidth=2)
    axis.set_xlabel(r"$\eta$")
    axis.set_ylabel(r"Mean edge-wise variance explained by QC ($R^2$)")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    archive_specs = sorted(args.archive, key=lambda item: item[0])
    first_ec, first_subject_ids = load_archive(archive_specs[0][1], args.ec_key)
    qc = load_qc(
        args.phenotype_csv,
        first_subject_ids,
        args.subject_id_column,
        tuple(args.qc_columns),
        args.delimiter,
    )
    design, valid = build_design(qc)

    summary = []
    edge_rows = []
    for eta, path in archive_specs:
        ec, subject_ids = load_archive(path, args.ec_key)
        if ec.shape != first_ec.shape or not np.array_equal(subject_ids, first_subject_ids):
            raise ValueError("All eta archives must have identical shape and subject order")
        r2 = edgewise_r2(ec, design, valid, args.include_diagonal)
        finite = r2[np.isfinite(r2)]
        summary.append({
            "eta": eta,
            "n_subjects": int(valid.sum()),
            "n_edges": int(finite.size),
            "mean_r2": float(np.mean(finite)),
            "median_r2": float(np.median(finite)),
            "std_r2": float(np.std(finite)),
            "min_r2": float(np.min(finite)),
            "max_r2": float(np.max(finite)),
        })
        for source, target in zip(*np.where(np.isfinite(r2))):
            edge_rows.append({
                "eta": eta,
                "source_index": int(source),
                "target_index": int(target),
                "r2": float(r2[source, target]),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / "multivariate_qc_r2_summary.csv", summary)
    write_edge_table(args.output_dir / "multivariate_qc_r2_edges.csv", edge_rows)
    if args.plot:
        plot_summary(args.output_dir / "multivariate_qc_r2_vs_eta.png", summary)
    for row in summary:
        print(f"eta={row['eta']:.4g} mean_R2={row['mean_r2']:.6f} n={row['n_subjects']}")
    print(f"Saved: {(args.output_dir / 'multivariate_qc_r2_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
