#!/usr/bin/env python3
"""Visualize BD-Core20 signals before and after the frozen denoising pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import index_img, resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from scipy.signal import welch

try:
    from .extract_bd_core20 import find_func_dir
    from .paths import BDCORE20_DIR, FMRIPREP_DIR, SPACE
    from .preprocess import load_bdcore20_atlas
except ImportError:
    from extract_bd_core20 import find_func_dir
    from paths import BDCORE20_DIR, FMRIPREP_DIR, SPACE
    from preprocess import load_bdcore20_atlas


def zscore_columns(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=0, keepdims=True)
    standard_deviations = np.std(values, axis=0, ddof=1, keepdims=True)
    standard_deviations[standard_deviations == 0] = 1.0
    return (values - means) / standard_deviations


def extract_unprocessed(bold_path: Path, atlas_img, roi_names: list[str]) -> np.ndarray:
    bold_img = nib.load(bold_path)
    atlas_resampled = resample_to_img(
        atlas_img,
        index_img(bold_img, 0),
        interpolation="nearest",
    )
    masker = NiftiLabelsMasker(
        labels_img=atlas_resampled,
        labels=["Background"] + roi_names,
        resampling_target=None,
        standardize=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        smoothing_fwhm=None,
        memory=None,
        memory_level=0,
        verbose=0,
        reports=False,
    )
    return masker.fit_transform(bold_img)


def power_summary(values: np.ndarray, repetition_time: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    frequencies, power = welch(
        values,
        fs=1.0 / repetition_time,
        nperseg=min(128, values.shape[0]),
        axis=0,
    )
    mean_power = np.mean(power, axis=1)
    total = float(np.trapezoid(mean_power, frequencies))
    normalized = mean_power / total if total > 0 else mean_power

    def fraction(mask: np.ndarray) -> float:
        if np.sum(mask) < 2 or total <= 0:
            return 0.0
        return float(np.trapezoid(mean_power[mask], frequencies[mask]) / total)

    summary = {
        "power_below_0.01_hz": fraction(frequencies < 0.01),
        "power_0.01_to_0.1_hz": fraction((frequencies >= 0.01) & (frequencies <= 0.1)),
        "power_above_0.1_hz": fraction(frequencies > 0.1),
    }
    return frequencies, normalized, summary


def mean_absolute_fd_correlation(values: np.ndarray, framewise_displacement: np.ndarray) -> float:
    valid = np.isfinite(framewise_displacement)
    if np.sum(valid) < 3 or np.std(framewise_displacement[valid]) == 0:
        return float("nan")
    correlations = []
    for column in range(values.shape[1]):
        signal = values[valid, column]
        if np.std(signal) == 0:
            continue
        correlations.append(abs(float(np.corrcoef(signal, framewise_displacement[valid])[0, 1])))
    return float(np.mean(correlations)) if correlations else float("nan")


def mean_lag_one_autocorrelation(values: np.ndarray) -> float:
    correlations = []
    for column in range(values.shape[1]):
        first = values[:-1, column]
        second = values[1:, column]
        if np.std(first) == 0 or np.std(second) == 0:
            continue
        correlations.append(float(np.corrcoef(first, second)[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="10159")
    parser.add_argument("--group", default="HC", choices=["BD", "HC", "SCHZ", "ADHD"])
    parser.add_argument("--atlas-dir", type=Path, default=BDCORE20_DIR)
    parser.add_argument("--fmriprep-dir", type=Path, default=FMRIPREP_DIR)
    parser.add_argument("--processed-dir", type=Path, default=BDCORE20_DIR)
    parser.add_argument("--outdir", type=Path, default=BDCORE20_DIR / "qc")
    args = parser.parse_args()

    subject_id = args.subject.replace("sub-", "")
    atlas_img, roi_names = load_bdcore20_atlas(args.atlas_dir.resolve())
    func_dir, detected_group = find_func_dir(args.fmriprep_dir.resolve(), subject_id, args.group)
    bold_files = sorted(func_dir.glob(f"*_space-{SPACE}*_desc-preproc_bold.nii.gz"))
    confounds_files = sorted(func_dir.glob("*_desc-confounds_timeseries.tsv"))
    if not bold_files or not confounds_files:
        raise FileNotFoundError(f"sub-{subject_id}: missing BOLD or confounds")

    bold_path = bold_files[0]
    confounds_path = confounds_files[0]
    processed_path = args.processed_dir.resolve() / detected_group / f"sub-{subject_id}.txt"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed BD-Core20 time series: {processed_path}")

    json_path = Path(str(bold_path).replace(".nii.gz", ".json"))
    with json_path.open() as handle:
        repetition_time = float(json.load(handle)["RepetitionTime"])

    unprocessed = extract_unprocessed(bold_path, atlas_img, roi_names)
    processed = np.loadtxt(processed_path, delimiter="\t", skiprows=1)
    if unprocessed.shape != processed.shape:
        raise RuntimeError(f"Shape mismatch: unprocessed={unprocessed.shape}, processed={processed.shape}")

    unprocessed_display = zscore_columns(unprocessed)
    processed_display = zscore_columns(processed)
    confounds = pd.read_csv(confounds_path, sep="\t")
    framewise_displacement = pd.to_numeric(
        confounds.get("framewise_displacement", pd.Series(np.nan, index=confounds.index)),
        errors="coerce",
    ).to_numpy()

    raw_frequency, raw_power, raw_power_summary = power_summary(unprocessed_display, repetition_time)
    clean_frequency, clean_power, clean_power_summary = power_summary(processed_display, repetition_time)
    raw_fd_correlation = mean_absolute_fd_correlation(unprocessed_display, framewise_displacement)
    clean_fd_correlation = mean_absolute_fd_correlation(processed_display, framewise_displacement)
    raw_lag_one = mean_lag_one_autocorrelation(unprocessed_display)
    clean_lag_one = mean_lag_one_autocorrelation(processed_display)

    output_dir = args.outdir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"sub-{subject_id}_bdcore20_preprocessing_comparison"

    metrics = pd.DataFrame(
        [
            {
                "stage": "unprocessed",
                "mean_absolute_fd_correlation": raw_fd_correlation,
                "mean_lag1_autocorrelation": raw_lag_one,
                **raw_power_summary,
            },
            {
                "stage": "processed",
                "mean_absolute_fd_correlation": clean_fd_correlation,
                "mean_lag1_autocorrelation": clean_lag_one,
                **clean_power_summary,
            },
        ]
    )
    metrics.to_csv(output_dir / f"{output_stem}.tsv", sep="\t", index=False, float_format="%.6f")

    time_seconds = np.arange(unprocessed.shape[0]) * repetition_time
    selected_indices = [0, 6, 8, 10, 18]
    selected_colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))

    figure = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15])
    raw_axis = figure.add_subplot(grid[0, :])
    clean_axis = figure.add_subplot(grid[1, :])
    trace_axis = figure.add_subplot(grid[2, 0])
    power_axis = figure.add_subplot(grid[2, 1])

    heatmap_options = dict(aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-3, vmax=3)
    raw_image = raw_axis.imshow(unprocessed_display.T, **heatmap_options)
    raw_axis.set_title("Before denoising: ROI means only (z-scored for visualization)")
    raw_axis.set_ylabel("BD-Core20 ROI")
    raw_axis.set_yticks(np.arange(20))
    raw_axis.set_yticklabels([f"{index + 1:02d} {name}" for index, name in enumerate(roi_names)], fontsize=7)

    clean_axis.imshow(processed_display.T, **heatmap_options)
    clean_axis.set_title("After denoising: 24 motion + 5 aCompCor + outliers + 0.01–0.1 Hz + z-score")
    clean_axis.set_xlabel("Time point")
    clean_axis.set_ylabel("BD-Core20 ROI")
    clean_axis.set_yticks(np.arange(20))
    clean_axis.set_yticklabels([f"{index + 1:02d} {name}" for index, name in enumerate(roi_names)], fontsize=7)
    figure.colorbar(raw_image, ax=[raw_axis, clean_axis], label="Standardized amplitude", shrink=0.75)

    vertical_offset = 7.0
    for row, (roi_index, color) in enumerate(zip(selected_indices, selected_colors)):
        offset = (len(selected_indices) - row - 1) * vertical_offset
        trace_axis.plot(time_seconds, unprocessed_display[:, roi_index] + offset, color="0.65", linewidth=1.0)
        trace_axis.plot(time_seconds, processed_display[:, roi_index] + offset, color=color, linewidth=1.25)
        trace_axis.text(time_seconds[-1] + repetition_time, offset, roi_names[roi_index], color=color, va="center", fontsize=8)
    trace_axis.plot([], [], color="0.65", label="Before")
    trace_axis.plot([], [], color="tab:blue", label="After")
    trace_axis.set_title("Representative ROI time series")
    trace_axis.set_xlabel("Time (s)")
    trace_axis.set_yticks([])
    trace_axis.legend(loc="upper right", frameon=False)
    trace_axis.grid(alpha=0.15)

    power_axis.plot(raw_frequency, raw_power, color="0.55", linewidth=2, label="Before")
    power_axis.plot(clean_frequency, clean_power, color="tab:blue", linewidth=2, label="After")
    power_axis.axvspan(0.01, 0.1, color="tab:green", alpha=0.12, label="Pass band 0.01–0.1 Hz")
    power_axis.axvline(0.01, color="tab:green", linestyle="--", linewidth=1)
    power_axis.axvline(0.1, color="tab:green", linestyle="--", linewidth=1)
    power_axis.set_xlim(0, 0.25)
    power_axis.set_title("Mean normalized power spectrum across 20 ROIs")
    power_axis.set_xlabel("Frequency (Hz)")
    power_axis.set_ylabel("Normalized power density")
    power_axis.legend(frameon=False)
    power_axis.grid(alpha=0.2)

    outside_before = raw_power_summary["power_below_0.01_hz"] + raw_power_summary["power_above_0.1_hz"]
    outside_after = clean_power_summary["power_below_0.01_hz"] + clean_power_summary["power_above_0.1_hz"]
    figure.suptitle(
        f"BD-Core20 preprocessing comparison — sub-{subject_id} ({detected_group})\n"
        f"Mean |corr(FD)|: {raw_fd_correlation:.3f} → {clean_fd_correlation:.3f}    "
        f"Out-of-band power: {outside_before:.1%} → {outside_after:.1%}",
        fontsize=15,
    )
    figure.savefig(output_dir / f"{output_stem}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved comparison figure: {output_dir / f'{output_stem}.png'}")
    print(f"Saved metrics: {output_dir / f'{output_stem}.tsv'}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
