"""Build Graph-BEC AAL90 time series from fMRIPrep outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MOTION_COLUMNS = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
PHENOTYPE_COLUMNS = (
    "FILE_ID", "SUB_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX",
    "FIQ", "VIQ", "PIQ", "func_mean_fd", "func_fd_gt_0_2",
    "func_fd_gt_0_5", "func_dvars", "func_quality",
)


def load_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["subject"]: row["site"] for row in csv.DictReader(handle, delimiter="\t")}


def subject_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("sub-"):
            return parent.name
    raise ValueError(f"Could not find subject ID in {path}")


def find_confounds(bold_path: Path) -> Path:
    candidates = sorted(bold_path.parent.glob("*_desc-confounds_timeseries.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No confounds TSV next to {bold_path}")
    return candidates[0]


def read_tr(bold_path: Path, explicit_tr: float | None) -> float:
    if explicit_tr is not None:
        return explicit_tr
    candidates = [
        bold_path.with_name(bold_path.name.replace("_desc-preproc_bold.nii.gz", "_bold.json")),
        *sorted(bold_path.parent.glob("*_bold.json")),
    ]
    for path in candidates:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("RepetitionTime")
                if value is not None:
                    return float(value)
            except (OSError, ValueError, TypeError):
                continue
    raise ValueError(f"Cannot determine TR for {bold_path}; pass --tr")


def numeric_column(frame, name):
    import pandas as pd

    if name not in frame:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def build_motion_24p(frame):
    import pandas as pd

    motion = frame.reindex(columns=MOTION_COLUMNS).apply(pd.to_numeric, errors="coerce")
    motion = motion.interpolate(limit_direction="both").fillna(0.0)
    derivative = motion.diff().fillna(0.0)
    values = pd.concat((motion, derivative, motion.pow(2), derivative.pow(2)), axis=1)
    values.columns = [f"motion24_{index:02d}" for index in range(values.shape[1])]
    return values


def select_confounds(frame, acompcor_components: int):
    import pandas as pd

    selected = build_motion_24p(frame)
    acompcor = sorted(
        (column for column in frame.columns if column.startswith("a_comp_cor_")),
        key=lambda value: (
            0,
            int(value.rsplit("_", 1)[-1]),
        )
        if value.rsplit("_", 1)[-1].isdigit()
        else (1, value),
    )[:acompcor_components]
    if acompcor:
        components = frame[acompcor].apply(pd.to_numeric, errors="coerce")
        components = components.interpolate(limit_direction="both").fillna(0.0)
        selected = pd.concat((selected, components), axis=1)
    selected = selected.loc[:, selected.nunique(dropna=False) > 1]
    if selected.empty:
        raise ValueError("No usable motion or aCompCor confounds found")
    return selected.to_numpy(dtype=np.float64)


def qc_values(frame):
    fd = numeric_column(frame, "framewise_displacement").fillna(0.0).to_numpy(dtype=float)
    dvars = numeric_column(frame, "dvars").to_numpy(dtype=float)
    finite_dvars = dvars[np.isfinite(dvars)]
    mean_dvars = float(np.mean(finite_dvars)) if len(finite_dvars) else float("nan")
    return {
        "func_mean_fd": float(np.mean(fd)),
        "func_fd_gt_0_2": float(np.mean(fd > 0.2)),
        "func_fd_gt_0_5": float(np.mean(fd > 0.5)),
        "func_dvars": mean_dvars,
        "func_quality": float(np.mean(fd <= 0.2)),
    }


def load_phenotype(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {row["FILE_ID"].strip(): row for row in rows if row.get("FILE_ID", "").strip()}


def write_phenotype(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHENOTYPE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())


def extract_one(bold_path, atlas_path, output_path, tr, args, manifest, nib, resample_to_img, NiftiLabelsMasker, clean):
    import pandas as pd

    subject = subject_from_path(bold_path)
    site = manifest.get(subject, "unknown")
    image = nib.load(str(bold_path))
    if len(image.shape) != 4:
        raise ValueError(f"Expected 4D BOLD, got {image.shape}")
    if image.shape[-1] < args.min_timepoints:
        raise ValueError(f"Only {image.shape[-1]} time points; need {args.min_timepoints}")

    confounds_path = find_confounds(bold_path)
    confound_frame = pd.read_csv(confounds_path, sep="\t")
    confounds = select_confounds(confound_frame, args.acompcor_components)
    qc = qc_values(confound_frame)

    atlas = resample_to_img(nib.load(str(atlas_path)), image, interpolation="nearest")
    labels = np.asarray(atlas.get_fdata(), dtype=np.int16)
    labels[(labels < 1) | (labels > 116)] = 0
    label_count = len(np.unique(labels[labels > 0]))
    if label_count != 116:
        raise ValueError(f"Atlas has {label_count} labels after resampling; expected 116")
    atlas = nib.Nifti1Image(labels, atlas.affine, atlas.header)

    masker = NiftiLabelsMasker(labels_img=atlas, standardize=False, detrend=False, strategy="mean")
    values = masker.fit_transform(image)
    if values.shape[1] != 116:
        raise ValueError(f"Expected 116 extracted AAL labels, got {values.shape}")
    cleaned = clean(
        values,
        confounds=confounds,
        t_r=tr,
        detrend=True,
        standardize=False,
        low_pass=args.high_cutoff,
        high_pass=args.low_cutoff,
    )
    cleaned = np.asarray(cleaned[:, :90], dtype=np.float32)
    if not np.isfinite(cleaned).all():
        raise ValueError("Non-finite values after temporal denoising")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, cleaned, fmt="%.8g")
    return subject, site, qc, cleaned.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmriprep-root", type=Path, default=Path("ABIDEII/derivatives/fmriprep"))
    parser.add_argument("--atlas", type=Path, required=True, help="AAL116 atlas NIfTI")
    parser.add_argument("--phenotype", type=Path, default=Path("dataset/ABIDE-II/ABIDEII_phenotype_graphbec.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("ABIDEII/bids/abideii_bids_manifest.tsv"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/ABIDE-II/cpac/filt_noglobal"))
    parser.add_argument("--space", default="MNI152NLin6Asym")
    parser.add_argument("--low-cutoff", type=float, default=0.01)
    parser.add_argument("--high-cutoff", type=float, default=0.1)
    parser.add_argument("--acompcor-components", type=int, default=5)
    parser.add_argument("--min-timepoints", type=int, default=78)
    parser.add_argument("--tr", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import nibabel as nib
        import pandas as pd  # noqa: F401
        from nilearn.image import resample_to_img
        from nilearn.maskers import NiftiLabelsMasker
        from nilearn.signal import clean
    except ImportError as error:
        raise SystemExit("需要 nibabel、nilearn、pandas；请在分析环境或容器中运行") from error

    if not args.atlas.is_file():
        raise SystemExit(f"Atlas not found: {args.atlas}")
    bold_files = sorted(args.fmriprep_root.glob(f"sub-*/**/*space-{args.space}*desc-preproc_bold.nii.gz"))
    if not bold_files:
        raise SystemExit(f"No fMRIPrep BOLD found under {args.fmriprep_root}")
    by_subject = {}
    for path in bold_files:
        by_subject.setdefault(subject_from_path(path), path)
    manifest = load_manifest(args.manifest)
    rows = load_phenotype(args.phenotype)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Temporal strategy: Friston-24 + aCompCor + linear detrend + 0.01-0.1 Hz")
    print("Global signal regression: disabled")
    print("Frame deletion/scrubbing: disabled; FD/DVARS saved as subject-level QC")
    print(f"Subjects: {len(by_subject)}")
    for index, (subject, bold_path) in enumerate(sorted(by_subject.items()), 1):
        output_path = args.output_root / f"{subject}_rois_aal.1D"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(by_subject)}] exists {output_path}")
            continue
        try:
            shape = None
            subject, site, qc, shape = extract_one(
                bold_path, args.atlas, output_path, read_tr(bold_path, args.tr), args,
                manifest, nib, resample_to_img, NiftiLabelsMasker, clean,
            )
            row = rows.get(subject)
            if row is None:
                print(f"[{index}/{len(by_subject)}] WARNING phenotype missing: {subject}")
                continue
            row["SITE_ID"] = site if site != "unknown" else row.get("SITE_ID", "")
            row.update({key: f"{value:.8g}" for key, value in qc.items()})
            print(f"[{index}/{len(by_subject)}] {subject}: {shape[0]}x{shape[1]}")
        except Exception as error:
            print(f"[{index}/{len(by_subject)}] ERROR {subject}: {type(error).__name__}: {error}")
    write_phenotype(args.phenotype, rows)
    print(f"Saved phenotype/QC: {args.phenotype.resolve()}")
    print(f"Saved ROI files: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
