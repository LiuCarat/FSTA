"""Extract CPAC-like filt_noglobal AAL90 time series from fMRIPrep output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_manifest(path: Path):
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            site, subject = line.split("\t", 1)
            rows[subject] = site
    return rows


def parse_site_subject(path: Path, manifest):
    for parent in path.parents:
        if parent.name.startswith("sub-"):
            subject = parent.name
            return manifest.get(subject, "unknown"), subject
    return "unknown", path.stem


def find_confounds(bold_path: Path):
    candidates = list(bold_path.parent.glob("*_desc-confounds_timeseries.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No confounds TSV next to {bold_path}")
    return candidates[0]


def find_json(bold_path: Path):
    candidates = [bold_path.with_name(bold_path.name.replace("_desc-preproc_bold.nii.gz", "_bold.json"))]
    candidates.extend(bold_path.parent.glob("*_bold.json"))
    return next((path for path in candidates if path.is_file()), None)


def read_tr(bold_path: Path, explicit_tr):
    if explicit_tr is not None:
        return float(explicit_tr)
    json_path = find_json(bold_path)
    if json_path is not None:
        try:
            value = json.loads(json_path.read_text(encoding="utf-8")).get("RepetitionTime")
            if value is not None:
                return float(value)
        except (OSError, ValueError, TypeError):
            pass
    raise ValueError(f"Cannot determine TR for {bold_path}; pass --tr")


def select_cpac_confounds(frame, fd_threshold, dvars_threshold):
    """Build CPAC-like filt_noglobal nuisance regressors without global signal."""
    import pandas as pd

    def columns(names):
        return [name for name in names if name in frame.columns]

    base = columns([
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
        "white_matter", "csf",
    ])
    motion = columns([
        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
        "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
        "trans_x_power2", "trans_y_power2", "trans_z_power2",
        "rot_x_power2", "rot_y_power2", "rot_z_power2",
        "trans_x_derivative1_power2", "trans_y_derivative1_power2",
        "trans_z_derivative1_power2", "rot_x_derivative1_power2",
        "rot_y_derivative1_power2", "rot_z_derivative1_power2",
    ])
    selected = list(dict.fromkeys(base + motion))
    if not selected:
        raise ValueError("fMRIPrep confounds contain no motion/WM/CSF regressors")

    confounds = frame[selected].apply(pd.to_numeric, errors="coerce")
    confounds = confounds.replace([np.inf, -np.inf], np.nan)
    confounds = confounds.fillna(confounds.median()).fillna(0.0)
    confounds = confounds.loc[:, confounds.nunique(dropna=False) > 1]

    fd = pd.to_numeric(frame.get("framewise_displacement"), errors="coerce")
    dvars = pd.to_numeric(frame.get("dvars"), errors="coerce")
    bad = np.zeros(len(frame), dtype=bool)
    if fd_threshold is not None and fd is not None:
        bad |= fd.fillna(0.0).to_numpy() > float(fd_threshold)
    if dvars_threshold is not None and dvars is not None:
        finite = dvars[np.isfinite(dvars)]
        if len(finite):
            threshold = float(dvars_threshold)
            if threshold <= 1.0:
                threshold = float(np.nanmedian(finite) + threshold * (np.nanpercentile(finite, 75) - np.nanpercentile(finite, 25)))
            bad |= dvars.fillna(0.0).to_numpy() > threshold
    sample_mask = np.flatnonzero(~bad)
    return confounds.to_numpy(dtype=np.float64), sample_mask, int(bad.sum())


def extract_one(bold_path, atlas_img, manifest, args, nib, resample_to_img, NiftiLabelsMasker, clean):
    import pandas as pd

    site, subject = parse_site_subject(bold_path, manifest)
    target_dir = args.output_root / site / subject.removeprefix("sub-")
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"{subject.removeprefix('sub-')}_aal_TCs.1D"
    if output.exists() and not args.overwrite:
        return f"exists {output}"

    image = nib.load(str(bold_path))
    if len(image.shape) != 4:
        return f"skip non-4D {bold_path}"
    tr = read_tr(bold_path, args.tr)
    confounds_path = find_confounds(bold_path)
    confound_frame = pd.read_csv(confounds_path, sep="\t")
    confounds, sample_mask, censored = select_cpac_confounds(
        confound_frame, args.fd_threshold, args.dvars_threshold
    )
    if len(sample_mask) < args.min_timepoints:
        return f"skip after scrubbing ({len(sample_mask)} points) {bold_path}"

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=False,
        detrend=False,
        strategy="mean",
    )
    values = masker.fit_transform(image)
    values = values[sample_mask]
    values = clean(
        values,
        t_r=tr,
        detrend=True,
        standardize=False,
        low_pass=args.high_cutoff,
        high_pass=args.low_cutoff,
        confounds=confounds[sample_mask],
        ensure_finite=True,
    )
    if values.shape[1] < args.roi_count:
        raise RuntimeError(f"Expected at least {args.roi_count} AAL labels, got {values.shape}")
    values = values[:, :args.roi_count]
    if values.shape[0] < args.min_timepoints:
        return f"skip short cleaned series ({values.shape[0]}) {bold_path}"
    np.savetxt(output, values, fmt="%.8g", delimiter=" ")
    return f"{site}/{subject}: {values.shape}, censored={censored} -> {output}"


def main():
    parser = argparse.ArgumentParser(description="CPAC-like filt_noglobal AAL90 extraction")
    parser.add_argument("--fmriprep-root", type=Path, default=Path("ABIDEII/derivatives/fmriprep"))
    parser.add_argument("--atlas", type=Path, required=True, help="AAL116 NIfTI in fMRIPrep MNI space")
    parser.add_argument("--output-root", type=Path, default=Path("dataset/ABIDE-II/AAL_TCs_filtfix"))
    parser.add_argument("--manifest", type=Path, default=Path("ABIDEII/bids/abideii_bids_manifest.tsv"))
    parser.add_argument("--space", default="MNI152NLin6Asym")
    parser.add_argument("--roi-count", type=int, default=90)
    parser.add_argument("--min-timepoints", type=int, default=78)
    parser.add_argument("--fd-threshold", type=float, default=0.2, help="FD scrubbing threshold in mm; CPAC-like default")
    parser.add_argument("--dvars-threshold", type=float, default=None, help="DVARS absolute threshold; omitted by default")
    parser.add_argument("--low-cutoff", type=float, default=0.01)
    parser.add_argument("--high-cutoff", type=float, default=0.1)
    parser.add_argument("--tr", type=float, default=None, help="Override TR if BOLD JSON lacks RepetitionTime")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import nibabel as nib
        import pandas as pd  # noqa: F401
        from nilearn.image import resample_to_img
        from nilearn.maskers import NiftiLabelsMasker
        from nilearn.signal import clean
    except ImportError as error:
        raise SystemExit("需要 nibabel、nilearn、pandas；建议在 nilearn/fmriprep 分析容器中运行") from error

    bold_files = sorted(args.fmriprep_root.glob(f"sub-*/**/*space-{args.space}*desc-preproc_bold.nii.gz"))
    if not bold_files:
        raise SystemExit(f"No fMRIPrep BOLD found under {args.fmriprep_root}")
    manifest = load_manifest(args.manifest)
    first_image = nib.load(str(bold_files[0]))
    atlas_img = resample_to_img(nib.load(str(args.atlas)), first_image, interpolation="nearest")
    atlas_data = np.asarray(atlas_img.get_fdata())
    atlas_data[(atlas_data < 1) | (atlas_data > 116)] = 0
    atlas_img = nib.Nifti1Image(atlas_data.astype(np.int16), atlas_img.affine, atlas_img.header)

    print("CPAC-like strategy: filt_noglobal")
    print(f"  nuisance: motion 24P + white_matter + csf")
    print(f"  band-pass: {args.low_cutoff}-{args.high_cutoff} Hz")
    print(f"  scrubbing: FD > {args.fd_threshold} mm")
    for index, bold_path in enumerate(bold_files, 1):
        try:
            message = extract_one(
                bold_path, atlas_img, manifest, args, nib,
                resample_to_img, NiftiLabelsMasker, clean,
            )
        except Exception as error:
            message = f"ERROR {bold_path}: {type(error).__name__}: {error}"
        print(f"[{index}/{len(bold_files)}] {message}")


if __name__ == "__main__":
    main()
