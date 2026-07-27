#!/usr/bin/env python3
"""Build and validate the custom BD-Core20 atlas from user-supplied AAL3 files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes

ROI_NAMES = [
    "vmPFC_mOFC_L", "vmPFC_mOFC_R", "dlPFC_L", "dlPFC_R",
    "vlPFC_L", "vlPFC_R", "Anterior_Insula_L", "Anterior_Insula_R",
    "sgACC_L", "sgACC_R", "Amygdala_L", "Amygdala_R",
    "NAcc_L", "NAcc_R", "Caudate_L", "Caudate_R", "Putamen_L",
    "Putamen_R", "Thalamus_L", "Thalamus_R",
]

NETWORKS = {
    "vmPFC_mOFC": "Default/valuation", "dlPFC": "Executive control",
    "vlPFC": "Executive control", "Anterior_Insula": "Salience/interoception",
    "sgACC": "Limbic/mood", "Amygdala": "Limbic/mood", "NAcc": "Reward",
    "Caudate": "Striatal", "Putamen": "Striatal", "Thalamus": "Thalamic",
}

@dataclass(frozen=True)
class Label:
    value: int
    name: str


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def hemisphere(name: str) -> str | None:
    n = norm(name)
    if re.search(r"(?:^|[ _-])(left|l)(?:$|[ _-])", name, re.I) or n.endswith("l"):
        return "L"
    if re.search(r"(?:^|[ _-])(right|r)(?:$|[ _-])", name, re.I) or n.endswith("r"):
        return "R"
    return None


def read_labels(path: Path) -> list[Label]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    suffix = path.suffix.lower()
    labels: list[Label] = []
    if suffix == ".xml":
        root = ET.fromstring(text)
        for node in root.iter():
            attrs = {str(k).lower(): str(v).strip() for k, v in node.attrib.items()}
            value = next((attrs[k] for k in ("index", "id", "value", "label") if k in attrs), None)
            name = next((attrs[k] for k in ("name", "label", "region", "title", "text") if k in attrs), None)
            if value is None:
                value = next((node.findtext(k) for k in ("index", "id", "value") if node.findtext(k)), None)
            if name is None:
                name = next((node.findtext(k) for k in ("name", "region", "title", "text") if node.findtext(k)), None)
            if name is None and node.text and node.text.strip():
                name = node.text.strip()
            if value is not None and name is not None and re.fullmatch(r"[-+]?\d+", value.strip()):
                labels.append(Label(int(value), name.strip()))
    else:
        rows = list(csv.reader(text.splitlines(), delimiter="\t"))
        if len(rows) == 1 and "," in text:
            rows = list(csv.reader(text.splitlines()))
        for row in rows:
            if len(row) < 2:
                continue
            value = next((x.strip() for x in row[:3] if re.fullmatch(r"[-+]?\d+", x.strip())), None)
            if value is None:
                continue
            value_i = int(value)
            name_candidates = [x.strip() for x in row if x.strip() and not re.fullmatch(r"[-+]?\d+", x.strip())]
            if name_candidates:
                labels.append(Label(value_i, name_candidates[-1]))
    unique = {label.value: label for label in labels if label.value != 0}
    if not unique:
        raise ValueError(f"未能从标签文件解析任何整数标签: {path}")
    return sorted(unique.values(), key=lambda x: x.value)


def show_candidates(title: str, candidates: list[Label]) -> None:
    print(f"[Candidate] {title}: {len(candidates)} 个候选标签")
    for item in candidates:
        print(f"  value={item.value}\tname={item.name}")


def resolve(label_set: list[Label], title: str, predicate, expected: int = 1) -> list[Label]:
    candidates = [label for label in label_set if predicate(label.name)]
    if len(candidates) != expected:
        show_candidates(title, candidates)
        raise RuntimeError(f"{title} 解析到 {len(candidates)} 个标签，期望 {expected}；拒绝静默选择。")
    return candidates



def resolve_regions(labels: list[Label]) -> dict[str, list[Label]]:
    result: dict[str, list[Label]] = {}
    for side in ("L", "R"):
        suffix = f"_{side}"
        result[f"vmPFC_mOFC{suffix}"] = resolve(
            labels, f"vmPFC_mOFC_{side}",
            lambda n, s=side: hemisphere(n) == s and "med" in norm(n) and "orb" in norm(n)
            and "rect" not in norm(n),
        )
        result[f"dlPFC{suffix}"] = resolve(
            labels, f"dlPFC_{side}",
            lambda n, s=side: hemisphere(n) == s and (
                norm(n).startswith("frontalmid") or "middlefrontal" in norm(n)
            ) and not any(x in norm(n) for x in ("orb", "supp", "med")),
        )
        # This intentionally requires exactly one label per side for each component.
        vl_candidates = [label for label in labels if hemisphere(label.name) == side and any(
            token in norm(label.name) for token in ("frontalinfoper", "frontalinftri", "inferiorfrontalop", "inferiorfrontaltri")
        )]
        show_candidates(f"vlPFC_{side} (frontal inferior opercular/triangular)", vl_candidates)
        if len(vl_candidates) != 2:
            raise RuntimeError(f"vlPFC_{side} 必须解析出恰好两个候选标签（盖部+三角部），实际 {len(vl_candidates)}。")
        result[f"vlPFC{suffix}"] = vl_candidates
        result[f"Anterior_Insula{suffix}"] = resolve(
            labels, f"Anterior_Insula_{side}", lambda n, s=side: hemisphere(n) == s and norm(n).endswith("insula" + s.lower()),
        )
        result[f"sgACC{suffix}"] = resolve(
            labels, f"sgACC_{side}", lambda n, s=side: hemisphere(n) == s and any(x in norm(n) for x in ("cingulumsubgenual", "subgenualanteriorcingulate", "accsub", "sgacc")),
        )
        direct = {
            f"Amygdala{suffix}": ("amygdala",),
            f"NAcc{suffix}": ("accumbens", "nucleusaccumbens", "nacc"),
            f"Caudate{suffix}": ("caudate",),
            f"Putamen{suffix}": ("putamen",),
        }
        for roi, tokens in direct.items():
            result[roi] = resolve(
                labels, roi, lambda n, s=side, ts=tokens: hemisphere(n) == s and any(t in norm(n) for t in ts),
            )
        result[f"Thalamus{suffix}"] = [
            label for label in labels if hemisphere(label.name) == side and "thal" in norm(label.name)
        ]
        if not result[f"Thalamus{suffix}"]:
            show_candidates(f"Thalamus_{side}", [])
            raise RuntimeError(f"Thalamus_{side} 未找到任何按名称和半球匹配的标签。")
    return result


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def orientation(img: nib.spatialimages.SpatialImage) -> str:
    return "".join(aff2axcodes(img.affine))


def plot_qc(img: nib.spatialimages.SpatialImage, outdir: Path) -> None:
    try:
        from nilearn import plotting
        plotting.plot_roi(img, title="BD-Core20 ROI atlas (orthographic)", draw_cross=True, colorbar=True).savefig(outdir / "BD_Core20_qc_ortho.png", dpi=180, bbox_inches="tight")
        plotting.plot_glass_brain(img, title="BD-Core20 ROI atlas (glass brain)", display_mode="lyrz", colorbar=True).savefig(outdir / "BD_Core20_qc_glass.png", dpi=180, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception as exc:
        raise RuntimeError(f"QC 图片生成失败（需要 nilearn/matplotlib）: {exc}") from exc


def build(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    atlas_img = nib.load(args.aal3_atlas)
    atlas_data = np.asanyarray(atlas_img.dataobj)
    if atlas_data.ndim != 3:
        raise ValueError(f"AAL3 图谱必须是 3D 整数标签图，实际 shape={atlas_data.shape}")
    if not np.all(np.isfinite(atlas_data)) or not np.allclose(atlas_data, np.round(atlas_data)):
        raise ValueError("AAL3 图谱包含非有限值或非整数标签")
    atlas_data = np.round(atlas_data).astype(np.int32)
    labels = read_labels(args.aal3_labels)
    label_by_value = {label.value: label for label in labels}
    regions = resolve_regions(labels)
    reference_img = nib.load(args.reference_bold)
    reference_3d = nib.Nifti1Image(np.zeros(reference_img.shape[:3], dtype=np.uint8), reference_img.affine, reference_img.header)

    source_masks: dict[str, np.ndarray] = {}
    target_specs: dict[str, tuple[list[Label], str]] = {}
    for roi_name in ROI_NAMES:
        selected = regions[roi_name]
        rule = "union of source labels"
        if roi_name.startswith("Anterior_Insula"):
            coords = nib.affines.apply_affine(atlas_img.affine, np.indices(atlas_data.shape).reshape(3, -1).T)
            mask = np.isin(atlas_data, [x.value for x in selected]).reshape(-1) & (coords[:, 1] > 0)
            source_masks[roi_name] = mask.reshape(atlas_data.shape)
            rule = "complete ipsilateral insula; retain only voxels with affine-derived MNI y > 0"
        else:
            source_masks[roi_name] = np.isin(atlas_data, [x.value for x in selected])
        target_specs[roi_name] = (selected, rule)

    out_data = np.zeros(atlas_data.shape, dtype=np.int16)
    overlap = np.zeros(atlas_data.shape, dtype=np.int16)
    for index, roi_name in enumerate(ROI_NAMES, 1):
        mask = source_masks[roi_name]
        overlap += mask.astype(np.int16)
        if not mask.any():
            raise RuntimeError(f"ROI {index} {roi_name} 在 AAL3 原始空间为空")
        out_data[mask] = index
    if np.any(overlap > 1):
        raise RuntimeError(f"存在未报告的 ROI 重叠体素: {int(np.sum(overlap > 1))}")

    atlas_out = nib.Nifti1Image(out_data, atlas_img.affine, atlas_img.header.copy())
    atlas_out.set_data_dtype(np.int16)
    bold_resampled = __import__("nilearn.image", fromlist=["resample_to_img"]).resample_to_img(atlas_out, reference_3d, interpolation="nearest")
    bold_data = np.rint(np.asanyarray(bold_resampled.dataobj)).astype(np.int16)
    bold_out = nib.Nifti1Image(bold_data, reference_3d.affine, reference_3d.header.copy())
    bold_out.set_data_dtype(np.int16)
    if np.any((bold_data < 0) | (bold_data > 20)) or set(np.unique(bold_data)) - {0, *range(1, 21)}:
        raise RuntimeError("重采样图谱包含非法标签值")

    labels_rows, mapping_rows, voxel_rows = [], [], []
    for index, roi_name in enumerate(ROI_NAMES, 1):
        selected, rule = target_specs[roi_name]
        side = hemisphere(roi_name)
        base = roi_name.rsplit("_", 1)[0]
        labels_rows.append({"index": index, "name": roi_name, "hemisphere": side, "network": NETWORKS[base], "source_atlas": "AAL3 (user supplied)", "construction": rule, "source_labels": "; ".join(x.name for x in selected)})
        for source in selected:
            mapping_rows.append({"source_value": source.value, "source_label": source.name, "bd_core20_index": index, "bd_core20_name": roi_name, "hemisphere": side, "construction": rule})
        raw_count = int(np.sum(out_data == index))
        bold_count = int(np.sum(bold_data == index))
        voxel_rows.append({"index": index, "name": roi_name, "aal3_voxel_count": raw_count, "reference_bold_voxel_count": bold_count, "coverage": f"{bold_count / raw_count:.6f}"})
        if bold_count == 0:
            raise RuntimeError(f"ROI {index} {roi_name} 在 reference BOLD 空间为空")
        if bold_count < 3:
            print(f"\033[1;33mWARNING: ROI {index} {roi_name} has only {bold_count} BOLD voxels.\033[0m", file=sys.stderr)
        world = nib.affines.apply_affine(atlas_img.affine, np.argwhere(out_data == index))
        median_x = float(np.median(world[:, 0]))
        if (side == "L" and median_x >= 0) or (side == "R" and median_x <= 0):
            raise RuntimeError(f"左右半球检查失败: {roi_name}, median MNI x={median_x:.3f}")

    write_tsv(outdir / "BD_Core20_labels.tsv", labels_rows, list(labels_rows[0]))
    write_tsv(outdir / "BD_Core20_mapping.tsv", mapping_rows, list(mapping_rows[0]))
    write_tsv(outdir / "BD_Core20_roi_voxels.tsv", voxel_rows, list(voxel_rows[0]))
    nib.save(atlas_out, outdir / "BD_Core20_dseg.nii.gz")
    nib.save(bold_out, outdir / "BD_Core20_space-MNI152NLin6Asym_res-bold_dseg.nii.gz")
    plot_qc(atlas_out, outdir)

    summary = [
        {"file": "BD_Core20_dseg.nii.gz", "shape": "x".join(map(str, atlas_out.shape)), "orientation": orientation(atlas_out), "affine": np.array2string(atlas_out.affine, precision=6, separator=",")},
        {"file": "BD_Core20_space-MNI152NLin6Asym_res-bold_dseg.nii.gz", "shape": "x".join(map(str, bold_out.shape)), "orientation": orientation(bold_out), "affine": np.array2string(bold_out.affine, precision=6, separator=",")},
    ]
    write_tsv(outdir / "BD_Core20_validation_summary.tsv", summary, list(summary[0]))
    print("BD-Core20 validation summary")
    print(f"  unique labels: {sorted(np.unique(out_data).tolist())}")
    print(f"  non-background labels: {len(np.unique(out_data)) - 1}")
    print(f"  original grid: shape={atlas_out.shape}, orientation={orientation(atlas_out)}")
    print(f"  BOLD grid: shape={bold_out.shape}, orientation={orientation(bold_out)}")
    print(f"  outputs: {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aal3-atlas", type=Path, required=True)
    parser.add_argument("--aal3-labels", type=Path, required=True)
    parser.add_argument("--reference-bold", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    build(args)

if __name__ == "__main__":
    main()
