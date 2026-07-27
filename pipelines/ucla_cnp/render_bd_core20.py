#!/usr/bin/env python3
"""使用 Matplotlib 绘制 BD-Core20 ROI atlas 的冠状面、矢状面和轴位三维视图。

改进点：
  1. 仅保留 Matplotlib 渲染流程。
  2. 使用 fsaverage 的 sulcal depth（脑沟深度）为皮层着色。
  3. 根据三角面法向量加入方向光照，增强脑回/脑沟层次。
  4. 默认提高皮层不透明度，并对远近 ROI 添加深度提示。
  5. 默认使用透视投影；需要标准正投影时可传入 --projection ortho。

用法示例:
  python render_bd_core20_matplotlib.py
  python render_bd_core20_matplotlib.py --surface-alpha 0.20
  python render_bd_core20_matplotlib.py --projection ortho
  python render_bd_core20_matplotlib.py --mesh fsaverage6 --dpi 400
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_ROIS = [
    "vmPFC_mOFC_L", "vmPFC_mOFC_R", "dlPFC_L", "dlPFC_R", "vlPFC_L", "vlPFC_R",
    "Anterior_Insula_L", "Anterior_Insula_R", "sgACC_L", "sgACC_R", "Amygdala_L",
    "Amygdala_R", "NAcc_L", "NAcc_R", "Caudate_L", "Caudate_R", "Putamen_L",
    "Putamen_R", "Thalamus_L", "Thalamus_R",
]

VIEWS = {
    "coronal": "Coronal",
    "sagittal": "Sagittal",
    "axial": "Axial",
}

MANAGED_OUTPUTS = {
    f"BD_Core20_render_{view}.png" for view in VIEWS
}

PAIR_COLORS = [
    "#D73027", "#4575B4", "#1A9850", "#762A83", "#E08214",
    "#C51B7D", "#2A9D8F", "#4D4D4D", "#8C510A", "#3F007D",
]

MATPLOTLIB_VIEWS = {
    "coronal": {"title": "Coronal", "elevation": 0, "azimuth": 90},
    "sagittal": {"title": "Sagittal", "elevation": 0, "azimuth": 0},
    "axial": {"title": "Axial", "elevation": 90, "azimuth": -90},
}


def repository_root() -> Path:
    """向上查找包含 dataset/BDCore20 的项目根目录。"""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "dataset" / "BDCore20").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing dataset/BDCore20")


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--atlas",
        type=Path,
        default=root / "dataset" / "BDCore20" / "provenance" / "BD_Core20_dseg.nii.gz",
        help="BD-Core20 NIfTI label atlas",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "dataset" / "BDCore20" / "provenance" / "BD_Core20_labels.tsv",
        help="ROI label table",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "dataset" / "BDCore20" / "figures",
        help="Output directory",
    )
    parser.add_argument(
        "--surface-alpha",
        type=float,
        default=0.20,
        help=(
            "Cortical surface opacity (0–1). "
            "Recommended range: 0.16–0.28; default: 0.20."
        ),
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=4.8,
        help="ROI sphere radius in coordinate units; default: 4.8.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution; default: 300.",
    )
    parser.add_argument(
        "--mesh",
        choices=("fsaverage5", "fsaverage6", "fsaverage7"),
        default="fsaverage5",
        help="FreeSurfer surface resolution; default: fsaverage5.",
    )
    parser.add_argument(
        "--projection",
        choices=("persp", "ortho"),
        default="persp",
        help=(
            "3D projection. 'persp' gives stronger depth cues; "
            "'ortho' is geometrically flatter; default: persp."
        ),
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> pd.DataFrame:
    """读取并验证 ROI 标签表。"""
    labels = (
        pd.read_csv(labels_path, sep="\t")
        .sort_values("index")
        .reset_index(drop=True)
    )
    required = {"index", "name", "hemisphere"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Label table is missing columns: {sorted(missing)}")
    if labels["index"].tolist() != list(range(1, 21)):
        raise ValueError("BD-Core20 label indices must be exactly 1 through 20")
    if labels["name"].tolist() != EXPECTED_ROIS:
        raise ValueError("BD-Core20 labels do not match the required ROI order")
    return labels


def compute_centers(atlas_path: Path, labels: pd.DataFrame) -> pd.DataFrame:
    """计算每个 ROI 的体素质心，并转换到 NIfTI 世界坐标。"""
    atlas_img = nib.load(atlas_path)
    atlas_data = np.asarray(atlas_img.dataobj)

    rows = []
    for row in labels.itertuples(index=False):
        voxel_indices = np.argwhere(atlas_data == row.index)
        if voxel_indices.size == 0:
            raise ValueError(f"ROI {row.index} ({row.name}) is empty")

        center = nib.affines.apply_affine(
            atlas_img.affine, voxel_indices
        ).mean(axis=0)

        expected_sign = -1 if row.hemisphere == "L" else 1
        if expected_sign * center[0] <= 0:
            raise ValueError(
                f"Hemisphere check failed for {row.name}: x={center[0]:.3f} mm"
            )

        rows.append(
            {
                "name": row.name,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "pair_id": (int(row.index) + 1) // 2,
            }
        )

    return pd.DataFrame(rows)


def clean_outputs(outdir: Path) -> None:
    """删除上一轮由本脚本生成的 PNG。"""
    for name in MANAGED_OUTPUTS:
        path = outdir / name
        if path.exists():
            path.unlink()


def copy_renderer(outdir: Path) -> None:
    """把当前脚本复制到输出目录，方便复现。"""
    source = Path(__file__).resolve()
    target = (outdir / source.name).resolve()
    if source != target:
        shutil.copy2(source, target)


def _setup_matplotlib() -> None:
    """配置 Matplotlib 使用无界面后端。"""
    import matplotlib

    matplotlib.use("Agg")


def make_pair_table(labels: pd.DataFrame) -> pd.DataFrame:
    """创建左右 ROI 配对名称和颜色表。"""
    left_names = labels["name"].iloc[::2].tolist()
    return pd.DataFrame(
        {
            "pair_id": range(1, 11),
            "legend_name": [
                f"{name.rsplit('_', 1)[0]} L&R" for name in left_names
            ],
            "hex_color": PAIR_COLORS,
        }
    )


def load_brain_surfaces(
    mesh_name: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """读取 fsaverage 左右半球的 pial 网格和 sulcal-depth 数据。"""
    from nilearn import datasets, surface

    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh_name)
    mesh_paths = (fsaverage.pial_left, fsaverage.pial_right)
    sulc_paths = (fsaverage.sulc_left, fsaverage.sulc_right)

    meshes = []
    for mesh_path, sulc_path in zip(mesh_paths, sulc_paths):
        coordinates, faces = surface.load_surf_mesh(mesh_path)
        sulc = surface.load_surf_data(sulc_path)

        meshes.append(
            (
                np.asarray(coordinates, dtype=float),
                np.asarray(faces, dtype=np.int32),
                np.asarray(sulc, dtype=float),
            )
        )
    return meshes


def robust_normalize(values: np.ndarray) -> np.ndarray:
    """使用 5%–95% 分位数稳健归一化到 0–1。"""
    values = np.asarray(values, dtype=float)
    low, high = np.nanpercentile(values, [5, 95])

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.full(values.shape, 0.5, dtype=float)

    return np.clip((values - low) / (high - low), 0.0, 1.0)


def make_surface_facecolors(
    coordinates: np.ndarray,
    faces: np.ndarray,
    sulc: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """根据脑沟深度和三角面法向量生成具有层次的灰色 RGBA。"""
    triangles = coordinates[faces]

    # 每个三角面的 sulcal-depth 值。
    face_sulc = sulc[faces].mean(axis=1)
    sulc_norm = robust_normalize(face_sulc)

    # 三角面法向量，用于方向光照。
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_1, edge_2)

    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normal_lengths[normal_lengths == 0] = 1.0
    normals = normals / normal_lengths

    light_direction = np.array([0.35, -0.45, 0.82], dtype=float)
    light_direction /= np.linalg.norm(light_direction)

    directional_light = np.clip(normals @ light_direction, 0.0, 1.0)

    # sulcal depth 提供脑沟/脑回对比，方向光提供立体明暗。
    gray = (
        0.69
        + 0.17 * sulc_norm
        + 0.12 * directional_light
    )
    gray = np.clip(gray, 0.66, 0.98)

    # 法线朝向光源的面稍微更不透明，轮廓更清楚。
    face_alpha = alpha * (0.78 + 0.22 * directional_light)
    face_alpha = np.clip(face_alpha, 0.0, 1.0)

    return np.column_stack((gray, gray, gray, face_alpha))


def add_brain_surface(axis, meshes, alpha: float) -> None:
    """添加带 sulcal-depth 和方向光照的半透明皮层表面。"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for coordinates, faces, sulc in meshes:
        collection = Poly3DCollection(
            coordinates[faces],
            facecolors=make_surface_facecolors(
                coordinates=coordinates,
                faces=faces,
                sulc=sulc,
                alpha=alpha,
            ),
            edgecolors="none",
            linewidths=0,
            antialiased=True,
            zsort="average",
        )
        collection.set_rasterized(True)
        axis.add_collection3d(collection)


def sphere_grid(
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成较平滑的球面网格。"""
    azimuth = np.linspace(0, 2 * np.pi, 48)
    polar = np.linspace(0, np.pi, 32)

    sphere_x = radius * np.outer(np.cos(azimuth), np.sin(polar))
    sphere_y = radius * np.outer(np.sin(azimuth), np.sin(polar))
    sphere_z = radius * np.outer(np.ones_like(azimuth), np.cos(polar))
    return sphere_x, sphere_y, sphere_z


def camera_depth(
    centers: pd.DataFrame,
    elevation: float,
    azimuth: float,
) -> np.ndarray:
    """估计 ROI 相对摄像机的远近，返回 0（远）到 1（近）。"""
    elev = np.deg2rad(elevation)
    azim = np.deg2rad(azimuth)

    camera_direction = np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=float,
    )

    xyz = centers[["x", "y", "z"]].to_numpy(dtype=float)
    raw_depth = xyz @ camera_direction
    return robust_normalize(raw_depth)


def mix_with_white(rgb: np.ndarray, amount: float) -> np.ndarray:
    """把颜色按 amount 比例与白色混合。"""
    return rgb * (1.0 - amount) + np.ones(3, dtype=float) * amount


def add_roi_spheres(
    axis,
    centers: pd.DataFrame,
    pairs: pd.DataFrame,
    radius: float,
    elevation: float,
    azimuth: float,
) -> None:
    """绘制 ROI 球体，并用远近明暗、透明度和大小增强层次感。"""
    from matplotlib.colors import LightSource, to_rgb

    pair_colors = pairs.set_index("pair_id")["hex_color"].to_dict()
    depth = camera_depth(centers, elevation=elevation, azimuth=azimuth)

    centers_with_depth = centers.copy()
    centers_with_depth["depth"] = depth

    # 远处先画、近处后画，减少后方球体覆盖前方球体的情况。
    centers_with_depth = centers_with_depth.sort_values("depth")

    light_source = LightSource(azdeg=315, altdeg=45)

    for center in centers_with_depth.itertuples(index=False):
        # 远处 ROI 更浅、更小、更透明；近处 ROI 更饱和、更大、更不透明。
        depth_value = float(center.depth)
        local_radius = radius * (0.90 + 0.12 * depth_value)
        local_alpha = 0.70 + 0.30 * depth_value
        whiten_amount = 0.22 * (1.0 - depth_value)

        rgb = np.asarray(to_rgb(pair_colors[center.pair_id]), dtype=float)
        rgb = mix_with_white(rgb, whiten_amount)

        sphere_x, sphere_y, sphere_z = sphere_grid(local_radius)

        axis.plot_surface(
            sphere_x + center.x,
            sphere_y + center.y,
            sphere_z + center.z,
            color=rgb,
            linewidth=0,
            antialiased=True,
            shade=True,
            lightsource=light_source,
            alpha=local_alpha,
        )


def legend_handles(pairs: pd.DataFrame) -> list:
    """创建图例中的彩色圆点。"""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=row.hex_color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            label=row.legend_name,
        )
        for row in pairs.itertuples(index=False)
    ]


def create_3d_axis(figure):
    """创建三维坐标轴，保留 Matplotlib 的自动深度排序。"""
    axis_rect = [0.02, 0.16, 0.96, 0.75]
    return figure.add_axes(axis_rect, projection="3d")


def set_projection(axis, projection: str) -> None:
    """设置透视或正投影，并兼容不同 Matplotlib 版本。"""
    if projection == "persp":
        try:
            axis.set_proj_type("persp", focal_length=1.25)
        except TypeError:
            axis.set_proj_type("persp")
    else:
        axis.set_proj_type("ortho")


def render_view(
    axis,
    view_name: str,
    meshes,
    centers: pd.DataFrame,
    pairs: pd.DataFrame,
    radius: float,
    alpha: float,
    projection: str,
) -> None:
    """绘制一个指定方向的三维视图。"""
    view = MATPLOTLIB_VIEWS[view_name]

    axis.view_init(
        elev=view["elevation"],
        azim=view["azimuth"],
    )
    set_projection(axis, projection)

    axis.set_xlim(-82, 82)
    axis.set_ylim(-115, 80)
    axis.set_zlim(-70, 90)
    axis.set_box_aspect((164, 195, 160), zoom=1.35)
    axis.set_axis_off()
    axis.set_facecolor("white")

    add_brain_surface(axis, meshes, alpha)
    add_roi_spheres(
        axis=axis,
        centers=centers,
        pairs=pairs,
        radius=radius,
        elevation=view["elevation"],
        azimuth=view["azimuth"],
    )


def trim_white_margin(image_path: Path, dpi: int) -> None:
    """裁掉图片外围白边。"""
    with Image.open(image_path).convert("RGB") as image:
        pixels = np.asarray(image)
        content = np.any(pixels < 248, axis=2)

        if not np.any(content):
            return

        rows, columns = np.where(content)
        padding = 18

        left = max(0, int(columns.min()) - padding)
        top = max(0, int(rows.min()) - padding)
        right = min(image.width, int(columns.max()) + padding + 1)
        bottom = min(image.height, int(rows.max()) + padding + 1)

        image.crop((left, top, right, bottom)).save(
            image_path,
            dpi=(dpi, dpi),
        )


def save_views(
    outdir: Path,
    meshes,
    centers: pd.DataFrame,
    pairs: pd.DataFrame,
    radius: float,
    alpha: float,
    dpi: int,
    projection: str,
) -> None:
    """保存冠状面、矢状面和轴位 PNG。"""
    import matplotlib.pyplot as plt

    handles = legend_handles(pairs)

    for view_name, view in MATPLOTLIB_VIEWS.items():
        figure = plt.figure(
            figsize=(7.6, 6.6),
            facecolor="white",
        )
        axis = create_3d_axis(figure)

        render_view(
            axis=axis,
            view_name=view_name,
            meshes=meshes,
            centers=centers,
            pairs=pairs,
            radius=radius,
            alpha=alpha,
            projection=projection,
        )

        figure.suptitle(
            view["title"],
            fontsize=15,
            fontweight="semibold",
            y=0.965,
        )
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            frameon=False,
            fontsize=8.2,
            columnspacing=1.0,
            handletextpad=0.35,
            bbox_to_anchor=(0.5, 0.025),
        )

        output_path = outdir / f"BD_Core20_render_{view_name}.png"
        figure.savefig(
            output_path,
            dpi=dpi,
            facecolor="white",
        )
        plt.close(figure)
        trim_white_margin(output_path, dpi)


def _verify_outputs(outdir: Path) -> None:
    """检查预期输出文件是否都已生成。"""
    expected = {
        f"BD_Core20_render_{view}.png" for view in VIEWS
    }
    missing = sorted(
        name for name in expected
        if not (outdir / name).is_file()
    )
    if missing:
        raise RuntimeError(
            f"Missing expected output files: {missing}"
        )


def main() -> None:
    """程序入口。"""
    args = parse_args()
    _setup_matplotlib()

    atlas_path = args.atlas.resolve()
    labels_path = args.labels.resolve()
    outdir = args.outdir.resolve()

    if not atlas_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing atlas or labels: {atlas_path}, {labels_path}"
        )

    if not 0.0 < args.surface_alpha < 1.0:
        raise ValueError(
            "--surface-alpha must be between 0 and 1"
        )
    if args.sphere_radius <= 0:
        raise ValueError(
            "--sphere-radius must be positive"
        )
    if args.dpi < 300:
        raise ValueError(
            "--dpi must be at least 300"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    clean_outputs(outdir)

    labels = load_labels(labels_path)
    centers = compute_centers(atlas_path, labels)
    pairs = make_pair_table(labels)
    meshes = load_brain_surfaces(args.mesh)

    save_views(
        outdir=outdir,
        meshes=meshes,
        centers=centers,
        pairs=pairs,
        radius=args.sphere_radius,
        alpha=args.surface_alpha,
        dpi=args.dpi,
        projection=args.projection,
    )

    copy_renderer(outdir)
    _verify_outputs(outdir)

    print(
        f"Generated three BD-Core20 views in {outdir}"
    )
    print(
        "Note: ROI centers use NIfTI world coordinates, while fsaverage is a "
        "FreeSurfer template surface. Confirm coordinate-space alignment "
        "before interpreting exact anatomical positions."
    )


if __name__ == "__main__":
    main()
