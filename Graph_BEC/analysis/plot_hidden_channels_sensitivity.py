import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helper
# ============================================================
def to_decimal(values):
    return np.array(values, dtype=float) / 100.0


def get_best_and_second(values, decimals=4):
    """
    Return the indices of the best and second-best DISTINCT values.
    Ties are supported.
    """
    values = np.round(np.asarray(values), decimals)

    unique_values = np.unique(values)
    unique_values = np.sort(unique_values)[::-1]

    best_value = unique_values[0]
    best_idx = np.where(values == best_value)[0]

    if len(unique_values) > 1:
        second_value = unique_values[1]
        second_idx = np.where(values == second_value)[0]
    else:
        second_idx = np.array([], dtype=int)

    return best_idx, second_idx


# ============================================================
# Data
# ============================================================
experiments = [

    # --------------------------------------------------------
    # (a) Loss weight alpha
    # --------------------------------------------------------
    {
        "caption": r"(a) Loss weight $\alpha$ in the EC Estimator",
        "xlabel": r"Loss weight $\alpha$",
        "xticklabels": ["0.00", "0.01", "0.02", "0.03", "0.04"],

        "line_color": "#2A9D8F",
        "fill_color": "#BFE5DF",
        "best_color": "#E76F51",
        "second_color": "#EDC948",

        "metrics": {

            "ACC": {
                "mean": to_decimal([
                    60.62,
                    60.39,
                    61.09,
                    61.19,
                    59.59
                ]),
                "std": to_decimal([
                    5.04,
                    5.12,
                    4.62,
                    4.53,
                    6.16
                ]),
            },

            "SEN": {
                "mean": to_decimal([
                    59.99,
                    70.50,
                    67.94,
                    64.25,
                    64.40
                ]),
                "std": to_decimal([
                    18.19,
                    13.83,
                    14.21,
                    15.85,
                    16.81
                ]),
            },

            "AUC": {
                "mean": to_decimal([
                    66.25,
                    66.85,
                    68.76,
                    67.49,
                    65.35
                ]),
                "std": to_decimal([
                    4.99,
                    4.32,
                    5.03,
                    3.25,
                    6.16
                ]),
            },

            "F1": {
                "mean": to_decimal([
                    57.31,
                    61.73,
                    61.22,
                    59.56,
                    58.63
                ]),
                "std": to_decimal([
                    7.94,
                    6.89,
                    5.80,
                    8.88,
                    8.66
                ]),
            },
        }
    },

    # --------------------------------------------------------
    # (b) Multimodal fusion coefficient beta
    # --------------------------------------------------------
    {
        "caption": r"(b) Multimodal fusion coefficient $\beta$",
        "xlabel": r"Fusion coefficient $\beta$",
        "xticklabels": ["0.2", "0.4", "0.6", "0.8", "1.0"],

        "line_color": "#7A3E9D",
        "fill_color": "#D8BFE8",
        "best_color": "#F28E2B",
        "second_color": "#EDC948",

        "metrics": {

            "ACC": {
                "mean": to_decimal([
                    60.17,
                    60.39,
                    60.39,
                    61.31,
                    60.05
                ]),
                "std": to_decimal([
                    6.59,
                    4.89,
                    5.12,
                    4.70,
                    4.08
                ]),
            },

            "SEN": {
                "mean": to_decimal([
                    64.23,
                    62.77,
                    70.50,
                    66.02,
                    66.32
                ]),
                "std": to_decimal([
                    16.86,
                    12.59,
                    13.83,
                    11.89,
                    14.98
                ]),
            },

            "AUC": {
                "mean": to_decimal([
                    65.94,
                    64.84,
                    66.85,
                    66.29,
                    65.21
                ]),
                "std": to_decimal([
                    5.26,
                    3.91,
                    4.32,
                    3.74,
                    3.38
                ]),
            },

            "F1": {
                "mean": to_decimal([
                    59.00,
                    59.00,
                    61.73,
                    60.84,
                    59.81
                ]),
                "std": to_decimal([
                    8.16,
                    6.76,
                    6.89,
                    6.41,
                    7.51
                ]),
            },
        }
    },

    # --------------------------------------------------------
    # (c) Hidden channels
    # --------------------------------------------------------
    {
        "caption": r"(c) Hidden channels in the QSR-EC refinement network",
        "xlabel": "Hidden channels",
        "xticklabels": ["2", "4", "8", "16", "32"],

        "line_color": "#1D4E89",
        "fill_color": "#B9D3EA",
        "best_color": "#F4A261",
        "second_color": "#EDC948",

        "metrics": {

            "ACC": {
                "mean": to_decimal([
                    61.08,
                    58.90,
                    60.39,
                    59.59,
                    59.47
                ]),
                "std": to_decimal([
                    5.25,
                    5.30,
                    5.12,
                    5.33,
                    5.35
                ]),
            },

            "SEN": {
                "mean": to_decimal([
                    60.81,
                    66.77,
                    70.50,
                    66.17,
                    67.74
                ]),
                "std": to_decimal([
                    19.90,
                    12.51,
                    13.83,
                    4.33,
                    14.90
                ]),
            },

            "AUC": {
                "mean": to_decimal([
                    67.12,
                    65.19,
                    66.85,
                    67.74,
                    66.20
                ]),
                "std": to_decimal([
                    4.65,
                    3.37,
                    4.32,
                    14.90,
                    4.32
                ]),
            },

            "F1": {
                "mean": to_decimal([
                    57.50,
                    59.73,
                    61.73,
                    60.21,
                    60.14
                ]),
                "std": to_decimal([
                    11.12,
                    5.06,
                    6.89,
                    6.92,
                    7.01
                ]),
            },
        }
    }
]

# ============================================================
# Size controls
# ============================================================
# 后面想调大小，只改这里即可

# ---------------- Text ----------------
FONT_TICK = 11.0          # x/y 坐标刻度数字
FONT_XLABEL = 12.0        # Loss weight α / Fusion coefficient β ...
FONT_YLABEL = 12.0        # ACC / SEN / AUC / F1
FONT_CAPTION = 14.0       # (a) / (b) / (c)
FONT_LEGEND = 10.5        # Best / Second best

# ---------------- Markers ----------------
LINE_MARKER_SIZE = 4.5    # 普通折线圆点
BEST_MARKER_SIZE = 48     # Best 圆点

# ---------------- Lines ----------------
LINE_WIDTH = 1.6
MARKER_EDGE_WIDTH = 0.8
GRID_LINE_WIDTH = 0.45
SPINE_WIDTH = 0.75

# ---------------- Layout ----------------
WSPACE = 0.24             # 左右子图间距
HSPACE = 0.46             # 上下子图间距，越小越紧凑

# Caption 与当前这一行图的距离
CAPTION_OFFSET = 0.056


# ============================================================
# Global style
# ============================================================

plt.rcParams.update({
    "font.size": FONT_TICK,
    "axes.labelsize": FONT_XLABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,

    "mathtext.fontset": "stix",

    # PDF 中保留可编辑字体
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


n_rows = len(experiments)
n_cols = 4


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    n_rows,
    n_cols,

    # 保留你原来的横向 3 × 4 布局
    figsize=(14.8, 9.0),

    dpi=300,
    squeeze=False
)


# ============================================================
# Plot
# ============================================================

for row_idx, exp in enumerate(experiments):

    metric_names = list(
        exp["metrics"].keys()
    )

    x_labels = exp["xticklabels"]
    x = np.arange(len(x_labels))


    for col_idx, metric_name in enumerate(metric_names):

        ax = axes[row_idx, col_idx]

        mean = exp["metrics"][metric_name]["mean"]
        std = exp["metrics"][metric_name]["std"]


        # ----------------------------------------------------
        # Line
        # ----------------------------------------------------

        ax.plot(
            x,
            mean,

            marker="o",
            markersize=LINE_MARKER_SIZE,

            linewidth=LINE_WIDTH,

            color=exp["line_color"],

            zorder=4
        )


        # ----------------------------------------------------
        # Mean ± std shadow
        # ----------------------------------------------------

        ax.fill_between(
            x,

            mean - std,
            mean + std,

            color=exp["fill_color"],

            alpha=0.32,

            linewidth=0,

            zorder=1
        )


        # ----------------------------------------------------
        # Best / second-best
        # ----------------------------------------------------

        best_idx, _ = get_best_and_second(mean)


        # Best
        for i in best_idx:

            ax.scatter(
                x[i],
                mean[i],

                s=BEST_MARKER_SIZE,

                color=exp["best_color"],

                edgecolor="black",
                linewidth=MARKER_EDGE_WIDTH,

                marker="o",

                zorder=6,

                label=(
                    "Best"
                    if row_idx == 0 and col_idx == 0
                    else None
                )
            )


        


        # ----------------------------------------------------
        # X ticks
        # ----------------------------------------------------

        ax.set_xticks(x)

        ax.set_xticklabels(
            x_labels,
            fontsize=FONT_TICK
        )


        # ----------------------------------------------------
        # X label
        # ----------------------------------------------------

        ax.set_xlabel(
            exp["xlabel"],

            fontsize=FONT_XLABEL,

            labelpad=3
        )


        # ----------------------------------------------------
        # Y label: ACC / SEN / AUC / F1
        # ----------------------------------------------------

        ax.set_ylabel(
            metric_name,

            fontsize=FONT_YLABEL,

            labelpad=5
        )


        # ----------------------------------------------------
        # Adaptive y-range
        # ----------------------------------------------------

        y_min = max(
            0.0,
            np.min(mean - std) - 0.015
        )

        y_max = min(
            1.0,
            np.max(mean + std) + 0.015
        )


        # Avoid overly narrow y-axis
        if y_max - y_min < 0.08:

            center = (
                y_min + y_max
            ) / 2

            y_min = max(
                0.0,
                center - 0.04
            )

            y_max = min(
                1.0,
                center + 0.04
            )


        ax.set_ylim(
            y_min,
            y_max
        )


        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.grid(
            True,

            linestyle="--",

            linewidth=GRID_LINE_WIDTH,

            alpha=0.28,

            zorder=0
        )


        # ----------------------------------------------------
        # Spines
        # ----------------------------------------------------

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(
            SPINE_WIDTH
        )

        ax.spines["bottom"].set_linewidth(
            SPINE_WIDTH
        )


        # ----------------------------------------------------
        # Tick style
        # ----------------------------------------------------

        ax.tick_params(
            axis="both",

            labelsize=FONT_TICK,

            length=3,
            width=0.65,

            pad=2
        )


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.060,
    right=0.995,

    top=0.965,
    bottom=0.060,

    # 左右距离
    wspace=WSPACE,

    # 上下距离
    # 原来是 0.72，现在明显缩小
    hspace=HSPACE
)


# ============================================================
# Draw canvas before locating captions
# ============================================================

fig.canvas.draw()


# ============================================================
# Captions below each row
# ============================================================

for row_idx, exp in enumerate(experiments):

    row_axes = axes[row_idx, :]

    row_bottom = min(
        ax.get_position().y0
        for ax in row_axes
    )


    # --------------------------------------------------------
    # Caption position
    # --------------------------------------------------------

    caption_y = (
        row_bottom
        - CAPTION_OFFSET
    )


    fig.text(
        0.5,
        caption_y,

        exp["caption"],

        ha="center",
        va="top",

        fontsize=FONT_CAPTION
    )


# ============================================================
# Legend
# ============================================================

handles, labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)


fig.legend(
    handles,
    labels,

    loc="upper right",

    bbox_to_anchor=(
        0.995,
        0.995
    ),

    fontsize=FONT_LEGEND,

    frameon=False,

    ncol=1,

    handletextpad=0.4,

    borderaxespad=0.2
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "parameter_sensitivity_abide1.pdf",

    bbox_inches="tight",

    pad_inches=0.02
)


plt.savefig(
    "parameter_sensitivity_abide1.png",

    dpi=600,

    bbox_inches="tight",

    pad_inches=0.02
)


plt.show()
