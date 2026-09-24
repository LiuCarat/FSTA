import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# Global size controls
# ============================================================

FONT_GLOBAL = 9.5

FONT_TITLE = 9.5
FONT_VALUE = 9.0
FONT_XTICK = 9.0
FONT_CONFIG = 8.5
FONT_LEGEND = 9.0

# Configuration matrix dots
SIZE_INACTIVE_DOT = 52
SIZE_ACTIVE_DOT = 64

# Best / second-best circles
SIZE_RANK_MARKER = 36

# Line widths
WIDTH_CONFIG_LINE = 1.35
WIDTH_ERROR = 1.0
WIDTH_GRID = 0.55
WIDTH_RANK_EDGE = 0.9


# ============================================================
# Configuration colors
# ============================================================
#
# Color now represents CONFIGURATION, not metric.
#
# The first five colors follow your uploaded palette.
# A muted purple is added for the sixth configuration.
# ============================================================

CONFIG_COLORS = [
    "#466F87",  # Ph
    "#78B6C8",  # Ph + QC
    "#F2D98E",  # FC
    "#E48578",  # FC + QC
    "#98AF1E",  # Ph + FC
    "#A78BC2",  # Ph + FC + QC
]


# ============================================================
# Helper
# ============================================================

def get_best_and_second(mean, decimals=2):
    """
    Return indices of the best and second-best DISTINCT mean values.

    Values are rounded to `decimals` before comparison.
    Ties are supported.
    """
    rounded = np.round(mean, decimals)

    unique_values = np.unique(rounded)
    unique_values = np.sort(unique_values)[::-1]

    best_value = unique_values[0]
    best_idx = np.where(
        rounded == best_value
    )[0]

    if len(unique_values) > 1:
        second_value = unique_values[1]
        second_idx = np.where(
            rounded == second_value
        )[0]
    else:
        second_idx = np.array([], dtype=int)

    return best_idx, second_idx


# ============================================================
# Data
# ============================================================

# Each mask corresponds to [Ph, FC, QC].
#
# Row 1: Ph
# Row 2: Ph + QC
# Row 3: FC
# Row 4: FC + QC
# Row 5: Ph + FC
# Row 6: Ph + FC + QC

configs = [
    ("Ph",           [1, 0, 0]),
    ("Ph + QC",      [1, 0, 1]),
    ("FC",           [0, 1, 0]),
    ("FC + QC",      [0, 1, 1]),
    ("Ph + FC",      [1, 1, 0]),
    ("Ph + FC + QC", [1, 1, 1]),
]


metrics = {

    "ACC": {
        "mean": np.array([
            61.54,
            60.04,
            60.05,
            60.05,
            60.39,
            60.39,
        ]),
        "std": np.array([
            4.12,
            3.69,
            4.08,
            5.17,
            4.93,
            5.12,
        ]),
    },

    "SEN": {
        "mean": np.array([
            61.29,
            68.02,
            66.32,
            67.50,
            68.99,
            70.50,
        ]),
        "std": np.array([
            18.06,
            10.37,
            14.98,
            12.28,
            14.68,
            13.83,
        ]),
    },

    "AUC": {
        "mean": np.array([
            65.53,
            65.61,
            65.21,
            66.37,
            65.79,
            66.85,
        ]),
        "std": np.array([
            3.97,
            3.10,
            3.38,
            5.72,
            4.89,
            4.32,
        ]),
    },

    "F1": {
        "mean": np.array([
            58.32,
            60.92,
            59.81,
            60.60,
            61.10,
            61.73,
        ]),
        "std": np.array([
            9.04,
            4.05,
            7.51,
            6.61,
            6.49,
            6.89,
        ]),
    },
}


# ============================================================
# Global style
# ============================================================

plt.rcParams.update({

    "font.size": FONT_GLOBAL,

    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_GLOBAL,

    "xtick.labelsize": FONT_XTICK,
    "ytick.labelsize": FONT_XTICK,

    # Embed editable fonts in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


inactive_color = "#D9DEE5"
error_color = "#555A60"
grid_color = "#E8EBEF"
text_color = "#1F2328"

n = len(configs)
y = np.arange(n)


# ============================================================
# Figure and layout
# ============================================================

fig = plt.figure(
    figsize=(7.15, 4.90),
    dpi=300
)


# Layout:
#
#            ACC            SEN
# Ph FC QC
#
#            AUC            F1
# Ph FC QC

gs = GridSpec(
    2,
    3,
    figure=fig,

    width_ratios=[
        1.00,   # configuration matrix
        3.15,   # metric 1
        3.15,   # metric 2
    ],

    height_ratios=[
        1,
        1
    ],

    wspace=0.035,
    hspace=0.32
)


# ============================================================
# Helper: draw configuration matrix
# ============================================================

def draw_config_matrix(ax):

    for row, (_, mask) in enumerate(configs):

        mask = np.array(mask)
        active = np.where(mask == 1)[0]

        row_color = CONFIG_COLORS[row]

        # ----------------------------------------------------
        # Inactive dots
        # ----------------------------------------------------

        ax.scatter(
            [0, 1, 2],
            [row] * 3,

            s=SIZE_INACTIVE_DOT,

            color=inactive_color,
            edgecolors="none",

            zorder=1
        )

        # ----------------------------------------------------
        # Connect active modules
        # ----------------------------------------------------

        if len(active) > 1:

            ax.plot(
                [active.min(), active.max()],
                [row, row],

                color=row_color,

                linewidth=WIDTH_CONFIG_LINE,

                solid_capstyle="round",

                zorder=2
            )

        # ----------------------------------------------------
        # Active dots
        # ----------------------------------------------------

        ax.scatter(
            active,
            [row] * len(active),

            s=SIZE_ACTIVE_DOT,

            color=row_color,

            edgecolors="white",
            linewidths=0.8,

            zorder=3
        )

    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------

    ax.set_xlim(
        -0.58,
        2.58
    )

    ax.set_ylim(
        n - 0.5,
        -0.40
    )

    # --------------------------------------------------------
    # Ph / FC / QC labels
    # --------------------------------------------------------

    ax.set_xticks(
        [0, 1, 2],
        labels=[
            "Ph",
            "FC",
            "QC"
        ]
    )

    ax.xaxis.tick_top()

    ax.tick_params(
        axis="x",

        length=0,
        pad=5,

        labelsize=FONT_CONFIG,

        colors=text_color
    )

    # for label in ax.get_xticklabels():
        # label.set_fontweight("semibold")

    ax.set_yticks([])

    # --------------------------------------------------------
    # Remove frame
    # --------------------------------------------------------

    for spine in ax.spines.values():
        spine.set_visible(False)

    # --------------------------------------------------------
    # Horizontal separators
    # --------------------------------------------------------

    for sep in np.arange(
        0.5,
        n - 0.5,
        1
    ):

        ax.axhline(
            sep,

            color=grid_color,

            linewidth=WIDTH_GRID,

            zorder=0
        )


# ============================================================
# Configuration matrices
# ============================================================

ax_cfg_top = fig.add_subplot(
    gs[0, 0]
)

draw_config_matrix(
    ax_cfg_top
)


ax_cfg_bottom = fig.add_subplot(
    gs[1, 0]
)

draw_config_matrix(
    ax_cfg_bottom
)


cfg_axes = [
    ax_cfg_top,
    ax_cfg_bottom,
]


# ============================================================
# Metric panels
# ============================================================

xmax = 92

cap_half_height = 0.11

metric_items = list(
    metrics.items()
)


for idx, (metric_name, item) in enumerate(metric_items):

    # --------------------------------------------------------
    # Position
    #
    # ACC -> row 0, col 1
    # SEN -> row 0, col 2
    # AUC -> row 1, col 1
    # F1  -> row 1, col 2
    # --------------------------------------------------------

    row = idx // 2
    col = idx % 2 + 1

    ax = fig.add_subplot(
        gs[row, col],
        sharey=cfg_axes[row]
    )

    mean = item["mean"]
    std = item["std"]


    # ========================================================
    # Bars
    #
    # Every row has its own fixed configuration color.
    # ========================================================

    ax.barh(
        y,
        mean,

        height=0.74,

        color=CONFIG_COLORS,

        edgecolor="none",

        zorder=2
    )


    # ========================================================
    # Error bars
    # ========================================================

    for yi, m, s in zip(
        y,
        mean,
        std
    ):

        # Horizontal error line
        ax.hlines(
            yi,
            m,
            m + s,

            colors=error_color,

            linewidth=WIDTH_ERROR,

            zorder=4
        )

        # End cap
        ax.vlines(
            m + s,

            yi - cap_half_height,
            yi + cap_half_height,

            colors=error_color,

            linewidth=WIDTH_ERROR,

            zorder=4
        )


    # ========================================================
    # Best / second-best
    # ========================================================

    best_idx, second_idx = get_best_and_second(
        mean,
        decimals=2
    )


    # ========================================================
    # Value labels
    # ========================================================

    for i, (m, s) in enumerate(
        zip(mean, std)
    ):

        label = f"{m:.2f}±{s:.2f}"

        ax.text(
            m * 0.50,
            i,
            label,

            ha="center",
            va="center",

            fontsize=FONT_VALUE,

            color="white",

            fontweight="medium",

            zorder=5
        )


    # ========================================================
    # Best markers
    #
    # Filled circle using the corresponding configuration color.
    # ========================================================

    for best_i in best_idx:

        ax.scatter(
            mean[best_i],
            best_i,

            s=SIZE_RANK_MARKER,

            color=CONFIG_COLORS[best_i],

            edgecolors="black",

            linewidths=WIDTH_RANK_EDGE,

            zorder=6
        )


    # ========================================================
    # Second-best markers
    #
    # White center + corresponding configuration color outline.
    # ========================================================

    for second_i in second_idx:

        ax.scatter(
            mean[second_i],
            second_i,

            s=SIZE_RANK_MARKER,

            facecolors="white",

            edgecolors=CONFIG_COLORS[second_i],

            linewidths=WIDTH_RANK_EDGE + 0.25,

            zorder=6
        )


    # ========================================================
    # X axis
    # ========================================================

    ax.set_xlim(
        0,
        xmax
    )

    ax.set_xticks(
        np.arange(
            0,
            91,
            15
        )
    )

    ax.tick_params(
        axis="x",

        labelsize=FONT_XTICK,

        length=3,
        width=0.8,

        pad=3,

        colors=text_color
    )

    for tick in ax.get_xticklabels():
        tick.set_fontweight("medium")


    # --------------------------------------------------------
    # Remove Y labels
    # --------------------------------------------------------

    ax.set_yticks([])


    # ========================================================
    # Metric title
    # ========================================================

    ax.set_title(
        metric_name,

        pad=7,

        fontsize=FONT_TITLE,

        # fontweight="semibold",

        color=text_color
    )


    # ========================================================
    # Vertical grid
    # ========================================================

    ax.grid(
        axis="x",

        color=grid_color,

        linewidth=WIDTH_GRID,

        alpha=0.95,

        zorder=0
    )


    # ========================================================
    # Horizontal separators
    # ========================================================

    for sep in np.arange(
        0.5,
        n - 0.5,
        1
    ):

        ax.axhline(
            sep,

            color=grid_color,

            linewidth=WIDTH_GRID,

            zorder=0
        )


    # ========================================================
    # Spines
    # ========================================================

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_visible(
        False
    )

    ax.spines["bottom"].set_color(
        "#7E848B"
    )

    ax.spines["bottom"].set_linewidth(
        0.9
    )


# ============================================================
# Overall layout
# ============================================================

fig.subplots_adjust(

    left=0.025,
    right=0.995,

    top=0.90,
    bottom=0.10
)


# ============================================================
# Legend
# ============================================================

fig.text(

    0.995,
    0.968,

    "● best    ○ second-best",

    ha="right",
    va="center",

    fontsize=FONT_LEGEND,

    fontweight="medium",

    color="#42474E"
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "fine_grained_ablation_config_colors.pdf",

    bbox_inches="tight",

    pad_inches=0.025
)


plt.savefig(
    "fine_grained_ablation_config_colors.png",

    dpi=600,

    bbox_inches="tight",

    pad_inches=0.025
)


plt.show()
