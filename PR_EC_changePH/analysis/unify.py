import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# Unified ICLR figure style
# ============================================================
# Layout:
#
#   LEFT                  MIDDLE                         RIGHT
#   Ablation              Parameter sensitivity         Top-k sensitivity
#   narrower              WIDEST                        narrower
#
# All figures have EXACTLY the same physical height.
# ============================================================

FIG_H = 4.20

FIG_W_SIDE = 4.20
FIG_W_MIDDLE = 6.45


# ============================================================
# Unified font sizes
# ============================================================

FONT_TICK = 10.0
FONT_LABEL = 10.5
FONT_TITLE = 10.5
FONT_VALUE = 9.0
FONT_LEGEND = 9.5
FONT_CAPTION = 10.5
FONT_SMALL = 8.5

# ------------------------------------------------------------
# Transparency of value text inside ablation bars
#
# 1.0 = completely opaque
# 0.8 = slightly transparent
# 0.6 = more transparent
# ------------------------------------------------------------
VALUE_ALPHA = 0.65


# ============================================================
# Unified line / marker style
# ============================================================

LINE_WIDTH = 1.55
GRID_WIDTH = 0.55
SPINE_WIDTH = 0.80
MARKER_SIZE = 4.8


# ============================================================
# Global Matplotlib settings
# ============================================================

plt.rcParams.update({

    "font.size": FONT_TICK,

    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_LABEL,

    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,

    "legend.fontsize": FONT_LEGEND,

    "mathtext.fontset": "stix",

    # Keep editable fonts in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Helper functions
# ============================================================

def get_best_and_second(values, decimals=4):

    values = np.round(
        np.asarray(values),
        decimals
    )

    unique_values = np.unique(values)

    unique_values = np.sort(
        unique_values
    )[::-1]

    best_value = unique_values[0]

    best_idx = np.where(
        values == best_value
    )[0]

    if len(unique_values) > 1:

        second_value = unique_values[1]

        second_idx = np.where(
            values == second_value
        )[0]

    else:

        second_idx = np.array(
            [],
            dtype=int
        )

    return best_idx, second_idx


def to_decimal(values):

    return (
        np.asarray(
            values,
            dtype=float
        )
        / 100.0
    )


def save_figure(fig, stem):

    fig.savefig(
        f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.025
    )

    fig.savefig(
        f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025
    )


# ============================================================
# ============================================================
#
# FIGURE 1
#
# LEFT:
# Fine-grained ablation
#
# ============================================================
# ============================================================

def make_ablation():

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    CONFIG_COLORS = [

        "#466F87",  # Ph

        "#78B6C8",  # Ph + QC

        "#F2D98E",  # FC

        "#E48578",  # FC + QC

        "#98AF1E",  # Ph + FC

        "#A78BC2",  # Ph + FC + QC
    ]

    inactive_color = "#D9DEE5"

    error_color = "#555A60"

    grid_color = "#E8EBEF"

    text_color = "#1F2328"


    # --------------------------------------------------------
    # Configurations
    # --------------------------------------------------------

    configs = [

        (
            "Ph",
            [1, 0, 0]
        ),

        (
            "Ph + QC",
            [1, 0, 1]
        ),

        (
            "FC",
            [0, 1, 0]
        ),

        (
            "FC + QC",
            [0, 1, 1]
        ),

        (
            "Ph + FC",
            [1, 1, 0]
        ),

        (
            "Ph + FC + QC",
            [1, 1, 1]
        ),
    ]


    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    metrics = {

        "ACC": {

            "mean": np.array([
                61.54,
                60.04,
                60.05,
                60.05,
                60.39,
                60.39
            ]),

            "std": np.array([
                4.12,
                3.69,
                4.08,
                5.17,
                4.93,
                5.12
            ])
        },


        "SEN": {

            "mean": np.array([
                61.29,
                68.02,
                66.32,
                67.50,
                68.99,
                70.50
            ]),

            "std": np.array([
                18.06,
                10.37,
                14.98,
                12.28,
                14.68,
                13.83
            ])
        },


        "AUC": {

            "mean": np.array([
                65.53,
                65.61,
                65.21,
                66.37,
                65.79,
                66.85
            ]),

            "std": np.array([
                3.97,
                3.10,
                3.38,
                5.72,
                4.89,
                4.32
            ])
        },


        "F1": {

            "mean": np.array([
                58.32,
                60.92,
                59.81,
                60.60,
                61.10,
                61.73
            ]),

            "std": np.array([
                9.04,
                4.05,
                7.51,
                6.61,
                6.49,
                6.89
            ])
        }
    }


    n = len(configs)

    y = np.arange(n)


    # ========================================================
    # Figure
    # ========================================================

    fig = plt.figure(

        figsize=(
            FIG_W_SIDE,
            FIG_H
        ),

        dpi=300
    )


    gs = GridSpec(

        2,
        3,

        figure=fig,

        width_ratios=[
            1.05,
            3.15,
            3.15
        ],

        height_ratios=[
            1,
            1
        ],

        wspace=0.055,

        hspace=0.34
    )


    # ========================================================
    # Configuration matrix
    # ========================================================

    def draw_config_matrix(ax):

        for row, (_, mask) in enumerate(configs):

            mask = np.asarray(mask)

            active = np.where(
                mask == 1
            )[0]

            row_color = CONFIG_COLORS[row]


            # ------------------------------------------------
            # Inactive dots
            # ------------------------------------------------

            ax.scatter(

                [0, 1, 2],

                [row] * 3,

                s=44,

                color=inactive_color,

                edgecolors="none",

                zorder=1
            )


            # ------------------------------------------------
            # Active connection
            # ------------------------------------------------

            if len(active) > 1:

                ax.plot(

                    [
                        active.min(),
                        active.max()
                    ],

                    [
                        row,
                        row
                    ],

                    color=row_color,

                    linewidth=1.25,

                    solid_capstyle="round",

                    zorder=2
                )


            # ------------------------------------------------
            # Active dots
            # ------------------------------------------------

            ax.scatter(

                active,

                [row] * len(active),

                s=54,

                color=row_color,

                edgecolors="white",

                linewidths=0.7,

                zorder=3
            )


        # ----------------------------------------------------
        # Axis
        # ----------------------------------------------------

        ax.set_xlim(
            -0.58,
            2.58
        )

        ax.set_ylim(
            n - 0.5,
            -0.40
        )


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

            pad=4,

            labelsize=FONT_SMALL,

            colors=text_color
        )


        ax.set_yticks([])


        # ----------------------------------------------------
        # Remove frame
        # ----------------------------------------------------

        for spine in ax.spines.values():

            spine.set_visible(False)


        # ----------------------------------------------------
        # Horizontal separators
        # ----------------------------------------------------

        for sep in np.arange(
            0.5,
            n - 0.5,
            1
        ):

            ax.axhline(

                sep,

                color=grid_color,

                linewidth=GRID_WIDTH,

                zorder=0
            )


    # ========================================================
    # Configuration axes
    # ========================================================

    ax_cfg_top = fig.add_subplot(
        gs[0, 0]
    )


    ax_cfg_bottom = fig.add_subplot(
        gs[1, 0]
    )


    draw_config_matrix(
        ax_cfg_top
    )


    draw_config_matrix(
        ax_cfg_bottom
    )


    cfg_axes = [

        ax_cfg_top,

        ax_cfg_bottom
    ]


    # ========================================================
    # Metric panels
    # ========================================================

    for idx, (
        metric_name,
        item
    ) in enumerate(
        metrics.items()
    ):


        row = idx // 2

        col = idx % 2 + 1


        ax = fig.add_subplot(

            gs[row, col],

            sharey=cfg_axes[row]
        )


        mean = item["mean"]

        std = item["std"]


        # ----------------------------------------------------
        # Bars
        # ----------------------------------------------------

        ax.barh(

            y,

            mean,

            height=0.74,

            color=CONFIG_COLORS,

            edgecolor="none",

            zorder=2
        )


        # ----------------------------------------------------
        # Error bars
        # ----------------------------------------------------

        for yi, m, s in zip(
            y,
            mean,
            std
        ):

            ax.hlines(

                yi,

                m,

                m + s,

                colors=error_color,

                linewidth=1.0,

                zorder=4
            )


            ax.vlines(

                m + s,

                yi - 0.11,

                yi + 0.11,

                colors=error_color,

                linewidth=1.0,

                zorder=4
            )


        # ----------------------------------------------------
        # Best / second best
        # ----------------------------------------------------

        best_idx, second_idx = (
            get_best_and_second(
                mean,
                decimals=2
            )
        )


        # ----------------------------------------------------
        # Value labels
        # ----------------------------------------------------

        for i, (
            m,
            s
        ) in enumerate(
            zip(mean, std)
        ):

            ax.text(

    m * 0.50,

    i,

    f"{m:.2f}±{s:.2f}",

    ha="center",

    va="center",

    fontsize=FONT_VALUE,

    color="white",

    alpha=VALUE_ALPHA,

    fontweight="medium",

    zorder=5
)


        # ----------------------------------------------------
        # Best
        # ----------------------------------------------------

        for i in best_idx:

            ax.scatter(

                mean[i],

                i,

                s=34,

                color=CONFIG_COLORS[i],

                edgecolors="black",

                linewidths=0.9,

                zorder=6
            )


        # ----------------------------------------------------
        # Second best
        # ----------------------------------------------------

        for i in second_idx:

            ax.scatter(

                mean[i],

                i,

                s=34,

                facecolors="white",

                edgecolors=CONFIG_COLORS[i],

                linewidths=1.1,

                zorder=6
            )


        # ----------------------------------------------------
        # X axis
        # ----------------------------------------------------

        ax.set_xlim(
            0,
            92
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

            labelsize=FONT_SMALL,

            length=3,

            width=0.8,

            pad=2,

            colors=text_color
        )


        ax.set_yticks([])


        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        ax.set_title(

            metric_name,

            pad=5,

            fontsize=FONT_TITLE,

            color=text_color
        )


        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.grid(

            axis="x",

            color=grid_color,

            linewidth=GRID_WIDTH,

            alpha=0.95,

            zorder=0
        )


        for sep in np.arange(
            0.5,
            n - 0.5,
            1
        ):

            ax.axhline(

                sep,

                color=grid_color,

                linewidth=GRID_WIDTH,

                zorder=0
            )


        # ----------------------------------------------------
        # Spines
        # ----------------------------------------------------

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
            SPINE_WIDTH
        )


    # ========================================================
    # Layout
    # ========================================================

    fig.subplots_adjust(

        left=0.025,

        right=0.995,

        top=0.88,

        bottom=0.10
    )


    # ========================================================
    # Legend
    # ========================================================

    fig.text(

        0.995,

        0.965,

        "● best    ○ second-best",

        ha="right",

        va="center",

        fontsize=FONT_LEGEND,

        fontweight="medium",

        color="#42474E"
    )


    save_figure(
        fig,
        "panel_left_ablation"
    )


    return fig


# ============================================================
# ============================================================
#
# FIGURE 2
#
# MIDDLE:
# Parameter sensitivity
#
# THIS IS THE WIDEST FIGURE
#
# ============================================================
# ============================================================

def make_parameter_sensitivity():

    # ========================================================
    # Data
    # ========================================================

    experiments = [

        # ----------------------------------------------------
        # (a) Alpha
        # ----------------------------------------------------
        {
            "caption":
                r"(a) Loss weight $\alpha$ in the Spatiotemporal Estimator",

            "xlabel":
                r"Loss weight $\alpha$",

            "xticklabels":
                ["0.00", "0.01", "0.02", "0.03", "0.04"],

            "line_color":
                "#2A9D8F",

            "fill_color":
                "#BFE5DF",

            "best_color":
                "#E76F51",

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
                    ])
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
                    ])
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
                    ])
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
                    ])
                }
            }
        },


        # ----------------------------------------------------
        # (b) Beta
        # ----------------------------------------------------
        {
            "caption":
                r"(b) Multimodal Population Graph fusion coefficient $\beta$",

            "xlabel":
                r"Fusion coefficient $\beta$",

            "xticklabels":
                ["0.2", "0.4", "0.6", "0.8", "1.0"],

            "line_color":
                "#7A3E9D",

            "fill_color":
                "#D8BFE8",

            "best_color":
                "#F28E2B",

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
                    ])
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
                    ])
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
                    ])
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
                    ])
                }
            }
        },


        # ----------------------------------------------------
        # (c) Hidden channels
        # ----------------------------------------------------
        {
            "caption":
                r"(c) Hidden channels in the QSR Network",

            "xlabel":
                "Hidden channels",

            "xticklabels":
                ["2", "4", "8", "16", "32"],

            "line_color":
                "#1D4E89",

            "fill_color":
                "#B9D3EA",

            "best_color":
                "#F4A261",

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
                    ])
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
                    ])
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
                    ])
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
                    ])
                }
            }
        }
    ]


    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(

        3,
        4,

        figsize=(
            FIG_W_MIDDLE,
            FIG_H
        ),

        dpi=300,

        squeeze=False
    )


    # ========================================================
    # Plot
    # ========================================================

    for row_idx, exp in enumerate(experiments):

        x = np.arange(
            len(exp["xticklabels"])
        )


        for col_idx, (
            metric_name,
            item
        ) in enumerate(
            exp["metrics"].items()
        ):

            ax = axes[
                row_idx,
                col_idx
            ]

            mean = item["mean"]

            std = item["std"]


            # =================================================
            # Line
            # =================================================

            ax.plot(

                x,
                mean,

                marker="o",

                markersize=MARKER_SIZE,

                linewidth=LINE_WIDTH,

                color=exp["line_color"],

                zorder=4
            )


            # =================================================
            # Mean ± std
            # =================================================

            ax.fill_between(

                x,

                mean - std,

                mean + std,

                color=exp["fill_color"],

                alpha=0.32,

                linewidth=0,

                zorder=1
            )


            # =================================================
            # Best point
            # =================================================

            best_idx, _ = (
                get_best_and_second(mean)
            )


            for i in best_idx:

                ax.scatter(

                    x[i],

                    mean[i],

                    s=42,

                    color=exp["best_color"],

                    edgecolor="black",

                    linewidth=0.8,

                    zorder=6
                )


            # =================================================
            # X ticks
            # =================================================

            ax.set_xticks(x)

            ax.set_xticklabels(

                exp["xticklabels"],

                fontsize=FONT_SMALL
            )


            # =================================================
            # IMPORTANT:
            # No xlabel here.
            #
            # We put the experiment description below the
            # entire row instead.
            # =================================================

            ax.set_xlabel("")


            # =================================================
            # Y label
            # =================================================

            ax.set_ylabel(

                metric_name,

                fontsize=FONT_SMALL,

                labelpad=3
            )


            # =================================================
            # Adaptive y range
            # =================================================

            y_min = max(

                0.0,

                np.min(
                    mean - std
                ) - 0.015
            )


            y_max = min(

                1.0,

                np.max(
                    mean + std
                ) + 0.015
            )


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


            # =================================================
            # Grid
            # =================================================

            ax.grid(

                True,

                linestyle="--",

                linewidth=GRID_WIDTH,

                alpha=0.28,

                zorder=0
            )


            # =================================================
            # Spines
            # =================================================

            ax.spines[
                "top"
            ].set_visible(
                False
            )


            ax.spines[
                "right"
            ].set_visible(
                False
            )


            ax.spines[
                "left"
            ].set_linewidth(
                SPINE_WIDTH
            )


            ax.spines[
                "bottom"
            ].set_linewidth(
                SPINE_WIDTH
            )


            # =================================================
            # Tick style
            # =================================================

            ax.tick_params(

                axis="both",

                labelsize=FONT_SMALL,

                length=2.5,

                width=0.65,

                pad=1.5
            )


    # ========================================================
    # Layout
    #
    # Larger hspace is intentional:
    # captions will sit BELOW each row.
    # ========================================================

    fig.subplots_adjust(

        left=0.075,

        right=0.995,

        top=0.975,

        bottom=0.12,

        wspace=0.38,

        hspace=0.90
    )


    # ========================================================
    # IMPORTANT
    #
    # Draw canvas first so axes positions are final.
    # ========================================================

    fig.canvas.draw()


    # ========================================================
    # Captions BELOW each row
    # ========================================================

    # Distance between bottom of subplot row and caption.
    #
    # Smaller -> closer to plots
    # Larger  -> farther below
    CAPTION_GAP = 0.065


    for row_idx, exp in enumerate(experiments):

        row_axes = axes[
            row_idx,
            :
        ]


        # ----------------------------------------------------
        # Actual left/right boundary of the WHOLE row
        # ----------------------------------------------------

        row_left = min(

            ax.get_position().x0

            for ax in row_axes
        )


        row_right = max(

            ax.get_position().x1

            for ax in row_axes
        )


        # ----------------------------------------------------
        # True horizontal center of the entire 4-panel row
        # ----------------------------------------------------

        row_center = (

            row_left
            +
            row_right

        ) / 2.0


        # ----------------------------------------------------
        # Bottom boundary of this row
        # ----------------------------------------------------

        row_bottom = min(

            ax.get_position().y0

            for ax in row_axes
        )


        # ----------------------------------------------------
        # Caption BELOW row
        # ----------------------------------------------------

        caption_y = (

            row_bottom

            -

            CAPTION_GAP
        )


        # ----------------------------------------------------
        # Add centered caption
        # ----------------------------------------------------

        fig.text(

            row_center,

            caption_y,

            exp["caption"],

            ha="center",

            va="top",

            fontsize=FONT_CAPTION
        )


    # ========================================================
    # Save
    # ========================================================

    save_figure(

        fig,

        "panel_middle_parameter_sensitivity"
    )


    return fig

# ============================================================
# ============================================================
#
# FIGURE 3
#
# RIGHT:
# Top-k sensitivity
#
# ============================================================
# ============================================================

def make_topk_sensitivity():


    # ========================================================
    # Data
    # ========================================================

    top_k = np.array([
        10,
        20,
        50,
        100,
        200
    ])


    gap_abide1 = np.array([

        0.0004,

        0.0051,

        0.0028,

        0.0003,

        0.0015
    ])


    gap_abide2 = np.array([

        0.0043,

        0.0018,

        0.0012,

        0.0003,

        0.0012
    ])


    shared_count = np.array([

        0,

        0,

        2,

        5,

        16
    ])


    shared_ratio = np.array([

        0.0,

        0.0,

        4.0,

        5.0,

        8.0
    ])


    x = np.arange(
        len(top_k)
    )


    # ========================================================
    # Figure
    # ========================================================

    fig, ax_gap = plt.subplots(

        figsize=(
            FIG_W_SIDE,
            FIG_H
        ),

        dpi=300
    )


    ax_shared = (
        ax_gap.twinx()
    )


    # ========================================================
    # Bars
    # ========================================================

    bars = ax_shared.bar(

        x,

        shared_ratio,

        width=0.52,

        alpha=0.22,

        label="Shared-edge ratio",

        zorder=1
    )


    # ========================================================
    # ABIDE-I
    # ========================================================

    line1, = ax_gap.plot(

        x,

        gap_abide1,

        marker="o",

        linewidth=LINE_WIDTH,

        markersize=MARKER_SIZE,

        label="ABIDE-I gap",

        zorder=4
    )


    # ========================================================
    # ABIDE-II
    # ========================================================

    line2, = ax_gap.plot(

        x,

        gap_abide2,

        marker="s",

        linestyle="--",

        linewidth=LINE_WIDTH,

        markersize=MARKER_SIZE,

        label="ABIDE-II gap",

        zorder=4
    )


    # ========================================================
    # ABIDE-I labels
    # ========================================================

    a1_offsets = {

        0: (0, 7),

        1: (0, 7),

        2: (0, 7),

        4: (0, 7)
    }


    for i, y in enumerate(
        gap_abide1
    ):


        # Top-k = 100:
        # shared value with ABIDE-II
        if i == 3:

            continue


        dx, dy = (
            a1_offsets[i]
        )


        ax_gap.annotate(

            f"{y:.4f}",

            (
                x[i],
                y
            ),

            xytext=(
                dx,
                dy
            ),

            textcoords=
                "offset points",

            ha="center",

            va="bottom",

            fontsize=FONT_VALUE,

            zorder=10,

            clip_on=False
        )


    # ========================================================
    # ABIDE-II labels
    # ========================================================

    a2_offsets = {

        0: (0, -10),

        1: (0, -10),

        2: (0, -10),

        4: (0, -10)
    }


    for i, y in enumerate(
        gap_abide2
    ):


        if i == 3:

            continue


        dx, dy = (
            a2_offsets[i]
        )


        ax_gap.annotate(

            f"{y:.4f}",

            (
                x[i],
                y
            ),

            xytext=(
                dx,
                dy
            ),

            textcoords=
                "offset points",

            ha="center",

            va="top",

            fontsize=FONT_VALUE,

            zorder=10,

            clip_on=False
        )


    # ========================================================
    # Shared Top-k = 100 label
    # ========================================================

    ax_gap.annotate(

        "0.0003",

        (
            x[3],
            gap_abide1[3]
        ),

        xytext=(
            0,
            8
        ),

        textcoords=
            "offset points",

        ha="center",

        va="bottom",

        fontsize=FONT_VALUE,

        zorder=10,

        clip_on=False
    )


    # ========================================================
    # Bar labels
    # ========================================================

    for (
        bar,
        count,
        ratio,
        k
    ) in zip(

        bars,

        shared_count,

        shared_ratio,

        top_k
    ):


        x_center = (

            bar.get_x()

            +

            bar.get_width()
            / 2
        )


        if ratio == 0:

            y_text = 0.25

        else:

            y_text = (
                ratio + 0.25
            )


        ax_shared.text(

            x_center,

            y_text,

            f"{count}/{k} "
            f"({ratio:.0f}%)",

            ha="center",

            va="bottom",

            fontsize=FONT_VALUE,

            zorder=10,

            clip_on=False
        )


    # ========================================================
    # X axis
    # ========================================================

    ax_gap.set_xticks(x)


    ax_gap.set_xticklabels(

        [
            str(v)
            for v in top_k
        ],

        fontsize=FONT_TICK
    )


    ax_gap.set_xlabel(

        "Top-$k$",

        fontsize=FONT_LABEL
    )


    # ========================================================
    # Y axes
    # ========================================================

    ax_gap.set_ylabel(

        "Effect-size gap at cutoff",

        fontsize=FONT_LABEL
    )


    ax_shared.set_ylabel(

        "Shared-edge ratio (%)",

        fontsize=FONT_LABEL
    )


    ax_gap.tick_params(

        axis="y",

        labelsize=FONT_TICK
    )


    ax_shared.tick_params(

        axis="y",

        labelsize=FONT_TICK
    )


    ax_gap.set_ylim(

        0,

        0.0061
    )


    ax_shared.set_ylim(

        0,

        10
    )


    # ========================================================
    # Grid
    # ========================================================

    ax_gap.grid(

        axis="y",

        linewidth=GRID_WIDTH,

        alpha=0.35,

        zorder=0
    )


    # ========================================================
    # Legend
    # ========================================================

    handles = [
        line1,
        line2,
        bars
    ]

    labels = [
        "ABIDE-I gap",
        "ABIDE-II gap",
        "Shared-edge ratio"
    ]

    ax_gap.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=1.6,
        columnspacing=0.8
    )


    # ========================================================
    # Style
    # ========================================================

    ax_gap.spines["top"].set_visible(False)
    ax_shared.spines["top"].set_visible(False)


    # ========================================================
    # Layout
    #
    # Make the Top-k panel vertically taller so that its
    # visual height better matches the other two panels
    # when placed side-by-side in the ICLR layout.
    # ========================================================

    fig.subplots_adjust(
        left=0.17,
        right=0.83,
        bottom=0.105,
        top=0.90
    )


    save_figure(

        fig,

        "panel_right_topk_sensitivity"
    )


    return fig


# ============================================================
# ============================================================
#
# MAIN
#
# ============================================================
# ============================================================

if __name__ == "__main__":


    # --------------------------------------------------------
    # Generate LEFT
    # --------------------------------------------------------

    fig1 = make_ablation()


    # --------------------------------------------------------
    # Generate MIDDLE
    # --------------------------------------------------------

    fig2 = (
        make_parameter_sensitivity()
    )


    # --------------------------------------------------------
    # Generate RIGHT
    # --------------------------------------------------------

    fig3 = (
        make_topk_sensitivity()
    )


    print()

    print(
        "========================================"
    )

    print(
        "All figures have been generated."
    )

    print(
        "========================================"
    )

    print()


    print(
        "LEFT:"
    )

    print(
        "  panel_left_ablation.pdf"
    )

    print(
        "  panel_left_ablation.png"
    )


    print()


    print(
        "MIDDLE:"
    )

    print(
        "  panel_middle_parameter_sensitivity.pdf"
    )

    print(
        "  panel_middle_parameter_sensitivity.png"
    )


    print()


    print(
        "RIGHT:"
    )

    print(
        "  panel_right_topk_sensitivity.pdf"
    )

    print(
        "  panel_right_topk_sensitivity.png"
    )


    print()

    print(
        "Recommended ICLR LaTeX widths:"
    )

    print(
        r"LEFT   = 0.27\linewidth"
    )

    print(
        r"MIDDLE = 0.42\linewidth"
    )

    print(
        r"RIGHT  = 0.27\linewidth"
    )


    print()

    print(
        "Figure sizes:"
    )

    print(
        f"LEFT   = {FIG_W_SIDE} x {FIG_H} inch"
    )

    print(
        f"MIDDLE = {FIG_W_MIDDLE} x {FIG_H} inch"
    )

    print(
        f"RIGHT  = {FIG_W_SIDE} x {FIG_H} inch"
    )


    # --------------------------------------------------------
    # Show figures
    # --------------------------------------------------------

    plt.show()