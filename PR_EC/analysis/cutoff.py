import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Data
# ============================================================

top_k = np.array([10, 20, 50, 100, 200])

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

x = np.arange(len(top_k))


# ============================================================
# Figure
# 30% smaller than 7.0 x 3.7
# ============================================================

fig, ax_gap = plt.subplots(
    figsize=(3.2, 3.2),
    dpi=300
)

ax_shared = ax_gap.twinx()


# ============================================================
# Bars
# ============================================================

bars = ax_shared.bar(
    x,
    shared_ratio,
    width=0.52,
    alpha=0.22,
    label="Shared-edge ratio",
    zorder=1
)


# ============================================================
# Lines
# ============================================================

line1, = ax_gap.plot(
    x,
    gap_abide1,
    marker="o",
    linewidth=1.4,
    markersize=4.2,
    label="ABIDE-I gap",
    zorder=4
)

line2, = ax_gap.plot(
    x,
    gap_abide2,
    marker="s",
    linestyle="--",
    linewidth=1.4,
    markersize=4.0,
    label="ABIDE-II gap",
    zorder=4
)


# ============================================================
# ABIDE-I labels
# ============================================================

a1_offsets = {
    0: (0, 7),
    1: (0, 7),
    2: (0, 7),
    4: (0, 7),
}

for i, y in enumerate(gap_abide1):

    # Top-k = 100 will be labeled only once later
    if i == 3:
        continue

    dx, dy = a1_offsets[i]

    ax_gap.annotate(
        f"{y:.4f}",
        (x[i], y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.3,
        zorder=10,
        clip_on=False
    )


# ============================================================
# ABIDE-II labels
# ============================================================

a2_offsets = {
    0: (0, -10),
    1: (0, -10),
    2: (0, -10),
    4: (0, -10),
}

for i, y in enumerate(gap_abide2):

    # Top-k = 100 will be labeled only once later
    if i == 3:
        continue

    dx, dy = a2_offsets[i]

    ax_gap.annotate(
        f"{y:.4f}",
        (x[i], y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=6.3,
        zorder=10,
        clip_on=False
    )


# ============================================================
# Shared label at Top-k = 100
# Both ABIDE-I and ABIDE-II = 0.0003
# ============================================================

ax_gap.annotate(
    "0.0003",
    (x[3], gap_abide1[3]),
    xytext=(0, 8),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=6.3,
    zorder=10,
    clip_on=False
)


# ============================================================
# Bar labels
# ============================================================

for bar, count, ratio, k in zip(
    bars,
    shared_count,
    shared_ratio,
    top_k
):

    x_center = (
        bar.get_x()
        + bar.get_width() / 2
    )

    if ratio == 0:
        y_text = 0.25
    else:
        y_text = ratio + 0.25

    ax_shared.text(
        x_center,
        y_text,
        f"{count}/{k} ({ratio:.0f}%)",
        ha="center",
        va="bottom",
        fontsize=6.2,
        zorder=10,
        clip_on=False
    )


# ============================================================
# X-axis
# ============================================================

ax_gap.set_xticks(x)

ax_gap.set_xticklabels(
    [str(v) for v in top_k],
    fontsize=7
)

ax_gap.set_xlabel(
    "Top-$k$",
    fontsize=7.5
)


# ============================================================
# Y-axes
# ============================================================

ax_gap.set_ylabel(
    "Effect-size gap at cutoff",
    fontsize=7.5
)

ax_shared.set_ylabel(
    "Shared-edge ratio (%)",
    fontsize=7.5
)

ax_gap.tick_params(
    axis="y",
    labelsize=7
)

ax_shared.tick_params(
    axis="y",
    labelsize=7
)

ax_gap.set_ylim(
    0,
    0.0061
)

ax_shared.set_ylim(
    0,
    10
)


# ============================================================
# Grid
# ============================================================

ax_gap.grid(
    axis="y",
    linewidth=0.45,
    alpha=0.35,
    zorder=0
)


# ============================================================
# Legend
# ============================================================

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
    bbox_to_anchor=(0.5, 1.01),
    ncol=3,
    frameon=False,
    fontsize=6.2,
    handlelength=1.7,
    columnspacing=1.0
)


# ============================================================
# Style
# ============================================================

ax_gap.spines["top"].set_visible(False)
ax_shared.spines["top"].set_visible(False)


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.13,
    right=0.87,
    bottom=0.22,
    top=0.80
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "topk_sensitivity.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "topk_sensitivity.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
