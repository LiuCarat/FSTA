import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# 0. 全局样式设置
# 以后主要修改这里即可
# ============================================================

# ---------- 字体 ----------
# FONT_FAMILY = "Times New Roman"

FONT_TITLE  = 9     # 标题
FONT_LABEL  = 7     # x/y轴标题
FONT_XTICK  = 4     # ROI名称
FONT_YTICK  = 7.0     # y轴数字
FONT_LEGEND = 7    # 图例

# plt.rcParams["font.family"] = FONT_FAMILY


# ---------- 图尺寸 ----------
FIG_WIDTH = 5.5
FIG_HEIGHT = 3.2


# ---------- 柱宽 ----------
BAR_WIDTH = 0.92


# ---------- 渐变颜色 ----------
# 数值越大颜色越深
#
# Enhanced:
# Blues / Greens / Purples / BuPu 等
#
# Reduced:
# Oranges / Reds / YlOrRd / PuRd 等

ENHANCED_CMAP = plt.cm.Oranges
REDUCED_CMAP  = plt.cm.Blues

# 控制渐变深浅范围
# 左侧使用 DARK，右侧使用 LIGHT
COLOR_DARK  = 0.88
COLOR_LIGHT = 0.32


# ============================================================
# 1. 数据
# ============================================================

data = [
    ('PreCG.L', 0, 6), ('PreCG.R', 0, 0),
    ('SFGdor.L', 0, 4), ('SFGdor.R', 0, 2),
    ('ORBsup.L', 0, 2), ('ORBsup.R', 0, 0),
    ('MFG.L', 0, 2), ('MFG.R', 0, 0),
    ('ORBmid.L', 0, 3), ('ORBmid.R', 0, 0),
    ('IFGoperc.L', 0, 0), ('IFGoperc.R', 4, 2),
    ('IFGtriang.L', 4, 0), ('IFGtriang.R', 6, 4),
    ('ORBinf.L', 0, 0), ('ORBinf.R', 0, 1),
    ('ROL.L', 2, 0), ('ROL.R', 0, 2),
    ('SMA.L', 0, 0), ('SMA.R', 0, 0),
    ('OLF.L', 0, 0), ('OLF.R', 0, 1),
    ('SFGmed.L', 3, 3), ('SFGmed.R', 0, 8),
    ('ORBsupmed.L', 1, 5), ('ORBsupmed.R', 4, 11),
    ('REC.L', 2, 0), ('REC.R', 6, 2),
    ('INS.L', 0, 4), ('INS.R', 1, 0),
    ('ACG.L', 0, 3), ('ACG.R', 4, 8),
    ('DCG.L', 0, 0), ('DCG.R', 6, 0),
    ('PCG.L', 2, 7), ('PCG.R', 6, 7),
    ('HIP.L', 8, 0), ('HIP.R', 11, 0),
    ('PHG.L', 2, 1), ('PHG.R', 1, 0),
    ('AMYG.L', 4, 0), ('AMYG.R', 4, 0),
    ('CAL.L', 0, 0), ('CAL.R', 0, 0),
    ('CUN.L', 0, 1), ('CUN.R', 0, 0),
    ('LING.L', 0, 0), ('LING.R', 0, 0),
    ('SOG.L', 1, 0), ('SOG.R', 0, 2),
    ('MOG.L', 0, 0), ('MOG.R', 1, 4),
    ('IOG.L', 0, 3), ('IOG.R', 0, 0),
    ('FFG.L', 7, 8), ('FFG.R', 5, 0),
    ('PoCG.L', 2, 4), ('PoCG.R', 2, 4),
    ('SPG.L', 4, 1), ('SPG.R', 2, 4),
    ('IPL.L', 0, 0), ('IPL.R', 0, 0),
    ('SMG.L', 0, 4), ('SMG.R', 2, 0),
    ('ANG.L', 0, 10), ('ANG.R', 0, 0),
    ('PCUN.L', 0, 9), ('PCUN.R', 2, 2),
    ('PCL.L', 2, 2), ('PCL.R', 4, 4),
    ('CAU.L', 5, 8), ('CAU.R', 1, 4),
    ('PUT.L', 1, 0), ('PUT.R', 5, 0),
    ('PAL.L', 2, 0), ('PAL.R', 9, 0),
    ('THA.L', 8, 6), ('THA.R', 14, 7),
    ('HES.L', 1, 0), ('HES.R', 0, 1),
    ('STG.L', 6, 1), ('STG.R', 8, 0),
    ('TPOsup.L', 0, 0), ('TPOsup.R', 0, 0),
    ('MTG.L', 18, 3), ('MTG.R', 15, 10),
    ('TPOmid.L', 1, 1), ('TPOmid.R', 7, 8),
    ('ITG.L', 4, 2), ('ITG.R', 2, 3)
]


# ============================================================
# 2. 提取数据
# ============================================================

roi = np.array([r[0] for r in data])
enhanced = np.array([r[1] for r in data])
reduced = np.array([r[2] for r in data])

total = enhanced + reduced


# ============================================================
# 3. 去掉没有任何 replicated EC 的脑区
# ============================================================

mask = total > 0

roi = roi[mask]
enhanced = enhanced[mask]
reduced = reduced[mask]
total = total[mask]


# ============================================================
# 4. 按总数量从高到低排序
# ============================================================

order = np.lexsort(
    (
        roi,        # 第三优先级
        -enhanced,  # 第二优先级
        -total      # 第一优先级
    )
)

roi = roi[order]
enhanced = enhanced[order]
reduced = reduced[order]
total = total[order]


# ============================================================
# 5. 创建渐变颜色
#
# 左边最深 -> 右边最浅
# ============================================================

n_roi = len(roi)

color_position = np.linspace(
    COLOR_DARK,
    COLOR_LIGHT,
    n_roi
)

enhanced_colors = ENHANCED_CMAP(color_position)
reduced_colors = REDUCED_CMAP(color_position)


# ============================================================
# 6. 绘图
# ============================================================

x = np.arange(n_roi)

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)


# Enhanced：向上
ax.bar(
    x,
    enhanced,
    width=BAR_WIDTH,
    color=enhanced_colors,
    label="ASD-enhanced"
)


# Reduced：向下
ax.bar(
    x,
    -reduced,
    width=BAR_WIDTH,
    color=reduced_colors,
    label="ASD-reduced"
)


# ============================================================
# 7. 中心线
# ============================================================

ax.axhline(
    0,
    color="black",
    linewidth=0.7
)


# ============================================================
# 8. X轴
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels(
    roi,
    rotation=90,
    ha="center",
    va="top",
    fontsize=FONT_XTICK
)

ax.tick_params(
    axis="x",
    length=0,
    pad=1
)


# ============================================================
# Y-axis
# ============================================================

top_ylim = enhanced.max() + 2   # 上面自动
bottom_ylim = 13                # 下面固定到 15

ax.set_ylim(-bottom_ylim, top_ylim)

ax.yaxis.set_major_formatter(
    FuncFormatter(
        lambda y, _: f"{abs(int(y))}"
    )
)

ax.tick_params(
    axis="y",
    labelsize=FONT_YTICK
)

ax.set_ylabel(
    "Replicated EC involvement count",
    fontsize=FONT_LABEL
)




# ============================================================
# 11. 图例
#
# 因为每个柱颜色都不同，
# legend 默认会取第一根柱子的深色
# ============================================================

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.06),
    fontsize=FONT_LEGEND,
    handlelength=1.4,
    columnspacing=1.0
)


# ============================================================
# 12. 网格
# ============================================================

ax.grid(
    axis="y",
    linewidth=0.4,
    alpha=0.20
)

ax.set_axisbelow(True)


# ============================================================
# 13. 边框
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)


# ============================================================
# 14. 去除左右空白
# ============================================================

ax.margins(x=0)

ax.set_xlim(
    -0.5,
    n_roi - 0.5
)


# ============================================================
# 15. Layout
# ============================================================

plt.subplots_adjust(
    left=0.07,
    right=0.995,
    top=0.81,
    bottom=0.31
)


# ============================================================
# 16. 保存
# ============================================================

plt.savefig(
    "panelA_gradient.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "panelA_gradient.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
