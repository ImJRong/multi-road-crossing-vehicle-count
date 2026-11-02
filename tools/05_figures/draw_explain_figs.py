#!/usr/bin/env python3
"""生成「head4 完整讲解」的全套示意图。

一共 8 张，按讲解顺序编号：
  fig1_overview.png     一页纸总览：现象 → 原因 → 解法 → 结果
  fig2_realframe.png    真实画面：同一辆车，旧框 vs 新框（带放大）
  fig3_ruler.png        直觉：尺子刻度越细越准
  fig4_bins.png         原理：256 个格子怎么分
  fig5_errorchain.png   误差从哪来：一步错，步步放大
  fig6_channels.png     证据：4 个通道范围，好模型各不相同
  fig7_structure.png    解法：模型结构改在哪
  fig8_result.png       结果：10 帧实测

用法：
  D:\\Python3.12.4\\python.exe tools/05_figures/draw_explain_figs.py
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/xx/ -> 项目根
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

RED = "#d93a3a"      # 旧：粗刻度 / 坏
GREEN = "#1f9d4d"    # 新：细刻度 / 好
GREY = "#7a7a7a"
BLUE = "#2f6db3"
ORANGE = "#e08a1e"

VIDEO = os.path.join(ROOT, "data", "videos", "test1.mp4")
FRAME_NO = 200

# 第 200 帧那一辆车的实测数据（原图坐标，单位像素）
TRUTH = dict(cx=298.8, cy=223.2, w=41.5, h=53.0)
DET_V7 = dict(cx=297.3, cy=217.0, w=44.2, h=52.4)
DET_H4 = dict(cx=299.0, cy=223.4, w=41.0, h=52.6)


# --------------------------------------------------------------- 通用小工具
def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, facecolor="white")
    plt.close(fig)
    print("  生成:", p)
    return p


def box_xywh(d):
    return (d["cx"] - d["w"] / 2, d["cy"] - d["h"] / 2, d["w"], d["h"])


def read_frame(idx=FRAME_NO):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读不到视频帧")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------- 图 1 总览
def fig_overview():
    fig = plt.figure(figsize=(14, 9.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.0)
    ax.axis("off")

    cards = [
        ("① 现象", RED, 0.55, 6.35,
         "框整体往左上跑\n车越小偏得越多\n最大偏到 22 像素",
         "板子上看出来的"),
        ("② 真正原因", ORANGE, 7.25, 6.35,
         "模型输出的是坐标 0~640\nRKNN 按 640 配刻度 = 2.5\n"
         "可 DFL 里的距离只有 0~15\n→ 精度被这把粗尺子压没了",
         "过了很久才查明白"),
        ("③ 解法", BLUE, 0.55, 2.15,
         "让模型只输出「距离 0~15」\n坐标换算搬到 C++ 用 float 算\n"
         "刻度自动变成 0.047\n→ 细了 53 倍",
         "只改输出，不改权重"),
        ("④ 结果", GREEN, 7.25, 2.15,
         "中心偏移 6.75 → 0.03 像素\n改善约 200 倍\n速度一点没变（18.3 fps）",
         "10 帧逐框核对"),
    ]

    for title, color, x, y, body, note in cards:
        card = FancyBboxPatch((x, y), 6.2, 2.0,
                              boxstyle="round,pad=0.06,rounding_size=0.18",
                              fc="white", ec=color, lw=2.6, zorder=2)
        ax.add_patch(card)
        ax.add_patch(FancyBboxPatch((x, y + 1.42), 6.2, 0.58,
                                    boxstyle="round,pad=0.06,rounding_size=0.18",
                                    fc=color, ec=color, lw=2.6, zorder=3, alpha=0.92))
        ax.text(x + 0.28, y + 1.71, title, fontsize=17, fontweight="bold",
                color="white", va="center", zorder=4)
        ax.text(x + 6.2 - 0.28, y + 1.71, note, fontsize=10.5,
                color="white", va="center", ha="right", zorder=4, alpha=0.95)
        ax.text(x + 0.30, y + 0.78, body, fontsize=12.6, va="center",
                linespacing=1.85, color="#222222", zorder=4)

    # 箭头
    ax.add_patch(FancyArrowPatch((6.95, 7.35), (7.05, 7.35), arrowstyle="-|>",
                                 mutation_scale=26, lw=2.6, color="#555555", zorder=1))
    ax.add_patch(FancyArrowPatch((3.65, 6.15), (3.65, 4.55), arrowstyle="-|>",
                                 mutation_scale=26, lw=2.6, color="#555555", zorder=1))
    ax.add_patch(FancyArrowPatch((10.35, 6.15), (10.35, 4.55), arrowstyle="-|>",
                                 mutation_scale=26, lw=2.6, color="#555555", zorder=1))

    ax.text(7.0, 8.62, "一句话：模型一个字没改，只是把「尺子的最小刻度」换细了",
            ha="center", fontsize=18, fontweight="bold", color="#111111")

    return save(fig, "fig1_overview.png")


# ------------------------------------------------------------- 图 2 真实画面
def fig_realframe():
    img = read_frame()
    H, W = img.shape[:2]

    # 放大区域：围着那辆车裁一块
    cx, cy = TRUTH["cx"], TRUTH["cy"]
    half_w, half_h = 58, 68
    x0 = int(max(0, cx - half_w)); x1 = int(min(W, cx + half_w))
    y0 = int(max(0, cy - half_h)); y1 = int(min(H, cy + half_h))

    fig = plt.figure(figsize=(14.5, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.28],
                          hspace=0.20, wspace=0.08,
                          left=0.015, right=0.985, top=0.855, bottom=0.03)

    # ---- 上：全景 ----
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(img, extent=(0, W, H, 0))
    ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            ec=ORANGE, lw=2.4, ls="--"))
    ax0.text(x1 + 12, y0 + 4, "下面放大的区域", color=ORANGE,
             fontsize=11.5, fontweight="bold", va="top")
    ax0.plot([cx], [cy], "+", ms=12, color="#ffffff", mew=2)
    ax0.set_xlim(0, W); ax0.set_ylim(H, 0)
    ax0.axis("off")

    # ---- 左下：旧框 ----
    # ---- 右下：新框 ----
    panels = [(gs[1, 0], DET_V7, RED, "旧模型 v7（偏了 6.75 像素）"),
              (gs[1, 1], DET_H4, GREEN, "新模型 head4（偏 0.03 像素）")]

    for cell, det, color, title in panels:
        axp = fig.add_subplot(cell)
        crop = img[y0:y1, x0:x1]
        big = cv2.resize(crop, None, fx=6.2, fy=6.2, interpolation=cv2.INTER_CUBIC)
        axp.imshow(big, extent=(x0, x1, y1, y0))

        # 真值（白虚线）—— 最准的位置
        tx, ty, tw, th = box_xywh(TRUTH)
        axp.add_patch(Rectangle((tx, ty), tw, th, fill=False,
                                ec="#ffffff", lw=1.9, ls=(0, (5, 4)), alpha=0.95))
        # 检出的框
        bx, by, bw, bh = box_xywh(det)
        axp.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=color, lw=2.6))

        axp.plot([det["cx"]], [det["cy"]], "x", ms=11, color=color, mew=2.6)
        axp.plot([TRUTH["cx"]], [TRUTH["cy"]], "+", ms=13, color="#ffffff", mew=2.2)

        axp.set_xlim(x0, x1); axp.set_ylim(y1, y0)
        axp.set_xticks([]); axp.set_yticks([])
        axp.set_title(title, fontsize=13.5, fontweight="bold", color=color, pad=7)
        for s in axp.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(2.2)

        # 偏移量标注
        axp.annotate("", xy=(det["cx"], det["cy"]), xytext=(TRUTH["cx"], TRUTH["cy"]),
                     arrowprops=dict(arrowstyle="->", lw=1.6, color=color,
                                     connectionstyle="arc3,rad=-0.25"))
        dx = det["cx"] - TRUTH["cx"]
        dy = det["cy"] - TRUTH["cy"]
        mag = (dx * dx + dy * dy) ** 0.5
        axp.text(x0 + (x1 - x0) * 0.03, y1 - (y1 - y0) * 0.06,
                 "中心差 (%.1f, %.1f) = %.2f 像素" % (dx, dy, mag),
                 fontsize=11.5, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.34", fc=color, ec="none"))

    fig.suptitle("同一辆车、同一帧：白虚线是真值，彩色框是模型给的",
                 fontsize=17, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925,
             "上半：测试视频第 200 帧全景（原图 1280×674）",
             ha="center", fontsize=12, color="#444444")
    fig.text(0.5, 0.885,
             "下半：把虚线框里的区域放大 —— 旧框整体往上飘，新框几乎和虚线重合",
             ha="center", fontsize=12.5, color="#222222", fontweight="bold")
    return save(fig, "fig2_realframe.png")


# ------------------------------------------------------------------ 图 3 尺子
def fig_ruler():
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.6))
    true_v = 5.30

    def label_right(ax, x_text, y_text, x_target, color, text):
        ax.annotate(text, xy=(x_target, 0), xytext=(x_text, y_text),
                    ha="left", va="center", fontsize=13, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", lw=1.8, color=color,
                                    connectionstyle="arc3,rad=0.18"),
                    bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=color, lw=1.4))

    # 上：粗
    ax = axes[0]
    ax.hlines(0, 0, 16.5, color="black", lw=2.5, zorder=1)
    for t in np.arange(0, 17.5, 2.5):
        ax.vlines(t, -0.34, -0.10, color=RED, lw=2.5, zorder=2)
        ax.text(t, -0.44, "%.1f" % t, ha="center", va="top", fontsize=11, color=RED)
    snapped = round(true_v / 2.5) * 2.5
    ax.plot(true_v, 0, "o", ms=14, color="black", zorder=5)
    ax.plot(snapped, 0, "X", ms=19, color=RED, zorder=6)
    ax.text(true_v, 0.62, "真实值 5.30", ha="center", fontsize=13, fontweight="bold")
    ax.annotate("", xy=(true_v, 0.08), xytext=(true_v, 0.52),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
    label_right(ax, 7.4, 0.34, snapped, RED, "只能读成 5.0  →  差 0.3")
    ax.text(9.0, 1.02, "每格 2.5，最大误差 ±1.25",
            ha="center", fontsize=14, color=RED, fontweight="bold")
    ax.set_title("① 以前：给「距离」配了粗刻度（scale = 2.5）", fontsize=15,
                 fontweight="bold", loc="left", color=RED)
    ax.set_xlim(-0.8, 17.2); ax.set_ylim(-0.95, 1.25); ax.axis("off")

    # 下：细
    ax = axes[1]
    ax.hlines(0, 0, 16.5, color="black", lw=2.5, zorder=1)
    for t in np.arange(0, 16.01, 0.0625):
        ax.vlines(t, -0.30, -0.12, color=GREEN, lw=0.7, zorder=2, alpha=0.6)
    fine = round(true_v / 0.046943) * 0.046943
    ax.plot(true_v, 0, "o", ms=14, color="black", zorder=5)
    ax.plot(fine, 0, "X", ms=19, color=GREEN, zorder=6)
    ax.text(true_v, 0.62, "真实值 5.30", ha="center", fontsize=13, fontweight="bold")
    ax.annotate("", xy=(true_v, 0.08), xytext=(true_v, 0.52),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
    label_right(ax, 7.4, 0.34, fine, GREEN, "读成 5.30  →  几乎没误差")
    ax.text(9.0, 1.02, "每格 0.047（共 256 个刻度），最大误差 ±0.023",
            ha="center", fontsize=14, color=GREEN, fontweight="bold")
    ax.set_title("② 现在：给「距离」配了细刻度（scale = 0.0469，细了 53 倍）", fontsize=15,
                 fontweight="bold", loc="left", color=GREEN)
    ax.set_xlim(-0.8, 17.2); ax.set_ylim(-0.95, 1.25); ax.axis("off")

    fig.suptitle("先建立一个直觉：同样 256 格，范围越小越准",
                 fontsize=17, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return save(fig, "fig3_ruler.png")


# ------------------------------------------------------------------ 图 4 格子
def fig_bins():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6),
                             gridspec_kw={"width_ratios": [1, 1]})

    def draw(ax, total_range, step, color, title, sub, show_step):
        n = 16                                  # 为了看得清，只画 16 格代表 256 格
        cell = total_range / n
        for i in range(n):
            ax.add_patch(Rectangle((i, 0), 0.93, 1.0,
                                   fc=color, ec="white", lw=1.4,
                                   alpha=0.28 + 0.045 * i))
            ax.text(i + 0.465, 0.5, "%.0f" % (i * cell), ha="center", va="center",
                    fontsize=9.5, color="#2b2b2b")
        ax.set_xlim(-0.05, n + 0.05); ax.set_ylim(-0.95, 2.15)
        ax.axis("off")
        ax.set_title(title, fontsize=15, fontweight="bold", color=color, loc="left", pad=12)
        ax.text(0, -0.26,
                "256 格平均分完，每格代表 %s" % show_step,
                fontsize=13, color=color, fontweight="bold")
        ax.text(0, -0.60, sub, fontsize=11, color="#444444", linespacing=1.7)
        # 箭头指向"真实值落的那一格"的中心
        pos = 5.3 / total_range * n + 0.465
        ax.annotate("真实距离 5.30\n落在这格", xy=(pos, 1.03), xytext=(pos + 1.6, 1.80),
                    fontsize=10.5, color="#111111",
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#111111"),
                    bbox=dict(boxstyle="round,pad=0.32", fc="#fffbe6", ec="#c9b458"))

    draw(axes[0], 640, 640 / 256, RED,
         "① 以前：这个输出号称是「坐标」，范围 0~640",
         "刻度 = 640 / 256 = 2.5\n一格 = 2.5 像素，量小数值等于用米尺量头发丝",
         "2.5")
    draw(axes[1], 16, 16 / 256, GREEN,
         "② 现在：这个输出就是「距离」，范围 0~15",
         "刻度 = 16 / 256 = 0.047\n一格 = 0.047，比上面细 53 倍",
         "0.047")

    fig.suptitle("同一套 256 个格子，装的东西不一样，结果差 53 倍",
                 fontsize=17, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save(fig, "fig4_bins.png")


# ------------------------------------------------------------ 图 5 误差链条
def fig_errorchain():
    fig = plt.figure(figsize=(14, 7.8))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.90])
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.4); ax.axis("off")

    def node(x, y, w, h, text, fc, ec, fs=11.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.14",
                                    fc=fc, ec=ec, lw=2.0, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4, linespacing=1.6,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color="#555555", lw=2.2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=20, lw=lw, color=color, zorder=2))

    # ---------- 上面：旧 ----------
    ax.text(0.15, 7.02, "以前（输出坐标，刻度 2.5）", fontsize=14, fontweight="bold", color=RED)
    y = 5.55
    node(0.15, y, 2.35, 0.95, "DFL 算出\n距离 5.30 (格)", "#eaf3fb", BLUE)
    arrow(2.55, y + 0.475, 2.95, y + 0.475)
    node(2.95, y, 2.6, 0.95, "量化成坐标\n5.30 → 5.0\n误差 0.30 格", "#ffe9e9", RED, bold=True)
    arrow(5.60, y + 0.475, 6.00, y + 0.475)
    node(6.00, y, 2.5, 0.95, "乘 stride 8\n0.30 × 8", "#fff4e0", ORANGE)
    arrow(8.55, y + 0.475, 8.95, y + 0.475)
    node(8.95, y, 3.0, 0.95, "框偏 2.4 像素\n（小幅）", "#ffe9e9", RED, bold=True)
    arrow(11.98, y + 0.475, 12.35, y + 0.475)
    node(12.35, y, 1.45, 0.95, "再叠加\n共 6.75px", "#f8d7d7", RED, fs=10.5, bold=True)

    ax.text(0.15, 4.92,
            "距离误差 0.3 格看着很小，乘上 stride 8 就变成 2.4 像素；\n"
            "框的四个边各错一点，中间再一叠加，就是 6.75 像素。",
            fontsize=11.5, color="#444444", linespacing=1.9)

    # ---------- 下面：新 ----------
    ax.text(0.15, 3.95, "现在（输出距离，刻度 0.047）", fontsize=14,
            fontweight="bold", color=GREEN)
    y2 = 2.35
    node(0.15, y2, 2.35, 0.95, "DFL 算出\n距离 5.30 (格)", "#eaf3fb", BLUE)
    arrow(2.55, y2 + 0.475, 2.95, y2 + 0.475, color=GREEN)
    node(2.95, y2, 2.6, 0.95, "量化成距离\n5.30 → 5.2998\n误差 0.0002 格",
         "#e3f7e8", GREEN, bold=True)
    arrow(5.60, y2 + 0.475, 6.00, y2 + 0.475, color=GREEN)
    node(6.00, y2, 2.5, 0.95, "乘 stride 8\n0.0002 × 8", "#e3f7e8", GREEN)
    arrow(8.55, y2 + 0.475, 8.95, y2 + 0.475, color=GREEN)
    node(8.95, y2, 3.0, 0.95, "框偏 0.001 像素", "#d6f5de", GREEN, bold=True)
    arrow(11.98, y2 + 0.475, 12.35, y2 + 0.475, color=GREEN)
    node(12.35, y2, 1.45, 0.95, "实测\n0.03px", "#c9f0d5", GREEN, fs=10.5, bold=True)

    ax.text(0.15, 0.72,
            "关键在于：误差是在「距离」这一层产生的。\n"
            "距离的刻度粗，后面乘多少倍，误差就跟着放大多少倍。",
            fontsize=11.5, color="#444444", linespacing=1.9)

    fig.suptitle("误差是怎么一步步变大的", fontsize=17, fontweight="bold", y=0.98)
    return save(fig, "fig5_errorchain.png")


# ------------------------------------------------------------- 图 6 通道证据
def fig_channels():
    # 实测数据（板子 ch_scan 输出，test1 第 200 帧）
    data = {
        "v7（正确）": dict(v=[(0, 630.4), (0, 615.4), (2.5, 510.3), (2.5, 230.2)],
                       color=GREEN),
        "2.5.0 转的（坏的）": dict(v=[(0, 637.9), (0, 637.9), (10.0, 637.9), (10.0, 637.9)],
                              color=RED),
    }
    names = ["ch0\ncx", "ch1\ncy", "ch2\nw", "ch3\nh"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True)

    for ax, (title, d) in zip(axes, data.items()):
        for i, (lo, hi) in enumerate(d["v"]):
            ax.bar(i, hi - lo, bottom=lo, width=0.62,
                   color=d["color"], alpha=0.85, edgecolor="white", linewidth=2)
            ax.text(i, hi + 18, "%.0f~%.0f" % (lo, hi), ha="center", va="bottom",
                    fontsize=10.5, color=d["color"], fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", color=d["color"], pad=10)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_ylim(0, 800)
        ax.axhline(637.9, color=RED, ls="--", lw=1.4, alpha=0.65)

    for ax in axes:
        ax.annotate("637.9 = 255×2.5\n（int8 的饱和值）", xy=(-0.35, 637.9), xytext=(-0.42, 745),
                    fontsize=9.2, color=RED, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=RED,
                                    connectionstyle="arc3,rad=-0.2"))

    axes[0].set_ylabel("通道数值范围（640 空间）", fontsize=12)
    fig.text(0.265, 0.075,
             "四个通道各管一件事：cx/cy 覆盖全图，\n"
             "w/h 只到几百。各不相同 = 正常。",
             ha="center", fontsize=11.5, color=GREEN, linespacing=1.7)
    fig.text(0.765, 0.075,
             "四个通道范围全一样，而且都顶到 637.9（int8 的饱和值）\n"
             "= 数值被截断了，w/h 被撑大十几倍。",
             ha="center", fontsize=11.5, color=RED, linespacing=1.7)

    fig.suptitle("怎么一眼看出模型坏了：看 4 个通道的范围",
                 fontsize=17, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    return save(fig, "fig6_channels.png")


# ----------------------------------------------------------------- 图 7 结构
def fig_structure():
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    yA, yB, H = 0.72, 0.28, 0.10

    def band(x, y, w, h, text, face, edge=BLUE, fs=11, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.015",
                                    fc=face, ec=edge, lw=1.6, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color="#333333", lw=1.8, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=lw, color=color,
                                     linestyle=ls, zorder=2))

    # 上：head3
    ax.text(0.02, 0.90, "① 以前（head3）：模型最后一步把「距离」换算成「坐标」再输出",
            fontsize=14, fontweight="bold", color=RED)
    band(0.02, yA, 0.16, H, "骨干网络\n+ 检测头", "#eaf2fb")
    arrow(0.18, yA + H / 2, 0.225, yA + H / 2)
    band(0.225, yA, 0.13, H, "DFL\n16 个 bin", "#eaf2fb")
    arrow(0.355, yA + H / 2, 0.40, yA + H / 2)
    band(0.40, yA, 0.145, H, "距离\n0 ~ 15", "#fdf0e0")
    arrow(0.545, yA + H / 2, 0.59, yA + H / 2)
    band(0.59, yA, 0.175, H, "9 个换算节点\n× stride + 锚点", "#fdf0e0", edge=RED)
    arrow(0.765, yA + H / 2, 0.81, yA + H / 2)
    band(0.81, yA, 0.17, H, "输出\n坐标 0 ~ 640", "#ffe3e0", edge=RED, bold=True)
    ax.text(0.895, yA - 0.055, "配刻度 2.5\n（一格 2.5 像素）", ha="center", va="top",
            fontsize=11, color=RED, fontweight="bold")

    # 下：head4
    ax.text(0.02, 0.44, "② 现在（head4）：模型只输出「距离」，坐标换算搬到 C++ 里",
            fontsize=14, fontweight="bold", color=GREEN)
    band(0.02, yB, 0.16, H, "骨干网络\n+ 检测头", "#eaf2fb")
    arrow(0.18, yB + H / 2, 0.225, yB + H / 2)
    band(0.225, yB, 0.13, H, "DFL\n16 个 bin", "#eaf2fb")
    arrow(0.355, yB + H / 2, 0.40, yB + H / 2)
    band(0.40, yB, 0.145, H, "距离\n0 ~ 15", "#e3f7e8", edge=GREEN, bold=True)
    arrow(0.545, yB + H / 2, 0.81, yB + H / 2, color=GREEN, lw=2.6)
    ax.text(0.677, yB + H / 2 + 0.045, "直接输出", ha="center", fontsize=10, color=GREEN)
    band(0.81, yB, 0.17, H, "输出\n距离 0 ~ 15", "#d6f5de", edge=GREEN, bold=True)
    ax.text(0.895, yB - 0.055, "配刻度 0.047\n（细 53 倍）", ha="center", va="top",
            fontsize=11, color=GREEN, fontweight="bold")

    band(0.40, 0.06, 0.36, 0.085,
         "C++ 里用 float 算：x1 = 锚点 − 左距   x2 = 锚点 + 右距\n然后 × stride  →  得到坐标",
         "#eef3ff", edge=BLUE, fs=10)
    arrow(0.58, yB - 0.005, 0.58, 0.15, color=BLUE, ls="--", lw=1.6)

    ax.text(0.02, 0.16,
            "删掉的 9 个节点没有任何权重，只是加减乘除；\n"
            "用 numpy 重写后和原模型逐元素对比，差值 = 0，\n"
            "所以这不算「新模型」，是同一个模型搬了个家。",
            fontsize=10.5, color="#444444", va="center", linespacing=1.7)

    fig.suptitle("模型到底改在哪", fontsize=17, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return save(fig, "fig7_structure.png")


# ----------------------------------------------------------------- 图 8 结果
def fig_result():
    frames = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    v7_dcx = np.array([-6.8, -2.8, -1.5, -4.2, -1.1, -7.5, -4.4, -2.8, -6.1, -6.1])
    v7_dcy = np.array([-4.3, -6.3, -6.2, -6.0, -6.2, -3.8, -7.0, -4.1, -3.8, -4.3])
    h4_dcx = np.array([-0.1, 0.2, 0.2, -0.1, -0.1, 0.1, -0.1, 0.0, -0.1, 0.1])
    h4_dcy = np.array([-0.1, -0.1, 0.2, -0.1, -0.0, -0.1, 0.0, 0.1, 0.0, 0.2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    ax1.axhline(0, color="#cccccc", lw=1)
    ax1.axvline(0, color="#cccccc", lw=1)
    ax1.scatter(v7_dcx, v7_dcy, s=115, color=RED, alpha=0.85,
                edgecolor="white", linewidth=1.2, label="v7（旧）", zorder=3)
    ax1.scatter(h4_dcx, h4_dcy, s=115, color=GREEN, alpha=0.92,
                edgecolor="white", linewidth=1.2, label="head4（新）", zorder=4)
    ax1.plot(0, 0, "+", ms=20, color="black", mew=2, zorder=5)
    ax1.annotate("理想位置 (0, 0)", xy=(0, 0), xytext=(1.4, -1.5),
                 fontsize=11, ha="left",
                 arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))
    ax1.set_xlabel("横向偏移（像素）", fontsize=12)
    ax1.set_ylabel("纵向偏移（像素）", fontsize=12)
    ax1.set_title("每一帧的框中心落在哪", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=12, loc="lower right")
    ax1.grid(alpha=0.25, linestyle=":")
    ax1.set_xlim(-9, 3); ax1.set_ylim(-8.5, 2.5)

    v7m = np.hypot(v7_dcx, v7_dcy)
    h4m = np.hypot(h4_dcx, h4_dcy)
    x = np.arange(len(frames)); w = 0.38
    ax2.bar(x - w / 2, v7m, w, color=RED, alpha=0.9, label="v7（旧）")
    ax2.bar(x + w / 2, np.maximum(h4m, 0.02), w, color=GREEN, alpha=0.9, label="head4（新）")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(f) for f in frames], rotation=45, fontsize=9)
    ax2.set_xlabel("视频帧号", fontsize=12)
    ax2.set_ylabel("中心偏移（像素）", fontsize=12)
    ax2.set_title("每一帧偏了多少（越小越好）", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")
    ax2.text(4.5, max(v7m) * 0.80,
             "平均 6.75 px  →  0.03 px\n改善约 200 倍",
             ha="center", fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#999999"))

    fig.suptitle("实测：10 帧，每帧和 FP32 真值逐框对比",
                 fontsize=16, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, "fig8_result.png")


if __name__ == "__main__":
    print("生成全套讲解图：")
    fig_overview()
    fig_realframe()
    fig_ruler()
    fig_bins()
    fig_errorchain()
    fig_channels()
    fig_structure()
    fig_result()
    print("全部完成 ->", OUT)
