#!/usr/bin/env python3
"""画三张图，把「为什么改输出就能修好」讲透。

  fig9_causal_chain.png   因果链：旧路径每一步都在 int8 里重新量化
  fig10_outputs.png       三种模型的输出分别长什么样（一个输出 vs 两个输出）
  fig11_freedom.png       为什么可以随便指定输出（ONNX 的 graph.output 是什么）

数据全部来自实测：
  - 各层 scale / min / max：tools/02_diagnose/show_cfg_scale.py 读 step1 的 quantization.cfg
  - 误差 6.75px / 0.03px：tools/02_diagnose/batch_box_eval.py 跑 10 帧和 FP32 真值对比

用法：
  D:\\Python3.12.4\\python.exe tools/05_figures/draw_causal_figs.py
"""
import os

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

RED = "#d93a3a"
GREEN = "#1f9d4d"
BLUE = "#2f6db3"
ORANGE = "#e08a1e"
GREY = "#8a8a8a"
PURPLE = "#7a4fa3"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, facecolor="white")
    plt.close(fig)
    print("  生成:", p)


# ============================================================ 图 9 因果链
def fig_causal_chain():
    fig = plt.figure(figsize=(15.5, 10.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    # 实测数据：(名字, 值域下限, 上限, 刻度)
    OLD = [
        ("DFL 输出的\n距离", 0.000, 11.970, 0.046943),
        ("anchor\n− 左距离", -3.470, 79.461, 0.325220),
        ("anchor\n+ 右距离", 0.000, 83.276, 0.326573),
        ("两者相加\n÷2", 0.000, 159.480, 0.625413),
        ("两者相减\n= 宽高", 0.000, 21.434, 0.084053),
        ("拼成 4 通道\n× stride", 0.000, 637.921, 2.501652),
    ]

    def card(x, y, w, h, name, lo, hi, sc, color, fs_name=10.5, dim=False):
        alpha = 0.35 if dim else 1.0
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.10",
                                    fc=color, ec=color, lw=1.8,
                                    alpha=0.16 * alpha + 0.05, zorder=3))
        # 名称
        ax.text(x + w / 2, y + h - 0.30, name, ha="center", va="center",
                fontsize=fs_name, fontweight="bold", color="#1a1a1a", zorder=5,
                linespacing=1.5)
        # 值域
        ax.text(x + w / 2, y + 0.44, "值域 %.0f ~ %.0f" % (lo, hi),
                ha="center", va="center", fontsize=9.2, color="#333333", zorder=5)
        # 刻度
        ax.text(x + w / 2, y + 0.16, "刻度 %.3f" % sc,
                ha="center", va="center", fontsize=9.2, fontweight="bold",
                color=color, zorder=5)
        return (x, y, w, h)

    # ---------------- 上排：旧 ----------------
    ax.text(0.35, 9.55, "① 以前的路径 —— 整条链都在 int8 里算，每一步都重新量化一次",
            fontsize=15, fontweight="bold", color=RED)

    W, H, GAP = 2.30, 1.72, 0.18
    x0, y_top = 0.35, 7.20
    scales = []
    for i, (nm, lo, hi, sc) in enumerate(OLD):
        x = x0 + i * (W + GAP)
        card(x, y_top, W, H, nm, lo, hi, sc, RED)
        scales.append(sc)
        if i < len(OLD) - 1:
            ax.add_patch(FancyArrowPatch((x + W, y_top + H / 2),
                                         (x + W + GAP, y_top + H / 2),
                                         arrowstyle="-|>", mutation_scale=15,
                                         lw=1.8, color="#444444", zorder=2))

    # 刻度增长说明（放在卡片下方，留够间距）
    ax.text(0.35, y_top - 0.62,
            "刻度从 0.047 一路被撑到 2.50（放大约 53 倍）——范围越大，一格代表越多，精度越糊",
            fontsize=11.5, color=RED, fontweight="bold")

    # 中间说明（再往下挪，别和上一行挤）
    ax.text(0.35, y_top - 1.55,
            "每一格运算的结果，都要先压回 int8 才能传给下一步。压的时候用的是「这一步算出来的值域」，\n"
            "所以范围一涨，刻度跟着变粗，前面积累的精度就在这一步被抹掉一部分。",
            fontsize=11, color="#444444", linespacing=1.95, va="top")

    # ---------------- 下排：新 ----------------
    ax.text(0.35, 4.42, "② 现在的路径 —— 在 DFL 就算完，直接出模型，剩下的用 float 算",
            fontsize=15, fontweight="bold", color=GREEN)

    y_bot = 2.30
    # 前两格：模型内（int8）
    card(x0, y_bot, W, H, OLD[0][0], OLD[0][1], OLD[0][2], OLD[0][3], GREEN)
    x1 = x0 + W + GAP
    card(x1, y_bot, W, H, "直接当输出", 0.000, 11.970, 0.046943, GREEN)
    ax.add_patch(FancyArrowPatch((x0 + W, y_bot + H / 2), (x1, y_bot + H / 2),
                                 arrowstyle="-|>", mutation_scale=15, lw=1.8,
                                 color="#444444", zorder=2))

    # 标注：模型到此为止（放到虚线框内的左下角，避开标题）
    ax.add_patch(Rectangle((x0 - 0.10, y_bot - 0.22), 2 * W + GAP + 0.20, H + 0.44,
                           fill=False, ec=BLUE, lw=2.0, ls="--", zorder=6))
    ax.text(x0 + (2 * W + GAP) / 2, y_bot + H + 0.36, "↑ 模型里（只量化这一次）",
            ha="center", va="bottom", fontsize=10.5, color=BLUE, fontweight="bold")

    # 后面：C++ float
    x2 = x1 + W + GAP
    ax.add_patch(FancyArrowPatch((x1 + W, y_bot + H / 2), (x2, y_bot + H / 2),
                                 arrowstyle="-|>", mutation_scale=15, lw=2.2,
                                 color=GREEN, zorder=2))
    ax.add_patch(FancyBboxPatch((x2, y_bot), 5.45, H,
                                boxstyle="round,pad=0.04,rounding_size=0.10",
                                fc="#e9f7ee", ec=GREEN, lw=2.0, zorder=3))
    ax.text(x2 + 5.45 / 2, y_bot + H - 0.30,
            "C++ 里用 float32 算（全都在模型外面）",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=GREEN, zorder=5)
    ax.text(x2 + 5.45 / 2, y_bot + 0.68,
            "x1 = anchor − 左距    x2 = anchor + 右距\n"
            "宽 = x2 − x1          然后统一 × stride",
            ha="center", va="center", fontsize=10.5, color="#1a1a1a",
            zorder=5, linespacing=1.8)
    ax.text(x2 + 5.45 / 2, y_bot + 0.16,
            "float32 有 24 位有效数字，这里根本没有「刻度」这回事",
            ha="center", va="center", fontsize=9.5, color=GREEN, zorder=5)

    # ---------------- 结论 ----------------
    ax.add_patch(FancyBboxPatch((0.35, 0.30), 14.8, 1.42,
                                boxstyle="round,pad=0.06,rounding_size=0.14",
                                fc="#fff8e6", ec="#c9a227", lw=2.0, zorder=3))
    ax.text(0.75, 1.46, "结论（这三条都是实测的）：", fontsize=12,
            fontweight="bold", color="#8a6d1a", zorder=5)
    ax.text(0.75, 0.92,
            "① 旧路径里，从「距离」到「坐标」一共 6 步，每步都在 int8 里重新量化，刻度被撑大 53 倍；"
            "新路径只量化 1 次，剩下全 float。\n"
            "② 实测中心偏移：旧 6.75 像素 → 新 0.03 像素（10 帧、每帧和 FP32 真值逐框对比）。\n"
            "③ 模型权重一个字节没改，量化方式（INT8、300 张校准图、1.6.0）也完全没改，速度不变（18.3 fps）。",
            fontsize=10.5, color="#333333", linespacing=2.05, va="center")

    save(fig, "fig9_causal_chain.png")


# ============================================================= 图 10 输出形态
def fig_outputs():
    fig = plt.figure(figsize=(15, 11.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 11.2)
    ax.axis("off")

    def tensor_block(x, y, rows, labels, colors, title, sub, row_h=0.40,
                     col_w=3.30, fs=10):
        """画一个张量：rows 行 x 1 列。标题画在块上方。"""
        top = y + rows * row_h
        ax.text(x, top + 0.62, title, fontsize=12.5, fontweight="bold",
                color="#1a1a1a", va="center")
        ax.text(x, top + 0.30, sub, fontsize=9.5, color=GREY, va="center")
        for i in range(rows):
            yy = y + (rows - 1 - i) * row_h
            ax.add_patch(Rectangle((x, yy), col_w, row_h - 0.04,
                                   fc=colors[i], ec="white", lw=1.6, alpha=0.88,
                                   zorder=3))
            ax.text(x + col_w / 2, yy + (row_h - 0.04) / 2, labels[i],
                    ha="center", va="center", fontsize=fs, zorder=4,
                    fontweight="bold")

    # ================= ① 原始 ONNX =================
    ax.text(0.35, 10.85, "① 原始模型  best_detrac.onnx —— 单输出，坐标和分数拼在一起",
            fontsize=14.5, fontweight="bold", color=BLUE)
    tensor_block(0.70, 7.90, 5,
                 ["第 1 行：cx（中心 x）", "第 2 行：cy（中心 y）",
                  "第 3 行：w（框宽）", "第 4 行：h（框高）",
                  "第 5 行：分数（这里有没有车）"],
                 ["#dbeafe"] * 4 + ["#fee2e2"],
                 "[1, 5, 8400]", "5 行 × 8400 列（8400 个锚点）")
    ax.text(4.60, 9.15,
            "YOLOv8 官方导出就是这一个输出。\n\n"
            "坐标是「已经乘过 stride 的像素坐标」\n"
            "→ 值域 0 ~ 640\n"
            "→ 量化只能配 2.5 的粗刻度",
            fontsize=10.8, color="#333333", linespacing=2.0, va="center")

    # ================= ② head3 =================
    ax.text(0.35, 7.02, "② head3（昨天）—— 拆成两个输出，内容没变",
            fontsize=14.5, fontweight="bold", color=ORANGE)
    tensor_block(0.70, 4.35, 4, ["cx", "cy", "w", "h"], ["#dbeafe"] * 4,
                 "[1, 4, 8400]", "坐标那 4 行单独拎出来")
    tensor_block(4.60, 5.55, 1, ["分数"], ["#fee2e2"],
                 "[1, 1, 8400]", "分数单独一个")
    ax.text(8.35, 5.65,
            "为什么要拆：拼在一起时，RKNN 会在末端\n"
            "自动插一个 Split 做拆分，那个 Split 触发过\n"
            "工具 bug（详见 MEMORY 5.3）。拆开后就没有了。\n\n"
            "注意：内容还是坐标，值域还是 0~640，\n"
            "刻度还是 2.5 —— 所以框还是偏 6.75 像素。",
            fontsize=10.5, color="#333333", linespacing=2.0, va="center")

    # ================= ③ head4 =================
    ax.text(0.35, 3.68, "③ head4（今天）—— 还是两个输出，但第一个换了含义",
            fontsize=14.5, fontweight="bold", color=GREEN)
    tensor_block(0.70, 0.68, 4,
                 ["左距离", "上距离", "右距离", "下距离"], ["#d1f2dc"] * 4,
                 "[1, 4, 8400]", "DFL 算出的距离，单位是「格」")
    tensor_block(4.60, 1.70, 1, ["分数"], ["#fee2e2"],
                 "[1, 1, 8400]", "分数没动")
    ax.text(8.35, 2.16,
            "形状和 head3 一模一样（[1,4,8400] + [1,1,8400]），\n"
            "变的只有「那 4 行数字代表什么」：坐标 → 距离。\n\n"
            "值域 0 ~ 12  →  刻度 0.047（细 53 倍）\n\n"
            "这是整件事的关键，就改了这一个字。",
            fontsize=10.5, color="#333333", linespacing=2.0, va="center")

    # 底部提示
    ax.add_patch(FancyBboxPatch((0.35, 0.06), 14.30, 0.30,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc="#f0f5ff", ec=BLUE, lw=1.2, zorder=2))
    ax.text(7.5, 0.21,
            "注意：head3 和 head4 的输出「形状完全相同」，只有那 4 行数字的物理含义不同。"
            "所以后处理代码必须能分辨是哪一个（代码里靠张量名区分）。",
            ha="center", va="center", fontsize=10, color="#1a3a66", zorder=4)

    save(fig, "fig10_outputs.png")


# ============================================================= 图 11 自由度
def fig_freedom():
    fig = plt.figure(figsize=(15, 9.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    ax.text(7.5, 9.22, "为什么可以随便指定「哪个张量当输出」",
            ha="center", fontsize=17, fontweight="bold", color="#1a1a1a")

    # ---------- 左：ONNX 文件里就一行 ----------
    ax.add_patch(FancyBboxPatch((0.35, 5.10), 6.55, 3.55,
                                boxstyle="round,pad=0.06,rounding_size=0.14",
                                fc="#f7f9fc", ec=BLUE, lw=2.0, zorder=2))
    ax.text(0.65, 8.34, "ONNX 文件里，输出就是一行声明",
            fontsize=13, fontweight="bold", color=BLUE, zorder=4)

    # 代码块：纯 ASCII，避免等宽字体缺汉字
    code = ('graph {\n'
            '  node { ... 200+ ops ... }\n'
            '  output: "/model.22/Mul_2_output_0"      <-- head3\n'
            '  output: "/model.22/Sigmoid_output_0"\n'
            '}')
    ax.text(0.62, 7.15, code, fontsize=9.0, family="monospace",
            color="#1a1a1a", va="top", linespacing=1.9, zorder=4,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc"))

    ax.text(0.65, 6.05,
            "把它改成：\n"
            '  output: "/model.22/dfl/Reshape_1_output_0"\n'
            "就这一处改动，模型输出从「坐标」变成「距离」。",
            fontsize=10.2, color=GREEN, fontweight="bold", va="top",
            linespacing=1.95, zorder=4)

    # ---------- 右：为什么合法 ----------
    ax.add_patch(FancyBboxPatch((7.35, 5.10), 7.30, 3.55,
                                boxstyle="round,pad=0.06,rounding_size=0.14",
                                fc="#f4fbf6", ec=GREEN, lw=2.0, zorder=2))
    ax.text(7.65, 8.34, "为什么这不是「作弊」", fontsize=13,
            fontweight="bold", color=GREEN, zorder=4)
    ax.text(7.65, 7.80,
            "① 这是 ONNX 规范的正常用法\n"
            "   图里任何一个张量都可以声明成输出，\n"
            "   官方 checker 校验通过、onnxruntime 正常跑。\n\n"
            "② 模型权重一个字节没改\n"
            "   我没动任何参数，只是把「结果从哪一站交出来」提前了。\n\n"
            "③ 数学上完全等价\n"
            "   剩下那几步我用 numpy 重写，和原模型输出逐元素对比，\n"
            "   最大差值 = 0（不是很小，是精确的 0）。",
            fontsize=10.0, color="#1a1a1a", va="top", linespacing=1.9, zorder=4)

    # ---------- 下：代价 ----------
    ax.add_patch(FancyBboxPatch((0.35, 1.35), 14.30, 3.42,
                                boxstyle="round,pad=0.06,rounding_size=0.14",
                                fc="#fff8e6", ec="#c9a227", lw=2.0, zorder=2))
    ax.text(0.65, 4.42, "但是——不是白拿的，是有代价的", fontsize=14,
            fontweight="bold", color="#8a6d1a", zorder=4)

    ax.text(0.95, 3.85, "你交出去的东西", fontsize=11.5, fontweight="bold",
            color=RED, zorder=4)
    ax.text(0.95, 3.35,
            "· 后面那几步的代码要自己写\n"
            "· 那 9 个算子的算法要自己复刻对\n"
            "· anchor 网格、stride 分段、距离顺序\n"
            "  这些细节错了，框就全错\n"
            "· 换个检测模型（比如别的 YOLO 版本）\n"
            "  这套换算要重写",
            fontsize=10.2, color="#333333", va="top", linespacing=1.95, zorder=4)

    ax.text(7.90, 3.85, "你换回来的东西", fontsize=11.5, fontweight="bold",
            color=GREEN, zorder=4)
    ax.text(7.90, 3.35,
            "· 那些计算从 int8 变成 float32，\n"
            "  精度不受量化影响\n"
            "· 量化刻度从 2.5 变 0.047，\n"
            "  DFL 那 16 个 bin 的精度保住\n"
            "· 速度完全不变（还是纯 int8，没有 fp16 层）\n"
            "· 实测中心偏移 6.75 → 0.03 像素",
            fontsize=10.2, color="#333333", va="top", linespacing=1.95, zorder=4)

    # ---------- 底部一句话 ----------
    ax.add_patch(FancyBboxPatch((0.35, 0.35), 14.30, 0.82,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="#eef3ff", ec=BLUE, lw=1.8, zorder=2))
    ax.text(7.5, 0.76,
            "一句话：ONNX 的输出就是一个「在哪一站下车」的选择，改它不违反任何规则；\n"
            "代价是下车之后的票得你自己买——那些计算要你自己写对。",
            ha="center", fontsize=11.5, color="#1a3a66", va="center",
            linespacing=1.9, zorder=4)

    save(fig, "fig11_freedom.png")


if __name__ == "__main__":
    print("生成因果链讲解图：")
    fig_causal_chain()
    fig_outputs()
    fig_freedom()
    print("完成 ->", OUT)
