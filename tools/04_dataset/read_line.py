# 读计数线：你在截图上画一条线，这个工具把它的坐标算出来。
#
# 用法：
#   1. 先用 tools/grab_frame.py 导出一张原图（或者是任意一帧）
#   2. 用画图工具在那张图上画一条线（颜色随便，画粗一点更好认），另存为新文件
#   3. 跑这个工具，它会比对"原图"和"画了线的图"，找出你画的线在哪
#   4. 加 --write 就直接把坐标写进 configs 里的配置
#
# 为什么用"两张图比对"而不是认颜色：视频里本来就有红色的车、红色的广告牌，
# 认颜色会认错。比两张图的差异最稳，你画成什么颜色都行。
#
# 想连 ROI 框一起画、又怕画出框外的话，先用 tools/make_line_base.py 生成一张
# 带框的"底图"，在底图上画线，然后加 --base <底图> 来读。
# 这样框的边就不算在你画的线里（因为底图上本来就有框）。
#
# 例：
#   python tools/read_line.py data/samples/test1_frame0.jpg data/samples/test1_line.jpg
#   python tools/read_line.py data/samples/test1_frame0.jpg data/samples/test1_line.jpg --write configs/test1.json 0
#   python tools/read_line.py data/samples/test3_frame300.jpg data/samples/test3_line.jpg --base data/samples/test3_frame300_base.jpg

import json
import os
import sys

import cv2
import numpy as np

THRESHOLD = 40          # 两张图差多少算"你画上去的"，0~255


def find_line(base_path, drawn_path):
    """返回 (端点1, 端点2) 的列表。画了多条就返回多条。"""
    orig = cv2.imread(base_path)
    drawn = cv2.imread(drawn_path)
    if orig is None:
        raise SystemExit("读不到参照图：%s" % base_path)
    if drawn is None:
        raise SystemExit("读不到画了线的图：%s" % drawn_path)

    if orig.shape != drawn.shape:
        raise SystemExit(
            "两张图尺寸不一样：参照图 %s，你那张 %s。\n"
            "可能是画图工具改了尺寸，或者你不是在同一张图上画的。\n"
            "请用 tools/grab_frame.py 导出（或 make_line_base.py 生成）的图来画，另存时不要改尺寸。"
            % (str(orig.shape), str(drawn.shape)))

    # 逐像素比差异，差异大的就是画上去的笔画
    diff = cv2.absdiff(orig, drawn).max(axis=2)
    ys, xs = np.where(diff > THRESHOLD)

    if len(xs) < 20:
        raise SystemExit("几乎没找到差异，你是不是没画上？或者另存的时候覆盖在原图上了。")

    # 去掉离群的像素（比如画图工具顺手写了个字）
    med_x, med_y = np.median(xs), np.median(ys)
    keep = (np.abs(xs - med_x) < 0.6 * orig.shape[1]) & (np.abs(ys - med_y) < 0.6 * orig.shape[0])
    xs, ys = xs[keep], ys[keep]

    return fit_segments(xs, ys, orig.shape)


def fit_segments(xs, ys, shape, gap=25):
    """把笔画像素按 y 分团（一条线一团），每团拟合出一条直线。"""
    order = np.argsort(ys)
    xs, ys = xs[order], ys[order]

    groups = []
    start = 0
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] > gap:
            groups.append((xs[start:i], ys[start:i]))
            start = i
    groups.append((xs[start:], ys[start:]))

    out = []
    for gx, gy in groups:
        if len(gx) < 20:
            continue
        # 线主要是横的 -> 拟合 y=f(x)；主要是竖的 -> 拟合 x=f(y)
        if gx.max() - gx.min() >= gy.max() - gy.min():
            a, b = np.polyfit(gx, gy, 1)
            x1, x2 = int(gx.min()), int(gx.max())
            out.append(((x1, int(round(a * x1 + b))), (x2, int(round(a * x2 + b)))))
        else:
            a, b = np.polyfit(gy, gx, 1)
            y1, y2 = int(gy.min()), int(gy.max())
            out.append(((int(round(a * y1 + b)), y1), (int(round(a * y2 + b)), y2)))
    if not out:
        raise SystemExit("找到笔画了，但凑不成一条线。是不是画得太短或者太散？")
    return out


def main():
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    write = "--write" in argv
    base = None
    if "--base" in argv:
        i = argv.index("--base")
        if i + 1 >= len(argv):
            raise SystemExit("--base 后面要跟底图路径")
        base = argv[i + 1]
        args = [a for a in args if a != base]

    if len(args) < 2:
        print("用法：python tools/read_line.py <原图> <画了线的图> [--base 底图] [--write <配置.json> <第几条线>]")
        print("例：  python tools/read_line.py data/samples/f.jpg data/samples/f_line.jpg --write configs/test1.json 0")
        print("      python tools/read_line.py data/samples/f.jpg data/samples/f_line.jpg --base data/samples/f_base.jpg")
        return

    ref = base or args[0]
    if base:
        print("参照图（带 ROI 框的底图）：%s" % base)

    segs = find_line(ref, args[1])
    print("找到 %d 条线：" % len(segs))
    for k, (A, B) in enumerate(segs):
        if A[1] == B[1]:
            shape = "水平线 y=%d" % A[1]
        elif A[0] == B[0]:
            shape = "竖直线 x=%d" % A[0]
        else:
            shape = "斜率 %.3f" % ((B[1] - A[1]) / (B[0] - A[0]))
        print("  [%d] A=[%d, %d]  B=[%d, %d]   （%s）" % (k, A[0], A[1], B[0], B[1], shape))

    # 画回去让你核对
    orig = cv2.imread(ref)
    for A, B in segs:
        cv2.line(orig, A, B, (0, 0, 255), 3)
    out = os.path.splitext(args[1])[0] + "_check.jpg"
    cv2.imwrite(out, orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("核对图：%s（打开看一眼，红线是不是和你画的重合）" % out)

    if not write:
        print()
        print("确认没问题后，加 --write <配置.json> <第几条线> 写进配置。")
        return

    if len(args) < 4:
        raise SystemExit("--write 后面要跟配置文件和线序号，比如 --write configs/test1.json 0")

    cfg_path, idx = args[2], int(args[3])
    A, B = segs[idx]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["lines"][idx]["a"] = [A[0], A[1]]
    cfg["lines"][idx]["b"] = [B[0], B[1]]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("已写入 %s 的第 %d 条线" % (cfg_path, idx))
    print("注意：车道 x 区间（lanes）还没改，那是另一回事，要按这条线的位置重新划。")


if __name__ == "__main__":
    main()
