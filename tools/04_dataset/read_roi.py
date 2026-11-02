# 读检测区域（ROI）：你在截图上画一个区域，这个工具把它的多边形坐标算出来。
#
# 思路和 read_line.py 一样：比对"原图"和"你画过的图"，找出你画上去的笔画。
# 好处是你画什么颜色都行——视频里本来就有红的黄的，认颜色会认错。
#
# 画法建议：用画图工具的「矩形」或者「多边形」工具，把要检测的区域圈起来，
#   画粗一点（3 像素以上）更好认。不用画得很精确，大概就行。
#
# ⚠️ 有个坑要注意：区域不能只圈住计数线，要往【车来的方向】多留一段。
#    跟踪器要看到一辆车连续好几帧，才知道它"上一帧在线哪一侧"。
#    区域框太紧，车刚进画面就到线上了，这一辆就漏了。
#    经验值：留够车跑 1~2 秒的距离。
#
# 例：
#   python tools/grab_frame.py data/videos/test1.mp4 1
#   （用画图工具在导出的图上圈一个区域，另存为 test1_roi.jpg）
#   python tools/read_roi.py data/samples/test1_frame1.jpg data/samples/test1_roi.jpg
#   python tools/read_roi.py data/samples/test1_frame1.jpg data/samples/test1_roi.jpg --write configs/test1.json

import json
import os
import re
import sys

import cv2
import numpy as np

THRESHOLD = 40          # 两张图差多少算"你画上去的"
APPROX = 0.01           # 多边形简化程度，越大点越少（取周长的百分比）


def _compact(text):
    """把 json.dumps 展开成多行的坐标/颜色数组压回一行 [a, b] / [a, b, c]。"""
    def one(m):
        nums = [g for g in m.groups() if g is not None]
        return "[%s]" % ", ".join(nums)

    return re.sub(r"\[\s+(-?\d+),\s+(-?\d+)(?:,\s+(-?\d+))?\s+\]", one, text)


def find_roi(orig_path, drawn_path):
    orig = cv2.imread(orig_path)
    drawn = cv2.imread(drawn_path)
    if orig is None:
        raise SystemExit("读不到原图：%s" % orig_path)
    if drawn is None:
        raise SystemExit("读不到画过的图：%s" % drawn_path)
    if orig.shape != drawn.shape:
        raise SystemExit(
            "两张图尺寸不一样：原图 %s，你那张 %s。\n"
            "请用 tools/grab_frame.py 导出的那张图来画，另存时不要改尺寸。"
            % (str(orig.shape), str(drawn.shape)))

    diff = cv2.absdiff(orig, drawn).max(axis=2)
    ys, xs = np.where(diff > THRESHOLD)
    if len(xs) < 50:
        raise SystemExit("几乎没找到差异，你是不是没画上？或者另存时覆盖在原图上了。")

    pts = np.stack([xs, ys], axis=1).astype(np.int32)

    # 取凸包：你画的是个框/多边形的"边"，凸包能把这些边还原成一个实心形状。
    # 代价是凹多边形的凹口会被填平——对圈检测区域来说够用了。
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, APPROX * peri, True).reshape(-1, 2)

    return poly


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv

    if len(args) < 2:
        print("用法：python tools/read_roi.py <原图> <画了区域的图> [--write <配置.json>]")
        print("例：  python tools/read_roi.py data/samples/f.jpg data/samples/f_roi.jpg --write configs/test1.json")
        return

    poly = find_roi(args[0], args[1])

    xs, ys = poly[:, 0], poly[:, 1]
    print("找到 %d 个顶点：" % len(poly))
    print("  " + "  ".join("[%d,%d]" % (p[0], p[1]) for p in poly))
    print()
    print("外接矩形：x %d~%d, y %d~%d   （裁图的时候按这个裁，尺寸 %dx%d）"
          % (xs.min(), xs.max(), ys.min(), ys.max(),
             xs.max() - xs.min(), ys.max() - ys.min()))

    orig = cv2.imread(args[0])
    cv2.polylines(orig, [poly.reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    out = os.path.splitext(args[1])[0] + "_check.jpg"
    cv2.imwrite(out, orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print()
    print("核对图：%s（打开看一眼，红线是不是把你想要的区域圈住了）" % out)

    if not write:
        print()
        print("确认没问题后，加 --write <配置.json> 写进配置。")
        return

    if len(args) < 3:
        raise SystemExit("--write 后面要跟配置文件路径")

    cfg_path = args[2]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["roi"] = [[int(p[0]), int(p[1])] for p in poly]
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(_compact(json.dumps(cfg, ensure_ascii=False, indent=2)) + "\n")
    print("已写入 %s" % cfg_path)
    print()
    print("⚠️ 计数线的坐标还是原来的（在整张图上的坐标），不用改。")
    print("   但如果线跑到区域外面了，那辆车就永远不会被检测到，计数会是 0。核对一下。")


if __name__ == "__main__":
    main()
