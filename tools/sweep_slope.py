# 扫计数线的斜率和位置：拿一段视频的轨迹，试很多条线，看哪条最合适。
#
# 用法（在项目根目录）：python tools/sweep_slope.py <轨迹缓存.json> <左右幅> <y中心> <x范围>
#   例：python tools/sweep_slope.py data/samples/_test2_traj.json L 560 150 700
#
# 判断"哪条线好"的两个指标：
#   1. 覆盖：有多少辆车会穿过它（越多越好，说明没漏）
#   2. 穿透速度：车每次前进跨过这条线的距离（越大越好，太小容易数重或数漏）

import json
import sys

import numpy as np


def vel(pts):
    n = len(pts) - 1
    return (pts[-1][0] - pts[0][0]) / n, (pts[-1][1] - pts[0][1]) / n


def cross(pts, y0, slope, xc_ref):
    """轨迹穿过直线 y = y0 + slope*(x - xc_ref) 的位置和穿透速度。"""
    # 每条线段和直线求交点
    for (x0, y0p), (x1, y1p) in zip(pts[:-1], pts[1:]):
        f0 = y0p - (y0 + slope * (x0 - xc_ref))
        f1 = y1p - (y0 + slope * (x1 - xc_ref))
        if f0 == 0 or f0 * f1 < 0:
            t = f0 / (f0 - f1) if f0 != f1 else 0
            xc = x0 + t * (x1 - x0)
            yc = y0p + t * (y1p - y0p)
            vx, vy = vel(pts)
            n = np.array([-slope, 1.0])
            n /= np.linalg.norm(n)
            return xc, yc, abs(vx * n[0] + vy * n[1]), vx, vy
    return None


def main():
    if len(sys.argv) < 5:
        print("用法：python tools/sweep_slope.py <轨迹.json> <L|R> <y中心> <x范围起> <x范围止>")
        return
    path, side = sys.argv[1], sys.argv[2]
    y0 = float(sys.argv[3])
    x_lo, x_hi = float(sys.argv[4]), float(sys.argv[5])

    data = json.load(open(path))
    tracks = [v for v in data.values() if len(v) >= 20]

    print("轨迹 %d 条，y中心 %.0f，x范围 %.0f~%.0f" % (len(tracks), y0, x_lo, x_hi))
    print()
    print("%-8s %-8s %-10s %-10s %s" % ("斜率", "覆盖车数", "穿透中位", "穿透最慢", "覆盖的x范围"))

    best = None
    for slope in np.arange(-1.6, 1.65, 0.2):
        xc_ref = (x_lo + x_hi) / 2
        got = []
        for pts in tracks:
            r = cross(pts, y0, slope, xc_ref)
            if r is None:
                continue
            xc, yc, sp, vx, vy = r
            if not (x_lo <= xc <= x_hi):
                continue
            # 方向过滤：L 幅只收向下走的，R 幅只收向上走的
            if side == "L" and vy <= 0.5:
                continue
            if side == "R" and vy >= -0.5:
                continue
            got.append((xc, sp))
        if not got:
            print("%-8.1f %-8d %s" % (slope, 0, "-"))
            continue
        sps = [g[1] for g in got]
        xcs = [g[0] for g in got]
        print("%-8.1f %-8d %-10.2f %-10.2f %.0f~%.0f"
              % (slope, len(got), np.median(sps), min(sps), min(xcs), max(xcs)))
        # 打分：覆盖数优先，再要穿透速度大
        score = len(got) * 10 + np.median(sps)
        if best is None or score > best[0]:
            best = (score, slope, len(got), np.median(sps))

    if best:
        print()
        print("最合适：斜率 %.1f，覆盖 %d 辆车，穿透中位 %.2f px/帧" % (best[1], best[2], best[3]))


if __name__ == "__main__":
    main()
