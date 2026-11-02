#!/usr/bin/env python3
"""对比 v7 与 head4 两个模型在四个视频上的评测结果。

数据来源：板子上 multi_eval 生成的
  output/full_v7_summary.csv
  output/full_head4_summary.csv

注意：四个视频的计数线条数不同（test4 有 A/B/C 三条，其余两条），
      所以 CSV 列数不齐，不能用 DictReader 按固定字段读，这里按行切分。

判读：
  total_ids  累计分配的轨迹编号数。同一视频、同样帧数下，编号越少说明断号越少
             （一辆车被反复换号）。
  counts     各计数线的过车数。断号会导致漏计，这个数偏小就是漏。

用法：
  python tools/02_diagnose/compare_models.py <v7.csv> <head4.csv>
"""
import csv
import sys

FIXED = ["route", "frames", "total_ids", "seconds", "fps", "total_boxes"]


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        line_names = header[len(FIXED):]          # 固定字段之后的都是线名
        for r in reader:
            if len(r) < len(FIXED):
                continue
            name = r[0]
            counts = {}
            for i, ln in enumerate(line_names):
                idx = len(FIXED) + i
                if idx < len(r) and r[idx].strip():
                    counts[ln] = int(r[idx])
            rows[name] = {
                "frames": int(r[1]),
                "ids": int(r[2]),
                "fps": float(r[4]),
                "counts": counts,
            }
    return rows


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    a, b = load(sys.argv[1]), load(sys.argv[2])

    routes = [r for r in a if r in b]
    if not routes:
        print("两份 CSV 没有共同的路线名，检查文件")
        return 1

    print("=" * 76)
    print("轨迹编号数（断号指标，越少越好）")
    print("=" * 76)
    print("%-10s %10s %10s %10s %10s" % ("路线", "v7", "head4", "减少", "降幅"))
    print("-" * 76)
    ta = tb = 0
    for r in routes:
        va, vb = a[r]["ids"], b[r]["ids"]
        ta += va
        tb += vb
        cut = va - vb
        pct = 100.0 * cut / va if va else 0
        print("%-10s %10d %10d %10d %9.1f%%" % (r, va, vb, cut, pct))
    print("-" * 76)
    cut = ta - tb
    pct = 100.0 * cut / ta if ta else 0
    print("%-10s %10d %10d %10d %9.1f%%" % ("合计", ta, tb, cut, pct))

    print()
    print("=" * 76)
    print("各线计数（越大说明漏得越少）")
    print("=" * 76)

    lines = []
    for r in routes:
        for k in a[r]["counts"]:
            if k not in lines:
                lines.append(k)
        for k in b[r]["counts"]:
            if k not in lines:
                lines.append(k)

    print("%-10s" % "路线" + "".join(" %16s" % ln for ln in lines))
    total_delta = 0
    for r in routes:
        row = "%-10s" % r
        for ln in lines:
            va = a[r]["counts"].get(ln)
            vb = b[r]["counts"].get(ln)
            if va is None and vb is None:
                row += " %16s" % "-"
            elif va is None:
                row += " %16s" % ("+" + str(vb))
            elif vb is None:
                row += " %16s" % ("-" + str(va))
            else:
                d = vb - va
                total_delta += d
                row += " %16s" % ("%d->%d(%+d)" % (va, vb, d))
        print(row)

    print("-" * 76)
    print("计数净变化：%+d" % total_delta)

    print()
    print("=" * 76)
    print("结论")
    print("=" * 76)
    if tb < ta:
        print("断号减少 %.1f%%（%d -> %d），这支持" % (pct, ta, tb))
        print("「框偏移 -> 框抖 -> 跟踪断号 -> 漏计」这条因果链。")
    else:
        print("断号没有减少（%d -> %d），需要重新分析。" % (ta, tb))

    if total_delta > 0:
        print("计数净增 %d，与「框偏移修复后漏计减少」一致。" % total_delta)
    elif total_delta == 0:
        print("计数无变化。")
    else:
        print("计数净减 %d，与预期相反，需查原因。" % abs(total_delta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
