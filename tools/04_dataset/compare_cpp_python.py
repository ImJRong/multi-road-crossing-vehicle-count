# 对拍：把 C++ 和 Python 两边的过线记录逐条对比。
#
# 用法（在项目根目录）：
#   python tools/compare_cpp_python.py outputs/records_test4.csv output/records_test4.csv
#   python tools/compare_cpp_python.py A.csv B.csv 8     第三个参数是帧号容差，默认 5
#
# 为什么要容差：两边跟踪器不同（Python 是 ByteTrack，C++ 是 IoU 贪心），
# 同一辆车过线的帧号可能差几帧。所以不要求帧号完全相等，落在容差内就算同一次过车。
#
# 输出三样：
#   1. 每条线的总数对比（这是最硬的结果）
#   2. 逐事件匹配：匹配上多少、A 独有多少、B 独有多少
#   3. 没匹配上的明细，拿去回看视频定位原因

import csv
import sys
from collections import defaultdict


def load(path):
    rows = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[r["line"]].append(int(r["frame"]))
    for k in rows:
        rows[k].sort()
    return rows


def match(a, b, tol):
    """贪心匹配两个升序帧号序列，返回 (配对列表, a 独有, b 独有)。"""
    pairs, only_a, only_b = [], [], []
    i = j = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tol:
            pairs.append((a[i], b[j]))
            i += 1
            j += 1
        elif a[i] < b[j]:
            only_a.append(a[i])
            i += 1
        else:
            only_b.append(b[j])
            j += 1
    only_a.extend(a[i:])
    only_b.extend(b[j:])
    return pairs, only_a, only_b


def main():
    if len(sys.argv) < 3:
        print(__doc__ or "用法: python tools/compare_cpp_python.py <A.csv> <B.csv> [帧号容差]")
        return 1
    path_a, path_b = sys.argv[1], sys.argv[2]
    tol = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    A, B = load(path_a), load(path_b)

    print("A = %s" % path_a)
    print("B = %s" % path_b)
    print("帧号容差: ±%d" % tol)
    print()
    print("%-8s %8s %8s %8s %8s %8s" % ("线", "A总数", "B总数", "匹配上", "A独有", "B独有"))
    print("-" * 54)

    total_a = total_b = total_m = 0
    details = []
    for line in sorted(set(A) | set(B)):
        a, b = A.get(line, []), B.get(line, [])
        pairs, only_a, only_b = match(a, b, tol)
        total_a += len(a)
        total_b += len(b)
        total_m += len(pairs)
        print("%-8s %8d %8d %8d %8d %8d" % (line, len(a), len(b), len(pairs), len(only_a), len(only_b)))
        if only_a or only_b:
            details.append((line, only_a, only_b))

    print("-" * 54)
    print("%-8s %8d %8d %8d" % ("合计", total_a, total_b, total_m))

    if total_a == 0 and total_b == 0:
        print("\n两边都是 0，检查一下 csv 路径和 columns 对不对。")
        return 0

    diff = abs(total_a - total_b)
    base = max(total_a, total_b)
    print("\n总数偏差: %d 次 (%.1f%%)" % (diff, diff * 100.0 / base if base else 0))

    if details:
        print("\n没匹配上的（前 20 条，拿帧号去回看视频）:")
        n = 0
        for line, only_a, only_b in details:
            for fr in only_a:
                print("  线 %-4s 第 %-6d 帧  只有 A 有" % (line, fr))
                n += 1
                if n >= 20:
                    break
            if n >= 20:
                break
            for fr in only_b:
                print("  线 %-4s 第 %-6d 帧  只有 B 有" % (line, fr))
                n += 1
                if n >= 20:
                    break
            if n >= 20:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
