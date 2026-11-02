#!/usr/bin/env python3
"""对比两个 toolkit 版本 dump 出来的逐层误差，只看尾部（DFL 之后）。"""
import os
import re

PROJ = os.path.expanduser("~/traffic_count")
A = os.path.join(PROJ, "trace_1_6_0", "error_analysis.txt")
B = os.path.join(PROJ, "trace_2_5_0", "error_analysis.txt")

ROW = re.compile(
    r"^\s*\[(?P<op>[^\]]+)\]\s+(?P<name>\S+)\s+"
    r"(?P<ec>[\d.]+)\s*\|\s*(?P<ee>[\d.]+)\s+"
    r"(?P<sc>[\d.]+)\s*\|\s*(?P<se>[\d.]+)")


def load(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ROW.match(line)
            if m:
                rows.append(dict(op=m.group("op"), name=m.group("name"),
                                 ec=float(m.group("ec")), ee=float(m.group("ee")),
                                 sc=float(m.group("sc")), se=float(m.group("se"))))
    return rows


def main():
    a, b = load(A), load(B)
    print("1.6.0 解析到 %d 层，2.5.0 解析到 %d 层" % (len(a), len(b)))
    if not a or not b:
        print("解析失败，检查文件：")
        print("  ", A)
        print("  ", B)
        return 1

    # 只保留 DFL 及之后的层（名字里有 dfl / Mul_2 / Concat_2 / Slice / Sigmoid）
    def tail(rows):
        out = []
        for r in rows:
            n = r["name"]
            if any(k in n for k in ("dfl", "Mul_2", "Concat_2", "Slice",
                                    "Sub_1", "Div_1", "Add_1", "Add_2", "Sigmoid")):
                out.append(r)
        return out

    ta, tb = tail(a), tail(b)
    print("\n" + "=" * 96)
    print("DFL 及其之后的层 —— 量化误差（entire cosine / entire euc，越小越准）")
    print("=" * 96)
    print("%-52s %-18s %-18s" % ("层名", "1.6.0 (cos|euc)", "2.5.0 (cos|euc)"))
    print("-" * 96)

    bmap = {r["name"]: r for r in tb}
    for r in ta:
        n = r["name"]
        m = bmap.get(n)
        if m:
            print("%-52s %-18s %-18s" % (n[-52:],
                                         "%.5f|%9.2f" % (r["ec"], r["ee"]),
                                         "%.5f|%9.2f" % (m["ec"], m["ee"])))

    print("\n" + "=" * 96)
    print("只看最终输出")
    print("=" * 96)
    for tag, rows in (("1.6.0", a), ("2.5.0", b)):
        for r in rows:
            if "Mul_2_output_0" in r["name"] and "rs" not in r["name"] and "int8" not in r["name"]:
                print("%s  Mul_2(最终坐标输出)  cos=%.5f  euc=%.2f" % (tag, r["ec"], r["ee"]))

    # 找两边差异最大的层
    print("\n" + "=" * 96)
    print("两版差异最大的层（|cos 差| 排序）")
    print("=" * 96)
    amap = {r["name"]: r for r in a}
    diffs = []
    for r in b:
        if r["name"] in amap:
            d = abs(r["ec"] - amap[r["name"]]["ec"])
            diffs.append((d, r["name"], amap[r["name"]]["ec"], r["ec"]))
    diffs.sort(reverse=True)
    for d, n, ca, cb in diffs[:15]:
        print("  差 %.6f  %-60s 1.6.0=%.5f  2.5.0=%.5f" % (d, n[-60:], ca, cb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
