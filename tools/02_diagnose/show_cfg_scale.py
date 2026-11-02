#!/usr/bin/env python3
"""从 step1 生成的 quantization.cfg 里，精确提取指定层的量化参数。

为什么要这么干：A 的结论是「w/h 放大 13 倍不是 DFL 算错，是某层的刻度配错」。
需要把 DFL 之后每一层的 min/max/scale 摊开看，找出哪一层的刻度和它的实际量纲对不上。
"""
import os
import re
import sys

CFG = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/traffic_count/models/step1_250/best_detrac_head3.quantization.cfg")

# 按 YAML 顶层 key 分段，只在 quantize_parameters 段里找
TARGETS = [
    "/model.22/dfl/Reshape_output_0",
    "/model.22/dfl/Softmax_output_0",
    "/model.22/dfl/conv/Conv_output_0",
    "/model.22/dfl/Reshape_1_output_0",
    "/model.22/Slice_output_0",
    "/model.22/Slice_1_output_0",
    "/model.22/Sub_output_0",
    "/model.22/Add_1_output_0",
    "/model.22/Add_2_output_0",
    "/model.22/Sub_1_output_0",
    "/model.22/Div_1_output_0",
    "/model.22/Concat_2_output_0",
    "/model.22/Mul_2_output_0",
    "/model.22/Sigmoid_output_0",
]


def main():
    if not os.path.isfile(CFG):
        print("找不到 cfg:", CFG)
        return 1

    lines = open(CFG, encoding="utf-8", errors="replace").read().splitlines()
    # 找到 quantize_parameters: 的行号
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("quantize_parameters:"):
            start = i
            break
    if start is None:
        print("cfg 里没有 quantize_parameters 段")
        return 1
    body = lines[start:]

    # 层名行形如 "    /model.xx/yyy:"（4 空格缩进，以冒号结尾）
    entries = {}
    cur = None
    buf = []
    for ln in body:
        m = re.match(r"^    (\S+):\s*$", ln)
        if m:
            if cur:
                entries[cur] = buf
            cur = m.group(1)
            buf = []
        elif cur is not None:
            buf.append(ln)
    if cur:
        entries[cur] = buf

    print("cfg:", CFG)
    print("quantize_parameters 里共 %d 个条目\n" % len(entries))

    def get(name):
        """名字可能带 -rs / _sw / _shape4 等后缀，模糊匹配。"""
        for k in entries:
            if k == name or k.startswith(name + "-") or k.startswith(name + "_"):
                return k
        for k in entries:
            if name in k:
                return k
        return None

    print("%-42s %-8s %-10s %-10s %-12s %s"
          % ("层", "dtype", "min", "max", "scale", "实际层名"))
    print("-" * 118)
    for t in TARGETS:
        k = get(t)
        if not k:
            print("%-42s %s" % (t, "(cfg 里没有)"))
            continue
        body_txt = "\n".join(entries[k])
        dtype = re.search(r"dtype:\s*(\S+)", body_txt)
        # min/max/scale 都是 "key:\n   - value" 的形式
        def num(key):
            m = re.search(r"%s:\s*\n\s*-\s*([-\d.eE+]+)" % key, body_txt)
            return float(m.group(1)) if m else None
        mn, mx, sc = num("min"), num("max"), num("scale")
        print("%-42s %-8s %-10s %-10s %-12s %s"
              % (t[-42:],
                 dtype.group(1) if dtype else "?",
                 "%.3f" % mn if mn is not None else "-",
                 "%.3f" % mx if mx is not None else "-",
                 "%.6f" % sc if sc is not None else "-",
                 k))

    print("\n判读方法：")
    print("  Slice_output_0 是 [锚点 − 左距离]，值域约 0~2.5（小数值）")
    print("  Mul_2_output_0 是最终坐标，值域 0~640")
    print("  如果 Slice 那层的 scale 也是 2.5 级别 → 小数值被粗刻度压扁 → 这就是 13 倍的来源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
