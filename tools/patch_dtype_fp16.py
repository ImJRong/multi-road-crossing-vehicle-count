"""手工把 DFL 那几层的 dtype 从 int8 改成 float16, 不碰 custom_quantize_layers.

为什么不用"追加到 custom_quantize_layers"的办法:
    实测 1.6.0 的 step2 在 quant_optimizer._p_adjust_tanh_sigmoid 里
    硬查 '/model.22/Concat_1_output_0', 而 step1 生成的 cfg 里这一项叫
    '/model.22/Concat_1_output_0_shape4_/model.22/Sigmoid' —— 名字对不上,
    必定 KeyError。只改 2 层一样崩, 所以不是"改多了"的问题, 是工具自身的问题。

这个办法绕开那条路径:
    不往 custom_quantize_layers 里加东西(保持 step1 原样),
    只在 quantize_parameters 段里把目标张量的 dtype 改成 float16。

用法(WSL, 必须在 ~/traffic_count 下):
    conda activate rknn16
    cd ~/traffic_count
    python /mnt/d/multi-road-crossing-vehicle-count/tools/patch_dtype_fp16.py
"""
import os
import re
import shutil
import sys

CANDIDATES = [
    "/mnt/c/Windows/System32/best_detrac_head3.quantization.cfg",
    os.path.expanduser("~/traffic_count/best_detrac_head3.quantization.cfg"),
]

# 只改这几个：DFL 的 Softmax 和它的转置，以及最终乘 stride 之后的结果
TARGETS = [
    "/model.22/dfl/Softmax_output_0",
    "/model.22/dfl/Transpose_1_output_0",
    "/model.22/Mul_2_output_0",
    "/model.22/Mul_2_output_0_shape4",
]


def find_src():
    for p in CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def main():
    src = find_src()
    if not src:
        print("找不到 quantization.cfg")
        for p in CANDIDATES:
            print("  ", p)
        return 1

    print("读:", src)
    bak = src + ".orig"
    if not os.path.isfile(bak):
        shutil.copy(src, bak)
        print("备份原始配置 ->", bak)

    with open(src, encoding="utf-8") as f:
        lines = f.readlines()

    # 状态机：定位 "    <张量名>:" 那一行，然后把它下面第一个 "dtype: int8" 改成 float16
    changed = []
    cur = None
    depth = 0
    for i, line in enumerate(lines):
        m = re.match(r"^    (/\S+?):\s*$", line.rstrip("\n"))
        if m:
            cur = m.group(1)
            continue
        if cur and cur in TARGETS and re.match(r"^\s+dtype:\s*int8\s*$", line):
            lines[i] = line.replace("int8", "float16")
            changed.append(cur)
            cur = None          # 一个张量只改第一处

    # 也要处理 custom_quantize_layers 段里已有的 float16 行（保持原样即可）

    if not changed:
        print("\n⚠️ 一个都没改到。可能:")
        print("   1. 这些张量名在当前 cfg 里不存在")
        print("   2. 它们已经是 float16 了")
        print("\ncfg 里实际有哪些 dfl 相关张量:")
        for line in lines:
            if "dfl" in line and re.match(r"^    /", line):
                print("  ", line.strip())
        return 1

    with open(src, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\n已改 %d 个张量为 float16:" % len(changed))
    for c in changed:
        print("   ", c)

    # 校验：打印改后的那几段
    print("\n改后确认:")
    with open(src, encoding="utf-8") as f:
        all_lines = f.readlines()
    for i, line in enumerate(all_lines):
        m = re.match(r"^    (/\S+?):\s*$", line.rstrip("\n"))
        if m and m.group(1) in TARGETS:
            for j in range(i + 1, min(i + 8, len(all_lines))):
                if "dtype" in all_lines[j]:
                    print("   %s -> %s" % (m.group(1), all_lines[j].strip()))
                    break

    print("\n下一步:")
    print("  python /mnt/d/multi-road-crossing-vehicle-count/tools/hybrid_step2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
