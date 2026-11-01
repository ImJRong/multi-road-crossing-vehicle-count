"""把混合量化配置里"坐标分支"的层改成 float16, 保持不变的部分照旧.

为什么要手工改:
    step1 的 proposal 自动分析选的是骨干最前面 7 层(model.1/model.2),
    对坐标精度没有任何帮助。它优化的是"整模型输出相似度",
    而我们要的是"框坐标准", 这两件事不是一回事。

为什么只改坐标分支:
    model.22 检测头里有两路:
      分类分支 cv3.x  -> Sigmoid -> scores
      回归分支 cv2.x  -> DFL(Softmax + conv 加权) -> boxes
    我们关心的是 boxes。全改 fp16 会拖慢速度, 只改回归那一路最划算。
    另外 DFL 的 Softmax 输出是概率分布(16 个 bin 和为 1),
    量化 scale 只有 1/255, 每个 bin 平均 0.0625 只剩 16 格,
    加权求坐标时误差累加 —— 这就是框偏移的数学根源。

用法(WSL 里, 必须在 ~/traffic_count 目录下跑):
    conda activate rknn16
    cd ~/traffic_count
    python /mnt/d/multi-road-crossing-vehicle-count/tools/fix_hybrid_cfg.py
"""
import os
import re
import shutil
import sys

# step1 生成的文件名固定是 <onnx名>.quantization.cfg / <onnx名>.data
# 但如果当前目录不对, 它们会落在别处, 所以这里支持从参数指定
SRC_CANDIDATES = [
    os.path.expanduser("~/traffic_count/best_detrac_head3.quantization.cfg"),
    "/mnt/c/Windows/System32/best_detrac_head3.quantization.cfg",
]

# ---- 要改成 float16 的张量：只碰 DFL 核心那几步 ----
#
# 2026-11-01 第一次尝试改多了（cv2.x 回归骨干 + Slice/Sub/Add/Concat 全改成
# float16, 35 个), step2 崩在:
#     KeyError: '/model.22/Concat_1_output_0'
# 原因: 工具在 quant_optimizer._p_adjust_tanh_sigmoid 那一步要按层名查表,
# 而某些被融合/拆分的层在 quantize_parameters 里没有条目。改的层越多,
# 撞上的概率越大。
#
# 所以收窄到最小集合: 只改 DFL 那一段(Softmax + 加权平均)。
# 坐标偏移的数学根源就在这里 —— Softmax 输出是 16 个 bin 的概率分布
# (和为 1, 每个平均 0.0625), scale 只有 1/255, 一个 bin 只剩 16 格,
# 加权求距离时误差直接累加。
FP16_PATTERNS = [
    r"^/model\.22/dfl/.*Output",
    r"^/model\.22/Mul_2_output_0$",
    r"^/model\.22/Mul_2_output_0_shape4$",
]

# 明确不动的
KEEP_INT8_PATTERNS = [
    r"/model\.22/cv3\.\d+/",
    r"/model\.22/Concat_1",
    r"/model\.22/Sigmoid",
    r"/model\.22/Mul_2_output_0_int8",     # 已经是拆好的 int8 版本
    r"/model\.22/Sigmoid_output_0_int8",
    r"^images$",
    r"/model\.0/",
    r"/model\.22/Slice",                   # 这几个改过就崩, 别碰
    r"/model\.22/Sub",
    r"/model\.22/Add",
    r"/model\.22/Concat_2",
    r"/model\.22/Div_1",
]


def find_cfg():
    """优先读原始配置。改过的 _dfl_fp16 不能作为输入，否则会叠加。"""
    for p in SRC_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def pick_src(arg):
    """决定从哪个文件读。如果给的是改过的，自动退回 .bak。"""
    if arg:
        base = arg
    else:
        base = find_cfg()
        if not base:
            return None
    if "_dfl_fp16" in base:
        # 传进来了改过的版本 -> 用它的备份（原始 step1 输出）
        if os.path.isfile(base + ".bak"):
            return base + ".bak"
        orig = base.replace("_dfl_fp16", "")
        if os.path.isfile(orig):
            return orig
    return base


def should_fp16(name):
    for pat in KEEP_INT8_PATTERNS:
        if re.search(pat, name):
            return False
    for pat in FP16_PATTERNS:
        if re.search(pat, name):
            return True
    return False


def main():
    src = pick_src(sys.argv[1] if len(sys.argv) > 1 else None)
    if not src:
        print("找不到 quantization.cfg。")
        print("试过这些位置:")
        for p in SRC_CANDIDATES:
            print("  ", p)
        print("\n请在 ~/traffic_count 目录下重跑 step1, 或把 cfg 路径作为参数传进来。")
        return 1

    print("读:", src)
    shutil.copy(src, src + ".bak")

    with open(src, encoding="utf-8") as f:
        lines = f.readlines()

    # ---- 第一步：找 custom_quantize_layers 段，追加我们的层 ----
    out = []
    in_custom = False
    custom_indent = None
    existing = []
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("custom_quantize_layers:"):
            in_custom = True
            out.append(line)
            continue
        if in_custom:
            # 空行或下一个顶层 key 就结束
            if stripped and not stripped.startswith(" "):
                in_custom = False
            elif stripped.startswith("    ") and ":" in stripped:
                name = stripped.strip().split(":")[0]
                existing.append(name)

    # 收集 quantize_parameters 里所有张量名
    all_tensors = []
    for line in lines:
        m = re.match(r"^    (/\S+?):\s*$", line.rstrip("\n"))
        if m:
            all_tensors.append(m.group(1))
    all_tensors = sorted(set(all_tensors))

    to_add = [t for t in all_tensors if should_fp16(t) and t not in existing]

    print("\n配置里已有的 float16 层: %d 个" % len(existing))
    for n in existing:
        print("   ", n)

    print("\n本次要追加成 float16 的层: %d 个" % len(to_add))
    for n in to_add[:30]:
        print("   ", n)
    if len(to_add) > 30:
        print("    ... 还有 %d 个" % (len(to_add) - 30))

    if not to_add:
        print("\n⚠️ 没有匹配到任何层, 检查 FP16_PATTERNS 是否和模型结构对得上")
        return 1

    # ---- 第二步：把新层插进 custom_quantize_layers 段 ----
    result = []
    inserted = False
    for line in lines:
        result.append(line)
        stripped = line.rstrip("\n")
        # 在 custom_quantize_layers 的最后一条之后插入
        if not inserted and stripped.startswith("custom_quantize_layers:"):
            # 找到这个段的所有后续行，插在最后一行后面
            pass

    # 用更简单可靠的方式: 在 custom_quantize_layers 段的末尾插入
    final = []
    i = 0
    while i < len(lines):
        line = lines[i]
        final.append(line)
        if line.rstrip("\n").startswith("custom_quantize_layers:"):
            # 跳过这个段的所有缩进行
            j = i + 1
            while j < len(lines) and (lines[j].startswith("    ") or lines[j].strip() == ""):
                final.append(lines[j])
                j += 1
            # 在段末插入新层
            for name in to_add:
                final.append("    %s: float16\n" % name)
            i = j
            continue
        i += 1

    out_path = src.replace(".quantization.cfg", "_dfl_fp16.quantization.cfg")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(final)

    print("\n写出: %s" % out_path)
    print("备份: %s.bak" % src)
    print("\n下一步:")
    print("  在 ~/traffic_count 目录下, 用 step2 读这个 cfg 生成模型")
    print("  python tools/hybrid_step2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
