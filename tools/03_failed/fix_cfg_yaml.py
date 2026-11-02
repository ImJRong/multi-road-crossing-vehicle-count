#!/usr/bin/env python3
"""把 DFL 相关层标记为 float16 —— 只改 custom_quantize_layers。

查证过的规则（来自 2.5.0 的报错，不是猜的）:
   1. quantize_parameters 是 step1 生成的量化参数表，**只读**。
      动了里面任何一层的 dtype 就会报:
        The quantize_parameters['X']['dtype'] is not allowed to be modified!
      fp16 层的量化参数由 step2 自己重算，用户不用管。
   2. custom_quantize_layers 是唯一该改的地方，格式:
        custom_quantize_layers:
            /model.xx/yy: float16
      注意 step1 生成的是 `custom_quantize_layers: {}`（空字典同行），
      直接往下追加会让 YAML 解析失败，必须先把 {} 去掉。
   3. 层名必须和 cfg 里 quantize_parameters 的 key 一模一样，
      否则报 Invalid operands name。2.5.0 的实际名字:
        /model.22/dfl/Reshape_output_0
        /model.22/dfl/Softmax_output_0_sw_sw
        /model.22/dfl/conv/Conv_output_0
   4. Reshape / Transpose / Concat / Slice / Split 这些布局层没有独立
      量化参数，别往里写。

用法:
    conda activate rknn
    cd ~/traffic_count
    python /mnt/d/multi-road-crossing-vehicle-count/tools/fix_cfg_yaml.py
"""
import glob
import os
import re
import sys

WORK = os.path.expanduser("~/traffic_count/models/step1_250")

# 要保留 fp16 的：只到 DFL 的 Softmax 和加权卷积为止。
#
# 为什么不带 /model.22/Mul_2_output_0（输出去量化）:
#   那一层是最终坐标输出。一旦把它也设成 fp16，编译器会为输出插一个
#   NC1HWC2 布局重排（日志里能看到 116 Reshape ...-rs__float16），
#   输出的元素顺序就乱了：anchor 0 的 cy 本该是 5，读出来变成 397。
#   排布错乱比精度损失严重得多，C++ 那边完全没法用。
#
#   而输出保持 int8 的代价很小：坐标量化步长 2.5px，只有 ±1.25px 误差；
#   实测总偏差 6~22px 主要来自 Softmax 那步（1/255 的格子，每个 bin
#   只剩 16 格，再乘 stride 放大 8/16/32 倍）。所以让 DFL 内部走 fp16
#   就能消掉大头，输出那点小误差留着不影响使用。
PATTERNS = [
    r"/model\.22/dfl/",
]

# 布局层：没有独立量化参数，写了会被拒
SKIP_PATTERNS = [
    r"/Reshape",
    r"/Transpose",
    r"/Concat",
    r"/Slice",
    r"/Split",
    r"/Squeeze",
    r"/Unsqueeze",
]

RE_TENSOR = re.compile(r"^    (/\S+?):\s*$")


def main():
    cfgs = glob.glob(os.path.join(WORK, "*.quantization.cfg"))
    if not cfgs:
        print("找不到 cfg:", WORK)
        return 1
    cfg = cfgs[0]

    # 永远从 step1 的原始产物重建，避免在改过的文件上叠加
    orig = cfg + ".orig"
    src = orig if os.path.isfile(orig) else cfg
    print("cfg :", cfg)
    print("源  :", src, "(原始 step1 输出)" if src == orig else "")

    with open(src, encoding="utf-8") as f:
        lines = f.readlines()

    # ---- 1. 收集真实张量名 ----
    tensors = sorted({m.group(1) for m in (RE_TENSOR.match(l.rstrip("\n")) for l in lines) if m})
    print("\ncfg 里共 %d 个张量" % len(tensors))

    todo = []
    skipped = []
    for t in tensors:
        if not any(re.search(p, t) for p in PATTERNS):
            continue
        if any(re.search(p, t) for p in SKIP_PATTERNS):
            skipped.append(t)
            continue
        todo.append(t)

    print("\n将标记为 float16 (%d 个):" % len(todo))
    for t in todo:
        print("   ", t)
    if skipped:
        print("\n跳过这些布局层 (%d 个，没有独立量化参数):" % len(skipped))
        for t in skipped:
            print("   ", t)
    if not todo:
        print("\n⚠️ 没匹配到可改的层")
        return 1

    # ---- 2. 重建文件：只替换 custom_quantize_layers 段 ----
    out = []
    i = 0
    done = False
    while i < len(lines):
        line = lines[i]
        if line.rstrip("\n").startswith("custom_quantize_layers:"):
            out.append("custom_quantize_layers:\n")
            for name in todo:
                out.append("    %s: float16\n" % name)
            done = True
            i += 1
            # 跳过原来的缩进内容（{} 之后如果还有行）
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
                i += 1
            continue
        out.append(line)
        i += 1

    if not done:
        out = ["custom_quantize_layers:\n"] + \
              ["    %s: float16\n" % n for n in todo] + out

    with open(cfg, "w", encoding="utf-8") as f:
        f.writelines(out)
    print("\n已写入:", cfg)

    # ---- 3. 校验：YAML 能解析 + 名字都在 + quantize_parameters 未被改动 ----
    try:
        from ruamel.yaml import YAML
        y = YAML()
        with open(cfg, encoding="utf-8") as f:
            new = y.load(f)
        with open(src, encoding="utf-8") as f:
            old = y.load(f)

        cq = new.get("custom_quantize_layers") or {}
        qp = new.get("quantize_parameters") or {}

        print("\n✅ YAML 通过, custom_quantize_layers 共 %d 项" % len(cq))

        missing = [k for k in cq if k not in qp]
        if missing:
            print("❌ 这些名字不在 quantize_parameters 里:")
            for m in missing:
                print("   ", m)
            return 1
        print("✅ 层名都能在 quantize_parameters 里找到")

        # 确认 quantize_parameters 一个字节没变
        old_qp = old.get("quantize_parameters") or {}
        if old_qp != qp:
            diffs = [k for k in qp if old_qp.get(k) != qp.get(k)]
            print("❌ quantize_parameters 被改动了! 差异层: %s" % diffs[:5])
            return 1
        print("✅ quantize_parameters 保持原样（只读，未改动）")
    except ImportError:
        print("(yaml 校验跳过: 无 ruamel.yaml)")
    except Exception as e:
        print("❌ YAML 有问题:", e)
        return 1

    print("\n下一步:")
    print("  python /mnt/d/multi-road-crossing-vehicle-count/tools/hybrid_250_step2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
