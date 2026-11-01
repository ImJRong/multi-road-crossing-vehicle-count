#!/usr/bin/env python3
"""用 rknn-toolkit2 2.5.0 做混合量化(坐标分支留 fp16).

背景:
    1.6.0 的 hybrid_quantization_step2 必定崩在
        quant_optimizer._p_adjust_tanh_sigmoid
        KeyError: '/model.22/Concat_1_output_0'
    它要找的名字和 cfg 里实际的
        '/model.22/Concat_1_output_0_shape4_/model.22/Sigmoid'
    对不上。改 35 层、改 2 层、只改 dtype, 三次全崩同一行,
    说明和改什么无关, 是工具自身的问题。换 2.5.0 试。

用法(WSL, 必须在 ~/traffic_count 下):
    conda activate rknn
    cd ~/traffic_count
    python /mnt/d/multi-road-crossing-vehicle-count/tools/hybrid_250.py

产出:
    ~/traffic_count/models/best_detrac_v8_h250.rknn
    ~/traffic_count/models/step1_250/  (step1 的 cfg/data)
"""
import glob
import os
import re
import shutil
import sys

from rknn.api import RKNN

ONNX = "/mnt/d/multi-road-crossing-vehicle-count/models/best_detrac_head3.onnx"
CALIB_DIR = "/mnt/d/rknn_calib"
PROJ = os.path.expanduser("~/traffic_count")
MODELS = os.path.join(PROJ, "models")
WORK = os.path.join(MODELS, "step1_250")
DATASET = os.path.join(PROJ, "dataset_calib_250.txt")
OUT = os.path.join(MODELS, "best_detrac_v8_h250.rknn")

# 想保留 fp16 的层：DFL 核心 + 最终坐标输出
FP16_PATTERNS = [
    r"/model\.22/dfl/",
    r"^/model\.22/Mul_2_output_0$",
    r"^/model\.22/Mul_2_output_0_shape4$",
]

RE_TENSOR = re.compile(r"^    (/\S+?):\s*$")
RE_DTYPE_INT8 = re.compile(r"^\s+dtype:\s*int8\s*$")


def make_dataset():
    imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))
    with open(DATASET, "w", encoding="utf-8") as f:
        for n in imgs:
            f.write(os.path.join(CALIB_DIR, n) + "\n")
    print("校准图 %d 张" % len(imgs))


def new_rknn():
    r = RKNN(verbose=False)
    base = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    try:
        # 2.5.0 特有: 不加这个 build 会崩 IndexError
        r.config(disable_rules=["seperate_conv_rs_split"], **base)
        print("已加 disable_rules (2.5.0 必需)")
    except TypeError:
        r.config(**base)
        print("此版本无 disable_rules, 用默认参数")
    if r.load_onnx(model=ONNX) != 0:
        raise RuntimeError("load_onnx 失败")
    return r


def step1(use_proposal=False):
    """生成量化配置。

    use_proposal=False 的原因（2026-11-01 实测）:
        2.5.0 带 proposal=True 时崩在
            hybrid_proposal.proposal -> expand_batch -> infer_shapes
            KeyError: '/model.22/dfl/conv/Conv_output_0'
        这条路要遍历整图推算 shape, 而 DFL 那段的层名它认不出来。
        我们本来也不需要它的"建议"(1.6.0 那次的建议全落在骨干前 7 层,
        对坐标精度毫无帮助)。所以直接生成默认配置, 自己改。
    """
    print("\n" + "=" * 62)
    print("2.5.0 step1: 生成量化配置 (proposal=%s)" % use_proposal)
    print("=" * 62)
    os.makedirs(WORK, exist_ok=True)
    old_cwd = os.getcwd()
    os.chdir(WORK)          # 生成的文件落在固定目录, 不污染当前目录
    try:
        rknn = new_rknn()
        ret = rknn.hybrid_quantization_step1(
            dataset=DATASET,
            rknn_batch_size=1,
            proposal=use_proposal,
            proposal_dataset_size=8 if use_proposal else 1,
        )
        rknn.release()
    finally:
        os.chdir(old_cwd)
    if ret != 0:
        raise RuntimeError("step1 失败, 返回 %d" % ret)

    files = sorted(glob.glob(os.path.join(WORK, "*")))
    print("\nstep1 生成:")
    for p in files:
        print("  %s (%d 字节)" % (p, os.path.getsize(p)))
    return files


def patch_cfg(cfg_path):
    """把 DFL 相关层改成 float16（同时写 custom_quantize_layers 和 dtype 字段）。"""
    print("\n" + "-" * 62)
    print("修改配置: DFL 层 -> float16")
    print("-" * 62)
    shutil.copy(cfg_path, cfg_path + ".orig")
    with open(cfg_path, encoding="utf-8") as f:
        lines = f.readlines()

    # 配置里已有的 float16 层
    existing = set()
    in_custom = False
    for line in lines:
        s = line.rstrip("\n")
        if s.startswith("custom_quantize_layers:"):
            in_custom = True
            continue
        if in_custom:
            if s and not s.startswith(" "):
                in_custom = False
            elif "float16" in s:
                name = s.strip().split(":")[0].strip()
                existing.add(name)
    print("原本的 float16 层: %d 个" % len(existing))
    for n in sorted(existing):
        print("   ", n)

    # quantize_parameters 里的所有张量
    tensors = sorted({m.group(1) for m in (RE_TENSOR.match(l.rstrip("\n")) for l in lines) if m})
    todo = [t for t in tensors
            if any(re.search(p, t) for p in FP16_PATTERNS) and not t.endswith("_int8")]

    print("\n要改成 float16 的: %d 个" % len(todo))
    for t in todo:
        print("   ", t)
    if not todo:
        print("⚠️ 一个都没匹配到，检查 FP16_PATTERNS")
        return 0

    # (1) 追加到 custom_quantize_layers 段末尾
    final = []
    i = 0
    while i < len(lines):
        line = lines[i]
        final.append(line)
        if line.rstrip("\n").startswith("custom_quantize_layers:"):
            j = i + 1
            while j < len(lines) and (lines[j].startswith("    ") or lines[j].strip() == ""):
                final.append(lines[j])
                j += 1
            for name in todo:
                if name not in existing:
                    final.append("    %s: float16\n" % name)
            i = j
            continue
        i += 1

    # (2) 把对应张量的 dtype 也改成 float16
    cur = None
    for idx, line in enumerate(final):
        m = RE_TENSOR.match(line.rstrip("\n"))
        if m:
            cur = m.group(1)
            continue
        if cur in todo and RE_DTYPE_INT8.match(line):
            final[idx] = line.replace("int8", "float16")
            cur = None

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.writelines(final)

    n = sum(1 for l in final if "float16" in l)
    print("\n已写入:", cfg_path)
    print("备份:", cfg_path + ".orig")
    print("cfg 里 float16 行数:", n)
    return n


def step2(cfg, data, model_file=None):
    """生成模型。

    关键: step2 之前**不能**调 load_onnx —— 2.5.0 会报
        "The model already exists. Do not call 'load_onnx' before
         executing 'hybrid_quantization_step2'"
    模型从 step1 生成的 .model 文件读(model_input 传那个路径)。
    """
    print("\n" + "=" * 62)
    print("2.5.0 step2: 生成模型")
    print("=" * 62)

    # 找 step1 的 .model 文件
    if not model_file:
        cands = glob.glob(os.path.join(WORK, "*.model"))
        model_file = cands[0] if cands else None
    if not model_file or not os.path.isfile(model_file):
        raise RuntimeError("找不到 step1 生成的 .model 文件, 目录: %s" % WORK)
    print("model:", model_file)
    print("data :", data)
    print("cfg  :", cfg)

    rknn = RKNN(verbose=False)
    base = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    try:
        rknn.config(disable_rules=["seperate_conv_rs_split"], **base)
    except TypeError:
        rknn.config(**base)

    # 注意: 这里故意不 load_onnx
    ret = rknn.hybrid_quantization_step2(
        model_input=model_file,
        data_input=data,
        model_quantization_cfg=cfg,
    )
    if ret != 0:
        rknn.release()
        raise RuntimeError("step2 失败, 返回 %d" % ret)
    if rknn.export_rknn(OUT) != 0:
        rknn.release()
        raise RuntimeError("export 失败")
    rknn.release()
    print("\n完成: %s (%d 字节)" % (OUT, os.path.getsize(OUT)))


def main():
    if not os.path.isfile(ONNX):
        print("找不到 ONNX:", ONNX)
        return 1
    make_dataset()
    files = step1(use_proposal=False)

    cfg = next((p for p in files if p.endswith(".json") or p.endswith(".cfg")), None)
    data = next((p for p in files if p.endswith(".data") or p.endswith(".npy")), None)
    if not cfg or not data:
        print("\n⚠️ 没找到 cfg/data。生成的文件:")
        for p in files:
            print("  ", p)
        return 1

    n = patch_cfg(cfg)
    if n == 0:
        return 1
    step2(cfg, data)

    print("\n下一步: 拷到板子验")
    print("  ./rknn_probe models/best_detrac_v8_h250.rknn data/videos/test1.mp4 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
