#!/usr/bin/env python3
"""混合量化: 把 DFL / 坐标相关的层保留 fp16, 其余 int8.

为什么要混合:
    纯 INT8 时 boxes 张量 scale=2.5017, 一格 2.5 像素(640空间)。
    而 YOLOv8 的坐标是 DFL 加权平均出来的(58 组 Sigmoid+Mul),
    每个 bin 被量化后误差累加, 实测框中心偏左上 6~15 px,
    车越小(越远)偏得越厉害。
    纯 FP16 完全没这个问题, 但 RK3588 上只有 6.6 fps。
    混合量化只让那几十个轻量层走 fp16, 速度和精度折中。

流程(rknn-toolkit2 1.6.0 两阶段 API):
    step1: 生成量化配置 json + 校准数据文件
    step2: 读改好的配置, 生成最终模型

用法(WSL 里):
    conda activate rknn16
    python /mnt/d/multi-road-crossing-vehicle-count/tools/convert_head3_hybrid.py

产出:
    ~/traffic_count/models/best_detrac_v8_hybrid.rknn
    step1 生成的 cfg/data 会打印出真实路径
"""
import glob
import json
import os
import sys

from rknn.api import RKNN

ONNX = "/mnt/d/multi-road-crossing-vehicle-count/models/best_detrac_head3.onnx"
CALIB_DIR = "/mnt/d/rknn_calib"
PROJ = os.path.expanduser("~/traffic_count")
MODELS = os.path.join(PROJ, "models")

DATASET = os.path.join(PROJ, "dataset_calib_hybrid.txt")
OUT = os.path.join(MODELS, "best_detrac_v8_hybrid.rknn")


def make_dataset():
    imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))
    with open(DATASET, "w", encoding="utf-8") as f:
        for n in imgs:
            f.write(os.path.join(CALIB_DIR, n) + "\n")
    print("校准图 %d 张 -> %s" % (len(imgs), DATASET))
    return len(imgs)


def new_rknn():
    r = RKNN(verbose=False)
    r.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    if r.load_onnx(model=ONNX) != 0:
        raise RuntimeError("load_onnx 失败")
    return r


def step1():
    """生成量化配置。proposal=True 让工具自己分析哪些层该留 fp16。"""
    print("\n" + "=" * 62)
    print("step1: 生成混合量化配置 (proposal 自动分析)")
    print("=" * 62)

    before = set(glob.glob(os.path.join(MODELS, "*")))
    rknn = new_rknn()
    ret = rknn.hybrid_quantization_step1(
        dataset=DATASET,
        rknn_batch_size=1,
        proposal=True,
        proposal_dataset_size=8,   # 用 8 张图做敏感度分析, 快一点
    )
    rknn.release()
    if ret != 0:
        raise RuntimeError("hybrid step1 失败, 返回 %d" % ret)

    after = set(glob.glob(os.path.join(MODELS, "*")))
    new_files = sorted(after - before)
    print("\nstep1 新生成的文件:")
    if not new_files:
        print("  ⚠️ 一个都没生成, 去 %s 下手工看" % MODELS)
    for p in new_files:
        print("  %s (%d 字节)" % (p, os.path.getsize(p)))
    return new_files


def inspect_cfg(cfg_path):
    """看配置里建议了哪些层保留 fp16。"""
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    # step1 的配置是多层嵌套的 dict, 递归找 quantized_dtype
    fp16, int8, other = [], [], []

    def walk(node, path=""):
        if isinstance(node, dict):
            dt = node.get("quantized_dtype")
            if dt is not None:
                item = (path or node.get("name", "?"), dt)
                if "float16" in str(dt):
                    fp16.append(item)
                elif "int8" in str(dt):
                    int8.append(item)
                else:
                    other.append(item)
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, path + "/" + str(k) if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, "%s[%d]" % (path, i))

    walk(cfg)

    print("\n配置统计:")
    print("  保留 fp16 的层/张量: %d 个" % len(fp16))
    print("  量化 int8 的:         %d 个" % len(int8))
    if other:
        print("  其他类型:             %d 个" % len(other))

    if fp16:
        print("\n建议 fp16 的(最多列 25 个):")
        for name, dt in fp16[:25]:
            print("   ", name, "->", dt)
        if len(fp16) > 25:
            print("    ... 还有 %d 个" % (len(fp16) - 25))
    else:
        print("\n  ⚠️ 一个 fp16 层都没建议 -> 等于全 int8, 起不到作用.")
        print("     需要手工改配置, 把 DFL 相关层(Softmax/Sigmoid/Mul/Concat)设为 float16")
    return len(fp16)


def step2(data_path, cfg_path):
    print("\n" + "=" * 62)
    print("step2: 按配置生成模型")
    print("=" * 62)
    rknn = new_rknn()
    ret = rknn.hybrid_quantization_step2(
        model_input=ONNX,
        data_input=data_path,
        model_quantization_cfg=cfg_path,
    )
    if ret != 0:
        rknn.release()
        raise RuntimeError("hybrid step2 失败, 返回 %d" % ret)
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
    new_files = step1()

    cfg = next((p for p in new_files if p.endswith(".json")), None)
    data = next((p for p in new_files if p.endswith(".npy") or p.endswith(".data")), None)
    if not cfg or not data:
        print("\n⚠️ 没找到 cfg/data 文件, 停下. 生成的文件在上面的列表里.")
        return 1

    print("\n" + "-" * 62)
    inspect_cfg(cfg)
    print("-" * 62)

    step2(data, cfg)

    print("\n下一步(板子上):")
    print("  ./rknn_probe models/best_detrac_v8_hybrid.rknn data/videos/test1.mp4 200")
    print("  判据: scores 的 ==zp(-128) 约 8340 = 正常")
    print("  再跑 400 帧看帧率和框位置:")
    print("  /tmp/box_check models/best_detrac_v8_hybrid.rknn data/videos/test1.mp4 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
