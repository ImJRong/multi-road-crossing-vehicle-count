#!/usr/bin/env python3
"""混合量化 step2: 用改好的 cfg 生成最终模型.

配合 fix_hybrid_cfg.py 使用: 先跑 step1 生成 cfg, 再用 fix 脚本把
坐标分支改成 float16, 最后跑这个生成模型.

用法(必须在 ~/traffic_count 目录下):
    conda activate rknn16
    cd ~/traffic_count
    python tools/hybrid_step2.py
"""
import os
import sys

from rknn.api import RKNN

ONNX = "/mnt/d/multi-road-crossing-vehicle-count/models/best_detrac_head3.onnx"
PROJ = os.path.expanduser("~/traffic_count")

# 优先用改过的 cfg
CFG_CANDIDATES = [
    os.path.join(PROJ, "best_detrac_head3_dfl_fp16.quantization.cfg"),
    os.path.join(PROJ, "best_detrac_head3.quantization.cfg"),
    "/mnt/c/Windows/System32/best_detrac_head3_dfl_fp16.quantization.cfg",
]
DATA_CANDIDATES = [
    os.path.join(PROJ, "best_detrac_head3.data"),
    "/mnt/c/Windows/System32/best_detrac_head3.data",
]
OUT = os.path.join(PROJ, "models", "best_detrac_v8_hybrid.rknn")


def pick(cands, what):
    for p in cands:
        if os.path.isfile(p):
            return p
    print("找不到 %s, 试过:" % what)
    for p in cands:
        print("  ", p)
    return None


def main():
    cfg = pick(CFG_CANDIDATES, "quantization.cfg")
    data = pick(DATA_CANDIDATES, "data")
    if not cfg or not data:
        return 1

    print("cfg :", cfg)
    print("data:", data)
    print("onnx:", ONNX)

    # 统计一下要跑多少 fp16 层，心里有数
    n_fp16 = 0
    with open(cfg, encoding="utf-8") as f:
        for line in f:
            if "float16" in line and ":" in line:
                n_fp16 += 1
    print("cfg 里 float16 行数: %d" % n_fp16)

    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    if rknn.load_onnx(model=ONNX) != 0:
        print("load_onnx 失败")
        return 1

    print("\nstep2 生成中...")
    if rknn.hybrid_quantization_step2(
            model_input=ONNX,
            data_input=data,
            model_quantization_cfg=cfg) != 0:
        rknn.release()
        print("step2 失败")
        return 1

    if rknn.export_rknn(OUT) != 0:
        rknn.release()
        print("export 失败")
        return 1
    rknn.release()

    print("\n完成: %s (%d 字节)" % (OUT, os.path.getsize(OUT)))
    print("\n下一步: 拷到板子验")
    print("  scp %s orangepi@192.168.1.105:~/code/models/" % OUT)
    print("  ./rknn_probe models/best_detrac_v8_hybrid.rknn data/videos/test1.mp4 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
