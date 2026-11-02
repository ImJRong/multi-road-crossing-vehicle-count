#!/usr/bin/env python3
"""只跑混合量化的 step2（step1 已经跑过了）。

2.5.0 的坑:
    step2 之前**不能** load_onnx, 否则报
        The model already exists. Do not call 'load_onnx' before
        executing 'hybrid_quantization_step2'
    model_input 要传 step1 生成的 .model 文件。

用法(WSL):
    conda activate rknn
    cd ~/traffic_count
    python /mnt/d/multi-road-crossing-vehicle-count/tools/hybrid_250_step2.py
"""
import glob
import os
import sys

from rknn.api import RKNN

PROJ = os.path.expanduser("~/traffic_count")
WORK = os.path.join(PROJ, "models", "step1_250")
# v9 = DFL 内部 fp16、输出仍 int8（v8 把输出也搞成 fp16，布局乱了不能用）
OUT = os.path.join(PROJ, "models", "best_detrac_v9_h250.rknn")


def main():
    model = glob.glob(os.path.join(WORK, "*.model"))
    data = glob.glob(os.path.join(WORK, "*.data"))
    cfg = glob.glob(os.path.join(WORK, "*.quantization.cfg"))

    if not (model and data and cfg):
        print("step1 的产物不全, 目录:", WORK)
        for p in sorted(glob.glob(os.path.join(WORK, "*"))):
            print("  ", os.path.basename(p), os.path.getsize(p))
        return 1

    model, data, cfg = model[0], data[0], cfg[0]
    print("model:", model)
    print("data :", data)
    print("cfg  :", cfg)

    n_fp16 = sum(1 for l in open(cfg, encoding="utf-8") if "float16" in l)
    print("cfg 里 float16 行数:", n_fp16)
    if n_fp16 == 0:
        print("⚠️ 没有 fp16 层，先跑 hybrid_250.py 的 patch 部分")
        return 1

    rknn = RKNN(verbose=True)
    base = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    try:
        rknn.config(disable_rules=["seperate_conv_rs_split"], **base)
    except TypeError:
        rknn.config(**base)

    # 关键：不调 load_onnx，直接把 step1 的 .model 喂进去
    print("\nstep2 生成中...")
    ret = rknn.hybrid_quantization_step2(
        model_input=model,
        data_input=data,
        model_quantization_cfg=cfg,
    )
    if ret != 0:
        rknn.release()
        print("step2 失败, 返回", ret)
        return 1

    if rknn.export_rknn(OUT) != 0:
        rknn.release()
        print("export 失败")
        return 1
    rknn.release()

    print("\n完成: %s (%d 字节)" % (OUT, os.path.getsize(OUT)))
    print("\n下一步: 拷到板子验")
    print("  scp %s orangepi@192.168.1.105:~/code/models/" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
