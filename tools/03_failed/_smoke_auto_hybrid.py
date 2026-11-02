#!/usr/bin/env python3
"""冒烟测试：用极小的假 ONNX 验证 rknn-toolkit2 2.5.0 的
    build(do_quantization=True, dataset=..., auto_hybrid=True)
这个调用到底能不能跑通。

为什么要先做这个：
  auto_hybrid 是 2.5.0 才加的开关，Python 层的 rknn.py 里确实有，
  但真正干活的是编译好的 rknn_base.so —— 如果它不接受这个关键字，
  会在跑完 10 分钟的后才报 TypeError。所以先拿 30 秒的小模型试。

小模型结构故意模仿 YOLO 尾巴：Conv -> Conv -> Reshape -> Softmax -> Conv
（DFL 就是这个形状），这样顺便验证 auto_hybrid 在 Softmax 附近不崩。

跑法（WSL，rknn 环境）：
  conda activate rknn
  python /mnt/d/multi-road-crossing-vehicle-count/tools/_smoke_auto_hybrid.py
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

WORK = os.path.expanduser("~/traffic_count/_smoke")
os.makedirs(WORK, exist_ok=True)
ONNX = os.path.join(WORK, "tiny.onnx")
DATASET = os.path.join(WORK, "dataset.txt")
OUT_INT8 = os.path.join(WORK, "tiny_int8.rknn")
OUT_AUTO = os.path.join(WORK, "tiny_auto.rknn")


def make_tiny_onnx():
    """1x3x64x64 -> conv -> conv -> 输出 [1,4,256]，故意留一个 Softmax。"""
    rng = np.random.default_rng(0)
    inits, nodes = [], []

    def conv(name, cin, cout, k=3):
        w = rng.standard_normal((cout, cin, k, k)).astype(np.float32) * 0.1
        b = rng.standard_normal((cout,)).astype(np.float32) * 0.01
        inits.append(numpy_helper.from_array(w, name + "_w"))
        inits.append(numpy_helper.from_array(b, name + "_b"))
        nodes.append(helper.make_node(
            "Conv", ["x" if name == "c1" else "r1", name + "_w", name + "_b"],
            [name], name=name, pads=[1, 1, 1, 1]))
        return name

    conv("c1", 3, 8)
    nodes.append(helper.make_node("Relu", ["c1"], ["r1"], name="r1"))
    conv("c2", 8, 16)
    # 模仿 DFL：reshape 成一个 (1,16,16,H*W) 再 Softmax
    nodes.append(helper.make_node(
        "Reshape", ["c2", "shape"], ["rs"], name="rs"))
    inits.append(numpy_helper.from_array(np.array([1, 16, -1], np.int64), "shape"))
    nodes.append(helper.make_node("Softmax", ["rs"], ["sm"], name="sm", axis=1))
    # 再卷回 4 通道
    nodes.append(helper.make_node(
        "Reshape", ["sm", "shape2"], ["o"], name="rs2"))
    inits.append(numpy_helper.from_array(np.array([1, 1, -1], np.int64), "shape2"))

    g = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("o", TensorProto.FLOAT, [1, 1, 16384])],
        initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 12)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, ONNX)
    print("小模型 ->", ONNX, "%.2f KB" % (os.path.getsize(ONNX) / 1024))


def make_dataset():
    import cv2
    img_dir = os.path.join(WORK, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(1)
    paths = []
    for i in range(4):
        a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        p = os.path.join(img_dir, "s%d.png" % i)
        cv2.imwrite(p, a)
        paths.append(p)
    with open(DATASET, "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")
    print("校准图 %d 张 -> %s" % (len(paths), DATASET))


def build(auto_hybrid, out):
    from rknn.api import RKNN
    tag = "auto_hybrid=True" if auto_hybrid else "基线（auto_hybrid 不传）"
    print("\n" + "=" * 66)
    print(">>> 开始 build：%s" % tag)
    rknn = RKNN(verbose=False)
    if rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                   target_platform="rk3588") != 0:
        print("config 失败")
        return False
    if rknn.load_onnx(model=ONNX) != 0:
        print("load_onnx 失败")
        return False
    try:
        if auto_hybrid:
            ret = rknn.build(do_quantization=True, dataset=DATASET, auto_hybrid=True)
        else:
            ret = rknn.build(do_quantization=True, dataset=DATASET)
    except TypeError as e:
        print("!! 这个版本的 rknn_base 不接受 auto_hybrid 参数：", e)
        rknn.release()
        return False
    if ret != 0:
        print("build 返回非 0：", ret)
        rknn.release()
        return False
    if rknn.export_rknn(out) != 0:
        print("export 失败")
        rknn.release()
        return False
    print("   成功 ->", out, "%d 字节" % os.path.getsize(out))
    rknn.release()
    return True


if __name__ == "__main__":
    make_tiny_onnx()
    make_dataset()
    ok1 = build(False, OUT_INT8)
    ok2 = build(True, OUT_AUTO)
    print("\n" + "=" * 66)
    print("基线 INT8 :", "OK" if ok1 else "失败")
    print("auto_hybrid:", "OK" if ok2 else "失败")
    if ok1 and ok2:
        s1 = os.path.getsize(OUT_INT8)
        s2 = os.path.getsize(OUT_AUTO)
        print("两模型大小：基线 %d / auto %d（差 %d 字节）" % (s1, s2, s2 - s1))
        print("大小不同 => auto_hybrid 确实起了作用（插了 fp16 层）")
    sys.exit(0 if (ok1 and ok2) else 1)
