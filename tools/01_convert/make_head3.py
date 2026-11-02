#!/usr/bin/env python3
"""生成真正"拆头"的 ONNX：删掉末端 Concat，让它的两个输入直接当输出。

和现有 best_detrac_head2.onnx 的区别：
  head2: Concat_3 -> output0 -> Split -> boxes/scores   （Split 是后加的）
  head3: 删掉 Concat_3，Mul_2_output_0 / Sigmoid_output_0 直接当输出

用法：
  D:\\miniconda3\\envs\\yolo\\python.exe tools/01_convert/make_head3.py
校验：两路拼回去必须和原模型 output0 逐元素相等。
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/xx/ -> 项目根
SRC = os.path.join(ROOT, "models", "best_detrac.onnx")
DST = os.path.join(ROOT, "models", "best_detrac_head3.onnx")


def build():
    m = onnx.load(SRC)
    g = m.graph
    concat = [n for n in g.node if n.op_type == "Concat" and "output0" in n.output]
    if not concat:
        raise SystemExit("没找到末端 Concat")
    c = concat[0]
    boxes, scores = c.input[0], c.input[1]

    g.node.remove(c)
    del g.output[:]
    g.output.extend([
        helper.make_tensor_value_info(boxes, TensorProto.FLOAT, [1, 4, 8400]),
        helper.make_tensor_value_info(scores, TensorProto.FLOAT, [1, 1, 8400]),
    ])
    onnx.checker.check_model(m)
    onnx.save(m, DST)
    print("已生成 %s (%.2f MB), nodes=%d" % (DST, os.path.getsize(DST) / 1024 / 1024, len(g.node)))
    return boxes, scores


def verify(boxes, scores):
    rng = np.random.default_rng(0)
    blob = rng.random((1, 3, 640, 640), dtype=np.float32)
    ref = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"]).run(None, {"images": blob})[0]
    out = ort.InferenceSession(DST, providers=["CPUExecutionProvider"]).run(
        None, {"images": blob})
    joined = np.concatenate([out[0], out[1]], axis=1)
    d = float(np.abs(joined - ref).max())
    print("输出名: %s / %s" % (boxes, scores))
    print("拼回 vs 原 output0 最大差值 = %g  %s" % (d, "OK" if d == 0 else "有问题"))
    if d != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    b, s = build()
    verify(b, s)
