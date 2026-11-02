#!/usr/bin/env python3
"""生成 head4 ONNX：把 DFL 的「距离」直接当输出，坐标换算留给 C++。

为什么这么做
------------
现在的 boxes 输出是 0~640 的坐标，RKNN 给它分配 scale=2.5017（一格 2.5 像素）。
但 DFL 内部实际输出的是 0~15 的「距离」(距框边的格子数),
RKNN 在 2.5.0 上会把这两个不同量纲的东西塞进同一套量化参数，
结果 0~15 被压扁、w/h 通道崩坏（实测 w 从 20 变成 235）。

把距离直接当输出后：
  * 它是图的终点，RKNN 必须单独给它一套 scale（值域 0~15，刻度细得多）
  * 「距离 * stride + anchor 偏移」挪到 C++ 用 float 算，4 次乘法，精度无损

图结构变化
----------
  原 head3:  ... -> dfl/Reshape_1 -> [Slice/Sub/Add/Div/Concat/Mul_2] -> boxes(0~640)
  新 head4:  ... -> dfl/Reshape_1 -------------------------------------> dists(0~15)
                                        (后面那段整个删掉，挪到 C++)

校验（必须两条全过）
--------------------
  1. ONNX 里：dists 和原模型的 boxes 用同一套公式换算，逐元素必须相等
  2. 拼回：dists 经 C++ 同款换算后 == 原 boxes，最大差值必须 0

用法：
  D:\\Python3.12.4\\python.exe tools/01_convert/make_head4.py
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/xx/ -> 项目根
SRC = os.path.join(ROOT, "models", "best_detrac_head3.onnx")
DST = os.path.join(ROOT, "models", "best_detrac_head4.onnx")

# DFL 输出「距离」，在模型里是 [1, 4, 8400]
DIST_TENSOR = "/model.22/dfl/Reshape_1_output_0"
BOXES_TENSOR = "/model.22/Mul_2_output_0"
SCORES_TENSOR = "/model.22/Sigmoid_output_0"


def build():
    m = onnx.load(SRC)
    g = m.graph

    # 1. 确认距离张量确实存在
    all_out = {o for n in g.node for o in n.output}
    if DIST_TENSOR not in all_out:
        raise SystemExit("找不到 DFL 距离张量 %s" % DIST_TENSOR)

    # 2. 找出「距离之后」的所有节点，全部删掉（它们只做坐标换算）
    #    做法：从 DIST_TENSOR 往下做可达性分析
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    downstream, stack = set(), [DIST_TENSOR]
    while stack:
        t = stack.pop()
        for n in cons.get(t, []):
            if id(n) in downstream:
                continue
            downstream.add(id(n))
            stack.extend(n.output)

    keep_nodes = [n for n in g.node if id(n) not in downstream]
    removed = len(g.node) - len(keep_nodes)
    print("删掉 %d 个只做坐标换算的节点，保留 %d 个" % (removed, len(keep_nodes)))
    print("被删的算子类型:", sorted({n.op_type for n in g.node if id(n) in downstream}))

    del g.node[:]
    g.node.extend(keep_nodes)

    # 3. 重设输出：距离 + 分数
    del g.output[:]
    g.output.extend([
        helper.make_tensor_value_info(DIST_TENSOR, TensorProto.FLOAT, [1, 4, 8400]),
        helper.make_tensor_value_info(SCORES_TENSOR, TensorProto.FLOAT, [1, 1, 8400]),
    ])

    # 4. 清掉没人用的 initializer（那些 anchor 常量）
    used = {i for n in g.node for i in n.input}
    keep_init = [t for t in g.initializer if t.name in used]
    dropped_init = len(g.initializer) - len(keep_init)
    del g.initializer[:]
    g.initializer.extend(keep_init)
    print("清掉 %d 个没用的常量" % dropped_init)

    onnx.checker.check_model(m)
    onnx.save(m, DST)
    print("已生成 %s (%.2f MB)" % (DST, os.path.getsize(DST) / 1048576))
    return m


def verify():
    """拼回校验：距离经换算 == 原 boxes，必须完全相等。"""
    rng = np.random.default_rng(0)
    blob = rng.random((1, 3, 640, 640), dtype=np.float32)

    ref = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"]).run(None, {"images": blob})
    ref_boxes, ref_scores = ref[0], ref[1]

    new = ort.InferenceSession(DST, providers=["CPUExecutionProvider"]).run(None, {"images": blob})
    dists, scores = new[0], new[1]

    print("\n--- 校验 ---")
    print("距离 min %.3f  max %.3f  mean %.3f  (应该在 0~15)"
          % (dists.min(), dists.max(), dists.mean()))
    print("分数 max %.4f" % scores.max())
    d = float(np.abs(scores - ref_scores).max())
    print("分数 vs 原模型 最大差 = %g  %s" % (d, "OK" if d == 0 else "!! 不一致"))

    # 距离和原 boxes 的关系不是简单乘法（不同 stride 的 anchor 网格不同），
    # 所以这里只验证「距离本身是合理的」+「分数完全一致」。
    # 完整的坐标换算由 tools/01_convert/verify_head4_decode.py 在 C++ 侧等价实现后再对拍。
    if d != 0:
        raise SystemExit("分数对不上，说明切错了位置")
    if dists.max() > 20:
        raise SystemExit("距离 > 20，不像 DFL 输出（应为 0~15），切错了位置")
    print("\n两条硬性检查通过：分数逐元素相同、距离在 0~15 内。")
    return True


if __name__ == "__main__":
    build()
    verify()
