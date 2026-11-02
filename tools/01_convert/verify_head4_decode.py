#!/usr/bin/env python3
"""验证 head4 的坐标换算公式：距离 -> 框坐标，必须和原 head3 的 boxes 完全相等。

这是给 C++ 用的公式, 先在 Python 里证明它对, 再照抄进 rknn_detector.cpp。

推导（从 head3 的 ONNX 图里读出来的常量）
---------------------------------------
  d = dists            [1,4,8400]   DFL 输出的距离, 0~16
  d_left  = d[0:2]     Slice(axis=1, start=0, end=2)
  d_right = d[2:4]     Slice(axis=1, start=2, end=4)

  anchor   = Constant_12 = [0.5, 1.5, 2.5, ...]   形状 [1,2,8400]  (x 和 y 各一份)
  stride   = Constant_15 = 8

  x1 = anchor - d_left
  y1 = anchor - d_left 的 y 那份
  x2 = anchor + d_right
  y2 = anchor + d_right 的 y 那份
  cx = (x1 + x2) / 2
  cy = (y1 + y2) / 2
  w  = x2 - x1
  h  = y2 - y1
  boxes = concat(cx, cy, w, h) * stride

注意：anchor 只有 0~80，不是 0~640。stride 在最后才乘上去。

用法：
  D:\\Python3.12.4\\python.exe tools/01_convert/verify_head4_decode.py
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/xx/ -> 项目根
HEAD3 = os.path.join(ROOT, "models", "best_detrac_head3.onnx")
HEAD4 = os.path.join(ROOT, "models", "best_detrac_head4.onnx")


def read_constants():
    """从 head3 的图里把 anchor 网格和 stride 读出来，不写死。"""
    m = onnx.load(HEAD3)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    anchor = inits["/model.22/Constant_12_output_0"]      # [1,2,8400]
    stride = inits["/model.22/Constant_15_output_0"]      # [1,8400] 不是标量！
    print("anchor 形状 %s  范围 %.2f ~ %.2f"
          % (anchor.shape, anchor.min(), anchor.max()))
    print("stride 形状 %s  唯一值 %s"
          % (stride.shape, np.unique(stride.ravel())))
    # 分段统计，方便核对
    for lo, hi, name in [(0, 6400, "stride 8"), (6400, 8000, "stride 16"),
                         (8000, 8400, "stride 32")]:
        seg = stride.ravel()[lo:hi]
        print("  [%4d,%4d) %-9s x %d 个 anchor  stride 唯一值 %s"
              % (lo, hi, name, hi - lo, np.unique(seg)))
    return anchor.astype(np.float32), stride.reshape(8400).astype(np.float32)


def decode(dists, anchor, stride):
    """距离 -> boxes，完全照 ONNX 图里的算法规格来。

    关键：stride 是 8400 长的数组（8/16/32 三段），不是标量。
    """
    d = np.asarray(dists, dtype=np.float32).reshape(4, 8400)
    d_left = d[0:2]        # x 的左右, y 的上下
    d_right = d[2:4]

    cx_cy_part = anchor.reshape(2, 8400)     # [2,8400]
    x1y1 = cx_cy_part - d_left
    x2y2 = cx_cy_part + d_right

    center = (x1y1 + x2y2) / 2.0
    size = x2y2 - x1y1
    boxes = np.concatenate([center, size], axis=0) * stride.reshape(1, 8400)
    return boxes.reshape(4, 8400)


def main():
    anchor, stride = read_constants()

    rng = np.random.default_rng(0)
    blob = rng.random((1, 3, 640, 640), dtype=np.float32)

    ref = ort.InferenceSession(HEAD3, providers=["CPUExecutionProvider"]).run(None, {"images": blob})
    ref_boxes = ref[0].reshape(4, 8400)

    new = ort.InferenceSession(HEAD4, providers=["CPUExecutionProvider"]).run(None, {"images": blob})
    dists, scores = new[0], new[1]

    mine = decode(dists, anchor, stride)

    diff = np.abs(mine - ref_boxes)
    print("\n--- 结果 ---")
    print("我的解码 vs 原 boxes 最大差 = %g" % diff.max())
    print("原 boxes 范围 %.2f ~ %.2f" % (ref_boxes.min(), ref_boxes.max()))
    print("我的范围   %.2f ~ %.2f" % (mine.min(), mine.max()))

    if diff.max() < 1e-4:
        print("\n[通过] 公式完全正确，可以照抄进 C++。")
        print("C++ 里要做的（每帧 8400 次）:")
        print("  stride[i] 按段取值: i<6400 -> 8, i<8000 -> 16, else 32")
        print("  anchor[i] 也要按段算: 80x80 / 40x40 / 20x20 网格")
        print("  x1 = anchor[i] - d[0][i];  y1 = anchor[i] - d[1][i];")
        print("  x2 = anchor[i] + d[2][i];  y2 = anchor[i] + d[3][i];")
        print("  cx = (x1+x2)*0.5f; cy = (y1+y2)*0.5f;")
        print("  w  = x2-x1;        h  = y2-y1;")
        print("  四者都 *= stride[i]")
        return 0
    print("\n[失败] 公式不对，差值 %.6f" % diff.max())
    bad = np.unravel_index(np.argmax(diff), diff.shape)
    print("最差的点 %s: 我的 %.4f / 原 %.4f" % (bad, mine[bad], ref_boxes[bad]))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
