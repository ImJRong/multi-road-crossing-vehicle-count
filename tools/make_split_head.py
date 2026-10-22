# 把导出的 YOLOv8 ONNX 拆成双输出：boxes [1,4,8400] + scores [1,1,8400]。
#
# 跑在哪：Windows 或 WSL 都行，只要装了 onnx 和 onnxruntime
#   python tools/make_split_head.py
#
# 为什么要拆：
#   原模型 output0 是 [1,5,8400]，末端一个 Concat 把坐标（实测 1.3~636.6）和
#   分数（0~0.877）拼进同一个张量。INT8 量化是按张量算一套 scale 的，
#   量程被坐标撑到 636.6 后分辨率变成 636.6/255≈2.5，0~1 的分数全部舍入成 0。
#   实测日志里 INT8 的分数 max 就是 0.0000，坐标只剩 2.5 的整数倍 —— 就是这个原因。
#   拆开之后 boxes 和 scores 各自一套 scale，分数才能保住。
#
# 代价：双输出模型在板端要读两个输出，`rknn_detector.cpp` 得跟着改。
#
# 产出：models/best_detrac_head2.onnx
# 自带校验：把两路输出拼回去和原模型 output0 逐元素比，最大差值必须是 0。

import os

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "models", "best_detrac.onnx")
DST = os.path.join(ROOT, "models", "best_detrac_head2.onnx")
VIDEO = os.path.join(ROOT, "data", "videos", "test3.mp4")
IMGSZ = 640


def build_split():
    m = onnx.load(SRC)
    concat = [n for n in m.graph.node if n.op_type == "Concat" and "output0" in n.output]
    if not concat:
        raise SystemExit("没找到末端 Concat，可能已经拆过了")
    concat = concat[0]
    boxes, scores = concat.input[0], concat.input[1]

    m.graph.node.remove(concat)          # 删掉拼接节点
    del m.graph.output[:]                # 原来的 output0 不再存在
    m.graph.output.extend([
        helper.make_tensor_value_info(boxes, TensorProto.FLOAT, [1, 4, 8400]),
        helper.make_tensor_value_info(scores, TensorProto.FLOAT, [1, 1, 8400]),
    ])
    onnx.checker.check_model(m)
    onnx.save(m, DST)
    print("已生成 %s (%.2f MB)" % (DST, os.path.getsize(DST) / 1024 / 1024))


def sample_input():
    """取 test3 的一帧，按主流程裁 ROI + letterbox 到 640，转成 NCHW。"""
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("（找不到 test3.mp4，跳过校验）")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    x1, y1, x2, y2 = 196, 340, 1256, 686      # 与 configs/test3.json 的 ROI 外接矩形一致
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    h, w = crop.shape[:2]
    s = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = cv2.resize(crop, (nw, nh))
    pad = cv2.copyMakeBorder(canvas, (IMGSZ - nh) // 2, IMGSZ - nh - (IMGSZ - nh) // 2,
                            (IMGSZ - nw) // 2, IMGSZ - nw - (IMGSZ - nw) // 2,
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.transpose(pad.astype(np.float32) / 255.0, (2, 0, 1))[None]


def verify():
    """把双输出的两路拼回去，必须和原模型逐元素相等，否则拆分改坏了。"""
    blob = sample_input()
    if blob is None:
        return
    ref = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"]).run(None, {"images": blob})[0]
    out = ort.InferenceSession(DST, providers=["CPUExecutionProvider"]).run(None, {"images": blob})
    joined = np.concatenate([out[0], out[1]], axis=1)
    diff = float(np.abs(joined - ref).max())
    print("校验：原输出 %s，拆分后 %s + %s" % (ref.shape, out[0].shape, out[1].shape))
    print("      拼回后与原模型最大差值 = %g  %s" % (diff, "OK" if diff == 0 else "有问题！"))
    if diff != 0:
        raise SystemExit("拆分结果和原模型不一致，别用这份模型")


if __name__ == "__main__":
    if not os.path.exists(SRC):
        raise SystemExit("缺 %s" % SRC)
    build_split()
    verify()
