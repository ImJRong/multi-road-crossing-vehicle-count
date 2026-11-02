#!/usr/bin/env python3
"""用 FP32 ONNX 当真值，导出第 N 帧的真实框，给板子做对比标尺。

板子上两个模型都只能在 NPU 上跑，没有真值可比。
这里在 PC 上用 onnxruntime 跑同一个 ONNX 的 FP32 结果，
按板子完全相同的 ROI 裁切 + letterbox 流程还原成原图坐标，
输出成一行一个框的文本，传到板子上和 box_cmp 的结果对照。

用法：
  D:\\Python3.12.4\\python.exe tools/02_diagnose/dump_truth_boxes.py [帧号]
产出：
  outputs/truth_boxes.txt
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/xx/ -> 项目根
ONNX = os.path.join(ROOT, "models", "best_detrac.onnx")
VIDEO = os.path.join(ROOT, "data", "videos", "test1.mp4")
OUT = os.path.join(ROOT, "outputs", "truth_boxes.txt")
IMGSZ = 640
ROI = (278, 90, 983, 541)      # x1,y1,x2,y2  和板子上 main.cpp 算出来的一致
CONF = 0.4
IOU_TH = 0.4


def letterbox(frame):
    x1, y1, x2, y2 = ROI
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    s = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    dx, dy = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(crop, (nw, nh))
    return canvas, s, dx, dy


def nms(boxes):
    keep = []
    for b in sorted(boxes, key=lambda x: -x[4]):
        drop = False
        for k in keep:
            ix = min(b[2], k[2]) - max(b[0], k[0])
            iy = min(b[3], k[3]) - max(b[1], k[1])
            if ix <= 0 or iy <= 0:
                continue
            inter = ix * iy
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if ua > 0 and inter / ua > IOU_TH:
                drop = True
                break
        if not drop:
            keep.append(b)
    return keep


def main():
    frame_no = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读不到视频帧")

    canvas, s, dx, dy = letterbox(frame)
    blob = (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    blob = blob.transpose(2, 0, 1)[None]

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": blob})[0][0]        # [5, 8400]
    print("ONNX 输出 shape:", out.shape)

    boxes5 = out[:4].T                                  # [8400,4]
    scores = out[4]
    dets = []
    for i in range(boxes5.shape[0]):
        if scores[i] < CONF:
            continue
        cx, cy, w, h = boxes5[i]
        # 640 空间 -> 原图（先扣补边、再除 scale、最后加 ROI 原点）
        x1 = (cx - w / 2 - dx) / s + ROI[0]
        y1 = (cy - h / 2 - dy) / s + ROI[1]
        x2 = (cx + w / 2 - dx) / s + ROI[0]
        y2 = (cy + h / 2 - dy) / s + ROI[1]
        dets.append([x1, y1, x2, y2, float(scores[i])])
    dets = nms(dets)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# FP32 ONNX 真值, test1 第 %d 帧, conf>%.1f, ROI=%s\n" % (frame_no, CONF, ROI))
        f.write("# x1 y1 x2 y2 cx cy w h score\n")
        for d in dets:
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            f.write("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.4f\n"
                    % (d[0], d[1], d[2], d[3], cx, cy, d[2] - d[0], d[3] - d[1], d[4]))

    print("真值框 %d 个 -> %s" % (len(dets), OUT))
    print("（原图坐标，已加回 ROI 原点）")
    for d in dets[:8]:
        print("  cx=%7.1f cy=%7.1f w=%6.1f h=%6.1f score=%.3f"
              % ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2, d[2] - d[0], d[3] - d[1], d[4]))


if __name__ == "__main__":
    main()
