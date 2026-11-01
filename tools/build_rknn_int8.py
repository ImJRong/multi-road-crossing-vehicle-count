# 用 rknn-toolkit2 1.6.0 把拆分头模型转成 INT8（双输出 + 带归一化）
#
# 为什么用 best_detrac_head2.onnx 而不是 best_detrac.onnx：
#   单输出模型把坐标(0~640)和分数(0~1)拼在一个张量里，INT8 量化时一个张量只有一套
#   scale，坐标的量程会把分数的分辨率摊薄到 2.5，0~1 的分数全部舍入成 0。
#   拆成 boxes / scores 两个输出，各自一套 scale，分数才保得住。
#
# 跑法（WSL，rknn20 环境）：
#   cd /mnt/d/multi-road-crossing-vehicle-count
#   python tools/build_rknn_int8.py

import os

import cv2
import numpy as np

from rknn.api import RKNN

ROOT = "/mnt/d/multi-road-crossing-vehicle-count"
ONNX = ROOT + "/models/best_detrac_head2.onnx"
CALIB = ROOT + "/outputs/calib_u8_list.txt"
OUT = ROOT + "/models/best_detrac_int8_v5.rknn"
VIDEO = ROOT + "/data/videos/test1.mp4"
IMGSZ = 640
ROI = (185, 125, 941, 566)


def make_canvas():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读不到视频帧")
    x1, y1, x2, y2 = ROI
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    s = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    canvas[(IMGSZ - nh) // 2:(IMGSZ - nh) // 2 + nh,
           (IMGSZ - nw) // 2:(IMGSZ - nw) // 2 + nw] = cv2.resize(crop, (nw, nh))
    return canvas


def stat(boxes, scores, tag):
    b = np.asarray(boxes).reshape(-1, 8400)      # [4, 8400]
    s = np.asarray(scores).reshape(-1, 8400)[0]  # [8400]
    print("\n--- %s ---" % tag)
    print("  boxes 形状", b.shape, " scores 形状", s.shape)
    print("  分数 %.4f ~ %.4f" % (s.min(), s.max()))
    for th in (0.1, 0.3, 0.5):
        print("  > %.1f: %d 个" % (th, int((s > th).sum())))
    top = np.argsort(s)[-3:][::-1]
    for i in top:
        print("   top %.3f  cx=%.1f cy=%.1f w=%.1f h=%.1f"
              % (s[i], b[0][i], b[1][i], b[2][i], b[3][i]))


def main():
    if not os.path.exists(ONNX):
        raise SystemExit("找不到 " + ONNX)
    if not os.path.exists(CALIB):
        raise SystemExit("找不到校准清单 " + CALIB)

    canvas = make_canvas()

    rknn = RKNN(verbose=False)
    print("[1/5] config（带归一化：输入 0~255，内部除以 255）")
    if rknn.config(target_platform="rk3588",
                   mean_values=[[0, 0, 0]],
                   std_values=[[255, 255, 255]]) != 0:
        raise SystemExit("config 失败")

    print("[2/5] load_onnx: %s" % os.path.basename(ONNX))
    if rknn.load_onnx(model=ONNX) != 0:
        raise SystemExit("load_onnx 失败")

    print("[3/5] build INT8（要 5~10 分钟，用 200 张校准图）")
    if rknn.build(do_quantization=True, dataset=CALIB) != 0:
        raise SystemExit("build 失败")

    print("[4/5] 导出 %s" % OUT)
    if rknn.export_rknn(OUT) != 0:
        raise SystemExit("export_rknn 失败")
    print("    大小 %.2f MB" % (os.path.getsize(OUT) / 1048576))

    print("[5/5] 模拟器验证")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime 失败")

    rgb = canvas[..., ::-1].copy()[None]        # uint8 0~255, NHWC
    out = rknn.inference(inputs=[rgb], data_format="nhwc")
    print("  inference 返回 %d 个输出" % len(out))
    if len(out) == 2:
        stat(out[0], out[1], "INT8 双输出")
    else:
        stat(out[0][0][:4].reshape(4, -1) if False else out[0], out[0], "INT8 单输出（异常）")

    rknn.release()
    print("\n分数落在 0~1、且 >0.3 的有几十个，就说明 INT8 转好了。")


if __name__ == "__main__":
    main()
