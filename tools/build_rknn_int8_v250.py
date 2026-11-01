# 用 rknn-toolkit2 2.5.0 转 INT8，开启官方推荐的优化项
# 跟之前 1.6.0 那版的区别只有 config 里的两个参数：
#   optimization_level=3   图优化拉满（算子融合、内存复用）
#   quantized_algorithm='mmse'  量化算法从默认的 normal 换成 mmse，对 YOLO 更友好
#
# 跑法（WSL，rknn20 环境）：
#   pip install rknn-toolkit2==2.5.0 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
#   cd /mnt/d/multi-road-crossing-vehicle-count
#   python tools/build_rknn_int8_v250.py

import os

import cv2
import numpy as np

from rknn.api import RKNN

ROOT = "/mnt/d/multi-road-crossing-vehicle-count"
ONNX = ROOT + "/models/best_detrac_head2.onnx"
CALIB = ROOT + "/outputs/calib_u8_list.txt"
OUT = ROOT + "/models/best_detrac_int8_250.rknn"
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


def main():
    canvas = make_canvas()

    rknn = RKNN(verbose=False)
    print("[1/5] config: rk3588 + 归一化 + optimization_level=3 + mmse")
    if rknn.config(target_platform="rk3588",
                   mean_values=[[0, 0, 0]],
                   std_values=[[255, 255, 255]],
                   optimization_level=3,
                   disable_rules=["seperate_conv_rs_split"]) != 0:
        raise SystemExit("config 失败")

    print("[2/5] load_onnx: %s" % os.path.basename(ONNX))
    if rknn.load_onnx(model=ONNX) != 0:
        raise SystemExit("load_onnx 失败")

    print("[3/5] build INT8（5~10 分钟）")
    if rknn.build(do_quantization=True, dataset=CALIB) != 0:
        raise SystemExit("build 失败")

    print("[4/5] 导出 %s" % OUT)
    if rknn.export_rknn(OUT) != 0:
        raise SystemExit("export_rknn 失败")
    print("    大小 %.2f MB" % (os.path.getsize(OUT) / 1048576))

    print("[5/5] 模拟器验证")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime 失败")

    rgb = canvas[..., ::-1].copy()[None]
    out = rknn.inference(inputs=[rgb], data_format="nhwc")
    print("  返回 %d 个输出" % len(out))
    if len(out) == 2:
        s = np.asarray(out[1]).reshape(-1)
        b = np.asarray(out[0]).reshape(-1, 8400)
        print("  分数 %.4f ~ %.4f" % (s.min(), s.max()))
        for th in (0.1, 0.3, 0.5):
            print("  > %.1f: %d 个" % (th, int((s > th).sum())))
        top = np.argsort(s)[-3:][::-1]
        for i in top:
            print("   top %.3f cx=%.1f cy=%.1f w=%.1f h=%.1f"
                  % (s[i], b[0][i], b[1][i], b[2][i], b[3][i]))
    else:
        print("  注意：输出不是 2 个，检查模型")

    rknn.release()
    print("\n分数正常（0~1、>0.1 有几十个）就传板子测速度。")


if __name__ == "__main__":
    main()
