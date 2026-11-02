# 验证：模型到底期望什么值域的输入。
# 同一个模型，分别喂 float32 0~1 / uint8 0~255 / float32 0~255，比较输出的分数分布。
# 如果后两种明显偏高，说明"没归一化 + 传 0~255 = 输入放大 255 倍"这个判断成立。
#
# 跑法（WSL，rknn20 环境）：
#   cd /mnt/d/multi-road-crossing-vehicle-count
#   python tools/02_diagnose/verify_norm.py

import cv2
import numpy as np

from rknn.api import RKNN

ROOT = "/mnt/d/multi-road-crossing-vehicle-count"
ONNX = ROOT + "/models/best_detrac.onnx"
VIDEO = ROOT + "/data/videos/test1.mp4"
IMGSZ = 640
ROI = (185, 125, 941, 566)   # x1,y1,x2,y2，和 configs/test1.json 一致


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


def stat(out, tag):
    d = np.asarray(out).reshape(1, 5, 8400)[0]
    sc = d[4]
    print("\n--- %s ---" % tag)
    print("  分数范围 %.4f ~ %.4f" % (sc.min(), sc.max()))
    for th in (0.1, 0.3, 0.5):
        print("  > %.1f 的候选框: %d 个" % (th, int((sc > th).sum())))
    top = np.argsort(sc)[-3:][::-1]
    for i in top:
        print("   top: %.3f  cx=%.1f cy=%.1f w=%.1f h=%.1f"
              % (sc[i], d[0][i], d[1][i], d[2][i], d[3][i]))


def main():
    canvas = make_canvas()
    rgb_u8 = canvas[..., ::-1].copy()          # BGR -> RGB，uint8

    rknn = RKNN(verbose=False)
    # 注意：这里故意不设 mean_values / std_values，跟原来的转换脚本一样
    if rknn.config(target_platform="rk3588") != 0:
        raise SystemExit("config 失败")
    if rknn.load_onnx(model=ONNX) != 0:
        raise SystemExit("load_onnx 失败")
    if rknn.build(do_quantization=False) != 0:
        raise SystemExit("build 失败")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime 失败")

    # 1) float32 0~1（PC 上一直用的、模拟器验证过的）
    a = (rgb_u8.astype(np.float32) / 255.0)[None]
    stat(rknn.inference(inputs=[a], data_format="nhwc")[0], "float32 0~1")

    # 2) uint8 0~255（板子上现在传的）
    b = rgb_u8[None]
    stat(rknn.inference(inputs=[b], data_format="nhwc")[0], "uint8 0~255")

    # 3) float32 0~255（等价于 2，换个类型验证）
    c = rgb_u8.astype(np.float32)[None]
    stat(rknn.inference(inputs=[c], data_format="nhwc")[0], "float32 0~255")

    rknn.release()
    print("\n结论：哪一组分数明显偏高，就说明那种输入跟模型期望的值域不符。")


if __name__ == "__main__":
    main()
