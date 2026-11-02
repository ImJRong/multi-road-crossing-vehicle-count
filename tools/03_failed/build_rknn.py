# 在 PC 上用 rknn-toolkit2 走完整流程：ONNX -> build(fp16) -> 模拟器推理。
#
# 跑在哪：WSL 的 rknn 环境
#   cd /mnt/d/multi-road-crossing-vehicle-count
#   python tools/build_rknn.py
#
# 为什么要这样写：
#   rknn-toolkit2 不允许用 load_rknn 加载现成的 .rknn 做模拟器推理，
#   必须在同一个进程里 load_onnx -> build -> init_runtime。
#   所以这个脚本同时承担两个职责：
#     1. 验证转换后模型能否正常推理（第 1 步）
#     2. 以后换模型时重新生成 .rknn（步骤写死，可复现）
#
# 判断输入格式：
#   build 之后模型的输入类型由 config 决定。这里不设 mean/std，
#   分别用 uint8(0~255) 和 float(0~1) 试一次，看哪组输出的分数落在 0~1。
#
# build 阶段要几分钟，属正常。

import os

import cv2
import numpy as np

ROOT = "/mnt/d/multi-road-crossing-vehicle-count"
ONNX_PATH = ROOT + "/models/best_detrac.onnx"
RKNN_OUT = ROOT + "/models/best_detrac_new.rknn"   # 转出来放项目里, 不放 /tmp（重启就没了）
VIDEO = ROOT + "/data/videos/test3.mp4"

IMGSZ = 640
ROI_RECT = (196, 340, 1256, 686)   # 和 configs/test3.json 一致
FRAME_NO = 600


def make_input():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit("打不开视频: " + VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NO)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读不到第 %d 帧" % FRAME_NO)

    x1, y1, x2, y2 = ROI_RECT
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    scale = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    resized = cv2.resize(crop, (nw, nh))
    dx, dy = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    print("裁剪 %dx%d -> 缩放 %dx%d (scale=%.4f), 补边 dx=%d dy=%d"
          % (w, h, nw, nh, scale, dx, dy))
    return canvas


def describe(name, arr):
    arr = np.asarray(arr)
    print("\n--- %s ---" % name)
    print("形状:", arr.shape, "类型:", arr.dtype)
    if arr.ndim != 3:
        return
    d = arr[0]
    scores = d[4]
    print("分数行: min=%.4f max=%.4f" % (scores.min(), scores.max()))
    print("分数 > 0.4 的候选框: %d 个" % int((scores > 0.4).sum()))
    top = np.argsort(scores)[-3:][::-1]
    print("前 3 名的分数和坐标 (分数 / cx / cy / w / h):")
    for i in top:
        print("   %.4f  %.1f  %.1f  %.1f  %.1f"
              % (scores[i], d[0][i], d[1][i], d[2][i], d[3][i]))


def main():
    if not os.path.exists(ONNX_PATH):
        raise SystemExit("找不到 ONNX: " + ONNX_PATH)

    canvas = make_input()

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)

    print("\n[1/4] 配置: target=rk3588, 关掉出问题的融合规则")
    if rknn.config(target_platform="rk3588",
                   disable_rules=["seperate_conv_rs_split"]) != 0:
        raise SystemExit("config 失败")

    print("[2/4] 加载 ONNX: %s" % ONNX_PATH)
    if rknn.load_onnx(model=ONNX_PATH) != 0:
        raise SystemExit("load_onnx 失败")

    print("[3/4] build（fp16，不做量化）... 要几分钟")
    if rknn.build(do_quantization=False) != 0:
        raise SystemExit("build 失败")
    print("    build 完成")

    print("[4/4] 导出到 %s" % RKNN_OUT)
    if rknn.export_rknn(RKNN_OUT) != 0:
        raise SystemExit("export_rknn 失败")
    print("    大小: %.2f MB" % (os.path.getsize(RKNN_OUT) / 1024 / 1024))

    # 同一个进程里已经 build 过，现在模拟器可用
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime 失败")

    # YOLOv8 训练时喂的是 RGB。cv2 读出来是 BGR，必须换过来再送。
    # RKNN 自己不认通道顺序，它只是按内存搬运，所以这一步错了模型就在看错颜色。
    rgb = canvas[..., ::-1]
    inp_rgb = (rgb.astype(np.float32) / 255.0)[None]        # (1,640,640,3) 0~1 RGB
    out_rgb = rknn.inference(inputs=[inp_rgb], data_format="nhwc")
    describe("RKNN 输入 float32 0~1 (RGB)", out_rgb[0])

    inp_bgr = (canvas.astype(np.float32) / 255.0)[None]     # 对照组: BGR
    out_bgr = rknn.inference(inputs=[inp_bgr], data_format="nhwc")
    describe("RKNN 输入 float32 0~1 (BGR，错误示范)", out_bgr[0])

    rknn.release()

    # 有 onnxruntime 就顺手对比一下 ONNX 原始输出（同样是 RGB）
    try:
        import onnxruntime as ort

        blob = np.transpose(inp_rgb[0], (2, 0, 1))[None]     # HWC -> NCHW
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        out_onnx = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
        describe("ONNX 原始模型（基准，RGB）", out_onnx)
    except ImportError:
        print("\n（rknn 环境里没有 onnxruntime，跳过 ONNX 基准对比）")

    print("\n分数落在 0~1 之间、且有若干个超过 0.4 的那组，就是正确的输入格式。")


if __name__ == "__main__":
    main()
