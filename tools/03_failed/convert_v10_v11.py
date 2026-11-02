#!/usr/bin/env python3
"""用 rknn-toolkit2 2.5.0 转两个模型，做「普通 INT8」vs「自动混合量化」的对照。

产出：
  ~/traffic_count/models/best_detrac_v10_int8.rknn    普通 INT8（对照组）
  ~/traffic_count/models/best_detrac_v11_auto.rknn    自动混合量化（实验组）

为什么要用 2.5.0 转 INT8：
  auto_hybrid 这个开关只有 2.5.0 有。要公平对比，对照组必须用同一个版本转，
  否则分不清是「版本差异」还是「混合量化」带来的差别。
  （v7 是 1.6.0 转的，保留不动，作为最终兜底）

参数来源（全部是照抄 v7 的配方 + 官方文档，没有自己发挥）：
  ONNX       models/best_detrac_head3.onnx  ← 删掉末端 Concat 的那份，唯一能转好的输入
  mean/std   [[0,0,0]] / [[255,255,255]]    ← 和 v7 一致，板子传 uint8 0~255
  platform   rk3588
  其余全部用默认值（不传 optimization_level、不传 quantized_*）

跑法（WSL）：
  conda activate rknn
  cd ~/traffic_count
  python /mnt/d/multi-road-crossing-vehicle-count/tools/convert_v10_v11.py

日志：~/traffic_count/v10_v11_build.log
"""
import os
import sys
import time

ROOT = "/mnt/d/multi-road-crossing-vehicle-count"
ONNX = ROOT + "/models/best_detrac_head3.onnx"
CALIB_DIR = "/mnt/d/rknn_calib"
VIDEO = ROOT + "/data/videos/test1.mp4"
HOME = os.path.expanduser("~")
DATASET = HOME + "/traffic_count/dataset_calib_v10.txt"
LOG = HOME + "/traffic_count/v10_v11_build.log"
OUT_INT8 = HOME + "/traffic_count/models/best_detrac_v10_int8.rknn"
OUT_AUTO = HOME + "/traffic_count/models/best_detrac_v11_auto.rknn"

# 参考值：v7（1.6.0 转的）的实测结果，用来判断新模型是否正常
V7_REF = dict(
    boxes_mean=182.0,      # boxes 反量化后均值
    scores_nonzero=60,     # scores int8 里非零个数（约 60）
    scores_max=0.78,       # scores 最大概率
)

# test1 第 200 帧的取样参数
IMGSZ = 640
FRAME_IDX = 200
CFG = ROOT + "/configs/test1.json"


def roi_rect_from_cfg():
    """从 configs/test1.json 读 ROI 多边形，算外接矩形。

    不写死坐标，是因为 board 上的 test1.json 和 Windows 这份可能不同步，
    写死的话模拟器复现的就不是板子的输入了。
    这里复现的是 main.cpp 第 245 行的做法：
        roi_rect = cv::boundingRect(poly) & cv::Rect(0,0,width,height)
    """
    import json
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    poly = cfg.get("roi") or []
    if len(poly) < 3:
        return None, "配置里没有 ROI，板子会跑整帧"
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    rect = (min(xs), min(ys), max(xs), max(ys))
    return rect, "ROI 外接矩形 %s（来自 %s）" % (str(rect), os.path.basename(CFG))


class Tee:
    """屏幕和日志各写一份，出问题能回看。"""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def preflight():
    """开跑前的硬校验：任何一条不满足就立刻退出，不浪费 10 分钟。"""
    print("=" * 72)
    print("开跑前自检")
    print("-" * 72)

    errors = []

    # 1. 工具版本必须是 2.5.0（auto_hybrid 只在 2.5.0 有）
    import rknn
    from rknn.api import RKNN
    pkg = getattr(rknn, "__path__", [None])[0]
    ver = open(os.path.join(pkg, "VERSION")).read().strip()
    print("  rknn-toolkit2 版本 : %s" % ver)
    if ver != "2.5.0":
        errors.append("版本必须是 2.5.0，当前是 %s（auto_hybrid 在 2.5.0 才有）" % ver)

    # 2. build 必须接受 auto_hybrid 这个关键字
    import inspect
    build_params = list(inspect.signature(RKNN.build).parameters)
    print("  build 参数         : %s" % ", ".join(build_params))
    if "auto_hybrid" not in build_params:
        errors.append("build() 不接受 auto_hybrid，版本不对")

    # 2b. disable_rules 必须能通过 kwargs 传进去（2.5.0 转 YOLO 必需）
    cfg_sig = inspect.signature(RKNN.config)
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in cfg_sig.parameters.values())
    print("  config 吃 kwargs   : %s（disable_rules 靠它传）" % has_kwargs)
    if not has_kwargs:
        errors.append("config() 不吃 **kwargs，没法传 disable_rules")

    # 3. config 必须有阈值参数
    cfg_params = list(inspect.signature(RKNN.config).parameters)
    print("  cos 阈值参数       : %s" %
          ("auto_hybrid_cos_thresh" in cfg_params))
    if "auto_hybrid_cos_thresh" not in cfg_params:
        errors.append("config() 没有 auto_hybrid_cos_thresh")

    # 4. ONNX 存在且是 head3（两个输出，末端无 Concat）
    print("  ONNX               : %s" % ONNX)
    if not os.path.isfile(ONNX):
        errors.append("找不到 ONNX")
    else:
        import onnx
        m = onnx.load(ONNX)
        outs = [o.name for o in m.graph.output]
        print("   输出              : %s" % outs)
        print("   大小              : %.2f MB" % (os.path.getsize(ONNX) / 1048576))
        if len(outs) != 2:
            errors.append("head3 应该有 2 个输出，实际 %d 个" % len(outs))

    # 5. 校准集
    imgs = []
    if not os.path.isdir(CALIB_DIR):
        errors.append("找不到校准图目录 %s" % CALIB_DIR)
    else:
        imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))
        print("  校准图             : %d 张 (%s)" % (len(imgs), CALIB_DIR))
        if len(imgs) < 20:
            errors.append("校准图太少（%d 张），至少要 20 张" % len(imgs))

    # 6. 视频和配置（后面做模拟器验证要用）
    if not os.path.isfile(CFG):
        print("  !! 配置文件不存在，模拟器验证用整帧：%s" % CFG)
    else:
        print("  配置文件           : %s" % CFG)
    if not os.path.isfile(VIDEO):
        print("  !! 视频不存在，模拟器验证会跳过：%s" % VIDEO)
    else:
        print("  验证视频           : %s（第 %d 帧）" % (VIDEO, FRAME_IDX))

    print("-" * 72)
    if errors:
        for e in errors:
            print("  [失败] %s" % e)
        print("=" * 72)
        raise SystemExit("自检没通过，不往下跑")
    print("  自检全部通过")
    print("=" * 72)
    return imgs


def write_dataset(imgs):
    with open(DATASET, "w", encoding="utf-8") as f:
        for name in imgs:
            f.write(os.path.join(CALIB_DIR, name) + "\n")
    print("校准清单 -> %s（%d 行）" % (DATASET, len(imgs)))


def make_canvas():
    """复现板子上的裁切+letterbox，用来在模拟器上验证模型输出。

    板子流程（main.cpp 第 284 行 + rknn_detector.cpp detect()）：
      按 ROI 外接矩形裁 -> 等比缩放到 640 -> 补边 114 -> BGR 转 RGB -> 送 NPU
    """
    import cv2
    import numpy as np

    rect, why = roi_rect_from_cfg()
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, None

    if rect is None:
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
    else:
        x1, y1, x2, y2 = rect
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    s = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(crop, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, dict(scale=s, pad_x=left, pad_y=top, crop=(x1, y1, x2, y2), why=why)


def check_model(rknn, tag):
    """在 PC 模拟器上跑一帧，判断模型是否正常（板子上的 rknn_probe 的等价物）。"""
    import numpy as np

    rgb, geo = make_canvas()
    if rgb is None:
        print("   [跳过] 读不到视频帧")
        return None
    if rknn.init_runtime() != 0:
        print("   [失败] init_runtime")
        return None
    outs = rknn.inference(inputs=[rgb[None]], data_format="nhwc")
    if len(outs) != 2:
        print("   [失败] 输出个数 %d，应为 2" % len(outs))
        return None

    boxes = np.asarray(outs[0]).reshape(4, -1).astype(np.float32)
    scores = np.asarray(outs[1]).reshape(-1).astype(np.float32)

    print("   boxes  min %.2f  max %.2f  均值 %.2f   (v7 参考均值 %.0f)"
          % (boxes.min(), boxes.max(), boxes.mean(), V7_REF["boxes_mean"]))
    print("   scores max %.4f   非零 %d 个    >0.5 的 %d 个   (v7 参考: max %.2f / 非零 %d)"
          % (scores.max(), int((scores != 0).sum()), int((scores > 0.5).sum()),
             V7_REF["scores_max"], V7_REF["scores_nonzero"]))
    print("   4 个通道各自范围（坏模型会四路一样）:")
    for i, name in enumerate(("cx", "cy", "w ", "h ")):
        print("     ch%d(%s) %7.1f ~ %7.1f" % (i, name, boxes[i].min(), boxes[i].max()))

    top = np.argsort(scores)[-3:][::-1]
    for i in top:
        if scores[i] < 0.1:
            continue
        print("     top %.3f  cx=%.1f cy=%.1f w=%.1f h=%.1f"
              % (scores[i], boxes[0][i], boxes[1][i], boxes[2][i], boxes[3][i]))
    return dict(boxes_mean=float(boxes.mean()), scores_max=float(scores.max()),
                nonzero=int((scores != 0).sum()))


def convert(out_path, tag, auto_hybrid, extra_config=None):
    from rknn.api import RKNN

    print("\n" + "=" * 72)
    print(">>> %s" % tag)
    print("    产出: %s" % out_path)
    print("=" * 72)
    t0 = time.time()

    rknn = RKNN(verbose=False)
    cfg = dict(mean_values=[[0, 0, 0]],
               std_values=[[255, 255, 255]],
               target_platform="rk3588",
               # 2.5.0 必须加这一条：不加会在 _p_seperate_conv_rs_split 崩 IndexError。
               # 这不是我们选的优化，是绕开工具自身的 bug（工具日志里自己提示的）。
               disable_rules=["seperate_conv_rs_split"])
    if extra_config:
        cfg.update(extra_config)
    print("    config 实际传入: %s" % cfg)
    if rknn.config(**cfg) != 0:
        print("   [失败] config")
        return None

    if rknn.load_onnx(model=ONNX) != 0:
        print("   [失败] load_onnx")
        return None

    build_kw = dict(do_quantization=True, dataset=DATASET)
    if auto_hybrid:
        build_kw["auto_hybrid"] = True
    print("    build  实际传入: %s" % build_kw)
    if rknn.build(**build_kw) != 0:
        print("   [失败] build")
        return None

    if rknn.export_rknn(out_path) != 0:
        print("   [失败] export_rknn")
        return None
    size = os.path.getsize(out_path)
    print("    导出成功: %d 字节 (%.2f MB)，耗时 %.1f 分钟"
          % (size, size / 1048576, (time.time() - t0) / 60))

    print("   --- 模拟器验证 ---")
    stat = check_model(rknn, tag)
    rknn.release()
    return dict(path=out_path, size=size, stat=stat, minutes=(time.time() - t0) / 60)


def main():
    imgs = preflight()
    write_dataset(imgs)

    if "--preflight-only" in sys.argv:
        print("\n只做自检，不转换（去掉 --preflight-only 才真跑）。")
        rect, why = roi_rect_from_cfg()
        print("  %s" % why)
        rgb, geo = make_canvas()
        if rgb is not None:
            print("  模拟器输入: shape %s dtype %s  mean %.1f"
                  % (rgb.shape, rgb.dtype, rgb.mean()))
            print("  裁切信息  : %s" % geo)
        return

    tee = Tee(LOG)
    sys.stdout = tee
    results = {}
    try:
        results["int8"] = convert(OUT_INT8, "对照组：普通 INT8（2.5.0，全默认参数）",
                                  auto_hybrid=False)
        results["auto"] = convert(OUT_AUTO, "实验组：自动混合量化 auto_hybrid=True（默认阈值 cos=0.98）",
                                  auto_hybrid=True)
    finally:
        sys.stdout = tee.stdout
        tee.flush()
        tee.f.close()

    print("\n" + "=" * 72)
    print("汇总")
    print("-" * 72)
    print("%-8s %-12s %-10s %s" % ("组", "大小", "耗时", "状态"))
    for k, v in results.items():
        if v is None:
            print("%-8s %-12s %-10s %s" % (k, "-", "-", "失败"))
        else:
            print("%-8s %-12d %-10.1f分 %s"
                  % (k, v["size"], v["minutes"], "ok"))

    a, b = results.get("int8"), results.get("auto")
    if a and b:
        print("-" * 72)
        if a["size"] == b["size"]:
            print("!! 两个模型字节数完全一样 => auto_hybrid 一层都没切")
            print("   说明默认阈值 cos=0.98 太松，这个模型的量化误差没触发。")
            print("   下一步：调严 auto_hybrid_cos_thresh 再跑一次。")
        else:
            print("两个模型大小差 %+d 字节 => auto_hybrid 确实切了层" % (b["size"] - a["size"]))
    print("=" * 72)
    print("日志: %s" % LOG)


if __name__ == "__main__":
    main()
