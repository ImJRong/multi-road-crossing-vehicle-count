#!/usr/bin/env python3
"""验证 head4 方案是否成立：用 1.6.0 转 head4，看「距离」输出的量化刻度。

这是整个方案的生死线。只看一件事：
  head4 的距离输出（真实值 0~16）拿到的 scale，
  是不是真的比 head3 的 boxes 输出（真实值 0~640）小很多。

  如果 scale 还是 2.5 左右 => 距离照样被压扁 => 方案不成立，白做
  如果 scale 变成 0.0xx 级别  => 刻度细了 16 倍以上 => 方案成立

跑法（WSL）：
  conda activate rknn16
  cd ~/traffic_count
  python /mnt/d/multi-road-crossing-vehicle-count/tools/01_convert/convert_head4_probe.py

产出：
  ~/traffic_count/models/best_detrac_v12_head4.rknn  （给板子验证用）
"""
import os
import sys

ONNX = "/mnt/d/multi-road-crossing-vehicle-count/models/best_detrac_head4.onnx"
CALIB_DIR = "/mnt/d/rknn_calib"
HOME = os.path.expanduser("~")
DATASET = HOME + "/traffic_count/dataset_calib_head4.txt"
OUT = HOME + "/traffic_count/models/best_detrac_v12_head4.rknn"


def make_dataset():
    imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))
    with open(DATASET, "w", encoding="utf-8") as f:
        for n in imgs:
            f.write(os.path.join(CALIB_DIR, n) + "\n")
    print("校准清单 %d 张 -> %s" % (len(imgs), DATASET))


def main():
    import rknn
    from rknn.api import RKNN

    pkg = getattr(rknn, "__path__", [None])[0]
    ver = open(os.path.join(pkg, "VERSION")).read().strip()
    print("=" * 68)
    print("rknn-toolkit2 版本: %s" % ver)
    if ver != "1.6.0":
        print("[中止] 这个脚本必须在 rknn16 环境跑（1.6.0）")
        return 1

    if not os.path.isfile(ONNX):
        print("[中止] 找不到 %s" % ONNX)
        return 1
    print("ONNX: %s (%.2f MB)" % (ONNX, os.path.getsize(ONNX) / 1048576))
    print("=" * 68)

    make_dataset()

    rknn = RKNN(verbose=False)
    if rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                   target_platform="rk3588") != 0:
        print("[失败] config")
        return 1
    if rknn.load_onnx(model=ONNX) != 0:
        print("[失败] load_onnx")
        return 1

    print("build 中（约 1~2 分钟）...")
    if rknn.build(do_quantization=True, dataset=DATASET) != 0:
        print("[失败] build")
        return 1
    if rknn.export_rknn(OUT) != 0:
        print("[失败] export")
        return 1
    print("导出: %s (%d 字节)" % (OUT, os.path.getsize(OUT)))

    # ---- 关键一步：查两个输出的量化参数 ----
    print("\n" + "=" * 68)
    print("关键检查：输出张量的量化刻度")
    print("=" * 68)
    if rknn.init_runtime() != 0:
        print("[失败] init_runtime")
        return 1

    # 用真实的一张校准图跑一次，拿到输出数值
    import cv2
    import numpy as np

    img = cv2.imread(os.path.join(CALIB_DIR, sorted(os.listdir(CALIB_DIR))[0]))
    if img is None:
        print("[跳过] 读不到校准图")
        rknn.release()
        return 1
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outs = rknn.inference(inputs=[rgb[None]], data_format="nhwc")

    print("返回 %d 个输出" % len(outs))
    for i, o in enumerate(outs):
        a = np.asarray(o)
        print("  输出[%d] shape=%-16s min=%-8.3f max=%-8.3f mean=%.3f"
              % (i, str(a.shape), a.min(), a.max(), a.mean()))

    rknn.release()

    print("\n" + "=" * 68)
    print("下一步（在板子上跑 rknn_probe 看真实 scale）:")
    print("  cd ~/code && ./rknn_probe models/best_detrac_v12_head4.rknn data/videos/test1.mp4 200")
    print("  看 [OUT] index=0 那一行的 scale 值：")
    print("    正常: 0.0x ~ 0.1   （距离 0~16，刻度细）")
    print("    失败: 2.5 左右     （和 head3 的 boxes 一样，刻度粗）")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
