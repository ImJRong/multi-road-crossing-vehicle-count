#!/usr/bin/env python3
"""查 A：为什么 2.5.0 转出来的模型，w/h 会被放大 13 倍。

已知线索（都是实测的）：
  v7（1.6.0 转）  : w = 20.0  h = 20.0     框正常
  v10（2.5.0 转） : w = 235.2 h = 260.2    放大 13 倍
  v10 的 4 个通道最大值都是 637.92 = 255 × 2.5016（int8 饱和值）
  v10 的 float 输出完全等于昨天那个 int8_normal.rknn（逐字节一样）

「单纯差 0.3 格」只能解释 2.4 像素的误差，解释不了放大 13 倍。
所以一定是某一步的语义被破坏了。

做法：
  用 toolkit 的 accuracy_analysis 把两个模型逐层 dump，
  算每层和 FP32 golden 的余弦距离，找第一个误差突然变大的层。
  ——哪里先崩，问题就在哪。

跑法（分别在两个环境跑，对比结果）：
  conda activate rknn16  && python convert_trace.py 1.6.0
  conda activate rknn    && python convert_trace.py 2.5.0

产出：~/traffic_count/trace_<版本>/error_analysis.txt + 打印对比
"""
import os
import sys

ROOT = "/mnt/d/multi-road-crossing-vehicle-count"
ONNX_HEAD3 = ROOT + "/models/best_detrac_head3.onnx"
CALIB_DIR = "/mnt/d/rknn_calib"
HOME = os.path.expanduser("~")
PROJ = HOME + "/traffic_count"


def make_dataset(n=20):
    """accuracy_analysis 用不着 300 张，20 张够看趋势，也快。"""
    imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))[:n]
    path = os.path.join(PROJ, "dataset_trace_%d.txt" % n)
    with open(path, "w", encoding="utf-8") as f:
        for name in imgs:
            f.write(os.path.join(CALIB_DIR, name) + "\n")
    return path, [os.path.join(CALIB_DIR, x) for x in imgs[:2]]


def main():
    import rknn
    from rknn.api import RKNN

    pkg = getattr(rknn, "__path__", [None])[0]
    ver = open(os.path.join(pkg, "VERSION")).read().strip()
    tag = sys.argv[1] if len(sys.argv) > 1 else ver
    outdir = os.path.join(PROJ, "trace_%s" % tag.replace(".", "_"))

    print("=" * 70)
    print("rknn-toolkit2 版本:", ver)
    print("dump 目录:", outdir)
    print("=" * 70)

    dataset, probe_imgs = make_dataset(20)

    r = RKNN(verbose=False)
    cfg = dict(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
               target_platform="rk3588")
    # 2.5.0 必须加 disable_rules，不加会在 _p_seperate_conv_rs_split 崩；
    # 1.6.0 不认这个参数（会报 not supported），所以先试带、失败再退回。
    configured = False
    try:
        if r.config(disable_rules=["seperate_conv_rs_split"], **cfg) == 0:
            configured = True
    except (ValueError, TypeError):
        pass
    if not configured:
        r2 = RKNN(verbose=False)
        if r2.config(**cfg) != 0:
            raise SystemExit("config 失败")
        r = r2
        print("（这个版本不吃 disable_rules，用默认参数）")

    if r.load_onnx(model=ONNX_HEAD3) != 0:
        raise SystemExit("load_onnx 失败")
    if r.build(do_quantization=True, dataset=dataset) != 0:
        raise SystemExit("build 失败")

    # ★ 关键：逐层 dump，和 FP32 golden 对比
    print("\n开始 accuracy_analysis（逐层量化误差）...")
    ret = r.accuracy_analysis(inputs=probe_imgs, output_dir=outdir)
    if ret != 0:
        print("accuracy_analysis 失败，返回", ret)
    else:
        print("完成 ->", outdir)

    r.release()

    # 打印误差最大的那些层
    txt = os.path.join(outdir, "error_analysis.txt")
    if not os.path.isfile(txt):
        print("\n没找到 error_analysis.txt")
        return 1

    print("\n" + "=" * 70)
    print("误差最大的层（按整个网络的余弦距离排，越小越像 FP32）")
    print("=" * 70)

    rows = []
    with open(txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # 每行形如: [Op] name   1.00000 | 0.0   1.00000 | 0.0
            parts = line.replace("|", " ").split()
            if len(parts) < 4:
                continue
            try:
                vals = [float(x) for x in parts if _is_num(x)]
            except ValueError:
                continue
            if len(vals) < 2:
                continue
            cos = vals[0]
            name = " ".join(p for p in parts if not _is_num(p))
            if name and cos < 0.999:
                rows.append((cos, name.strip(), line.rstrip()))

    rows.sort()
    if not rows:
        print("所有层的余弦距离都 >= 0.999（没有明显崩坏的层）")
    for cos, name, raw in rows[:25]:
        print("  cos=%.6f  %s" % (cos, raw[:150]))

    print("\n完整文件:", txt)
    print("重点看：dfl 相关的层、Mul_2、Sigmoid 这些")
    return 0


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    sys.exit(main())
