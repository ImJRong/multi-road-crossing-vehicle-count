#!/usr/bin/env python3
"""统计检测框的分数分布，用来给每个视频单独定 conf。

背景：板端配置里 conf 卡在 0.3，但跟踪器的分界线是 high=0.25 / low=0.1。
conf >= 0.25 时低分那一堆永远是空的，ByteTrack 的第二阶段全程空转。
要把 conf 降下来又怕误检，就得先知道"分数到底分布在哪"。

做法：用 FP32 的 .pt 模型（真值，不含量化误差），把 conf 放到最低 0.001
把所有候选框都捞出来，再统计：
  1. 每个分数区间有多少框
  2. 每个候选 conf 阈值下，平均每帧剩几个框
  3. 每个视频"真车的分数"落在什么范围

怎么判读"哪些区间是真车"：
  低于 0.3 的框里既有被遮挡的真车，也有栅栏/摩托车/广告牌这类假框。
  所以脚本末尾会存一组对比图（每个视频 3 帧）：
    绿框 = 分数 >= conf_hi（默认 0.3，板端当前用的）
    红框 = 分数在 conf_lo ~ conf_hi 之间（默认 0.1 ~ 0.3，板端现在全扔掉的）
  统计红框中的真车数量，可判断 conf 是否还有下调空间。

用法（在项目根目录）：
  D:\\miniconda3\\envs\\yolo\\python.exe tools/02_diagnose/score_dist.py
  D:\\miniconda3\\envs\\yolo\\python.exe tools/02_diagnose/score_dist.py --videos test1 test3
  D:\\miniconda3\\envs\\yolo\\python.exe tools/02_diagnose/score_dist.py --stride 5 --no-figs

输出：
  output/score_dist_<视频>.png       分数直方图（每个视频一张）
  output/score_dist_summary.txt      全部视频的对比表
  output/lowscores_<视频>_f<帧号>.jpg 低分框可视化
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "models", "best_detrac.pt")
VIDEO_DIR = os.path.join(ROOT, "data", "videos")
OUT_DIR = os.path.join(ROOT, "output")

# 分数区间，用于直方图。下限 0.001 是 mAP 评测用的，尽量不丢框
BINS = [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]

# 候选阈值，用来算"降到这个 conf 每帧会剩几个框"
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def letterbox(frame, size=640, pad=114):
    """和板端 detector.cpp 一致的补边方式：等比缩放 + 灰色填充。"""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def run_video(model, name, stride, conf_lo, iou, conf_hi, make_figs):
    path = os.path.join(VIDEO_DIR, name + ".mp4")
    if not os.path.isfile(path):
        print("[跳过] 没有这个视频: %s" % path)
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("[跳过] 打不开: %s" % path)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("\n=== %s === %dx%d, %.1f fps, %d 帧, 每 %d 帧取 1 帧"
          % (name, w, h, fps, total, stride))

    all_scores = []          # 所有采样帧的全部框分数
    per_frame_counts = {}    # 阈值 -> 该视频累计框数
    sampled = 0
    fig_frames = []          # 留着画对比图用的原始帧

    # 每隔 stride 帧取一帧；另外固定取 3 个位置画图
    fig_targets = {int(total * r) for r in (0.15, 0.45, 0.75)}

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        want_stat = (index % stride == 0)
        want_fig = make_figs and index in fig_targets
        if not (want_stat or want_fig):
            index += 1
            continue

        canvas, scale, dx, dy = letterbox(frame)
        res = model(canvas, conf=conf_lo, iou=iou, verbose=False)[0]

        boxes = res.boxes
        scores = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))

        if want_stat:
            sampled += 1
            all_scores.append(scores)
            for t in THRESHOLDS:
                per_frame_counts[t] = per_frame_counts.get(t, 0) + int((scores >= t).sum())

        if want_fig:
            fig_frames.append((index, frame.copy(), scores.copy(), xyxy.copy(),
                               scale, dx, dy))

        index += 1

    cap.release()

    if not all_scores:
        print("没采到任何帧")
        return None

    flat = np.concatenate(all_scores) if all_scores else np.zeros(0)
    print("采样 %d 帧，共 %d 个框（NMS 后，conf>=%.3f, iou=%.2f）"
          % (sampled, flat.size, conf_lo, iou))

    if flat.size == 0:
        return None

    hist, _ = np.histogram(flat, bins=BINS)
    print("\n分数区间           框数    占比")
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        n = hist[i]
        if n == 0:
            continue
        print("  %.3f ~ %.3f   %6d   %5.1f%%" % (lo, min(hi, 1.0), n, 100.0 * n / flat.size))

    print("\n分位数：")
    for q in (10, 25, 50, 75, 90, 95, 99):
        print("  P%-3d = %.3f" % (q, np.percentile(flat, q)))

    print("\n候选 conf 阈值 -> 平均每帧框数（相对 0.05 保留比例）")
    base = per_frame_counts[THRESHOLDS[0]] / max(sampled, 1)
    rows = []
    for t in THRESHOLDS:
        avg = per_frame_counts[t] / max(sampled, 1)
        keep = 100.0 * avg / base if base > 0 else 0.0
        flag = "  <- 板端当前" if abs(t - 0.30) < 1e-6 else ""
        print("  conf>=%.2f   %6.2f 框/帧   %5.1f%%%s" % (t, avg, keep, flag))
        rows.append((t, avg, keep))

    # 低分框可视化：红 = 板端现在扔掉的，绿 = 板端保留的
    if make_figs:
        for (fi, frame, sc, xyxy, scale, dx, dy) in fig_frames:
            vis = frame.copy()
            n_low = n_high = 0
            for s, bb in zip(sc, xyxy):
                # canvas 坐标 -> 原图坐标
                x1 = (bb[0] - dx) / scale
                y1 = (bb[1] - dy) / scale
                x2 = (bb[2] - dx) / scale
                y2 = (bb[3] - dy) / scale
                if s >= conf_hi:
                    color, n_high = (0, 220, 0), n_high + 1
                elif s >= conf_lo:
                    color, n_low = (0, 0, 255), n_low + 1
                else:
                    continue
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(vis, "%.2f" % s, (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            out = os.path.join(OUT_DIR, "lowscores_%s_f%04d.jpg" % (name, fi))
            cv2.imwrite(out, vis)
            print("  图: %s  (绿 %d 个 >=%.2f, 红 %d 个 %.2f~%.2f)"
                  % (os.path.basename(out), n_high, conf_hi, n_low, conf_lo, conf_hi))

    return {
        "name": name,
        "sampled": sampled,
        "boxes": int(flat.size),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "rows": rows,
    }


def draw_hist(name, all_scores, out_path):
    """画直方图。柱子上标数量，横轴是分数。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    counts, _ = np.histogram(all_scores, bins=BINS)
    labels = ["%.2f-%.2f" % (BINS[i], min(BINS[i + 1], 1.0)) for i in range(len(BINS) - 1)]
    total = max(counts.sum(), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#c0392b" if BINS[i + 1] <= 0.30 else "#27ae60" for i in range(len(counts))]
    bars = ax.bar(range(len(counts)), counts, color=colors)

    for b, c in zip(bars, counts):
        if c > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    "%d\n%.0f%%" % (c, 100.0 * c / total),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("检测框分数")
    ax.set_ylabel("框数")
    ax.set_title("%s 检测框分数分布（红=低于 0.30，板端现在全扔掉）" % name)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["test1", "test2", "test3", "test4"])
    ap.add_argument("--stride", type=int, default=3, help="每 N 帧统计一次，1=全部")
    ap.add_argument("--conf-lo", type=float, default=0.001, help="统计下限，越低越完整")
    ap.add_argument("--conf-hi", type=float, default=0.30, help="板端当前用的 conf，画图分色用")
    ap.add_argument("--iou", type=float, default=0.3, help="NMS 的 IoU，与配置保持一致")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(MODEL):
        print("找不到模型:", MODEL)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(MODEL)
    print("模型:", MODEL)

    results = []
    for name in args.videos:
        r = run_video(model, name, args.stride, args.conf_lo, args.iou,
                      args.conf_hi, not args.no_figs)
        if r:
            results.append(r)

    if not results:
        print("\n没有任何结果")
        return 1

    # 汇总表
    lines = []
    lines.append("检测框分数分布汇总（FP32 .pt 模型，conf>=%.3f, iou=%.2f）"
                 % (args.conf_lo, args.iou))
    lines.append("")
    head = "%-8s %6s %8s %7s %7s" % ("视频", "帧数", "框数", "P50", "P90")
    for t in THRESHOLDS:
        head += " %7s" % ("%.2f" % t)
    lines.append(head + "   (每列=该 conf 下平均每帧框数)")
    for r in results:
        row = "%-8s %6d %8d %7.3f %7.3f" % (r["name"], r["sampled"], r["boxes"],
                                            r["p50"], r["p90"])
        for (_, avg, _) in r["rows"]:
            row += " %7.2f" % avg
        lines.append(row)

    text = "\n".join(lines)
    print("\n" + text)

    out_txt = os.path.join(OUT_DIR, "score_dist_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n汇总表: %s" % out_txt)

    if not args.no_figs:
        for r in results:
            p = os.path.join(OUT_DIR, "score_dist_%s.png" % r["name"])
            print("直方图: %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
