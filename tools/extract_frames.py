# 抽帧：把视频里的画面抽成图片，作为标注用的原料。
#
# 为什么要抽帧而不是每帧都用：同一个摄像头画面高度重复，相邻帧几乎一样，
# 抽太密等于标同一张图好几遍，纯浪费人力。所以宁可多个机位各抽几张。
#
# 三种用法（在项目根目录跑）：
#
#   1. 每隔几秒抽一张（默认，适合自己拍的视频）
#      python tools/extract_frames.py data/videos/test1.mp4 5
#
#   2. 整个视频均匀抽 N 张（适合一整个数据集批量处理）
#      python tools/extract_frames.py data/videos/test1.mp4 count=15
#
#   3. 指定输出目录（不指定就写到 data/dataset/images）
#      python tools/extract_frames.py data/videos/test1.mp4 count=15 --out data/aicity/images
#
# 文件名格式：<前缀>_<视频名>_<帧号>.jpg
#   例：aic_cam_1_000123.jpg
#   这样 split_dataset.py 能靠文件名认出"这是哪个机位"，划分时同机位的图不会被拆散。

import os
import sys
import cv2

# ============ 配置区 ============
OUT_DIR = "data/dataset/images"        # 默认输出目录
DEFAULT_STEP = 2.0                     # 默认每隔几秒抽一张
PREFIX = ""                            # 文件名前缀，用 --prefix 改
# ===============================


def parse_args(argv):
    """手动解析：位置参数是 视频路径 和 抽帧方式，其余用 --xxx 指定。"""
    opts = {"video": None, "step": None, "count": None,
            "out": OUT_DIR, "prefix": PREFIX}
    for a in argv:
        if a.startswith("--out="):
            opts["out"] = a.split("=", 1)[1]
        elif a.startswith("--prefix="):
            opts["prefix"] = a.split("=", 1)[1]
        elif a.startswith("count="):
            opts["count"] = int(a.split("=", 1)[1])
        elif opts["video"] is None:
            opts["video"] = a
        elif opts["step"] is None:
            opts["step"] = float(a)
    if opts["step"] is None:
        opts["step"] = DEFAULT_STEP
    return opts


def main():
    opts = parse_args(sys.argv[1:])
    video = opts["video"]
    if not video:
        print("用法：")
        print("  python tools/extract_frames.py <视频> [每隔几秒] [--out 目录] [--prefix 前缀]")
        print("  python tools/extract_frames.py <视频> count=15 [--out 目录] [--prefix 前缀]")
        return

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = os.path.splitext(os.path.basename(video))[0]
    os.makedirs(opts["out"], exist_ok=True)

    if opts["count"]:
        # 均匀抽 N 张：算好目标帧号，直接跳过去读。
        # 大视频里逐帧读一遍太慢，跳着读快得多。
        n = min(opts["count"], total)
        targets = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
        saved = 0
        for t in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, frame = cap.read()
            if not ok:
                continue
            out = os.path.join(opts["out"], "%s%s_%06d.jpg" % (opts["prefix"], name, t + 1))
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        mode = "均匀抽 %d 张" % n
    else:
        # 每隔几秒抽一张
        step = max(1, int(round(fps * opts["step"])))
        saved = 0
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if (frame_no - 1) % step != 0:
                continue
            out = os.path.join(opts["out"], "%s%s_%06d.jpg" % (opts["prefix"], name, frame_no))
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        mode = "每隔 %.1f 秒抽一张" % opts["step"]

    cap.release()
    print("%-22s %dx%d %.0f帧/秒 %d帧  ->  %s，存了 %d 张"
          % (name, w, h, fps, total, mode, saved))


if __name__ == "__main__":
    main()
