# 调速度：把视频放慢或加快。
#
# 原理：画面内容一帧不改，只改【每秒钟播放多少帧】。
#   例如 30fps 的视频写进 24fps 的容器，播放时间就变成原来的 30/24 = 1.25 倍（变慢）。
#
# 注意：没有声音（重新编码时丢掉了）。
#
# 用法（在项目根目录）：
#   python tools/speed_video.py <输入视频> <speed> [--out 输出路径] [--replace]
#
#   speed 是"速度倍数"：
#     0.8  → 放慢到 0.8 倍速（时长变成 1/0.8 = 1.25 倍，即慢 25%）
#     1.0  → 原速
#     2.0  → 2 倍速（时长减半）
#
# 例：
#   python tools/speed_video.py data/videos/test3.mp4 0.8 --out data/videos/test3_slow.mp4

import os
import sys

import cv2

FPS_CHOICES = [10, 12, 15, 20, 24, 25, 30, 48, 50, 60]


def parse_args(argv):
    if len(argv) < 2:
        print("用法：python tools/speed_video.py <输入视频> <speed> [--out 路径] [--replace]")
        print("例：  python tools/speed_video.py data/videos/test3.mp4 0.8 --out data/videos/test3_slow.mp4")
        return None

    video = argv[0]
    speed = float(argv[1])
    if speed <= 0:
        raise SystemExit("speed 必须大于 0")

    out = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            raise SystemExit("--out 后面要跟输出路径")
        out = argv[i + 1]

    if out is None:
        base, ext = os.path.splitext(video)
        out = "%s_%gx%s" % (base, speed, ext or ".mp4")

    return video, speed, out, ("--replace" in argv)


def pick_fps(old_fps, speed):
    """挑一个播放器普遍认识的帧率，让实际倍速尽量贴近要的 speed。"""
    want = old_fps * speed
    return min(FPS_CHOICES, key=lambda f: abs(f - want))


def main():
    parsed = parse_args(sys.argv[1:])
    if parsed is None:
        return
    video, speed, out, replace = parsed

    if not os.path.exists(video):
        raise SystemExit("找不到输入视频：%s" % video)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % video)

    old_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_fps = pick_fps(old_fps, speed)
    real = old_fps / new_fps          # 实际倍速（>1 表示变慢）

    if os.path.exists(out) and not replace:
        cap.release()
        raise SystemExit("输出文件已存在：%s\n要覆盖它，请加 --replace。" % out)

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("输入：%s" % video)
    print("  %dx%d，原帧率 %.2f，共 %d 帧（%.1f 秒）" % (width, height, old_fps, total, total / old_fps))
    print("要求倍速 %.2f → 写入帧率 %.0f，实际倍速 %.2f（时长 %.1f → %.1f 秒）"
          % (speed, new_fps, real, total / old_fps, total / new_fps))
    print("输出：%s" % out)

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), new_fps, (width, height))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
        if n % 300 == 0:
            print("  已写 %d / %d 帧" % (n, total), flush=True)

    cap.release()
    writer.release()

    print()
    print("完成：写了 %d 帧，播放时长 %.1f 秒，文件 %.1f MB"
          % (n, n / new_fps, os.path.getsize(out) / 1024 / 1024))
    print("→ %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()
