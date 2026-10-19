# 剪视频：把一段视频从第几秒剪到第几秒，去掉头尾没用的部分。
#
# 用法（在项目根目录）：
#   python tools/cut_video.py <输入视频> <起点秒> <终点秒>
#   python tools/cut_video.py <输入视频> <起点秒> <终点秒> --out 输出路径
#   python tools/cut_video.py <输入视频> <起点秒> <终点秒> --out data/videos/test4.mp4 --replace
#
# 说明：
#   - 终点秒写 0 表示"一直剪到结尾"。
#   - 不加 --out 时，输出成 <输入名>_cut.mp4，放在输入视频旁边。
#   - 加 --replace 才会覆盖 --out 指向的已有文件（默认不覆盖，防止误删）。
#   - 用 OpenCV 重新编码，画面尺寸和帧率不变，但【没有声音】。
#
# 例：
#   python tools/cut_video.py "C:/Users/RONGG/Desktop/test4.mp4" 5 640 --out data/videos/test4.mp4 --replace

import os
import sys

import cv2


def parse_args(argv):
    if len(argv) < 3:
        print("用法：python tools/cut_video.py <输入视频> <起点秒> <终点秒> [--out 路径] [--replace]")
        print("例：  python tools/cut_video.py data/videos/test4.mp4 5 640 --out data/videos/test4_new.mp4")
        return None

    video = argv[0]
    start = float(argv[1])
    end = float(argv[2])

    out = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            raise SystemExit("--out 后面要跟输出路径")
        out = argv[i + 1]

    if out is None:
        base, ext = os.path.splitext(video)
        out = base + "_cut" + (ext or ".mp4")

    return video, start, end, out, ("--replace" in argv)


def main():
    parsed = parse_args(sys.argv[1:])
    if parsed is None:
        return
    video, start, end, out, replace = parsed

    if not os.path.exists(video):
        raise SystemExit("找不到输入视频：%s" % video)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    if end <= 0 or end > duration:
        end = duration

    f_start = int(start * fps)
    f_end = int(end * fps)
    if f_start < 0:
        f_start = 0
    if f_end > total:
        f_end = total
    if f_end <= f_start:
        cap.release()
        raise SystemExit("终点必须大于起点。这段视频一共 %.1f 秒。" % duration)

    if os.path.exists(out) and not replace:
        cap.release()
        raise SystemExit(
            "输出文件已存在：%s\n要覆盖它，请加 --replace。" % out)

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("输入：%s" % video)
    print("  尺寸 %dx%d，fps %.2f，共 %d 帧（%.1f 秒）" % (width, height, fps, total, duration))
    print("剪取：第 %.1f 秒 ~ 第 %.1f 秒  →  第 %d 帧 ~ 第 %d 帧（共 %d 帧，%.1f 秒）"
          % (start, end, f_start, f_end, f_end - f_start, (f_end - f_start) / fps))
    print("输出：%s" % out)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    n = 0
    while n < f_end - f_start:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
        if n % 300 == 0:
            print("  已写 %d / %d 帧" % (n, f_end - f_start), flush=True)

    cap.release()
    writer.release()

    if n == 0:
        raise SystemExit("一帧都没写出来，检查一下起点秒是不是超出视频长度了")

    size = os.path.getsize(out) / 1024 / 1024
    print()
    print("完成：写了 %d 帧，%.1f 秒，文件 %.1f MB" % (n, n / fps, size))
    print("→ %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()
