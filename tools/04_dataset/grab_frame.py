# 取帧：从视频里导出一张原图，拿去画图工具上画计数线。
#
# 用法（在项目根目录）：
#   python tools/grab_frame.py data/videos/test1.mp4
#   python tools/grab_frame.py data/videos/test1.mp4 500    第二个参数=要第几帧，默认第 1 帧
# 输出：data/samples/<视频名>_frame<帧号>.jpg

import os
import sys
import cv2

OUT_DIR = "data/samples"


def main():
    if len(sys.argv) < 2:
        print("用法：python tools/grab_frame.py <视频路径> [第几帧]")
        return

    video = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % video)

    frame = None
    for i in range(1, want + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise SystemExit("视频没这么长，读到第 %d 帧就没了" % i)
    cap.release()

    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(video))[0]
    path = os.path.join(OUT_DIR, "%s_frame%d.jpg" % (name, want))
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    h, w = frame.shape[:2]
    print("导出了第 %d 帧，尺寸 %dx%d" % (want, w, h))
    print("→ %s" % path)
    print()
    print("接下来：用画图工具打开它，画一条线，另存成另一个名字（别覆盖原图），")
    print("再用 tools/read_line.py 读出坐标。")


if __name__ == "__main__":
    main()
