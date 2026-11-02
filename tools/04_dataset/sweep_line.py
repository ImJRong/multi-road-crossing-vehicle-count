# 选线工具：扫一遍视频，看不同高度的横线各会有多少辆车穿过，用来给配置挑计数线的 y。
#
# 换视频、换机位都要重新扫一次。挑"穿过次数多、又不在车排队等待那一带"的高度。
#
# 用法（在项目根目录）：
#   python tools/sweep_line.py data/videos/test2.mp4
#   python tools/sweep_line.py data/videos/test2.mp4 600     第二个参数是跑多少帧

import sys
import cv2
from ultralytics import YOLO

VEHICLE = [2, 3, 5, 7]
CLASSES = [0, 1, 2, 3, 5, 7]


def collect(video, n_frames):
    """跑一遍跟踪，记下每个 ID 每帧的中心点。"""
    model = YOLO("models/yolov8n.pt")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % video)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracks = {}
    idx = 0
    while idx < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        idx += 1
        r = model.track(f, persist=True, verbose=False, classes=CLASSES,
                        conf=0.3, tracker="bytetrack.yaml")[0]
        if r.boxes.id is None:
            continue
        for b in r.boxes:
            if int(b.cls) not in VEHICLE:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            tracks.setdefault(int(b.id), []).append(((x1 + x2) / 2, (y1 + y2) / 2))
    cap.release()
    return tracks, idx, w, h, fps


def crossings(tracks, y, margin=5):
    """这条高度上穿过了多少次，顺便把穿线时的 x 记下来。"""
    n = 0
    xs = []
    for pts in tracks.values():
        prev = None
        for cx, cy in pts:
            s = cy - y
            if abs(s) <= margin:
                continue
            if prev is not None and (prev < 0) != (s < 0):
                n += 1
                xs.append(cx)
            prev = s
    return n, xs


def main():
    if len(sys.argv) < 2:
        print("用法：python tools/sweep_line.py <视频路径> [帧数]")
        return
    video = sys.argv[1]
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    tracks, frames, w, h, fps = collect(video, n_frames)
    print("视频：%s" % video)
    print("尺寸 %dx%d  帧率 %.1f  跑了 %d 帧  出现过的车辆 ID %d 个"
          % (w, h, fps, frames, len(tracks)))
    print()
    print("  y      穿过次数   穿线 x 的范围")
    step = max(25, h // 24)
    for y in range(step, h - step // 2, step):
        n, xs = crossings(tracks, y)
        if n == 0:
            print("  %-5d  %-8d  -" % (y, n))
        else:
            print("  %-5d  %-8d  %d ~ %d" % (y, n, int(min(xs)), int(max(xs))))


if __name__ == "__main__":
    main()
