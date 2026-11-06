import cv2
import numpy as np
import os
from PIL import Image

VID_DIR = r"D:/multi-road-crossing-vehicle-count/data/videos"
OUT_DIR = r"D:/rknn_calib"
SIZE = 640
PER_VIDEO = 75

os.makedirs(OUT_DIR, exist_ok=True)


def letterbox(img, size=SIZE):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


count = 0
for name in ["test1", "test2", "test3", "test4"]:
    path = os.path.join(VID_DIR, name + ".mp4")
    if not os.path.exists(path):
        print("跳过（不存在）:", path)
        continue
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("%s: %d 帧" % (name, total))
    # 均匀抽，跳过首尾 5%
    start = int(total * 0.05)
    end = int(total * 0.95)
    step = max(1, (end - start) // PER_VIDEO)
    n = 0
    for fno in range(start, end, step):
        if n >= PER_VIDEO:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            continue
        lb = letterbox(frame)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)   # 模型吃 RGB
        out = os.path.join(OUT_DIR, "%s_%04d.jpg" % (name, fno))
        Image.fromarray(rgb).save(out, quality=95)   # PIL 存，保持 RGB
        n += 1
        count += 1
    cap.release()
    print("  抽了 %d 张" % n)

print("\n总计 %d 张 -> %s" % (count, OUT_DIR))
