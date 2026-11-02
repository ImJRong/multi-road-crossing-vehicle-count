# 预标注：用现在的模型先自动标一遍，生成 YOLO 格式的 txt，你只改它标错的地方。
#
# 为什么这么做：从零标一张路口图要半分钟以上，改一个已经标好的框只要几秒。
# 哪怕模型只标对一半，也能省掉大半体力。
#
# 类别只有 3 个，映射关系（官方模型的编号 -> 我们训练的编号）：
#   car  2 -> 0
#   bus  5 -> 1
#   truck 7 -> 2
#
# 用法（在项目根目录）：
#   python tools/prelabel.py
#   python tools/prelabel.py 0.35        第二个参数=分数门槛，默认 0.3
# 输入：data/dataset/images/
# 输出：data/dataset/labels/   （同名 .txt）
#       data/dataset/classes.txt

import glob
import os
import cv2
from ultralytics import YOLO

# ============ 配置区 ============
MODEL = "models/yolov8n.pt"            # 先拿官方模型顶着，以后换成自己训的
IMAGE_DIR = "data/dataset/images"      # 图片放哪
LABEL_DIR = "data/dataset/labels"      # 标注写哪
CLASSES_FILE = "data/dataset/classes.txt"
CONF = 0.3                             # 分数门槛

# 官方 80 类 -> 我们的 3 类
MAP = {2: 0, 5: 1, 7: 2}
NAMES = ["car", "bus", "truck"]
# ===============================


def main(conf=CONF):
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not images:
        raise SystemExit("没找到图片，先跑 tools/extract_frames.py")

    os.makedirs(LABEL_DIR, exist_ok=True)
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(NAMES) + "\n")

    model = YOLO(MODEL)
    print("共 %d 张图，分数门槛 %.2f" % (len(images), conf))

    total_box = 0
    empty = 0

    for i, path in enumerate(images, 1):
        img = cv2.imread(path)
        h, w = img.shape[:2]

        r = model(img, verbose=False, conf=conf)[0]

        lines = []
        for box in r.boxes:
            old = int(box.cls)
            if old not in MAP:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # YOLO 要的是"归一化后的中心点 + 宽高"，不是左上右下角
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append("%d %.6f %.6f %.6f %.6f" % (MAP[old], cx, cy, bw, bh))

        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(LABEL_DIR, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        total_box += len(lines)
        if not lines:
            empty += 1
            # 一个框都没有的图也留个空 txt，这样训练时不会把这张图当"漏掉了"
        print("\r已处理 %d/%d 张，累计 %d 个框" % (i, len(images), total_box), end="")

    print()
    print("共 %d 个框，其中 %d 张图一个框都没标出来（这些要你手工补）" % (total_box, empty))
    print("标注 → %s" % LABEL_DIR)
    print("类别表 → %s" % CLASSES_FILE)
    print()
    print("下一步：拿标注工具打开 %s，对着图把错的框删掉、漏的补上。" % IMAGE_DIR)


if __name__ == "__main__":
    import sys
    main(float(sys.argv[1]) if len(sys.argv) > 1 else CONF)
