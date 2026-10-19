# 整理 UA-DETRAC：从 60 个摄像头里均匀抽一批图，转成 YOLO 格式。
#
# 为什么要"均匀抽"：UA-DETRAC 一共 8.4 万张。全用的话训练要跑上百小时，
#   Kaggle 一周只给 30 小时，跑不完。抽 1/10 就够，但不能瞎抽——
#   要保证 60 个摄像头都有份，不然多样性丢了，模型见过的场景反而变窄。
#
# 类别：只有一类 vehicle（就是"车"），不做轿车/公交/卡车的区分。
#   标注里还有个 ignored 类，是"忽略区域"（天空、被切掉一半的车之类），训练时丢掉。
#
# 用法（在项目根目录）：
#   python tools/prepare_detrac.py
#   python tools/prepare_detrac.py 200        第二个参数=每个摄像头抽多少张
# 输入：DETRAC_DIR 指向的 UA-DETRAC 训练集
# 输出：data/dataset/images/、data/dataset/labels/、data/dataset/classes.txt

import csv
import os
import shutil
import sys

# ============ 配置区 ============
DETRAC_DIR = r"C:\Users\RONGG\Desktop\ua_detrac_training_set"
OUT_IMAGE = "data/dataset/images"
OUT_LABEL = "data/dataset/labels"
PER_CAM = 130          # 每个摄像头抽多少张。60 个 × 130 ≈ 7800 张
NAMES = ["vehicle"]    # 就一类
# ===============================


def pick_images(files, k):
    """从排好序的图片列表里，等间隔地挑 k 张。这样早中晚的画面都能覆盖到。"""
    n = len(files)
    if n <= k:
        return files
    step = n / k
    return [files[int(i * step)] for i in range(k)]


def convert_one(cam_dir, cam_name, want):
    """处理一个摄像头文件夹，返回 (抽了几张, 写了几个框, 跳过几张没车的)。"""
    jpgs = sorted(f for f in os.listdir(cam_dir) if f.lower().endswith(".jpg"))
    csv_path = os.path.join(cam_dir, "annotations.csv")
    if not os.path.exists(csv_path):
        return 0, 0, 0

    # 先把标注按文件名归拢：每张图 -> (宽, 高, [(xmin,ymin,xmax,ymax), ...])
    boxes = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            # 列：空,filename,width,height,class,xmin,ymin,xmax,ymax,confidence,target_id
            if len(row) < 9 or row[4] != "vehicle":
                continue                      # 表头行 和 ignored 都跳过
            try:
                w, h = float(row[2]), float(row[3])
                rec = (float(row[5]), float(row[6]), float(row[7]), float(row[8]))
            except ValueError:
                continue
            boxes.setdefault(row[1], (w, h, []))[2].append(rec)

    picked = pick_images(jpgs, want)
    n_img = n_box = n_skip = 0

    for jpg in picked:
        if jpg not in boxes:
            n_skip += 1                       # 这张图没车，不要
            continue
        w, h, recs = boxes[jpg]

        stem = "%s_%s" % (cam_name, os.path.splitext(jpg)[0])   # 例：detrac10_image000001
        shutil.copyfile(os.path.join(cam_dir, jpg),
                        os.path.join(OUT_IMAGE, stem + ".jpg"))

        lines = []
        for xmin, ymin, xmax, ymax in recs:
            cx = (xmin + xmax) / 2 / w
            cy = (ymin + ymax) / 2 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            # 夹到 0~1，防止个别越界的框把训练搞崩
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
            lines.append("0 %.6f %.6f %.6f %.6f" % (cx, cy, bw, bh))

        with open(os.path.join(OUT_LABEL, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        n_img += 1
        n_box += len(lines)

    return n_img, n_box, n_skip


def main(per_cam=PER_CAM):
    if not os.path.isdir(DETRAC_DIR):
        raise SystemExit("找不到 UA-DETRAC：%s" % DETRAC_DIR)

    os.makedirs(OUT_IMAGE, exist_ok=True)
    os.makedirs(OUT_LABEL, exist_ok=True)
    with open("data/dataset/classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(NAMES) + "\n")

    cams = sorted(d for d in os.listdir(DETRAC_DIR)
                  if os.path.isdir(os.path.join(DETRAC_DIR, d)))
    print("找到 %d 个摄像头，每个抽 %d 张" % (len(cams), per_cam))

    total_img = total_box = total_skip = 0
    for i, cam in enumerate(cams, 1):
        # detrac_10_MVI_20062 -> detrac10
        tag = "".join(cam.split("_")[:2])
        ni, nb, ns = convert_one(os.path.join(DETRAC_DIR, cam), tag, per_cam)
        total_img += ni
        total_box += nb
        total_skip += ns
        print("\r  [%d/%d] %-24s 已累计 %d 张 / %d 个框" % (i, len(cams), cam, total_img, total_box), end="")

    print()
    print()
    print("抽出 %d 张图，%d 个框，平均每张 %.1f 个框" % (total_img, total_box, total_box / max(total_img, 1)))
    print("跳过 %d 张（那些图里一辆车都没有）" % total_skip)
    print()
    print("→ %s" % OUT_IMAGE)
    print("→ %s" % OUT_LABEL)
    print()
    print("下一步：python tools/split_dataset.py   （按摄像头分训练集/验证集）")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else PER_CAM)
