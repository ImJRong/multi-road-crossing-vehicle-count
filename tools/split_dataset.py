# 划分训练集 / 验证集。把所有图分两份，一份用来学，一份用来看学得怎么样。
#
# ★ 关键：按【摄像头】分组划分，不是按图片随机划分。
#   同一个摄像头连拍的几张图长得几乎一样。如果随机分，
#   同一辆车的两个瞬间会一张进训练、一张进验证，
#   验证分数就会虚高——模型其实是"背"下来的，不是真会了。
#   所以这里先按文件名前缀分组，同一组的图整组进同一侧。
#
# 文件名要求：<摄像头名>_<序号>.jpg
#   例：cam01_000012.jpg、cam01_000240.jpg、cam02_000015.jpg
#   cam01 的两张会一起走，不会一个进训练一个进验证。
#
# 用法（在项目根目录）：
#   python tools/split_dataset.py
#   python tools/split_dataset.py 0.2        第二个参数=验证集占多少，默认 0.2
# 输出：data/dataset/train.txt、val.txt、data.yaml

import glob
import os
import random
import re
import sys

# ============ 配置区 ============
DATA_DIR = "data/dataset"              # 数据集根目录
IMAGE_DIR = "data/dataset/images"
LABEL_DIR = "data/dataset/labels"
VAL_RATIO = 0.2                        # 验证集占多少
SEED = 0                               # 固定随机种子，每次划分结果一样，方便对比
# 类别顺序从这里读，不写死——单类（vehicle）和三类（car/bus/truck）都靠它区分
CLASSES_FILE = "data/dataset/classes.txt"
# ===============================


def group_of(name):
    """从文件名里取摄像头名。把结尾那种序号去掉：
       detrac10_image000001 -> detrac10
       cam01_000012         -> cam01
       test1_000061         -> test1

    ⚠️ 这里的正则要能兼容两种命名。写成只认 '_数字' 的话，
       UA-DETRAC 的 '_image000001' 就匹配不上，会退化成"一张图一个组"，
       同机位的图就漏到两边去了，验证分数会虚高。
    """
    m = re.match(r"^(.*?)_[A-Za-z]*\d+$", name)
    return m.group(1) if m else name


def main(val_ratio=VAL_RATIO):
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not images:
        raise SystemExit("没找到图片：%s 是空的" % IMAGE_DIR)

    # 类别表由 prepare_detrac.py / prelabel.py 生成，这里只读不写
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, encoding="utf-8") as f:
            class_names = [ln.strip() for ln in f if ln.strip()]
    else:
        raise SystemExit("找不到类别表 %s，先跑 tools/prepare_detrac.py" % CLASSES_FILE)

    # 按摄像头分组
    groups = {}
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        groups.setdefault(group_of(stem), []).append(path)

    print("共 %d 张图，来自 %d 个摄像头/场景" % (len(images), len(groups)))
    if len(groups) < 2:
        print()
        print("⚠️ 只有一个摄像头。这样没法按摄像头分家，验证分数会虚高。")
        print("   建议再收集几个不同机位的素材。")

    # 检查每张图有没有对应的标注文件
    missing = [p for p in images
               if not os.path.exists(os.path.join(
                   LABEL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt"))]
    if missing:
        print()
        print("⚠️ 有 %d 张图没有对应的标注文件，它们会被跳过。前几个：" % len(missing))
        for p in missing[:5]:
            print("   ", p)

    names = sorted(groups)
    random.seed(SEED)
    random.shuffle(names)

    # 按摄像头切，凑到验证集比例为止（至少留一个摄像头给验证集）
    n_val = max(1, int(round(len(names) * val_ratio)))
    n_val = min(n_val, len(names) - 1) if len(names) > 1 else 0
    val_groups = set(names[:n_val])

    train_paths, val_paths = [], []
    for g, paths in groups.items():
        (val_paths if g in val_groups else train_paths).extend(paths)

    train_paths.sort()
    val_paths.sort()

    def write_list(paths, out):
        # ⚠️ 必须写绝对路径。
        # ultralytics 找标注的办法是把路径里的 "images" 换成 "labels"，
        # 它要求这个 "images" 两边有路径分隔符。写成相对路径 "images/xxx.jpg"
        # 的时候前面没有分隔符，换不成，就会报 "No labels found"。
        with open(out, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(os.path.abspath(p).replace("\\", "/") + "\n")

    write_list(train_paths, os.path.join(DATA_DIR, "train.txt"))
    write_list(val_paths, os.path.join(DATA_DIR, "val.txt"))

    yaml_path = os.path.join(DATA_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % os.path.abspath(DATA_DIR).replace("\\", "/"))
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("nc: %d\n" % len(class_names))
        f.write("names: %s\n" % class_names)

    print()
    print("训练集 %d 张，来自 %d 个摄像头" % (len(train_paths), len(groups) - len(val_groups)))
    print("验证集 %d 张，来自 %d 个摄像头" % (len(val_paths), len(val_groups)))
    print()
    if val_groups:
        shown = sorted(val_groups)
        if len(shown) > 12:
            print("划到验证集的摄像头（共 %d 个）：%s ..."
                  % (len(shown), ", ".join(shown[:12])))
        else:
            print("划到验证集的摄像头：%s" % ", ".join(shown))
    print()
    print("→ %s" % os.path.join(DATA_DIR, "train.txt"))
    print("→ %s" % os.path.join(DATA_DIR, "val.txt"))
    print("→ %s" % yaml_path)
    print()
    print("下一步：python src/07_train.py")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else VAL_RATIO)
