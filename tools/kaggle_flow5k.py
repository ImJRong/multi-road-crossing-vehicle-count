# ===========================================================================
# Intersection-Flow-5K → 单类 vehicle，然后在 Kaggle 上接着 best_detrac.pt 训
#
# 用法：每一段单独贴进一个 Kaggle 代码格，按顺序跑，跑完一段再跑下一段。
# ===========================================================================


# ============ 第 3 段：转换成单类 ============
import os, shutil
from collections import Counter

ROOT = '/kaggle/input/datasets/starsw/intersection-flow-5k/Intersection-Flow-5K'
WORK = '/kaggle/working/flow5k'

SRC_NAMES = ['vehicle', 'bus', 'bicycle', 'pedestrian',
             'engine', 'truck', 'tricycle', 'obstacle']
KEEP = {0, 1, 5}                 # 只留 车 / 公交 / 卡车
CLASS_MAP = {i: 0 for i in KEEP} # 全部映射成 0 = vehicle

if os.path.exists(WORK):
    shutil.rmtree(WORK)

kept = Counter()
dropped = Counter()

for split in ['train', 'val', 'test']:
    src_img = os.path.join(ROOT, 'images', split)
    src_lab = os.path.join(ROOT, 'labels', split)
    dst_img = os.path.join(WORK, 'images', split)
    dst_lab = os.path.join(WORK, 'labels', split)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lab, exist_ok=True)

    n_keep = n_skip = 0
    for name in sorted(os.listdir(src_lab)):
        if not name.endswith('.txt'):
            continue
        stem = name[:-4]

        img_src = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            p = os.path.join(src_img, stem + ext)
            if os.path.exists(p):
                img_src = p
                break
        if img_src is None:
            continue

        out_lines = []
        for line in open(os.path.join(src_lab, name)):
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(float(p[0]))
            if cid in CLASS_MAP:
                out_lines.append('%d %s\n' % (CLASS_MAP[cid], ' '.join(p[1:5])))
                kept[SRC_NAMES[cid]] += 1
            else:
                dropped[SRC_NAMES[cid]] += 1

        if not out_lines:          # 整张图都没有要的车，跳过
            n_skip += 1
            continue

        shutil.copy(img_src, os.path.join(dst_img, os.path.basename(img_src)))
        with open(os.path.join(dst_lab, name), 'w') as f:
            f.writelines(out_lines)
        n_keep += 1

    print('%-6s 保留 %5d 张，跳过 %4d 张（没有车）' % (split, n_keep, n_skip))

print()
print('保留的框:')
for k, v in kept.most_common():
    print('  %-10s %6d' % (k, v))
print('丢掉的框:')
for k, v in dropped.most_common():
    print('  %-10s %6d' % (k, v))

# ============ 第 4 段：重排成 YOLO 要的结构，生成 data.yaml ============
import os, glob, shutil, random

WORK = '/kaggle/working/flow5k'
WORK2 = '/kaggle/working/dataset_flow5k'

if os.path.exists(WORK2):
    shutil.rmtree(WORK2)
os.makedirs(os.path.join(WORK2, 'images'), exist_ok=True)
os.makedirs(os.path.join(WORK2, 'labels'), exist_ok=True)

for split in ['train', 'val', 'test']:
    d = os.path.join(WORK, 'images', split)
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        shutil.copy(os.path.join(d, name), os.path.join(WORK2, 'images', name))
        stem = os.path.splitext(name)[0]
        lab = os.path.join(WORK, 'labels', split, stem + '.txt')
        if os.path.exists(lab):
            shutil.copy(lab, os.path.join(WORK2, 'labels', stem + '.txt'))

imgs = sorted(glob.glob(os.path.join(WORK2, 'images', '*')))
print('合并后共 %d 张' % len(imgs))

random.seed(0)
random.shuffle(imgs)
n_val = max(1, int(len(imgs) * 0.15))
val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

with open(os.path.join(WORK2, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_imgs) + '\n')
with open(os.path.join(WORK2, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_imgs) + '\n')
with open(os.path.join(WORK2, 'data.yaml'), 'w') as f:
    f.write('path: %s\ntrain: train.txt\nval: val.txt\nnc: 1\nnames: [\'vehicle\']\n' % WORK2)

print('train %d 张，val %d 张' % (len(train_imgs), len(val_imgs)))
print('data.yaml:', os.path.join(WORK2, 'data.yaml'))

# ============ 第 5 段：接着 best_detrac.pt 训练 ============
from ultralytics import YOLO

BASE = '/kaggle/input/datasets/imrong/best-model/best_detrac.pt'
WORK2 = '/kaggle/working/dataset_flow5k'

model = YOLO(BASE)
model.train(
    data=os.path.join(WORK2, 'data.yaml'),
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    degrees=5.0,
    scale=0.5,
    mosaic=1.0,
    workers=2,
    project='/kaggle/working',
    name='flow5k_vehicle_v1',
)
