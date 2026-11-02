# 接着已训好的 flow5k 模型再训 30 轮。
# 用法：贴进 Kaggle 一格，把 NEWBASE 换成你刚下载/挂上的 best.pt 路径，跑。
# 跑完这段会自动打包成 zip，记得下载。

import os, shutil
from ultralytics import YOLO

# ============ 改这里 ============
# 上一次训好的权重（刚 Upload 成 Kaggle 数据集的那个，或者是 /kaggle/working 下的）
NEWBASE = '/kaggle/input/<你的数据集名>/best.pt'

# 数据集目录（如果换了 session，得先把第 2、3 格重跑一遍生成这个目录）
WORK2 = '/kaggle/working/dataset_flow5k'
# ==============================

# 先检查两样东西在不在，省得跑到一半才报错
assert os.path.exists(NEWBASE), '找不到权重：%s' % NEWBASE
assert os.path.exists(os.path.join(WORK2, 'data.yaml')), \
    '找不到数据集，先把第 2、3 格重跑一遍：%s' % WORK2

print('权重:', NEWBASE)
print('数据:', os.path.join(WORK2, 'data.yaml'))

model = YOLO(NEWBASE)
model.train(
    data=os.path.join(WORK2, 'data.yaml'),
    epochs=30,
    imgsz=640,
    batch=32,
    device=0,
    degrees=5.0,
    scale=0.5,
    mosaic=1.0,
    workers=8,
    project='/kaggle/working',
    name='flow5k_vehicle_v2',      # 换个名字，别覆盖上一轮的
    resume=False,
)

# ============ 训练完自动打包 ============
wdir = '/kaggle/working/flow5k_vehicle_v2/weights'
if os.path.isdir(wdir):
    out_zip = shutil.make_archive('/kaggle/working/model_v2', 'zip', wdir)
    print('权重已打包:', out_zip)
    print('大小: %.1f MB' % (os.path.getsize(out_zip) / 1024 / 1024))
    print('→ 去 Output 下载 model_v2.zip，里面就是 best.pt')
else:
    print('没找到权重目录，训练可能没跑完')
