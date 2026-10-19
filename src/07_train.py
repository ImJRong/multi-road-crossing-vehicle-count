"""训练脚本 —— 以官方 YOLOv8n 为起点，在你自己标的数据上微调。

为什么叫"微调"而不是"从零训"：
  官方模型已经会认车了（它见过几百万张图）。我们要教的只是
  「监控俯视角度下的车长什么样」，所以不用从零学，几步就能有明显改善。

先决条件：
  1. 图放 data/dataset/images/，标注放 data/dataset/labels/（YOLO txt 格式）
  2. 跑过 python tools/split_dataset.py，生成 data.yaml

跑法：在项目根目录执行  python src/07_train.py
输出：output/<NAME>/
        weights/best.pt   整个训练里最好的一份（要部署就用它）
        weights/last.pt   最后一轮（要接着训就指向它）
        results.csv       每轮一行的指标
        results.png       曲线图

⚠️ 本机没有独立显卡，DEVICE 只能是 "cpu"，速度很慢。
   正式训练建议放云 GPU（Kaggle 免费），把 DEVICE 改成 "0" 即可。
"""

from ultralytics import YOLO

# ============ 配置区 ============

# ---------- 数据 ----------
DATA = "data/dataset/data.yaml"

# ---------- 从哪开始学 ----------
# 官方预训练模型当起点（迁移学习）。要接着上次的训，就换成 last.pt 的路径。
BASE_MODEL = "models/yolov8n.pt"

# ---------- 训练参数 ----------
EPOCHS = 50         # 跑多少轮。
                    # 微调不需要 100 轮（那是从零训练才要的量）。
                    # 参考：7600 张图 @50 轮，Kaggle 的 T4 上大约 1 小时
IMGSZ = 640         # 图片缩放到多大
                    # UA-DETRAC 原图是 960x540，640 足够，用 1280 是白烧算力
                    # ⚠️ 以后推理时也用 640。因为我们会先把 ROI 区域裁出来再送进去，
                    #    裁完图变小了，车在画面里反而更"大"，所以 640 不吃亏
BATCH = 16          # 一次处理几张。T4（16G）上 16 很稳；本机 CPU 跑就调到 2~4
DEVICE = "cpu"      # "cpu"；云 GPU 上写 "0"

# ---------- 数据增强 ----------
# 这几个是专门针对"监控俯视"调的：让模型对角度、尺度变化更稳
DEGREES = 5.0       # 随机旋转角度（监控机位固定，不用转太多）
SCALE = 0.5         # 随机缩放（远处车和近处车大小差别很大，这个很重要）
MOSAIC = 1.0        # 四图拼接，能显著增加小目标样本

# ---------- 输出 ----------
NAME = "detrac_vehicle_v1"     # 换个名字就不会覆盖上次的结果
PROJECT = r"D:\multi-road-crossing-vehicle-count\output"   # 必须绝对路径

# ===============================

model = YOLO(BASE_MODEL)

model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,

    degrees=DEGREES,
    scale=SCALE,
    mosaic=MOSAIC,

    # Windows 上多进程读数据容易报错，设 0 表示不用多进程（慢一点但稳）
    workers=0,

    # 结果输出位置必须写绝对路径——写相对路径的话 ultralytics 会拼到 runs/detect/ 下面
    project=PROJECT,
    name=NAME,
)

print()
print("=" * 62)
print("训练结束。结果在 output\\%s\\ ：" % NAME)
print("  weights/best.pt  整个训练里成绩最好的一份，部署就用它")
print("  weights/last.pt  最后一轮（要接着训就把它填到 BASE_MODEL）")
print("  results.csv/png  每轮指标 + 曲线图")
print()
print("下一步：看 results.png 里 mAP50 涨到多少，再拿它跑一遍那 4 个测试视频。")
print("=" * 62)
