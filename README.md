# 多路口车流量统计 — Python 原型与模型转换

这是板端项目的前半部分，干两件事：Python 原型和模型转换。

原型跑检测、跟踪、穿线计数，用来验证算法、标定参数、准备训练数据。模型转换把 YOLOv8 的 ONNX 改成适合 RKNN INT8 量化的双输出，给板端用。

板端部署、网页监控和量化精度问题在 [multi-road-crossing-vehicle-count-cpp](https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp) 里解决。两个仓库共用 `configs/*.json` 和模型文件。

## 脚本

- `src/01_detect_video.py` — 逐帧检测、画框
- `src/02_track_video.py` — 加跟踪、编号
- `src/03_count_video.py` — 单条计数线，分方向计数
- `src/04_count_lanes.py` — 主入口：ROI + 跟踪 + 多线计数
- `src/05_export_csv.py` — 汇总报告
- `src/06_multistream.py` — 多路并行
- `src/07_train.py` — 训练
- `src/counter.py` — 计数核心，04 和 06 共用

模型是单类检测（vehicle），基于 UA-DETRAC 7633 张微调，精度 P 0.88 / R 0.818 / mAP50 0.908。

模型转换脚本在 `tools/01_convert/`，阈值标定脚本在 `tools/02_diagnose/`。

## 环境

Windows 11，Miniconda，conda 环境 `yolo`：

| 包 | 版本 |
|---|---|
| Python | 3.11.16 |
| ultralytics | 8.4.157 |
| torch | 2.14.0+cpu |
| opencv-python | 5.0.0.93 |
| lap | 0.5.13 |
| numpy | 2.4.6 |
| onnx | 1.23.0 |
| onnxruntime | 1.30.0 |
| matplotlib | 3.11.2 |

本机是核显（Intel Iris Xe），纯 CPU 推理，单路约 22 帧/秒，四路并行约 31 帧/秒。训练用云 GPU（Kaggle）。

模型转换在 WSL 里做，用 rknn-toolkit2 1.6.0+81f21f4d（conda 环境 `rknn16`）。2.5.0 转出来的模型宽高放大 13 倍，不能用，原因见 [docs/模型转换.md](docs/模型转换.md)。

## 安装

```bat
git clone https://github.com/ImJRong/multi-road-crossing-vehicle-count.git
cd multi-road-crossing-vehicle-count
conda create -n yolo python=3.11.16 -y
conda activate yolo
pip install ultralytics==8.4.157 torch==2.14.0+cpu opencv-python==5.0.0.93 lap==0.5.13 numpy==2.4.6 onnx==1.23.0 onnxruntime==1.30.0 matplotlib==3.11.2
```

或者直接双击 `start-traffic.bat`。

## 运行

视频文件（test1~test4.mp4）要自己放到 `data/videos/`，仓库里不带。

```bat
:: 主入口：多线分区计数
python src\04_count_lanes.py test3

:: 其他
python src\01_detect_video.py test1        :: 只检测
python src\02_track_video.py test1         :: 检测 + 跟踪
python src\03_count_video.py test1         :: 单线计数
python src\05_export_csv.py                :: 汇总 CSV
python src\06_multistream.py test1 test3   :: 多路并行
python src\07_train.py                     :: 训练
```

结果写到 `outputs/`：`records_<名字>.csv` 一定生成；带框 mp4 只在配置 `output` 为 `save` 时才写，`show` 是开窗口实时播放。test2 用 `save`，其余三个用 `show`。

## 配置

`configs/test1.json` ~ `test4.json` 对应四段视频，`roi`、`lines`、`detect` 不同。加新路口复制一份最接近的改数值：

```json
{
  "name": "test1",
  "video": "data/videos/test1.mp4",
  "max_frames": 0,
  "detect": { "imgsz": 640, "conf": 0.1, "iou": 0.4 },
  "roi": [[940,125],[185,125],[185,565],[940,565]],
  "lines": [
    { "name": "A", "a": [400,150], "b": [665,150],
      "count_dir": "neg", "margin": 6, "label_at": [406,140] },
    { "name": "D", "a": [730,150], "b": [880,280],
      "count_dir": "neg", "margin": 6, "label_at": [736,140] }
  ],
  "draw": { "box": true, "id": true, "line": true },
  "output": "show"
}
```

字段说明：

- 一条线只有 `a` 和 `b`，同时决定位置、方向、长度。
- `margin` 是线两侧的死区，压线抖动不计入，防止一辆车被数两次。
- `count_dir` 只数一个方向：`pos` 是负侧到正侧，`neg` 是正侧到负侧。
- `output` 只认 `save` 和 `show`。板端 C++ 还支持 `none`（纯测速）。

注意：这份配置是 Python 原型调参用的（test1 是 conf 0.1 / iou 0.4）。板端在 INT8 模型上重新标定过（同名文件 conf 0.3 / iou 0.3），两边不要直接互相覆盖。

## 模型转换

ONNX 转 RKNN，分三步：

```bat
:: 第 1 步：Windows 侧改检测头、导出 ONNX（每步带校验）
D:\Python3.12.4\python.exe tools\01_convert\make_head3.py
D:\Python3.12.4\python.exe tools\01_convert\make_head4.py
D:\Python3.12.4\python.exe tools\01_convert\verify_head4_decode.py
```

```bash
# 第 2 步：WSL 侧转 RKNN
conda activate rknn16
cd ~/traffic_count
python /mnt/d/multi-road-crossing-vehicle-count/tools/01_convert/convert_head4_probe.py
```

```bash
# 第 3 步：板端验证（看数值，不是看帧率）
cd ~/code
./rknn_probe models/新模型.rknn data/videos/test1.mp4 200
```

判据：距离输出的 `scale` 应该是 `0.046943`；`[OUT1 int8]` 那行的 `==zp(-128)` 在 8340 左右算正常，8400 说明分数全 0，模型不能用。

检测头为什么要拆成"距离"输出、量化刻度怎么算，见 [docs/模型转换.md](docs/模型转换.md)。

## 文档

- [docs/模型转换.md](docs/模型转换.md) — 检测头改造、量化参数、验证方法
- [docs/实操手册.md](docs/实操手册.md) — 训练数据准备、参数标定、量化实验
- [tools/README.md](tools/README.md) — 模型转换和阈值标定脚本
- 板端 C++ 的代码和文档在 [multi-road-crossing-vehicle-count-cpp](https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp)
