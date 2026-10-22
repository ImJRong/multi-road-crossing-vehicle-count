# 多路口车流量统计（RK3588 部署）

多路监控视频同时输入，检测车辆，统计各方向的过车数，最后部署到 RK3588 上跑。

Python 版负责验证算法、调参数；C++ 版是板端用的正式实现。两份代码的计数逻辑一一对应。

## 目录结构

| 目录 | 放什么 |
|---|---|
| `data/videos/` | 原始路口视频（test1~test4，各一个文件） |
| `data/samples/` | 从视频里抽的帧，用来画 ROI / 计数线 |
| `data/dataset/`、`data/aicity/` | 训练数据，不进仓库 |
| `models/` | `best_detrac.pt`（当前主力）、`.onnx`（中转）、`.rknn`（板子用） |
| `src/` | Python 主流程：检测、跟踪、计数、报告、多路并行 |
| `configs/` | 每个路口一份 json：ROI、计数线、检测参数、画法 |
| `outputs/` | 跑出来的结果：带框的视频、计数 csv |
| `tools/` | 辅助脚本：抽帧、画线、读坐标、准备数据集 |
| `docs/` | `项目记忆.md`（进度和踩坑，接着干先看这个）、`设计要点.md`（待做改进清单） |

C++ 版**不在本仓库**，它单独放在 WSL：`/home/imrong/traffic_count`（另一个 git 仓库）。
两个环境各自独立，只共用 `configs/` 和 `data/` 这两份数据，不来回搬产物。

## 跑法

所有 Python 脚本都在项目根目录跑（脚本里用的是相对路径）。

```bat
cd /d D:\multi-road-crossing-vehicle-count
D:\yolo-practice\start-yolo.bat        :: 或者双击 start-traffic.bat
python src\04_count_lanes.py test3
```

| 脚本 | 干什么 | 输出 |
|---|---|---|
| `src/01_detect_video.py` | 逐帧检测，画框 | `outputs/test1_detect.mp4` |
| `src/02_track_video.py` | 加跟踪，给每辆车编号 | `outputs/test1_track.mp4` |
| `src/03_count_video.py` | 单条计数线，分两个方向数 | `outputs/test1_count.mp4` |
| `src/04_count_lanes.py` | 按配置跑：ROI 裁剪 + 跟踪 + 多条线计数 | `outputs/testN_lanes.mp4`、`outputs/records_testN.csv` |
| `src/05_export_csv.py` | 把原始记录汇总成报告 | `outputs/report.csv` |
| `src/06_multistream.py` | 多路并行，一个进程管一路 | `outputs/<名字>_lanes.mp4` |
| `src/counter.py` | 计数核心，04 和 06 共用，不单独跑 | — |
| `src/07_train.py` | 训练（以官方 yolov8n 为起点，imgsz=640） | `runs/.../weights/best.pt` |

`06_multistream.py` 可以只跑指定的几路：`python src\06_multistream.py test1 test3`。

### 配置文件的约定

`configs/<名字>.json`，四个视频共用一套结构：`video` / `detect` / `roi` / `lines` / `draw` / `output` / `max_frames`。

一条线只有一个 `a` 和一个 `b`。这两个点同时决定位置、方向和长度，线画到哪就只管那一段。想调高低就改 `a`/`b` 的 y，想调长短就拉长或缩短两端。数字（A/B/C）写在哪由 `label_at` 定，不写就自动放在线的上端。

`output` 填 `"save"` 存 mp4，填 `"show"` 开窗口实时播放（按 Q 退出）。

### 工具

| 工具 | 干什么 |
|---|---|
| `tools/grab_frame.py` | 导出某一帧：`python tools\grab_frame.py data\videos\test4.mp4 300` |
| `tools/make_line_base.py` | 生成带 ROI 框的底图，在底图上画线不会画出界 |
| `tools/read_roi.py` | 读你在图上画的 ROI，写进配置：`--write configs/xxx.json` |
| `tools/read_line.py` | 读你在图上画的线，支持 `--base <底图>` |
| `tools/sweep_line.py` | 扫不同高度的线，看各位置覆盖多少车 |
| `tools/sweep_slope.py` | 扫不同斜率的线 |
| `tools/cut_video.py` | 剪掉头尾（水印、片尾黑屏） |
| `tools/extract_frames.py` | 抽帧做训练数据 |
| `tools/prelabel.py` | 用现有模型预标注，省手工标注的力 |
| `tools/split_dataset.py` | 按摄像头分组划分训练/验证集 |
| `tools/prepare_detrac.py` | 把 UA-DETRAC 转成 YOLO 格式 |
| `tools/kaggle_flow5k.py` | 在 Kaggle 上处理 Intersection-Flow-5K |
| `tools/check_gpu.py`、`check_obstacle.py` | 环境诊断 |

## 进度

- [x] 建工程目录
- [x] 单路检测 / 跟踪 / 穿线计数
- [x] 分车道、分方向计数，结果存 csv
- [x] 多路并行（PC 上 4 路，CPU 合计约 31 帧/秒）
- [x] ROI 裁剪（外接矩形裁剪 + 多边形过滤中心点）
- [x] 模型：UA-DETRAC 7633 张微调，`P 0.88 / R 0.818 / mAP50 0.908`
- [x] 导出 ONNX（12.2 MB，输入 640，输出 `[1,5,8400]`）
- [x] ONNX → RKNN（8.1 MB，fp16 未量化，`target_platform='rk3588'`）
- [x] C++ 版：ROI + 跟踪 + 计数，PC 上跑通（16~18 帧/秒）
- [x] INT8 量化验证：1510 张验证集，mAP50 -3.9 点、mAP50-95 -10.7 点
- [x] C++ 的 RKNN 后端代码（`rknn_detector.cpp` + `USE_RKNN` 开关）；模拟器验过输入格式，fp16 输出与 ONNX 一致
- [ ] C++ 的 RKNN 版编译验证、RK3588 上多路实时跑、测功耗
### INT8 量化掉了多少精度（2026-10-22 实测）

同一份权重导出三个模型，在验证集 1510 张（10795 个标注框）上对比。评估口径 conf=0.001、NMS IoU=0.7，单类。

| 模型 | mAP50 | mAP50-95 | 每图框数 |
|---|---|---|---|
| ONNX FP32 | 0.9426 | 0.7196 | 26.1 |
| RKNN FP16 | 0.9426 | 0.7191 | 26.2 |
| RKNN INT8 | 0.9037 | 0.6125 | 26.2 |

- INT8 相对 FP32：mAP50 下降 3.9 个百分点（4.1%），mAP50-95 下降 10.7 个百分点（14.9%）
- 损失主要在定位精度：坐标量化步长 2.5 px，匹配框中心平均偏移 1.86 px；每图框数 26.1 → 26.2，检出能力不变
- FP16 与 FP32 在本口径下无可测差异，可视为无损

INT8 转换前必须先拆分输出头（boxes + scores 两路输出），否则坐标与分数共用一个量化参数，
0~1 的分数会被压成 0。生成脚本为 `tools/make_split_head.py`，板端 C++ 需按双输出解析。
完整数据与说明见 `docs/项目记忆.md` 第十四节。

- [ ] 收集 20 个以上不同机位的训练素材
- [ ] 标注数据、混训
- [ ] 人工真值对照

## 训练数据怎么准备

⚠️ `data/videos/` 里那 4 个视频是最终测试用的，不能拿去训练。拿它们训练等于用考卷当练习册，测出来的分数是虚的。

数据来源是**多个不同摄像头的画面**，不是同一个视频切很多帧。

| 项 | 目标 |
|---|---|
| 总张数 | 150 ~ 300 张 |
| 摄像头数量 | 20 个以上 |
| 每个摄像头 | 5 ~ 20 张，覆盖不同时段和车流密度 |

同一个摄像头连拍的帧相似度 95% 以上，标 100 张等于标了同一张的 100 个复印本。宁要 20 个机位各 10 张，不要 1 个机位 200 张。

```bat
python tools\extract_frames.py data\videos\其他视频.mp4 10
python tools\prelabel.py 0.3          :: 先自动标一遍，再手工改，能省大半力气
:: 用标注工具打开 data\dataset\images 改标注，存 YOLO txt 到 data\dataset\labels
python tools\split_dataset.py         :: 按摄像头分组划分，防止同机位的图两边都出现
python src\07_train.py
```

标注标准：肉眼能认出是车的都框，认不出的都不框。远处的小车要框，只框中间的大车，模型会学会忽略远处。骑电动车的人、行人、自行车不框。

**混训的教训**：第 2 轮用 Intersection-Flow-5K 6928 张整个替换掉 UA-DETRAC 接着训，结果 test3/test4 反而更差，属于灾难性遗忘，只能退回旧模型。下次要么新旧按比例混着训，要么别换数据集。

## 关键参数怎么定

计数线的 y 不是猜的，两条路。

自动扫（先用这个）：

```bat
python tools\sweep_line.py data\videos\test3.mp4 200
```

它打印每个高度各有多少辆车穿过。挑次数多、又不在车排队等红灯那一带的高度。

手工画：

```bat
python tools\grab_frame.py data\videos\test1.mp4 1
:: 用画图工具在那张图上画一条线，另存成别的名字（别覆盖原图）
python tools\read_line.py data\samples\test1_frame1.jpg data\samples\我画的.jpg --write configs\test1.json 0
```

它会比对两张图的差异，把线拟合出来，再生成一张核对图让你确认。

ROI 两条规矩：一是不能贴计数线，要留出一个车身的助跑区，否则车刚进视野就到线上，跟踪器还没看清就过线，必漏；二是多边形只用来过滤框中心，裁剪用的是它的外接矩形，把区域外涂黑反而会检出更多垃圾框。

## 输入尺寸

监控画面里车很小，YOLO 默认压到 640×640，远处的车基本消失。同一张 1920×1080 的图实测检出数量：

| 输入尺寸 | 检出的车 |
|---|---|
| 640 | 29 |
| 960 | 51 |
| 1280 | 71 |
| 1600 | 84 |

裁出中心区域再送进去效果更明显。同一张图测过：

| 做法 | 中心区检出 | 耗时中位数 |
|---|---|---|
| 整图 → 640 | 21 辆 | 32 ms |
| 整图 → 1280 | 32 辆 | 103 ms |
| 裁中心 960 → 640 | 31 辆 | 48 ms |
| 裁中心 960 → 800 | 37 辆 | 73 ms |

所以 ROI 裁剪不只是为了排除干扰，它本身就等于放大。

**但当前全链路是 640**：训练 640、配置 640、ONNX 640、RKNN 640。1280 只在实验里测过检出数，没有真跑起来过。原因是训练尺寸和部署尺寸必须一致，要上 1280 得重训、重导 ONNX 和 RKNN，还要在板子上复测帧率。这件事没做，是目前精度的主要缺口。

## 环境

Windows 本机（Python 原型）：

- conda 环境 `yolo`，`D:\miniconda3\envs\yolo\python.exe`
- Python 3.11.16，ultralytics 8.4.157，torch 2.14.0+cpu，opencv 5.0.0，lap 0.5.13
- 没有独立显卡，只有 Intel 核显。单路约 22 帧/秒，4 路并行合计约 31 帧/秒。训练必须走云 GPU（Kaggle）

WSL `Ubuntu`（C++ 和模型转换）：

- C++ 工程在 `~/traffic_count`（不在本仓库），`cd ~/traffic_count && cmake . && make` 编译，`./traffic_count test3` 运行
- 依赖 OpenCV、`~/onnxruntime-linux-x64-1.31.0`
- conda 环境 `rknn`（Python 3.10.22，RKNN-Toolkit2 2.5.0），`source ~/rknn_env.sh` 激活，负责 ONNX → RKNN
- 交叉编译工具链 `gcc-aarch64-linux-gnu` 15.2.0 已装好，能产出 ARM 可执行文件；还缺 ARM 版 OpenCV 和 aarch64 的 `librknnrt.so`（建议板子到了从板子 rsync，别下 deb）

## C++ 版（板端用）

源码在 WSL 的 `~/traffic_count`。程序按 `CMakeLists.txt` 里的 `TRAFFIC_PROJECT_ROOT` 找 `configs/` 和 `data/videos/`，默认写死 `/mnt/d/multi-road-crossing-vehicle-count`，换机器改这一行。

```
CMakeLists.txt          编译配置，USE_RKNN 开关切推理后端
src/main.cpp            读配置 → 裁 ROI → 推理 → 跟踪 → 计数 → 画 → 存/播
src/detector.cpp/.hpp   ONNX Runtime 封装：letterbox 640、分数过滤、NMS
src/rknn_detector.cpp   RKNN 封装（板子用），前处理和后处理与 ONNX 版一字不差
src/counter.cpp/.hpp    side / near_segment / 死区 / count_dir
src/config.cpp/.hpp     解析 configs/*.json
src/tracker.cpp/.hpp    跟踪：卡尔曼预测 + 匈牙利 + 两阶段关联
src/kalman_track.hpp    卡尔曼滤波封装（只有头文件）
src/mini_json.hpp       自写的 JSON 解析，不依赖第三方
src/recorder.hpp        过线记录写 csv
```

跟踪已经是完整的两阶段 ByteTrack，和 ultralytics 对齐：卡尔曼预测位置 → 高分框（≥0.25）匈牙利匹配，代价 `1-IoU`、阈值 0.8 → 没配上的轨迹再和低分框（0.1~0.25）匹配一次，阈值 0.5 → 高分框没人认领的建新轨迹，低分框没人认领的直接丢 → 两轮都没配上的 `lost+1`，超 30 帧删。
两条容易搞错的地方：**Lost 状态的轨迹必须参与第一轮**，否则"留着等它回来"是空话；**匈牙利求出的是完美匹配**，必须再剔除代价超阈值的配对。

另一条规则是静止目标过滤：轨迹离出生位置不到 `kMinTravel`（4 像素）就算没动，这类目标既不计数也不画框。红绿灯、路牌、广告牌都被这条挡住。

实测（CPU，核显机器）：

| 视频 | 帧数 | 帧率 | A 线 | B 线 | 备注 |
|---|---|---|---|---|---|
| test3 | 2535 | 18~19 FPS | 243 | 186 | B 线人工真值 ≥190，差 4 次（2%） |
| test2 | 27241 | 17.5 FPS | 842 | 893 | Python 版是 838 / 893，总数差 4 次（0.2%） |

test2 累计分配 4608 个编号，实际计数 1735 次。比例高是因为俯视画面里大量车只在 ROI 内经过、从不穿线，照样会建轨迹发编号，属正常。

## 已知问题

- 官方 YOLOv8n 没在路口监控的机位和光照上训练过，远景小车、被遮挡的车会漏检。这是要自己标数据微调的原因。
- 跟踪 ID 会断。车被遮挡或漏检一帧就换新号，所以不能拿「出现过多少 ID」当车流量，必须用穿线计数。
- 红绿灯和杆子会被检成车（置信度 0.37 左右）。调 `conf` 去不掉，因为远处真车也在这个区间。静止过滤能不计数，但根治要标数据。
- C++ 的跟踪已经和 Python 的 ByteTrack 对齐（卡尔曼 + 匈牙利 + 两阶段）。test2 全片两边只差 4 次（0.2%）；test3 的 B 线 C++ 186、Python 165，差的这 21 次来自检测能力（远处小车、遮挡），不是跟踪。
- C++ 的 RKNN 版还没编译过，也没上板。`rknn_detector.cpp` 送的是 float32 NCHW 0~1，这类问题只有真编译、真跑才会暴露。
- test4 那辆被植被挡住的白色轿车，模型直接漏掉，至今没解决。
- 静止过滤有副作用：等红灯的车在开动前不画框。计数不受影响，因为过线那一刻它已经在动了。
