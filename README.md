# 多路口车流量统计系统 · Python 原型与模型转换

**Multi-Road Crossing Vehicle Counting System (Python Prototype)**

## 一、项目简介

本项目面向城市多路口监控场景，基于 YOLOv8 单类车辆检测与 ByteTrack 多目标跟踪，实现车辆实时检测、轨迹编号、穿线计数与分方向统计，并提供从训练、模型转换到边缘部署的完整链路。

**项目背景**：路口车流量调查长期依赖人工计数或埋设线圈检测器，前者人力成本高、后者需要破路施工。基于视觉的方案可直接复用既有监控摄像头，但受限于边缘设备算力，必须采用定点量化模型，由此带来精度损失。

**解决的核心问题**：在 RK3588 NPU 上以 INT8 量化运行 YOLOv8 时，检测框定位精度严重劣化。项目通过改写检测头输出、将 DFL 距离换算移至 CPU 完成，把框中心偏移从 6.75 px 降至 0.03 px，轨迹断号下降 34.3%。

**应用场景**：路口车流量调查、交通信号配时优化、固定机位交通流统计、边缘侧视频结构化。

**核心技术栈**：Python 3.11 / Ultralytics YOLOv8 / ByteTrack / ONNX / RKNN-Toolkit2。

**仓库分工**

| 仓库 | 内容 |
|---|---|
| 本仓库 | Python 原型（算法验证、参数标定）、模型转换脚本、技术文档 |
| [multi-road-crossing-vehicle-count-cpp](https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp) | 板端 C++ 实现，含四路并行网页监控 |

两套代码独立运行，共用同一份模型与同一套 `configs/*.json` 配置。

## 二、技术栈

| 分类 | 内容 |
|---|---|
| 开发语言 | Python 3.11.16 |
| 核心框架 | Ultralytics 8.4.157、PyTorch 2.14.0+cpu |
| 视觉库 | OpenCV 5.0.0.93 |
| 多目标跟踪 | ByteTrack（Ultralytics 内置实现），依赖 lap 0.5.13 |
| 模型转换 | ONNX 1.23.0、ONNX Runtime 1.30.0、RKNN-Toolkit2 1.6.0+81f21f4d |
| 数值与绘图 | NumPy 2.4.6、Matplotlib 3.11.2 |
| 开发环境 | Windows 11、Miniconda（conda 环境 `yolo`） |

## 三、系统架构

系统按职责分为四层：应用层负责具体业务脚本，算法层封装检测 / 跟踪 / 计数逻辑，
数据处理层负责配置解析与结果落盘，模型层负责训练与格式转换。

```
┌──────────────────────────────────────────────────────────────┐
│ 应用层  01_detect_video.py  只检测                           │
│         02_track_video.py   加跟踪                           │
│         03_count_video.py   单线计数                         │
│         04_count_lanes.py   多线分区计数（主入口）           │
│         05_export_csv.py    汇总报告                         │
│         06_multistream.py   多路并行                         │
│         07_train.py         训练                             │
├───────────────────────────────┬──────────────────────────────┤
                                │
├───────────────────────────────┴──────────────────────────────┤
│ 算法层  counter.py          穿线计数核心（04 与 06 共用）    │
│         ultralytics YOLO    检测 + ByteTrack 跟踪            │
├───────────────────────────────┬──────────────────────────────┤
                                │
├───────────────────────────────┴──────────────────────────────┤
│ 数据层  configs/*.json     ROI、计数线、检测参数             │
│         outputs/*.csv       过线记录                         │
│         outputs/*.mp4       带框画面                         │
├───────────────────────────────┬──────────────────────────────┤
                                │
├───────────────────────────────┴──────────────────────────────┤
│ 模型层  tools/01_convert/   改写检测头、生成 ONNX            │
│         tools/03_failed/    失败方案存档（勿重复）           │
│         models/*.onnx       中间产物                         │
└──────────────────────────────────────────────────────────────┘
```

## 四、环境依赖

### 4.1 Windows 本机（Python 原型）

| 项 | 版本 |
|---|---|
| 操作系统 | Windows 11 家庭中文版（build 22000） |
| Python | 3.11.16（conda 环境 `yolo`，路径 `D:\miniconda3\envs\yolo\python.exe`） |
| ultralytics | 8.4.157 |
| torch | 2.14.0+cpu |
| opencv-python | 5.0.0.93 |
| lap | 0.5.13 |
| numpy | 2.4.6 |
| onnx | 1.23.0 |
| onnxruntime | 1.30.0 |
| matplotlib | 3.11.2 |

本机无独立显卡，仅 Intel Iris Xe 核显。单路约 22 帧/秒，四路并行合计约 31 帧/秒。
训练必须使用云 GPU（Kaggle）。

### 4.2 WSL Ubuntu（模型转换）

| 项 | 版本 |
|---|---|
| 操作系统 | Ubuntu 26.04.1 LTS（WSL2） |
| 编译器 | g++ 15.2.0 |
| 构建工具 | CMake 4.2.3 |
| conda | Miniforge3（`~/miniforge3`） |
| 模型转换 | rknn-toolkit2 1.6.0+81f21f4d，conda 环境 `rknn16`，Python 3.10.22 |
| 对照环境 | rknn-toolkit2 2.5.0，conda 环境 `rknn` / `rknn20`，Python 3.10.22 |

只允许使用 1.6.0 完成转换。2.5.0 转出的模型宽高放大 13 倍，原因见 §八 难点三。

### 4.3 板端 RK3588（部署目标）

| 项 | 版本 |
|---|---|
| 硬件 | Orange Pi 5 Plus（RK3588，8 核，5.8 GiB） |
| 操作系统 | Ubuntu 22.04.5，内核 5.10.160-rockchip-rk3588 |
| 编译器 | g++ 11.4.0、CMake 3.22.1 |
| NPU 运行库 | librknnrt.so 2.5.0 |
| OpenCV | 4.5.4 |

板端编译与运行见 [C++ 仓库 README](https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp)。

## 五、项目结构

```
multi-road-crossing-vehicle-count/
├── README.md               本文件
├── MEMORY.md               交接文档：板子怎么连、踩过哪些坑、当前卡在哪
├── start-traffic.bat       一键激活 conda 环境并进入项目目录
├── .gitignore
├── .gitattributes          统一行尾（仓库内 LF）
├── LICENSE                 学术使用许可（Academic Use Only）
├── configs/                每个路口一份 JSON
│   └── test1.json ~ test4.json
├── data/                   数据（不进仓库）
│   ├── videos/             原始路口视频 test1~test4
│   ├── samples/            抽帧图，用于画 ROI / 计数线
│   ├── dataset/            训练数据
│   └── aicity/             外部数据集
├── src/                    Python 主流程
│   ├── 01_detect_video.py  逐帧检测、画框
│   ├── 02_track_video.py   加跟踪、编号
│   ├── 03_count_video.py   单条计数线，分方向计数
│   ├── 04_count_lanes.py   按配置跑：ROI + 跟踪 + 多线计数（主入口）
│   ├── 05_export_csv.py    汇总报告
│   ├── 06_multistream.py   多路并行
│   ├── 07_train.py         训练
│   └── counter.py          计数核心（04 / 06 共用）
├── models/                 模型文件
│   ├── best_detrac.onnx    中转 ONNX
│   ├── best_detrac_head2.onnx  双输出头 ONNX（坐标，旧）
│   ├── best_detrac_head3.onnx  双输出头 ONNX（坐标）
│   └── best_detrac_head4.onnx  双输出头 ONNX（DFL 距离，当前主力）
├── tools/                  辅助脚本，按用途分 5 组
│   ├── 01_convert/         模型转换主线
│   ├── 02_diagnose/        诊断验证
│   ├── 03_failed/          失败方案存档
│   ├── 04_dataset/         数据集处理
│   ├── 05_figures/         讲解配图生成
│   └── README.md           完整清单与用法
├── docs/                   文档
│   ├── 实操手册.md         训练数据、参数标定、量化实验、板端说明
│   ├── head4_原理详解.md   框偏移方案推导（11 节，11 张图）
│   ├── 设计要点.md         设计决策与待做清单
│   ├── INT8验证方案.md     INT8 精度验证结论
│   ├── 归档-项目记忆-2026-10-22.md  已归档的历史记录
│   ├── figures/            原理讲解配图（11 张）
│   └── screenshots/        INT8 偏移现象的现场截图（3 张）
└── outputs/                运行产物：过线记录 CSV、带框视频（不进仓库）
```

**配套文档**

| 想知道什么 | 去哪 |
|---|---|
| 接手项目、排查环境问题 | [MEMORY.md](MEMORY.md) |
| 框偏移是怎么修的 | [docs/head4_原理详解.md](docs/head4_原理详解.md) |
| 训练数据、参数标定、量化实验 | [docs/实操手册.md](docs/实操手册.md) |
| 转模型 / 换模型 | [tools/README.md](tools/README.md) |
| 板端 C++ 代码 | [multi-road-crossing-vehicle-count-cpp](https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp) |
| C++ 代码怎么读（逐文件逐函数讲解） | 板端仓库 [docs/代码走读.md](https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp/blob/main/docs/%E4%BB%A3%E7%A0%81%E8%B5%B0%E8%AF%BB.md) |

## 六、编译 / 运行 / 部署

本项目为 Python 工程，无需编译。以下步骤可在 Windows 本机直接复现。

### 6.1 准备环境

```bat
:: 1. 取代码
git clone https://github.com/ImJRong/multi-road-crossing-vehicle-count.git
cd multi-road-crossing-vehicle-count

:: 2. 创建并激活 conda 环境
conda create -n yolo python=3.11.16 -y
conda activate yolo

:: 3. 安装依赖
pip install ultralytics==8.4.157 torch==2.14.0+cpu opencv-python==5.0.0.93 ^
            lap==0.5.13 numpy==2.4.6 onnx==1.23.0 onnxruntime==1.30.0 matplotlib==3.11.2

:: 或者直接双击 start-traffic.bat（内部已包含环境激活）
```

### 6.2 准备数据与配置

`configs/` 已随仓库提供 `test1.json` ~ `test4.json`，无需手工创建。
视频文件因体积原因不入库，需自行放入：

```bat
mkdir data\videos
:: 把 test1.mp4 ~ test4.mp4 放进 data\videos\
```

四个配置分别对应四段视频，`roi`、`lines`、`detect` 参数各不相同，
新增路口时复制一份最接近的配置改数值即可。配置结构见 §七 模块 6。

### 6.3 运行

所有脚本都在**项目根目录**执行（脚本内使用相对路径）。

**单路计数（主入口）**

```bat
python src\04_count_lanes.py test3
```

输出：`outputs\records_test3.csv`（过线记录，始终生成）。
带框视频仅在配置的 `output` 为 `"save"` 时生成到 `outputs\test3_lanes.mp4`；
为 `"show"` 时改为实时播放窗口，不落盘。四个配置里 test2 用 `save`，其余三个用 `show`。

**其他脚本**

| 命令 | 输出 |
|---|---|
| `python src\01_detect_video.py test1` | `outputs\test1_detect.mp4` |
| `python src\02_track_video.py test1` | `outputs\test1_track.mp4` |
| `python src\03_count_video.py test1` | `outputs\test1_count.mp4` |
| `python src\05_export_csv.py` | `outputs\report.csv` |
| `python src\06_multistream.py test1 test3` | `outputs\<名字>_lanes.mp4` |
| `python src\07_train.py` | `runs\...\weights\best.pt` |

### 6.4 模型转换（ONNX → RKNN）

```bat
:: 第 1 步：Windows 侧生成 ONNX（每步都带校验）
D:\Python3.12.4\python.exe tools\01_convert\make_head3.py
D:\Python3.12.4\python.exe tools\01_convert\make_head4.py
D:\Python3.12.4\python.exe tools\01_convert\verify_head4_decode.py
:: 必须看到差值 0，否则不要往下走
```

```bash
# 第 2 步：WSL 侧转 RKNN
conda activate rknn16
cd ~/traffic_count
python /mnt/d/multi-road-crossing-vehicle-count/tools/01_convert/convert_head4_probe.py
```

```bash
# 第 3 步：板端验证（不是看帧率，是看数值）
cd ~/code
./rknn_probe models/新模型.rknn data/videos/test1.mp4 200
```

判据：距离输出 `scale` 应为 `0.046943`；`[OUT1 int8]` 行的 `==zp(-128)` 约 8340 为正常，
8400 表示分数全部为 0，模型不可用。

### 6.5 部署到板端

只同步模型文件：

```bash
scp models/best_detrac.rknn orangepi@192.168.1.105:~/code/models/
```

> **配置文件不要直接覆盖。**两个仓库的 `configs/*.json` 同名但内容不同：
> 本仓库面向 Python 原型调参（如 test1 为 conf 0.1 / iou 0.4），
> 板端在 INT8 模型上重新标定过（同名文件为 conf 0.3 / iou 0.3）。
> 板端配置已随 C++ 仓库部署，需要改参数时直接改板上的文件，
> 比对后手动同步。

板端编译与运行步骤见 [C++ 仓库 README](https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp)。

## 七、核心功能模块

### 1. 检测模块（`src/01_detect_video.py`）

**作用**：逐帧调用 YOLOv8 检测车辆，绘制检测框。

**实现原理**：`ultralytics` 加载 `best_detrac.onnx`，按配置的 `imgsz`（640）与 `conf` 阈值推理，
结果绘制后写入 mp4。

**关键技术点**：单类检测（`vehicle`），模型基于 UA-DETRAC 7633 张微调，精度 P 0.88 / R 0.818 / mAP50 0.908。

### 2. 跟踪模块（`src/02_track_video.py`）

**作用**：跨帧关联同一车辆，分配稳定编号。

**实现原理**：调用 Ultralytics 内置 ByteTrack（`model.track()`），卡尔曼滤波预测 + 匈牙利匹配 + 两阶段关联。

**关键技术点**：跟踪 ID 会因遮挡或漏检而断裂，因此「出现过多少 ID」不能当作车流量，必须使用穿线计数。

### 3. 计数模块（`src/counter.py`）

**作用**：判定车辆是否穿过计数线，并按方向累计。

**实现原理**：每帧对每辆车计算其相对计数线的侧别。用叉积判断在哪一侧，用投影比例判断是否落在线段范围内，
符号变化即视为换边。设置死区（`margin` 像素）防止车辆压线时反复计数。

**关键技术点**：`count_dir` 支持单方向统计（`"pos"` = 负侧→正侧，`"neg"` = 正侧→负侧）。
单行道场景下这是设计意图，不是缺陷。

### 4. 多线分区计数（`src/04_count_lanes.py`）

**作用**：按配置文件执行 ROI 裁剪 + 跟踪 + 多条线计数，是项目的标准入口。

**实现原理**：读取 `configs/<名字>.json`，按 `roi` 多边形裁剪（外接矩形裁剪 + 中心点过滤），
对每条 `lines` 分别执行穿线计数，结果写入 CSV。

**关键技术点**：ROI 多边形只用于过滤框中心，裁剪用的是它的外接矩形。把区域外涂黑反而会检出更多垃圾框。

### 5. 多路并行模块（`src/06_multistream.py`）

**作用**：一个进程管理多路视频，可指定只跑某几路。

**实现原理**：使用 `multiprocessing` 的 `spawn` 上下文开进程，每路一个进程，各自加载一份模型实例与计数器，共用同一份计数核心代码。
选进程不选线程的原因：Python 的 GIL 使得多线程无法真正并行执行 Python 层代码。

**关键技术点**：四路并行合计约 31 帧/秒（本机核显，纯 CPU 推理）。进程数超过 CPU 核数反而变慢，因为每个进程都要独立加载一份模型。

### 6. 配置模块（`configs/*.json`）

**作用**：为所有脚本提供统一参数。

**配置结构**

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

**关键技术点**：

- 一条线只有一个 `a` 和一个 `b`。两点同时决定位置、方向与长度，线画到哪就只管那一段。
- `margin` 是线两侧的死区像素数，压线抖动不计入，防止同一辆车被计两次。
- `count_dir` 只统计一个方向：`"pos"` 为负侧到正侧、`"neg"` 为正侧到负侧。单行道场景下这是设计意图。
- `output` 在 Python 侧只接受 `"save"`（存 mp4）与 `"show"`（开窗口实时播放，按 Q 退出）。
  板端 C++ 侧额外支持 `"none"`（不输出，纯测速），无显示器环境下不能用 `"show"`。
- JSON 中的 `说明_xxx` 键是写给人看的注释，解析时被忽略。

### 7. 模型转换模块（`tools/01_convert/`）

**作用**：把训练好的 PyTorch 权重改写为适合 INT8 量化的双输出结构。

**实现原理**：删除 ONNX 末端的 Concat（合并坐标与分数）以及 DFL 之后的换算节点，
把检测头拆成两路独立输出，使坐标与分数各自拥有独立的量化参数。

**关键技术点**：`make_head4.py` 生成后必须用 `verify_head4_decode.py` 校验 ——
把两路输出拼回去与原模型逐元素比较，差值必须为 0。

## 八、关键技术难点与解决方案

### 难点一：INT8 量化导致检测框整体向左上偏移

**问题**：RKNN INT8 模型输出的检测框相比 PC 端 FP32 结果整体向左上偏移，且目标越小偏移越大。
实测中心偏移 6.75 px（640 空间 1.86 px），在 1920 宽画面上肉眼明显。

**成因**：RKNN 为每个输出张量按 `范围 / 256` 分配量化刻度。原模型把坐标（0~640）与分数（0~1）
拼在同一个 `[1,5,8400]` 张量里，刻度被坐标撑到 2.5017；而 DFL 内部的距离值只有 0~15，
在粗刻度下仅剩 6 个有效台阶。误差链条经六步逐级放大 53 倍：

```
DFL 距离 0~11.97        刻度 0.047
anchor − 左距离          刻度 0.325   ← 大数减小数，小数值精度被稀释
anchor + 右距离          刻度 0.327
两者相加 ÷ 2             刻度 0.625
两者相减 = 宽高          刻度 0.084
拼 4 通道 × stride       刻度 2.502
```

**解决方案**：把模型输出改为 DFL 距离本身，坐标换算移到 C++ 侧用 float 完成。

- 从 ONNX 图中删除 DFL 之后的 9 个换算节点（Slice×2 / Sub×2 / Add×2 / Div / Concat / Mul），
  将 `/model.22/dfl/Reshape_1_output_0` 设为图输出。这 9 个节点不含权重。
- 距离成为独立输出后，RKNN 按其真实值域分配刻度 `0.046943`，精度提升 53 倍。
- C++ 换算：`x1 = (anchor_x − 左距离) × stride`，`y1 = (anchor_y − 上距离) × stride`，右/下边界同理用加号。

**效果**：中心偏移 6.75 px → 0.03 px。模型权重未改、量化方式未改、推理速度未变。
三道验证全过：ONNX 逐元素差值 0、numpy 公式复算差值 0、板端实测偏移 ≤ 0.04 px。

**易错点**：`stride` 是 8400 长度的数组（前 6400 为 8、中 1600 为 16、后 400 为 32），
当作标量计算会产生最大 453 像素的误差。anchor 网格也需分段生成 80×80 / 40×40 / 20×20。

### 难点二：拆输出头时分数被全部量化成 0

**问题**：拆分 `[1,5,8400]` 张量时，若在 Concat 之后加 Split，RKNN 编译器会把该节点融合掉，
导致分数通道全部输出 `-128`（即 zp，对应 0.0）。

**尝试过但失败的方案**：

| 方案 | 结果 |
|---|---|
| Concat 之后加 Split | 1.6.0：8400 个分数全为 -128；2.5.0：分数正常但框塌 |
| Split + `custom_quantize_layers` | `KeyError: '/model.22/Concat_1_output_0'`，配置结构不匹配 |
| Split + `proposal=False` 的混合量化 | `KeyError: '/model.22/dfl/conv/Conv_output_0'` |

**解决方案**：直接删除末端的 Concat，把 `Mul_2`（坐标）与 `Sigmoid`（分数）各自作为图输出，
图中不存在可被融合的节点。

### 难点三：rknn-toolkit2 2.5.0 转出的模型宽高放大 13 倍

**问题**：同一份 ONNX、同一批校准图、同一套配置，仅更换工具版本，2.5.0 转出的模型
宽高放大约 13 倍，四个输出通道的实测范围全部顶到 637.92（= 255 × 2.5 的饱和值），
而正常模型四个通道范围各不相同。

**排查过程**：先用基线 INT8 与 `auto_hybrid`（自动混合量化）各转一版，两者皆坏。
再通过冒烟测试确认 `auto_hybrid` 开关本身工作正常（阈值变化确实改变了模型体积与输入 dtype）。
逐层比对后确认：2.5.0 把 DFL 距离（0~15）与坐标（0~640）塞进同一套量化参数，
属于工具链缺陷，与混合量化功能无关。

**解决方案**：转换固定使用 rknn-toolkit2 **1.6.0+81f21f4d**。2.5.0 仅用于对照实验。

**补充坑**：2.5.0 的 `build()` 必须加 `disable_rules=["seperate_conv_rs_split"]`，
否则报 `IndexError: list index out of range`；该参数在 1.6.0 中不被识别。

### 难点四：转弯车辆漏计

**问题**：车辆转弯时计数线判定失效，出现漏计。

**排查与结论**：改用 head4 模型后，12135 帧测试数据上轨迹断号从 2559 降至 1680（下降 34.3%）：

| 视频 | 帧数 | 原模型编号数 | head4 编号数 | 降幅 |
|---|---|---|---|---|
| test1 | 3600 | 553 | 450 | 18.6% |
| test2 | 3600 | 625 | 441 | 29.4% |
| test3 | 2535 | 909 | 539 | 40.7% |
| test4 | 2400 | 472 | 250 | 47.0% |
| 合计 | 12135 | 2559 | 1680 | 34.3% |

计数净增 12 次，无任何一条线下降。说明框抖动导致的编号断裂是一个成因，
但只解释了约三分之一的漏计。剩余成因未进一步查证，投入产出比不高，已标记收尾。

### 难点五：INT8 量化本身带来的精度损失

**问题**：INT8 相对 FP32 的精度损失需要量化，才能判断是否可接受。

**实测方案**：同一份权重导出三个模型，在验证集 1510 张（10795 个标注框）上对比，
评估口径 conf=0.001、NMS IoU=0.7，单类，AP 按全点插值。

| 模型 | mAP50 | mAP50-95 | 每图框数 |
|---|---|---|---|
| ONNX FP32 | 0.9426 | 0.7196 | 26.1 |
| RKNN FP16 | 0.9426 | 0.7191 | 26.2 |
| RKNN INT8 | 0.9037 | 0.6125 | 26.2 |

**结论**：INT8 相对 FP32，mAP50 下降 3.9 个百分点（相对 4.1%），mAP50-95 下降 10.7 个百分点（相对 14.9%）。
损失集中在定位精度而非检出能力（每图框数 26.1 → 26.2 基本不变）。FP16 与 FP32 无可测差异，可视为无损。

### 难点六：混训导致灾难性遗忘

**问题**：第 2 轮训练用 Intersection-Flow-5K 的 6928 张整个替换 UA-DETRAC 继续训练，
结果 test3 / test4 反而变差。

**解决方案**：回退到旧模型。后续若要扩展数据集，必须新旧按比例混合训练，不能整体替换。

## 九、版本迭代记录

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.0.0 | 2026-11-06 | 补齐工程文档：README 按规范重写，操作层内容归档至 `docs/实操手册.md` |
| v0.9.0 | 2026-11-02 | 修复框偏移：模型输出改为 DFL 距离，中心偏移 6.75 px → 0.03 px；修复计数漏（断号下降 34.3%） |
| v0.8.0 | 2026-11-01 | 修复计数记录只增不删的内存泄漏；模型转换配方定型 |
| v0.7.0 | 2026-10-22 | INT8 量化精度验证：拆分输出头，完成 FP32 / FP16 / INT8 三方对比 |
| v0.6.0 | 2026-10-19 | 补充 RKNN 转换脚本与 Python / C++ 对拍工具 |
| v0.5.0 | 2026-10-18 | 项目初版：Python 原型与模型导出 |

## 十、作者与声明

| 项 | 内容 |
|---|---|
| 作者 | jiarong（RONGG） |
| 仓库 | https://github.com/ImJRong/multi-road-crossing-vehicle-count |
| 相关仓库 | https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp （板端 C++ 实现） |
| 日期 | 2026-11-06 |

**License**：仅供学术研究（Academic Use Only）。完整条款见 [LICENSE](LICENSE)。
未经作者书面许可，不得用于商业用途、不得再分发与对外提供服务。

模型基于 UA-DETRAC 数据集微调，数据集版权归原作者所有。YOLOv8 采用 Ultralytics 开源实现，
遵循其 AGPL-3.0 许可。
