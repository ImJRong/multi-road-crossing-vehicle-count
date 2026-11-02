# 项目交接文档（车流量统计 · RK3588 板子端）

> 最后更新：**2026-11-02 白天**。**先看完这一份再动手。**
> 同目录 `2026-10-2x.md` 是当天流水，里面的结论可能有错；**与本文冲突时以本文为准**。
> **框偏移已解决**，原理详见 `docs/head4_原理详解.md`（带图，从零讲起）。

---

## 〇、30 秒速览（赶时间只看这段）

**项目**：YOLOv8 车流量检测跑在 Orange Pi 5 Plus（RK3588）NPU 上，带网页实时看。

**板子**：`ssh orangepi@192.168.1.105`（静态 IP，密码 orangepi），代码在 `~/code`。

**现在什么能用**：推理 + 跟踪 + 计数 + 网页，18.3 fps，跑 `models/best_detrac.rknn`（= **head4 新模型**）。

**框偏移：已解决。** 中心偏移从 6.75px 降到 **0.03px**（改善约 200 倍），速度不变。
做法：把 ONNX 的输出从「坐标 0~640」改成「DFL 距离 0~15」，坐标换算搬到 C++。
**模型权重一个字节没改，量化方式也没改。** 原理详见 `docs/head4_原理详解.md`。

**计数漏：已收尾。** 实测断号减少 34.3%（2559 → 1680 个编号），
计数净增 12，无一条线下降。剩余漏计成因未查证，投入产出比不高，暂不深究。
详见 6.1 节。

**已经堵死的路（别再试）**：
- rknn-toolkit2 的**手动混合量化**：1.6.0 崩 `KeyError: Concat_1_output_0`，2.5.0 转出来是坏的（详见 5.8）
- **2.5.0 转普通 INT8 也是坏的**（w/h 通道被撑大十几倍，见 6.2）
- `mmse` / `kl_divergence` / `hybrid_level` 这些量化参数：1 像素级收益，没用
- `count_dir` 只数一个方向 —— **不是缺陷**，这几个路口是单行道，设计如此

---

## 〇点五、文档地图

| 文档 | 内容 | 什么时候看 |
|---|---|---|
| `MEMORY.md`（本文） | 状态、交接信息、踩坑、待办、**性能实测（六点五）** | **接手第一份** |
| `README.md` | 怎么跑、目录结构、进度 | 想跑起来 |
| `docs/head4_原理详解.md` | 框偏移从零推导，11 节 11 图 | 想懂原理 |
| `tools/README.md` | 50+ 脚本分 5 组，标清能跑/别跑 | 要转模型或诊断 |
| `docs/设计要点.md` | 待做改进清单（已标完成情况） | 规划下一步 |
| `docs/INT8验证方案.md` | INT8 掉了多少精度（已执行，含结论） | 评估量化代价 |
| `docs/归档-项目记忆-2026-10-22.md` | 10-17~10-22 的流水（**已归档，别当依据**） | 追溯历史决策 |

> **只想看性能数字**：直接跳到 **六点五节**（吞吐、延迟、实时流模拟、硬件上限）。

---

## 一、项目目标

把 YOLOv8 车流量检测跑在 Orange Pi 5 Plus（RK3588）的 NPU 上：

1. NPU 实时推理（目标 20fps+）
2. 跟踪 + 过线计数
3. 网页实时看画面和计数（板子没显示器，只能靠网页）
4. 最终用途：竞赛/论文的端侧部署 demo

---

## 二、连板子

```bash
ssh orangepi@192.168.1.105     # 密码 orangepi，sudo 也是 orangepi
```

| 项 | 值 |
|---|---|
| IP | **192.168.1.105（已设成静态）** |
| 网卡 | Realtek 无线（一块芯片，两个接口） |
| 系统 | Ubuntu 22.04.5，内核 5.10.160-rockchip |
| NPU 运行库 | `/usr/lib/librknnrt.so` **2.5.0**（别升级） |
| NPU 驱动 | 0.9.6（官方镜像自带，别动） |

**静态 IP 是怎么设的**（2026-11-01 做的，记下来免得再折腾）：

- NetworkManager 连接名 `700`
- `nmcli con mod 700 ipv4.method manual ipv4.addresses 192.168.1.105/24 ipv4.gateway 192.168.1.1 ipv4.dns 192.168.1.1`
- 已经重启验证过，MAC 怎么变都是 `.105`

**为什么必须用静态 IP**：这块卡会同时暴露 `wlan0` / `wlan1` / `wlP2p33s0` 三个接口，**接口名和 MAC（d0:39:57:cc:9b:eb 与 d2:...）每次开机可能互换**，靠路由器绑 MAC 根本治不住。

**板子现在没有外网**：路由器里留了两条 IP-MAC 绑定（`.105`/`.107`），板子的 MAC 一换就不匹配，路由器会把外网挡掉（SSH 和局域网不受影响）。要外网就让用户在路由器里删掉那两条绑定。

**WSL 的限制**：`wsl.exe` 在安全黑名单，助手**只能读写 WSL 文件，不能执行命令**。所有 WSL 操作都要用户在终端里敲。

---

## 三、项目在哪

| 位置 | 路径 | 内容 |
|---|---|---|
| 板子 | `/home/orangepi/code` | C++ 主项目（编译、运行、网页都在这里） |
| WSL | `/home/imrong/traffic_count` | PC 端版本 + **模型转换脚本** |
| Windows | `D:\multi-road-crossing-vehicle-count` | 原始项目、视频、配置、ONNX |
| 校准图 | `D:\rknn_calib` | 300 张（4 个视频各 75 张） |

板子关键文件：

```
/home/orangepi/code/
├── traffic_count          # 命令行版（跑视频出结果）
├── traffic_web_single     # 网页版（浏览器实时看）
├── configs/test1~4.json   # ROI / 计数线 / 阈值
├── models/                # 见第五节
├── data/videos/           # test1~4.mp4
├── output/                # 结果视频、CSV、抓拍
└── src/                   # main.cpp / web_main_single.cpp / rknn_detector.cpp
                           # tracker.cpp / counter.cpp / web_server.cpp / config.cpp
```

---

## 四、编译和运行

```bash
cd ~/code
make traffic_count          # 命令行版
make traffic_web_single     # 网页版
```

改了 `CMakeLists.txt` 的路径后**必须**：

```bash
rm -f CMakeCache.txt && cmake . -DUSE_RKNN=ON && make
```

两个路径必须对：`TRAFFIC_PROJECT_ROOT=/home/orangepi/code`、`RKNN_ROOT=/usr`

运行：

```bash
cd ~/code && ./traffic_count test1        # 跑完退出
cd ~/code && ./traffic_web_single 8080    # 网页版，浏览器开 http://192.168.1.105:8080
```

- `output: "show"` 在板子上**必崩**（没显示器），只能用 `"save"` 或 `"none"`
- 网页版是常驻进程，**测完必须 `pkill -f traffic_web_single`**
- 启动时打印「服务已启动」**不代表成功**：端口被占时 `web_server.cpp` 会报错，但 `main.cpp` 还会无条件再打印一遍（代码 bug，别被忽悠）

---

## 五、★模型转换（转不出来先看这一节）

### 5.1 板子上现有的模型

| 文件 | 大小 | 说明 |
|---|---|---|
| `best_detrac.rknn` | 5,969,848 | **当前在用 = head4**（md5 `af1bfa9f...`） |
| `best_detrac_v12_head4.rknn` | 5,969,848 | 同一个 head4，备份 |
| `best_detrac_v7.rknn` | 6,482,288 | 旧模型（1.6.0 + head3），框偏 6.75px，**留作对照** |
| `best_detrac_old_ok.rknn` | 6,482,352 | 10-30 那份旧模型（md5 `5f2ae464...`），可用 |
| `best_detrac_fp16.rknn` | 9,730,814 | fp16 不量化，精度好但 6.6 fps |
| `best_detrac_v10_int8.rknn` | 5,169,767 | 2.5.0 普通 INT8，**坏的**（w/h 撑大） |
| `best_detrac_v11_auto.rknn` | 7,652,903 | 2.5.0 + auto_hybrid，**也是坏的** |
| `best_detrac_v6.rknn` | 6,538,889 | 坏的，分数全 0 |
| `best_detrac_int8_normal.rknn` | 5,169,344 | 坏的，和 v10 输出完全一样 |
| `best_detrac_v8_h250.rknn` | 5,171,354 | 混合量化，输出布局错乱 |
| `best_detrac_v9_h250.rknn` | 5,171,927 | 混合量化，通道语义被破坏 |

**只有前三个是好的，其中 head4 是最优。** 后面全是失败产物，留着为了不重复踩坑。

### 5.2 唯一验证通过的转换配方（v7，照这个抄）

| 项 | 值 |
|---|---|
| 工具 | **rknn-toolkit2 1.6.0+81f21f4d**（WSL，`conda activate rknn16`，Python 3.10） |
| 输入 ONNX | `best_detrac_head3.onnx`（12.77 MB，**删掉了末端 Concat**） |
| ONNX 输出名 | `/model.22/Mul_2_output_0`（boxes [1,4,8400]）+ `/model.22/Sigmoid_output_0`（scores [1,1,8400]） |
| config 参数 | `target_platform='rk3588'`、`mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]` |
| build 参数 | `do_quantization=True`，其余全默认（w8a8 / normal / optimization_level 默认 1） |
| 校准集 | `D:\rknn_calib` 300 张（`dataset_calib_head3.txt` 清单） |
| 产出 | `~/traffic_count/models/best_detrac_v7.rknn`，**6,482,288 字节** |
| 验证结果 | 20.6 fps、4.1 框/帧、test1 前 400 帧 A 线 4 辆、box0 `x1=161.8 y1=-2.7` |

### 5.3 最关键的一条：ONNX 末端绝对不能有 Split

**失败现象**：`want_float=1` 时分数全 0。

**原因**：之前把「拆双输出」做成了「在末端 Concat 后面加一个 `Split(axis=1,[4,1])`」。RKNN 编译器把输出上的 Split 融合掉，两个版本各有各的错法：

| ONNX | toolkit | 结果 |
|---|---|---|
| Concat 保留 + 末尾 Split（head2） | 1.6.0 | scores 的 8400 个 int8 **全是 -128**（= zp = 0.0） |
| 同上 | 2.5.0 | scores 正常，但 boxes 塌（float 均值 330 vs 正常的 182），框挤在左边缘 |
| **删掉 Concat，两个张量直接当输出**（head3） | 1.6.0 | ✅ 正常 |

**正确做法**（`tools/01_convert/make_head3.py` 干的事）：

```python
# 删掉末端 Concat，把它的两个输入直接当 graph output
m.graph.node.remove(concat)          # concat.name = '/model.22/Concat_3'
del m.graph.output[:]
m.graph.output.extend([
    helper.make_tensor_value_info(boxes,  TensorProto.FLOAT, [1, 4, 8400]),
    helper.make_tensor_value_info(scores, TensorProto.FLOAT, [1, 1, 8400]),
])
```

### 5.4 ★head4：框偏移的最终解法（2026-11-02 验证通过）

> **要完整搞懂这套方法，看 `docs/head4_原理详解.md`（11 节、11 张图，从「什么是量化刻度」讲起）。**
> 本节只记结论和参数，方便写代码时查。

**一句话**：把模型输出从「坐标 0~640」换成「DFL 距离 0~15」，坐标换算搬到 C++。

**为什么有效**：RKNN 给输出配刻度是 `范围/256`。原来输出是坐标（0~640）→ 刻度 2.5，
但 DFL 内部的距离只有 0~15，被这把粗尺子一压，精度全丢。
距离单独当输出后，刻度变成 `0.047`，**细了 53 倍**。

**更完整的原因**（2026-11-02 补充，读 `quantization.cfg` 实测）：
不只是「最后配了个粗刻度」，而是**中间那几步每一步都在 int8 里重新量化**。
从 DFL 到输出一共 6 步，刻度一路被撑大：

| 步骤 | 值域 | 刻度 |
|---|---|---|
| DFL 输出距离 | 0 ~ 11.97 | **0.047** |
| anchor − 左距离 | −3.47 ~ 79.46 | 0.325 |
| anchor + 右距离 | 0 ~ 83.28 | 0.327 |
| 两者相加 ÷2 | 0 ~ 159.48 | 0.625 |
| 两者相减 = 宽高 | 0 ~ 21.43 | 0.084 |
| 拼 4 通道 × stride | 0 ~ 637.92 | **2.502** |

关键在**第 2 步**：`anchor`（0.5~79.5 的大数）减去 `距离`（0~12 的小数），
结果范围被大数主导，小数的精度立刻被稀释约 7 倍。后面又乘又加，一路涨到 2.5。

**所以 head4 修的是「后面 5 次量化全都不发生了」**，不只是换个刻度。
C++ 里用 float32（24 位有效数字），根本没有「刻度」这回事。

**实测结果**（10 帧，每帧和 FP32 真值逐框对比）：

| | 左边缘 | 上边缘 | 中心偏移 |
|---|---|---|---|
| v7（旧） | −4.94 | −5.50 | **6.75 px** |
| head4（新） | +0.04 | −0.00 | **0.03 px** |

速度不变（18.3 fps），因为全是 int8，没有 fp16 层。

**改了什么**（`tools/01_convert/make_head4.py`）：
1. 删掉 DFL 之后的 9 个节点（Slice/Sub/Add/Div/Concat/Mul ×2）——**这些节点没有任何权重**
2. 把 `/model.22/dfl/Reshape_1_output_0` 直接设成图输出

**C++ 侧换算**（`rknn_detector.cpp`）：

```cpp
x1 = (anchor_x - 左距离) * stride;
y1 = (anchor_y - 上距离) * stride;
x2 = (anchor_x + 右距离) * stride;
y2 = (anchor_y + 下距离) * stride;
```

| 元素 | 规格 |
|---|---|
| anchor 网格 | 三段：80×80、40×40、20×20，值 = (列号+0.5, 行号+0.5) |
| stride | **是 8400 长的数组，不是标量！** 前 6400 个=8，中 1600 个=16，后 400 个=32 |
| 距离顺序 | 左、上、右、下 |

**自动识别**：`rknn_detector.cpp` 看坐标输出张量名里有没有 `dfl`
（head4 是 `/model.22/dfl/Reshape_1_output_0`，head3 是 `/model.22/Mul_2_output_0`）。
**所以 v7 和 head4 可以随便换，代码都能跑。**

**三道验证关**（改模型必过）：
1. ONNX 合法 + **分数输出和原模型差值必须为 0**
2. numpy 重写换算公式，和原模型 boxes 差值必须为 0（实测精确 0）
3. 板子上和 FP32 真值逐框对比

**关于 `head4` 这个名字**

`head` 指模型的**检测头**（head），即负责输出结果的最后一段网络。
YOLOv8 的检测头在 ONNX 图里节点名前缀是 `/model.22/`，官方叫 `Detect` head。

本项目为了绕开 RKNN 的量化问题，对导出结构改了三版，用编号区分：

| 名称 | 输出结构 | 改动 | 状态 |
|---|---|---|---|
| 原始 | 单输出 `[1,5,8400]` | 无（ultralytics 导出原样） | 分数全 0，不可用 |
| head2 | 双输出 | 末端加 `Split` 拆分 | 触发工具 bug，不可用 |
| head3 | 双输出 `[1,4,8400]` + `[1,1,8400]` | 删掉末端 `Concat` | 可用，但框偏 6.75 px |
| **head4** | 同上形状，内容改为距离 | 再删 DFL 之后 9 个换算节点 | **当前主力，框偏 0.03 px** |

**这三个编号（head2/3/4）是本项目自定的，不是 YOLO 或 RKNN 的官方术语。**
官方只有 `Detect` head 一个概念，没有 head3、head4 之分。

换个说法也成立：head4 就是「输出 DFL 距离而不是坐标」的那版模型。
如果名字让人困惑，直接按特性称呼即可。

**三种模型的输出形状**（容易混，记一下）：

| | 输出 | 那 4 行代表 | 值域 |
|---|---|---|---|
| 原始 ONNX | `[1,5,8400]` 单个 | cx cy w h + 分数 | 0~640 |
| head3 | `[1,4,8400]` + `[1,1,8400]` | cx cy w h | 0~640 |
| head4 | `[1,4,8400]` + `[1,1,8400]` | 左 上 右 下 距离 | 0~15 |

**head3 和 head4 形状完全一样，只有含义不同**，所以后处理必须靠张量名判断。

**为什么可以随便改输出**：ONNX 规范里 `graph.output` 就是一行声明，
任何张量都能声明成输出。`onnx.checker` 通过、onnxruntime 正常跑、
权重一个字节没改、换算公式验证差值为 0。
**代价**是后面那几步得自己写对（anchor 网格、stride 分段、距离顺序，错一个全错）。

### 5.6 怎么复现

**head4（当前主线）：**

```bash
# Windows 侧
cd D:\multi-road-crossing-vehicle-count
D:\Python3.12.4\python.exe tools\01_convert\make_head3.py
D:\Python3.12.4\python.exe tools\01_convert\make_head4.py
D:\Python3.12.4\python.exe tools\01_convert\verify_head4_decode.py   # 必须看到差值 0

# WSL 侧
conda activate rknn16
cd ~/traffic_count
python /mnt/d/multi-road-crossing-vehicle-count/tools/01_convert/convert_head4_probe.py
```

**head3（老路线，对照用）：**

```bash
conda activate rknn16
python /mnt/d/multi-road-crossing-vehicle-count/tools/03_failed/convert_head3.py
```

### 5.7 怎么验证（两步，别只看帧率）
**第一步：模型体检**（工具已经在板子项目里：`src/rknn_probe.cpp`，编出来是 `~/code/rknn_probe`）

```bash
cd ~/code
make rknn_probe        # 第一次或改了源码才编，之后直接用
./rknn_probe models/新模型.rknn data/videos/test1.mp4 200
```

它干的事：打印每个输入/输出张量的形状、类型、scale、零点，然后真跑一帧，把输出**原样摊开**（反量化后的 float + 未换算的 int8）。换新模型先跑它，别拿业务程序猜。

合格标准（第 200 帧，test1）：

| 输出 | 合格表现 |
|---|---|
| boxes float | min 0、max ≈630、**均值 ≈182** |
| scores int8 原始值 | 8400 个里约 **60 个非零、9 个 >0.5**（8000 多个 -128 是正常的，大多数 anchor 没车） |
| scores float | max ≈0.78~0.80 |

**第二步：跑 400 帧看真实结果**

```bash
sed -e 's/"name": "test1"/"name": "tmpv7"/' -e 's/"max_frames": 0/"max_frames": 400/' \
    configs/test1.json > /tmp/t1v7.json
# 换模型前先备份 models/best_detrac.rknn，跑完必须还原
cd /tmp && /home/orangepi/code/traffic_count /tmp/t1v7.json
```

合格：`box0` 的 `x1` 在 150~170、`y1` 是 -3 左右、约 4 框/帧、A 线 4 辆。

### 5.8 混合量化：两条路都试过了，都不行（2026-11-01 详细记录）

**目的**：INT8 的框偏移 6~22px 来自 DFL 量化，想让 DFL 那几层走 fp16、其余仍 int8，
既保住精度又不掉速度。

**结论：1.6.0 和 2.5.0 都走不通。别再重复。** 下面是完整过程，省得下次重新试。

#### 1.6.0：step2 必定崩

```
hybrid_quantization_step2 → quant_optimizer._p_adjust_tanh_sigmoid
KeyError: '/model.22/Concat_1_output_0'
```

工具要查的层名，和 cfg 里实际的名字对不上：

| 工具要找 | cfg 里实际是 |
|---|---|
| `/model.22/Concat_1_output_0` | `/model.22/Concat_1_output_0_shape4_/model.22/Sigmoid` |

试过三种配置：改 35 层、改 2 层、只改 `quantize_parameters` 的 dtype 不碰
`custom_quantize_layers` —— **三次崩在同一行**，说明和改什么无关，是工具自身 bug。

#### 2.5.0：step1 能过，但生成的模型不能用

`proposal=True` 崩在 `hybrid_proposal.proposal → expand_batch → infer_shapes`：
`KeyError: '/model.22/dfl/conv/Conv_output_0'`。**用 `proposal=False` 能过**，
而且本来也不需要它的自动建议（1.6.0 那次的建议全落在骨干前 7 层，对坐标精度毫无帮助）。

step2 的三个坑（按踩到顺序）：

1. **`step2` 之前不能 `load_onnx`** —— 报
   `The model already exists. Do not call 'load_onnx'...`。
   `model_input` 要传 step1 生成的 `.model` 文件，不是 ONNX。
2. **`custom_quantize_layers` 的格式** —— step1 生成的是 `custom_quantize_layers: {}`
   （空字典同一行）。直接往下追加缩进条目会导致 YAML 解析失败：
   `expected <block end>, but found '<block mapping start>'`。
   必须先把 `{}` 删掉，改成块状映射。
3. **`quantize_parameters` 是只读的** —— 改了里面任何一层的 `dtype` 就报
   `The quantize_parameters['X']['dtype'] is not allowed to be modified!`。
   fp16 层的量化参数由 step2 自己重算，用户只该动 `custom_quantize_layers`。
4. **层名必须和 cfg 里完全一致**，写错报 `Invalid operands name 'X' in custom_quantize_layers!`。
   2.5.0 的 DFL 实际只有 3 个张量，名字也和 1.6.0 不同：
   ```
   /model.22/dfl/Reshape_output_0
   /model.22/dfl/Softmax_output_0_sw_sw      ← 注意 _sw_sw 后缀
   /model.22/dfl/conv/Conv_output_0
   ```
   没有 `Transpose_1_output_0`，也没有 `Mul_2_output_0_shape4`。
5. **Reshape/Transpose/Concat/Slice 这类布局层不能写进去**（没有独立量化参数）。

#### 终于生成出来的两个模型，都不能用

| 模型 | 配置 | 结果 |
|---|---|---|
| `best_detrac_v8_h250.rknn` | DFL + `Mul_2_output_0` 都 fp16 | **输出布局错乱** |
| `best_detrac_v9_h250.rknn` | 只有 DFL 内部 fp16，输出仍 int8 | **4 个通道语义被破坏** |

**v8 的问题**：boxes 输出变 FLOAT16 后，编译器插了 NC1HWC2 布局重排
（编译日志：`116 Reshape /model.22/Mul_2_output_0-rs__float16 FLOAT16 NC1HWC2`），
元素顺序不再是标准 NCHW。实测 anchor 0 的 `cy` 应该是 5，读出来是 397。

**v9 的问题**：DFL 输出被 `exDataConvert` 从 float16 转回 int8 时，量化参数选错了。
DFL 算出来的"距离"范围是 0~15，和下游坐标（0~640, scale=2.5）共用一套参数，
0~15 被压成只剩 6 个台阶。实测 4 个通道的范围全都变成 0~638（v7 里 4 个通道范围各不相同），
说明通道语义被破坏了。

**判断依据**（下次可以直接用）：
```bash
/tmp/ch_scan 模型.rknn data/videos/test1.mp4 200
```
好的模型，4 个通道范围应该各不相同：
```
ch0(cx) 0~630 | ch1(cy) 0~615 | ch2(w) 2.5~510 | ch3(h) 2.5~230
```
如果 4 个通道范围一样（都是 0~638），就是坏了。

#### 以后想再提精度，只剩这些路

1. **更好的校准集**（最大收益，且已经做过一轮）
2. **直接上 fp16 模型**：精度和 FP32 无差，但 6.6 fps（NPU 计算 132ms），需要配跳帧
3. **QAT（量化感知训练）**：训练时把量化误差算进去，最彻底，但要重训
4. `mmse` / `kl_divergence`：只是换裁剪范围，1 像素级收益，不值得

### 5.9 试过但没用的（别再重复）

- `quantized_dtype`、`hybrid_level`、`custom_string`、`quantized_input_type` —— 全是猜的，无效
- `mmse` / `kl_divergence` —— 只是换裁剪范围，1 像素级收益；mmse 还容易 OOM
- 2.5.0 的 `disable_rules=['seperate_conv_rs_split']` —— 只对 2.5.0，1.6.0 没这个参数
- `optimization_level=3` —— 会把 DFL 坐标算飞（框巨大错位），只能用默认/1
- **rknn-toolkit2 的混合量化**（详见 5.6）—— 两个版本都有 bug

**结论：模型这块就停在 1.6.0 + head3 + INT8（v7）。**

---

## 六、已知问题

### 6.1 计数漏（2026-11-02 收尾）

**现状**：问题已处理告一段落，不再继续追查。断号这一支已证实并修复，
剩余漏计不影响交付，记录如下备查。

#### 原来的三个可疑点，逐一交代

| # | 可疑点 | 结论 |
|---|---|---|
| 1 | `counter.cpp` 的 `near_segment` 不通过就 `continue`，连 `prev_` 状态都不更新 | **未查证**。理论上转弯车轨迹偏出线段端点会判不到，但这只是推测，没有实测数据支撑 |
| 2 | `count_dir: "neg"` 只数一个方向 | ❌ **不是问题**。这几个路口都是**单行道**，单方向计数是设计意图。除非出现逆行车辆，否则不会有漏计 |
| 3 | 跟踪断号：转弯时 IoU 匹配容易掉，新 ID 没有「上次在哪一侧」的记录 | ✅ **已证实并修复**，见下 |

> **第 2 条不要再当成缺陷去查。** 单行道场景下它是对的，
> 改动它反而会把逆行误判成正常车流。

#### 实测：框偏移修复后，断号减少 34.3%

2026-11-02 用 `multi_eval`（板端多路并行评测）跑完整视频，两模型同配置对比：

| 视频 | 帧数 | v7 编号数 | head4 编号数 | 降幅 |
|---|---|---|---|---|
| test1 | 3600 | 553 | **450** | 18.6% |
| test2 | 3600 | 625 | **441** | 29.4% |
| test3 | 2535 | 909 | **539** | 40.7% |
| test4 | 2400 | 472 | **250** | 47.0% |
| **合计** | 12135 | **2559** | **1680** | **34.3%** |

（编号数 = `Tracker::total_ids()`，累计分配过多少编号。同一视频同样帧数下，
编号越少说明断号越少。）

计数变化：净增 **+12**（test1 A +2、test3 D +8、test4 A/B 各 +1），**没有一条线下降**。

#### 结论

**断号减少 34.3%，四个视频无一例外，越密的场景降得越多。**
这支持因果链：**框偏移 → 框抖 → IoU 匹配失败 → 断号 → 漏计**。

但计数只多 12，且 test2 的 A/B 两条线完全没变。
说明框偏移只解释了约三分之一的漏计。

**剩下的漏计暂不深究** —— 视频里过车基数本来就小（5 分钟几十辆），
计数差异的统计意义有限；再看下去投入产出比不高。

#### 复现方法

```bash
# 板子上，四路并行，各跑 3600 帧（test3/test4 视频本身更短，跑完即止）
cd ~/code
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --out output/full_head4
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --model models/best_detrac_v7.rknn --out output/full_v7

# 把两个 summary.csv 拉回 PC 后对比
python tools/02_diagnose/compare_models.py full_v7_summary.csv full_head4_summary.csv
```

实测速度：4 路并行合计 **62~73 fps**（单路 18~22 fps），12135 帧约 3 分钟跑完。

### 6.2 框偏移（2026-11-02 **已解决**）

**现在的状态**：中心偏移 6.75px → **0.03px**，速度不变。解法见 **5.4 节**，
完整原理（带图）见 `docs/head4_原理详解.md`。

下面是当时的诊断过程，留着备查。

**现象**：板端框整体向左上偏，**车越小（越远）偏得越厉害**。截图见 `docs/screenshots/int8_shift_*.png`。

**真正的根因**（2026-11-02 查明，此前一直猜错）：
RKNN 给模型输出配量化刻度用的是 `刻度 = 输出范围 / 256`。
原来模型输出的是**坐标（0~640）**，所以刻度 = 640/256 = **2.5**。
但 DFL 内部那个「距离」只有 0~15，被这把粗尺子压得只剩几个台阶 → 精度全丢。

**决定性证据**（同一帧、同一个车，anchor #2180）：

| 模型 | cx | cy | **w** | **h** |
|---|---|---|---|---|
| v7（正确） | 160.1 | 215.1 | **20.0** | **20.0** |
| v10（2.5.0 坏的） | 120.1 | 195.1 | **235.2** | **260.2** |

w/h 被撑大 **13 倍**——所以「框偏左上」其实是「框被撑大后中心偏移」。
另外 v10 里 w/h 通道的最大值正好是 **637.92 = 255 × 2.5**（int8 的饱和值），
说明距离值超出了刻度能表示的范围，被截断了。

**4 个通道范围对比**（好模型四路各不相同，坏模型四路一样）：

```
v7  : ch0 0~630 | ch1 0~615 | ch2 2.5~510 | ch3 2.5~230   ← 正常
v10 : ch0 0~638 | ch1 0~638 | ch2 10~638  | ch3 10~638    ← 坏了
v11 : 四路全是 0~638                                        ← 坏得更彻底
```

**当时猜错的地方**（教训）：
一直以为是「DFL 的 16 个 bin 精度不够」，于是去折腾混合量化。
实际是**刻度配错了范围**——不是精度不够，是精度被浪费了。
方向错了，所以怎么调都没用。

**已排除的怀疑**（都验证过）：
- 不是 letterbox 补边算错（ONNX 侧同样算法结果正常）
- 不是反量化公式错（`(q-zp)*scale`，zp=-128 验证过）
- 不是 `want_float` 的问题（探针实测反量化后的值是对的）
- 不是模型权重问题（旧模型、v7 模型偏移量一致）

**其他方案的现状**：

| 方案 | 精度 | 速度 | 结论 |
|---|---|---|---|
| **head4（现在用的）** | **偏 0.03px** | 18.3 fps | **✅ 已采用** |
| v7（旧） | 偏 6.75px | 18.3 fps | 留作对照 |
| 纯 FP16 | 与 FP32 无差 | 6.6 fps | 太慢，没必要了 |
| 混合量化 | — | — | 两个 toolkit 版本都失败（5.8） |
| 后处理补偿 | 治标 | 不变 | 不需要了 |


### 6.3 框抖

已删掉重复的 EMA（见下），只保留卡尔曼一层平滑。

2026-11-01 改动：**删掉了 tracker.cpp 里的 EMA 平滑**（`0.6×旧 + 0.4×新`）。它和卡尔曼重复平滑——卡尔曼那层滤波结果才是真正该用的，输出框现在直接取 `kf.state()`。实测计数和框位置完全不变（400 帧 A=6 前后一致），说明 EMA 只是白加 1~2 帧滞后。

### 6.4 其他

- 网页端口被占时仍打印「服务已启动」：`main.cpp` 无条件打印

### 6.5 已修：prev_ 只增不删

`Counter::prev_` 原来只增不删，接摄像头连跑几天会攒到几十万条。2026-11-01 加了 `retain(live_ids)`：每帧问跟踪器"现在还有谁活着"，不在名单上的清掉。跟踪器删 id 的时机是它自己定的（`track_buffer_=30` 帧），计数器不自己数帧，两边不会错位。

---

## 六点五、性能实测数据（2026-11-02）

三组测试，覆盖三种用法：批量多视频、单视频流水线、模拟实时流。

### ① 批量多视频并行（`multi_eval`）

每路一个线程，按 `i % 3` 绑 NPU 核。不画图不存视频。

| 线路数 | 合计吞吐 | 相对单路 |
|---|---|---|
| 1 | 17.3 fps | 1.00× |
| 2 | 37.9 fps | 2.19× |
| 3 | 58.3 fps | 3.37× |
| 4 | 71.2 fps | 4.12× |

**线性扩展。** 12135 帧四路并行约 3 分钟跑完（单路串行要 11 分钟）。

### ② 单视频流水线（`pipeline_bench`）

三段流水：读帧线程 → 推理线程（K 个，各绑一核）→ 跟踪线程（单线程串行）。
帧号作钥匙，按序重组后再送跟踪。

| 推理线程数 | 吞吐 | 相对单线程 |
|---|---|---|
| 1 | 18.4 fps | 1.00× |
| 2 | 41.7 fps | 2.27× |
| 3 | 65.9 fps | 3.58× |
| 4 | 85.0 fps | 4.62× |
| 6 | 105.5 fps | 5.73× |
| 8 | 113.2 fps | 6.15× |
| 10 | **118~121 fps** | **6.6×** |
| 12 | 117.0 fps | 6.36× |
| 16 | 120.1 fps | 6.53× |

**拐点在 10 线程。** 超过 10 之后不再上升，因为 3 个核已经吃满。

### ③ 跟踪到底占多少

| 线程数 | 只检测 | 检测+跟踪 | 差异 |
|---|---|---|---|
| 3 | 70.7 fps | 66.7 fps | -5.7% |
| 6 | 107.3 fps | 107.6 fps | 0 |
| 10 | 117.4 fps | 116.9 fps | 0 |

**跟踪几乎不占时间。** 因为跟踪约 2~3ms/帧，只有 NPU 的 5%，
而且跑在独立线程里，被 NPU 的时间完全掩盖。

**结论：不需要把跟踪挪到后面批量做。「边跑边追」没有额外成本。**

### ④ 延迟拆解（为什么大数字会骗人）

| 线程数 | 纯推理 | 排队等待 | 端到端 |
|---|---|---|---|
| 1 | 52.5 ms | 1670 ms | 1725 ms |
| 3 | 49.8 ms | 501 ms | 553 ms |
| 10 | 118.7 ms | 296 ms | 418 ms |

**端到端那 1.7 秒里，1.67 秒是在队列里排队，真正算一帧只要 52ms。**

排队时间 = 队列长度 × 单帧耗时（32 帧 × 52ms ≈ 1.66 秒，与实测吻合）。

### ⑤ MP4 模拟实时流（`--realtime N`）

让读帧线程按指定 fps 节奏送帧，等价于摄像头推流。
判据看**队列占用率**，不看吞吐（送帧速度受节奏限制，吞吐必然约等于送帧速度）。

| 送帧速度 | 1 线程 | 3 线程 | 6 线程 | 8 线程 |
|---|---|---|---|---|
| 15 fps | ✅ 65ms | ✅ 62ms | — | — |
| 30 fps | ❌ 队列 100% | ✅ **47ms** | — | — |
| 45 fps | ❌ | ✅ 47ms | — | — |
| 60 fps | ❌ | ✅ 47ms（队列10%） | — | — |
| 90 fps | ❌ | ❌ | ✅ 62ms | ✅ 68ms |
| 120 fps | ❌ | ❌ | — | ✅ 146ms |

**关键结论：**

- **单线程只能扛 15fps 的流**（单核推理 20fps，30fps 就积压）
- **3 线程能扛 60fps 的流，延迟仅 47ms**
- 8 线程能扛 120fps，但延迟涨到 146ms

**所以接摄像头确实必须开多线程。** 2 线程就够 30fps，3 线程有余量。

### ⑥ 实测的硬件上限

```
单帧推理（单核）:  42~52 ms  →  单核 20~24 fps
3 核理论极限:      3 × 24 = 72 fps（无竞争时）
流水线实测峰值:    120 fps（10 线程，靠队列深度掩盖延迟）
```

**120 fps 已接近这块板子的物理上限**，再堆线程无收益。

### ⑦ 用这些数据的场景建议

| 场景 | 配置 | 预期 |
|---|---|---|
| 批量跑多个 mp4 | `multi_eval`，4 路并行 | 71 fps，约 3 分钟/12135 帧 |
| 单个 mp4 要快 | `pipeline_bench --workers 10 --buf 64` | 118 fps |
| 接 30fps 摄像头 | `--workers 3 --buf 30 --realtime 30` | 47ms 延迟，跟得上 |
| 接 60fps 摄像头 | `--workers 3 --buf 30 --realtime 60` | 47ms，队列 10% |

> `--buf` 是队列上限。批处理时设大（64）提高吞吐；实时流设小（30 或更小）压低延迟。

### ⑧ GPU 情况

`/dev/mali*`、`g18p0` 驱动、`fb000000.gpu` 调频节点都在，是 **Mali-G610 MP4，最高 1.0 GHz**。

**但当前项目完全没用它。** 推理在 NPU，跟踪/计数/画图在 CPU。

值得用的只有一处：存视频时用硬件编码器（rkmpp），能省 CPU。但目前不做视频输出。

**不能用来训练**：Mali 没有可用的 PyTorch 后端，且 6GB 共享内存、25GB/s 带宽都不够。

---


## 七、踩的坑（都是真踩的）

1. **`output: "show"` 必崩**（没显示器）
2. **改 `CMakeLists.txt` 路径后不删 `CMakeCache.txt`** → 一直用旧路径
3. **CPU 调频**：默认 `ondemand`，实测同一段 400 帧 **16.0 fps**；设 `performance` 后 **20.6 fps**。开关在
   ```bash
   /sys/devices/system/cpu/cpufreq/policy{0,4,6}/scaling_governor
   ```
   重启失效。用户倾向 ondemand（觉得够流畅），跑性能测试时要切 performance 才公平。
4. **板子重启后 `/tmp` 清空**，探针、临时脚本都要重传
5. **WSL 文件从 Windows 侧能读，覆盖写不可靠**（新建有时成功，覆盖/Edit 常失败）→ 脚本放 D 盘，让 WSL 通过 `/mnt/d/...` 读
6. **网页参数只存内存**，不改 json，重启即恢复
7. **`pkill -f traffic_web_single` 会杀掉自己那条命令**（pattern 匹配到 bash -c），
   要用 `pkill -f "[t]raffic_web_single"`。用 `pkill -x` 也不行，进程名被截断成 15 字符。
8. **在错误的工作目录下跑 rknn 脚本**：step1 生成的文件会落在当前目录。
   在 `~/traffic_count` 下跑才对；之前误在 `/mnt/c/Windows/System32` 跑，
   文件全掉到 System32 里了。
9. **rknn-toolkit2 的 hybrid 打印信息是编译后打包的**，`quant_optimizer` 等模块只有 `.so`，
   看不到源码。出问题时只能靠报错原文反推规则。
10. **`rknn_probe` 的 `head` 只打印前 12 个值**，看不出是哪个通道坏了。
    要判断通道语义，得用 `tools/02_diagnose/ch_scan.cpp`（每通道独立统计 min/max）。
11. **DFL 的 stride 是 8400 长的数组，不是标量**。
    `Constant_15` 里前 6400 个是 8、中 1600 个是 16、后 400 个是 32。
    当成标量 8 算，后两段全错（实测最大差 453 像素）。
12. **anchor 网格也要分段生成**：80×80 / 40×40 / 20×20，值 = (列号+0.5, 行号+0.5)。
    直接线性排 0.5~79.5 是错的。
13. **MSYS 的 bash 会把 `D:\path` 改写成 `/d/path`**，传给 Windows 的 Python 就找不到文件。
    要在 ops_bash 里调 Windows Python 且带路径参数，注意这一点。
14. **`\\wsl.localhost\` 的 UNC 路径要用正斜杠**（`//wsl.localhost/...`），
    反斜杠在 Python/paramiko 里会报 `FileNotFoundError`。
15. **脚本挪目录后 ROOT 会算错**。这些脚本用 `dirname(dirname(dirname(abspath(__file__))))` 往上退三层算项目根，
    现在它们在 `tools/01_convert/` 这类子目录下才对。再挪目录要重新检查。

---

## 八、待办（下一个 agent 从这里开始）

### 当前状态一句话

板子跑 `best_detrac.rknn = head4`（1.6.0 + head4 + INT8），18.3 fps，
**框偏移已解决（0.03px），计数漏已收尾（断号降 34.3%）**。
模型转换这条路已经走到头，不要再碰。

### 仓库提交状态（2026-11-02 收工时的状态）

| 仓库 | 提交 | 推送 |
|---|---|---|
| `D:\multi-road-crossing-vehicle-count` | `a4f3e48`「修复框偏移」 | ✅ 已推送，与 origin 同步 |
| WSL `~/traffic_count` | `9eff195`「修复框偏移」 | ❌ **未推送**，需手动执行下面这条 |

```bash
cd ~/traffic_count && git push --force-with-lease
```

> 为什么是 force：WSL 那边 amend 过提交，本地和远端分叉了（ahead 1 / behind 1）。
> 本地代码是全的，推送只是把远端同步过来，**不影响任何功能**，什么时候推都行。
> 走代理时 WSL 里 `git push` 会卡死，试不通就直接跳过，不要反复重试。

### tools 目录结构（2026-11-02 整理过）

```
tools/
├── 01_convert/     转换主线 —— 要转模型只看这里
├── 02_diagnose/    诊断验证 —— 判断模型好坏
├── 03_failed/      失败记录 —— 别再跑
├── 04_dataset/     数据集相关
└── 05_figures/     画讲解图
```

详细说明见 `tools/README.md`。

### 当前状态：没有待办

框偏移已解决，计数漏已收尾，本轮工作结束。

如果以后要继续提升，按投入产出比排序：

1. **`near_segment` 那条（未查证）** —— 唯一还没验证过的怀疑点。
   做法：给 `counter.cpp` 加轨迹日志（每帧每车：id、中心、在线的哪一侧、
   是否 near_segment），跑完整 test1 和 CSV 对账。
2. **扩大统计基数** —— 现在 5 分钟视频只有几十辆车，计数差异统计意义有限。
   跑完整 15 分钟视频（`--frames 0`）、或多跑几个场景，结论会更稳。
3. **板端多路实时跑、测功耗** —— 部署收尾工作，与算法无关。

> **不要做的事**：`count_dir` 只数一个方向不是缺陷（单行道场景），
> 模型转换这条路也走到头了（见速览的「已经堵死的路」）。


### 已验证可用的工具

**板子 `~/code` 下（永久）：**

| 工具 | 用途 |
|---|---|
| `./rknn_probe 模型 视频 帧号` | 模型体检：打印张量属性 + int8 原始值 + 反量化值 |
| `./traffic_count 配置` | 命令行跑视频（画图存视频，单路） |
| `./multi_eval 配置... --frames N` | **多路并行评测**，不画图不存视频，最快；支持 `--model` 换模型对比 |
| `./traffic_web_single 8080` | 网页实时看（`pkill -f "[t]raffic_web_single"` 关掉） |

**`multi_eval` 用法**（2026-11-02 新增）：

```bash
cd ~/code
# 四路并行，各跑前 3600 帧
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --out output/full_head4

# 换模型对比（同一批数据）
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --model models/best_detrac_v7.rknn --out output/full_v7
```

每个线程创建独立的 `RknnDetector`，按 `i % 3` 绑定到 0/1/2 号 NPU 核。
实测 4 路并行合计 62~73 fps（单路 18~22 fps）。

**`tools/02_diagnose/` 下（源码在 D 盘，要用时传到板子编译）：**

| 工具 | 用途 |
|---|---|
| `ch_scan.cpp` | 打印 boxes 四个通道各自的范围，判断模型有没有坏 |
| `box3.cpp` | 两个模型 + FP32 真值三方对比，逐框算误差 |
| `box_cmp.cpp` | 两个模型互相配对比较（不需要真值） |
| `batch_box_eval.py` | 批量跑多帧，汇总平均误差（调 box3） |
| `dump_truth_boxes.py` | PC 上用 FP32 ONNX 生成真值文件 |
| `trace_layers.py` | 逐层 dump 量化误差，查哪层崩了 |
| `compare_models.py` | **对比两个模型的评测结果**（断号数 + 各线计数），读 multi_eval 的 CSV |
| `show_cfg_scale.py` | 从 quantization.cfg 提每层的 min/max/scale |

板子重启后 `/tmp` 会清空，这些工具要重传重编：

```bash
# 在板子上编译（先把源码传到 /tmp）
g++ -O2 -std=c++17 -I~/code/src /tmp/ch_scan.cpp -o /tmp/ch_scan \
    -lrknnrt $(pkg-config --cflags --libs opencv4)
```

源码在 `D:\multi-road-crossing-vehicle-count\tools\` 对应的 `.cpp` 里
（ch_scan / box_check 需要时我可以重新给）。

