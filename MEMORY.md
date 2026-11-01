# 项目交接文档（车流量统计 · RK3588 板子端）

> 最后更新：2026-11-01 夜。**先看完这一份再动手。**
> 同目录 `2026-10-2x.md` 是当天流水，里面的结论可能有错；**与本文冲突时以本文为准**。

---

## 〇、30 秒速览（赶时间只看这段）

**项目**：YOLOv8 车流量检测跑在 Orange Pi 5 Plus（RK3588）NPU 上，带网页实时看。

**板子**：`ssh orangepi@192.168.1.105`（静态 IP，密码 orangepi），代码在 `~/code`。

**现在什么能用**：推理 + 跟踪 + 计数 + 网页，20.6 fps，跑 `models/best_detrac.rknn`（= v7）。

**两个没解决的问题**：
1. **计数漏**（主要矛盾）——转弯车经常漏计
2. **框偏左上 6~22px**——车越小偏得越多

**已经堵死的路（别再试）**：
- rknn-toolkit2 的**混合量化**：1.6.0 和 2.5.0 都有 bug，生成出来的模型输出是乱的（详见 5.6）
- `mmse` / `kl_divergence` / `hybrid_level` 这些量化参数：1 像素级收益，没用

**下一步建议**：先修计数漏（和框偏移解耦），详见第八节。

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
| `best_detrac.rknn` | 6,482,288 | **当前在用 = v7**（md5 `1c925d65c06925e317dee6c264e8f9ab`） |
| `best_detrac_v7.rknn` | 6,482,288 | 同一个 v7，备份 |
| `best_detrac_old_ok.rknn` | 6,482,352 | 10-30 那份旧模型（md5 `5f2ae464...`），可用 |
| `best_detrac_fp16.rknn` | 9,730,814 | fp16 不量化，精度好但 6.6 fps。**单输出 `[1,5,8400]`** |
| `best_detrac_v6.rknn` | 6,538,889 | 坏的，分数全 0，别用 |
| `best_detrac_int8_normal.rknn` | 5,169,344 | 2.5.0 转的，分数正常但框塌，别用 |
| `best_detrac_v8_h250.rknn` | 5,171,354 | 混合量化，**输出布局错乱**，别用 |
| `best_detrac_v9_h250.rknn` | 5,169,127 | 混合量化，**通道语义被破坏**，别用 |

v6 / int8_normal / v8 / v9 这四个都是失败产物，留着只是为了不重复踩坑。确认没用可以删。

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

**正确做法**（`tools/make_head3.py` 干的事）：

```python
# 删掉末端 Concat，把它的两个输入直接当 graph output
m.graph.node.remove(concat)          # concat.name = '/model.22/Concat_3'
del m.graph.output[:]
m.graph.output.extend([
    helper.make_tensor_value_info(boxes,  TensorProto.FLOAT, [1, 4, 8400]),
    helper.make_tensor_value_info(scores, TensorProto.FLOAT, [1, 1, 8400]),
])
```

### 5.4 怎么复现

```bash
# WSL
conda activate rknn16
python /mnt/d/multi-road-crossing-vehicle-count/tools/make_head3.py     # 生成+校验 ONNX（拼回差值必须 0）
python /mnt/d/multi-road-crossing-vehicle-count/tools/convert_head3.py  # 转 rknn
```

### 5.5 怎么验证（两步，别只看帧率）

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

### 5.6 混合量化：两条路都试过了，都不行（2026-11-01 详细记录）

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

### 5.7 试过但没用的（别再重复）

- `quantized_dtype`、`hybrid_level`、`custom_string`、`quantized_input_type` —— 全是猜的，无效
- `mmse` / `kl_divergence` —— 只是换裁剪范围，1 像素级收益；mmse 还容易 OOM
- 2.5.0 的 `disable_rules=['seperate_conv_rs_split']` —— 只对 2.5.0，1.6.0 没这个参数
- `optimization_level=3` —— 会把 DFL 坐标算飞（框巨大错位），只能用默认/1
- **rknn-toolkit2 的混合量化**（详见 5.6）—— 两个版本都有 bug

**结论：模型这块就停在 1.6.0 + head3 + INT8（v7）。**

---

## 六、已知问题

### 6.1 计数漏（主要矛盾，未解决）

转弯车经常漏计。已读代码，可疑点：

1. `counter.cpp` 第 14 行：`near_segment` 不通过就 `continue`，**连 prev_ 状态都不更新**；转弯车轨迹偏出线段端点后，过线要么判不到，要么位置错
2. `count_dir: "neg"` 只数一个方向，反方向穿线的转弯车直接被过滤
3. 跟踪断号：转弯时 IoU 匹配容易掉，新 ID 没有「上次在哪一侧」的记录 → 漏

### 6.2 框偏移（2026-11-01 查明，未解决）

**现象**：板端框整体向左上偏，**车越小（越远）偏得越厉害**。截图见 `docs/screenshots/int8_shift_*.png`。

**实测量化**（test1 第 200 帧，同一帧逐框对比 FP32 ONNX 与 INT8 RKNN）：

| 框 | ONNX FP32 (640空间) | RKNN INT8 | 偏差（折算原图） |
|---|---|---|---|
| 1 | cx=145.8 cy=262.4 w=28.5 h=27.2 | cx=142.6 cy=255.2 w=30.0 h=27.5 | 左 6.4px，上 14.5px |
| 2 | cx=337 cy=147 | cx=320 cy=128 | 左 17px，上 22px |
| 3 | cx=311 cy=187 | cx=300 cy=168 | 左 10px，上 20px |

**根因**：`boxes` 张量 `scale=2.5017`，一格 = 2.5 像素（640 空间），折原图 5 像素。实测 RKNN 输出除以 scale 全是精确整数（57.00、102.00…），证实坐标被吸附到网格。而 YOLOv8 的坐标是 **DFL 加权平均**出来的（58 组 Sigmoid+Mul，16 个 bin），每个 bin 被量化后误差在加权时累加，小框的相对误差更大 —— 这解释了"越远越偏"。

**已排除**：
- 不是 letterbox 补边算错（ONNX 侧同样算法结果正常）
- 不是反量化公式错（`(q-zp)*scale`，zp=-128 验证过）
- 不是模型本身的问题（旧模型、v7 模型偏移量一致）

**方案评估**（混合量化已于 2026-11-01 试完并失败，详见 5.6）：

| 方案 | 精度 | 速度 | 结论 |
|---|---|---|---|
| 现用 INT8（v7） | 偏 6~22px | 20.6 fps（performance） | **当前在用** |
| 纯 FP16 | 与 FP32 无差（用户实测） | **6.6 fps**（NPU 计算 132ms） | 太慢，可配跳帧 |
| 混合量化 | — | — | **两个 toolkit 版本都失败** |
| 后处理补偿 | 治标 | 不变 | 换机位要重新拟合 |

**已排除的怀疑**（都验证过）：
- 不是 letterbox 补边算错（ONNX 侧同样算法结果正常）
- 不是反量化公式错（`(q-zp)*scale`，zp=-128 验证过）
- 不是 `want_float` 的问题（探针实测反量化后的值是对的，min 0 / max 638 / mean 180）
- 不是模型权重问题（旧模型、v7 模型偏移量一致）

**DFL 为什么是元凶**：YOLOv8 的框距离是 16 个 bin 的概率分布加权求出来的。
`Softmax` 输出的概率每个 bin 平均只有 1/16 ≈ 0.0625，而量化 scale 是 1/255 ≈ 0.0039，
也就是**一个 bin 只剩 16 个台阶**。16 个数加权再乘 stride（8/16/32），
误差直接放大。框越小（远处车），同样的绝对误差占比越大 —— 这就是"越远越偏"。

### 6.3 框抖

已删掉重复的 EMA（见下），只保留卡尔曼一层平滑。

2026-11-01 改动：**删掉了 tracker.cpp 里的 EMA 平滑**（`0.6×旧 + 0.4×新`）。它和卡尔曼重复平滑——卡尔曼那层滤波结果才是真正该用的，输出框现在直接取 `kf.state()`。实测计数和框位置完全不变（400 帧 A=6 前后一致），说明 EMA 只是白加 1~2 帧滞后。

### 6.4 其他

- 网页端口被占时仍打印「服务已启动」：`main.cpp` 无条件打印

### 6.5 已修：prev_ 只增不删

`Counter::prev_` 原来只增不删，接摄像头连跑几天会攒到几十万条。2026-11-01 加了 `retain(live_ids)`：每帧问跟踪器"现在还有谁活着"，不在名单上的清掉。跟踪器删 id 的时机是它自己定的（`track_buffer_=30` 帧），计数器不自己数帧，两边不会错位。

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

---

## 八、待办（明天下一个 agent 从这里开始）

### 当前状态一句话

板子跑 `best_detrac.rknn = v7`（1.6.0 + head3 + INT8），20.6 fps，框偏左上 6~22px，
计数漏未修。模型转换这条路已经走到头（混合量化两个版本都失败），**不要再碰**。

### 三个候选方向（按我的建议排序）

**C. 先修计数漏（推荐先做）**
这是项目最初的主要目标，且和框偏移解耦。框偏 6~22px 对"过没过线"的判断影响有限。
做法：给 `counter.cpp` 加轨迹日志（每帧每车：id、中心、在线的哪一侧、是否 near_segment），
跑完整 test1，和 CSV 对账，看漏在「线段范围」「方向过滤」还是「断号」。

**A. 框偏移做后处理补偿**
偏移是系统性的（固定向左上、小框偏得多）。按框面积拟合一个补偿公式，能压掉大部分。
风险：换机位要重新拟合，属于治标。

**B. FP16 + 跳帧**
FP16 精度和 FP32 无差，但 6.6 fps。视频源 12fps，每 2 帧推理 1 帧 → 等效 13fps。
风险：跟踪更新频率减半，转弯时更容易断号。

### 已验证可用的工具（都在板子 `~/code` 下）

| 工具 | 用途 |
|---|---|
| `./rknn_probe 模型 视频 帧号` | 模型体检：打印张量属性 + int8 原始值 + 反量化值 |
| `./traffic_count 配置` | 命令行跑视频 |
| `./traffic_web_single 8080` | 网页实时看 |

`/tmp` 下还有几个临时诊断工具（重启会丢，需要时重传）：

| 工具 | 用途 |
|---|---|
| `/tmp/ch_scan` | 打印 boxes 4 个通道各自的范围，判断模型输出有没有坏 |
| `/tmp/box_check 模型 视频 帧号` | 打印检测框，可带人工真值对比 |
| `/tmp/layout_test` | 同时按 NCHW/NHWC 解析，判断输出布局 |
| `/tmp/cmp_out A B 视频 帧号` | 两个模型原始输出逐值对比 |

源码在 `D:\multi-road-crossing-vehicle-count\tools\` 对应的 `.cpp` 里
（ch_scan / box_check 需要时我可以重新给）。

