# 项目交接文档（车流量统计 · RK3588 板子端）

> 最后更新：2026-11-01。**先看完这一份再动手。**
> 同目录 `2026-10-2x.md` 是当天流水，里面的结论可能有错；**与本文冲突时以本文为准**。

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
| `best_detrac_fp16.rknn` | 9,730,814 | fp16，不量化、更准但慢一倍 |
| `best_detrac_v6.rknn` | 6,538,889 | **坏的，分数全 0，别用** |

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

### 5.6 试过但没用的（别再重复）

- `quantized_dtype`、`hybrid_level`、`custom_string`、`quantized_input_type` —— 全是猜的，无效
- `mmse` / `kl_divergence` —— 只是换裁剪范围，1 像素级收益；mmse 还容易 OOM
- 2.5.0 的 `disable_rules=['seperate_conv_rs_split']` —— 只对 2.5.0，1.6.0 没这个参数
- `optimization_level=3` —— 会把 DFL 坐标算飞（框巨大错位），只能用默认/1

**结论：先用 1.6.0 + head3。** 想再提精度，按这个顺序试：更好的校准集 → 混合量化（部分层 fp16）→ 直接上 fp16 模型 → QAT。

---

## 六、已知问题

### 6.1 计数漏（主要矛盾，未解决）

转弯车经常漏计。已读代码，可疑点：

1. `counter.cpp` 第 14 行：`near_segment` 不通过就 `continue`，**连 prev_ 状态都不更新**；转弯车轨迹偏出线段端点后，过线要么判不到，要么位置错
2. `count_dir: "neg"` 只数一个方向，反方向穿线的转弯车直接被过滤
3. 跟踪断号：转弯时 IoU 匹配容易掉，新 ID 没有「上次在哪一侧」的记录 → 漏

### 6.2 框抖

INT8 量化网格 2.5px（640 空间）。

2026-11-01 改动：**删掉了 tracker.cpp 里的 EMA 平滑**（`0.6×旧 + 0.4×新`）。它和卡尔曼重复平滑——卡尔曼那层滤波结果才是真正该用的，输出框现在直接取 `kf.state()`。实测计数和框位置完全不变（400 帧 A=6 前后一致），说明 EMA 只是白加 1~2 帧滞后。

### 6.3 其他

- 框轻微向左上偏 1~2px：INT8 固有
- 网页端口被占时仍打印「服务已启动」：`main.cpp` 无条件打印

### 6.4 已修：prev_ 只增不删

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

---

## 八、待办（按优先级）

1. **修计数漏**：先给 `counter.cpp` 加轨迹日志（每帧每车：id、中心、在线的哪一侧、是否 near_segment），跑完整 test1，和 CSV 对账，看漏在「线段范围」「方向过滤」还是「断号」
2. **按 5.5 的流程**把 fp16 模型跑同一段 400 帧，量化影响到底多大
3. 想让框更稳 → 混合量化或 QAT
4. `docs/设计要点.md` 的清单：跳帧检测、ID 重连、轨迹日志、人工真值对比
