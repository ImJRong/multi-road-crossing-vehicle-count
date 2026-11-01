# 模型转换与诊断工具

## 主线（能用）

| 脚本 | 用途 |
|---|---|
| `make_head3.py` | 从 `best_detrac.onnx` 删掉末端 Concat，生成 `best_detrac_head3.onnx`（唯一能转出好模型的输入） |
| `convert_head3.py` | 用 rknn-toolkit2 **1.6.0** 转 INT8 → 产出当前在用的 v7 模型 |

跑法（WSL）：
```bash
conda activate rknn16
python /mnt/d/multi-road-crossing-vehicle-count/tools/make_head3.py
python /mnt/d/multi-road-crossing-vehicle-count/tools/convert_head3.py
```

## 混合量化（试过，失败，保留备查）

这一组脚本**产出的模型都不能用**，留在这里是因为：
1. 记录了完整踩坑过程，避免下次重来
2. 万一以后 rknn-toolkit2 修了 bug，这套流程可以直接复用

| 脚本 | 状态 |
|---|---|
| `hybrid_250.py` | 2.5.0 的 step1，**能跑通**（`proposal=False`） |
| `fix_cfg_yaml.py` | 修 cfg 的 YAML 结构 + 标记 DFL 为 fp16，**能跑通** |
| `hybrid_250_step2.py` | 2.5.0 的 step2，**能生成模型，但模型输出是坏的** |
| `convert_head3_hybrid.py` | 1.6.0 版，step2 必定崩（`KeyError: Concat_1_output_0`），废弃 |
| `fix_hybrid_cfg.py` | 早期尝试，被 `fix_cfg_yaml.py` 取代，废弃 |
| `patch_dtype_fp16.py` | 早期尝试（直接改 dtype），会报 `not allowed to be modified`，废弃 |
| `hybrid_step2.py` | 1.6.0 版 step2，废弃 |

失败原因详见 `MEMORY.md` 第 5.6 节。

## 其他

| 脚本 | 用途 |
|---|---|
| `verify_norm.py` | 校验 mean/std 归一化是否生效 |
| `build_rknn*.py`、`convert_v5.sh` | 10-30 那几轮的转换尝试，历史遗留 |
| `eval_int8_precision.py`（在 WSL） | INT8 量化精度评估，1510 张验证集 |

## 板子上的诊断工具

`src/rknn_probe.cpp` 已经编进项目（`make rknn_probe`），是最常用的一个。

`/tmp` 下临时用过的（重启会丢）：

| 工具 | 用途 |
|---|---|
| `ch_scan` | 打印 boxes 4 个通道各自范围，判断模型输出有没有坏 |
| `box_check` | 打印检测框，可带人工真值对比 |
| `layout_test` | 同时按 NCHW/NHWC 解析，判断输出布局 |
| `cmp_out` | 两个模型原始输出逐值对比 |

**判断模型好坏最快的方法**：`./rknn_probe 模型 视频 帧号`，看 scores 的 `==zp(-128)`
是不是约 8340。8400 就是分数全 0，模型坏了。
