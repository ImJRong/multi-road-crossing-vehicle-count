# tools 目录

模型转换和阈值标定用的脚本。

## 01_convert —— 模型转换

按顺序跑这 4 个，从原始 ONNX 得到可用的板端 RKNN 模型。

| 脚本 | 作用 | 跑在哪 |
|---|---|---|
| `make_head3.py` | 从 `best_detrac.onnx` 删掉末端 Concat → `head3.onnx` | Windows |
| `make_head4.py` | 从 head3 再删掉 DFL 之后的 9 个坐标换算节点 → `head4.onnx` | Windows |
| `verify_head4_decode.py` | 验证「距离→坐标」换算公式，差值必须是 0 | Windows |
| `convert_head4_probe.py` | 用 rknn-toolkit2 1.6.0 转 RKNN | WSL |

```bash
# Windows 侧
cd D:\multi-road-crossing-vehicle-count
D:\Python3.12.4\python.exe tools\01_convert\make_head3.py
D:\Python3.12.4\python.exe tools\01_convert\make_head4.py
D:\Python3.12.4\python.exe tools\01_convert\verify_head4_decode.py
# 必须看到「最大差值 = 0」才算过

# WSL 侧
conda activate rknn16
cd ~/traffic_count
python /mnt/d/multi-road-crossing-vehicle-count/tools/01_convert/convert_head4_probe.py
```

head4 把模型输出从「坐标 0～640」换成「DFL 距离 0～15」。RKNN 配量化刻度用 `范围 / 256`，输出坐标拿到刻度 2.5 精度被压扁，输出距离拿到刻度 0.047，细 53 倍。原理见 C++ 仓库 [docs/head4_原理详解.md](https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp/blob/main/docs/head4_%E5%8E%9F%E7%90%86%E8%AF%A6%E8%A7%A3.md)。

## 02_diagnose —— 阈值标定

| 脚本 | 作用 |
|---|---|
| `score_dist.py` | 统计检测框的分数分布，给每个视频单独定 conf 阈值 |

模型换过之后，先用它看新模型的分数分布，再决定 `configs/*.json` 里 `detect.conf` 该定多少，别拍脑袋。
