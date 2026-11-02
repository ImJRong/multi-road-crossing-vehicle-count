#!/bin/bash
# 用 rknn-toolkit2 2.2.1 把 ONNX 转成老格式的 RKNN 模型（给驱动 0.9.6 的板子用）
# 用法：bash /mnt/d/multi-road-crossing-vehicle-count/tools/convert_v5.sh

set -e

echo "==== 1. 激活 rknn20 环境 ===="
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rknn20
python --version

echo
echo "==== 2. 确认 toolkit 版本 ===="
pip show rknn-toolkit2 2>/dev/null | grep -E "^Version" || echo "（pip 里查不到版本，继续）"

echo
echo "==== 3. 准备转换脚本（输出改名为 best_detrac_v5.rknn）===="
cd /mnt/d/multi-road-crossing-vehicle-count
cp -f tools/build_rknn.py tools/build_rknn_v5.py
sed -i 's|best_detrac_new.rknn|best_detrac_v5.rknn|' tools/build_rknn_v5.py
grep "RKNN_OUT" tools/build_rknn_v5.py | head -1

echo
echo "==== 4. 开始转换（要几分钟）===="
python tools/build_rknn_v5.py

echo
echo "==== 5. 结果 ===="
ls -lh models/best_detrac_v5.rknn
echo "--- 模型内部格式版本 ---"
strings models/best_detrac_v5.rknn | head -3

echo
echo "如果上面写着 version 5（不是 6），就成了，把这个文件传板子。"
