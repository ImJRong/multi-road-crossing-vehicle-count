#!/usr/bin/env python3
"""转换前参数自检：把每个要用的参数从已安装的工具包里读出来，不靠记忆。

跑法：在各自 conda 环境里跑这个脚本（脚本会打印它看到的一切）
"""
import inspect
import os
import sys

print("=" * 70)
print("python:", sys.version.split()[0], "| exe:", sys.executable)

try:
    import rknn
    from rknn.api import RKNN
    pkg = getattr(rknn, "__path__", [None])[0]
    ver = open(os.path.join(pkg, "VERSION")).read().strip()
    print("rknn-toolkit2 VERSION:", ver)
except Exception as e:
    print("导入 rknn 失败:", e)
    raise SystemExit(1)

print("-" * 70)
print("[1] RKNN.config 全部参数名 + 默认值")
sig = inspect.signature(RKNN.config)
for name, p in sig.parameters.items():
    if name == "self":
        continue
    print("    %-24s = %r" % (name, p.default))

print("-" * 70)
print("[2] RKNN.build 全部参数名 + 默认值")
sig = inspect.signature(RKNN.build)
for name, p in sig.parameters.items():
    if name == "self":
        continue
    print("    %-24s = %r" % (name, p.default))

print("-" * 70)
print("[3] RKNNBase.build 是否真的接受 auto_hybrid（编译进 .so 的真实签名）")
if hasattr(RKNN, "hybrid_quantization_step1"):
    print("    hybrid_quantization_step1 参数:")
    for name, p in inspect.signature(RKNN.hybrid_quantization_step1).parameters.items():
        if name == "self":
            continue
        print("        %-24s = %r" % (name, p.default))

# 编译后的 rknn_base.build 是 Cython 函数，签名可能读不到，试试 doc
base = RKNN("__probe__").rknn_base if False else None  # 不实例化，避免副作用
print("-" * 70)
print("[4] 精度分析接口")
print("    accuracy_analysis 存在:", hasattr(RKNN, "accuracy_analysis"))
if hasattr(RKNN, "accuracy_analysis"):
    for name, p in inspect.signature(RKNN.accuracy_analysis).parameters.items():
        if name == "self":
            continue
        print("        %-24s = %r" % (name, p.default))
print("=" * 70)
