#!/usr/bin/env python3
"""验证 auto_hybrid 到底会不会真的把层切成 fp16。

上一轮冒烟：auto_hybrid=True 跑通了，但产出模型和基线**字节数完全一样**，
说明它没切任何层（小模型误差没到阈值）。这不能算验证成功——
必须证明"阈值调严 -> 确实有层变 fp16 -> 模型大小真的变了"。

做法：同一个极小模型，跑 4 组：
  1. 基线（不传 auto_hybrid）
  2. auto_hybrid=True + 默认阈值 cos=0.98
  3. auto_hybrid=True + cos=0.99999（几乎不可能达到 -> 逼它切）
  4. auto_hybrid=True + cos=0.99999 + euc=0.0（双重逼迫）

看每组的输出字节数，只有大小不同才能证明开关生效。

跑法（WSL，rknn 环境）：
  conda activate rknn
  python /mnt/d/multi-road-crossing-vehicle-count/tools/_smoke_auto_hybrid2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _smoke_auto_hybrid import DATASET, ONNX, WORK, make_dataset, make_tiny_onnx

from rknn.api import RKNN


def build(tag, out, **kwargs):
    if not os.path.isfile(ONNX) or not os.path.isfile(DATASET):
        make_tiny_onnx()
        make_dataset()
    print("\n" + "=" * 66)
    print(">>>", tag)
    rknn = RKNN(verbose=False)
    cfg = dict(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
               target_platform="rk3588")
    cfg.update(kwargs.get("config", {}))
    if rknn.config(**cfg) != 0:
        print("   config 失败")
        return None
    if rknn.load_onnx(model=ONNX) != 0:
        print("   load_onnx 失败")
        return None
    bkw = kwargs.get("build", {})
    try:
        ret = rknn.build(do_quantization=True, dataset=DATASET, **bkw)
    except TypeError as e:
        print("   !! build 不接受参数:", e)
        rknn.release()
        return None
    if ret != 0:
        print("   build 失败 ret=", ret)
        rknn.release()
        return None
    if rknn.export_rknn(out) != 0:
        print("   export 失败")
        rknn.release()
        return None
    n = os.path.getsize(out)
    print("   -> %s  %d 字节" % (os.path.basename(out), n))
    rknn.release()
    return n


def main():
    results = {}
    results["1_基线"] = build("1. 基线（auto_hybrid 不传）",
                            os.path.join(WORK, "a1_base.rknn"))
    results["2_默认阈值"] = build("2. auto_hybrid=True，默认 cos=0.98",
                              os.path.join(WORK, "a2_def.rknn"),
                              build=dict(auto_hybrid=True))
    results["3_cos0.99999"] = build(
        "3. auto_hybrid=True，cos=0.99999（逼它切）",
        os.path.join(WORK, "a3_strict.rknn"),
        config=dict(auto_hybrid_cos_thresh=0.99999),
        build=dict(auto_hybrid=True))
    results["4_cos+euc"] = build(
        "4. auto_hybrid=True，cos=0.99999 + euc=0.0",
        os.path.join(WORK, "a4_strict2.rknn"),
        config=dict(auto_hybrid_cos_thresh=0.99999, auto_hybrid_euc_thresh=0.0),
        build=dict(auto_hybrid=True))

    print("\n" + "=" * 66)
    print("%-18s %s" % ("配置", "产出字节"))
    base = results.get("1_基线")
    for k, v in results.items():
        mark = ""
        if v is not None and base is not None:
            mark = "  同基线" if v == base else "  差 %+d" % (v - base)
        print("%-18s %s%s" % (k, v, mark))

    print("\n结论：")
    changed = [k for k, v in results.items()
               if v is not None and base is not None and v != base]
    if changed:
        print("  ✅ auto_hybrid 真的会切层，变化组：", changed)
        return 0
    print("  ❌ 四组大小全一样 —— auto_hybrid 在这个模型上没切任何层，")
    print("     要么阈值机制不生效，要么这个模型本身误差太小。需要换真模型再判。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
