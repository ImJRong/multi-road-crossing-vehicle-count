#!/usr/bin/env python3
"""批量对比：多帧、用 FP32 真值，分别量 v7 和 head4 的框误差。

单帧只配上 1 个框，说明不了问题。这个脚本跑多帧，
每帧都导出真值，板子上用 box3 对比，汇总所有配对的误差。

流程：
  1. PC 上对 N 帧跑 FP32 ONNX，导出真值文件
  2. 传真值到板子
  3. 板子上对每帧跑 box3，收集误差
  4. 汇总平均

用法：
  D:\\Python3.12.4\\python.exe tools\02_diagnose\batch_box_eval.py
"""
import os
import re
import subprocess
import sys

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/xx/ -> 项目根
FRAMES = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
HOST, USER, PASS = "192.168.1.105", "orangepi", "orangepi"


def ssh(cmd, timeout=300):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, password=PASS, timeout=15, banner_timeout=30)
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    cli.close()
    return o, e


def put(local, remote):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = cli.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    cli.close()


def gen_truth(frame_no):
    """调 dump_truth_boxes.py 生成该帧真值"""
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "tools", "02_diagnose", "dump_truth_boxes.py"),
                        str(frame_no)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print("  生成真值失败:", r.stderr.strip()[:200])
        return None
    path = os.path.join(ROOT, "outputs", "truth_boxes.txt")
    if not os.path.isfile(path):
        return None
    return path


def parse_avg(text, model_tag):
    """从 box3 输出里抠出某个模型的平均误差"""
    # 找到 "===== <model>  检出" 之后到下一个 ===== 之间的内容
    m = re.search(r"=====\s*%s\s+检出.*?----\s*平均误差:\s*左\s*([+-][\d.]+)\s*上\s*([+-][\d.]+)"
                  r"\s*宽\s*([+-][\d.]+)\s*高\s*([+-][\d.]+)\s*中心偏\(([+-][\d.]+),([+-][\d.]+)\)"
                  % re.escape(model_tag), text, re.S)
    if not m:
        return None
    g = [float(x) for x in m.groups()]
    return dict(left=g[0], top=g[1], dw=g[2], dh=g[3], dcx=g[4], dcy=g[5])


def main():
    v7 = "models/best_detrac_v7.rknn"
    h4 = "models/best_detrac_v12_head4.rknn"

    rows = []
    for fr in FRAMES:
        print("帧 %d ..." % fr, end="", flush=True)
        tp = gen_truth(fr)
        if not tp:
            print(" 跳过")
            continue
        put(tp, "/tmp/truth_%d.txt" % fr)
        out, err = ssh("cd ~/code && /tmp/box3 %s %s data/videos/test1.mp4 %d /tmp/truth_%d.txt 2>&1"
                       % (v7, h4, fr, fr), timeout=240)
        a = parse_avg(out, v7)
        b = parse_avg(out, h4)
        if not a or not b:
            print(" 没配上对")
            continue
        rows.append((fr, a, b))
        print("  v7 中心(%+.1f,%+.1f)  head4 中心(%+.1f,%+.1f)"
              % (a["dcx"], a["dcy"], b["dcx"], b["dcy"]))

    print("\n" + "=" * 78)
    if not rows:
        print("一帧都没配上，检查真值生成")
        return 1

    def avg(key, which):
        return sum(r[which][key] for r in rows) / len(rows)

    print("汇总 %d 帧" % len(rows))
    print("-" * 78)
    print("%-14s %10s %10s %10s %10s" % ("", "左边缘", "上边缘", "中心X", "中心Y"))
    for tag, which, name in [("v7 (旧)", 1, "v7"), ("head4 (新)", 2, "head4")]:
        print("%-14s %+10.2f %+10.2f %+10.2f %+10.2f"
              % (tag, avg("left", which), avg("top", which),
                 avg("dcx", which), avg("dcy", which)))
    print("-" * 78)

    v7c = (avg("dcx", 1) ** 2 + avg("dcy", 1) ** 2) ** 0.5
    h4c = (avg("dcx", 2) ** 2 + avg("dcy", 2) ** 2) ** 0.5
    print("中心偏移模长:  v7 = %.2f px   head4 = %.2f px" % (v7c, h4c))
    if h4c < v7c:
        print("head4 更准，改善 %.1f 倍" % (v7c / h4c if h4c > 1e-6 else float("inf")))
    else:
        print("head4 没有改善")
    return 0


if __name__ == "__main__":
    sys.exit(main())
