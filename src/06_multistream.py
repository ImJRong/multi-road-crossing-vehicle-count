# 多路并行：几个路口视频同时跑，一个进程管一路。
#
# 为什么用进程（Process）而不是线程（Thread）：
#   Python 的线程受 GIL 限制，同一时刻只有一个线程能跑 Python 代码，
#   而 YOLO 的推理大段都在 Python 里调度，开线程几乎不加速。
#   开进程各自独立，才能真正并行吃满多核 CPU。
#
# 代价：每个进程都要单独加载一份模型、单独占内存。路数开到 CPU 核数以上反而变慢。
#
# 跑法（在项目根目录）：
#   python src/06_multistream.py                      跑 configs/ 下全部
#   python src/06_multistream.py test1 test3          只跑指定的几路
# 输出：outputs/<名字>_lanes.mp4、outputs/records_<名字>.csv
#
# 注意：多路并行时子进程开不了窗口，所以每路的 output 一律按 save 处理（存文件）。
#       想实时播放就在配置文件里写 "output": "show"，然后用 04_count_lanes.py 单跑那一路。

import csv
import json
import multiprocessing as mp
import os
import sys
import time

# ============ 配置区 ============
CONFIG_DIR = "configs"                 # 路口配置放哪
OUT_DIR = "outputs"                    # 输出放哪
# ===============================


def worker(cfg_path, threads):
    """跑一路。这个函数必须在模块顶层，子进程要靠名字找到它。"""
    import torch
    torch.set_num_threads(threads)         # 每个进程只分这么多线程，免得几路互相抢

    import counter

    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)

    name = config["name"]
    out_video = os.path.join(OUT_DIR, "%s_lanes.mp4" % name)
    out_record = os.path.join(OUT_DIR, "records_%s.csv" % name)

    detect = config.get("detect") or {}
    t0 = time.perf_counter()
    records, frames = counter.run_video(
        config["video"],
        config,
        out_video=out_video,
        max_frames=config.get("max_frames", 0),
        conf=detect.get("conf", config.get("conf", 0.3)),
        iou=detect.get("iou", config.get("iou", 0.5)),
    )
    cost = time.perf_counter() - t0

    with open(out_record, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["line", "group", "class", "frame", "x", "y"])
        w.writeheader()
        w.writerows(records)

    # 子进程之间传不回返回值，把这一路的成绩写个小文件，主进程再读
    with open(os.path.join(OUT_DIR, "_stream_%s.json" % name), "w",
              encoding="utf-8") as f:
        json.dump({"name": name, "frames": frames, "seconds": cost,
                   "hits": len(records)}, f, ensure_ascii=False)

    return name, frames, cost, len(records)


def main():
    names = sys.argv[1:]
    paths = []
    for fn in sorted(os.listdir(CONFIG_DIR)):
        if not fn.endswith(".json"):
            continue
        if names and fn[:-5] not in names:
            continue
        paths.append(os.path.join(CONFIG_DIR, fn))

    if not paths:
        print("configs/ 下没找到配置文件")
        return

    n = len(paths)
    cpu = os.cpu_count() or 1
    threads = max(1, cpu // n)

    print("并行路数：%d    机器核数：%d    每路分到线程：%d" % (n, cpu, threads))
    print()

    ctx = mp.get_context("spawn")
    procs = []
    t0 = time.perf_counter()

    for path in paths:
        p = ctx.Process(target=worker, args=(path, threads))
        p.start()
        procs.append((path, p))
        print("  已启动 %s (pid=%d)" % (path, p.pid))

    results = []
    for path, p in procs:
        p.join()
        if p.exitcode != 0:
            print("  %s 退出码 %s，出错了" % (path, p.exitcode))
    wall = time.perf_counter() - t0

    # 子进程的返回值拿不到（没用 Pool），所以读它们写下的小成绩单
    print()
    print("  %-8s %-8s %-9s %-9s %s" % ("路口", "帧数", "耗时(秒)", "帧/秒", "过线次数"))
    total_frames = 0.0
    total_hits = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            name = json.load(f)["name"]
        stat = os.path.join(OUT_DIR, "_stream_%s.json" % name)
        if not os.path.exists(stat):
            print("  %-8s 没写出成绩，看上面的报错" % name)
            continue
        with open(stat, encoding="utf-8") as f:
            s = json.load(f)
        total_frames += s["frames"]
        total_hits += s["hits"]
        print("  %-8s %-8d %-9.1f %-9.1f %d"
              % (name, s["frames"], s["seconds"], s["frames"] / s["seconds"], s["hits"]))

    print()
    print("%d 路一起跑完，总耗时 %.1f 秒，合计 %d 帧、过线 %d 次"
          % (n, wall, int(total_frames), total_hits))
    print("整体吞吐 %.1f 帧/秒（所有路加在一起）" % (total_frames / wall if wall else 0))
    print()
    print("想知道单路能跑多快，就用 04_count_lanes.py 单独跑一路对比。")


if __name__ == "__main__":
    main()
