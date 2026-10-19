# 分方向计数：一条计数线 = 一条路 / 一个方向，只出总数。
#
# 跑法（在项目根目录）：
#   python src/04_count_lanes.py              → 默认跑 test1
#   python src/04_count_lanes.py test3        → 跑 configs/test3.json
#
# 输出：outputs/records_<名字>.csv    每辆车过线的原始记录
#       outputs/<名字>_lanes.mp4      画了线和数字的视频（output 为 show 时改成实时播放）
#
# 视频上画什么、怎么画，都在配置的 "draw": {...} 里：
#   "draw": { "roi": false, "box": false, "id": false, "label": false, ... }
#   正式测试时把这些关掉，画面就干净了。

import csv
import json
import sys
import time

from counter import run_video, aggregate, aggregate_group

# ============ 配置区 ============
NAME = sys.argv[1] if len(sys.argv) > 1 else "test1"
CONFIG = "configs/%s.json" % NAME
OUT_VIDEO = "outputs/%s_lanes.mp4" % NAME
OUT_RECORD = "outputs/records_%s.csv" % NAME
# ===============================

with open(CONFIG, encoding="utf-8") as f:
    config = json.load(f)

# 输出方式由配置里的 output 决定：
#   "save"   → 跑完存成视频文件（默认）
#   "show"   → 开窗口实时播放，不存文件，按 Q 或 Esc 退出
OUTPUT = config.get("output", "save")
SAVE = OUTPUT == "save"
SHOW = OUTPUT == "show"
if OUTPUT not in ("save", "show"):
    raise SystemExit('配置里的 output 只能填 "save" 或 "show"，现在填的是 %r' % OUTPUT)

DETECT = config.get("detect") or {}

print("配置文件：%s" % CONFIG)
print("视频：%s" % config["video"])
print("输出方式：%s" % ("存成视频文件 " + OUT_VIDEO if SAVE else "实时播放（不保存）"))
for line in config["lines"]:
    print("  计数线 %s：%s -> %s" % (line["name"], line["a"], line["b"]))

t0 = time.perf_counter()
records, frames = run_video(
    config["video"],
    config,
    out_video=OUT_VIDEO if SAVE else None,
    max_frames=config.get("max_frames", 0),
    conf=DETECT.get("conf", config.get("conf", 0.3)),
    iou=DETECT.get("iou", config.get("iou", 0.5)),
    display=SHOW,
)
cost = time.perf_counter() - t0

# 每辆车过线的原始记录，后面 05 拿它做汇总
with open(OUT_RECORD, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["line", "group", "class", "frame", "x", "y"])
    w.writeheader()
    w.writerows(records)

print()
print("按线统计：")
print("  %-14s %-8s %s" % ("线", "类别", "数量"))
table = aggregate(records)
total = 0
for (line, cls), n in sorted(table.items()):
    print("  %-14s %-8s %d" % (line, cls, n))
    total += n

# 只有当有线的 group 跟线名不一样时, 才多打一张合并表 (否则跟上面重复)
if any(L.get("group") and L["group"] != L["name"] for L in config["lines"]):
    print()
    print("按方向合并（group）：")
    print("  %-14s %-8s %s" % ("方向", "类别", "数量"))
    for (g, cls), n in sorted(aggregate_group(records).items()):
        print("  %-14s %-8s %d" % (g, cls, n))

print()
print("共 %d 帧，过线 %d 次，耗时 %.1f 秒（%.1f 帧/秒）"
      % (frames, total, cost, frames / cost))
print("记录：%s" % OUT_RECORD)
if SAVE:
    print("视频：%s" % OUT_VIDEO)
