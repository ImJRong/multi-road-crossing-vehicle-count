# 汇总报告：把 outputs/records_*.csv 这些过线原始记录，合并成一张 report.csv。
#
# 原始记录是"每过一辆车一行"，一辆车一行不方便看。
# 这一步按 视频 / 计数线 / 类别 归并，算出每格的数量。
#
# 跑法（在项目根目录）：python src/05_export_csv.py
# 输入：outputs/records_*.csv
# 输出：outputs/report.csv

import csv
import glob
import os

# ============ 配置区 ============
RECORD_DIR = "outputs"                 # 原始记录放哪
RECORD_GLOB = "records_*.csv"          # 收哪些文件
OUT = "outputs/report.csv"             # 汇总报告存哪
# ===============================

files = sorted(glob.glob(os.path.join(RECORD_DIR, RECORD_GLOB)))
if not files:
    print("没找到 %s，先跑一遍 src/04_count_lanes.py" % RECORD_GLOB)
    exit()

table = {}
for path in files:
    # records_test1.csv -> test1
    name = os.path.basename(path)[len("records_"):-len(".csv")]
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (name, row["line"], row["class"])
            table[key] = table.get(key, 0) + 1
    print("读了 %s" % path)

rows = []
for (name, line, cls), n in sorted(table.items()):
    rows.append({
        "视频": name,
        "计数线": line,
        "类别": cls,
        "数量": n,
    })

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["视频", "计数线", "类别", "数量"])
    w.writeheader()
    w.writerows(rows)

print()
print("汇总：")
print("  %-8s %-12s %-8s %s" % ("视频", "计数线", "类别", "数量"))
for r in rows:
    print("  %-8s %-12s %-8s %d" % (r["视频"], r["计数线"], r["类别"], r["数量"]))

print()
for name in sorted({r["视频"] for r in rows}):
    n = sum(r["数量"] for r in rows if r["视频"] == name)
    print("  %s 合计：%d" % (name, n))
print()
print("报告：%s（%d 行）" % (OUT, len(rows)))
