import pprint

base = r"C:/Users/RONGG/WorkBuddy/2026-10-27-21-12-12/"

rows = []
with open(base + "dets_test1.csv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        frame, x1, y1, x2, y2, score = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
        rows.append((frame, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, score))

print("总框数:", len(rows))

# 按帧分组
by_frame = {}
for r in rows:
    by_frame.setdefault(r[0], []).append(r)

frames = sorted(by_frame)
print("帧数:", len(frames), " 帧号范围:", frames[0], "~", frames[-1])

# 贪心关联
tracks = []
active = []
for fr in frames:
    dets = sorted(by_frame[fr], key=lambda r: -r[5])
    used = set()
    new_active = []
    for d in dets:
        best, best_d = None, 1e9
        for ti in active:
            if ti in used:
                continue
            last = tracks[ti][-1]
            if fr - last[0] > 15:
                continue
            dist = ((d[1] - last[1]) ** 2 + (d[2] - last[2]) ** 2) ** 0.5
            if dist < best_d and dist < 60:
                best, best_d = ti, dist
        if best is not None:
            tracks[best].append(d)
            used.add(best)
            new_active.append(best)
        else:
            tracks.append([d])
            new_active.append(len(tracks) - 1)
    active = new_active

print("轨迹数:", len(tracks))

# 找静止轨迹
still = []
for t in tracks:
    if len(t) < 30:
        continue
    xs = [p[1] for p in t]
    ys = [p[2] for p in t]
    span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    if span <= 10:
        still.append(t)

print("\n静止轨迹（>=30帧，总位移<=10px）:", len(still))


def std(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


all_dx, all_dy, all_dw, all_dh = [], [], [], []
for t in still:
    xs = [p[1] for p in t]
    ys = [p[2] for p in t]
    ws = [p[3] for p in t]
    hs = [p[4] for p in t]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    all_dx += [x - mx for x in xs]
    all_dy += [y - my for y in ys]
    mw, mh = sum(ws) / len(ws), sum(hs) / len(hs)
    all_dw += [w / mw for w in ws]
    all_dh += [h / mh for h in hs]
    print("  轨迹 %d 帧, 框 %.0fx%.0f, cx σ=%.2f, cy σ=%.2f"
          % (len(t), mw, mh, std(xs), std(ys)))

if all_dx:
    print("\n=== 观测噪声 R ===")
    print("cx 抖动 σ = %.2f px" % std(all_dx))
    print("cy 抖动 σ = %.2f px" % std(all_dy))
    print("w 相对 σ = %.2f%%" % (std(all_dw) * 100))
    print("h 相对 σ = %.2f%%" % (std(all_dh) * 100))
