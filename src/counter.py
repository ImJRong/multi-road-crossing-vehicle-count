# 计数核心。04_count_lanes.py 和 06_multistream.py 都调用这里，
# 免得两边的计数逻辑各写一遍、以后改一处忘一处。
#
# 一个视频跑一遍：检测 -> 跟踪 -> 判断有没有穿过配置里的计数线 -> 记一条过线记录。
# 统计（按车道、按方向、按类别）不在这里做，交给调用方 aggregate()。

import math

import cv2
import numpy as np
from ultralytics import YOLO

# 类别名的白名单。自己训的模型只有一个类叫 vehicle，COCO 那个叫 car/bus/truck。
# 两种模型的类别编号不一样（自己的是 0，COCO 是 2/5/7），所以按【名字】去认，不写死编号。
VEHICLE_NAMES = ("vehicle", "car", "bus", "truck")


def vehicle_ids(names):
    """从模型的类别表里挑出算车的那些编号。挑不到就退回用全部类别。"""
    out = [i for i, n in names.items() if str(n).strip().lower() in VEHICLE_NAMES]
    return out if out else list(names.keys())

COLORS = [(0, 255, 0), (0, 255, 255), (255, 128, 0), (255, 0, 255)]

# 画面画成什么样。配置里写 "draw": {...} 就能覆盖这里的默认值。
DEFAULT_STYLE = {
    "roi": True,                # 画不画检测区域框
    "box": True,               # 画不画车的方框
    "box_width": 2,            # 方框线宽
    "box_color": [0, 255, 0],  # 方框颜色 [B, G, R]
    "id": True,                # 画不画跟踪 ID
    "label": True,             # 画不画类别名和置信度
    "font_scale": 0.6,         # 框上文字大小
    "text_width": 2,           # 框上文字粗细
    "counter_scale": 0.9,      # 左上角计数数字大小
    "text_color": [0, 255, 255],
    "line_color": [0, 0, 255],  # 计数线颜色
    "line_width": 2,            # 计数线线宽
    "roi_color": [0, 200, 255], # 检测区域框颜色
}


def make_style(config):
    """把配置里的 draw 小节和默认值合并。draw 里以 说明 开头的条目会被忽略。"""
    style = dict(DEFAULT_STYLE)
    for k, v in (config.get("draw") or {}).items():
        if k.startswith("说明"):
            continue
        if k in style:
            style[k] = list(v) if isinstance(v, list) else v
    return style


def draw_boxes(crop, r, style, ids):
    """在裁剪图上画检测框。返回画好的图（没开任何显示项就原样返回）。"""
    if not (style["box"] or style["id"] or style["label"]):
        return crop

    out = crop.copy()
    if r.boxes.id is None:
        return out

    for box in r.boxes:
        if int(box.cls) not in ids:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        tid = int(box.id)
        name = r.names[int(box.cls)]
        conf = float(box.conf)

        if style["box"]:
            cv2.rectangle(out, (x1, y1), (x2, y2), tuple(style["box_color"]),
                          style["box_width"])

        parts = []
        if style["label"]:
            parts.append("%s %.2f" % (name, conf))
        if style["id"]:
            parts.append("id:%d" % tid)
        if parts:
            cv2.putText(out, " ".join(parts), (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, style["font_scale"],
                        tuple(style["box_color"]), style["text_width"])
    return out


def side(p, a, b):
    """点 p 在直线 AB 的哪一侧。正负各代表一边，0 就是压在线上。用叉积算，线斜着也行。"""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def line_segment(line):
    """计数线的两个端点，也就是 a、b。线画到哪，就只管那一段，不用再写 x_min/x_max。"""
    return tuple(line["a"]), tuple(line["b"])


def near_segment(p, a, b, pad=0):
    """
    点 p 是否落在线段 a-b 上（pad 是两端额外放宽的像素数）。

    做法是把 p 投影到线段方向，看投影长度占全长的比例在不在 0~1 之间。
    横线、竖线、斜线都适用，所以不再需要 x_min/x_max 那种按 x 切的办法。
    """
    ax, ay = b[0] - a[0], b[1] - a[1]
    l2 = ax * ax + ay * ay
    if l2 <= 0:                       # a 和 b 重合，这条线无效
        return False
    t = ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / l2
    slack = pad / math.sqrt(l2)
    return -slack <= t <= 1 + slack


def run_video(video, config, model_path="models/best_detrac.pt",
              out_video=None, max_frames=0, conf=0.3, iou=0.5,
              tracker="bytetrack.yaml", display=False):
    """
    跑一个视频，返回 (过线记录列表, 帧数)。

    每条记录是 dict：line / class / frame / x / y，
    坐标都是【整张图】上的坐标，不是裁剪区域的坐标。

    out_video 给路径就把画好线的视频写出去，给 None 就只算不画。
    max_frames 为 0 表示整段跑完。
    display 为 True 就开窗口实时播放（不写文件），按 Q 或 Esc 退出。
    iou 是 NMS 的重叠阈值，默认 0.5。调小 → 重叠框合并得更狠（挨着的车不容易被拆成两辆）。

    画面上的框、文字、线怎么画，由配置里的 draw 小节控制，
    没写的字段用 DEFAULT_STYLE 里的默认值。

    配置里可选的 roi 字段（多边形顶点列表）：
      给了就只把这块区域裁出来送进模型：区域外的东西不会被误检成车，
      而且裁剪等于放大，车在模型眼里更大更清楚。
      ⚠️ 区域不能只圈住计数线，要往车来的方向多留一段"助跑区"，
         让跟踪器连续看到车好几帧，否则车刚进画面就到线上，会漏。
    """
    lines = config["lines"]
    roi = config.get("roi")                    # [[x,y], ...] 或 None
    detect = config.get("detect") or {}        # 检测参数都在这一小节里
    imgsz = detect.get("imgsz", config.get("imgsz", 640))
    style = make_style(config)

    model = YOLO(model_path)
    # 这个模型的哪些类别算车。自己训的模型只有一类 vehicle，COCO 的是 2/5/7
    ids = vehicle_ids(model.names)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("打不开视频：%s" % video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 裁剪区域取 ROI 的外接矩形；多边形本身用来判断"框中心在不在区域内"
    if roi:
        poly = np.array(roi, dtype=np.int32)
        rx1, ry1 = int(poly[:, 0].min()), int(poly[:, 1].min())
        rx2, ry2 = int(poly[:, 0].max()), int(poly[:, 1].max())
    else:
        poly = None
        rx1 = ry1 = 0
        rx2, ry2 = width, height

    writer = None
    if out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    records = []
    counters = {}       # (线序号, 车道名, 方向) -> 次数，只用来在画面上显示
    prev = {}           # (线序号, 跟踪ID) -> 上一次在线哪一侧
    frame_no = 0

    while True:
        isread, frame = cap.read()
        if not isread:
            break
        frame_no += 1

        # 只把区域裁出来送进模型。没有 roi 时 crop 就是整帧，行为跟以前一样
        crop = frame[ry1:ry2, rx1:rx2]

        # iou 是 NMS 的重叠阈值：两个框重叠超过这个比例，就只留分数高的那个。
        # 默认 0.7 太松，挨在一起的卡车会被拆成两辆；调到 0.5 左右更稳。
        r = model.track(crop, persist=True, verbose=False, imgsz=imgsz,
                        classes=ids, conf=conf, iou=iou, tracker=tracker)[0]

        if r.boxes.id is not None:
            for box in r.boxes:
                if int(box.cls) not in ids:
                    continue
                tid = int(box.id)
                name = model.names[int(box.cls)]

                # 模型给的坐标是"裁剪图"上的，要加回偏移量才是整图坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, x2 = x1 + rx1, x2 + rx1
                y1, y2 = y1 + ry1, y2 + ry1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # 中心点落在多边形外面就不算——裁剪的是外接矩形，多边形可能更小
                if poly is not None and cv2.pointPolygonTest(
                        poly, (float(cx), float(cy)), False) < 0:
                    continue

                for li, line in enumerate(lines):
                    a = line["a"]
                    b = line["b"]
                    margin = line.get("margin", 5)

                    # 只算落在这条线自己那一段上的过线，线外的不算
                    if not near_segment((cx, cy), a, b, margin):
                        continue

                    key = (li, tid)
                    s = side((cx, cy), a, b)

                    # side() 的单位是"像素 × 线长", 除以线长才是车离线多少像素。
                    # 不除的话 margin 等于按 0.02 像素在判, 死区永远不生效。
                    line_len = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
                    if abs(s) / line_len <= margin:    # 贴着线，不更新也不计数
                        continue

                    old = prev.get(key)
                    if old is not None and (old < 0) != (s < 0):
                        # count_dir：只数某个方向，不写就两个方向都数
                        #   "pos" → 只数 负侧→正侧 的（画面里：从上往下穿）
                        #   "neg" → 只数 正侧→负侧 的（从下往上穿）
                        want = line.get("count_dir")
                        passed = s > 0
                        if (want == "pos" and not passed) or (want == "neg" and passed):
                            prev[key] = s
                            continue

                        # group 是"同方向合并"用的：几条线填同一个 group 名，就加在一起算
                        group = line.get("group", line["name"])
                        records.append({
                            "line": line["name"],
                            "group": group,
                            "class": name,
                            "frame": frame_no,
                            "x": int(cx),
                            "y": int(cy),
                        })
                        ck = li
                        counters[ck] = counters.get(ck, 0) + 1

                    prev[key] = s

        if writer or display:
            frame[ry1:ry2, rx1:rx2] = draw_boxes(crop, r, style, ids)

            if style["roi"] and poly is not None:
                cv2.polylines(frame, [poly.reshape(-1, 1, 2)], True, style["roi_color"], 2)

            # 已经写过的数字位置，后面撞上了就往下挪，避免几条线的字叠在一起
            placed = []
            for li, line in enumerate(lines):
                a, b = line_segment(line)
                cv2.line(frame, a, b, style["line_color"], style["line_width"])

                # 数字就写在这条线自己的端点旁边，一眼能看出是哪条路
                n = counters.get(li, 0)
                txt = "%s: %d" % (line["name"], n)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                              style["counter_scale"], style["text_width"])
                # 位置优先用配置里的 label_at；没写就放在线段上端点的上方
                if line.get("label_at"):
                    tx, ty = int(line["label_at"][0]), int(line["label_at"][1])
                else:
                    p = a if a[1] <= b[1] else b
                    tx = p[0] + 6
                    ty = p[1] - 10 if p[1] - 10 > th + 6 else p[1] + 26
                for _ in range(6):                       # 最多往下挪 6 次，避免几条线的字叠在一起
                    box = (tx, ty - th, tx + tw, ty + 4)
                    if not any(box[0] < q[2] and q[0] < box[2] and
                               box[1] < q[3] and q[1] < box[3] for q in placed):
                        break
                    ty += th + 6
                placed.append((tx, ty - th, tx + tw, ty + 4))
                cv2.putText(frame, txt, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            style["counter_scale"], style["text_color"],
                            style["text_width"])

            if writer:
                writer.write(frame)
            if display:
                cv2.imshow("count", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):    # Esc 或 Q 提前退出
                    break

        if max_frames and frame_no >= max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    return records, frame_no


def aggregate(records):
    """把过线记录汇总成 (线, 类别) -> 数量。"""
    table = {}
    for rec in records:
        key = (rec["line"], rec["class"])
        table[key] = table.get(key, 0) + 1
    return table


def aggregate_group(records):
    """按 group 汇总 (方向名, 类别) -> 数量。没写 group 的线用自己的名字。"""
    table = {}
    for rec in records:
        key = (rec.get("group", rec["line"]), rec["class"])
        table[key] = table.get(key, 0) + 1
    return table
