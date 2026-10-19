# 逐帧检测 + 跟踪 + 穿线计数。
#
# 02 里的 seen 只统计"出现过多少个不同 ID"，ID 一断就会多算，不能当车流量。
# 这一步改成一个规矩：车必须真的从线的一边走到另一边，才 +1。
#
# 跑法（在项目根目录）：python src/03_count_video.py
# 输出：outputs/test1_count.mp4
#
# 线的位置在配置区改：LINE_A / LINE_B 是线两端的坐标。
#   竖直的线：两个点的 x 一样，比如 (640,0) (640,674)
#   水平的线：两个点的 y 一样，比如 (0,220) (1280,220)
#
# 这条 y=220 不是随便定的：我在 test1 前 300 帧上把 y 从 150 扫到 650，
# 车中心的活动范围集中在 y=100~400，y=220 附近穿过的最多。换视频要重新扫。

from ultralytics import YOLO
import cv2

# ============ 配置区 ============
MODEL = "models/yolov8n.pt"            # 用哪个模型
VIDEO = "data/videos/test1.mp4"        # 输入视频
OUT = "outputs/test1_count.mp4"        # 输出视频存哪
TRACKER = "bytetrack.yaml"             # 用哪个跟踪器

LINE_A = (0, 220)                      # 计数线起点
LINE_B = (1280, 220)                   # 计数线终点
MARGIN = 5                             # 死区：车中心离线的距离小于它就不算，防止贴着线来回抖被反复数

CLASSES = [2, 5, 7]                    # 只检测机动车：汽车 公交 卡车
VEHICLE = [2, 5, 7]                    # 上面这几类都算车，全都要数
MAX_FRAMES = 300                       # 只跑前 300 帧，0 = 全跑
CONF = 0.3                             # 分数门槛
# ===============================


# 判断点 p 在直线 AB 的哪一侧。结果正负各代表一边，等于 0 就是压在线上。
# 用的是叉积，车走哪边都能用，线斜着画也不用改。
def side(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("打不开视频，检查路径：", VIDEO)
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("视频：%s" % VIDEO)
print("计数线：%s -> %s" % (str(LINE_A), str(LINE_B)))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, fps, (width, height))

prev_side = {}      # ID -> 上一帧在线哪一侧
count_down = 0      # 从上往下穿过
count_up = 0        # 从下往上穿过
index = 0

while True:
    isread, frame = cap.read()

    if not isread:
        break

    # persist=True 让跟踪器记住上一帧，ID 才接得上
    results = model.track(frame, persist=True, verbose=False,
                          classes=CLASSES, conf=CONF, tracker=TRACKER)
    r = results[0]

    n_veh = 0

    # 这一帧一个框都没有时 boxes.id 是 None，直接遍历会报错
    if r.boxes.id is not None:
        for box in r.boxes:
            cid = int(box.cls)
            if cid not in VEHICLE:
                continue

            tid = int(box.id)
            n_veh = n_veh + 1

            # 取框的左上右下角，算出中心点，用中心点判断在线的哪一侧
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            s = side((cx, cy), LINE_A, LINE_B)

            # 贴着线（离线的距离没超过死区）就不管，不然车停在线上抖来抖去会被数好几次
            if abs(s) > MARGIN:
                old = prev_side.get(tid)
                if old is not None and (old < 0) != (s < 0):
                    # 上一帧在一边、这一帧在另一边 = 这一帧穿过去了
                    if s > 0:
                        count_down = count_down + 1
                    else:
                        count_up = count_up + 1

                prev_side[tid] = s

            # 在框上写 ID，方便肉眼核对是哪一辆车被数的
            cv2.putText(frame, "id:%d" % tid, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    frame = r.plot()                               # 画框

    # 把计数线和两个方向的数字画上去
    cv2.line(frame, LINE_A, LINE_B, (0, 0, 255), 2)
    cv2.putText(frame, "down: %d" % count_down, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.putText(frame, "up: %d" % count_up, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    writer.write(frame)

    index = index + 1
    print("\r已处理 %d 帧，本帧车辆 %d，往下 %d 往上 %d"
          % (index, n_veh, count_down, count_up), end="")

    if MAX_FRAMES > 0 and index >= MAX_FRAMES:
        break

cap.release()
writer.release()

print("\nDONE，共 %d 帧，往下 %d 辆，往上 %d 辆，合计 %d → %s"
      % (index, count_down, count_up, count_down + count_up, OUT))
