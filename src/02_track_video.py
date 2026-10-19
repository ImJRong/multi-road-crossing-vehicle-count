# 逐帧检测 + 跟踪，给每辆车一个编号（ID）。
#
# 01 里每帧独立数，同一辆车在 1000 帧里会被数 1000 次。有了 ID 才能认出"这是同一辆车"。
#
# 跑法（在项目根目录）：python src/02_track_video.py
# 输出：outputs/test1_track.mp4
#
# 和 01 的差别只有一处：model() 换成 model.track()，并且 persist=True。
# persist 让跟踪器跨帧保留"上一帧谁在哪"的名单，少了它 ID 会乱跳。
#
# 要看的还是两处：
#   model.track(..., persist=True)                    一帧的推理 + 匹配
#   if r.boxes.id is not None / for box in r.boxes    取 ID 和类别

from ultralytics import YOLO
import cv2

# ============ 配置区 ============
MODEL = "models/yolov8n.pt"            # 用哪个模型
VIDEO = "data/videos/test1.mp4"        # 输入视频
OUT = "outputs/test1_track.mp4"        # 输出视频存哪
TRACKER = "bytetrack.yaml"             # 用哪个跟踪器，YOLO 自带 7 种，先用这个

CLASSES = [2, 5, 7]                    # 只检测机动车：汽车 公交 卡车
VEHICLE = [2, 5, 7]                    # 上面这几类都算车，全都要数
MAX_FRAMES = 300                       # 只跑前 300 帧，0 = 全跑
CONF = 0.3                             # 分数门槛
# ===============================

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("打不开视频，检查路径：", VIDEO)
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("视频：%s" % VIDEO)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, fps, (width, height))

seen = set()        # 出现过哪些车辆 ID，set 自动去重
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
            if cid in VEHICLE:
                n_veh = n_veh + 1
                seen.add(int(box.id))              # box.id 是跟踪编号

    frame = r.plot()                               # 画框，ID 也会画上

    cv2.putText(frame, "frame: %d" % n_veh, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.putText(frame, "seen: %d" % len(seen), (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    writer.write(frame)

    index = index + 1
    print("\r已处理 %d 帧，本帧车辆 %d，累计出现车辆 %d" % (index, n_veh, len(seen)), end="")

    if MAX_FRAMES > 0 and index >= MAX_FRAMES:
        break

cap.release()
writer.release()

print("\nDONE，共 %d 帧，累计出现车辆 %d → %s" % (index, len(seen), OUT))
