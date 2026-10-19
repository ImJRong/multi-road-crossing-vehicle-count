# 逐帧检测，把画好框的结果存成新视频。
#
# 这是第 1 步，只检测、不计数：先确认模型认得出车、框画得对。
# 计数放在 02 里做。
#
# 跑法（在项目根目录）：python src/01_detect_video.py
# 输出：outputs/test1_detect.mp4
# 注意：mp4v 编码 Windows 自带"电影和电视"打不开，用 PotPlayer 或 VLC。
#
# 代码里真正要看的是两处：
#   model(frame, classes=, conf=)   一帧的推理
#   for box in r.boxes              从结果里把框取出来
# 其余的开视频、建 VideoWriter、putText、写帧都是固定套路，不用记。

from ultralytics import YOLO
import cv2

# ============ 配置区 ============
MODEL = "models/yolov8n.pt"            # 用哪个模型
VIDEO = "data/videos/test1.mp4"        # 输入视频
OUT = "outputs/test1_detect.mp4"       # 输出视频存哪

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
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("视频：%s" % VIDEO)
print("尺寸：%dx%d   帧率：%.1f   总帧数：%d" % (width, height, fps, total))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, fps, (width, height))

index = 0

while True:
    isread, frame = cap.read()

    if not isread:
        break

    # 只跑我们关心的 6 类，比跑全部 80 类快不少；低于 conf 的框直接丢
    # 这里用 model()，每帧独立，看完就忘，不记得上一帧
    results = model(frame, verbose=False, classes=CLASSES, conf=CONF)
    r = results[0]

    n_veh = 0
    for box in r.boxes:                            # 一个 box 就是一个框
        cid = int(box.cls)                         # box.cls 是张量，转成整数
        if cid in VEHICLE:
            n_veh = n_veh + 1

    frame = r.plot()                               # 画框

    cv2.putText(frame, "vehicles: %d" % n_veh, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    writer.write(frame)

    index = index + 1
    print("\r已处理 %d 帧，本帧车辆 %d" % (index, n_veh), end="")

    if MAX_FRAMES > 0 and index >= MAX_FRAMES:
        break

cap.release()
writer.release()

print("\nDONE，共 %d 帧 → %s" % (index, OUT))
