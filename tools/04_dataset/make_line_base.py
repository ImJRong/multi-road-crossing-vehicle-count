# 做"画线底图"：把 ROI 框画到原帧上，你在这张图上画计数线，就不会画出框外。
#
# 为什么需要：read_line.py 是靠"比对两张图找差异"来认出你画的线的。
# 如果框和线画在同一张图上，框的边也会被当成线读进去，坐标就乱了。
# 解决办法：先生成一张已经带框的图（底图）。你画完以后，
# read_line.py 拿【底图】跟你画的图比，差异就只剩你新画的那条线。
#
# 用法（在项目根目录）：
#   python tools/make_line_base.py data/samples/test3_frame300.jpg configs/test3.json
# 输出：data/samples/test3_frame300_base.jpg

import json
import os
import sys

import cv2
import numpy as np

ROI_COLOR = (0, 200, 255)      # 橙色，和常见的红线蓝线都不撞
THICKNESS = 2


def main():
    if len(sys.argv) < 3:
        print("用法：python tools/make_line_base.py <原帧图> <配置.json>")
        print("例：  python tools/make_line_base.py data/samples/test3_frame300.jpg configs/test3.json")
        return

    frame_path, cfg_path = sys.argv[1], sys.argv[2]

    img = cv2.imread(frame_path)
    if img is None:
        raise SystemExit("读不到图：%s" % frame_path)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    roi = cfg.get("roi")
    if not roi:
        raise SystemExit("%s 里没有 roi 字段，先去跑 tools/read_roi.py 把区域写进去。" % cfg_path)

    poly = np.array(roi, dtype=np.int32)
    cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, ROI_COLOR, THICKNESS)

    # 边上标一下这是框，不是线
    cv2.putText(img, "ROI", (int(poly[:, 0].min()) + 8, int(poly[:, 1].min()) + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ROI_COLOR, 2)

    base, ext = os.path.splitext(frame_path)
    out = base + "_base" + ext
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("底图已生成：%s" % out)
    print("  图上的橙色框就是 ROI，画线时别超出它。")
    print()
    print("接下来：")
    print("  1. 用画图工具打开这张底图，画你们的计数线")
    print("  2. 另存成新文件（别覆盖底图，建议叫 %s_line.jpg）" % cfg["name"])
    print("  3. 读线：")
    print("     python tools/read_line.py %s %s_line.jpg --base %s"
          % (frame_path, os.path.join(os.path.dirname(frame_path), cfg["name"]), out))


if __name__ == "__main__":
    main()
