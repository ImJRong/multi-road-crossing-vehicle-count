import os, cv2, random
import numpy as np

ROOT = '/kaggle/input/datasets/starsw/intersection-flow-5k/Intersection-Flow-5K'
OUT = '/kaggle/working/check_obstacle'
os.makedirs(OUT, exist_ok=True)

# 挑出 obstacle 框最多的前 6 张图
cands = []
d = os.path.join(ROOT, 'labels', 'train')
for name in os.listdir(d):
    if not name.endswith('.txt'):
        continue
    n = 0
    for line in open(os.path.join(d, name)):
        p = line.split()
        if len(p) >= 5 and int(float(p[0])) == 7:   # 7 = obstacle
            n += 1
    if n > 0:
        cands.append((n, name))
cands.sort(reverse=True)
print('含 obstacle 的图共 %d 张，最多的 6 张：' % len(cands))
for n, name in cands[:6]:
    print('   %s  %d 个 obstacle 框' % (name, n))

COLORS = {0:(0,255,0), 1:(255,0,0), 2:(0,255,255), 3:(255,0,255),
          4:(128,0,128), 5:(0,128,255), 6:(255,255,0), 7:(0,0,255)}
NAMES = ['vehicle','bus','bicycle','pedestrian','engine','truck','tricycle','obstacle']

random.seed(0)
for idx, (n, name) in enumerate(cands[:6]):
    stem = name[:-4]
    img = None
    for ext in ('.jpg','.jpeg','.png','.bmp'):
        p = os.path.join(ROOT,'images','train', stem+ext)
        if os.path.exists(p):
            img = cv2.imread(p); break
    if img is None: continue
    H, W = img.shape[:2]
    for line in open(os.path.join(ROOT,'labels','train', name)):
        p = line.split()
        if len(p) < 5: continue
        cid = int(float(p[0]))
        cx,cy,w,h = map(float, p[1:5])
        x1,y1 = int((cx-w/2)*W), int((cy-h/2)*H)
        x2,y2 = int((cx+w/2)*W), int((cy+h/2)*H)
        col = COLORS[cid]
        th = 4 if cid==7 else 2
        cv2.rectangle(img,(x1,y1),(x2,y2),col,th)
        if cid==7:
            cv2.putText(img,'OBST',(x1,max(y1-6,12)),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
    # 缩到 1280 宽方便看
    if W > 1280:
        img = cv2.resize(img,(1280,int(H*1280/W)))
    out = os.path.join(OUT, 'obstacle_%d.jpg'%idx)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('->', out)
