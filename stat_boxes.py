import joblib
import sys
import numpy as np

ls = []
ws = []
hs = []

for file in sys.argv[1:]:
    boxes = joblib.load(file)
    for b in boxes:
        name, img_left, img_top, img_right, img_bottom, center_x, center_y, center_z, l, w, h, yaw = b
        ls.append(float(l))
        ws.append(float(w))
        hs.append(float(h))

print(np.mean(ls), np.mean(ws), np.mean(hs))
print(np.std(ls), np.std(ws), np.std(hs))
