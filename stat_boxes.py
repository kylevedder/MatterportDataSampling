import joblib
import sys

for file in sys.argv[1:]:
    print(file)
    boxes = joblib.load(file)
    print(boxes.shape)
    for b in boxes:
        name, img_left, img_top, img_right, img_bottom, center_x, center_y, center_z, l, w, h, yaw = b
        print(l, w, h)
