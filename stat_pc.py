import joblib
import sys
import numpy as np

points_lst = []
for file in sys.argv[1:]:
    points_lst.append(joblib.load(file))

points = np.concatenate(points_lst, axis=0)
print([(name, points[:, idx].min(), points[:, idx].max())
           for name, idx in [("X", 0), ("Y", 1), ("Z", 2)]])

    

