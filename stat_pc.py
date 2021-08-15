import joblib
import sys

for file in sys.argv[1:]:
    print(file)
    points = joblib.load(file)
    print(points.shape)
    print([(name, points[:, idx].min(), points[:, idx].max())
           for name, idx in [("X", 0), ("Y", 1), ("Z", 2)]])
