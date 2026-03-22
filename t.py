import numpy as np

path = r"C:\Users\ZSwinB\Documents\GitHub\2DGeoDistanceField\outputs\wall\0.npy"

walls = np.load(path, allow_pickle=True)

xs = []
ys = []

for A, B in walls:
    xs.extend([A[0], B[0]])
    ys.extend([A[1], B[1]])

print("x min/max:", min(xs), max(xs))
print("y min/max:", min(ys), max(ys))