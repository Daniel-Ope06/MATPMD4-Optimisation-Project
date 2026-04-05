import math
import numpy as np

coords: np.ndarray = np.array([
    # Stores 1-23
    [35, 16], [90, 92], [25, 82], [95, 77], [85, 85], [51, 37], [46, 68],
    [31, 43], [71, 40], [91, 68], [68, 29], [52, 21], [57, 44], [11, 10],
    [41, 45], [34, 87], [26, 41], [47, 60], [58, 9], [27, 12], [9, 92],
    [95, 60], [23, 64],
    # Warehouses W1, W2
    [21, 25], [37, 70]
])

N: int = len(coords)
W1_IDX: int = 23
W2_IDX: int = 24

# Euclidean Distance Matrix
dist: np.ndarray = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        dist[i][j] = np.linalg.norm(coords[i] - coords[j])

# Polar Angles Array (Stores relative to W1 and W2)
angle_w1: np.ndarray = np.zeros(N)
angle_w2: np.ndarray = np.zeros(N)

for i in range(23):  # Only calculate angles for the 23 stores
    angle_w1[i] = math.atan2(
        coords[i][1] - coords[W1_IDX][1],
        coords[i][0] - coords[W1_IDX][0]
    )
    angle_w2[i] = math.atan2(
        coords[i][1] - coords[W2_IDX][1],
        coords[i][0] - coords[W2_IDX][0]
    )
