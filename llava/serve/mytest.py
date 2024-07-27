import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 定义8个点的坐标
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

# 计算凸包
hull = ConvexHull(points)

# 绘制3D图形
ax = plt.subplot(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# 绘制凸包的外框
for simplex in hull.simplices:
    ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

# 显示图形
plt.show()

# 检查绘图后的点和边
print("Points:\n", points)
print("Hull vertices:\n", hull.vertices)
