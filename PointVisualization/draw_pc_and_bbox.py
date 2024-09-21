import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 绘制3D图形
ax = plt.subplot(projection='3d')

npy_filname = "Table_23724_0_1.npy"
npy_path = "./playground/data/shapellm/gapartnet_pcs/" + npy_filname
points = np.load(npy_path)
sampled_points = points[np.random.choice(points.shape[0], 5000, replace=False), :]

# 定义8个点的坐标
bboxes = np.array([
    [[-0.6, -0.74, -0.03], [-0.02, -0.74, -0.03], [-0.02, -0.61, -0.03], [-0.6, -0.61, -0.03], [-0.6, -0.74, -0.1], [-0.02, -0.74, -0.1], [-0.02, -0.61, -0.1], [-0.6, -0.61, -0.1]],
    [[-0.75, -0.6, -0.03], [-0.02, -0.6, -0.03], [-0.02, -0.75, -0.03], [-0.75, -0.75, -0.03], [-0.75, -0.6, -0.1], [-0.02, -0.6, -0.1], [-0.02, -0.75, -0.1], [-0.75, -0.75, -0.1]]
    ])

for point in sampled_points:
    color = (point[3], point[4], point[5])
    ax.scatter(-point[0], point[1], point[2], color=color, s=10, alpha=0.2)

bbox_colors = ['r', 'b']
for idx in range(len(bboxes)):
    bbox = bboxes[idx]
    bbox_color = bbox_colors[idx]
    ax.scatter(-bbox[:, 0], bbox[:, 1], bbox[:, 2])
    # 计算凸包
    hull = ConvexHull(bbox)

    # 绘制凸包的外框
    for simplex in hull.simplices:
        ax.plot(-bbox[simplex, 0], bbox[simplex, 1], bbox[simplex, 2], bbox_color + '-')

    # # 检查点和边
    # print("Points:\n", bbox)
    # print("Hull vertices:\n", hull.vertices)

# 设置摄像机视角
ax.view_init(elev=45, azim=45)

# 显示图形
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
plt.savefig('./PointVisualization/fig/sine_wave.png', bbox_inches='tight', dpi=600, pad_inches=0)
print("image saved")
# plt.show()

