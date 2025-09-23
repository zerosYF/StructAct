import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata

# 原始数据
theta = np.array([0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
lam   = np.array([0.16, 0.2, 0.24, 0.16, 0.2, 0.24, 0.16, 0.2, 0.24])
acc   = np.array([0.475, 0.475, 0.425, 0.55, 0.5, 0.375, 0.475, 0.425, 0.425])

# 构建网格
grid_x, grid_y = np.meshgrid(
    np.linspace(min(lam), max(lam), 100),
    np.linspace(min(theta), max(theta), 100)
)

# 插值
grid_z = griddata((lam, theta), acc, (grid_x, grid_y), method='cubic')

# 绘制平滑热力图
plt.figure(figsize=(6,5))
im = plt.imshow(grid_z, extent=(min(lam), max(lam), min(theta), max(theta)),
                origin='lower', aspect='auto', cmap='YlGnBu')
plt.colorbar(im, label='Accuracy')
plt.scatter(lam, theta, c='red', marker='o')  # 标出原始采样点
plt.title("Smoothed Accuracy Heatmap (theta vs lambda)")
plt.xlabel("lambda")
plt.ylabel("theta")
plt.savefig("accuracy_heatmap.png", dpi=300, bbox_inches='tight')  # 保存为 PNG，300dpi
plt.show()