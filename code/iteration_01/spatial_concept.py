"""
迭代 1 - 实践 1: 可视化 3D 空间中的点

学习目标:
- 理解 3D 坐标系 (x, y, z)
- 使用 matplotlib 进行 3D 可视化
- 理解空间中点的表示

运行方法:
    python spatial_concept.py
"""

import sys
sys.path.insert(0, '/Users/gelin/Desktop/store/dev/python/3.10/WorldModel')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.matplotlib_config import setup_chinese_font

# 配置中文字体支持（自动检测操作系统）
setup_chinese_font(verbose=False)


def visualize_3d_point(point, label="Point"):
    """
    可视化一个 3D 点

    Args:
        point: np.array([x, y, z]) - 3D 坐标
        label: str - 点的标签名称
    """
    # 创建图形和 3D 坐标轴
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点 (红色，大小 100)
    ax.scatter(*point, c='red', s=100, marker='o')

    # 在点旁边显示标签
    ax.text(*point, label, fontsize=12)

    # 设置坐标轴标签
    ax.set_xlabel('X 轴 (宽度/米)')
    ax.set_ylabel('Y 轴 (深度/米)')
    ax.set_zlabel('Z 轴 (高度/米)')
    ax.set_title('3D 空间中的点')

    # 设置等比例坐标轴范围
    max_range = np.array([point[0], point[1], point[2]]).max()
    ax.set_xlim([0, max_range * 1.5])
    ax.set_ylim([0, max_range * 1.5])
    ax.set_zlim([0, max_range * 1.5])

    plt.show()


if __name__ == "__main__":
    # 定义一个沙发中心点的位置
    # x=2.0米（距离房间左墙）
    # y=1.5米（距离房间后墙）
    # z=0.5米（沙发座面高度）
    sofa_position = np.array([2.0, 1.5, 0.5])

    print("=" * 50)
    print("迭代 1 - 实践 1: 可视化 3D 空间中的点")
    print("=" * 50)
    print(f"\n沙发中心位置: {sofa_position}")
    print(f"  - X 坐标: {sofa_position[0]} 米")
    print(f"  - Y 坐标: {sofa_position[1]} 米")
    print(f"  - Z 坐标: {sofa_position[2]} 米")
    print("\n正在打开 3D 可视化窗口...")

    visualize_3d_point(sofa_position, "沙发中心")

    print("\n✅ 完成！你已经学会了在 3D 空间中表示和可视化点。")
