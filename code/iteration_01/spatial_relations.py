"""
迭代 1 - 实践 2: 可视化物体之间的空间关系

学习目标:
- 理解多个物体在 3D 空间中的位置
- 计算物体之间的距离
- 可视化空间关系

运行方法:
    python spatial_relations.py
"""

import sys
sys.path.insert(0, '/Users/gelin/Desktop/store/dev/python/3.10/WorldModel')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.matplotlib_config import setup_chinese_font

# 配置中文字体支持（自动检测操作系统）
setup_chinese_font(verbose=False)


def visualize_spatial_relations():
    """
    可视化客厅中的多个物体及其空间关系
    """
    # 定义物体及其位置 (x, y, z 坐标，单位：米)
    objects = {
        '沙发': np.array([2.0, 1.5, 0.5]),     # 靠后墙
        '茶几': np.array([2.0, 3.0, 0.3]),     # 沙发前面
        '电视': np.array([2.0, 5.5, 1.0]),     # 对面墙上
        '台灯': np.array([0.5, 1.5, 1.2])      # 沙发旁边
    }

    print("=" * 50)
    print("迭代 1 - 实践 2: 可视化空间关系")
    print("=" * 50)
    print("\n客厅物体列表:")
    for name, pos in objects.items():
        print(f"  {name}: 位置 {pos}")

    # 创建 3D 图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 为每个物体分配颜色
    colors = ['red', 'blue', 'green', 'orange']

    # 绘制所有物体
    for (name, pos), color in zip(objects.items(), colors):
        ax.scatter(*pos, c=color, s=200, marker='o', label=name)
        ax.text(*pos, name, fontsize=10)

    # 绘制物体之间的关系线（沙发到茶几）
    sofa_pos = objects['沙发']
    table_pos = objects['茶几']
    ax.plot(
        [sofa_pos[0], table_pos[0]],
        [sofa_pos[1], table_pos[1]],
        [sofa_pos[2], table_pos[2]],
        'k--', alpha=0.3, linewidth=2, label='空间关系'
    )

    # 设置坐标轴
    ax.set_xlabel('X 轴 (宽度/米)')
    ax.set_ylabel('Y 轴 (深度/米)')
    ax.set_zlabel('Z 轴 (高度/米)')
    ax.set_title('客厅物体的空间关系', fontsize=14)
    ax.legend()

    # 设置视角
    ax.view_init(elev=20, azim=45)

    plt.show()

    # 计算并输出关系
    print("\n📏 物体之间的距离:")
    distance = np.linalg.norm(sofa_pos - table_pos)
    print(f"  沙发到茶几: {distance:.2f} 米")

    tv_pos = objects['电视']
    distance_tv = np.linalg.norm(sofa_pos - tv_pos)
    print(f"  沙发到电视: {distance_tv:.2f} 米")

    lamp_pos = objects['台灯']
    distance_lamp = np.linalg.norm(sofa_pos - lamp_pos)
    print(f"  沙发到台灯: {distance_lamp:.2f} 米")

    print("\n✅ 完成！你已经学会了表示和可视化空间关系。")


if __name__ == "__main__":
    visualize_spatial_relations()
