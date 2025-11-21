"""
迭代 1 - 实践 3: 简单的"世界状态"表示

学习目标:
- 理解如何用代码表示 3D 世界
- 使用面向对象编程组织物体
- 实现简单的空间推理（距离、碰撞）

运行方法:
    python world_state.py
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Object3D:
    """
    3D 物体的简单表示

    属性:
        name: 物体名称
        position: 位置 [x, y, z]，单位：米
        size: 尺寸 [宽, 深, 高]，单位：米
        color: 颜色描述
    """
    name: str
    position: np.ndarray
    size: np.ndarray
    color: str

    def distance_to(self, other):
        """
        计算到另一个物体的距离（中心点距离）

        Args:
            other: 另一个 Object3D 实例

        Returns:
            float: 欧氏距离
        """
        return np.linalg.norm(self.position - other.position)

    def __repr__(self):
        return (f"Object3D(name='{self.name}', "
                f"pos={self.position}, size={self.size})")


class WorldState:
    """
    世界状态的表示

    管理场景中的所有物体，提供查询和推理功能
    """
    def __init__(self):
        self.objects: List[Object3D] = []

    def add_object(self, obj: Object3D):
        """添加物体到世界"""
        self.objects.append(obj)
        print(f"✅ 添加物体: {obj.name}")

    def describe(self):
        """描述当前世界状态"""
        print(f"\n🌍 当前世界包含 {len(self.objects)} 个物体:")
        for obj in self.objects:
            print(f"  - {obj.name}:")
            print(f"      位置: {obj.position}")
            print(f"      尺寸: {obj.size}")
            print(f"      颜色: {obj.color}")

    def check_collision(self):
        """
        检查物体是否可能重叠（简化版）

        使用物体中心距离和尺寸进行粗略判断
        """
        print("\n🔍 检查物体碰撞...")
        has_collision = False

        for i, obj1 in enumerate(self.objects):
            for obj2 in self.objects[i+1:]:
                distance = obj1.distance_to(obj2)

                # 估算最小安全距离（两个物体对角线的一半）
                min_distance = (
                    np.linalg.norm(obj1.size) +
                    np.linalg.norm(obj2.size)
                ) / 2

                if distance < min_distance:
                    print(f"  ⚠️  {obj1.name} 和 {obj2.name} "
                          f"可能重叠! (距离: {distance:.2f}m)")
                    has_collision = True

        if not has_collision:
            print("  ✅ 没有检测到碰撞")

    def find_nearest(self, obj: Object3D):
        """找到离指定物体最近的其他物体"""
        if len(self.objects) < 2:
            return None

        nearest = None
        min_dist = float('inf')

        for other in self.objects:
            if other != obj:
                dist = obj.distance_to(other)
                if dist < min_dist:
                    min_dist = dist
                    nearest = other

        return nearest, min_dist


def create_living_room():
    """
    创建一个简单的客厅场景

    Returns:
        WorldState: 包含家具的世界状态
    """
    world = WorldState()

    # 创建家具对象
    sofa = Object3D(
        name="灰色沙发",
        position=np.array([2.0, 1.5, 0.5]),
        size=np.array([2.0, 0.9, 0.8]),  # 宽2米，深0.9米，高0.8米
        color="灰色"
    )

    table = Object3D(
        name="玻璃茶几",
        position=np.array([2.0, 3.0, 0.3]),
        size=np.array([1.2, 0.6, 0.4]),
        color="透明"
    )

    tv = Object3D(
        name="55寸电视",
        position=np.array([2.0, 5.5, 1.0]),
        size=np.array([1.2, 0.1, 0.7]),
        color="黑色"
    )

    lamp = Object3D(
        name="落地台灯",
        position=np.array([0.5, 1.5, 1.2]),
        size=np.array([0.3, 0.3, 1.5]),
        color="白色"
    )

    # 添加到世界
    world.add_object(sofa)
    world.add_object(table)
    world.add_object(tv)
    world.add_object(lamp)

    return world, sofa, table


if __name__ == "__main__":
    print("=" * 50)
    print("迭代 1 - 实践 3: 世界状态表示")
    print("=" * 50)

    # 创建客厅场景
    world, sofa, table = create_living_room()

    # 描述世界
    world.describe()

    # 检查碰撞
    world.check_collision()

    # 计算关系
    print(f"\n📏 空间关系:")
    distance = sofa.distance_to(table)
    print(f"  沙发到茶几的距离: {distance:.2f} 米")

    # 找到离沙发最近的物体
    nearest, dist = world.find_nearest(sofa)
    print(f"  离沙发最近的物体: {nearest.name} ({dist:.2f} 米)")

    print("\n💡 思考:")
    print("  这就是世界模型的基础 - 用数据结构表示世界！")
    print("  Marble 的世界表示比这复杂得多，但核心思想相同。")
    print("\n✅ 完成！你已经理解了世界状态的基本表示。")
