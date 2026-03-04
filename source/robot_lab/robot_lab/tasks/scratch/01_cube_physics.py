#!/usr/bin/env python3
"""
第二个示例：添加物理物体

学习目标：
1. 理解 World 类的作用 —— 管理物理引擎的步进
2. 理解 simulation_app.update() vs world.step() 的区别
3. 刚体动力学：重力、碰撞、反弹

核心概念：
- SimulationApp: 负责渲染窗口和 Kit 运行时
- World: 负责物理引擎的创建、重置、步进
- DynamicCuboid: 自带刚体（Rigid Body）+ 碰撞体（Collider）的立方体

⚠️ 关键理解：
  simulation_app.update() → 只刷新渲染，物理不动
  world.step()            → 驱动物理引擎 + 渲染，物体才会运动
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.api.objects import DynamicSphere
import numpy as np  # noqa: E402


def create_scene_with_cube():
    """创建场景并添加一个立方体，观察物理自由落体"""

    # ========================================
    # 第一步：创建 World（物理引擎管理器）
    # ========================================
    # physics_dt: 物理步进间隔（1/60s = 约60Hz）
    # rendering_dt: 渲染帧间隔
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)

    # ========================================
    # 第二步：创建场景物体
    # ========================================
    # 使用 world 自带方法添加地面（会自动添加默认灯光）
    world.scene.add_default_ground_plane()

    # 动态立方体（自带刚体物理 + 碰撞体）
    # - size: float 标量，立方体边长（不是数组！）
    # - mass: 质量 kg
    # - position: 初始位置 [x, y, z]

    for i in range(10):
        falling_cube = DynamicCuboid(
            prim_path=f"/World/cubes/cube{i}",
            name=f"falling_cube_{i}",
            position=np.array(
                [i + 0.2, i * 0.2, 2.0 + i * 0.1]
            ),  # 从 2m 高处开始，每个立方体间隔 0.1m
            size=0.5,  # 边长 0.1m
            mass=1.0,  # 质量 1kg
            color=np.array([0.0, 1.0, 0.0]),  # 红色
        )

    ball = DynamicSphere(
        prim_path="/World/balls",
        name="falling_ball",
        position=np.array([2.0, 2.0, 2.0]),
        radius=0.5,  # 半径 0.5m
        mass=1.0,  # 质量 1kg
        color=np.array([1.0, 0.0, 0.0]),
    )

    # ========================================
    # 第三步：重置物理引擎（必须调用！）
    # ========================================
    # world.reset() 做了什么：
    # 1. 初始化 PhysX 仿真后端
    # 2. 将所有物体设置到初始状态
    # 3. 没有这一步，物理引擎不会激活
    world.reset()

    print("=" * 60, flush=True)
    print("场景已创建！", flush=True)
    print("=" * 60, flush=True)
    print("\n你应该能看到：", flush=True)
    print("  - 地面（z=0）", flush=True)
    print("  - 一个红色立方体从 z=2m 掉落", flush=True)
    print("\n按 Ctrl+C 停止\n", flush=True)

    # ========================================
    # 第四步：仿真主循环
    # ========================================
    step_count = 0
    sim_dt = world.get_physics_dt()

    try:
        while simulation_app.is_running():
            # 检查仿真是否被停止（用户在 GUI 点了 Stop）
            if world.is_stopped():
                break

            # 如果仿真暂停，仍需调用 step 保持 GUI 响应
            if not world.is_playing():
                world.step(render=True)
                continue

            # ✅ 核心调用：world.step() 驱动物理 + 渲染
            world.step()
            step_count += 1

            # 每 60 步（约 1 秒）打印一次位置
            if step_count % 60 == 0:
                pos, _ = falling_cube.get_world_pose()
                elapsed = step_count * sim_dt
                print(f"时间 {elapsed:.1f}s: 高度 = {pos[2]:.3f}m", flush=True)
                print(f"位置：{pos}", flush=True)

    except KeyboardInterrupt:
        print("\n\n仿真已停止", flush=True)

    simulation_app.close()
    print("程序退出", flush=True)


if __name__ == "__main__":
    create_scene_with_cube()
