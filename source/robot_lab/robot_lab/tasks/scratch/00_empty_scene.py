#!/usr/bin/env python3
"""
第一个示例：创建空的仿真场景

这是从零学习强化学习的起点 - 理解仿真的基本概念
"""

from isaacsim import SimulationApp

# ⚠️ 重要：必须在 SimulationApp() 之后导入其他 Isaac Sim 模块
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.objects import GroundPlane


def create_empty_scene():
    """
    创建最简单的仿真环境

    Isaac Sim 的基本概念：
    1. SimulationApp: 仿真应用程序实例
    2. Stage: 3D 场景（类似 Unity/Unreal 的场景）
    3. Prim: 场景中的基本元素（物体、灯光等）
    4. Physics: 物理引擎
    """

    # 创建地面
    # prim_path: 物体在场景中的路径（类似文件路径）
    ground = GroundPlane(prim_path="/World/ground")

    print("=" * 60, flush=True)
    print("🎮 仿真环境已启动！", flush=True)
    print("=" * 60, flush=True)
    print("\n你应该能看到一个窗口，显示：", flush=True)
    print("  - 灰色地面（无限平面）", flush=True)
    print("  - 网格辅助线", flush=True)
    print("  - 黑色背景", flush=True)
    print("\n📝 基本操作：", flush=True)
    print("  - 左键拖动：旋转视角", flush=True)
    print("  - 中键拖动：平移视角", flush=True)
    print("  - 滚轮：缩放", flush=True)
    print("\n按 Ctrl+C 停止仿真\n", flush=True)
    print("=" * 60, flush=True)

    # 仿真主循环
    # simulation_app.is_running() 会在窗口关闭时返回 False
    # 对于简单场景，我们只需要保持循环，让仿真自动运行
    try:
        while simulation_app.is_running():
            # Isaac Sim 会自动步进仿真
            # 这里可以添加自定义逻辑
            pass
    except KeyboardInterrupt:
        print("\n\n🛑 仿真已停止")

    # 关闭仿真
    simulation_app.close()
    print("✅ 清理完成，程序退出")


if __name__ == "__main__":
    create_empty_scene()
    # 关闭仿真（会在函数末尾调用，这里保持一致性）
