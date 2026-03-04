# 从零学习强化学习 - 代码示例

这个目录包含从零开始学习强化学习的完整代码示例。

## 📁 文件组织

```
scratch/
├── 00_empty_scene.py           # 示例1：空场景
├── 01_cube_physics.py          # 示例2：物理物体
├── 02_terrain_generation.py    # 示例3：地形生成
├── 03_import_robot.py          # 示例4：导入机器人
├── 04_joint_control.py         # 示例5：关节控制
├── 05_observation_design.py    # 示例6：观察空间
├── 06_action_design.py         # 示例7：动作空间
├── 07_reward_design.py         # 示例8：奖励函数
├── 08_full_env.py              # 示例9：完整环境
├── 09_test_env.py              # 示例10：测试环境
├── 10_policy_network.py        # 示例11：策略网络
├── 11_value_network.py         # 示例12：价值网络
├── 12_compute_gae.py           # 示例13：GAE计算
├── 13_ppo_trainer.py           # 示例14：PPO训练器
├── 14_train.py                 # 示例15：运行训练
├── 15_visualize.py             # 示例16：可视化
└── 16_test_policy.py           # 示例17：测试策略
```

## 🚀 快速开始

### 1. 运行第一个示例

```bash
# 确保设置了 DISPLAY 环境变量
export DISPLAY=:1

# 运行空场景示例
python source/robot_lab/robot_lab/tasks/scratch/00_empty_scene.py
```

### 2. 预期输出

你应该能看到一个窗口显示：
- 灰色无限地面
- 网格辅助线
- 可以用鼠标控制视角

### 3. 学习顺序

按照文件编号顺序学习，每个文件都建立在前一个的基础上：

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16
```

## 📊 学习进度

### 第一阶段：仿真基础（00-04）
- [x] 00_empty_scene.py
- [x] 01_cube_physics.py
- [ ] 02_terrain_generation.py
- [ ] 03_import_robot.py
- [ ] 04_joint_control.py

### 第二阶段：MDP设计（05-09）
- [ ] 05_observation_design.py
- [ ] 06_action_design.py
- [ ] 07_reward_design.py
- [ ] 08_full_env.py
- [ ] 09_test_env.py

### 第三阶段：PPO算法（10-14）
- [ ] 10_policy_network.py
- [ ] 11_value_network.py
- [ ] 12_compute_gae.py
- [ ] 13_ppo_trainer.py
- [ ] 14_train.py

### 第四阶段：训练调试（15-16）
- [ ] 15_visualize.py
- [ ] 16_test_policy.py

## 🎯 每个示例的学习目标

### 00_empty_scene.py
**目标**：理解 Isaac Sim 的基本架构
- SimulationApp 的作用
- Stage 和 Prim 的概念
- 仿真主循环

### 01_cube_physics.py
**目标**：理解物理引擎
- 刚体动力学
- 重力和碰撞
- 物理步进

### ... 更多示例在学习路径文档中 ...

## 💡 学习建议

1. **不要跳过示例**：每个示例都建立了重要的基础概念
2. **修改代码实验**：改变参数，观察效果
3. **阅读注释**：代码中有详细的解释
4. **记录笔记**：写下你的理解和疑问

## 🔧 常见问题

### Q: 为什么窗口是黑屏？
A: 检查 DISPLAY 环境变量是否正确设置

### Q: 如何停止仿真？
A: 按 Ctrl+C

### Q: 看不到物体？
A: 检查物体位置是否在相机视野内，尝试用鼠标旋转视角

## 📚 相关文档

- 完整学习路径：`docs/LEARNING_PATH_FROM_SCRATCH.md`
- Isaac Lab 官方文档：https://isaac-sim.github.io/IsaacLab/

---

**开始你的强化学习之旅吧！** 🚀
