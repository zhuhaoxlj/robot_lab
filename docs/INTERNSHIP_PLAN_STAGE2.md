# 第二阶段：机械臂抓取策略设计（第 5-8 周）

## 📋 阶段目标

在强化学习基础之上，完成**机械臂抓取任务**的策略设计、训练与评估，掌握操作控制的核心技术。

---

## 🎯 Week 5-6: 机械臂仿真环境搭建

### Week 5: Isaac Lab 机械臂环境

#### 学习目标
- 理解机械臂的 URDF/USD 模型结构
- 掌握 Isaac Lab 中添加机械臂的方法
- 实现基础的运动控制接口

#### 具体任务

**Day 1-2: 机械臂模型导入**

```bash
# 任务：导入一个机械臂模型到 Isaac Lab
# 选择：Franka Emika Panda 或 Unitree 机械臂（如果有）

# 步骤：
1. 下载或准备 URDF/USD 文件
2. 创建 asset 配置文件
3. 验证模型加载成功
4. 测试关节控制接口
```

```python
# 示例代码结构
# source/robot_lab/assets/franka.py

from isaaclab.assets.articulation import ArticulationCfg

FRANKA_PANDA_CFG = ArticulationCfg(
    prim_path="{TARGET_ASSET_PATH}/Franka",
    spawn=sim_utils.SpawnCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    # ... 其他配置
)
```

**Day 3-4: 创建抓取环境**

```python
# source/robot_lab/tasks/manager_based/manipulation/reach/reach_env_cfg.py

@configclass
class FrankaReachEnvCfg(ManagerBasedRLEnvCfg):
    """机械臂到达任务环境配置"""

    # 场景配置
    scene = SceneCfg(num_envs=512, env_spacing=2.5)

    # 机器人配置
    robot: ArticulationCfg = FRANKA_PANDA_CFG

    # 任务相关配置
    ee_frame_name: str = "panda_hand"  # 末端执行器 frame
    target_asset_cfg = SceneEntityCfg("target")

    # 奖励配置
    rewards_policy = RewardTermCfg(func=mdp.reach_target, weight=1.0)

    # 终止配置
    terminations = TerminationTermCfg(
        time_out=DoneTerm(func=mdp.time_out, time_out=True),
        is_success=DoneTerm(func=mdp.reach_goal, params={"distance": 0.02}),
    )
```

**Day 5: 运动学和动力学基础**

```python
# 任务：实现正向运动学验证

def test_forward_kinematics():
    """验证正向运动学计算是否正确"""
    # 1. 设置已知的关节角度
    joint_angles = torch.tensor([0.0, -0.785, 0.0, 2.356, 0.0, 1.571, 0.785])

    # 2. 计算 EE 位置（使用 Isaac Lab 的 API）
    ee_pos = robot.data.body_state_w[:, ee_body_idx, :3]

    # 3. 与理论值对比（使用 Franka 官方 FK 验证）
    expected_pos = torch.tensor([0.487, 0.0, 0.3])

    # 4. 验证误差 < 1mm
    assert torch.allclose(ee_pos[0], expected_pos, atol=0.001)
```

#### 🔍 自验证节点

**验证 1：机械臂模型测试**
```bash
# 测试脚本
python scripts/tools/test_manipulator.py --robot=franka

# 验证项：
# ✅ 所有关节可以独立控制
# ✅ 关节角度、速度、力矩读取正确
# ✅ 无碰撞穿透问题
# ✅ 末端执行器位置计算准确
```

**验证 2：基础控制测试**
```python
# 测试 PID 控制
def test_pid_control():
    """测试关节的 PID 控制性能"""
    # 阶跃响应测试
    target_angle = 1.57  # 90 度

    # 记录响应曲线
    angles_over_time = []
    for step in range(100):
        action = pid.compute(target_angle, current_angle)
        env.step(action)
        angles_over_time.append(current_angle)

    # 评估：
    # - 上升时间 < 0.5s
    # - 超调量 < 5%
    # - 稳态误差 < 0.01 rad
```

---

### Week 6: 抓取任务设计

#### 学习目标
- 设计抓取任务的状态空间
- 实现抓取相关的观察和奖励函数
- 掌握稀疏奖励下的训练技巧

#### 具体任务

**Day 1-3: 简单 Reach 任务**

```python
# 任务：末端执行器到达随机目标点

class ReachEnvCfg:
    """到达任务配置"""

    # 观察空间
    observations.policy = [
        "ee_pos",           # 末端执行器位置 (3,)
        "ee_quat",          # 末端执行器姿态 (4,)
        "target_pos",       # 目标位置 (3,)
        "joint_pos",        # 关节角度 (7,)
        "joint_vel",        # 关节速度 (7,)
    ]  # 总计 24-dim

    # 动作空间
    actions = "joint_velocity"  # 7-dim

    # 奖励函数
    rewards = [
        # 主要奖励
        reach_target_reward(weight=1.0),

        # 辅助奖励
        joint_limit_penalty(weight=-0.1),
        smooth_action_reward(weight=-0.01),
    ]
```

**奖励函数实现：**
```python
def reach_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """计算到达目标的奖励"""
    # 获取 EE 和目标位置
    ee_pos = env.scene["robot"].data.body_state_w[:, ee_idx, :3]
    target_pos = env.scene["target"].data.root_pos_w[:, :3]

    # 计算距离
    distance = torch.norm(ee_pos - target_pos, dim=-1)

    # 指数衰减奖励（奖励接近目标的行为）
    reward = torch.exp(-distance / 0.1)  # 0.1m 为尺度参数

    return reward
```

**Day 4-5: Pick and Place 任务**

```python
# 任务：抓取物体并放置到目标位置

class PickPlaceEnvCfg:
    """抓取放置任务配置"""

    # 观察空间
    observations.policy = [
        # 机器人状态
        "ee_pos", "ee_quat",
        "joint_pos", "joint_vel",

        # 物体状态
        "object_pos", "object_quat",
        "object_rel_pos",    # 相对 EE 的位置

        # 目标位置
        "target_pos",

        # 接触状态
        "gripper_contact",   # 夹爪是否接触物体 (1,)
        "object_velocity",   # 物体速度 (3,)
    ]  # 总计约 40-dim

    # 任务阶段
    phases = ["reach", "grasp", "lift", "place"]
```

**多阶段奖励设计：**
```python
def pick_place_reward(env):
    """Pick and Place 奖励函数"""

    # 阶段 1: 接近物体
    if env.phase == "reach":
        dist_to_object = norm(ee_pos - object_pos)
        reward = exp(-dist_to_object / 0.1)

    # 阶段 2: 抓取物体
    elif env.phase == "grasp":
        # 奖励闭合夹爪且物体在夹爪内
        gripper_width = env.gripper_width
        object_in_gripper = is_object_in_gripper()

        reward = (
            1.0 if object_in_gripper else 0.0  # 稀疏奖励
            + 0.1 * (1.0 - gripper_width / max_width)  # 鼓励闭合
        )

    # 阶段 3: 提起物体
    elif env.phase == "lift":
        object_height = object_pos[:, 2]
        target_height = 0.3  # 30cm
        reward = min(object_height / target_height, 1.0)

    # 阶段 4: 放置到目标
    elif env.phase == "place":
        dist_to_target = norm(object_pos - target_pos)
        reward = exp(-dist_to_target / 0.1)

    return reward
```

#### 🔍 自验证节点

**验证 1：Reach 任务训练**
```bash
# 训练 Reach 任务
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Manipulation-Reach-Franka-v0 \
    --num_envs 512 \
    --max_iterations 2000 \
    --headless

# 成功标准：
# - Episode reward > 0.8（目标 1.0）
# - 成功率 > 80%（距离 < 2cm）
# - 收敛 iterations < 1500
```

**验证 2：Pick-Place 任务训练**
```bash
# 训练 Pick-Place 任务（需要 curriculum）
python train.py \
    --task=RobotLab-Isaac-Manipulation-PickPlace-Franka-v0 \
    --curriculum=easy_to_hard \
    --max_iterations 5000

# 成功标准：
# - 最终成功率 > 50%（抓起并放到目标位置）
# - 平均 episode 长度 < 200 steps
# - 能处理不同的物体初始位置
```

**验证 3：奖励函数有效性分析**
```python
# 分析各个奖励项的贡献
def analyze_reward_components(log_dir):
    """分析各个奖励组件"""

    # 加载训练日志
    logs = load_tensorboard_logs(log_dir)

    # 可视化每个奖励项随时间的变化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    reward_terms = [
        "reach_reward",
        "grasp_reward",
        "lift_reward",
        "place_reward",
        "joint_penalty",
        "action_penalty",
    ]

    for i, term in enumerate(reward_terms):
        axes[i//3, i%3].plot(logs[term])
        axes[i//3, i%3].set_title(term)
        axes[i//3, i%3].set_xlabel("Iterations")
        axes[i//3, i%3].set_ylabel("Reward")

    plt.savefig("reward_components_analysis.png")
```

---

## 🎯 Week 7-8: 稀疏奖励与探索策略

### Week 7: 稀疏奖励问题

#### 学习目标
- 理解稀疏奖励下的挑战
- 掌握 Curriculum Learning 在操作任务中的应用
- 学习 Hindsight Experience Replay (HER)

#### 具体任务

**Day 1-2: Curriculum Learning**

```python
# 为 Pick-Place 设计课程学习

class PickPlaceCurriculum:
    """Pick-Place 课程学习"""

    def __init__(self):
        self.stages = [
            # Stage 1: 固定的简单配置
            {
                "object_pos": "fixed_front",  # 物体固定在前面
                "target_pos": "fixed_above",   # 目标在正上方
                "difficulty": "easy",
            },
            # Stage 2: 小范围随机
            {
                "object_pos": "random_small",  # 10cm x 10cm 范围
                "target_pos": "random_small",
                "difficulty": "medium",
            },
            # Stage 3: 完全随机
            {
                "object_pos": "random_large",  # 50cm x 50cm 范围
                "target_pos": "random_large",
                "difficulty": "hard",
            },
        ]

    def get_stage(self, success_rate):
        """根据成功率自动升级"""
        if success_rate > 0.8:
            return self.stages[2]
        elif success_rate > 0.5:
            return self.stages[1]
        else:
            return self.stages[0]
```

**Day 3-5: Hindsight Experience Replay**

```python
# 实现 HER 算法

class HERReplayBuffer:
    """HER 经验回放缓冲区"""

    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity

    def add_episode(self, episode_transitions):
        """添加一个 episode 的所有转换"""
        # episode_transitions: [(s, a, r, s', done), ...]
        self.buffer.extend(episode_transitions)

    def sample_hindsight_transitions(self, batch_size, strategy='final'):
        """采样并使用 hindsight 标签重新标注奖励"""

        # 1. 随机采样 episodes
        episodes = random.sample(self.buffer, batch_size)

        # 2. 对于每个 episode，选择 hindsight 目标
        for episode in episodes:
            if strategy == 'final':
                # 使用 episode 最后达到的状态作为目标
                hindsight_goal = episode[-1].next_state
            elif strategy == 'future':
                # 随机选择 episode 中的一个未来状态作为目标
                t = random.randint(0, len(episode) - 1)
                hindsight_goal = episode[t].next_state
            elif strategy == 'random':
                # 随机选择 episode 中的一个状态作为目标
                t = random.randint(1, len(episode))
                hindsight_goal = episode[t].state

            # 3. 重新计算奖励（使用 hindsight 目标）
            for transition in episode[:episode.index(hindsight_goal) + 1]:
                # 计算到 hindsight 目标的距离
                distance = norm(transition.state['ee_pos'] - hindsight_goal['target_pos'])
                transition.reward = exp(-distance / 0.1)
                transition.done = (distance < 0.02)  # 2cm 阈值

        # 4. 返回重新标注的 batch
        return episodes
```

#### 🔍 自验证节点

**验证 1：Curriculum Learning 效果**
```bash
# 对比实验
python train.py --task=PickPlace --curriculum=False --run_name=no_curriculum
python train.py --task=PickPlace --curriculum=True --run_name=with_curriculum

# 分析：
python scripts/tools/compare_curriculum.py \
    --logdir1=logs/no_curriculum \
    --logdir2=logs/with_curriculum

# 成功标准：
# - 使用 CL 的收敛速度提升 > 30%
# - 使用 CL 的最终性能提升 > 20%
```

**验证 2：HER 效果验证**
```python
# 对比有无 HER 的性能差异
results = {
    "no_her": train_with_her(use_her=False),
    "her_final": train_with_her(use_her=True, strategy='final'),
    "her_future": train_with_her(use_her=True, strategy='future'),
}

# 评估：
for method, reward in results.items():
    print(f"{method}: {reward:.3f}")

# 成功标准：
# - HER (final) 比无 HER 提升 > 50%
# - HER (future) 效果最好
```

---

### Week 8: 模仿学习与强化学习结合

#### 学习目标
- 理解模仿学习的基本原理
- 掌握 Behavior Cloning (BC)
- 学习 BC + RL 的混合训练方法

#### 具体任务

**Day 1-2: 收集演示数据**

```python
# 收集专家演示

# 方案 1：使用键盘/鼠标遥控
python scripts/tools/collect_demonstrations.py \
    --task=PickPlace \
    --num_demos 100 \
    --control=keyboard

# 方案 2：使用运动规划器生成演示
python scripts/tools/generate_demos.py \
    --task=PickPlace \
    --planner=ompl \
    --num_demos 100

# 方案 3：从真实机器人录制
# (如果有真实机器人的话)
```

**Day 3-4: Behavior Cloning**

```python
# 实现行为克隆

class BehaviorCloningTrainer:
    """行为克隆训练器"""

    def __init__(self, demo_dataset):
        self.dataset = demo_dataset
        # demo_dataset: [(state, action), ...]

        # 创建神经网络
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),  # 假设动作已归一化到 [-1, 1]
        )

    def train(self, num_epochs=100):
        """训练 BC 策略"""

        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            # 随机打乱数据
            random.shuffle(self.dataset)

            for state, action in self.dataset:
                # 前向传播
                pred_action = self.policy(state)

                # 计算损失
                loss = loss_fn(pred_action, action)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证
            if epoch % 10 == 0:
                eval_loss = self.evaluate()
                print(f"Epoch {epoch}: Loss = {eval_loss:.4f}")

    def evaluate(self):
        """评估策略"""
        total_loss = 0.0
        for state, action in self.dataset:
            pred_action = self.policy(state)
            loss = nn.MSELoss()(pred_action, action)
            total_loss += loss.item()
        return total_loss / len(self.dataset)
```

**Day 5: BC + RL 混合训练**

```python
# 实现混合训练策略

class HybridTrainer:
    """BC + RL 混合训练器"""

    def __init__(self, demo_dataset, env):
        self.demo_dataset = demo_dataset
        self.env = env

        # 创建两个策略
        self.bc_policy = BCPolicy(demo_dataset)
        self.rl_policy = PPOPolicy(env)

    def train(self, num_iterations=2000):
        """混合训练流程"""

        for iteration in range(num_iterations):
            # 阶段 1: BC 预训练（前 500 iterations）
            if iteration < 500:
                # 使用 BC 损失更新策略
                self.update_policy_with_bc()

            # 阶段 2: BC + RL 混合（500-1000 iterations）
            elif iteration < 1000:
                # 混合损失：α * BC_loss + (1-α) * RL_loss
                alpha = 1.0 - (iteration - 500) / 500  # 从 1 逐渐降到 0

                bc_loss = self.compute_bc_loss()
                rl_loss = self.compute_rl_loss()

                loss = alpha * bc_loss + (1 - alpha) * rl_loss
                self.update_policy(loss)

            # 阶段 3: 纯 RL（1000+ iterations）
            else:
                # 标准 PPO 更新
                self.update_policy_with_ppo()

            # 评估和日志
            if iteration % 100 == 0:
                success_rate = self.evaluate()
                print(f"Iter {iteration}: Success Rate = {success_rate:.2%}")
```

#### 🔍 自验证节点

**验证 1：BC 性能评估**
```python
# 评估 BC 策略在测试集上的性能
def evaluate_bc_policy(policy, test_envs):
    """评估 BC 策略"""

    success_count = 0
    total_count = len(test_envs)

    for env in test_envs:
        # 使用 BC 策略执行任务
        state = env.reset()
        done = False

        while not done and steps < 200:
            action = policy(state)
            state, reward, done, _ = env.step(action)
            steps += 1

        if success:  # 根据任务定义成功条件
            success_count += 1

    return success_count / total_count

# 评估
bc_success_rate = evaluate_bc_policy(bc_policy, test_envs)
print(f"BC Policy Success Rate: {bc_success_rate:.2%}")

# 成功标准：
# - BC 策略成功率 > 30%（演示数据质量好）
# - BC 策略成功率 > 50%（演示数据质量很好）
```

**验证 2：混合训练效果**
```bash
# 对比三种训练方式
python train.py --method=bc --run_name=bc_only
python train.py --method=rl --run_name=rl_only
python train.py --method=hybrid --run_name=hybrid

# 可视化对比
python scripts/tools/plot_training_comparison.py \
    --logdirs=logs/bc_only,logs/rl_only,logs/hybrid

# 成功标准：
# - Hybrid 比纯 RL 收敛快 > 40%
# - Hybrid 比纯 BC 最终性能高 > 30%
# - Hybrid 结合了两者的优势
```

**验证 3：演示数据质量分析**
```python
# 分析演示数据的质量

def analyze_demo_quality(demo_dataset):
    """分析演示数据"""

    # 1. 覆盖度分析
    # 计算状态空间覆盖的百分比
    coverage = compute_state_coverage(demo_dataset)

    # 2. 多样性分析
    # 计算演示之间的平均距离
    diversity = compute_demo_diversity(demo_dataset)

    # 3. 一致性分析
    # 检查是否有矛盾的演示
    consistency = check_demo_consistency(demo_dataset)

    return {
        "coverage": coverage,
        "diversity": diversity,
        "consistency": consistency,
    }

# 成功标准：
# - 状态空间覆盖 > 70%
# - 演示多样性高（平均距离 > 阈值）
# - 无矛盾演示
```

---

## 📝 第二阶段总结检查点

### 必须完成（Must Have）

- [ ] **环境搭建**
  - 成功导入机械臂模型
  - 实现基础控制接口
  - 完成 Reach 和 Pick-Place 环境

- [ ] **策略训练**
  - Reach 任务成功率 > 80%
  - Pick-Place 任务成功率 > 50%
  - 理解稀疏奖励下的探索策略

- [ ] **技术掌握**
  - 会使用 Curriculum Learning
  - 理解 HER 的原理和应用
  - 掌握 BC + RL 混合训练

### 加分项（Nice to Have）

- [ ] 实现了新颖的探索策略
- [ ] 收集了真实机器人的演示数据
- [ ] 实现了 Sim-to-Real 的初步迁移
- [ ] 发表了操作任务相关的技术博客

---

## 🎯 下一步预告

第三阶段将进入**四足+机械臂全身控制**，这是最具挑战性的部分：
- 多自由度协调控制
- 强动力学耦合处理
- 全身运动规划
- 稳定性和鲁棒性验证
