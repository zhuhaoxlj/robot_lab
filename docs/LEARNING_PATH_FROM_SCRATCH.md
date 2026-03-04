# 从零开始学习强化学习机器人控制

## 🎯 核心理念

**不依赖现成训练脚本，从底层逐步构建完整理解**

```
空场景 → 机器人导入 → 基础控制 → PPO算法 → 训练调试 → 高级优化
```

---

## 📅 学习路径（8-10周）

### 第一阶段：Isaac Sim 基础（Week 1-2）

#### Week 1: 场景与物理

**目标**：理解仿真环境的基本组成

**Day 1-2: 创建空场景**
```python
# tasks/scratch/00_empty_scene.py
from isaacsim import SimulationApp

# 创建最简单的仿真环境
simulation_app = SimulationApp({"headless": False})

# 添加地面
from omni.isaac.core.objects import GroundPlane
ground = GroundPlane(prim_path="/World/ground")

# 运行仿真
while simulation_app.is_running():
    simulation_app.step()

simulation_app.close()
```

**验证**：能看到空地面场景

**Day 3-4: 创建地形**
```python
# tasks/scratch/01_terrain_generation.py
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np

# 创建平地
# 创建粗糙地形（高度场）
# 创建台阶地形
```

**验证**：能生成不同类型地形

**Day 5: 物理材质与交互**
```python
# 添加物理材质
# 设置摩擦力、弹性
# 验证物理效果（球体滚动、碰撞等）
```

#### Week 2: 机器人导入与控制

**目标**：导入机器人并理解其结构

**Day 1-2: 导入 Unitree A1**
```python
# tasks/scratch/02_import_robot.py
from isaacsim.core.api.scenes import Scene
from isaacsim.core.objects import usd

# 方法1：从 URDF 导入
# 方法2：从 USD 导入
# 理解：关节、连杆、刚体、碰撞体
```

**理解重点**：
- 机器人的 URDF/USD 结构
- 关节类型（revolute, prismatic等）
- 自由度（DoF）概念
- 质量中心和惯性

**Day 3-4: 关节控制**
```python
# tasks/scratch/03_joint_control.py
# 读取当前关节角度
# 设置目标关节角度（位置控制）
# 设置关节速度
# 设置关节扭矩

# 验证：手动让机器人抬起一条腿
```

**Day 5: 基础运动测试**
```python
# tasks/scratch/04_basic_motion.py
# 实现简单的周期性动作（如坐下起立）
# 理解动作周期（action frequency）
```

**验证**：机器人能执行简单的预定义动作

---

### 第二阶段：强化学习环境搭建（Week 3-4）

#### Week 3: MDP 设计

**目标**：理解并实现完整的 MDP

**Day 1: 理论学习**
- 马尔可夫决策过程（MDP）定义
- 状态空间、动作空间、奖励函数、转移概率
- 折扣因子 γ 的作用

**Day 2-3: 状态空间设计**
```python
# tasks/scratch/05_observation_design.py
"""
设计观察空间（Observation Space）

观察什么？
1. 机器人本体状态
   - 基座位置、速度、姿态（四元数）
   - 关节角度、角速度
   - 重力方向投影

2. 环境信息
   - 目标速度命令
   - 地形高度扫描（可选）

3. 历史信息
   - 上一步动作
"""

class ObservationManager:
    def get_observations(self, env):
        obs = []
        # 基座角速度 (3)
        obs.append(env.base_angular_velocity)
        # 重力方向 (3)
        obs.append(env.projected_gravity)
        # 速度命令 (3)
        obs.append(env.velocity_command)
        # 关节位置 (12)
        obs.append(env.joint_positions)
        # 关节速度 (12)
        obs.append(env.joint_velocities)
        # 上一步动作 (12)
        obs.append(env.last_actions)

        return np.concatenate(obs, axis=-1)  # (45,)
```

**Day 4: 动作空间设计**
```python
# tasks/scratch/06_action_design.py
"""
设计动作空间（Action Space）

动作类型：
1. 关节位置控制（最常用）
2. 关节速度控制
3. 关节扭矩控制

对于 A1：12 个关节 → 12 维动作空间
"""

class ActionManager:
    def __init__(self, num_actions=12):
        # 动作范围：[-1, 1] 标准化
        # 需要缩放到实际关节角度范围
        self.action_scale = 0.5  # 弧度

    def process_actions(self, actions):
        # 将 [-1, 1] 映射到关节角度偏移
        target_angles = self.current_angles + actions * self.action_scale
        return target_angles
```

**Day 5: 奖励函数设计**
```python
# tasks/scratch/07_reward_design.py
"""
设计奖励函数（Reward Function）

奖励 = 正向奖励 - 负向惩罚

主要奖励项：
1. 线速度跟踪（前进、侧向）
2. 角速度跟踪（旋转）

惩罚项：
1. 关节扭矩（平滑运动）
2. 关节加速度（防止抖动）
3. 关节位置限制（防止超限）
4. 机器人倒地（高度、姿态）
"""

class RewardManager:
    def compute_rewards(self, env):
        rewards = {}

        # 1. 线速度跟踪奖励
        # reward = exp(-((current_vel - target_vel)^2) / sigma^2)
        rewards["track_lin_vel"] = self._velocity_tracking_reward(env)

        # 2. 角速度跟踪奖励
        rewards["track_ang_vel"] = self._angular_velocity_reward(env)

        # 3. 关节扭矩惩罚
        rewards["joint_torques"] = -1.0 * torch.mean(env.joint_torques**2)

        # 4. 机器人高度惩罚
        rewards["base_height"] = -2.0 * (env.base_height - 0.4)**2

        # 总奖励
        total_reward = sum(rewards.values())
        return total_reward, rewards
```

#### Week 4: 环境集成

**目标**：组装完整的 RL 环境

**Day 1-3: 实现完整环境**
```python
# tasks/scratch/08_full_env.py
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.robots import Robot
import torch

class QuadrupedEnv:
    def __init__(self, num_envs=1, device="cuda:0"):
        self.num_envs = num_envs
        self.device = device

        # 初始化仿真
        self._setup_scene()
        self._setup_robot()

        # 初始化管理器
        self.obs_manager = ObservationManager()
        self.action_manager = ActionManager()
        self.reward_manager = RewardManager()

        # 重置环境
        self.reset()

    def reset(self):
        """重置所有环境到初始状态"""
        # 随机化初始姿态
        # 随机化初始位置
        return self.get_observations()

    def step(self, actions):
        """执行一步仿真"""
        # 1. 应用动作
        self._apply_actions(actions)

        # 2. 物理仿真步进
        for _ in range(self.physics_steps_per_frame):
            self.sim.step()

        # 3. 计算奖励
        reward, reward_dict = self.reward_manager.compute_rewards(self)

        # 4. 检查终止条件
        done = self._check_termination()

        # 5. 获取新观察
        obs = self.get_observations()

        return obs, reward, done, reward_dict

    def get_observations(self):
        return self.obs_manager.get_observations(self)
```

**Day 4-5: 测试环境**
```python
# tasks/scratch/09_test_env.py
# 验证 MDP 的完整性
# - 观察维度是否正确
# - 动作维度是否正确
# - 奖励值范围是否合理
# - 终止条件是否触发

# 运行随机策略测试
env = QuadrupedEnv(num_envs=10)
obs = env.reset()

for i in range(1000):
    actions = np.random.randn(10, 12)  # 随机动作
    obs, reward, done, info = env.step(actions)
    print(f"Step {i}: reward={reward:.2f}")
```

**验证**：环境能正常运行，观察/动作/奖励维度正确

---

### 第三阶段：PPO 算法实现（Week 5-6）

#### Week 5: 策略网络与价值网络

**目标**：实现 Actor-Critic 架构

**Day 1-2: 理论学习**
```python
"""
PPO 核心概念：

1. 策略梯度（Policy Gradient）
   - 目标：最大化期望回报
   - 梯度：∇J(θ) = E[∇log π(a|s) * A(s,a)]

2. 优势函数（Advantage Function）
   - A(s,a) = Q(s,a) - V(s)
   - 表示当前动作比平均好多少

3. GAE（Generalized Advantage Estimation）
   - 平衡偏差和方差
   - A_t = Σ(γλ)^l δ_t+l
"""
```

**Day 3-4: 实现策略网络**
```python
# tasks/scratch/10_policy_network.py
import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    """策略网络（Actor）"""

    def __init__(self, obs_dim=45, action_dim=12, hidden_dim=[256, 128]):
        super().__init__()

        layers = []
        input_dim = obs_dim
        for dim in hidden_dim:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ELU()
            ])
            input_dim = dim

        # 输出层：动作均值
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # 动作标准差（log_std 为可学习参数）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        """前向传播，返回动作分布"""
        mean = self.network(obs)
        std = torch.exp(self.log_std)

        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        return dist

    def act(self, obs, deterministic=False):
        """采样动作"""
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        return action

# 测试
actor = ActorNetwork()
obs = torch.randn(1, 45)
action = actor.act(obs)
print(f"Action shape: {action.shape}")  # (1, 12)
```

**Day 5: 实现价值网络**
```python
# tasks/scratch/11_value_network.py
class CriticNetwork(nn.Module):
    """价值网络（Critic）"""

    def __init__(self, obs_dim=48, hidden_dim=[256, 128]):
        super().__init__()

        layers = []
        input_dim = obs_dim
        for dim in hidden_dim:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ELU()
            ])
            input_dim = dim

        # 输出层：状态价值 V(s)
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        """前向传播，返回状态价值"""
        value = self.network(obs)
        return value.squeeze(-1)  # (num_envs,)

# 测试
critic = CriticNetwork()
obs = torch.randn(1, 48)
value = critic(obs)
print(f"Value shape: {value.shape}")  # (1,)
```

#### Week 6: PPO 核心算法

**目标**：实现 PPO 训练循环

**Day 1-2: 理解 PPO Clip 目标**
```python
"""
PPO Clip 目标函数：

L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

其中：
- r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) （概率比率）
- A_t：优势函数
- ε：clip 参数（通常 0.2）

作用：
- 防止策略更新过大
- 当 A_t > 0 时，限制 r_t 不超过 1+ε
- 当 A_t < 0 时，限制 r_t 不低于 1-ε
"""

def compute_ppo_loss(logits, old_logits, actions, advantages, epsilon=0.2):
    """
    计算 PPO Clip 损失

    Args:
        logits: 新策略的 log_prob
        old_logits: 旧策略的 log_prob
        actions: 采取的动作
        advantages: 优势函数值
        epsilon: clip 参数

    Returns:
        ppo_loss: PPO 损失
    """
    # 计算概率比率
    ratio = torch.exp(logits - old_logits)

    # Clipped 损失
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    return policy_loss
```

**Day 3-4: 实现 GAE**
```python
# tasks/scratch/12_compute_gae.py
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计（GAE）

    Args:
        rewards: (num_steps, num_envs)
        values: (num_steps, num_envs)
        dones: (num_steps, num_envs)
        gamma: 折扣因子
        lam: GAE 参数

    Returns:
        advantages: (num_steps, num_envs)
        returns: (num_steps, num_envs)
    """
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    # 从后向前计算
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = 0
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        # TD 残差
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

        # GAE
        advantages[t] = last_advantage = delta + gamma * lam * next_non_terminal * last_advantage

    # 计算回报
    returns = advantages + values

    return advantages, returns

# 测试
rewards = torch.randn(100, 10)
values = torch.randn(100, 10)
dones = torch.zeros(100, 10)

advantages, returns = compute_gae(rewards, values, dones)
print(f"Advantages shape: {advantages.shape}")
```

**Day 5: 实现完整训练循环**
```python
# tasks/scratch/13_ppo_trainer.py
class PPOTrainer:
    def __init__(self, env, actor, critic, config):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.config = config

        # 优化器
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        # 经验缓冲区
        self.buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }

    def collect_rollouts(self, num_steps):
        """收集经验"""
        obs = self.env.reset()

        for _ in range(num_steps):
            with torch.no_grad():
                # 获取动作
                dist = self.actor(obs)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

                # 获取价值
                value = self.critic(obs)

            # 环境步进
            next_obs, reward, done, _ = self.env.step(action)

            # 存储经验
            self.buffer['observations'].append(obs)
            self.buffer['actions'].append(action)
            self.buffer['rewards'].append(reward)
            self.buffer['values'].append(value)
            self.buffer['log_probs'].append(log_prob)
            self.buffer['dones'].append(done)

            obs = next_obs

        # 转换为张量
        for key in self.buffer:
            self.buffer[key] = torch.stack(self.buffer[key], dim=0)

    def update_policy(self):
        """更新策略"""
        # 1. 计算 GAE 和 returns
        advantages, returns = compute_gae(
            self.buffer['rewards'],
            self.buffer['values'],
            self.buffer['dones'],
            gamma=self.config['gamma'],
            lam=self.config['lam']
        )

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. 多次更新
        for _ in range(self.config['num_epochs']):
            # 前向传播
            dist = self.actor(self.buffer['observations'])
            new_log_prob = dist.log_prob(self.buffer['actions']).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            values = self.critic(self.buffer['observations'])

            # 计算损失
            policy_loss = compute_ppo_loss(
                new_log_prob,
                self.buffer['log_probs'],
                self.buffer['actions'],
                advantages
            )

            value_loss = nn.MSELoss()(values, returns)

            # 总损失
            actor_loss = policy_loss - self.config['entropy_coef'] * entropy
            critic_loss = value_loss

            # 反向传播
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # 清空缓冲区
        for key in self.buffer:
            self.buffer[key] = []

    def train(self, num_iterations):
        """完整训练循环"""
        for iteration in range(num_iterations):
            # 收集经验
            self.collect_rollouts(self.config['num_steps'])

            # 更新策略
            self.update_policy()

            # 日志
            if iteration % 10 == 0:
                mean_reward = self.buffer['rewards'].mean().item()
                print(f"Iteration {iteration}: Mean Reward = {mean_reward:.2f}")
```

---

### 第四阶段：训练与调试（Week 7-8）

#### Week 7: 训练与可视化

**Day 1-2: 运行第一次训练**
```python
# tasks/scratch/14_train.py
from scratch_13_ppo_trainer import PPOTrainer
from scratch_08_full_env import QuadrupedEnv
from scratch_10_policy_network import ActorNetwork
from scratch_11_value_network import CriticNetwork

# 创建环境
env = QuadrupedEnv(num_envs=512, device="cuda:0")

# 创建网络
actor = ActorNetwork(obs_dim=45, action_dim=12).to("cuda:0")
critic = CriticNetwork(obs_dim=48).to("cuda:0")

# 训练配置
config = {
    'num_steps': 24,  # 每次收集 24 步
    'num_epochs': 5,
    'gamma': 0.99,
    'lam': 0.95,
    'entropy_coef': 0.01,
}

# 创建训练器
trainer = PPOTrainer(env, actor, critic, config)

# 开始训练
trainer.train(num_iterations=1000)

# 保存模型
torch.save(actor.state_dict(), "actor.pt")
torch.save(critic.state_dict(), "critic.pt")
```

**Day 3-4: 可视化训练过程**
```python
# tasks/scratch/15_visualize.py
# 使用 TensorBoard 记录训练过程
# 绘制奖励曲线
# 录制视频

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/scratch_ppo")

# 记录奖励
writer.add_scalar("Reward/mean", mean_reward, iteration)
writer.add_scalar("Reward/max", max_reward, iteration)

# 记录损失
writer.add_scalar("Loss/policy", policy_loss, iteration)
writer.add_scalar("Loss/value", value_loss, iteration)
```

**Day 5: 测试训练好的策略**
```python
# tasks/scratch/16_test_policy.py
# 加载模型
actor = ActorNetwork()
actor.load_state_dict(torch.load("actor.pt"))
actor.eval()

# 测试
env = QuadrupedEnv(num_envs=10)
obs = env.reset()

for i in range(1000):
    with torch.no_grad():
        action = actor.act(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
```

#### Week 8: 调试与优化

**Day 1-2: 诊断训练问题**
```python
# 常见问题检查清单
# 1. 奖励值是否合理（不会被某项主导）
# 2. 观察值是否归一化
# 3. 梯度是否爆炸或消失
# 4. 学习率是否合适

def diagnose_training(trainer):
    """诊断训练问题"""
    # 检查梯度
    for name, param in trainer.actor.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm = {param.grad.norm():.4f}")

    # 检查奖励分布
    rewards = trainer.buffer['rewards']
    print(f"Reward: mean={rewards.mean():.2f}, std={rewards.std():.2f}")

    # 检查优势分布
    advantages = trainer.buffer['advantages']
    print(f"Advantage: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
```

**Day 3-4: 超参数调优**
```python
# 网格搜索或使用 Optuna
# 调优参数：学习率、clip参数、entropy系数、gamma、lambda
```

**Day 5: 总结与记录**
```python
# 记录实验结果
# 绘制对比图表
# 撰写实验报告
```

---

## 📊 学习成果检查

### 理论理解
- [ ] 能解释 MDP 的四个要素
- [ ] 能推导 PPO 的目标函数
- [ ] 理解 GAE 的作用和实现
- [ ] 能说明 Actor-Critic 的分工

### 实践能力
- [ ] 能从零创建仿真环境
- [ ] 能设计观察/动作/奖励空间
- [ ] 能实现完整的 PPO 算法
- [ ] 能诊断和修复训练问题

### 工程技能
- [ ] 掌握 Isaac Sim 的基本 API
- [ ] 理解 PyTorch 的自动微分
- [ ] 能使用 TensorBoard 可视化
- [ ] 能管理实验和版本

---

## 🎓 与实习计划的对应

| 从零开始路径 | 对应实习计划 | 补充说明 |
|-------------|-------------|---------|
| Week 1-2 | - | 新增：Isaac Sim 基础 |
| Week 3-4 | Week 2 | 更深入：从头实现 MDP |
| Week 5-6 | Week 1 | 更深入：手写 PPO |
| Week 7-8 | Week 3-4 | 巩固：训练与调优 |
| Week 9+ | Stage 2-3 | 可直接进入进阶内容 |

---

## 💡 学习建议

1. **不要着急**：从零开始需要时间，但理解更深刻
2. **多动手**：每段代码都要实际运行和验证
3. **记笔记**：记录每个概念的理解和疑问
4. **做对比**：将手写版本与框架代码对比学习
5. **求甚解**：不懂的地方一定要弄清楚再继续

---

## 📚 参考资源

### 必读论文
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438

### 代码参考
- CleanRL: https://github.com/vwxyzjn/cleanrl
- Spinning Up: https://github.com/openai/spinningup

### 教程
- Isaac Lab 官方文档
- PyTorch 官方教程

---

**开始你的从零学习之旅吧！** 🚀
