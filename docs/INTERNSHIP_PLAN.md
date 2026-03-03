# 实习期学习计划：强化学习机器人控制

## 📋 总体目标

围绕 **强化学习 → 机械臂抓取 → 四足+机械臂全身控制** 技术主线，在试用期内完成从单一控制能力到复杂全身协同控制的系统性能力建设。

---

## 🎯 第一阶段：强化学习基础与控制框架构建（第 1-4 周）

### Week 1: 理论基础与代码理解

#### 学习目标
- 深入理解 PPO/SAC 算法原理
- 掌握现有代码库架构
- 理解 MDP 建模在机器人控制中的应用

#### 具体任务

**Day 1-2: 理论学习**
- [ ] **PPO 算法深入理解**
  - 阅读 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  - 重点：Clip 目标函数、GAE 优势估计、策略梯度
  - 笔记：用自己的话解释 PPO 核心思想

- [ ] **SAC 算法了解**
  - 阅读 [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning](https://arxiv.org/abs/1801.01290)
  - 理解最大熵原理及其在探索中的作用

**Day 3-5: 代码库深入分析**
- [ ] **训练流程梳理**
  ```bash
  # 追踪完整训练流程
  1. 分析 scripts/reinforcement_learning/rsl_rl/train.py
  2. 理解 RSL-RL 的 Runner 实现
  3. 绘制训练流程图（environment → data → policy → loss → update）
  ```

- [ ] **环境配置理解**
  - 分析 `velocity_env_cfg.py` 的配置结构
  - 理解 Manager 系统设计（Action/Observation/Reward/Termination Managers）
  - 画出模块依赖关系图

- [ ] **数据流追踪**
  ```python
  # 在代码中添加断点，追踪一次完整的训练步骤：
  # obs → policy → action → env.step → next_obs → reward → done
  # 并记录每个张量的形状和数值范围
  ```

#### 🔍 自验证节点

**验证方式 1：算法原理测试**
```python
# 创建 tests/test_theory_understanding.py
"""
回答以下问题（每个答案不超过 200 字）：
1. PPO 的 Clip 机制如何防止过大的策略更新？
2. GAE (Generalized Advantage Estimation) 相比 TD(0) 有什么优势？
3. 为什么要使用 GAE 而不是 Monte Carlo returns？
4. Actor-Critic 架构中，Actor 和 Critic 各自的作用是什么？
5. 什么是 On-Policy 和 Off-Policy？PPO 属于哪一种？
"""

def test_ppo_understanding():
    # 通过代码注释展示对算法的理解
    pass
```

**验证方式 2：代码流程复现**
```bash
# 运行一个简单训练任务，并画出数据流图
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0 \
    --num_envs 64 \
    --max_iterations 10

# 产出：
# - 训练流程图（markdown 格式）
# - 关键函数调用栈（至少 3 层深度）
```

**验证方式 3：MDP 建模练习**
```python
# 文档：为 Unitree G1 走路任务设计 MDP
"""
1. 状态空间 (State Space)
   - 基座姿态（位置、速度、四元数）
   - 关节角度、角速度
   - 速度命令
   - 预期：列出所有观测维度及其物理意义

2. 动作空间 (Action Space)
   - 关节目标位置控制
   - 动作范围：[-1, 1] 标准化到实际关节角度
   - 预期：画出控制框图

3. 奖励函数 (Reward Function)
   - 线速度跟踪奖励
   - 角速度跟踪奖励
   - 姿态稳定奖励
   - 惩罚项（关节限制、扭矩等）
   - 预期：给出数学表达式并解释每一项的作用

4. 终止条件 (Termination Conditions)
   - 倒地检测
   - 超出边界
   - 超时
   - 预期：列出具体阈值及其物理依据
"""
```

**预期产出：**
- [ ] 算法学习笔记（1000+ 字）
- [ ] 代码流程图（使用 draw.io 或 mermaid）
- [ ] Unitree G1 MDP 设计文档
- [ ] 数据流追踪报告（含 tensor shape 和范围）

---

### Week 2: 状态空间设计与观察系统

#### 学习目标
- 掌握状态空间设计原则
- 理解本体感知 vs 特权信息的区别
- 实现自定义观察项

#### 具体任务

**Day 1-3: 观察系统实现**

- [ ] **任务：为 Unitree G1 添加新的观察项**
  ```python
  # 在 source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/observations.py
  # 实现以下观察函数：

  def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
      """返回基座高度，用于机器人高度控制"""
      # TODO: 实现
      pass

  def feet_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
      """返回每只脚的接触力，用于步态分析"""
      # TODO: 实现
      pass

  def joint_accelerations(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
      """返回关节加速度，用于平滑控制"""
      # TODO: 实现
      pass
  ```

- [ ] **配置观察项**
  ```python
  # 在 Unitree G1 配置中添加新的观察项
  self.observations.policy.base_height = ObsTerm(
      func=mdp.base_height,
      params={"asset_cfg": SceneEntityCfg("robot")}
  )
  ```

**Day 4-5: 非对称 Actor-Critic 实验**

- [ ] **理解特权信息**
  ```python
  # 设计对比实验：
  # Actor 观测：仅本体感知（无接触力、无速度命令历史）
  # Critic 观测：特权信息（完整状态、真实速度、摩擦系数）

  # 预期：对比训练收敛速度和最终性能
  ```

- [ ] **实现观察空间配置**
  ```python
  # 创建 asymmetric_obs 配置
  # Actor: 37-dim (基础本体感知)
  # Critic: 187-dim (包含 height_scan 和其他特权信息)
  ```

#### 🔍 自验证节点

**验证 1：观察函数单元测试**
```python
# tests/test_observations.py
def test_base_height():
    """测试 base_height 观察是否正确"""
    # 1. 创建环境
    # 2. 重置到已知状态
    # 3. 验证观察值与实际高度一致（误差 < 0.01m）

def test_feet_contact_forces():
    """测试接触力观察"""
    # 1. 抬起一条腿
    # 2. 验证该腿接触力为 0
    # 3. 放下该腿
    # 4. 验证接触力 > 0
```

**验证 2：观察重要性分析**
```python
# 进行观察消融实验（Observation Ablation Study）
"""
依次移除以下观察，观察性能变化：
1. projected_gravity（重力投影）
2. base_lin_vel（线速度）
3. base_ang_vel（角速度）
4. joint_pos（关节位置）
5. joint_vel（关节速度）

预期产出：
- 每个观察的重要性排序
- 最小有效观察集合
"""
```

**验证 3：观察质量检查**
```bash
# 训练 1000 iterations，观察以下指标：
tensorboard --logdir=logs/experiments/observation_ablation

# 检查点：
# - 各个观察值的分布是否合理（无异常值）
# - 观察值是否在合理范围内归一化
# - 梯度是否稳定（无 NaN/Inf）
```

**预期产出：**
- [ ] 3 个新观察函数的实现代码
- [ ] 观察函数单元测试（测试覆盖率 > 80%）
- [ ] 观察消融实验报告（含训练曲线）
- [ ] 非对称 Actor-Critic 配置文档

---

### Week 3: 动作空间与奖励函数设计

#### 学习目标
- 掌握动作空间设计（位置/速度/力矩）
- 学习奖励函数设计技巧（Reward Shaping）
- 理解 Curriculum Learning 的应用

#### 具体任务

**Day 1-2: 动作空间对比**

- [ ] **实验：不同动作空间的性能对比**
  ```python
  # 对比三种动作空间：
  1. Joint Position Control（位置控制）
  2. Joint Velocity Control（速度控制）
  3. Joint Torque Control（力矩控制）

  # 训练配置：
  task = RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0
  num_envs = 512
  max_iterations = 2000

  # 记录：
  - 收敛速度（iterations to convergence）
  - 最终性能（episode reward）
  - 训练稳定性（variance of reward）
  - 动作平滑度（action change rate）
  ```

- [ ] **分析报告**
  ```python
  # 产出：动作空间对比表
  """
  | Action Space | Convergence Speed | Final Reward | Stability | Smoothness |
  |--------------|-------------------|--------------|-----------|------------|
  | Position     |                   |              |           |            |
  | Velocity     |                   |              |           |            |
  | Torque       |                   |              |           |            |

  结论：选择哪种动作空间及理由
  """
  ```

**Day 3-5: 奖励函数设计与调试**

- [ ] **奖励函数组件实现**
  ```python
  # 在 rewards.py 中实现以下奖励函数：

  def feet_energy_penalty(env, asset_cfg):
      """惩罚过高的足端冲击力"""
      # 实现：接触力 > 100N 时施加惩罚
      pass

  def slip_penalty(env, asset_cfg):
      """惩罚足端打滑"""
      # 实现：当接触力 > 0 但速度仍大时惩罚
      pass

  def gait_frequency_reward(env, asset_cfg):
      """奖励自然的步态频率"""
      # 实现：足端接触周期性
      pass

  def smooth_action_reward(env, asset_cfg):
      """奖励平滑的动作序列"""
      # 实现：惩罚动作变化率
      pass
  ```

- [ ] **奖励函数调试流程**
  ```python
  # 1. 创建奖励分析工具
  def analyze_rewards(log_dir):
      """
      加载训练日志，分析：
      - 各个奖励项的分布（mean, std, min, max）
      - 奖励项之间的相关性
      - 稀疏奖励项的触发频率
      """
      pass

  # 2. 可视化奖励权重影响
  # 使用 grid search 搜索最优权重组合
  ```

#### 🔍 自验证节点

**验证 1：动作空间效果对比**
```bash
# 运行三个实验，使用相同随机种子
for action_type in "position" "velocity" "torque"; do
    python train.py \
        --task=RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0 \
        --action_type=$action_type \
        --seed=42 \
        --max_iterations=2000 \
        --run_name=action_space_study/${action_type}
done

# 分析结果：
python scripts/tools/analyze_action_spaces.py --logdir=logs/action_space_study
```

**验证 2：奖励函数消融**
```python
# 依次移除每个奖励项，观察性能影响
reward_terms = [
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "flat_orientation",
    "feet_air_time",
    "slip_penalty",
    # ...
]

for term in reward_terms:
    # 训练不带该奖励项的策略
    # 记录性能下降百分比

# 产出：奖励项重要性排序
```

**验证 3：奖励值分布检查**
```python
# 在训练过程中记录每个奖励项的统计信息
# 检查：
# 1. 是否有奖励项始终为 0（无效）
# 2. 是否有奖励项方差过大（不稳定）
# 3. 是否有奖励项被其他项主导（权重不平衡）

# 可视化：
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, term in enumerate(reward_terms):
    axes[i//3, i%3].hist(term_values, bins=50)
    axes[i//3, i%3].set_title(f"{term}: mean={mean:.2f}, std={std:.2f}")
plt.savefig("reward_distributions.png")
```

**预期产出：**
- [ ] 动作空间对比实验报告
- [ ] 4 个新奖励函数实现
- [ ] 奖励分析工具脚本
- [ ] 奖励消融实验报告

---

### Week 4: 训练稳定性与性能优化

#### 学习目标
- 掌握训练调参技巧
- 学习常见训练问题诊断
- 理解 Curriculum Learning 应用

#### 具体任务

**Day 1-2: 训练调参实战**

- [ ] **超参数敏感性分析**
  ```python
  # 关键超参数及其影响：
  1. learning_rate: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
  2. clip_param: [0.1, 0.2, 0.3]
  3. entropy_coef: [0.0, 0.01, 0.05, 0.1]
  4. gamma: [0.95, 0.97, 0.99, 0.995]
  5. lambda (GAE): [0.9, 0.95, 0.98]

  # 使用 Weights & Biases 或 TensorBoard 记录
  ```

- [ ] **Batch Size 和环境数量实验**
  ```python
  # 研究：
  # 1. batch_size 对样本效率和训练稳定性的影响
  # 2. num_envs 对训练速度和内存使用的影响

  # 实验设计：
  batch_sizes = [512, 1024, 2048, 4096]
  num_envs_list = [128, 256, 512, 1024, 2048]

  # 找到最优配置
  ```

**Day 3-4: 常见问题诊断与解决**

- [ ] **问题诊断清单**
  ```python
  # 实现诊断工具：
  def diagnose_training_issue(log_dir):
      """
      自动诊断常见训练问题：

      1. 奖励不收敛或震荡
      2. 梯度爆炸/消失
      3. 策略过早收敛到次优解
      4. 训练速度过慢
      5. 内存溢出

      返回：问题诊断结果 + 建议解决方案
      """
      pass
  ```

- [ ] **实际问题演练**
  ```python
  # 制造以下问题并学会修复：
  1. 观察归一化失败（导致 NaN）
  2. 奖励权重失衡（导致某些行为被忽略）
  3. 学习率过大（训练发散）
  4. 批次大小过小（训练不稳定）

  # 每个问题：制造 → 识别 → 修复 → 验证
  ```

**Day 5: Curriculum Learning 实现**

- [ ] **实现 Curriculum Learning**
  ```python
  # 在 curriculums.py 中实现：

  class TerrainDifficultyCurriculum:
      """地形难度课程学习"""
      def __init__(self):
          self.levels = [
              {"roughness": 0.0, "max_height": 0.0},      # 平地
              {"roughness": 0.1, "max_height": 0.05},     # 简单崎岖
              {"roughness": 0.2, "max_height": 0.1},      # 中等崎岖
              {"roughness": 0.3, "max_height": 0.15},     # 困难崎岖
          ]

      def update_level(self, env_measures):
          """根据性能自动升级难度"""
          # 如果连续 100 次 episode 成功，升级难度
          pass

  class CommandVelocityCurriculum:
      """速度命令课程学习"""
      def __init__(self):
          self.velocity_ranges = [
              (0.0, 0.5),   # 初始：低速
              (0.5, 1.0),   # 中速
              (1.0, 1.5),   # 高速
          ]
  ```

#### 🔍 自验证节点

**验证 1：超参数优化实验**
```python
# 使用 Optuna 进行自动超参数搜索
import optuna

def objective(trial):
    """超参数优化目标函数"""
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    clip_param = trial.suggest_float("clip_param", 0.1, 0.3)
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.1)

    # 训练 500 iterations，返回最终 reward
    reward = train_and_evaluate(lr, clip_param, entropy_coef)
    return reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 输出最优超参数
print(f"Best hyperparameters: {study.best_params}")
```

**验证 2：训练问题诊断测试**
```python
# 准备 5 个有问题的训练日志
test_cases = [
    "divergent_training",      # 发散的训练
    "oscillating_training",    # 震荡的训练
    "slow_convergence",        # 收敛过慢
    "suboptimal_policy",       # 次优策略
    "memory_overflow",         # 内存溢出
]

# 你的诊断工具应该能正确识别所有问题
for case in test_cases:
    diagnosis = diagnose_training_issue(f"logs/{case}")
    print(f"{case}: {diagnosis}")
    assert diagnosis.issue_detected == True
```

**验证 3：Curriculum Learning 效果验证**
```bash
# 对比实验：
# 1. 无 Curriculum Learning
# 2. 有 Curriculum Learning

# 运行相同 iterations，比较：
# - 最终性能（在困难地形上的表现）
# - 收敛速度
# - 样本效率

python train.py --task=... --use_curriculum=False
python train.py --task=... --use_curriculum=True

# 可视化对比：
python scripts/tools/plot_curriculum_comparison.py
```

**预期产出：**
- [ ] 超参数敏感性分析报告
- [ ] 训练诊断工具脚本
- [ ] Curriculum Learning 实现代码
- [ ] 5 个实际问题的修复案例记录

---

## 🎯 第一阶段总结检查点

### 必须完成（Must Have）

- [ ] **代码能力**
  - 能独立追踪和理解完整训练流程
  - 能实现新的观察/奖励函数
  - 能诊断和修复常见训练问题

- [ ] **理论理解**
  - 能清晰解释 PPO 算法原理
  - 能设计合理的 MDP（状态、动作、奖励）
  - 理解非对称 Actor-Critic 的优势

- [ ] **实践经验**
  - 至少完成 3 次完整的训练实验
  - 能使用 TensorBoard 分析训练曲线
  - 能对比不同配置的性能差异

### 加分项（Nice to Have）

- [ ] 实现了新颖的观察/奖励函数
- [ ] 发表了训练日志分析博客
- [ ] 参与了代码库的 Issue 或 PR
- [ ] 实现了 Curriculum Learning 并证明其有效性

---

## 📚 学习资源

### 核心论文
1. PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. SAC: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
3. GAE: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
4. Asymmetric AC: [Asymmetric Actor-Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542)

### 实践课程
1. [UCL RL Course](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
2. [Spinning Up in Deep RL](https://spinningup.openai.com/)
3. [Deep RL for Robotics (Stanford)](http://rll.berkeley.edu/deeprlcourse/)

### 代码参考
1. [CleanRL](https://github.com/vwxyzjn/cleanrl) - 简洁的 RL 实现
2. [RLLIB](https://docs.ray.io/en/latest/rllib/) - Ray 的 RL 库
3. [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)

---

## 📝 周报模板

### 每周汇报内容

```markdown
## Week N 总结

### 完成任务
- [ ] 任务 1：具体描述
  - 实现方式
  - 遇到的问题
  - 解决方案

- [ ] 任务 2：具体描述
  - ...

### 关键收获
1. 理论方面：
2. 实践方面：
3. 工程技能：

### 遇到的问题
1. 问题描述
2. 尝试的解决方案
3. 最终解决方案

### 下周计划
- [ ] 计划任务 1
- [ ] 计划任务 2
- ...

### 资源消耗
- 代码行数：XXX
- 训练小时数：XX
- 论文阅读：X 篇
```

---

## 🎓 下一步预告

第二阶段将进入**机械臂抓取策略设计**，涵盖：
- 抓取任务的 MDP 建模
- 稀疏奖励下的探索策略
- 模仿学习与强化学习结合
- Sim-to-Real 迁移技术
