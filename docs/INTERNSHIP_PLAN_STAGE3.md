# 第三阶段：四足+机械臂全身控制（第 9-12 周）

## 📋 阶段目标

这是最具挑战性的阶段：将**四足移动平台**与**机械臂操作**结合，实现全身协同控制策略，验证在强动力学耦合条件下的稳定性、泛化性与工程可行性。

---

## 🎯 Week 9-10: 全身控制基础

### Week 9: 组合仿真环境搭建

#### 学习目标
- 理解四足+机械臂系统的复杂性
- 掌握多体系统的仿真配置
- 实现基础的运动学和动力学接口

#### 具体任务

**Day 1-3: 系统建模**

```python
# 组合系统配置
# 示例：Unitree Go2 + 机械臂

class QuadrupedWithArmCfg:
    """四足+机械臂组合系统配置"""

    # 四足平台
    base = UNITREE_GO2_CFG

    # 机械臂（安装在四足背部）
    arm = ARTICULATED_ARM_CFG
    arm_mount_offset = (0.0, 0.0, 0.2)  # 相对于四足背部的偏移

    # 关键配置
    total_dof = 12 + 7  # 四足 12 DOF + 机械臂 7 DOF

    # 耦合分析
    coupling_effects = {
        "base_motion_affects_arm": True,   # 四足运动影响机械臂
        "arm_motion_affects_base": True,   # 机械臂运动影响四足稳定性
        "payload_changes": "variable",     # 机械臂抓取物体改变负载
    }
```

**仿真环境搭建：**
```python
# source/robot_lab/tasks/manager_based/locomotion/manipulation/whole_body_env_cfg.py

@configclass
class QuadrupedManipulationEnvCfg(ManagerBasedRLEnvCfg):
    """四足+机械臂全身控制环境"""

    # 场景配置
    scene = SceneCfg(
        num_envs=256,  # 减少环境数量（计算量增大）
        env_spacing=3.0,
    )

    # 机器人配置
    robot = QUADRUPED_WITH_ARM_CFG

    # 任务配置
    task_type = "locomotion_manipulation"

    # 观察空间
    observations.policy = [
        # 四足状态
        "base_pos", "base_quat",
        "base_lin_vel", "base_ang_vel",
        "quadruped_joint_pos", "quadruped_joint_vel",

        # 机械臂状态
        "arm_joint_pos", "arm_joint_vel",
        "ee_pos", "ee_quat",

        # 目标/物体状态
        "target_pos",
        "object_pos", "object_quat",

        # 接触状态
        "feet_contact",  # 四足足端接触
        "gripper_contact",  # 夹爪接触
    ]  # 总计约 80-100 dim

    # 动作空间
    actions = {
        "quadruped": "joint_position",  # 12-dim
        "arm": "joint_position",        # 7-dim
    }  # 总计 19-dim

    # 奖励配置
    rewards = [
        # 四足运动奖励
        "track_base_velocity",

        # 机械臂操作奖励
        "reach_target",
        "grasp_object",

        # 全身协调奖励
        "minimize_base_motion_during_manipulation",
        "maintain_stability",
    ]
```

**Day 4-5: 动力学耦合分析**

```python
# 分析动力学耦合效应

class DynamicsCouplingAnalyzer:
    """动力学耦合分析器"""

    def __init__(self, env):
        self.env = env

    def analyze_coupling_effects(self):
        """分析不同部分之间的动力学耦合"""

        results = {}

        # 实验 1: 四足运动对机械臂的影响
        print("测试：四足运动对机械臂的影响")
        base_velocities = [0.0, 0.5, 1.0, 1.5]  # m/s
        ee_position_error = []

        for vel in base_velocities:
            # 设置四足速度
            self.env.set_base_velocity(vel)

            # 机械臂保持静止
            ee_pos_initial = self.env.get_ee_position()

            # 运行 100 步
            for _ in range(100):
                self.env.step(zero_action)

            # 测量 EE 位置变化
            ee_pos_final = self.env.get_ee_position()
            error = np.linalg.norm(ee_pos_final - ee_pos_initial)
            ee_position_error.append(error)

        results['base_motion_on_arm'] = ee_position_error

        # 实验 2: 机械臂运动对四足稳定性的影响
        print("测试：机械臂运动对四足稳定性的影响")
        arm_velocities = [0.0, 0.5, 1.0, 1.5]  # rad/s
        base_tilt_error = []

        for vel in arm_velocities:
            # 设置机械臂运动
            self.env.set_arm_velocity(vel)

            # 四足保持站立
            base_orientation_initial = self.env.get_base_orientation()

            # 运行 100 步
            for _ in range(100):
                self.env.step(zero_action)

            # 测量基座姿态变化
            base_orientation_final = self.env.get_base_orientation()
            error = compute_orientation_error(base_orientation_initial, base_orientation_final)
            base_tilt_error.append(error)

        results['arm_motion_on_base'] = base_tilt_error

        # 实验 3: 负载变化的影响
        print("测试：负载变化对系统动态的影响")
        payload_masses = [0.0, 0.5, 1.0, 1.5]  # kg
        stability_metrics = []

        for mass in payload_masses:
            # 设置负载质量
            self.env.set_payload_mass(mass)

            # 执行标准动作序列
            for _ in range(100):
                self.env.step(standard_action)

            # 测量稳定性
            stability = self.env.compute_stability_metric()
            stability_metrics.append(stability)

        results['payload_effect'] = stability_metrics

        return results

    def visualize_coupling(self, results):
        """可视化耦合效应"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 图 1：四足运动对机械臂的影响
        axes[0].plot(base_velocities, results['base_motion_on_arm'])
        axes[0].set_xlabel('Base Velocity (m/s)')
        axes[0].set_ylabel('EE Position Error (m)')
        axes[0].set_title('Effect of Base Motion on Arm')

        # 图 2：机械臂运动对四足稳定性的影响
        axes[1].plot(arm_velocities, results['arm_motion_on_base'])
        axes[1].set_xlabel('Arm Velocity (rad/s)')
        axes[1].set_ylabel('Base Tilt Error (rad)')
        axes[1].set_title('Effect of Arm Motion on Base Stability')

        # 图 3：负载变化的影响
        axes[2].plot(payload_masses, results['payload_effect'])
        axes[2].set_xlabel('Payload Mass (kg)')
        axes[2].set_ylabel('Stability Metric')
        axes[2].set_title('Effect of Payload on System Dynamics')

        plt.tight_layout()
        plt.savefig('coupling_analysis.png')
```

#### 🔍 自验证节点

**验证 1：系统仿真正确性**
```bash
# 测试脚本能正确加载组合系统
python scripts/tools/test_quadruped_arm.py --test=simulation

# 验证项：
# ✅ 所有关节可独立控制
# ✅ 无碰撞穿透
# ✅ 正向运动学计算正确
# ✅ 动力学仿真稳定（无爆炸）
```

**验证 2：耦合效应分析**
```python
# 运行完整的耦合分析
analyzer = DynamicsCouplingAnalyzer(env)
results = analyzer.analyze_coupling_effects()

# 生成报告：
# - 耦合效应的量化描述
# - 主要的耦合来源
# - 对控制策略设计的启示
analyzer.visualize_coupling(results)

# 预期产出：
# - 耦合分析报告（PDF）
# - 可视化图表
# - 控制策略设计建议
```

---

### Week 10: 全身运动规划

#### 学习目标
- 理解全身运动规划的挑战
- 掌握分层控制策略
- 学习 Whole-Body Control (WBC) 方法

#### 具体任务

**Day 1-3: 分层控制架构**

```python
# 实现分层控制器

class HierarchicalController:
    """分层全身控制器"""

    def __init__(self, env):
        self.env = env

        # 高层策略：任务级决策
        self.high_level_policy = HighLevelPolicy(
            state_dim=100,
            action_dim=20,  # 粗粒度的动作指令
        )

        # 低层策略：关节级控制
        self.low_level_policy = LowLevelPolicy(
            state_dim=150,
            action_dim=19,  # 所有关节
        )

        # 中间层：任务分解
        self.task_decomposer = TaskDecomposer()

    def compute_action(self, state):
        """分层动作计算"""

        # 第 1 层：高层任务规划
        task_command = self.high_level_policy.plan(state)
        # task_command: {
        #     "base_velocity": (vx, vy, vz),
        #     "arm_target": (x, y, z, qx, qy, qz, qw),
        #     "gripper_command": open/close,
        # }

        # 第 2 层：任务分解
        quadruped_task, arm_task = self.task_decomposer.decompose(task_command)

        # 第 3 层：低层控制
        quadruped_action = self.low_level_policy.control_quadruped(
            state, quadruped_task
        )
        arm_action = self.low_level_policy.control_arm(
            state, arm_task
        )

        # 组合动作
        action = np.concatenate([quadruped_action, arm_action])
        return action


class HighLevelPolicy:
    """高层策略：任务规划"""

    def __init__(self):
        # 神经网络策略（输出粗粒度指令）
        self.network = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),  # 粗粒度动作空间
        )

    def plan(self, state):
        """生成任务级命令"""

        # 提取高层状态
        high_level_state = self.extract_high_level_state(state)

        # 前向传播
        task_command = self.network(high_level_state)

        # 解析命令
        base_velocity = task_command[:3]
        arm_target = task_command[3:10]
        gripper_cmd = task_command[10]

        return {
            "base_velocity": base_velocity,
            "arm_target": arm_target,
            "gripper_command": gripper_cmd,
        }


class LowLevelPolicy:
    """低层策略：关节控制"""

    def __init__(self):
        # 四足控制器（可能使用传统控制方法）
        self.quadruped_controller = QuadrupedMPC()

        # 机械臂控制器（可能使用 IK）
        self.arm_controller = ArmIKController()

    def control_quadruped(self, state, task):
        """四足低层控制"""
        # 将任务转换为关节命令
        base_velocity_cmd = task["base_velocity"]

        # 使用 MPC 或其他方法计算关节动作
        joint_actions = self.quadruped_controller.compute(
            base_velocity_cmd,
            state["quadruped_joint_pos"],
            state["quadruped_joint_vel"],
        )

        return joint_actions

    def control_arm(self, state, task):
        """机械臂低层控制"""
        # 将目标转换为关节命令
        ee_target = task["arm_target"]

        # 使用 IK 计算关节角度
        joint_angles = self.arm_controller.compute_ik(
            ee_target,
            state["arm_joint_pos"],
        )

        return joint_angles
```

**Day 4-5: Whole-Body Control (WBC)**

```python
# 实现 WBC 方法

class WholeBodyController:
    """全身控制器（基于 QP）"""

    def __init__(self, env):
        self.env = env

        # 任务优先级
        self.task_hierarchy = [
            "grasp",          # 优先级 1：抓取任务
            "stability",      # 优先级 2：保持稳定
            "locomotion",     # 优先级 3：运动
            "energy",         # 优先级 4：节能
        ]

    def compute_optimal_action(self, state, desired_tasks):
        """计算最优全身动作（QP 求解）"""

        # 构建二次规划问题：
        # min  0.5 * x^T * H * x + f^T * x
        # s.t. A_eq * x = b_eq
        #      A_ineq * x <= b_ineq
        #      lb <= x <= ub

        # 决策变量 x：关节加速度（或力矩）
        n_joints = 19  # 12（四足） + 7（机械臂）

        # 目标函数：最小化控制努力
        H = np.eye(n_joints)  # 权重矩阵
        f = np.zeros(n_joints)

        # 等式约束：动力学方程
        # M(q) * q_ddot + C(q, q_dot) = tau
        M = self.compute_mass_matrix(state)
        C = self.compute_coriolis_centrifugal(state)
        tau = self.compute_joint_torques(state)

        A_eq = np.block([[M, -np.eye(n_joints)]])
        b_eq = -C

        # 不等式约束：摩擦锥、关节限制等
        A_ineq, b_ineq = self.compute_inequality_constraints(state)

        # 求解 QP
        from osqp import OSQP
        solver = OSQP()
        solver.setup(H, f, A_eq, b_eq, A_ineq, b_ineq)

        # 任务约束（软约束，用惩罚项处理）
        for task in self.task_hierarchy:
            weight = self.get_task_weight(task)
            desired = desired_tasks[task]

            # 添加任务约束到目标函数
            J = self.compute_task_jacobian(task, state)
            error = self.compute_task_error(task, state, desired)

            # H += weight * J^T * J
            # f += -weight * J^T * error
            H += weight * J.T @ J
            f += -weight * J.T @ error

        # 更新并求解
        solver.update(H, f)
        solution = solver.solve()

        q_ddot_optimal = solution.x[:n_joints]

        # 积分得到动作（位置/速度）
        action = self.integrate_acceleration(q_ddot_optimal, state)

        return action

    def compute_task_jacobian(self, task, state):
        """计算任务的雅可比矩阵"""

        if task == "grasp":
            # 抓取任务的雅可比：末端执行器位置/速度
            J_ee = self.env.compute_ee_jacobian(state)
            return J_ee

        elif task == "stability":
            # 稳定性任务的雅可比：ZMP（零力矩点）
            J_zmp = self.env.compute_zmp_jacobian(state)
            return J_zmp

        elif task == "locomotion":
            # 运动任务的雅可比：基座速度
            J_base = self.env.compute_base_jacobian(state)
            return J_base

        elif task == "energy":
            # 能量任务的雅可比：功率
            J_power = self.env.compute_power_jacobian(state)
            return J_power

    def get_task_weight(self, task):
        """获取任务权重（基于优先级）"""
        weights = {
            "grasp": 1000.0,      # 最高优先级
            "stability": 100.0,
            "locomotion": 10.0,
            "energy": 1.0,
        }
        return weights.get(task, 1.0)
```

#### 🔍 自验证节点

**验证 1：分层控制性能**
```bash
# 对比分层控制和端到端控制

# 分层控制
python train.py --method=hierarchical --run_name=hierarchical
python train.py --method=end_to_end --run_name=end_to_end

# 分析：
python scripts/tools/compare_control_methods.py

# 评估指标：
# - 收敛速度
# - 最终性能
# - 训练稳定性
# - 动作平滑度

# 成功标准：
# - 分层控制比端到端快 > 30%
# - 分层控制最终性能接近端到端
```

**验证 2：WBC 有效性**
```python
# 测试 WBC 在不同场景下的表现

test_scenarios = [
    "static_grasp",        # 静止抓取
    "walking_grasp",       # 行走中抓取
    "rough_terrain",        # 粗糙地形
    "payload_variations",   # 不同负载
]

for scenario in test_scenarios:
    # 设置场景
    env.set_scenario(scenario)

    # 测试 WBC 控制器
    success_rate = test_wbc_controller(env)

    print(f"{scenario}: {success_rate:.2%}")

# 成功标准：
# - 所有场景成功率 > 70%
# - walking_grasp 最困难（预期最低）
```

---

## 🎯 Week 11-12: 全身协同控制与验证

### Week 11: 全身协同控制策略

#### 学习目标
- 实现复杂的全身协调任务
- 掌握多目标优化方法
- 学习任务切换和优先级管理

#### 具体任务

**Day 1-3: 多任务学习**

```python
# 实现 Multi-Task RL

class MultiTaskWholeBodyController:
    """多任务全身控制器"""

    def __init__(self):
        self.tasks = [
            "locomotion",       # 任务 1：移动
            "manipulation",     # 任务 2：操作
            "loco_manip",       # 任务 3：边走边抓
            "patrol",           # 任务 4：巡逻（边走边观察）
            "transport",        # 任务 5：运输（抓取并移动物体）
        ]

        # 共享策略网络
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # 任务特定头部
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(256, action_dim)
            for task in self.tasks
        })

        # 任务识别器
        self.task_identifier = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.tasks)),
            nn.Softmax(dim=-1),
        )

    def select_action(self, state):
        """根据当前任务选择动作"""

        # 识别当前任务
        task_probs = self.task_identifier(state)
        current_task = self.tasks[torch.argmax(task_probs)]

        # 共享特征提取
        shared_features = self.shared_encoder(state)

        # 任务特定动作
        action_head = self.task_heads[current_task]
        action = action_head(shared_features)

        return action


class MultiObjectiveOptimizer:
    """多目标优化器"""

    def __init__(self):
        # 多个优化目标
        self.objectives = {
            "task_success": 1.0,        # 任务成功
            "energy_efficiency": 0.3,   # 能量效率
            "smoothness": 0.2,          # 动作平滑
            "stability": 0.5,           # 稳定性
            "speed": 0.4,               # 速度
        }

    def compute_multi_objective_reward(self, env):
        """计算多目标奖励"""

        rewards = {}

        # 计算各个目标的奖励
        for obj_name in self.objectives:
            if obj_name == "task_success":
                rewards[obj_name] = self.compute_task_success(env)

            elif obj_name == "energy_efficiency":
                rewards[obj_name] = -self.compute_energy_consumption(env)

            elif obj_name == "smoothness":
                rewards[obj_name] = -self.compute_action_jerk(env)

            elif obj_name == "stability":
                rewards[obj_name] = self.compute_stability_metric(env)

            elif obj_name == "speed":
                rewards[obj_name] = self.compute_speed_metric(env)

        # 加权求和
        total_reward = sum(
            self.objectives[obj] * rewards[obj]
            for obj in self.objectives
        )

        return total_reward, rewards
```

**Day 4-5: 任务切换和优先级**

```python
# 实现动态任务切换

class DynamicTaskScheduler:
    """动态任务调度器"""

    def __init__(self):
        self.current_task = None
        self.task_priority = {
            "safety": 1,        # 最高优先级：安全（避免摔倒、碰撞等）
            "critical": 2,      # 关键任务：避免物体掉落
            "normal": 3,        # 普通任务：常规操作
            "optional": 4,      # 可选任务：优化性能
        }

    def update_task_priority(self, env_state):
        """根据环境状态动态调整任务优先级"""

        # 检测紧急情况
        if self.detect_fall_risk(env_state):
            # 防止摔倒：最高优先级
            self.current_task = "maintain_stability"
            priority = "safety"

        elif self.detect_dropping_object(env_state):
            # 防止掉落：高优先级
            self.current_task = "secure_grasp"
            priority = "critical"

        elif self.detect_obstacle(env_state):
            # 避障：普通优先级
            self.current_task = "avoid_obstacle"
            priority = "normal"

        else:
            # 执行主要任务：低优先级
            self.current_task = "execute_primary_task"
            priority = "optional"

        return priority

    def detect_fall_risk(self, state):
        """检测摔倒风险"""
        # 检查：
        # 1. 基座倾斜角度
        # 2. ZMP（零力矩点）是否接近支撑多边形边缘
        # 3. 质心高度是否过低

        base_tilt = compute_base_tilt(state)
        zmp_margin = compute_zmp_margin(state)
        com_height = state["base_pos"][2]

        if abs(base_tilt) > 0.3:  # 倾斜 > 17°
            return True
        if zmp_margin < 0.02:  # ZMP 距离边缘 < 2cm
            return True
        if com_height < 0.2:  # 质心过低
            return True

        return False

    def detect_dropping_object(self, state):
        """检测物体掉落风险"""
        # 检查：
        # 1. 夹爪接触力是否突然减小
        # 2. 物体是否在夹爪范围内

        gripper_force = state["gripper_contact_force"]
        object_in_gripper = state["object_in_gripper"]

        if object_in_gripper and gripper_force < 1.0:  # 接触力 < 1N
            return True

        return False
```

#### 🔍 自验证节点

**验证 1：多任务学习效果**
```bash
# 训练多任务策略
python train.py \
    --task=MultiTaskWholeBody \
    --num_tasks 5 \
    --max_iterations 5000 \
    --run_name=multi_task

# 评估每个任务的表现
for task in ["locomotion", "manipulation", "loco_manip", "patrol", "transport"]:
    success_rate = evaluate_task(task)
    print(f"{task}: {success_rate:.2%}")

# 成功标准：
# - 所有任务成功率 > 60%
# - 简单任务（locomotion, manipulation）> 80%
# - 复杂任务（loco_manip, transport）> 50%
```

**验证 2：任务切换性能**
```python
# 测试任务切换能力

# 场景 1：从静态到动态
# 任务：静止抓取 → 行走中抓取
static_grasp_success = test_static_grasp()
walking_grasp_success = test_walking_grasp()

# 场景 2：紧急情况处理
# 任务：正常操作 → 检测到摔倒风险 → 恢复稳定
recovery_success = test_emergency_recovery()

# 场景 3：多任务并行
# 任务：边走边观察（patrol）
patrol_performance = test_patrol_task()

# 评估指标：
# - 任务切换延迟（< 10 steps）
# - 切换成功率（> 90%）
# - 性能保持率（切换后性能 > 原性能的 80%）
```

---

### Week 12: 系统验证与部署

#### 学习目标
- 完成全身控制策略的全面验证
- 测试泛化能力和鲁棒性
- 准备 Sim-to-Real 迁移（如有条件）

#### 具体任务

**Day 1-3: 全面的系统验证**

```python
# 系统验证测试套件

class SystemValidationTestSuite:
    """系统验证测试套件"""

    def __init__(self, policy_path):
        self.policy = load_policy(policy_path)

    def run_all_tests(self):
        """运行所有验证测试"""

        results = {}

        # 测试 1：基础功能测试
        print("测试 1：基础功能...")
        results['basic_functionality'] = self.test_basic_functionality()

        # 测试 2：鲁棒性测试
        print("测试 2：鲁棒性...")
        results['robustness'] = self.test_robustness()

        # 测试 3：泛化能力测试
        print("测试 3：泛化能力...")
        results['generalization'] = self.test_generalization()

        # 测试 4：性能边界测试
        print("测试 4：性能边界...")
        results['performance_limits'] = self.test_performance_limits()

        # 测试 5：长期运行测试
        print("测试 5：长期运行...")
        results['long_term_stability'] = self.test_long_term_stability()

        # 生成测试报告
        self.generate_validation_report(results)

        return results

    def test_basic_functionality(self):
        """基础功能测试"""

        tests = {
            "locomotion": test_robot_can_move_forward,
            "manipulation": test_robot_can_grasp_object,
            "loco_manip": test_robot_can_walk_and_grasp,
            "stability": test_robot_remains_stable,
            "safety": test_robot_no_collision,
        }

        results = {}
        for test_name, test_func in tests.items():
            success = test_func(self.policy)
            results[test_name] = success
            print(f"  {test_name}: {'✓' if success else '✗'}")

        return results

    def test_robustness(self):
        """鲁棒性测试：在扰动下的性能"""

        robustness_tests = {
            "sensor_noise": test_with_sensor_noise,
            "actuator_delay": test_with_actuator_delay,
            "external_force": test_with_external_force,
            "terrain_variation": test_on_different_terrains,
            "payload_variation": test_with_different_payloads,
        }

        results = {}
        for test_name, test_func in robustness_tests.items():
            performance = test_func(self.policy)
            results[test_name] = performance
            print(f"  {test_name}: {performance:.2%}")

        return results

    def test_generalization(self):
        """泛化能力测试：在未见过的场景下的性能"""

        generalization_tests = {
            "unseen_terrain": test_on_new_terrain,
            "unseen_object": test_with_new_object,
            "unseen_goal": test_with_new_goal,
            "velocity_variation": test_with_different_velocities,
            "lighting_condition": test_under_different_lighting,
        }

        results = {}
        for test_name, test_func in generalization_tests.items():
            performance = test_func(self.policy)
            results[test_name] = performance
            print(f"  {test_name}: {performance:.2%}")

        return results

    def test_performance_limits(self):
        """性能边界测试：找出系统的极限"""

        # 测试不同速度下的性能
        velocities = np.linspace(0.5, 2.0, 8)  # m/s
        success_rates = []

        for vel in velocities:
            success = test_at_velocity(self.policy, vel)
            success_rates.append(success)

        # 找到最大可行速度
        max_velocity = velocities[np.argmax(success_rates)]

        # 测试不同负载下的性能
        payloads = np.linspace(0.0, 2.0, 9)  # kg
        payload_success_rates = []

        for payload in payloads:
            success = test_with_payload(self.policy, payload)
            payload_success_rates.append(success)

        # 找到最大负载
        max_payload = payloads[np.argmax(payload_success_rates)]

        return {
            "max_velocity": max_velocity,
            "max_payload": max_payload,
            "velocity_curve": (velocities, success_rates),
            "payload_curve": (payloads, payload_success_rates),
        }

    def test_long_term_stability(self):
        """长期稳定性测试：长时间运行是否稳定"""

        # 运行 10000 步（约 3.3 小时，假设 dt=0.02s）
        num_steps = 10000

        performance_over_time = []
        for step in range(num_steps):
            # 执行策略
            obs, reward, done, _ = env.step(policy(obs))

            # 记录性能
            if step % 100 == 0:
                performance_over_time.append({
                    "step": step,
                    "reward": reward,
                    "episode_length": step - last_reset,
                })

            # 重置
            if done:
                obs = env.reset()

        # 分析性能是否下降
        early_performance = np.mean([p["reward"] for p in performance_over_time[:10]])
        late_performance = np.mean([p["reward"] for p in performance_over_time[-10:]])
        performance_degradation = early_performance - late_performance

        return {
            "performance_over_time": performance_over_time,
            "degradation": performance_degradation,
            "stable": performance_degradation < 0.1,  # 性能下降 < 10%
        }
```

**Day 4-5: Sim-to-Real 迁移准备**

```python
# Sim-to-Real 技术

class Sim2RealTransfer:
    """Sim-to-Real 迁移"""

    def __init__(self):
        self.domain_randomization = DomainRandomization()
        self.system_identification = SystemIdentification()
        self.reality_gap_analysis = RealityGapAnalyzer()

    def prepare_for_transfer(self):
        """准备迁移到真实机器人"""

        # 1. 系统识别
        print("系统识别...")
        real_robot_params = self.system_identification.identify_real_robot()

        # 2. 域随机化配置
        print("配置域随机化...")
        dr_config = self.domain_randomization.configure_randomization(
            real_robot_params
        )

        # 3. 训练鲁棒策略
        print("使用域随机化训练...")
        robust_policy = self.train_with_domain_randomization(dr_config)

        # 4. 迁移到真实机器人
        print("迁移到真实机器人...")
        transfer_success = self.transfer_to_real_robot(robust_policy)

        return transfer_success

    def train_with_domain_randomization(self, dr_config):
        """使用域随机化训练鲁棒策略"""

        # 随机化参数：
        randomized_params = {
            # 物理参数
            "mass": lambda: np.random.uniform(0.8, 1.2),  # ±20%
            "friction": lambda: np.random.uniform(0.5, 1.0),  # ±50%
            "joint_damping": lambda: np.random.uniform(0.8, 1.2),

            # 延迟
            "actuator_delay": lambda: np.random.uniform(0, 0.02),  # 0-20ms

            # 传感器噪声
            "sensor_noise": lambda: np.random.uniform(0.001, 0.01),

            # 初始状态
            "init_joint_pos": lambda: np.random.uniform(-0.1, 0.1),
            "init_base_pos": lambda: np.random.uniform(-0.05, 0.05),
        }

        # 训练
        for iteration in range(max_iterations):
            # 每次迭代重新采样参数
            sampled_params = {
                name: sampler() for name, sampler in randomized_params.items()
            }

            # 更新环境参数
            env.update_parameters(**sampled_params)

            # 训练步骤
            policy.update(env.step(policy(obs)))

        return policy

    def analyze_reality_gap(self, sim_policy, real_robot):
        """分析仿真与现实的差距"""

        # 收集数据
        sim_data = collect_data(sim_policy, simulation)
        real_data = collect_data(sim_policy, real_robot)

        # 对比分析
        gaps = {
            "state_distribution": compare_distributions(sim_data, real_data),
            "action_distribution": compare_distributions(sim_data, real_data),
            "task_success_rate": compare_success_rates(sim_data, real_data),
            "safety_metrics": compare_safety(sim_data, real_data),
        }

        return gaps
```

#### 🔍 自验证节点

**验证 1：系统完整性测试**
```bash
# 运行完整验证测试套件
python scripts/tools/validate_whole_body_control.py \
    --policy_path=logs/best_model.pt \
    --output=validation_report.html

# 验证项：
# ✅ 所有基础功能测试通过（> 90%）
# ✅ 鲁棒性测试：性能下降 < 20%
# ✅ 泛化测试：未见场景成功率 > 50%
# ✅ 长期运行：性能无明显下降
```

**验证 2：性能基准测试**
```python
# 与基线方法对比

baselines = {
    "hierarchical_control": HierarchicalController(),
    "end_to_end_rl": EndToEndPolicy(),
    "wbc": WholeBodyController(),
    "our_method": OurMethod(),
}

methods_results = {}

for method_name, method in baselines.items():
    results = evaluate_method(method, test_scenarios)
    methods_results[method_name] = results

# 可视化对比
plot_method_comparison(methods_results)

# 成功标准：
# - 我们的方法至少在 3/5 场景中优于所有基线
# - 没有场景明显差于所有基线
```

**验证 3：准备部署**
```python
# 检查部署清单

deployment_checklist = {
    # 策略性能
    "policy_meets_requirements": check_policy_performance(),
    "policy_is_safe": check_safety_constraints(),
    "policy_is_robust": check_robustness(),

    # 代码质量
    "code_passes_linter": run_linter(),
    "code_has_tests": check_test_coverage(),
    "code_is_well_documented": check_documentation(),

    # 仿真验证
    "sim_validation_passed": run_validation_tests(),
    "edge_cases_handled": test_edge_cases(),

    # 真实机器人准备
    "real_robot_available": check_real_robot_access(),
    "safety_protocols_in_place": check_safety_protocols(),
    "emergency_stop_configured": check_emergency_stop(),
}

all_passed = all(deployment_checklist.values())

if all_passed:
    print("✓ 准备部署到真实机器人！")
else:
    print("✗ 还需要解决以下问题：")
    for item, passed in deployment_checklist.items():
        if not passed:
            print(f"  - {item}")
```

---

## 📝 第三阶段总结检查点

### 必须完成（Must Have）

- [ ] **系统搭建**
  - 成功搭建四足+机械臂组合系统
  - 实现分层控制或 WBC 方法
  - 系统能稳定运行（无崩溃、无异常）

- [ ] **策略训练**
  - 完成至少 3 个复杂任务（loco-manip, patrol, transport）
  - 任务成功率 > 60%
  - 多任务学习效果优于单任务

- [ ] **系统验证**
  - 通过所有基础功能测试
  - 鲁棒性和泛化测试达标
  - 长期运行稳定

### 加分项（Nice to Have）

- [ ] 实现了创新的控制方法
- [ ] 完成真实机器人部署
- [ ] 发表了技术论文或专利
- [ ] 开源了高质量代码

---

## 🎓 最终交付物清单

### 代码
- [ ] 完整的四足+机械臂控制代码
- [ ] 训练脚本和配置文件
- [ ] 测试套件和评估工具
- [ ] 部署脚本（如有真实机器人）

### 文档
- [ ] 系统设计文档（架构、接口）
- [ ] 训练报告（超参数、曲线、分析）
- [ ] 验证报告（性能、鲁棒性、泛化）
- [ ] 用户手册（如何使用、调试）

### 演示
- [ ] 视频：展示系统能力（3-5 个场景）
- [ ] 现场演示：如有条件，展示真实机器人
- [ ] 代码演示：walkthrough 主要功能

### 总结
- [ ] 实习总结报告（收获、挑战、未来工作）
- [ ] 技术博客（可选，分享经验）

---

## 🎉 恭喜！完成所有三个阶段

你现在应该已经：
- ✅ 掌握了强化学习在机器人控制中的应用
- ✅ 实现了从简单到复杂的机器人控制任务
- ✅ 具备了独立开展机器人强化学习项目的能力
- ✅ 积累了丰富的仿真和部署经验

**下一步方向：**
- 深入研究某个特定方向（Sim-to-Real、元学习、安全RL等）
- 参与开源项目，贡献代码
- 发表研究成果
- 将技术应用到实际产品中

祝实习顺利！🚀
