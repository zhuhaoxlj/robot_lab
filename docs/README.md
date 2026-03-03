# 实习期学习计划文档索引

本目录包含完整的实习期（12 周）强化学习机器人控制学习计划。

## 📚 文档结构

```
docs/
├── INTERNSHIP_PLAN_OVERVIEW.md    # 总览：快速参考和周度检查清单
├── INTERNSHIP_PLAN.md              # 第一阶段：强化学习基础与控制框架（Week 1-4）
├── INTERNSHIP_PLAN_STAGE2.md       # 第二阶段：机械臂抓取策略设计（Week 5-8）
└── INTERNSHIP_PLAN_STAGE3.md       # 第三阶段：四足+机械臂全身控制（Week 9-12）
```

## 🎯 使用指南

### 1. 开始前：阅读总览文档
```bash
# 从这里开始！
cat docs/INTERNSHIP_PLAN_OVERVIEW.md
```

总览文档包含：
- ✅ 完整的时间规划表
- ✅ 每周核心任务和验证方式
- ✅ 周度检查清单
- ✅ 快速启动指南
- ✅ 常见问题 FAQ

### 2. 执行中：参考详细计划

**第一阶段（Week 1-4）**：`INTERNSHIP_PLAN.md`
- 理论基础与代码理解
- 状态空间设计
- 动作空间与奖励设计
- 训练稳定性与优化

**第二阶段（Week 5-8）**：`INTERNSHIP_PLAN_STAGE2.md`
- 机械臂仿真环境搭建
- 抓取任务设计
- 稀疏奖励与探索策略
- 模仿学习与强化学习结合

**第三阶段（Week 9-12）**：`INTERNSHIP_PLAN_STAGE3.md`
- 组合仿真环境搭建
- 全身运动规划
- 全身协同控制
- 系统验证与部署

### 3. 每周：使用检查清单

从总览文档中复制"周度检查清单"，每周检查进度。

---

## 📋 快速导航

### 按主题查找

| 主题 | 文档 | 章节 |
|------|------|------|
| PPO 算法原理 | INTERNSHIP_PLAN.md | Week 1, Day 1-2 |
| 状态空间设计 | INTERNSHIP_PLAN.md | Week 2 |
| 奖励函数设计 | INTERNSHIP_PLAN.md | Week 3 |
| 超参数调优 | INTERNSHIP_PLAN.md | Week 4 |
| 机械臂导入 | INTERNSHIP_PLAN_STAGE2.md | Week 5 |
| Reach 任务 | INTERNSHIP_PLAN_STAGE2.md | Week 6 |
| Pick-Place 任务 | INTERNSHIP_PLAN_STAGE2.md | Week 6 |
| Curriculum Learning | INTERNSHIP_PLAN_STAGE2.md | Week 7 |
| Hindsight Experience Replay | INTERNSHIP_PLAN_STAGE2.md | Week 7 |
| Behavior Cloning | INTERNSHIP_PLAN_STAGE2.md | Week 8 |
| 四足+机械臂组合 | INTERNSHIP_PLAN_STAGE3.md | Week 9 |
| 分层控制 | INTERNSHIP_PLAN_STAGE3.md | Week 10 |
| Whole-Body Control | INTERNSHIP_PLAN_STAGE3.md | Week 10 |
| 多任务学习 | INTERNSHIP_PLAN_STAGE3.md | Week 11 |
| 系统验证 | INTERNSHIP_PLAN_STAGE3.md | Week 12 |

### 按验证类型查找

| 验证类型 | 位置 | 描述 |
|----------|------|------|
| 理论测试 | INTERNSHIP_PLAN.md | Week 1, 验证方式 1 |
| 代码流程测试 | INTERNSHIP_PLAN.md | Week 1, 验证方式 2 |
| 观察函数测试 | INTERNSHIP_PLAN.md | Week 2, 验证 1 |
| 动作空间对比 | INTERNSHIP_PLAN.md | Week 3, 验证 1 |
| 超参数优化 | INTERNSHIP_PLAN.md | Week 4, 验证 1 |
| 机械臂模型测试 | INTERNSHIP_PLAN_STAGE2.md | Week 5, 验证 1 |
| Reach 任务测试 | INTERNSHIP_PLAN_STAGE2.md | Week 6, 验证 1 |
| Curriculum Learning | INTERNSHIP_PLAN_STAGE2.md | Week 7, 验证 1 |
| HER 效果验证 | INTERNSHIP_PLAN_STAGE2.md | Week 7, 验证 2 |
| BC 策略评估 | INTERNSHIP_PLAN_STAGE2.md | Week 8, 验证 1 |
| 系统耦合分析 | INTERNSHIP_PLAN_STAGE3.md | Week 9, 验证 2 |
| 分层控制性能 | INTERNSHIP_PLAN_STAGE3.md | Week 10, 验证 1 |
| 多任务测试 | INTERNSHIP_PLAN_STAGE3.md | Week 11, 验证 1 |
| 系统验证 | INTERNSHIP_PLAN_STAGE3.md | Week 12, 验证 1 |

---

## 🚀 快速开始

### 第一周快速启动命令

```bash
# 1. 阅读 PPO 论文
# https://arxiv.org/abs/1707.06347

# 2. 运行简单训练
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0 \
    --num_envs 64 \
    --max_iterations 10

# 3. 查看训练曲线
tensorboard --logdir=logs

# 4. 完成理论测试
# 见 INTERNSHIP_PLAN.md 中的"验证方式 1"
```

---

## 📊 进度跟踪

### 使用 Markdown 任务列表

在每个文档的章节开头使用任务列表：

```markdown
### Week 1: 理论基础

**核心任务：**
- [ ] 深入理解 PPO/SAC 算法原理
- [ ] 掌握代码库架构
- [ ] 绘制数据流图

**验证方式：**
- [ ] 算法原理测试
- [ ] 代码流程复现
- [ ] MDP 建模设计
```

### 使用 Git 追踪代码贡献

```bash
# 创建分支用于每周任务
git checkout -b week-1-theory-and-code

# 提交每日进度
git commit -m "feat: complete PPO algorithm notes"
git commit -m "feat: implement base_height observation"
git commit -m "feat: finish code flow diagram"

# 周末合并到主分支
git checkout main
git merge week-1-theory-and-code
```

---

## 📝 周报模板位置

每周使用此模板提交周报：

```
docs/weekly_reports/
├── week-1_YYYY-MM-DD.md
├── week-2_YYYY-MM-DD.md
├── ...
└── week-12_YYYY-MM-DD.md
```

周报模板见 `INTERNSHIP_PLAN_OVERVIEW.md` 中的"周报模板"章节。

---

## 💡 提示和技巧

### 时间管理
- **周一上午**（1h）：阅读本周计划，列出任务清单
- **每天晚上**（5min）：记录当天进度和遇到的问题
- **周五下午**（1h）：总结本周工作，评估完成度

### 学习效率
- 使用番茄工作法：25 分钟专注 + 5 分钟休息
- 每学习 50 分钟，休息 10 分钟（避免疲劳）
- 在精力最好的时间段做最困难的任务

### 问题解决
- 遇到问题先尝试自己解决（15 分钟）
- 如果无法解决，记录下来并继续其他任务
- 定期（如每周五）统一询问导师

### 代码管理
- 每完成一个功能，立即提交代码
- 提交信息使用 Conventional Commits 格式
- 定期推送到远程仓库（备份）

---

## 🎓 关键里程碑

### 第一阶段里程碑（Week 4）
- [ ] 理解 PPO 算法原理
- [ ] 能实现新的观察/奖励函数
- [ ] 能诊断和修复训练问题
- [ ] 至少完成 3 次完整训练实验

### 第二阶段里程碑（Week 8）
- [ ] 成功搭建机械臂仿真环境
- [ ] Reach 任务成功率 > 80%
- [ ] Pick-Place 任务成功率 > 50%
- [ ] 掌握稀疏奖励下的训练技巧

### 第三阶段里程碑（Week 12）
- [ ] 成功搭建四足+机械臂系统
- [ ] 至少 3 个复杂任务成功率 > 60%
- [ ] 通过所有基础功能测试
- [ ] 完成系统验证报告

---

## 🔗 相关资源

### 内部资源
- 代码库：`/home/xiao/zh/robot_lab`
- 日志目录：`logs/`
- 演示视频：`logs/*/videos/`

### 外部资源
- Isaac Lab 文档：[isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab/)
- PPO 论文：[arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- 强化学习课程：[spinningup.openai.com](https://spinningup.openai.com/)

### 工具安装
```bash
# TensorBoard
pip install tensorboard

# Optuna（超参数优化）
pip install optuna

# matplotlib（可视化）
pip install matplotlib

# jupyter（实验记录）
pip install jupyter
```

---

## 📈 技能发展路径

### 从新手到熟练
**Week 1-2（新手）**：
- 跟随教程完成第一个训练
- 理解基本概念和术语
- 能运行和修改示例代码

**Week 3-6（进阶）**：
- 能独立实现新功能
- 能诊断和解决训练问题
- 能设计和运行实验

**Week 7-10（熟练）**：
- 能优化现有系统
- 能处理复杂任务
- 能进行创新性工作

**Week 11-12（专家）**：
- 能指导他人
- 能进行系统性改进
- 能发表技术成果

---

## ✅ 最终检查清单

### 实习结束前确认

**代码质量**：
- [ ] 代码通过 linter 检查
- [ ] 关键功能有单元测试
- [ ] 代码有充分注释
- [ ] 有 README 和使用说明

**文档完整性**：
- [ ] 设计文档完整
- [ ] 训练报告详细
- [ ] 验证报告全面
- [ ] 用户手册清晰

**演示准备**：
- [ ] 录制演示视频（3-5 个场景）
- [ ] 准备现场演示（如适用）
- [ ] 准备代码 walkthrough
- [ ] 准备 Q&A

**总结材料**：
- [ ] 实习总结报告
- [ ] 技术博客（可选）
- [ ] 未来工作建议

---

**祝实习顺利！记住：循序渐进、持续学习、及时总结、保持热情！** 🚀✨

---

## 📞 联系方式

如有问题或建议，请：
- 提交 GitHub Issue
- 发送邮件
- 在 Discord 群组讨论

**文档版本**：v1.0
**最后更新**：2026-03-03
**维护者**：[Your Name]
