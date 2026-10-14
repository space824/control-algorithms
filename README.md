# control-algorithms

一个用纯 Python 实现的控制算法库与仿真合集——PID、滤波器、状态估计（KF/EKF）、最优控制（LQR/MPC）。

> 🚧 正在建设中，按四周计划逐步添加算法。当前进度：Day 1 位置式 PID。

## 算法列表

- [x] 位置式 PID（含抗积分饱和）
- [ ] 增量式 PID
- [ ] 一阶/二阶低通滤波
- [ ] 互补滤波
- [ ] 卡尔曼滤波 KF / 扩展卡尔曼滤波 EKF
- [ ] LQR 求解器 + 仿真
- [ ] 简易 MPC

## 快速开始

```bash
pip install -r requirements.txt
python examples/day1_position_pid.py   # 生成 docs/images/day1_position_pid.png
```

## 目录结构

```
src/        # 算法实现（按模块分类）
examples/   # 每个算法一个可运行示例脚本，跑一次出一张图
docs/images/ # 仿真结果图
```
