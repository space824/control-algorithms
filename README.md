# control-algorithms

一个用纯 Python 实现的控制算法库与仿真合集——PID、滤波器、状态估计（KF/EKF）、最优控制（LQR/MPC）。

> 🚧 正在建设中，按四周计划逐步添加算法。当前进度：Day 8 状态空间模型。

## 算法列表

- [x] 位置式 PID（含抗积分饱和）
- [x] 增量式 PID（天然抗积分饱和；与位置式差分等价，不平滑掉微分冲击）
- [x] 一阶低通滤波（指数移动平均，调 α 控制截止频率）
- [x] 二阶低通滤波（巴特沃斯 biquad，−40 dB/十倍频，Q 可调）
- [x] 互补滤波（陀螺 + 加速度计融合，固定增益观测器）
- [x] 状态空间模型（A/B/C/D 建模、可控性/可观性、ZOH 离散）
- [ ] 卡尔曼滤波 KF / 扩展卡尔曼滤波 EKF
- [ ] LQR 求解器 + 仿真
- [ ] 简易 MPC

## 专题

- [x] 离散化（前向欧拉 / 后向欧拉 / ZOH，采样周期怎么选）

## 快速开始

```bash
pip install -r requirements.txt
python examples/day1_position_pid.py   # 生成 docs/images/day1_position_pid.png
python examples/day2_incremental_pid.py  # 生成 docs/images/day2_incremental_pid.png
python examples/day3_lowpass_filter.py   # 生成 docs/images/day3_lowpass_filter.png
python examples/day4_second_order_lowpass.py  # 生成 docs/images/day4_second_order_lowpass.png
python examples/day5_complementary_filter.py  # 生成 docs/images/day5_complementary_filter.png
python examples/day6_discretization.py   # 生成 docs/images/day6_discretization.png
python examples/day8_state_space.py     # 生成 docs/images/day8_state_space.png
```

> 目前只依赖 **numpy + matplotlib**。`requirements.txt` 里的 scipy 尚未用到——Day 6 的矩阵指数是自己实现的（缩放-平方 + 泰勒），这样不用为了一课装依赖，也能看清 ZOH 到底在算什么。

## Day 6 的一条结论

同一个控制器、同一个对象，**只把采样周期从 10 ms 放大到 200 ms，PI 闭环的超调就从 40% 涨到 84%**。
原因不是"算得粗"，是"延迟变大"：1 拍计算延迟 = 1 个 Ts，ZOH 还额外约 Ts/2。
所以采样周期必须和**闭环速度**挂钩，没有一个通用的数字。

## Day 8 的一条结论
同一个二阶系统，Day 4 叫它 `Q=0.707`（巴特沃斯，频域最平坦），今天叫它 `ζ=0.707`（时域 4.32% 超调）。
**两个「临界」不是一回事**：通带鼓包的临界是 Q>0.707，阶跃超调的临界是 ζ<1。

## 目录结构

```
src/          # 算法实现（按模块分类）
  pid/        # 位置式 / 增量式 PID
  filter/     # 一阶 / 二阶低通、互补滤波
  discrete/   # 连续→离散（前向/后向欧拉、ZOH）
  state_space/ # 状态空间建模（A/B/C/D、可控可观、离散化）
examples/     # 每个算法一个可运行示例脚本，跑一次出一张图
docs/images/  # 仿真结果图
matlab实现/    # 同一批算法的 MATLAB 实现 + 逐段解析（Day 1–5）
每日学习内容调试/ # 每天的原理讲解、图形解读与调试记录
```
