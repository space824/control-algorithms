# control-algorithms

一个用纯 Python 实现的控制算法库与仿真合集——PID、滤波器、状态估计（KF/EKF）、最优控制（LQR/MPC）。

> 🚧 正在建设中，按四周计划逐步添加算法。当前进度：Day 5 互补滤波。

## 算法列表

- [x] 位置式 PID（含抗积分饱和）
- [x] 增量式 PID（天然抗积分饱和；与位置式差分等价，不平滑掉微分冲击）
- [x] 一阶低通滤波（指数移动平均，调 α 控制截止频率）
- [x] 二阶低通滤波（巴特沃斯 biquad，−40 dB/十倍频，Q 可调）
- [x] 互补滤波（陀螺 + 加速度计融合，固定增益观测器）
- [ ] 卡尔曼滤波 KF / 扩展卡尔曼滤波 EKF
- [ ] LQR 求解器 + 仿真
- [ ] 简易 MPC

## 快速开始

```bash
pip install -r requirements.txt
python examples/day1_position_pid.py   # 生成 docs/images/day1_position_pid.png
python examples/day2_incremental_pid.py  # 生成 docs/images/day2_incremental_pid.png
python examples/day3_lowpass_filter.py   # 生成 docs/images/day3_lowpass_filter.png
python examples/day4_second_order_lowpass.py  # 生成 docs/images/day4_second_order_lowpass.png
python examples/day5_complementary_filter.py  # 生成 docs/images/day5_complementary_filter.png
```

## 最近两条结论

**Day 4**：一阶和二阶在**白噪声**下几乎打平（噪声放大 0.334 vs 0.281，只差 1.5 dB）——等效噪声带宽只与延迟成反比，与阶数无关。
二阶真正的价值在**窄带干扰**：120 Hz 处一阶 −10.7 dB、二阶 −21.9 dB，差 11.2 dB。

**Day 5**：互补滤波把角度 RMSE 从 1.49°（纯加速度计）降到 **0.32°**，比好 4.7 倍；纯陀螺积分 30 秒后漂了 14.8°。
它的结构 `angle = pred + K·(meas − pred)` 就是**固定增益的观测器**——Day 9 的卡尔曼只是把 K 换成随协方差自动变化的值。

## 目录结构

```
src/          # 算法实现（按模块分类）
examples/     # 每个算法一个可运行示例脚本，跑一次出一张图
docs/images/  # 仿真结果图
matlab实现/    # 同一批算法的 MATLAB 实现 + 逐段解析
每日学习内容调试/ # 每天的原理讲解、图形解读与调试记录
```
