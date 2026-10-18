"""Day 1:位置式 PID 控制一阶惯性环节，阶跃响应仿真。

被控对象：一阶惯性环节
    dx/dt = -a*x + b*u
离散化（前向欧拉）：
    x[k+1] = x[k] + dt * (-a*x[k] + b*u[k])

本脚本对比三种参数整定，直观感受 P / I / D 的作用：
    1. P only —— 有稳态误差（一阶系统纯比例无法消除稳态误差）
    2. P + I   —— 积分消除稳态误差
    3. PID     —— 完整形式（注意：对一阶系统 D 收益很小，
                D 的价值要在二阶/有振荡的系统上才明显,Day 3+ 会看到）

运行：在项目根目录执行
    python examples/day1_position_pid.py
生成图片:docs/images/day1_position_pid.png
"""

import sys
from pathlib import Path

# 让脚本能 import 到 src/ 下的算法包（从任意目录运行都能找到）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能正常出图；想弹窗查看可去掉这行
import matplotlib.pyplot as plt

from src.pid.position_pid import PositionPID


class FirstOrderPlant:
    """一阶惯性环节 dx/dt = -a*x + b*u，前向欧拉离散。"""

    def __init__(self, a=1.0, b=0.3, dt=0.01, x0=0.0):
        self.a = a
        self.b = b
        self.dt = dt
        self.x = x0

    def step(self, u):
        self.x += self.dt * (-self.a * self.x + self.b * u)
        return self.x


def simulate(pid, plant, setpoint=1.0, duration=5.0):
    """跑一个阶跃响应，返回时间轴、设定值、输出、控制量。"""
    n = int(duration / plant.dt)
    t = np.zeros(n)
    y = np.zeros(n)
    u = np.zeros(n)
    r = np.full(n, setpoint)

    for k in range(n):
        t[k] = k * plant.dt
        y[k] = plant.x
        u[k] = pid.update(setpoint, y[k])
        plant.step(u[k])

    return t, r, y, u


def main():
    dt = 0.001
    duration = 10.0

    # 三组整定参数，用来对比 P/I/D 各自的作用
    configs = [
        ("P only", dict(kp=3.0, ki=2.0, kd=0.0)),
        ("PI",     dict(kp=3.0, ki=8.0, kd=0.0)),
        ("PID",    dict(kp=4.0, ki=2.0, kd=0.3)),
    ]

    fig, (ax_y, ax_u) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    for name, params in configs:
        pid = PositionPID(dt=dt, **params)
        plant = FirstOrderPlant(dt=dt)
        t, r, y, u = simulate(pid, plant, duration=duration)
        ax_y.plot(t, y, label=name)
        ax_u.plot(t, u, label=name)

    ax_y.plot(t, r, "k--", label="setpoint")
    ax_y.set_ylabel("output y")
    ax_y.set_title("Position PID: step response of first-order plant")
    ax_y.legend()
    ax_y.grid(True, alpha=0.3)

    ax_u.set_xlabel("time [s]")
    ax_u.set_ylabel("control u")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "day1_position_pid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
