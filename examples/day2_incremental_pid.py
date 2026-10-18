"""Day 2:增量式 PID —— 对比位置式与增量式的关键区别。

本脚本用相同的增益、不同的 PID 结构，对比两件事：
    1. 设定值阶跃时的微分冲击（derivative kick）：
       增益配平后两种结构完全等价，(b) 里两条曲线在 t=0 跳出同一个尖峰
       （本例 33.02）。微分冲击来自"对误差求导"，不是增量式能消除的，
       真正的解法是微分先行（让 D 作用在测量值上）。这一组图的价值在于
       看清尖峰打哪来，以及两种形式其实是同一回事。
    2. 积分饱和（integral windup）：
       输出被限幅时，位置式（若无抗饱和）会无限累积积分，松开后造成超调；
       增量式没有积分累加器，天然无此问题。—— 这才是两者真正的区别。

运行：在项目根目录执行
    python examples/day2_incremental_pid.py
生成图片:docs/images/day2_incremental_pid.png
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
from src.pid.incremental_pid import IncrementalPID


class SecondOrderPlant:
    """二阶系统 x'' + 2*zeta*wn*x' + wn^2*x = b*u，前向欧拉离散。

    欠阻尼的二阶系统会振荡，微分项的价值在这种系统上才明显，
    设定值阶跃时也最容易看清"微分冲击"那根尖峰。
    """

    def __init__(self, zeta=0.3, wn=2.0, b=4.0, dt=0.01, x0=0.0, v0=0.0):
        self.zeta = zeta
        self.wn = wn
        self.b = b
        self.dt = dt
        self.x = x0
        self.v = v0

    def step(self, u):
        self.v += self.dt * (-2 * self.zeta * self.wn * self.v
                             - self.wn ** 2 * self.x
                             + self.b * u)
        self.x += self.dt * self.v
        return self.x


class FirstOrderPlant:
    """一阶惯性环节 dx/dt = -a*x + b*u，前向欧拉离散。"""

    def __init__(self, a=1.0, b=1.0, dt=0.01, x0=0.0):
        self.a = a
        self.b = b
        self.dt = dt
        self.x = x0

    def step(self, u):
        self.x += self.dt * (-self.a * self.x + self.b * u)
        return self.x


class NaivePositionPID:
    """无抗饱和的位置式 PID，仅用于演示"积分饱和"是怎么产生的。

    注意：它与 src/pid/position_pid.py 里带条件积分抗饱和的版本不同，
    这里故意不做任何抗饱和处理，积分项会无条件累积。
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.01,
                 output_limits=(None, None)):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.lo, self.hi = output_limits
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, setpoint, measurement):
        e = setpoint - measurement
        self._integral += e * self.dt  # 无条件累积，饱和时也会涨
        derivative = (e - self._prev_error) / self.dt
        u = self.kp * e + self.ki * self._integral + self.kd * derivative
        if self.hi is not None and u > self.hi:
            u = self.hi
        elif self.lo is not None and u < self.lo:
            u = self.lo
        self._prev_error = e
        return u


def current_setpoint(t, setpoint_seq):
    """setpoint_seq 是 (开始时刻, 设定值) 的有序列表，返回当前时刻的设定值。"""
    sp = setpoint_seq[0][1]
    for t_start, sp_val in setpoint_seq:
        if t >= t_start:
            sp = sp_val
    return sp


def simulate(pid, plant, setpoint_seq, duration):
    """按时间轴跑仿真，返回时间轴、设定值、输出、控制量。"""
    n = int(duration / plant.dt)
    t = np.zeros(n)
    r = np.zeros(n)
    y = np.zeros(n)
    u = np.zeros(n)

    for k in range(n):
        t[k] = k * plant.dt
        r[k] = current_setpoint(t[k], setpoint_seq)
        y[k] = plant.x
        u[k] = pid.update(r[k], y[k])
        plant.step(u[k])

    return t, r, y, u


def main():
    dt = 0.01
    gains = dict(kp=3.0, ki=2.0, kd=0.3)  # 两组共用同一组增益：配平后两者应当完全等价

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ---------- 面板 1：设定值阶跃的微分冲击 ----------
    # 二阶欠阻尼系统，设定值在 t=0 从 0 阶跃到 1
    pos = PositionPID(dt=dt, **gains)
    plant_pos = SecondOrderPlant(dt=dt)
    t1, r1, y_pos, u_pos = simulate(pos, plant_pos, [(0.0, 1.0)], duration=6.0)

    inc = IncrementalPID(dt=dt, **gains)
    plant_inc = SecondOrderPlant(dt=dt)
    _, _, y_inc, u_inc = simulate(inc, plant_inc, [(0.0, 1.0)], duration=6.0)

    ax = axes[0, 0]
    ax.plot(t1, r1, "k--", label="setpoint")
    ax.plot(t1, y_pos, label="positional")
    ax.plot(t1, y_inc, label="incremental")
    ax.set_title("(a) Step response: second-order plant (output y)")
    ax.set_ylabel("output y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t1, u_pos, label="positional")
    ax.plot(t1, u_inc, label="incremental")
    ax.set_title("(b) Control u: derivative kick at the setpoint step")
    ax.set_ylabel("control u")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---------- 面板 2：积分饱和 ----------
    # 一阶系统，输出限幅到 (0, 1)；设定值先要 3（迫使控制量饱和），
    # 5 秒后再降到 0.5，观察松开饱和后两种控制器是否超调
    limits = (0.0, 1.0)

    naive = NaivePositionPID(dt=dt, output_limits=limits, **gains)
    plant_naive = FirstOrderPlant(dt=dt)
    t2, r2, y_naive, u_naive = simulate(
        naive, plant_naive, [(0.0, 3.0), (5.0, 0.5)], duration=10.0)

    inc2 = IncrementalPID(dt=dt, output_limits=limits, **gains)
    plant_inc2 = FirstOrderPlant(dt=dt)
    _, _, y_inc2, u_inc2 = simulate(
        inc2, plant_inc2, [(0.0, 3.0), (5.0, 0.5)], duration=10.0)

    ax = axes[1, 0]
    ax.plot(t2, r2, "k--", label="setpoint")
    ax.plot(t2, y_naive, label="positional (no anti-windup)")
    ax.plot(t2, y_inc2, label="incremental")
    ax.set_title("(c) Integral windup: overshoot after limit released (y)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t2, u_naive, label="positional (no anti-windup)")
    ax.plot(t2, u_inc2, label="incremental")
    ax.set_title("(d) Control u: positional integral keeps growing")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("control u")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "day2_incremental_pid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
