"""Day 5: 互补滤波 —— 陀螺仪 + 加速度计融合角度。

    angle[k] = α·(angle[k-1] + (gyro[k] - bias)·dt) + (1-α)·angle_acc[k]

陀螺积分：短时间准、长时间漂移（零偏积分）；加速度计：长期不漂、短期噪声大。
互补滤波让两者各管一段频带。

本脚本做四件事（四张子图），核心是回答：
    「为什么要融合？融合之后到底比单独用强多少？」

    (a) 时域：真值 / 纯陀螺积分 / 纯加速度计 / 互补滤波
    (b) 误差曲线：三种估计各自的误差随时间怎么长（陀螺是线性发散）
    (c) α 扫描：RMSE 随 α 怎么变，最优点在哪
    (d) 「互补」到底互补在哪：加速度计路径的低通、陀螺路径的高通，两者和恒为 1

运行：在项目根目录执行
    python examples/day5_complementary_filter.py
生成图片: docs/images/day5_complementary_filter.png
"""

import sys
from pathlib import Path

# 让脚本能 import 到 src/ 下的算法包（从任意目录运行都能找到）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能正常出图；想弹窗查看可去掉这行
import matplotlib.pyplot as plt

from src.filter.complementary import ComplementaryFilter


# ---------------------------------------------------------------- 工具函数
def simulate(dt, duration, rng, f_tilt=0.05, amp=20.0,
             gyro_bias=0.5, gyro_noise=0.1, accel_noise=1.5):
    """造一段「真值 + 两个传感器的脏测量」。

    返回 dict，包含 t、真值、陀螺读数、加速度计读数。
    """
    n = int(duration / dt)
    t = np.arange(n) * dt

    # 真实角度：缓慢倾斜（0.05 Hz，30 秒走 1.5 个周期）
    angle_true = amp * np.sin(2.0 * np.pi * f_tilt * t)
    # 真实角速度 = 角度求导（解析式，不用差分）
    rate_true = amp * 2.0 * np.pi * f_tilt * np.cos(2.0 * np.pi * f_tilt * t)

    # 陀螺：真实角速度 + 零偏 + 白噪声
    gyro = rate_true + gyro_bias + gyro_noise * rng.standard_normal(n)
    # 加速度计：真实角度 + 明显更大的白噪声
    accel = angle_true + accel_noise * rng.standard_normal(n)

    return dict(n=n, t=t, angle_true=angle_true, gyro=gyro, accel=accel,
                gyro_bias=gyro_bias)


def run_estimators(sim, alpha=0.98, dt=0.01):
    """三种估计：纯陀螺积分 / 纯加速度计 / 互补滤波。"""
    n, gyro, accel = sim["n"], sim["gyro"], sim["accel"]

    # ① 纯陀螺积分（不做零偏补偿，模拟"没标定"的真实情况）
    angle_gyro = np.empty(n)
    angle_gyro[0] = 0.0
    for k in range(1, n):
        angle_gyro[k] = angle_gyro[k - 1] + gyro[k] * dt

    # ② 纯加速度计（直接用）
    angle_acc = accel.copy()

    # ③ 互补滤波
    cf = ComplementaryFilter(alpha=alpha, dt=dt, angle0=0.0)
    angle_cf = np.empty(n)
    for k in range(n):
        angle_cf[k] = cf.update(gyro[k], accel[k])

    return angle_gyro, angle_acc, angle_cf, cf


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def accel_path_response(alpha, dt, f):
    """加速度计路径的频响：一阶低通 H = (1-α)/(1-α·e^{-jω·dt})。"""
    w = 2.0 * np.pi * np.asarray(f, dtype=float) * dt
    return (1.0 - alpha) / (1.0 - alpha * np.exp(-1j * w))


# ---------------------------------------------------------------- 主流程
def main():
    dt = 0.01                  # 100 Hz，MPU6050 的典型采样率
    duration = 30.0            # 30 秒，够让陀螺漂移到肉眼可见
    alpha = 0.98               # 课本里的经典值
    rng = np.random.default_rng(0)   # 固定随机种子，结果可复现

    sim = simulate(dt=dt, duration=duration, rng=rng)
    t, angle_true = sim["t"], sim["angle_true"]

    a_gyro, a_acc, a_cf, cf = run_estimators(sim, alpha=alpha, dt=dt)

    r_gyro, r_acc, r_cf = rmse(a_gyro, angle_true), rmse(a_acc, angle_true), rmse(a_cf, angle_true)
    drift_end = float(a_gyro[-1] - angle_true[-1])

    print(f"dt = {dt}s (100Hz) | 时长 {duration}s | alpha = {alpha}")
    print(f"陀螺参数：零偏 {sim['gyro_bias']}°/s，噪声 0.1°/s | 加速度计噪声 1.5°")
    print(f"互补滤波：tau = {cf.tau:.3f}s，fc = {cf.cutoff_hz:.2f}Hz")
    print("-" * 60)
    print(f"{'估计方法':16s}{'RMSE(对真值)':>14s}{'末态误差':>12s}")
    print(f"{'纯陀螺积分':16s}{r_gyro:14.4f}{drift_end:11.2f}°")
    print(f"{'纯加速度计':16s}{r_acc:14.4f}{0.0:11.2f}°")
    print(f"{'互补滤波':16s}{r_cf:14.4f}{float(a_cf[-1]-angle_true[-1]):11.2f}°")
    print("-" * 60)

    # ---------- α 扫描 ----------
    alphas = np.linspace(0.50, 0.995, 60)
    scan = []
    for a in alphas:
        _, _, y, _ = run_estimators(sim, alpha=a, dt=dt)
        scan.append(rmse(y, angle_true))
    scan = np.array(scan)
    best_i = int(np.argmin(scan))
    print(f"alpha 扫描：最优 alpha = {alphas[best_i]:.4f}，RMSE = {scan[best_i]:.4f}")
    for a in (0.90, 0.95, 0.98, 0.995):
        j = int(np.argmin(np.abs(alphas - a)))
        f = ComplementaryFilter(alpha=a, dt=dt)
        print(f"  alpha={alphas[j]:.3f} -> tau={f.tau:6.3f}s, fc={f.cutoff_hz:6.3f}Hz, "
              f"RMSE={scan[j]:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ---------- (a) 时域对比 ----------
    ax = axes[0, 0]
    ax.plot(t, angle_true, "k--", lw=2, label="true angle")
    ax.plot(t, a_acc, color="0.7", lw=0.6, alpha=0.8, label=f"accel only (RMSE {r_acc:.2f}°)")
    ax.plot(t, a_gyro, color="tab:red", lw=1.2, label=f"gyro integrate (RMSE {r_gyro:.2f}°)")
    ax.plot(t, a_cf, color="tab:blue", lw=2.0, label=f"complementary (RMSE {r_cf:.2f}°)")
    ax.set_title(f"(a) Fusion at alpha={alpha}, dt={dt*1000:.0f}ms, fc={cf.cutoff_hz:.2f}Hz")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [deg]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---------- (b) 误差曲线 ----------
    ax = axes[0, 1]
    ax.plot(t, a_gyro - angle_true, color="tab:red", lw=1.4, label="gyro integrate")
    ax.plot(t, a_acc - angle_true, color="0.6", lw=0.6, alpha=0.8, label="accel only")
    ax.plot(t, a_cf - angle_true, color="tab:blue", lw=1.8, label="complementary")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("(b) Error vs truth — gyro drifts linearly")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [deg]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---------- (c) alpha 扫描 ----------
    ax = axes[1, 0]
    ax.plot(alphas, scan, color="tab:blue", lw=1.8)
    ax.plot([alphas[best_i]], [scan[best_i]], "o", color="tab:red", ms=8, zorder=5)
    j = int(np.argmin(np.abs(alphas - 0.98)))
    ax.plot([alphas[j]], [scan[j]], "s", color="tab:orange", ms=7, zorder=5)
    ax.annotate(f"best a={alphas[best_i]:.3f}\nRMSE {scan[best_i]:.2f}°",
                (alphas[best_i], scan[best_i]), textcoords="offset points",
                xytext=(8, 12), fontsize=8)
    ax.annotate("a=0.98 (book)", (alphas[j], scan[j]), textcoords="offset points",
                xytext=(-58, 16), fontsize=8, color="tab:orange")
    ax.set_title("(c) RMSE vs alpha — too small trusts accel, too big trusts gyro")
    ax.set_xlabel("alpha (gyro weight)")
    ax.set_ylabel("RMSE [deg]")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # ---------- (d) 「互补」的含义：低通 + 高通 ----------
    ax = axes[1, 1]
    f = np.logspace(-2, np.log10(0.5 / dt), 600)
    H_lp = accel_path_response(alpha, dt, f)
    H_hp = 1.0 - H_lp
    ax.semilogx(f, np.abs(H_lp), color="tab:blue", lw=1.8,
                label="accel path |H| (low-pass)")
    ax.semilogx(f, np.abs(H_hp), color="tab:orange", lw=1.8,
                label="gyro path |H| (high-pass)")
    ax.axvline(cf.cutoff_hz, ls=":", color="gray", alpha=0.9)
    ax.text(cf.cutoff_hz * 1.08, 0.04, f"fc={cf.cutoff_hz:.2f}Hz", fontsize=8, rotation=90)
    ax.axhline(0.5, ls="--", color="k", lw=0.8, alpha=0.6)
    ax.set_title("(d) Why 'complementary': low-pass + high-pass = 1")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("weight")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "day5_complementary_filter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
