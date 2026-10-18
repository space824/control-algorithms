"""Day 4: 二阶低通滤波（巴特沃斯 biquad）—— 与一阶正面对比。

一阶：y[k] = α·x[k] + (1-α)·y[k-1]                → 高频 -20 dB/十倍频
二阶：y[k] = b0·x + b1·x[k-1] + b2·x[k-2]
             − a1·y[k-1] − a2·y[k-2]              → 高频 -40 dB/十倍频

本脚本做四件事（四张子图），核心是回答学习计划里那两个问题：
    「谁的噪声抑制更好？谁的延迟更大？」

    (a) 时间域：同一段含噪正弦，一阶 vs 二阶（同一截止频率）
    (b) 阶跃响应：一阶无超调，二阶巴特沃斯约 4.3% 超调
    (c) 幅频响应：滚降斜率差一倍，但 -3dB 点落在同一条竖线上
    (d) 权衡曲线：扫截止频率，画「相位延迟 vs 残余噪声」——
        白噪声下两条曲线几乎重合（阶数换不来好处），
        120Hz 窄带干扰下二阶明显更靠左下（这才是二阶的主场）。

运行：在项目根目录执行
    python examples/day4_second_order_lowpass.py
生成图片: docs/images/day4_second_order_lowpass.png
"""

import sys
from pathlib import Path

# 让脚本能 import 到 src/ 下的算法包（从任意目录运行都能找到）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能正常出图；想弹窗查看可去掉这行
import matplotlib.pyplot as plt

from src.filter.lowpass import FirstOrderLowPass
from src.filter.second_order import SecondOrderLowPass


# ---------------------------------------------------------------- 工具函数
def first_order_coeffs(lp):
    """把一阶滤波器也写成 biquad 形式 (b0,b1,b2,a1,a2)，方便共用分析函数。

    y[k] = α·x[k] + (1-α)·y[k-1]  ⟺  b0 = α, a1 = -(1-α)，其余为 0。
    """
    return (lp.alpha, 0.0, 0.0, -(1.0 - lp.alpha), 0.0)


def response(coeffs, dt, f):
    """通用 biquad 频率响应 H(e^{jω})，ω = 2π·f·dt。"""
    b0, b1, b2, a1, a2 = coeffs
    w = 2.0 * np.pi * np.asarray(f, dtype=float) * dt
    e1 = np.exp(-1j * w)
    e2 = e1 ** 2
    return (b0 + b1 * e1 + b2 * e2) / (1.0 + a1 * e1 + a2 * e2)


def noise_gain(coeffs, dt, n=8192):
    """白噪声放大倍数：sqrt(mean(|H|²))，ω 在 [0, π] 上均匀取样。

    输入噪声标准差 σ，输出噪声标准差 ≈ σ · noise_gain。
    """
    w = np.linspace(0.0, np.pi, n)
    H = response(coeffs, dt, w / (2.0 * np.pi * dt))
    return float(np.sqrt(np.mean(np.abs(H) ** 2)))


def gain_at(coeffs, dt, f0):
    """在单个频率 f0 处的幅度增益 |H(f0)|。"""
    return float(np.abs(response(coeffs, dt, f0)))


def phase_delay_ms(coeffs, dt, f_sig):
    """在频率 f_sig 处的相位延迟（毫秒）。相位 φ 为负表示滞后，等效延迟 = -φ/ω。"""
    H = response(coeffs, dt, f_sig)
    return -float(np.angle(H)) / (2.0 * np.pi * f_sig) * 1000.0


def filter_signal(filt, x):
    """把整段信号喂进滤波器（一拍拍更新），返回滤波后的数组。"""
    y = np.empty_like(x, dtype=float)
    for k in range(len(x)):
        y[k] = filt.update(x[k])
    return y


def step_response(filt, dt_step, n, amp=1.0):
    """对单位阶跃做响应，返回 (t, y)。"""
    y = np.empty(n, dtype=float)
    for k in range(n):
        y[k] = filt.update(amp)
    return np.arange(n) * dt_step, y


def step_metrics(t, y, amp=1.0):
    """阶跃指标：超调 %、上升时间(10%→90%)、调节时间(±2%)。"""
    yn = y / amp
    overshoot = (yn.max() - 1.0) * 100.0
    i10 = int(np.argmax(yn >= 0.1))
    i90 = int(np.argmax(yn >= 0.9))
    t_rise = (i90 - i10) * (t[1] - t[0])
    outside = np.where(np.abs(yn - 1.0) > 0.02)[0]
    i_set = int(outside[-1]) + 1 if len(outside) else 0
    return overshoot, t_rise, i_set * (t[1] - t[0])


# ---------------------------------------------------------------- 主流程
def main():
    dt = 0.001                 # 采样周期 1ms（1kHz），与 Day 3 保持一致
    f_sig = 3.0                # 正弦信号频率 3Hz
    f_int = 120.0              # 窄带干扰频率：位于 fc 上方约 3.4 倍
    duration = 1.0             # 1 秒（3 个周期）
    fc = 35.7                  # 截止频率：正好是 Day 3 里 α=0.2 对应的 fc
    noise_std = 0.4            # 高斯噪声标准差
    rng = np.random.default_rng(0)   # 固定随机种子，结果可复现

    n = int(duration / dt)
    t = np.arange(n) * dt

    x_clean = np.sin(2.0 * np.pi * f_sig * t)
    x_noisy = x_clean + rng.normal(0.0, noise_std, size=n)

    # 两个滤波器用同一个截止频率 fc —— 这是公平对比的前提
    lp1 = FirstOrderLowPass.from_cutoff(cutoff_hz=fc, dt=dt)
    lp2 = SecondOrderLowPass.from_cutoff(cutoff_hz=fc, dt=dt)
    c1 = first_order_coeffs(lp1)
    c2 = lp2.coefficients

    y1 = filter_signal(lp1, x_noisy)
    y2 = filter_signal(lp2, x_noisy)

    # ---------- 量化指标 ----------
    rms = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    rmse_noisy = rms(x_noisy, x_clean)
    rmse_1 = rms(y1, x_clean)
    rmse_2 = rms(y2, x_clean)
    ng1, ng2 = noise_gain(c1, dt), noise_gain(c2, dt)
    d1, d2 = phase_delay_ms(c1, dt, f_sig), phase_delay_ms(c2, dt, f_sig)
    g1_int, g2_int = gain_at(c1, dt, f_int), gain_at(c2, dt, f_int)

    dt_step, n_step = 0.0002, 400
    ts, ys1 = step_response(FirstOrderLowPass.from_cutoff(fc, dt_step), dt_step, n_step)
    _, ys2 = step_response(SecondOrderLowPass.from_cutoff(fc, dt_step), dt_step, n_step)
    ov1, tr1, tt1 = step_metrics(ts, ys1)
    ov2, tr2, tt2 = step_metrics(ts, ys2)

    print(f"fc = {fc} Hz | 采样 1/{1/dt:.0f}s | 信号 {f_sig}Hz | 干扰 {f_int}Hz")
    print(f"一阶 alpha = {lp1.alpha:.6f}")
    print(f"二阶 b0={lp2.b0:.6f} b1={lp2.b1:.6f} b2={lp2.b2:.6f} "
          f"a1={lp2.a1:.6f} a2={lp2.a2:.6f} (dc_gain={lp2.dc_gain:.6f})")
    print("-" * 66)
    print(f"{'':16s}{'RMSE':>9s}{'白噪放大':>10s}{'延迟@3Hz':>11s}{'120Hz衰减':>12s}")
    print(f"{'含噪输入':16s}{rmse_noisy:9.4f}{1.0:10.4f}{0.0:10.2f}ms{0.0:11.1f}dB")
    print(f"{'一阶 low-pass':16s}{rmse_1:9.4f}{ng1:10.4f}{d1:10.2f}ms{20*np.log10(g1_int):11.1f}dB")
    print(f"{'二阶 low-pass':16s}{rmse_2:9.4f}{ng2:10.4f}{d2:10.2f}ms{20*np.log10(g2_int):11.1f}dB")
    print("-" * 66)
    print(f"{'阶跃':16s}{'超调%':>9s}{'上升ms':>10s}{'调节ms':>10s}")
    print(f"{'一阶':16s}{ov1:9.2f}{tr1*1000:10.2f}{tt1*1000:10.2f}")
    print(f"{'二阶':16s}{ov2:9.2f}{tr2*1000:10.2f}{tt2*1000:10.2f}")
    print("-" * 66)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ---------- (a) 同一段含噪信号：一阶 vs 二阶 ----------
    ax = axes[0, 0]
    ax.plot(t, x_clean, "k--", lw=2, label="clean sine")
    ax.plot(t, x_noisy, color="0.75", lw=0.6, alpha=0.7, label="noisy input")
    ax.plot(t, y1, color="tab:orange", lw=1.4,
            label=f"1st order (delay {d1:.1f} ms)")
    ax.plot(t, y2, color="tab:blue", lw=1.8,
            label=f"2nd order (delay {d2:.1f} ms)")
    ax.set_title(f"(a) Same noise, same fc={fc} Hz")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 放大窗：只看 0.40~0.56s，让「二阶滞后更多」变得肉眼可见
    axin = ax.inset_axes([0.46, 0.08, 0.50, 0.38])
    m = (t >= 0.40) & (t <= 0.56)
    axin.plot(t[m] * 1000, x_clean[m], "k--", lw=1.2)
    axin.plot(t[m] * 1000, y1[m], color="tab:orange", lw=1.1)
    axin.plot(t[m] * 1000, y2[m], color="tab:blue", lw=1.3)
    axin.set_title("zoom 0.40–0.56 s", fontsize=7)
    axin.tick_params(labelsize=6)
    axin.grid(True, alpha=0.3)

    # ---------- (b) 阶跃响应 ----------
    ax = axes[0, 1]
    ax.plot(ts * 1000, ys1, color="tab:orange", lw=1.6,
            label=f"1st order (overshoot {ov1:.1f}%)")
    ax.plot(ts * 1000, ys2, color="tab:blue", lw=2.0,
            label=f"2nd order (overshoot {ov2:.1f}%)")
    ax.axhline(1.0, ls=":", color="k", lw=0.8)
    ax.set_title("(b) Step response — 2nd order rings slightly")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("output")
    ax.set_ylim(-0.05, 1.25)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---------- (c) 幅频响应 ----------
    f = np.logspace(-1, np.log10(0.5 / dt), 1000)   # 0.1Hz ~ 奈奎斯特 500Hz
    ax = axes[1, 0]
    for coeffs, color, name in ((c1, "tab:orange", "1st order: −20 dB/dec"),
                                (c2, "tab:blue", "2nd order: −40 dB/dec")):
        mag = np.maximum(np.abs(response(coeffs, dt, f)), 1e-12)
        ax.semilogx(f, 20.0 * np.log10(mag), color=color, lw=1.8, label=name)
    ax.axhline(-3.0, ls="--", color="k", alpha=0.7, lw=1)
    ax.axvline(fc, ls=":", color="gray", alpha=0.8)
    ax.axvline(f_int, ls=":", color="tab:red", alpha=0.5)
    ax.text(fc * 1.08, -52, f"fc={fc}Hz", fontsize=8, rotation=90)
    ax.text(f_int * 1.06, -52, f"interference\n{f_int:.0f}Hz", fontsize=7,
            color="tab:red", rotation=90, va="bottom")
    ax.text(0.12, -2.6, "-3 dB", fontsize=8)
    ax.set_title("(c) Magnitude response — same −3dB, steeper roll-off")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("magnitude [dB]")
    ax.set_ylim(-60, 2)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ---------- (d) 延迟 vs 残余噪声（白噪声 / 窄带干扰） ----------
    ax = axes[1, 1]
    fcs = np.logspace(np.log10(4.0), np.log10(300.0), 80)
    row = {"d1": [], "d2": [], "wn1": [], "wn2": [], "nb1": [], "nb2": []}
    for f_c in fcs:
        a = FirstOrderLowPass.from_cutoff(f_c, dt)
        b = SecondOrderLowPass.from_cutoff(f_c, dt)
        ca, cb = first_order_coeffs(a), b.coefficients
        row["d1"].append(phase_delay_ms(ca, dt, f_sig))
        row["d2"].append(phase_delay_ms(cb, dt, f_sig))
        row["wn1"].append(20 * np.log10(noise_gain(ca, dt)))
        row["wn2"].append(20 * np.log10(noise_gain(cb, dt)))
        row["nb1"].append(20 * np.log10(max(gain_at(ca, dt, f_int), 1e-12)))
        row["nb2"].append(20 * np.log10(max(gain_at(cb, dt, f_int), 1e-12)))

    ax.plot(row["d1"], row["wn1"], "--", color="tab:orange", lw=1.4,
            label="1st, white noise")
    ax.plot(row["d2"], row["wn2"], "--", color="tab:blue", lw=1.4,
            label="2nd, white noise")
    ax.plot(row["d1"], row["nb1"], "-", color="tab:orange", lw=1.8,
            label=f"1st, {f_int:.0f}Hz interference")
    ax.plot(row["d2"], row["nb2"], "-", color="tab:blue", lw=1.8,
            label=f"2nd, {f_int:.0f}Hz interference")
    ax.plot([d1], [20 * np.log10(ng1)], "o", color="tab:orange", ms=6, zorder=5)
    ax.plot([d2], [20 * np.log10(ng2)], "o", color="tab:blue", ms=6, zorder=5)
    ax.plot([d1], [20 * np.log10(g1_int)], "s", color="tab:orange", ms=6, zorder=5)
    ax.plot([d2], [20 * np.log10(g2_int)], "s", color="tab:blue", ms=6, zorder=5)
    ax.set_title(f"(d) Delay vs residual noise (fc={fc}Hz marked)")
    ax.set_xlabel("phase delay at 3 Hz [ms]")
    ax.set_ylabel("residual level [dB re input]")
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "day4_second_order_lowpass.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
