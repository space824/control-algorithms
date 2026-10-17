"""Day 3: 一阶低通滤波 —— 含噪正弦滤波 + α 与截止频率的关系。

一阶 RC 低通的离散形式（指数移动平均）：
    y[k] = α·x[k] + (1-α)·y[k-1]

本脚本做三件事：
    1. 生成含高斯噪声的正弦信号，用一阶低通滤波，对比"真实值/含噪/滤波后"。
    2. 用不同 α 滤波同一段含噪信号，观察"滤波强度 vs 延迟"的 trade-off。
    3. 画幅频响应和截止频率 fc(α) 曲线，看 α 如何移动 -3dB 截止频率。

运行：在项目根目录执行
    python examples/day3_lowpass_filter.py
生成图片: docs/images/day3_lowpass_filter.png
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


def magnitude_response(alpha, dt, f):
    """一阶低通在离散频率 f 处的幅频响应 |H(e^{jω})|，ω = 2π·f·dt。"""
    w = 2.0 * np.pi * np.asarray(f) * dt
    denom = np.sqrt(1.0 - 2.0 * (1.0 - alpha) * np.cos(w) + (1.0 - alpha) ** 2)
    return alpha / denom


def cutoff_freq(alpha, dt):
    """由 alpha 和采样周期 dt 计算离散域精确的 -3dB 截止频率。

    来源：cos(2π·fc·dt) = (2 - 2α - α²) / (2(1-α))。仅当 α ≤ 2(√2-1)≈0.828
    时存在 -3dB 点；更大的 α 整个通带几乎不衰减，返回 NaN。
    """
    alpha = np.asarray(alpha, dtype=float)
    c = (2.0 - 2.0 * alpha - alpha ** 2) / (2.0 * (1.0 - alpha))
    with np.errstate(invalid="ignore"):
        wc = np.arccos(c)
    return wc / (2.0 * np.pi * dt)


def filter_signal(x, alpha, y0=0.0):
    """对整段信号 x 逐点做一阶低通，返回滤波后的数组。"""
    lp = FirstOrderLowPass(alpha=alpha, y0=y0)
    y = np.empty_like(x, dtype=float)
    for k in range(len(x)):
        y[k] = lp.update(x[k])
    return y


def main():
    dt = 0.001                 # 采样周期 1ms（1kHz）
    f_sig = 3.0                # 正弦信号频率 3Hz
    duration = 1.0             # 1 秒（3 个周期）
    rng = np.random.default_rng(0)   # 固定随机种子，结果可复现

    n = int(duration / dt)
    t = np.arange(n) * dt

    # 干净正弦 + 高斯噪声
    x_clean = np.sin(2.0 * np.pi * f_sig * t)
    noise = rng.normal(0.0, 0.4, size=n)
    x_noisy = x_clean + noise

    alphas = [0.05, 0.2, 0.6]  # 三档平滑系数：低 / 中 / 高

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ---------- 面板 (a)：滤波前后对比（取中间档 α=0.2） ----------
    y_a = filter_signal(x_noisy, 0.2)
    ax = axes[0, 0]
    ax.plot(t, x_clean, "k--", lw=2, label="clean sine")
    ax.plot(t, x_noisy, color="0.6", lw=0.6, alpha=0.8, label="noisy input")
    ax.plot(t, y_a, "r", label="filtered (α=0.2)")
    ax.set_title("(a) Noisy sine → first-order low-pass (α=0.2)")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---------- 面板 (b)：不同 α 滤波同一段含噪信号 ----------
    ax = axes[0, 1]
    ax.plot(t, x_clean, "k--", lw=2, label="clean sine")
    ax.plot(t, x_noisy, color="0.7", lw=0.6, alpha=0.5, label="noisy input")
    for a in alphas:
        ax.plot(t, filter_signal(x_noisy, a), label=f"α={a}")
    ax.set_title("(b) Same noisy input, different α")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---------- 面板 (c)：幅频响应，看 -3dB 截止频率随 α 移动 ----------
    f = np.logspace(-1, np.log10(0.5 / dt), 1000)  # 0.1Hz ~ 奈奎斯特频率 500Hz
    ax = axes[1, 0]
    for a in alphas:
        mag_db = 20.0 * np.log10(magnitude_response(a, dt, f))
        ax.semilogx(f, mag_db, label=f"α={a}")
        fc = cutoff_freq(a, dt)
        ax.axvline(fc, ls=":", color="gray", alpha=0.7)
        ax.text(fc * 1.1, -37, f"{fc:.1f}Hz", fontsize=7, rotation=90)
    ax.axhline(-3.0, ls="--", color="k", alpha=0.7, lw=1)
    ax.text(0.12, -2.6, "-3 dB", fontsize=8)
    ax.set_title("(c) Magnitude response |H(f)| — cutoff shifts with α")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("magnitude [dB]")
    ax.set_ylim(-40, 1)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # ---------- 面板 (d)：截止频率 fc 与 α 的关系曲线 ----------
    ax = axes[1, 1]
    a_axis = np.linspace(0.001, 0.8, 1000)  # α>0.828 无 -3dB 点，故画到 0.8
    ax.plot(a_axis, cutoff_freq(a_axis, dt), "b", label="fc(α)")
    for a in alphas:
        fc = cutoff_freq(a, dt)
        ax.plot([a], [fc], "ro")
        ax.annotate(f"α={a}, fc={fc:.1f}Hz",
                    (a, fc), textcoords="offset points",
                    xytext=(8, -14), fontsize=8)
    ax.set_title("(d) Cutoff frequency vs α")
    ax.set_xlabel("α (smoothing factor)")
    ax.set_ylabel("-3dB cutoff frequency [Hz]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "day3_lowpass_filter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
