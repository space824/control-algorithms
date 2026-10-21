"""Day 6: 离散化专题 —— 前向欧拉 / 后向欧拉 / ZOH，以及采样时间怎么选。

连续对象（统一用一阶 ẋ = -x + u，时间常数 T = 1 s）：

    (a) 同一个 Ts 下，三条离散路线谁贴得紧
    (b) 前向欧拉的生死线：Ts 大到什么程度会发散
    (c) 误差随 Ts 的衰减速度（一阶精度 vs 精确）
    (d) 闭环视角：Ts 变大会让 PI 闭环的超调涨上去

运行：在项目根目录执行
    python examples/day6_discretization.py
生成图片: docs/images/day6_discretization.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.discrete.discretize import (
    euler_backward, euler_forward, zoh, step_response, closed_loop_step,
)

# 一阶对象 ẋ = -a·x + b·u，a = 1 时时间常数就是 1 s
A1 = np.array([[-1.0]])
B1 = np.array([[1.0]])

# 二阶对象 ẍ = -3ẋ - 2x + u（极点 -1、-2），闭环超调更能显出采样率的影响
A2 = np.array([[0.0, 1.0], [-2.0, -3.0]])
B2 = np.array([[0.0], [1.0]])

EXACT = lambda t: 1.0 - np.exp(-t)      # 一阶对象在单位阶跃下的解析解


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ---------------- (a) 三条路线 vs 解析解 ----------------
    Ts = 0.1
    n = 60
    t = np.arange(n) * Ts

    y_fw = step_response(*euler_forward(A1, B1, Ts), n=n)
    y_bw = step_response(*euler_backward(A1, B1, Ts), n=n)
    y_zoh = step_response(*zoh(A1, B1, Ts), n=n)

    d_fw = y_fw - EXACT(t)
    d_bw = y_bw - EXACT(t)
    d_zoh = y_zoh - EXACT(t)
    err_fw, signed_fw = np.max(np.abs(d_fw)), d_fw[np.argmax(np.abs(d_fw))]
    err_bw, signed_bw = np.max(np.abs(d_bw)), d_bw[np.argmax(np.abs(d_bw))]
    err_zoh = np.max(np.abs(d_zoh))
    print(f"Ts = {Ts}s 时的最大误差：前向欧拉 {signed_fw:+.5f}，后向欧拉 {signed_bw:+.5f}，ZOH {err_zoh:.2e}")

    ax = axes[0, 0]
    ax.axhline(0.0, color="k", lw=1)
    ax.plot(t, d_fw, "o-", ms=3, lw=1.2, color="tab:orange",
            label=f"forward Euler  ({signed_fw:+.4f})")
    ax.plot(t, d_bw, "s-", ms=3, lw=1.2, color="tab:green",
            label=f"backward Euler ({signed_bw:+.4f})")
    ax.plot(t, d_zoh, lw=2.0, color="tab:blue", label="ZOH (exact, ~1e-16)")
    ax.set_title(f"(a) Error vs exact at Ts={Ts}s — two Eulers err in opposite directions")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error  (discrete − exact)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---------------- (b) 前向欧拉的生死线 ----------------
    ax = axes[0, 1]
    for Ts_b in (0.5, 1.0, 1.5, 2.1):
        nb = 40
        tb = np.arange(nb) * Ts_b
        yb = step_response(*euler_forward(A1, B1, Ts_b), n=nb)
        ax.plot(tb, yb, "o-", ms=3, lw=1.2, label=f"Ts={Ts_b}s  (Ad={1-Ts_b:+.1f})")
    tt = np.linspace(0, 60, 400)
    ax.plot(tt, EXACT(tt), "k--", lw=1.5, label="exact")
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.set_title("(b) Forward Euler: unstable once Ts > 2/a")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---------------- (c) 误差 vs Ts ----------------
    ax = axes[1, 0]
    Ts_list = np.logspace(-3, 0.3, 45)          # 1ms ~ 2s
    t_final = 5.0
    errs = {"forward Euler": [], "backward Euler": [], "ZOH": []}
    for Ts_i in Ts_list:
        n_i = max(2, int(round(t_final / Ts_i)))
        t_i = np.arange(n_i) * Ts_i
        exact_i = EXACT(t_i)
        for name, route in (("forward Euler", euler_forward),
                            ("backward Euler", euler_backward),
                            ("ZOH", zoh)):
            y_i = step_response(*route(A1, B1, Ts_i), n=n_i)
            errs[name].append(np.max(np.abs(y_i - exact_i)))

    for name, color in (("forward Euler", "tab:orange"),
                        ("backward Euler", "tab:green"),
                        ("ZOH", "tab:blue")):
        ax.loglog(Ts_list, np.maximum(errs[name], 1e-17), "o-", ms=3, lw=1.4,
                  color=color, label=name)
    ax.axvline(2.0, ls="--", color="tab:red", alpha=0.7)
    ax.text(2.15, 1e-8, "Ts = 2/a\nforward Euler diverges", fontsize=7, color="tab:red")
    ax.set_title("(c) Error vs Ts — Euler is 1st order, ZOH is exact")
    ax.set_xlabel("Ts [s]")
    ax.set_ylabel("max |error| over 5 s")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ---------------- (d) 闭环：Ts 的影响 ----------------
    ax = axes[1, 1]
    Kp, Ki = 3.0, 6.0
    ts_ref = 8.0
    ref_Ts = 0.002
    n_ref = int(ts_ref / ref_Ts)
    t_ref, y_ref = closed_loop_step(A2, B2, Kp, Ki, ref_Ts, n_ref)

    print("\n闭环阶跃（二阶对象 + PI，Kp=3, Ki=6，含 1 拍计算延迟）：")
    print(f"{'Ts [ms]':>9s}{'超调 %':>9s}{'峰值':>9s}{'上升(10-90%) ms':>16s}")
    for Ts_c in (10, 40, 100, 200):
        Ts_c_s = Ts_c / 1000.0
        n_c = int(ts_ref / Ts_c_s)
        t_c, y_c = closed_loop_step(A2, B2, Kp, Ki, Ts_c_s, n_c)
        peak = float(np.max(y_c))
        overshoot = (peak - 1.0) * 100
        i10 = int(np.argmax(y_c >= 0.1))
        i90 = int(np.argmax(y_c >= 0.9))
        rise = (i90 - i10) * Ts_c_s * 1000
        print(f"{Ts_c:>9d}{overshoot:>9.1f}{peak:>9.2f}{rise:>16.1f}")
        ax.plot(t_c, y_c, "o-", ms=2.5, lw=1.2, label=f"Ts = {Ts_c} ms")

    ax.plot(t_ref, y_ref, "k--", lw=2, label="Ts = 2 ms (reference)")
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.set_title("(d) Closed loop: the same PI at different Ts")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output")
    ax.set_ylim(-0.1, 2.2)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "day6_discretization.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
