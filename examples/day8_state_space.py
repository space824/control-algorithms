"""Day 8: 状态空间模型 —— 从物理方程和传递函数写 A、B、C、D，再看极点、可控性、可观性。

统一对象：质量-弹簧-阻尼系统  m·ẍ + c·ẋ + k·x = F，取 m = 1、k = 4（ωn = 2 rad/s）。

    (a) 一个系统内部有两个状态，其中一个你没有传感器
    (b) 同样的 ωn，改阻尼 c → 时域形状（对应 Day 4 的 Q）
    (c) 极点随 c 的轨迹：始终在以 ωn 为半径的圆上
    (d) ZOH 把极点映到 z = e^{s·Ts}：左半平面 → 单位圆内

运行：在项目根目录执行
    python examples/day8_state_space.py
生成图片: docs/images/day8_state_space.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.state_space import (
    StateSpace, canonical_from_transfer, dc_motor, mass_spring_damper,
)

M, K = 1.0, 4.0
WN = np.sqrt(K / M)                       # 2 rad/s
TS_SIM = 0.002                            # 仿真步长（远小于 1/ωn）

# 四个典型阻尼比对应的 c（c = 2ζ√(km)）
ZETAS = [0.2, 1 / np.sqrt(2), 1.0, 2.0]
C_LIST = [2 * z * np.sqrt(K * M) for z in ZETAS]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ================================================== (a) 两个状态，一个传感器
    sys0 = mass_spring_damper(M, C_LIST[0], K)
    n_a = int(12.0 / TS_SIM)
    t_a, X_a, Y_a = sys0.step(TS_SIM, n_a)

    # x2 过零的时刻应该正好是 x1 到达峰值的时刻（因为 x2 = ẋ1）
    # 注意要跳过 t=0：那一刻 x2 本来就是 0（初值为零），会误判
    i_peak = int(np.argmax(X_a[:, 0]))
    sc = np.where(np.diff(np.signbit(X_a[:, 1]).astype(int)) != 0)[0]
    sc = sc[t_a[sc] > 0.5]
    i_zero = int(sc[0])
    print(f"(a) 位移峰值在 t = {t_a[i_peak]:.3f}s，速度过零点在 t = {t_a[i_zero]:.3f}s"
          f"  → 相差 {abs(t_a[i_peak] - t_a[i_zero]) * 1000:.1f} ms")

    ax = axes[0, 0]
    ax.plot(t_a, X_a[:, 0], lw=2, color="tab:blue",
            label="x1 = position  (measured, C=[1,0])")
    ax.plot(t_a, X_a[:, 1], lw=2, color="tab:orange",
            label="x2 = velocity  (NO sensor)")
    ax.axhline(1.0 / K, ls=":", color="gray", lw=0.9)
    ax.axvline(t_a[i_peak], ls="--", color="tab:blue", alpha=0.5, lw=1)
    ax.axvline(t_a[i_zero], ls="--", color="tab:orange", alpha=0.5, lw=1)
    ax.annotate("velocity = 0\nwhere position peaks",
                xy=(t_a[i_peak], 0.0), xytext=(3.0, 0.18), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_title("(a) One plant, two states — only one is measured")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("state value")
    ax.set_xlim(0, 12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ================================================== (b) 阻尼比 σ
    ax = axes[0, 1]
    n_b = int(10.0 / TS_SIM)
    print("\n(b) 同为 ωn=2 rad/s，只改阻尼 c：")
    print(f"{'ζ':>7s}{'c':>9s}{'Q=1/(2ζ)':>11s}{'超调 %':>9s}{'状态数':>8s}")
    for c_i, z_i in zip(C_LIST, ZETAS):
        sys_i = mass_spring_damper(M, c_i, K)
        t_i, X_i, Y_i = sys_i.step(TS_SIM, n_b)
        peak = float(np.max(Y_i))
        overshoot = (peak - 1.0 / K) * K * 100
        print(f"{z_i:>7.3f}{c_i:>9.3f}{1 / (2 * z_i):>11.3f}{overshoot:>9.1f}{sys_i.n:>8d}")
        ax.plot(t_i, Y_i, lw=1.8, label=f"$\\zeta$={z_i:.3f}  (c={c_i:.2f})")
    ax.axhline(1.0 / K, ls=":", color="gray", lw=0.9)
    ax.set_title("(b) Same $\\omega_n$, different damping $\\zeta$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output y = x1")
    ax.set_xlim(0, 10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ================================================== (c) 极点轨迹
    ax = axes[1, 0]
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(WN * np.cos(theta), WN * np.sin(theta), "--", color="gray", lw=0.9,
            label=f"circle of radius $\\omega_n$={WN:.1f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-12, 0, color="tab:green", alpha=0.06)
    ax.text(-11.5, 1.75, "stable half-plane", fontsize=8, color="tab:green")

    c_sweep = np.concatenate([np.linspace(0.02, 4.0, 120),
                              np.linspace(4.0, 20.0, 120)])
    Re, Im = [], []
    for c_i in c_sweep:
        lam = np.linalg.eigvals(mass_spring_damper(M, c_i, K).A)
        # 排序必须显式写死：eigvals 的返回顺序不保证，根变成实数后
        # 若不做 tie-break，两条分支会随机互换、画出横穿平面的假线。
        lam = sorted(lam, key=lambda z: (-z.imag, -z.real))
        Re.append([z.real for z in lam])
        Im.append([z.imag for z in lam])
    Re, Im = np.array(Re), np.array(Im)
    for j in range(2):
        ax.plot(Re[:, j], Im[:, j], lw=1.8, color="tab:red" if j == 0 else "tab:purple",
                label="pole locus  (c: 0.02 → 20)" if j == 0 else None)

    for c_i, z_i in zip(C_LIST, ZETAS):
        lam = np.linalg.eigvals(mass_spring_damper(M, c_i, K).A)
        ax.plot(lam.real, lam.imag, "o", ms=5, mfc="none",
                mec="tab:red" if z_i == ZETAS[0] else "tab:purple")

    ax.set_title("(c) Pole locus as c sweeps: |$\\lambda$| = $\\omega_n$ until breakaway")
    ax.set_xlabel("Re [1/s]")
    ax.set_ylabel("Im [1/s]")
    ax.set_xlim(-12, 1.5)
    ax.set_ylim(-3, 3)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ================================================== (d) s → z 映射
    ax = axes[1, 1]
    Ts_d = 0.1
    ax.plot(np.cos(theta), np.sin(theta), "--", color="gray", lw=0.9,
            label="unit circle |z|=1")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)

    zs = np.exp(np.array(Re) * Ts_d + 1j * np.array(Im) * Ts_d)
    for j in range(2):
        ax.plot(zs[:, j].real, zs[:, j].imag, lw=1.8,
                color="tab:red" if j == 0 else "tab:purple")

    for c_i in C_LIST:
        lam = np.linalg.eigvals(mass_spring_damper(M, c_i, K).A)
        zz = np.exp(lam * Ts_d)
        ax.plot(zz.real, zz.imag, "o", ms=5, mfc="none", mec="k")

    lam0 = sorted(np.linalg.eigvals(mass_spring_damper(M, 0.8, K).A),
                  key=lambda z: (-z.imag, -z.real))[0]
    z0 = np.exp(lam0 * Ts_d)
    ax.text(-1.08, 1.05,
            f"e.g.  $\\lambda$ = {lam0.real:+.3f}{lam0.imag:+.3f}j\n"
            f"      z = exp($\\lambda$Ts) = {z0.real:.3f}{z0.imag:+.3f}j\n"
            f"      |z| = {abs(z0):.4f} < 1  → stable",
            fontsize=7, color="tab:blue", va="top", ha="left")

    ax.set_title(f"(d) ZOH maps poles: z = exp($\\lambda$·Ts), Ts={Ts_d}s")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "day8_state_space.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved -> {out}")

    verify()


def verify():
    """把"同一系统两套写法"和"离散化保极点"这两件事都实测一遍。"""
    print("\n" + "=" * 68)
    print("验证 1：物理建模 vs 从传递函数写的能控标准型，是不是同一个系统")
    print("=" * 68)

    sys_phy = mass_spring_damper(M, 0.5, K)                 # 1/(s² + 0.5s + 4)
    sys_can = canonical_from_transfer([1.0], [M, 0.5, K])
    w_test = np.array([0.1, 1.0, 2.0, 10.0])

    print(f"{'ω [rad/s]':>10s}{'|G| 物理':>12s}{'|G| 标准型':>13s}{'相对差':>12s}")
    g1 = sys_phy.freq_response(w_test)
    g2 = sys_can.freq_response(w_test)
    for wi, a, b in zip(w_test, g1, g2):
        print(f"{wi:>10.2f}{abs(a):>12.6f}{abs(b):>13.6f}{abs(a - b) / abs(a):>12.2e}")
    print(f"直流增益：物理 {sys_phy.dcgain():.6f}，标准型 {sys_can.dcgain():.6f}"
          f"（理论 1/k = {1 / K:.6f}）")

    print("\n" + "=" * 68)
    print("验证 2：ZOH 离散精确保留极点 —— eig(Ad) = exp(eig(A)·Ts)")
    print("=" * 68)
    Ts = 0.1
    sys_p = mass_spring_damper(M, 0.8, K)
    dsys = sys_p.discretize(Ts)
    lam = np.sort_complex(np.linalg.eigvals(sys_p.A))
    z_eig = np.sort_complex(np.linalg.eigvals(dsys.A))
    z_map = np.sort_complex(np.exp(lam * Ts))
    print(f"{'连续极点 λ':>22s}{'eig(Ad)':>22s}{'exp(λTs)':>22s}")
    for a, b, c in zip(lam, z_eig, z_map):
        print(f"{a:>22.6f}{b:>22.6f}{c:>22.6f}")
    print(f"最大偏差 = {np.max(np.abs(z_eig - z_map)):.2e}  （浮点极限，视为精确）")
    print(f"极点模长 |eig(Ad)| = {np.abs(z_eig)}  → 全在单位圆内 = 稳定")

    print("\n" + "=" * 68)
    print("验证 3：可控性 / 可观性 —— Day 9 观测器和 Day 15 LQR 的前提")
    print("=" * 68)
    cases = [
        ("质量-弹簧-阻尼 (C=[1,0] 测位移)", mass_spring_damper(M, 0.5, K)),
        ("质量-弹簧-阻尼 (C=[0,1] 测速度)",
         StateSpace(mass_spring_damper(M, 0.5, K).A,
                    mass_spring_damper(M, 0.5, K).B,
                    [[0.0, 1.0]])),
        ("直流电机 (输出转速)", dc_motor()),
        ("人为构造：两个相同极点且只测第一个",
         StateSpace([[-1.0, 0.0], [0.0, -1.0]], [[1.0], [0.0]], [[1.0, 0.0]])),
    ]
    for label, sys_i in cases:
        print(f"{label:<34s} 可控={str(sys_i.is_controllable()):<5s}"
              f" 可观={str(sys_i.is_observable()):<5s}"
              f" 极点={np.round(sys_i.poles(), 4)}")

    print("\n注：最后一行是刻意造的——两个相同的极点，且 B 和 C 都只碰得到第一个状态。")
    print("    第二个状态既推不动、也看不见，所以可控和可观同时失败。")
    print("    现实里更常见的是'可控但不可观'：对象推得动，但传感器读不全。")
    print("    这时候就得上观测器（Day 9）——它的成立前提正是可观性。")


if __name__ == "__main__":
    main()
