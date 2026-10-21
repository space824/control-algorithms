"""连续系统 → 离散系统的三条路线：前向欧拉 / 后向欧拉 / 零阶保持（ZOH）。

前五天所有公式都是离散的（`y[k] = α·x[k] + (1-α)·y[k-1]`、`x[k+1] = A·x[k] + B·u[k]`），
但物理世界是连续的。中间这一步"怎么从连续跳到离散"，决定了仿真跑得对不对、
以及搬到单片机上会不会翻车。

统一的对象（连续时间状态空间）：

    ẋ(t) = A·x(t) + B·u(t)

目标：找到一个"一拍一拍"的递推式

    x[k+1] = Ad·x[k] + Bd·u[k]

三条路线的差别，全在 (Ad, Bd) 怎么算。

---

1) 前向欧拉（forward Euler / 显式欧拉）

    x[k+1] = x[k] + Ts·(A·x[k] + B·u[k])

    Ad = I + Ts·A
    Bd = Ts·B

   - 最省：一次矩阵乘加，不用求逆、不用矩阵指数；
   - **可能不稳定**：一阶对象 dx/dt = -a·x 时 Ad = 1 - a·Ts，
     Ts > 2/a 时 |Ad| > 1，数值直接发散。这是新手最常踩的坑。
   - 精度一阶：误差 ∝ Ts。

2) 后向欧拉（backward Euler / 隐式欧拉）

    x[k+1] = x[k] + Ts·(A·x[k+1] + B·u[k])

    Ad = (I - Ts·A)⁻¹
    Bd = (I - Ts·A)⁻¹·Ts·B

   - 每一步要解一个线性方程组（矩阵求逆，可离线预算好）；
   - **稳定域更保守**：连续系统稳定（A 的特征值在左半平面）→ 离散后一定也稳定，
     不管 Ts 多大。这是它相对前向欧拉最大的优势。
   - 精度同样是一阶：误差 ∝ Ts，只是误差的方向和大小不同。

3) 零阶保持（ZOH, Zero-Order Hold）—— 嵌入式里真正用的那个

    假设 u 在每个采样区间 [kTs, (k+1)Ts) 内保持不变（这正是 DAC / PWM 的行为），
    那么可以解析求出精确解：

    x[k+1] = e^{A·Ts}·x[k] + (∫₀^{Ts} e^{A·τ} dτ)·B·u[k]

    Ad = e^{A·Ts}
    Bd = ∫₀^{Ts} e^{A·τ} dτ · B

   - 输入真的保持恒定时，**它是精确的**（误差 0，不是"更小"）；
   - 保稳定：连续稳定 → 离散稳定，且能精确复现连续响应；
   - 代价：要算矩阵指数，但可以离线算好、烧进固件，运行时和欧拉一样便宜。

   Bd 的实用算法：把 A、B 拼成增广矩阵

        M = [[A, B],
             [0, 0]]

    则 e^{M·Ts} 的左上块就是 Ad、右上块就是 Bd。一行矩阵指数解决，不用手推积分。

---

一句话结论：
    前向欧拉 = 最省但会炸；后向欧拉 = 稳但只是近似；ZOH = 精确，且嵌入式里恰好成立。
    控制代码里要用的，几乎总是 ZOH（离线算好）或它的等价物。

本模块只用 numpy，不依赖 scipy——自己实现矩阵指数（缩放-平方 + 泰勒级数），
这样你不用为了这一课装新依赖，也能看清"ZOH 到底在算什么"。
"""

import numpy as np


def expm(A, terms=18):
    """矩阵指数 e^A，缩放-平方（scaling & squaring）+ 泰勒级数。

    纯 numpy 实现，避免额外依赖。对小尺寸矩阵（1~6 阶）精度足够。

    原理：
        1. 先把 A 除以 2^s，让它"变小"（此时泰勒级数收敛极快）；
        2. 用泰勒级数算 e^{A/2^s}；
        3. 连续平方 s 次，还原成 e^A。
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        raise ValueError("expm 需要一个方阵")

    # 缩放：把矩阵范数压到 1 以内
    norm = np.linalg.norm(A, np.inf)
    s = 0 if norm <= 1.0 else int(np.ceil(np.log2(norm)))
    As = A / (2.0 ** s)

    # 泰勒级数
    E = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, terms + 1):
        term = term @ As / k
        E = E + term

    # 平方还原
    for _ in range(s):
        E = E @ E
    return E


def euler_forward(A, B, Ts):
    """前向欧拉：Ad = I + Ts·A，Bd = Ts·B。简单，但 Ts 大了会发散。"""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    I = np.eye(A.shape[0])
    return I + Ts * A, Ts * B


def euler_backward(A, B, Ts):
    """后向欧拉：Ad = (I - Ts·A)⁻¹，Bd = Ad·(Ts·B)。保稳定，精度一阶。"""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    I = np.eye(A.shape[0])
    Ad = np.linalg.inv(I - Ts * A)
    return Ad, Ad @ (Ts * B)


def zoh(A, B, Ts):
    """零阶保持（精确）：Ad = e^{A·Ts}，Bd = ∫₀^{Ts} e^{A·τ}dτ·B。

    用增广矩阵一次算出两半：
        M = [[A, B], [0, 0]]  →  e^{M·Ts} = [[Ad, Bd], [0, I]]
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    n, m = A.shape[0], B.shape[1]

    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B

    E = expm(M * Ts)
    return E[:n, :n], E[:n, n:]


def step_response(Ad, Bd, n, C=None):
    """对单位阶跃（u ≡ 1）跑 n 拍，返回输出序列 y[k]（y[k] 对应 t = k·Ts）。"""
    Ad = np.atleast_2d(Ad)
    Bd = np.atleast_2d(Bd)
    C = np.atleast_2d(C) if C is not None else np.eye(Ad.shape[0])[:1, :]

    x = np.zeros(Ad.shape[0])
    y = np.empty(n)
    for k in range(n):
        y[k] = float((C @ x)[0])   # 先记录 t = k·Ts 的输出，再推进一步
        x = Ad @ x + Bd[:, 0]      # u = 1
    return y


def closed_loop_step(A, B, Kp, Ki, Ts, n, setpoint=1.0, delay_steps=1):
    """二阶结构：对象（ZOH 离散）+ PI 控制 + 计算延迟，跑闭环阶跃。

    控制器用后向欧拉离散的积分器：I[k] = I[k-1] + e[k]·Ts，u = Kp·e + Ki·I。

    delay_steps 模拟真实 MCU 的「采样 → 算 → 下一拍才输出」：
    默认 1 拍，也就是说同一个 Kp/Ki 下，Ts 越大、等效纯延迟越大，
    闭环超调越大 —— 这正是「控制周期不能随便放大」的根源。
    """
    Ad, Bd = zoh(A, B, Ts)
    x = np.zeros(Ad.shape[0])
    integ = 0.0
    pending = [0.0] * delay_steps          # 还没生效的控制量
    y_hist = np.empty(n)
    for k in range(n):
        y = float(x[0])
        y_hist[k] = y

        e = setpoint - y
        integ += e * Ts
        u_new = Kp * e + Ki * integ

        if delay_steps > 0:
            u_apply = pending.pop(0)
            pending.append(u_new)
        else:
            u_apply = u_new

        x = Ad @ x + Bd[:, 0] * u_apply
    return np.arange(n) * Ts, y_hist
