"""状态空间模型：ẋ = A·x + B·u，y = C·x + D·u。

前七天写的对象，全都是"一个输入进去、一个输出出来"：

    y = H(z)·u          # 一阶低通、biquad、互补滤波，都是这个形状

这种写法隐式地假设**系统内部只有一个量在动**。但真实的被控对象几乎都不是：
电机同时有转速和电流，倒立摆同时有角度、角速度、小车位置和速度。

状态空间做的事，就是把这些"内部正在动的量"显式地写出来：

    ẋ = A·x + B·u       # 内部怎么演化
    y = C·x + D·u       # 从哪一路读出去

四个矩阵各管一件事：

| 矩阵 | 管什么 | 一句话 |
|---|---|---|
| A | 系统自己怎么演化 | 决定极点、稳定性、振荡频率——**最关键的一个** |
| B | 输入从哪个门进来 | "我这一脚踹在哪" |
| C | 从哪一路读出去 | 同一个系统，C 换成 [0,1] 就输出速度而不是位移 |
| D | 输入直接穿到输出 | 真实系统有惯性，一般 D = 0 |

状态的三个条件（判据，不是定义）：

1. **足够**：知道 x[k] 和未来的 u，就能推出未来所有的 y；
2. **独立**：任意两个状态之间不能存在代数关系（否则可以约掉一个）；
3. **最少**：满足上面两条的最小个数 = 传递函数分母的阶数。

第 3 条最实用：**分母是几阶，就有几个状态**。所以 Day 4 的 biquad 是二阶的，
它内部就有两个状态（不管你代码里写了几行）。

和传递函数的关系：两者是同一个系统的两套写法，可以互相换算。

    G(s) = C·(sI − A)⁻¹·B + D

离散化直接接 Day 6：把 (A, B) 换成 (Ad, Bd)，C、D 不动。

    Ad = e^{A·Ts}
    Bd = ∫₀^{Ts} e^{A·τ} dτ · B

**ZOH 离散会精确保留极点**：eig(Ad) = exp(eig(A)·Ts)。这不是巧合——它正是
"s 左半平面 → z 单位圆内"这条稳定性映射的代数形式，本课最后一张图就在画这个。
"""

import numpy as np

from src.discrete.discretize import zoh as _zoh


class StateSpace:
    """连续时间状态空间模型。

    Parameters
    ----------
    A : (n, n) 状态矩阵
    B : (n, m) 输入矩阵
    C : (p, n) 输出矩阵
    D : (p, m) 直通矩阵，None 视为全零
    name : str 仅在打印时用
    """

    def __init__(self, A, B, C, D=None, name=""):
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.C = np.atleast_2d(np.asarray(C, dtype=float))

        n = self.A.shape[0]
        m = self.B.shape[1]
        p = self.C.shape[0]

        if self.A.shape != (n, n):
            raise ValueError(f"A 必须是方阵，得到 {self.A.shape}")
        if self.B.shape != (n, m):
            raise ValueError(f"B 必须是 ({n}, m)，得到 {self.B.shape}")
        if self.C.shape != (p, n):
            raise ValueError(f"C 必须是 (p, {n})，得到 {self.C.shape}")

        if D is None:
            D = np.zeros((p, m))
        self.D = np.atleast_2d(np.asarray(D, dtype=float))
        if self.D.shape != (p, m):
            raise ValueError(f"D 必须是 ({p}, {m})，得到 {self.D.shape}")

        self.name = name

    # ------------------------------------------------------------------ 基本尺寸
    @property
    def n(self):
        return self.A.shape[0]

    @property
    def m(self):
        return self.B.shape[1]

    @property
    def p(self):
        return self.C.shape[0]

    def _require_siso(self):
        if self.m != 1 or self.p != 1:
            raise ValueError("这个方法只支持单入单出（SISO）")

    # ------------------------------------------------------------------ 分析
    def poles(self):
        """极点 = A 的特征值。稳定性只看它的实部符号。"""
        return np.linalg.eigvals(self.A)

    def dcgain(self):
        """直流增益 G(0) = −C·A⁻¹·B + D（要求 A 可逆）。"""
        self._require_siso()
        return float(-(self.C @ np.linalg.solve(self.A, self.B) + self.D)[0, 0])

    def freq_response(self, w):
        """频率响应 G(jω) = C·(jωI − A)⁻¹·B + D，`w` 单位为 rad/s。"""
        self._require_siso()
        w = np.atleast_1d(np.asarray(w, dtype=float))
        I = np.eye(self.n)
        out = np.empty(w.size, dtype=complex)
        for i, wi in enumerate(w):
            x = np.linalg.solve(1j * wi * I - self.A, self.B)
            out[i] = (self.C @ x + self.D)[0, 0]
        return out

    def controllability_matrix(self):
        """[B, AB, A²B, …, A^{n-1}B]，秩为 n 才叫"可控"。"""
        blocks = [self.B]
        for _ in range(1, self.n):
            blocks.append(self.A @ blocks[-1])
        return np.hstack(blocks)

    def observability_matrix(self):
        """[C; CA; CA²; …; CA^{n-1}]，秩为 n 才叫"可观"。"""
        blocks = [self.C]
        for _ in range(1, self.n):
            blocks.append(blocks[-1] @ self.A)
        return np.vstack(blocks)

    def is_controllable(self, tol=None):
        """能不能通过输入把状态推到任意位置。Day 9 的观测器和 Day 15 的 LQR 都要它。"""
        rank = np.linalg.matrix_rank(self.controllability_matrix(), tol=tol)
        return rank == self.n

    def is_observable(self, tol=None):
        """能不能从输出反推出全部状态。Day 9 的观测器成立的前提就是它。"""
        rank = np.linalg.matrix_rank(self.observability_matrix(), tol=tol)
        return rank == self.n

    # ------------------------------------------------------------------ 离散 / 仿真
    def discretize(self, Ts, method="zoh"):
        """连续 → 离散，返回新的 StateSpace（C、D 不变）。

        目前只实现 ZOH（Day 6 的结论：嵌入式和它天然吻合）。
        """
        if method != "zoh":
            from src.discrete.discretize import euler_backward, euler_forward

            Ad, Bd = (euler_forward if method == "forward" else euler_backward)(
                self.A, self.B, Ts
            )
            return StateSpace(Ad, Bd, self.C, self.D, name=f"{self.name}@{method}")

        Ad, Bd = _zoh(self.A, self.B, Ts)
        return StateSpace(Ad, Bd, self.C, self.D, name=f"{self.name}@zoh")

    def step(self, Ts, n_steps, setpoint=1.0):
        """单位阶跃（u ≡ setpoint）跑 n 拍，返回 (t, X, Y)。

        X 的形状是 (n_steps, n)——**所有状态**的轨迹，不只是输出。
        这一点是状态空间相对传递函数的额外好处：那些你没有传感器的量，也能算出来。
        """
        d = self.discretize(Ts)
        X = np.empty((n_steps, self.n))
        Y = np.empty(n_steps)
        x = np.zeros(self.n)

        for k in range(n_steps):
            X[k] = x
            Y[k] = (self.C @ x + self.D[:, 0] * setpoint)[0]
            x = d.A @ x + d.B[:, 0] * setpoint

        return np.arange(n_steps) * Ts, X, Y

    def __repr__(self):
        tag = f" ({self.name})" if self.name else ""
        return f"StateSpace{tag}: n={self.n}, m={self.m}, p={self.p}"


# ====================================================================== 建模模板


def mass_spring_damper(m=1.0, c=0.5, k=1.0):
    """质量-弹簧-阻尼：m·ẍ + c·ẋ + k·x = F。

    选状态 x1 = 位移、x2 = 速度（"选什么当状态"是本课的第一个手艺）：

        ẋ1 = x2
        ẋ2 = −(k/m)·x1 − (c/m)·x2 + (1/m)·F

    C = [1, 0] 表示"只测得到位移"。换成 [0, 1] 就变成"只测得到速度"——
    同一个物理系统，C 不同就是不同的传感器配置。这正是 Day 9 观测器要解决的问题。
    """
    A = np.array([[0.0, 1.0], [-k / m, -c / m]])
    B = np.array([[0.0], [1.0 / m]])
    C = np.array([[1.0, 0.0]])
    return StateSpace(A, B, C, D=None, name="mass-spring-damper")


def dc_motor(R=1.0, L=0.5, J=0.01, b=1e-3, Ke=0.05, Kt=0.05):
    """直流电机的两状态模型（状态 = [电枢电流 i, 转速 ω]）。

        L·di/dt = −R·i − Ke·ω + u      # 电枢回路（Ke·ω 是反电动势）
        J·dω/dt =  Kt·i − b·ω          # 机械方程（Kt·i 是电磁转矩）

    C = [0, 1] 表示输出转速（P1 速度环用的就是这一路）。
    三个时间常数 R/L、J/b、以及机电耦合共同决定极点；
    L 很小时电气极点是快极点，正是"电流环要比速度环快得多"的原因。
    """
    A = np.array([[-R / L, -Ke / L], [Kt / J, -b / J]])
    B = np.array([[1.0 / L], [0.0]])
    C = np.array([[0.0, 1.0]])
    return StateSpace(A, B, C, D=None, name="dc-motor")


def canonical_from_transfer(num, den):
    """从传递函数系数写能控标准型（直接给你 A、B、C、D）。

    num、den 按 s 的降幂排列，例如 (3s+2)/(s²+5s+6) → num=[3,2], den=[1,5,6]。

    能控标准型（den 归一化成首一后，n = deg(den)）：

        A = [[-a1, -a2, …, -an],
             [  1,   0, …,   0],
             [  0,   1, …,   0],
             …                 ]
        B = [1, 0, …, 0]ᵀ
        D = b0（分子的首项，分子阶数 < 分母时为 0）
        C = [b1 − D·a1, b2 − D·a2, …]

    这套形式的好处：**任何**传递函数都能机械地转成状态空间，不用想物理意义。
    坏处：状态没有物理含义（它只是"x 的 0~n-1 阶导数"的打包），
    所以真要做观测器/LQR，还是得自己按物理量选状态。今天两种都写一遍来对比。
    """
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)

    if den.ndim != 1 or num.ndim != 1:
        raise ValueError("num / den 必须是一维系数数组")
    if abs(den[0]) < 1e-12:
        raise ValueError("分母首项不能为 0")

    den = den / den[0]
    n = den.size - 1
    if n < 1:
        raise ValueError("分母至少一阶")

    if num.size > n + 1:
        raise ValueError("分子阶数高于分母，不是真分式")
    num = np.pad(num, (n + 1 - num.size, 0))  # 前面补零对齐

    D = float(num[0])
    a = den[1:]                     # a1 … an
    b = num[1:]                     # b1 … bn

    A = np.zeros((n, n))
    A[0, :] = -a
    if n > 1:
        A[1:, :-1] = np.eye(n - 1)

    B = np.zeros((n, 1))
    B[0, 0] = 1.0

    C = (b - D * a).reshape(1, n)

    return StateSpace(A, B, C, np.array([[D]]), name="controllable-canonical")
