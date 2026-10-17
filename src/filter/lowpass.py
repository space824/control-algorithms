"""一阶低通滤波器（一阶 RC 低通的离散形式，也叫指数移动平均）。

连续时间的一阶 RC 低通电路，传递函数为
    H(s) = 1 / (1 + s·τ) ，其中时间常数 τ = RC
其 -3dB 截止角频率 ωc = 1/τ，截止频率 fc = 1 / (2π·RC)。

把它离散化（对 s·x = (x - y)/dt 这类关系做后向欧拉），整理后得到一阶差分方程：
    y[k] = α·x[k] + (1-α)·y[k-1]
其中
    α = dt / (dt + τ) = dt / (dt + RC)

即：新样本 x[k] 贡献 α，上一拍输出 y[k-1] 保留 (1-α)。

α 的物理意义（本课核心）：
- α 大（→1）：滤波器"信任新样本"，输出紧跟输入，几乎不过滤噪声。
- α 小（→0）：滤波器"信任历史输出"，输出平滑，但响应慢、有延迟。
- α 越大，截止频率越高（越能通过高频）；α 越小，截止频率越低（越能压噪声）。

截止频率与 α 的关系（离散域精确式）：
    cos(2π·fc·dt) = (2 - 2α - α²) / (2(1-α))
小 α 时近似为 fc ≈ α / (2π·dt)（此时 α ≈ 2π·fc·dt）。
"""

import math


class FirstOrderLowPass:
    """一阶低通滤波器。

    Parameters
    ----------
    alpha : float
        平滑系数，取值 (0, 1]。
        alpha 越大，截止频率越高、滤波越弱、延迟越小；
        alpha 越小，截止频率越低、滤波越强、延迟越大。
    y0 : float
        初始输出 y[-1]（滤波器唯一的"记忆"）。
    """

    def __init__(self, alpha=0.2, y0=0.0):
        self.alpha = float(alpha)
        self.reset(y0)

    @classmethod
    def from_cutoff(cls, cutoff_hz, dt):
        """由 -3dB 截止频率和采样周期反推 alpha（离散域精确解）。

        由 cos(2π·fc·dt) = (2 - 2α - α²) / (2(1-α)) 解出 (0,1) 内的根。
        比近似式 alpha ≈ 2π·fc·dt 精确，fc 接近奈奎斯特频率时也不会离谱。
        """
        c = math.cos(2.0 * math.pi * cutoff_hz * dt)
        k = 1.0 - c
        alpha = -k + math.sqrt(k * (k + 2.0))
        return cls(alpha=alpha)

    def reset(self, y0=0.0):
        self._y = float(y0)  # y[k-1]，滤波器唯一的"记忆"

    def update(self, x):
        """输入一个新样本 x[k]，返回滤波输出 y[k]。

        Returns
        -------
        float
            本次滤波输出。
        """
        self._y = self.alpha * x + (1.0 - self.alpha) * self._y
        return self._y
